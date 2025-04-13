import torch as t
from pathlib import Path
import json
import argparse
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Literal
from functools import partial
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType, PatchType, PruneScores,CircuitOutputs,Measurements
from auto_circuit.utils.graph_utils import patchable_model,edge_counts_util
from auto_circuit.data import load_datasets_from_json,PromptDataLoader
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune import run_circuits
from auto_circuit.metrics.prune_metrics.answer_value import measure_answer_val, batch_avg_answer_val
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff, batch_avg_answer_diff
from auto_circuit.metrics.prune_metrics.correct_answer_percent import measure_correct_ans_percent
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.patchable_model import PatchableModel

device = t.device("cuda" if t.cuda.is_available() else "mps")

def find_minimal_circuit(
        model: PatchableModel, 
        test_loader: PromptDataLoader, 
        prune_scores: PruneScores, 
        metrics: dict[str, Callable],
        ablation_type: AblationType, 
        patch_type: PatchType, 
        optimize_metric: str = "faithfulness",
        target_performance_pct: float=0.8,
        reverse_clean_corrupt: bool=False):
    # Evaluate baseline performance with all edges
    total_edge_count = model.n_edges
    
    # Calculate maximum allowed difference from baseline
    target_performance = target_performance_pct ## assume we use faithfulness as the metric    
    print(f"Target performance: {target_performance}")
    
    # Binary search parameters
    best_edge_count = total_edge_count
    min_edge_count = 1
    max_edge_count = total_edge_count
    step = min(50, total_edge_count // 100)  # Step size for binary search
    
    step_count = 0
    
    # Binary search to find minimal circuit
    best_measurements = baseline_measurements
    while min_edge_count <= max_edge_count:
        step_count += 1
        edge_count = (min_edge_count + max_edge_count) // 2
        
        # Evaluate circuit with current edge count
        outs = run_circuits(model, test_loader, [edge_count], prune_scores, 
                          ablation_type=ablation_type, patch_type=patch_type, reverse_clean_corrupt=reverse_clean_corrupt)
        measurements = {label: metric(model, test_loader, outs) for label, metric in metrics.items()}
        print(measurements)
        _, current_metric = measurements[optimize_metric][0]  # Get the metric for the current edge count
            
        print(f"Edges: {edge_count}/{total_edge_count} ({edge_count/total_edge_count:.2%}), Performance: {current_metric}")
        
        wandb.log({        
            "prune/current_edge_count": edge_count,
            "prune/edge_percentage": edge_count/total_edge_count,
            **{f"prune/{label}": value[0][-1] for label, value in measurements.items()}
        })
        
        # Check if within allowed difference from baseline        
        meets_target = current_metric >= target_performance
        
        if meets_target:
            best_edge_count = edge_count
            best_measurements = measurements
            max_edge_count = edge_count - step  # Try smaller
        else:
            min_edge_count = edge_count + step
    
    
    return best_edge_count, best_measurements


def invert_metric(metric: Callable):
    def inverted_metric(*args, **kwargs):
        results = metric(*args, **kwargs)
        return [[edge_count, -metric_value] for edge_count, metric_value in results]
    return inverted_metric


def measure_faithfulness(
    model: PatchableModel,  
    test_loader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    metric: Callable[[PatchableModel, PromptDataLoader, CircuitOutputs], Measurements],
    model_on_clean: float,
    model_on_corrupt: float,
) -> Measurements:
    """    
    Measures faithfulness by normalizing the metric results against baseline performance.
    
    This function takes a metric function and normalizes its results to a scale where:
    - 0 represents the model's performance on corrupt inputs
    - 1 represents the model's performance on clean inputs
    
    Args:
        model: The model being evaluated
        test_loader: The dataloader containing test data
        circuit_outs: Dictionary mapping edge counts to model outputs
        metric: The metric function to normalize
        model_on_clean: Baseline performance on clean inputs
        model_on_corrupt: Baseline performance on corrupt inputs
        
    Returns:
        Measurements: List of tuples (edge_count, normalized_result) where the 
        normalized_result ranges from 0 to 1  
    """

    measurements = []   
    circuit_metric = metric(model, test_loader, circuit_outs)

    for edge_count,result in circuit_metric:
      
        result = (result - model_on_corrupt) / (model_on_clean - model_on_corrupt)
        measurements.append((edge_count,result))
    return measurements

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find minimal circuits in transformer models")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size for processing")    
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--path", type=str, default="data/results/pile-wikipedia_en/contrastive_pile-wikipedia_en_0.75_gpt-neo-125m_50_50_bleu_divergence_ac_full.json", help="Path to the contrastive dataset")
    parser.add_argument("--output_dir", type=str, default="data/circuits", help="Directory to save circuit results")
    parser.add_argument("--pkl_prefix", type=str, default=None, help="Prefix for the output prune scores file")
    parser.add_argument("--ig", type=int, default=5, help="Number of integrated gradient steps")
    parser.add_argument("--ablation_type", type=str, default="RESAMPLE", choices=["RESAMPLE", "TOKENWISE_MEAN_CLEAN", "TOKENWISE_MEAN_CORRUPT", "TOKENWISE_MEAN_CLEAN_AND_CORRUPT", "BATCH_TOKENWISE_MEAN", "BATCH_ALL_TOK_MEAN", "ZERO"], help="Type of ablation")
    parser.add_argument("--patch_type", type=str, default="EDGE_PATCH", choices=["EDGE_PATCH", "TREE_PATCH"], help="Type of patching")
    parser.add_argument("--grad_function", type=str, default="logit", choices=["logit", "prob", "logprob", "logit_exp"], help="Function to apply to logits before taking the gradient")
    parser.add_argument("--loss_function", type=str, default="avg_diff", choices=["avg_diff", "neg_avg_diff", "avg_val", "mse", "avg_val_wrong"], help="Loss function")
    parser.add_argument("--optimize_metric", type=str, default="logit_diff", choices=["neg_logit_diff", "logit_diff", "answer_logit", "wrong_answer_logit", "answer_nll", "correct_ans_percent", "logit_diff", "neg_logit_diff"], help="Metric to optimize during circuit search")
    parser.add_argument("--reverse_clean_corrupt", action="store_true", help="Reverse the clean and corrupt inputs")
    parser.add_argument("--target_performance", type=float, default=0.85, help="Target performance as a fraction of baseline")
    parser.add_argument("--wandb_project", type=str, default="circuit-discovery", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun")
    
    args = parser.parse_args()   
    
    short_model_name = args.model_name.split("/")[-1]
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map string arguments to enum values
    ablation_type = getattr(AblationType, args.ablation_type)
    patch_type = getattr(PatchType, args.patch_type)    
    
    # Load the dataset
    path = Path(args.path)
    with open(path, 'r') as f:
        data = json.load(f)
    
    train_size = int(0.9*len(data["prompts"]))
    test_size = len(data["prompts"]) - train_size

    # Load and prepare the model
    model = load_tl_model(args.model_name, device)

    # slice_output = (slice(None),slice(50,100))
    slice_output = "last_seq"
    model = patchable_model(
        model,
        factorized=True,
        slice_output=slice_output,
        separate_qkv=True,
        device=device,
        ignore_tokens=[],
        # seq_len=50
    )
    
    # Create train and test dataloaders
    train_loader, test_loader = load_datasets_from_json(
        model, path, device, 
        return_seq_length=False, 
        tail_divergence=False, 
        train_test_size=(train_size, test_size), 
        batch_size=(min(args.batch_size, train_size), min(args.batch_size, test_size)),
        ablation_type=ablation_type
    )
    
    # Path for storing/loading prune scores    
    pkl_prefix = args.pkl_prefix or ""
    prune_scores_path = output_dir / f"{pkl_prefix}{short_model_name}_minimal_circuit_ig{args.ig}_{args.grad_function}_{args.loss_function}.pkl"
    patching_direction = "noising" if args.reverse_clean_corrupt else "denoising"
    results_path = output_dir / f"{pkl_prefix}{short_model_name}_ig{args.ig}_{args.ablation_type}_{args.patch_type}_{args.grad_function}_{args.loss_function}_{patching_direction}.json"

    # Load or compute prune scores
    if results_path.exists() and not args.force_rerun:
        print(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2))
    else:
        # Initialize wandb
        run_name = args.wandb_run_name or f"{short_model_name}_ig{args.ig}_{args.ablation_type}_{args.patch_type}_{args.grad_function}_{args.loss_function}_{patching_direction}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model_name": args.model_name,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "total_edges": model.n_edges,
                "batch_size": args.batch_size,
                "integrated_gradient_steps": args.ig,
                "ablation_type": args.ablation_type,
                "patch_type": args.patch_type,
                "target_performance": args.target_performance,
                "dataset_path": str(path),
                "total_samples": len(data["prompts"]),
                "train_size": train_size,
                "test_size": test_size,
                "grad_function": args.grad_function,
                "loss_function": args.loss_function,
                "reverse_clean_corrupt": args.reverse_clean_corrupt,
                "optimize_metric": args.optimize_metric
            }
        )

        print(f"Computing prune scores with IG steps={args.ig}")
        if args.ig is None:
            mask_val = 0.0
        else:
            mask_val = None
            
        if prune_scores_path.exists():
            print(f"Loading prune scores from {prune_scores_path}")
            prune_scores: PruneScores = t.load(prune_scores_path,weights_only=False)
        else:
            print(f"Computing prune scores with IG steps={args.ig}")
            if args.loss_function == "avg_val_wrong":
                answer_function = partial(batch_avg_answer_val, wrong_answer=True)
            elif args.loss_function == "neg_avg_diff":
                answer_function = partial(batch_avg_answer_diff, wrong_answer=True)
            else:
                answer_function = args.loss_function
            prune_scores: PruneScores = mask_gradient_prune_scores(
                model=model,
                dataloader=train_loader,
                official_edges=None,
                grad_function=args.grad_function,
                answer_function=answer_function,
                mask_val=mask_val, 
                integrated_grad_samples=args.ig,
                ablation_type=ablation_type
            )        
                        
            t.save(prune_scores, prune_scores_path)

        metrics = {
            "answer_logit": measure_answer_val,
            "wrong_answer_logit": partial(measure_answer_val, wrong_answer=True),
            "answer_nll": partial(measure_answer_val, prob_func="log_softmax"),
            "correct_ans_percent": measure_correct_ans_percent,
            "logit_diff": measure_answer_diff,
            "neg_logit_diff": invert_metric(measure_answer_diff)
        }
        
        outs = run_circuits(model, test_loader, [0, model.n_edges], prune_scores,ablation_type=ablation_type, patch_type=patch_type, reverse_clean_corrupt=args.reverse_clean_corrupt)
        baseline_measurements = {label: metric(model, test_loader, outs) for label, metric in metrics.items()}
        
        # noising
        if args.reverse_clean_corrupt:
            _, model_on_clean = baseline_measurements[args.optimize_metric][0] # 0 edges
            _, model_on_corrupt = baseline_measurements[args.optimize_metric][1] # all edges
        # denoising
        else:
            _, model_on_corrupt = baseline_measurements[args.optimize_metric][0] # 0 edges
            _, model_on_clean = baseline_measurements[args.optimize_metric][1] # all edges

        print(f"Model on clean: {model_on_clean}, Model on corrupt: {model_on_corrupt}")

        metrics["faithfulness"] = partial(measure_faithfulness, metric=metrics[args.optimize_metric], model_on_clean=model_on_clean, model_on_corrupt=model_on_corrupt)
        
        # Log to wandb
        wandb.log({        
            "prune/current_edge_count": model.n_edges,
            "prune/edge_percentage": 1,            
            **{f"prune/{label}": value[0][-1] for label, value in baseline_measurements.items()}
        })


        # Find minimal circuit using binary search
        print(f"Finding minimal circuit with target performance: {args.target_performance*100:.0f}% of baseline")    
        min_edge_count, final_metrics = find_minimal_circuit(
            model, test_loader, prune_scores, metrics, ablation_type=ablation_type, patch_type=patch_type,
            target_performance_pct=args.target_performance,
            reverse_clean_corrupt=args.reverse_clean_corrupt,            
        )

        _, final_metric = final_metrics[args.optimize_metric][0]        
        
        print(f"Found minimal circuit with {min_edge_count} ({min_edge_count/model.n_edges:.2%}) edges that's optimized for {args.optimize_metric} with {final_metric:.2%}")
        
        # Visualize the minimal circuit
        edge_counts = [10,20,50,100,min_edge_count]
        circuit_images = []
        
        for edge_count in edge_counts:
            threshold = prune_scores_threshold(prune_scores, edge_count)
            fig_path = output_dir / f"{args.model_name.split('/')[-1]}_minimal_circuit_ig{args.ig}_p{patch_type.name}_edge{edge_count}.png"
            
            # Draw the circuit graph and save to file
            draw_seq_graph(
                model, prune_scores, threshold.item(), layer_spacing=True, orientation="v", display_ipython=False, file_path=fig_path
            )
            
            # Log the circuit image to wandb
            circuit_images.append(wandb.Image(
                str(fig_path),  
                caption=f"Circuit with {edge_count} edges ({edge_count/model.n_edges:.2%})"
            ))
        
        # Log all circuit images to wandb
        wandb.log({"circuit_visualizations": circuit_images})
                
        # Save results to a JSON file
        results = {
            "model": args.model_name,
            "total_edges": model.n_edges,
            "minimal_edges": min_edge_count,
            "edge_percentage": min_edge_count/model.n_edges,
            "final_metric": final_metric,
            "target_performance": args.target_performance,
            "integrated_gradient_steps": args.ig,
            "ablation_type": args.ablation_type,
            "patch_type": args.patch_type,
            "grad_function": args.grad_function,
            "loss_function": args.loss_function
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save the results as an artifact
        results_artifact = wandb.Artifact(
            name=f"circuit_results_{run_name}", 
            type="results",
            description=f"Circuit finding results for {args.model_name}"
        )
        results_artifact.add_file(str(results_path))
        wandb.log_artifact(results_artifact)
        
        print(f"Saved results to {results_path}")

         # Save final results to wandb.summary
        wandb.summary["min_edge_count"] = min_edge_count
        wandb.summary["min_edge_percentage"] = min_edge_count/model.n_edges
        wandb.summary["final_metric"] = final_metric
        
        # Finish the wandb run
        wandb.finish()