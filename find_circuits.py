import torch as t
from pathlib import Path
import json
import argparse
import wandb
import matplotlib.pyplot as plt
from typing import Callable, Literal
from functools import partial
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType, PatchType, PruneScores
from auto_circuit.utils.graph_utils import patchable_model,edge_counts_util
from auto_circuit.data import load_datasets_from_json,PromptDataLoader
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune import run_circuits
from auto_circuit.metrics.prune_metrics.answer_value import measure_answer_val
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.patchable_model import PatchableModel
device = t.device("cuda" if t.cuda.is_available() else "mps")

def find_minimal_circuit(
        model: PatchableModel, 
        test_loader: PromptDataLoader, 
        prune_scores: PruneScores, 
        metrics: list[Callable],
        ablation_type: AblationType, 
        patch_type: PatchType, 
        target_performance_pct: float=0.8,
        invert_search_direction: bool=False,
        reverse_clean_corrupt: bool=False,
        metric_labels: list[str]=None):
    # Evaluate baseline performance with all edges
    total_edge_count = model.n_edges
    outs = run_circuits(model, test_loader, [total_edge_count], prune_scores,ablation_type=ablation_type, patch_type=patch_type, reverse_clean_corrupt=reverse_clean_corrupt)
    baseline_measurements = [metric(model, test_loader, outs) for metric in metrics]
    _, baseline_metric = baseline_measurements[0][-1]
    if metric_labels is None:
        metric_labels = [f"metric_{i}" for i in range(len(metrics))]
    print(f"Baseline metric: {baseline_metric}")
    
    # Log to wandb
    wandb.log({        
        "prune/current_edge_count": total_edge_count,
        "prune/edge_percentage": 1,        
        "prune/performance_ratio": 1,
        "prune/diff_from_baseline": 0,
        **{f"prune/{label}": baseline_measurements[i][0][-1] for i, label in enumerate(metric_labels)}
    })
    
    # Calculate maximum allowed difference from baseline
    target_performance = target_performance_pct * baseline_metric
    # max_allowed_diff = (1 - target_performance_pct) * abs(baseline_metric)
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
        measurements = [metric(model, test_loader, outs) for metric in metrics]
        print(measurements)
        _, current_metric = measurements[0][0]  # Get the metric for the current edge count
        
        # Calculate difference
        diff = abs(current_metric - baseline_metric)
        
        performance_ratio = 1 - (diff / abs(baseline_metric))
        print(f"Edges: {edge_count}/{total_edge_count} ({edge_count/total_edge_count:.2%}), Performance: {current_metric}/{baseline_metric}, Diff: {diff}, Ratio: {performance_ratio:.2%}")
        
        wandb.log({        
            "prune/current_edge_count": edge_count,
            "prune/edge_percentage": edge_count/total_edge_count,
            "prune/performance_ratio": performance_ratio,
            "prune/diff_from_baseline": diff,
            **{f"prune/{label}": measurements[i][0][-1] for i, label in enumerate(metric_labels)}
        })
        
        # Check if within allowed difference from baseline
        meets_target = current_metric >= target_performance #diff <= max_allowed_diff
        
        if meets_target:
            best_edge_count = edge_count
            best_measurements = measurements
            if invert_search_direction:
                min_edge_count = edge_count + step
            else:
                max_edge_count = edge_count - step  # Try smaller
        else:
            if invert_search_direction:
                # When inverted, REDUCE edges when diff is too large
                max_edge_count = edge_count - step
            else:
                # Normal behavior: INCREASE edges when diff is too large
                min_edge_count = edge_count + step
    
    return best_edge_count, best_measurements, baseline_metric


def invert_metric(metric: Callable):
    def inverted_metric(*args, **kwargs):
        results = metric(*args, **kwargs)
        return [[edge_count, -metric_value] for edge_count, metric_value in results]
    return inverted_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find minimal circuits in transformer models")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")    
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--path", type=str, default="data/results/pile-wikipedia_en/contrastive_pile-wikipedia_en_0.75_gpt-neo-125m_50_50_bleu_divergence_ac_full.json", help="Path to the contrastive dataset")
    parser.add_argument("--output_dir", type=str, default="data/circuits", help="Directory to save circuit results")
    parser.add_argument("--ig", type=int, default=5, help="Number of integrated gradient steps")
    parser.add_argument("--ablation_type", type=str, default="RESAMPLE", choices=["RESAMPLE", "MEAN", "ZERO"], help="Type of ablation")
    parser.add_argument("--patch_type", type=str, default="EDGE_PATCH", choices=["EDGE_PATCH", "TREE_PATCH"], help="Type of patching")
    parser.add_argument("--grad_function", type=str, default="logit", choices=["logit", "prob", "logprob", "logit_exp"], help="Function to apply to logits before taking the gradient")
    parser.add_argument("--loss_function", type=str, default="avg_diff", choices=["avg_diff", "avg_val", "mse"], help="Loss function")
    parser.add_argument("--reverse_clean_corrupt", type=bool, default=False, help="Reverse the clean and corrupt inputs")
    parser.add_argument("--target_performance", type=float, default=0.85, help="Target performance as a fraction of baseline")
    parser.add_argument("--wandb_project", type=str, default="circuit-discovery", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun")
    
    args = parser.parse_args()   
    
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
        batch_size=args.batch_size,        
    )
    
    # Path for storing/loading prune scores
    prune_scores_path = output_dir / f"{args.model_name.split('/')[-1]}_minimal_circuit_ig{args.ig}.pkl"
    results_path = output_dir / f"{args.model_name.split('/')[-1]}_minimal_circuit_ig{args.ig}_p{patch_type.name}.json"

    # Load or compute prune scores
    if results_path.exists() and not args.force_rerun:
        print(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2))
    else:
         # Initialize wandb
        run_name = args.wandb_run_name or f"{args.model_name.split('/')[-1]}_ig{args.ig}_{args.ablation_type}_{args.patch_type}_{args.grad_function}_{args.loss_function}"
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
                "reverse_clean_corrupt": args.reverse_clean_corrupt
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
            prune_scores: PruneScores = mask_gradient_prune_scores(
                model=model,
                dataloader=train_loader,
                official_edges=None,
                grad_function=args.grad_function,
                answer_function=args.loss_function,
                mask_val=mask_val, 
                integrated_grad_samples=args.ig,
                ablation_type=ablation_type
            )        
            
            t.save(prune_scores, prune_scores_path)

        metrics = [measure_answer_diff, partial(measure_answer_val, prob_func="log_softmax")] #invert_metric(measure_answer_diff)
        metric_labels = ["logit_diff", "nll"]
        # Find minimal circuit using binary search
        print(f"Finding minimal circuit with target performance: {args.target_performance*100:.0f}% of baseline")
        

        min_edge_count, final_metrics, baseline_metric = find_minimal_circuit(
            model, test_loader, prune_scores, metrics,
            ablation_type=ablation_type, patch_type=patch_type,
            target_performance_pct=args.target_performance,
            reverse_clean_corrupt=args.reverse_clean_corrupt,
            metric_labels=metric_labels
        )

        _, final_metric = final_metrics[0][0]
        performance_ratio = final_metric/baseline_metric
        
        print(f"Found minimal circuit with {min_edge_count} ({min_edge_count/model.n_edges:.2%}) edges that maintains {performance_ratio:.2%} of baseline performance")
        
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
            "baseline_metric": baseline_metric,
            "final_metric": final_metric,
            "performance_percentage": final_metric/baseline_metric,
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
        
        # Finish the wandb run
        wandb.finish()