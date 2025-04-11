import torch as t
from pathlib import Path
import json
import argparse
from typing import Callable
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
        target_performance_pct=0.8):
    # Evaluate baseline performance with all edges
    total_edge_count = model.n_edges
    outs = run_circuits(model, test_loader, [total_edge_count], prune_scores, 
                       ablation_type=ablation_type, patch_type=patch_type)
    baseline_measurements = [metric(model, test_loader, outs) for metric in metrics]
    _, baseline_metric = baseline_measurements[0][-1]
    
    # Calculate target performance
    target_performance = target_performance_pct * baseline_metric
    
    # Binary search parameters
    best_edge_count = total_edge_count
    min_edge_count = 1
    max_edge_count = total_edge_count
    step = max(1, total_edge_count // 100)  # Step size for binary search
    
    # Binary search to find minimal circuit
    while min_edge_count <= max_edge_count:
        edge_count = (min_edge_count + max_edge_count) // 2
        
        # Evaluate circuit with current edge count
        outs = run_circuits(model, test_loader, [edge_count], prune_scores, 
                          ablation_type=ablation_type, patch_type=patch_type)
        measurements = [metric(model, test_loader, outs) for metric in metrics]
        print(measurements)
        _, current_metric = measurements[0][0]  # Get the metric for the current edge count
        
        print(f"Edges: {edge_count}/{total_edge_count} ({edge_count/total_edge_count:.2%}), Performance: {current_metric}/{baseline_metric} ({current_metric/baseline_metric:.2%})")
        
        if current_metric >= target_performance:
            best_edge_count = edge_count
            max_edge_count = edge_count - step  # Try smaller
        else:
            min_edge_count = edge_count + step  # Try larger
    
    return best_edge_count, measurements, baseline_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find minimal circuits in transformer models")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")    
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--path", type=str, default="data/results/pile-wikipedia_en/contrastive_pile-wikipedia_en_0.75_gpt-neo-125m_50_50_bleu_divergence_ac_full.json", help="Path to the contrastive dataset")
    parser.add_argument("--output_dir", type=str, default="data/circuits", help="Directory to save circuit results")
    parser.add_argument("--ig", type=int, default=5, help="Number of integrated gradient steps")
    parser.add_argument("--ablation_type", type=str, default="RESAMPLE", choices=["RESAMPLE", "MEAN", "ZERO"], help="Type of ablation")
    parser.add_argument("--patch_type", type=str, default="EDGE_PATCH", choices=["EDGE_PATCH", "NODE_PATCH"], help="Type of patching")
    parser.add_argument("--target_performance", type=float, default=0.85, help="Target performance as a fraction of baseline")
    
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
    results_path = output_dir / f"{args.model_name.split('/')[-1]}_minimal_circuit_ig{args.ig}_p{patch_type.name}.json"    

    # Load or compute prune scores
    if results_path.exists():
        print(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(json.dumps(results, indent=2))
    else:
        print(f"Computing prune scores with IG steps={args.ig}")
        if args.ig is None:
            mask_val = 0.0
        else:
            mask_val = None
    
        prune_scores_path = results_path.with_suffix(".pkl")
        if prune_scores_path.exists():
            print(f"Loading prune scores from {prune_scores_path}")
            prune_scores: PruneScores = t.load(prune_scores_path,weights_only=False)
        else:
            print(f"Computing prune scores with IG steps={args.ig}")
            prune_scores: PruneScores = mask_gradient_prune_scores(
                model=model,
                dataloader=train_loader,
                official_edges=None,
                grad_function="logit",
                answer_function="avg_diff",
            mask_val=mask_val, 
            integrated_grad_samples=args.ig,
            ablation_type=ablation_type
            )
            t.save(prune_scores, prune_scores_path)
    

        metrics = [measure_answer_diff, partial(measure_answer_val, prob_func="log_softmax")]
        # Find minimal circuit using binary search
        print(f"Finding minimal circuit with target performance: {args.target_performance*100:.0f}% of baseline")
        min_edge_count, final_metrics, baseline_metric = find_minimal_circuit(
            model, test_loader, prune_scores, metrics,
            ablation_type=ablation_type, patch_type=patch_type,
            target_performance_pct=args.target_performance
        )

        _, final_metric = final_metrics[0][0]
        print(f"Found minimal circuit with {min_edge_count} ({min_edge_count/model.n_edges:.2%}) edges that maintains {final_metric/baseline_metric:.2%} of baseline performance")    
        
        # Visualize the minimal circuit
        threshold = prune_scores_threshold(prune_scores, min_edge_count)
        fig_path = output_dir / f"{args.model_name.split('/')[-1]}_minimal_circuit_ig{args.ig}_p{patch_type.name}.png"
        fig = draw_seq_graph(
            model, prune_scores, threshold.item(), layer_spacing=True, orientation="v", display_ipython=False, file_path=fig_path
        )
                
        # Save results to a JSON file        
        with open(results_path, 'w') as f:
            json.dump({
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
                "patch_type": args.patch_type
            }, f, indent=2)
        
        print(f"Saved results to {results_path}")