import torch as t
from pathlib import Path
import json
import argparse
import wandb
import matplotlib.pyplot as plt
from typing import Callable, Literal
from functools import partial
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType, PatchType, PruneScores,PatchWrapper
from auto_circuit.utils.graph_utils import patchable_model,edge_counts_util,patch_mode
from auto_circuit.utils.tensor_ops import desc_prune_scores,prune_scores_threshold,flat_prune_scores
from auto_circuit.data import load_datasets_from_json,PromptDataLoader,PromptPairBatch
from auto_circuit.prune import run_circuits
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.metrics.prune_metrics.answer_value import measure_answer_val
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.metrics.prune_metrics.correct_answer_percent import measure_correct_ans_percent
from auto_circuit.prune_algos.random_edges import random_prune_scores
from collections import defaultdict
from tqdm import tqdm
from auto_circuit.utils.misc import module_by_name

from typing import Optional
from torch.utils.data import Dataset

from find_circuits import METRICS, measure_faithfulness, invert_metric

device = t.device("cuda" if t.cuda.is_available() else "mps")

class SingleSampleDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.sample

def remove_padding(sample: dict[str, t.Tensor]):
    # Remove padding tokens using attention mask
    input_ids = sample["input_ids"]  
    attention_mask = sample["attention_mask"]          
    first_non_pad_pos = attention_mask.argmax()    
    # Extract the non-padded part of the sequence
    input_ids = input_ids[first_non_pad_pos:].unsqueeze(0)
    
    return {"input": input_ids, "attention_mask": t.ones_like(input_ids)}

def generate_with_circuit(
    model: PatchableModel,
    input_tokens: t.Tensor,
    prune_scores: PruneScores,
    edge_count: int,
    n_tokens: int,
    dataloader: PromptDataLoader,
    patch_type: PatchType = PatchType.EDGE_PATCH,
    ablation_type: AblationType = AblationType.ZERO,    
    reverse_clean_corrupt: bool = False,
) -> t.Tensor:
    """Generate tokens using a pruned circuit.

    Args:
        model: The model to use for generation.
        input_tokens: The input token IDs (shape: [batch_size, seq_len]).
        prune_scores: The scores that determine the ordering of edges for pruning.
        edge_count: The number of edges to include in the circuit.
        n_tokens: The number of tokens to generate.
        dataloader: The dataloader to use for patches.
        patch_type: Whether to patch the circuit or the complement.
        ablation_type: The type of ablation to use.
        reverse_clean_corrupt: Reverse clean and corrupt (for input and patches).

    Returns:
        The generated token IDs (shape: [batch_size, seq_len + n_tokens]).
    """

    
    # Create initial sample with the input tokens
    current_sample = PromptPairBatch(
        clean={"input_ids": input_tokens, "attention_mask": t.ones_like(input_tokens)},
        corrupt=dataloader.dataset[0].corrupt,
        key=0,
        batch_diverge_idx=0,
        answers=dataloader.dataset[0].answers,
        wrong_answers=dataloader.dataset[0].wrong_answers,
    )
    
    print(f"Starting token generation with circuit of {edge_count} edges")
    generated_tokens = input_tokens.clone()
    for i in tqdm(range(n_tokens), desc="Generating tokens"):
        # Create a new dataloader with just this sample
        current_dataloader = PromptDataLoader(
            SingleSampleDataset(current_sample),
            seq_len=None,  # Don't truncate
            diverge_idx=dataloader.diverge_idx,
            kv_cache=dataloader.kv_cache,
            seq_labels=dataloader.seq_labels,
            word_idxs=dataloader.word_idxs,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        
        # Run the circuit to get the next token
        outs = run_circuits(
            model, 
            current_dataloader, 
            [edge_count], 
            prune_scores,
            ablation_type=ablation_type, 
            patch_type=patch_type, 
            reverse_clean_corrupt=reverse_clean_corrupt
        )
        
        # Get the model output for the current sequence
        next_token_logits = list(outs[edge_count].values())[0]        
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
        
        # Append the predicted token to the sequence
        generated_tokens = t.cat([generated_tokens, next_token], dim=1)
        
        # Update the current sample with the new sequence
        current_sample.clean["input_ids"] = generated_tokens
        current_sample.clean["attention_mask"] = t.ones_like(generated_tokens)
        
        # Print the token generated in this step
        print(f"{model.tokenizer.decode(next_token[0])}", end="")
    
    return generated_tokens


def get_random_circuits(model: PatchableModel, test_loader: PromptDataLoader, prune_scores: PruneScores, edge_count: int, count: int=1):
    
    circuit_threshold = prune_scores_threshold(prune_scores, edge_count)         
    circuit_edges = set()
    for edge in model.edges:
        true_score = prune_scores[edge.dest.module_name][edge.patch_idx]    
        if true_score.item() >= circuit_threshold.item():
            circuit_edges.add(edge)
    

    print(f"Generating {count} random prune scores for evaluation...")    
    random_circuits = []
    for _ in range(count):
            
        random_ps = random_prune_scores(model, test_loader)
        # Get threshold for top edges in each method
        random_threshold = prune_scores_threshold(random_ps, edge_count)
        
        # Extract edges above thresholds
        random_edges = set()
        for edge in model.edges:          
            random_score = random_ps[edge.dest.module_name][edge.patch_idx]            
            if random_score.item() >= random_threshold.item():
                random_edges.add(edge)
  
        # Calculate intersection
        intersection = random_edges.intersection(circuit_edges)
        random_circuits.append(random_ps)    
        print(f"Number of random edges intersected with actual circuit: {len(intersection)} ({len(intersection)/edge_count*100:.2f}%)")
    return random_circuits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find minimal circuits in transformer models")
    parser.add_argument("--prune_scores_path", type=str, default="data/circuits/divergence_gpt-neo-125m_minimal_circuit_ig5_logit_avg_val.pkl", help="Path to the prune scores file")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size for processing")    
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--dataset_path", type=str, default="data/results/full-wiki_embeddings/contrastive_pile-wikipedia_en_1.0_gpt-neo-125m_50_50_bleu_dataset.json", help="Path to the contrastive dataset")
    parser.add_argument("--dataset_size", type=float, default=0.1, help="Fraction of prompts to use from the dataset")
    parser.add_argument("--edge_count", type=int, default=10, help="Number of edges to prune")
    parser.add_argument("--random_circuits", type=int, default=1, help="Number of random circuits to generate")
    parser.add_argument("--ablation_type", type=str, default="RESAMPLE", choices=["RESAMPLE", "TOKENWISE_MEAN_CLEAN", "TOKENWISE_MEAN_CORRUPT", "TOKENWISE_MEAN_CLEAN_AND_CORRUPT", "BATCH_TOKENWISE_MEAN", "BATCH_ALL_TOK_MEAN", "ZERO"], help="Type of ablation")
    parser.add_argument("--patch_type", type=str, default="EDGE_PATCH", choices=["EDGE_PATCH", "TREE_PATCH"], help="Type of patching")
    parser.add_argument("--reverse_clean_corrupt", action="store_true", help="Reverse the clean and corrupt inputs")
    parser.add_argument("--optimize_metric", type=str, default="logit_diff", choices=list(METRICS.keys()), help="Metric to optimize")
    args = parser.parse_args()   
    
    short_model_name = args.model_name.split("/")[-1]
    
    # Map string arguments to enum values
    ablation_type = getattr(AblationType, args.ablation_type)
    patch_type = getattr(PatchType, args.patch_type)    
    
    # Load the dataset
    path = Path(args.dataset_path)
    with open(path, 'r') as f:
        data = json.load(f)
    
    test_size = int(args.dataset_size*len(data["prompts"]))
    train_size = len(data["prompts"]) - test_size

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
    
    _, test_loader = load_datasets_from_json(
        model, path, device, 
        return_seq_length=False, 
        tail_divergence=False, 
        train_test_size=(train_size, test_size), 
        batch_size=(max(min(args.batch_size, train_size),1), max(min(args.batch_size, test_size),1)),
        ablation_type=ablation_type
    )
    print(f"Evaluating on dataset: {args.dataset_path} with {args.dataset_size*100}% of prompts")

    prune_scores: PruneScores = t.load(args.prune_scores_path,weights_only=False)
    
    # TODO: Implement generate() with circuit
    # sample = test_loader.dataset[0]    
    # input_ids = remove_padding(sample.clean)["input"]
    # patch_dataloader = test_loader
    # # if ablation_type.mean_over_dataset:
    # # else:
    # #     patch_dataloader = sample.corrupt        
    # #     patch_dataloader = remove_padding(patch_dataloader)   

    # print("Generating with circuit from:")
    # print(model.to_string(input_ids))
    # generated_tokens = generate_with_circuit(model, input_ids, prune_scores, args.edge_count, 16, patch_dataloader, patch_type=patch_type, ablation_type=ablation_type)
    # print("Generated:")
    # print(model.to_string(generated_tokens))

    base_ablation = AblationType.RESAMPLE
    outs = run_circuits(model, test_loader, [0, model.n_edges], prune_scores,ablation_type=base_ablation, patch_type=patch_type, reverse_clean_corrupt=args.reverse_clean_corrupt, verbose=False)
    baseline_measurements = {label: metric(model, test_loader, outs) for label, metric in METRICS.items()}
    
    # noising
    if args.reverse_clean_corrupt:
        # Make this vice versa as our baseline is on corrupt for noising (for both datasets)
        _, model_on_clean = baseline_measurements[args.optimize_metric][1] # all edges
        _, model_on_corrupt = baseline_measurements[args.optimize_metric][0] # 0 edges
    # denoising
    else:
        _, model_on_corrupt = baseline_measurements[args.optimize_metric][0] # 0 edges
        _, model_on_clean = baseline_measurements[args.optimize_metric][1] # all edges

    print("Baseline measurements:")
    print(f"Model on clean: {model_on_clean}, Model on corrupt: {model_on_corrupt}")
    print(json.dumps(baseline_measurements, indent=4))

    
    METRICS["faithfulness"] = partial(measure_faithfulness, metric=METRICS[args.optimize_metric], model_on_clean=model_on_clean, model_on_corrupt=model_on_corrupt)
        
    # random_circuits = get_random_circuits(model, test_loader, prune_scores, args.edge_count, args.random_circuits)

    # random_results = defaultdict(lambda: defaultdict(list))
    # for random_ps in random_circuits:
    #     random_outs = run_circuits(model, test_loader, [args.edge_count], random_ps,ablation_type=ablation_type, patch_type=patch_type, reverse_clean_corrupt=args.reverse_clean_corrupt)
    #     m = METRICS["faithfulness"]
    #     measurements = m(model, test_loader, random_outs)
    #     for edge_count, metric_value in measurements:
    #         random_results[edge_count]["faithfulness"].append(metric_value)
    # for edge_count, results in random_results.items():
    #     for label, values in results.items():
    #         random_results[edge_count][label] = sum(values) / len(values)
    # print("Random circuits results:")
    # print(json.dumps(random_results, indent=4))
    
    outs = run_circuits(model, test_loader, [args.edge_count], prune_scores,ablation_type=ablation_type, patch_type=patch_type, reverse_clean_corrupt=args.reverse_clean_corrupt,verbose=False)

    results = defaultdict(list)
    for label, metric in METRICS.items():
        measurements = metric(model, test_loader, outs)
        for edge_count, metric_value in measurements:
            results[edge_count].append((label, metric_value))

    for edge_count, metric_values in results.items():
        print(f"# Edges: {edge_count}")
        for label, metric_value in metric_values:
            print(f"* {label}: {metric_value}")
        print()
