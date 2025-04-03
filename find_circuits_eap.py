from functools import partial
from typing import Callable
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
import json
from pathlib import Path
from torch.nn.functional import kl_div

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute

def collate_EAP(xs):
    clean, corrupted, token_id_pairs = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(list(token_id_pairs))

    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath:Path, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer        
        with open(filepath, 'r') as f:
            data = json.load(f)
        prompts = data['prompts']
        self.df = pd.DataFrame(prompts)
        # Rename 'corrupt' column to 'corrupted' to match downstream expectations
        self.df.rename(columns={'corrupt': 'corrupted'}, inplace=True)
        # Extract first correct and incorrect answer strings
        self.df['correct_answer'] = self.df['answers'].apply(lambda x: x[0] if x else None)
        self.df['incorrect_answer'] = self.df['wrong_answers'].apply(lambda x: x[0] if x else None)
        # Drop original list columns
        self.df.drop(columns=['answers', 'wrong_answers'], inplace=True, errors='ignore')
        if 'label' in self.df.columns:
            self.df.drop(columns=['label'], inplace=True)

        # --- Pre-tokenize answers --- 
        correct_answers = self.df['correct_answer'].tolist()
        incorrect_answers = self.df['incorrect_answer'].tolist()

        # Batched tokenization
        # Handle potential None values if answers are missing
        correct_tokenized = self.tokenizer([ans if ans is not None else "" for ans in correct_answers], add_special_tokens=False)
        incorrect_tokenized = self.tokenizer([ans if ans is not None else "" for ans in incorrect_answers], add_special_tokens=False)

        # Extract first token ID for each answer (fallback to UNK)
        self.df['correct_token_id'] = [ids[0] if ids else self.tokenizer.unk_token_id 
                                        for ids in correct_tokenized['input_ids']]
        self.df['incorrect_token_id'] = [ids[0] if ids else self.tokenizer.unk_token_id 
                                            for ids in incorrect_tokenized['input_ids']]
        
        # Drop intermediate string columns
        self.df.drop(columns=['correct_answer', 'incorrect_answer'], inplace=True)
            # --- End Pre-tokenization ---

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Return clean, corrupted, and the tuple of token IDs
        return row['clean'], row['corrupted'], (row['correct_token_id'], row['incorrect_token_id'])
    
    def to_dataloader(self, batch_size: int, test_size: float = 0.1, seed: int = 42): # No tokenizer needed here
        """Splits the dataset into train and test sets and returns corresponding dataloaders."""
        dataset_size = len(self)
        test_split_size = int(test_size * dataset_size)
        train_split_size = dataset_size - test_split_size

        # Use a generator for reproducible splits
        generator = Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(
            self, [train_split_size, test_split_size], generator=generator
        )

        # Pass the simplified collate_EAP directly
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_EAP)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_EAP)
        
        return train_loader, test_loader

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def kl_div_(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    results = kl_div(logits.log_softmax(dim=-1), clean_logits.softmax(dim=-1), reduction='none').mean(-1)
    return results.mean() if mean else results

def find_minimal_circuit(model: HookedTransformer, g: Graph, test_dataloader: DataLoader,  metrics: list[Callable], target_performance_pct :float= 0.8) -> tuple[Graph,int]:
    # Find minimal circuit that maintains 80% performance of baseline
    # Evaluation is done on the test set
    target_performance = target_performance_pct * baseline_logit_diff
    best_n = len(g.edges)  # Start with a total number of edges
    min_n = 1
    max_n = best_n
    step = 10

    # Binary search to find minimal circuit
    while min_n <= max_n:
        n = (min_n + max_n) // 2    
        g.apply_topn(n, True)
        # Evaluate on test data
        results = evaluate_graph(model, g, test_dataloader, metrics,skip_clean=False)
        if not isinstance(results, list):
            results = [results]
        results = [result.mean().item() for result in results]
        
        performance = results[0] # assume main metric (logit_diff) is the first metric

        if performance >= target_performance:
            best_n = n
            max_n = n - step  # Try smaller
        else:
            min_n = n + step  # Try larger
        print(f"edges: {n}, Results: {results}")

    # Apply the minimal circuit size that maintains performance
    g.apply_topn(best_n, True)
    return g,best_n


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")    
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--path", type=str, default="data/results/contrastive_mem_0.5_gpt-neo-125m_50_50_bleu_filtered100.json", help="Path to the contrastive dataset")
    parser.add_argument("--output_dir", type=str, default="data/circuits", help="Directory to save circuit results")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    model_name = args.model_name
    path = Path(args.path)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)    

    model = HookedTransformer.from_pretrained(model_name, device=device,dtype="float16")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ds = EAPDataset(path, model.tokenizer) 
    # Create train and test dataloaders
    train_dataloader, test_dataloader = ds.to_dataloader(batch_size=batch_size, test_size=0.1, seed=42) 


    # Evaluate baseline on the test data
    baseline_logit_diff = evaluate_baseline(model, test_dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    print(f"Baseline logit_diff: {baseline_logit_diff}")

    g = Graph.from_model(model)
    # Attribute using the training data
    attribute(model, g, train_dataloader, 
            partial(logit_diff, loss=True, mean=True),
            method='EAP-IG-activations',
            ig_steps=5)

    eval_metrics = [partial(logit_diff, loss=False, mean=False),partial(kl_div_, loss=False, mean=False)]

    g, best_edges = find_minimal_circuit(model, g, test_dataloader, eval_metrics, target_performance_pct=0.8)
    # Final evaluation on test data
    logit_diff_val, kl_val = evaluate_graph(model, g, test_dataloader, eval_metrics,skip_clean=False)
    logit_diff_val = logit_diff_val.mean().item()
    kl_val = kl_val.mean().item()
    print(f"Found minimal circuit with {best_edges} ({best_edges/len(g.edges):.2%}) edges that maintains {logit_diff_val/baseline_logit_diff:.2%} of baseline performance")

    print("Number of included nodes:", g.count_included_nodes())
    
    # Generate filenames with dataset info
    circuit_json = output_dir / f'graph-{path.stem}.json'
    circuit_viz = output_dir / f'graph-{path.stem}.png'
    
    # Save circuit
    g.to_json(str(circuit_json))
    gz = g.to_graphviz(str(circuit_viz))