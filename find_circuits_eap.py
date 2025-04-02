from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
import json
from pathlib import Path

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

def kl_div(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
path = Path(f"data/results/contrastive_mem_0.5_pythia-70m-deduped_bleu_ac.json")

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset(path, model.tokenizer) 
# Create train and test dataloaders
train_dataloader, test_dataloader = ds.to_dataloader(batch_size=32, test_size=0.1, seed=42) 

g = Graph.from_model(model)
# Attribute using the training data
attribute(model, g, train_dataloader, 
          partial(logit_diff, loss=True, mean=True),
          method='EAP-IG-activations',
          ig_steps=5)

# Evaluate baseline on the test data
baseline = evaluate_baseline(model, test_dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Baseline performance: {baseline}")


# Find minimal circuit that maintains 80% performance of baseline
# Evaluation is done on the test set
target_performance = 0.8 * baseline
best_n = 3000  # Start with a large number
min_n = 1
max_n = 3000
step = 10

# Binary search to find minimal circuit
while min_n <= max_n:
    n = (min_n + max_n) // 2    
    g.apply_topn(n, True)
    # Evaluate on test data
    performance = evaluate_graph(model, g, test_dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    
    if performance >= target_performance:
        best_n = n
        max_n = n - step  # Try smaller
    else:
        min_n = n + step  # Try larger
    print(f"n: {n}, performance: {performance}")

# Apply the minimal circuit size that maintains performance
g.apply_topn(best_n, True)
# Final evaluation on test data
results = evaluate_graph(model, g, test_dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Found minimal circuit with {best_n} edges that maintains {results/baseline:.2%} of baseline performance")

print("Number of included nodes:", g.count_included_nodes())
g.to_json(f'graph-{MODEL_NAME.split("/")[-1]}-{results}.json')
gz = g.to_graphviz(f'graph-{MODEL_NAME.split("/")[-1]}-{results}.png')