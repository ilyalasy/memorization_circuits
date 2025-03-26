# %%
import pandas as pd
import torch as t
from pathlib import Path
from transformers import AutoTokenizer,AutoModel, GPTNeoXForCausalLM
from tqdm.auto import tqdm
import numpy as np
from pandarallel import pandarallel
from torch.utils.data import DataLoader,Dataset
from information_flow.information_flow import build_full_graph, get_best_tokens
import json
from pathlib import Path
from typing import Callable
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def escape(token:str):
    token = token.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    token = ''.join(c if ord(c) >= 32 else f'\\x{ord(c):02x}' for c in token)
    return token    

def plot_sample(tokens:list[str], contributions:list[float], cmap="cool"):    
    # Track min/max contributions for global normalization    
    
    global_min = min(contributions) 
    global_max = max(contributions)
    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)    
    # Pixels where you want to start first word
    start_x = 20
    start_y = 50

    # Whitespace in pixels
    whitespace = 20

    # Create figure
    figure = plt.figure(figsize=(14,1))

    # From renderer we can get textbox width in pixels
    rend = figure.canvas.get_renderer()
    for token, contrib in zip(tokens, contributions):
        token = escape(token)
        # Text box parameters and colors
        color = plt.cm.get_cmap(cmap)(norm(contrib))
        bbox = dict(boxstyle="round,pad=0.3", fc=color, ec="black", lw=1)

        txt = plt.text(start_x, start_y, token, color="black", bbox=bbox,fontsize=9,transform=None)        
        # Textbox width
        bb = txt.get_window_extent(renderer=rend)

        # Calculate where next word should be written
        start_x = bb.width + start_x + whitespace

    # Skip plotting axis
    plt.axis("off")
    # Save and plot figure
    return figure

# Calculate and plot average contributions across token positions
def plot_average_contributions(token_contributions, max_token_idx=32):
    # Initialize array to store sum of contributions and count for each position
    total_contributions = np.zeros(max_token_idx+1)
    contribution_counts = np.zeros(max_token_idx+1)
    
    # Sum up all contributions by position
    for contributions_dict, _ in token_contributions:
        for pos, value in contributions_dict.items():
            if pos <= max_token_idx:
                total_contributions[pos] += value
                contribution_counts[pos] += 1
    
    # Calculate averages (avoid division by zero)
    average_contributions = np.zeros(max_token_idx+1)
    for i in range(max_token_idx+1):
        if contribution_counts[i] > 0:
            average_contributions[i] = total_contributions[i] / contribution_counts[i]
    
    # Plot the averages
    plt.figure(figsize=(12, 6))
    plt.bar(range(max_token_idx+1), average_contributions)
    plt.xlabel('Token Position')
    plt.ylabel('Average Contribution')
    plt.title('Average Token Contribution by Position (0-32)')
    plt.xticks(range(0, max_token_idx+1, 2))  # Show every 2nd position for readability
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path("data/results/average_contributions.png"))
    plt.show()
    
    return average_contributions

pandarallel.initialize(nb_workers=16,progress_bar=True)

# %%
pile_df = pd.read_json("data/pile_samples.jsonl",lines=True)
pile_df.head()

# %%
device = t.device("cuda" if t.cuda.is_available() else "mps")

# %%
MODEL_NAME = "EleutherAI/pythia-70m-deduped"

# %%
dfs = [pd.read_parquet(f"data/mem/memorization_70m-deduped-v0_143000_rank{rank}.parquet") for rank in range(4)]
df = pd.concat(dfs, ignore_index=True)

# %%
df_perfect = df[df['acc'] == 1]

class PileDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        return t.tensor(self.df.iloc[idx]["context"]),t.tensor(self.df.iloc[idx]["true_continuation"])
    

# %%


model = HookedTransformer.from_pretrained(MODEL_NAME, device='mps')
model.cfg.use_split_qkv_input = False
model.cfg.use_attn_in = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ids = [3823,  4213,  4462,  4490,  4565,  4599,  9001,  9068,  9503,  9724,  9844,  9856,  9906,  14050,  14166,  14371,  14533,  14537,  19050,  19439,  19603,  19858,  19891,  20000]
samples = df_perfect[df_perfect.index.isin(ids)]

samples['decoded_context'] = model.to_str_tokens(samples['context'].to_numpy().tolist())
samples['decoded_continuation'] = model.to_str_tokens(samples['true_continuation'].to_numpy().tolist())

dataset = PileDataset(samples)
dataloader = DataLoader(dataset,batch_size=16,shuffle=False)

# %%

threshold = 0.0
save_path = Path(f"data/results/best_tokens_{threshold}.json")
best_tokens:list[tuple[dict[int,float],list[str]]] = []
# if save_path.exists():
#     with open(save_path, "r") as f:
#         best_tokens = json.load(f)

#     for sample in best_tokens:
#         sample[0] = {int(k):float(v) for k,v in sample[0].items()}
# else:
graphs = build_full_graph(model,dataloader,threshold)    
for graph,decoded_context in tqdm(zip(graphs,samples['decoded_context']),desc="Getting best tokens",total=len(graphs)):
    best_tokens.append((get_best_tokens(graph,threshold),decoded_context))

with open(save_path, "w") as f:
    json.dump(best_tokens, f, indent=2)


print(f"Best tokens saved to {save_path}")


first_sample = dataset[1][0]
logits, cache = model.run_with_cache(first_sample)

generated = model.generate(first_sample.unsqueeze(0),max_new_tokens=32,do_sample=False)
print("ORIGINAL")
print(model.to_string(generated[0][:32]) + "||" + model.to_string(generated[0][-32:]))


contribs, tokens = best_tokens[1]
# top_token_idx = list(contribs.keys())[0]
# top_candidates = logits[0,top_token_idx-1].topk(5)[1]
# bottom_candidates = (logits[0,top_token_idx-1] * -1).topk(5)[1]


first_sample_corrupt = first_sample.clone()
# Create a batch of corrupt versions, each with one token corrupted with its bottom logit
corrupt_samples = []
for idx in contribs.keys():
    # Get bottom candidate for this token position
    other_candidate = (logits[0,idx-1] * -1).argmax()
    # Get second candidate
    # other_candidate = (logits[0,idx-1]).topk(2)[1][1]
    
    # Create corrupted sample
    corrupt_sample = first_sample.clone()    
    corrupt_sample[idx] = other_candidate

    generated = model.generate(corrupt_sample.unsqueeze(0),max_new_tokens=32,do_sample=False)
    corrupt_samples.append((corrupt_sample,generated,contribs[idx], idx))

    print(f"Token: {idx} Contribution: {contribs[idx]}")
    print(model.to_string(generated[0][:32]) + "||" + model.to_string(generated[0][-32:]))



if best_tokens:    
    for sample in best_tokens[:5]:
        plot_sample(sample[1], sample[0])


# Plot average contributions for tokens 0-32
if best_tokens:
    avg_contributions = plot_average_contributions(best_tokens)
    print(f"Average contributions calculated and plotted for tokens 0-32")

test = 0

