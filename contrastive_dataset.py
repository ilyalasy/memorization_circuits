import argparse
import pandas as pd
import torch as t
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
# from pandarallel import pandarallel
import json

import evaluate
from typing import Literal

import torch as t
from auto_circuit.data import PromptDataLoader, PromptDataset
from torch.utils.data import Subset

# pandarallel.initialize(nb_workers=16,progress_bar=True)

bleu = evaluate.load("bleu")
device = t.device("cuda" if t.cuda.is_available() else "mps")


def convert_to_sequences(context_tensor:t.Tensor,
                         generated_outputs:dict[int,t.Tensor],
                         min_context_len:t.Tensor,
                         whole_samples:t.Tensor):
    
    # Create the requested sequences for each sample using stored data
    minimal_context = []    
    next_gt_tokens = []
    next_generated_tokens = []
    
    for i in range(len(batch)):
        min_len = int(min_context_len[i].item())
        
        # Get the context up to min_len
        min_context = context_tensor[i, :min_len]
        minimal_context.append(min_context)

        # take the next 2 tokens: first one will serve as clean/corrupted prompt, second one will be the correct/wrong answer
        next_gt_tokens.append(whole_samples[i, min_len:min_len+2].tolist())
        next_generated_tokens.append(generated_outputs[min_len][i, min_len:min_len+2].tolist()) 
    
    return {
        'minimal_context': minimal_context,
        'next_gt_tokens': next_gt_tokens,
        'next_generated_tokens': next_generated_tokens        
    }


def find_minimum_context_for_memorization(model: AutoModelForCausalLM, 
                                          batch:pd.DataFrame, 
                                          tokenizer:AutoTokenizer,
                                          significant_drop_threshold:float = 0.3, 
                                          benchmark_metric:Literal['bleu','memorization']='bleu'):
    context_len = len(batch["context"].iloc[0])    
    whole_samples = batch["context"] + batch["true_completion"]
    whole_samples = t.tensor(whole_samples.to_list()).to(device)
    context_tensor = t.tensor(batch["context"].to_list()).to(device)

    previous_scores = t.ones(len(batch))
    min_context_len = t.full((len(batch),),t.nan)
    scores = {}
    
    # Store the generated outputs at each context length
    generated_outputs = {}

    # Store scores before and after the drop
    scores_before = t.ones(len(batch))
    scores_after = t.ones(len(batch))
    
    for current_len in tqdm(range(context_len, 0, -1),desc="Finding minimum context length for memorization",total=context_len):
        current_context = context_tensor[:, :current_len]
        attn_mask = t.ones_like(current_context)

        with t.no_grad():
            generation_output = model.generate(
                inputs=current_context, 
                attention_mask=attn_mask,
                max_length=whole_samples.size(-1),
                do_sample=False,
                use_cache=True
            )
            
        # Store the generation output for this context length
        generated_outputs[current_len] = generation_output

        # Extract the generated completion (excluding the context)
        generated_completion = generation_output[:, current_len:]

        # decoded_context = tokenizer.batch_decode(current_context)
        decoded_completion = tokenizer.batch_decode(generated_completion)
        decoded_gt = tokenizer.batch_decode(whole_samples[:, current_len:])

        # Calculate memorization score    
        correct_tokens = (generated_completion == whole_samples[:, current_len:]).sum(-1)
        memorization_score = correct_tokens / generated_completion.size(-1)

        # Calculate BLEU-4 score for each sample in the batch
        bleu_scores = []
        for batch_i in range(len(decoded_completion)):
            # calculate bleu score only if we haven't found the minimum context length yet
            if t.isnan(min_context_len[batch_i]):
                bleu_result = bleu.compute(predictions=[decoded_completion[batch_i]], references=[decoded_gt[batch_i]],max_order=4)
                bleu_scores.append(bleu_result['bleu'])
            else:
                bleu_scores.append(previous_scores[batch_i])
        bleu_scores = t.tensor(bleu_scores)

        if benchmark_metric == 'bleu':
            benchmark_score = bleu_scores.cpu()
        elif benchmark_metric == 'memorization':
            benchmark_score = memorization_score.cpu()
        else:
            raise ValueError(f"Invalid benchmark metric: {benchmark_metric}")
        
        relative_drop = (previous_scores - benchmark_score) / previous_scores
        
        next_token_diff_mask = (generated_completion[:,0] != whole_samples[:, current_len]).cpu()

        # Only update min_context_len if:
        # 1. Relative drop is significant
        # 2. We haven't found a min_context_len yet
        # 3. The generated token is different from ground truth token
        mask = (relative_drop > significant_drop_threshold) & t.isnan(min_context_len) & next_token_diff_mask
        
        # Store scores before and after the drop
        scores_before[mask] = previous_scores[mask]
        scores_after[mask] = benchmark_score[mask]

        min_context_len[mask] = current_len

        # Update previous score        
        previous_scores = benchmark_score

        scores[current_len] = (bleu_scores.cpu(), memorization_score.cpu())

        if (~min_context_len.isnan()).all():
            break    

    # Ensure all samples have a valid min_context_len by using the full context if necessary
    min_context_len[t.isnan(min_context_len)] = context_len
    
    sequences = convert_to_sequences(context_tensor,generated_outputs,min_context_len,whole_samples)

    # Add scores to sequences
    sequences['score_before'] = scores_before.tolist()
    sequences['score_after'] = scores_after.tolist()
    sequences['diverging_position'] = min_context_len.tolist()
    return sequences

def save_jsonl(all_sequences:dict,path:Path,tokenizer:AutoTokenizer):
    total_skipped_count = 0
    print(f"Saving decoded data to {path}")        
    with open(path, 'w') as f:
        for i in range(len(all_sequences['minimal_context'])):
            # Filter those that cannot create clean/corrupt pair
            if all_sequences['next_gt_tokens'][i][0] == all_sequences['next_generated_tokens'][i][0]:
                total_skipped_count += 1
                continue
            sample = {                    
                'minimal_context': tokenizer.decode(all_sequences['minimal_context'][i]),
                'next_gt_tokens': tokenizer.batch_decode(all_sequences['next_gt_tokens'][i]),
                'next_generated_tokens': tokenizer.batch_decode(all_sequences['next_generated_tokens'][i]),

                'diverging_position': all_sequences['diverging_position'][i],
                'score_before': all_sequences['score_before'][i],
                'score_after': all_sequences['score_after'][i]
            }
            
            # Write as JSON line
            f.write(json.dumps(sample) + '\n')
    print(f"Total skipped count: {total_skipped_count}/{len(mem_df)}")

def save_autocircuit_ds(all_sequences:dict, path:Path, tokenizer:AutoTokenizer):
    """
    Save the dataset in a format compatible with AutoCircuit.
    
    Args:
        all_sequences: Dictionary containing the sequences data
        path: Path where to save the dataset
        tokenizer: Tokenizer to use for encoding if needed
    """
    autocircuit_data = {
        "prompts": []
    }
    
    for i in range(len(all_sequences['minimal_context'])):
        # Skip pairs where clean and corrupt are identical
        if all_sequences['next_gt_tokens'][i][0] == all_sequences['next_generated_tokens'][i][0]:
            continue
            
        # Create a prompt pair
        prompt_pair = {
            "clean": tokenizer.decode(all_sequences['minimal_context'][i].tolist() + [all_sequences['next_gt_tokens'][i][0]]),
            "corrupt": tokenizer.decode(all_sequences['minimal_context'][i].tolist() + [all_sequences['next_generated_tokens'][i][0]]),
            "answers": [tokenizer.decode(all_sequences['next_gt_tokens'][i][1])],
            "wrong_answers": [tokenizer.decode(all_sequences['next_generated_tokens'][i][1])]
        }
        
        autocircuit_data["prompts"].append(prompt_pair)
    
    # Save the dataset
    with open(path, 'w') as f:
        json.dump(autocircuit_data, f, indent=2)
    
    print(f"Saved AutoCircuit dataset with {len(autocircuit_data['prompts'])} prompt pairs to {path}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Memorization score threshold")
    parser.add_argument("--metric", type=str, default="bleu", choices=["memorization", "bleu"], help="Benchmark metric to use: 'memorization' for exact memorization or 'bleu' for approximate memorization")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Model to use")
    parser.add_argument("--prompt_tokens", type=int, default=32, help="Number of tokens to use as prompt")
    parser.add_argument("--generation_tokens", type=int, default=96, help="Number of tokens to generate")

    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    threshold = args.threshold
    metric = args.metric
    model_name = args.model_name
    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens

    df = pd.read_json(f"data/results/memorization_scores_{model_name.split('/')[-1]}_{prompt_tokens}_{generation_tokens}.json")

    mem_df = df[df['memorization_score'] > threshold]
    mem_df = mem_df.sort_values('memorization_score', ascending=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if device.type == "cuda":
        model = model.half()
    # Instead of processing everything at once, let's chunk it
    batches = [mem_df.iloc[i:i+batch_size] for i in range(0, len(mem_df), batch_size)]

    path = Path(f"data/results/contrastive_mem_{threshold}_{model_name.split('/')[-1]}_{prompt_tokens}_{generation_tokens}_{metric}.jsonl")

    if path.exists():
        with open(path, 'r') as f:
            all_sequences = [json.loads(line) for line in f]
    else:
        all_sequences = defaultdict(list)
        for batch in tqdm(batches,desc="Processing batches"):
            sequences = find_minimum_context_for_memorization(model,batch,tokenizer,benchmark_metric=metric)
            for key in sequences:
                all_sequences[key].extend(sequences[key])
        
        save_autocircuit_ds(all_sequences,path.with_name(f"{path.stem}_ac.json"),tokenizer)
        save_jsonl(all_sequences,path,tokenizer)