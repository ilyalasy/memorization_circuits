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
                         whole_samples:t.Tensor,
                         tokenizer:AutoTokenizer):
    
    # Create the requested sequences for each sample using stored data
    minimal_context_gt = []
    minimal_context_generated = []
    next_gt_token = []
    next_generated_token = []
    
    skipped_ids = []
    for i in range(len(batch)):
        min_len = int(min_context_len[i].item())
        
        # Get the context up to min_len
        min_context = context_tensor[i, :min_len]

        next_gt_from_completion = whole_samples[i, min_len]

        gen_output = generated_outputs[min_len][i]
        next_gen_token = gen_output[min_len]

        # Skip this sample if the next generated token is the same as the ground truth token
        if next_gen_token.item() == next_gt_from_completion.item():
            skipped_ids.append(i)
            continue
        
        # 1. Sequence of minimal context + ground truth token
        # Get next ground truth token after min_context        
        context_plus_gt = t.cat([min_context, next_gt_from_completion.unsqueeze(0)], dim=0)        
        minimal_context_gt.append(context_plus_gt)
        
        # 2. Sequence of minimal context + argmax token
        # Get the generation output for this context length
        
        context_plus_generated = t.cat([min_context, next_gen_token.unsqueeze(0)], dim=0)
        minimal_context_generated.append(context_plus_generated)
        
        # 3. Next ground truth token (after sequence 1)
        next_gt = whole_samples[i, min_len + 1].unsqueeze(0) if min_len + 1 < whole_samples.size(1) else t.tensor([tokenizer.eos_token_id]).to(device)
        next_gt_token.append(next_gt)
        
        # 4. Next argmax token (after sequence 2)
        next_gen = gen_output[min_len + 1].unsqueeze(0) if min_len + 1 < len(gen_output) else t.tensor([tokenizer.eos_token_id]).to(device)
        next_generated_token.append(next_gen)


        # Print decoded versions of all 4 sequences for inspection
        # print("Minimal Context + GT Token:")
        # print(tokenizer.decode(context_plus_gt))
        
        # print("\nNext GT Token:")
        # print(tokenizer.decode(next_gt))
        
        # print("\nMinimal Context + Generated Token:")
        # print(tokenizer.decode(context_plus_generated))
        
        # print("\nNext Generated Token:")
        # print(tokenizer.decode(next_gen))
    # if len(skipped_ids) > 0:
    #     print("Following samples were skipped as they had the same next token: ", skipped_ids)
    
    return {
        'minimal_context_gt': t.tensor(minimal_context_gt),
        'next_gt_token': t.tensor(next_gt_token),
        'minimal_context_generated': t.tensor(minimal_context_generated),
        'next_generated_token': t.tensor(next_generated_token)
    }, len(skipped_ids)


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
        min_context_len[mask] = current_len

        # Update previous score        
        previous_scores = benchmark_score

        scores[current_len] = (bleu_scores.cpu(), memorization_score.cpu())

        if (~min_context_len.isnan()).all():
            break    

    # Ensure all samples have a valid min_context_len by using the full context if necessary
    min_context_len[t.isnan(min_context_len)] = context_len
    
    sequences,skipped_count = convert_to_sequences(context_tensor,generated_outputs,min_context_len,whole_samples,tokenizer)
    
    return min_context_len, scores, sequences, skipped_count

def to_autocircuit_ds(calculated_sequences:dict[str,t.Tensor],return_seq_length: bool = False,tail_divergence: bool = False, test_size: float = 0.1, batch_size: int | tuple[int, int] = 8):

    dataset = PromptDataset(clean_prompts=calculated_sequences["minimal_context_gt"].to(device), 
                            corrupt_prompts=calculated_sequences["minimal_context_generated"].to(device),
                            answers=[calculated_sequences["next_gt_token"].unsqueeze(0).to(device)],
                            wrong_answers=[calculated_sequences["next_generated_token"].unsqueeze(0).to(device)])
     
    dataset_size = len(dataset)
    train_size = int(dataset_size * (1 - test_size))
    train_set = Subset(dataset, list(range(train_size)))
    test_set = Subset(dataset, list(range(train_size, dataset_size)))

    seq_len = None    
    diverge_idx: int = 0
    kvs = []
    if return_seq_length:        
        seq_len = df["context"].shape[1]
    
    if tail_divergence:
        diverge_idxs = (~(df["context"] == df["context"])).int().argmax(dim=1)
        diverge_idx = int(diverge_idxs.min().item())
    if diverge_idx > 0:
        raise NotImplementedError()

    train_loader = PromptDataLoader(
        train_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[0] if len(kvs) > 0 else None,
        seq_labels=None,
        word_idxs=None,
        batch_size=batch_size[0] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    test_loader = PromptDataLoader(
        test_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[-1] if len(kvs) > 0 else None,
        seq_labels=None,
        word_idxs=None,
        batch_size=batch_size[1] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    return train_loader, test_loader

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Memorization score threshold")
    parser.add_argument("--metric", type=str, default="bleu", choices=["memorization", "bleu"], help="Benchmark metric to use: 'memorization' for exact memorization or 'bleu' for approximate memorization")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Model to use")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    threshold = args.threshold
    metric = args.metric
    model_name = args.model_name

    df = pd.read_json("data/results/memorization_scores_pythia-70m-deduped_32_32.json")

    mem_df = df[df['memorization_score'] > threshold]
    mem_df = mem_df.sort_values('memorization_score', ascending=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if device.type == "cuda":
        model = model.half()
    # Instead of processing everything at once, let's chunk it
    batches = [mem_df.iloc[i:i+batch_size] for i in range(0, len(mem_df), batch_size)]

    path = Path(f"data/results/contrastive_mem_{threshold}_{model_name.split('/')[-1]}_{metric}.pt")

    if path.exists():
        all_sequences = t.load(path,weights_only=False)
    else:
        all_sequences = defaultdict(list)
        total_skipped_count = 0
        for batch in tqdm(batches,desc="Processing batches"):
            min_context_len,scores,sequences,skipped_count = find_minimum_context_for_memorization(model,batch,tokenizer,benchmark_metric=metric)
            for key in sequences:
                all_sequences[key].append(sequences[key])
            total_skipped_count += skipped_count
        print(f"Total skipped count: {total_skipped_count}/{len(mem_df)}")
        
        
        for key in all_sequences:
            all_sequences[key] = t.cat(all_sequences[key])
        t.save(dict(all_sequences), path)

        # Save decoded keys as jsonl
        jsonl_path = path.with_suffix(".jsonl")
        
        print(f"Saving decoded data to {jsonl_path}")        
        with open(jsonl_path, 'w') as f:
            for i in range(len(all_sequences['minimal_context_gt'])):
                # Create a dictionary for each example
                sample = {                    
                    'minimal_context_gt': tokenizer.decode(all_sequences['minimal_context_gt'][i]),
                    'next_gt_token': tokenizer.decode(all_sequences['next_gt_token'][i]),
                    'minimal_context_generated': tokenizer.decode(all_sequences['minimal_context_generated'][i]),
                    'next_generated_token': tokenizer.decode(all_sequences['next_generated_token'][i])
                }
                
                # Write as JSON line
                f.write(json.dumps(sample) + '\n')        

    train_loader, test_loader = to_autocircuit_ds(all_sequences,return_seq_length=True,tail_divergence=True,test_size=0.1,batch_size=batch_size)


