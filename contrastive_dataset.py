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

from metrics.nmt_bleu import compute_bleu
from typing import Literal

import torch as t

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# pandarallel.initialize(nb_workers=16,progress_bar=True)

device = t.device("cuda" if t.cuda.is_available() else "mps")


def convert_to_sequences(context_tensor:t.Tensor,
                         generated_outputs:dict[int,t.Tensor],
                         min_context_len:t.Tensor,
                         whole_samples:t.Tensor):
    
    # Create the requested sequences for each sample using stored data
    minimal_context = []    
    next_gt_tokens = []
    next_generated_tokens = []
    
    batch_size = context_tensor.size(0)
    
    for i in range(batch_size):
        min_len = int(min_context_len[i].item())
        
        # Get the context up to min_len
        min_context = context_tensor[i, :min_len]
        minimal_context.append(min_context)

        # Take the next 2 tokens: first one will serve as clean/corrupted prompt, second one will be the correct/wrong answer
        next_gt_tokens.append(whole_samples[i, min_len:].tolist())
        
        # Get generated tokens for this sample at this minimum context length
        if min_len in generated_outputs and i < generated_outputs[min_len].size(0):
            next_generated_tokens.append(generated_outputs[min_len][i, min_len:].tolist())
        else:
            # Fallback if we don't have generated output for this specific length
            # Find the closest available context length
            available_lens = sorted(generated_outputs.keys())
            closest_len = min(available_lens, key=lambda x: abs(x - min_len))
            next_generated_tokens.append(generated_outputs[closest_len][i, min_len:].tolist())
    
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
    
    # 1. First run model on full ground truth sequence and save logits for each token position
    with t.no_grad():
        outputs = model(context_tensor, return_dict=True)
        logits = outputs.logits
    
    # 2. Find argmax for each position and identify where they differ from ground truth
    # Shift logits and tokens for next-token prediction comparison
    shifted_logits = logits[:, :-1, :]
    shifted_tokens = context_tensor[:, 1:]
    
    # Get predictions (excluding the last token which has no corresponding target)
    predictions = t.argmax(shifted_logits, dim=-1)
    
    # Find positions where predictions differ from ground truth
    diff_positions = (predictions != shifted_tokens)

    diff_positions = diff_positions.nonzero(as_tuple=True)
    
    # Extract batch indices and position indices
    batch_indices, position_indices = diff_positions
    
    # Add 1 to position indices to align with original positions (due to shifted tokens)
    position_indices = position_indices + 1
    
    # Create a dictionary to group samples by position
    position_to_samples = defaultdict(list)
    for batch_idx, pos_idx in zip(batch_indices.tolist(), position_indices.tolist()):                
        position_to_samples[pos_idx].append(batch_idx)
    
    # Ensure all positions have at least the full context to check
    if 0 not in position_to_samples:
        position_to_samples[0] = list(range(len(batch)))
    
    # Sort positions from largest to smallest
    all_positions = sorted(position_to_samples.keys(), reverse=True)
    
    # 3. Iterate only over grouped positions that differ, instead of every possible position
    for current_len in tqdm(all_positions, desc="Finding minimum context length for memorization", total=len(all_positions)):
        # Get the batch indices for samples that have differences at this position
        batch_indices_to_check = position_to_samples[current_len]
        
        # Only process samples that don't have a min_context_len already
        batch_indices_to_process = [idx for idx in batch_indices_to_check if t.isnan(min_context_len[idx])]
        
        if not batch_indices_to_process:
            continue
            
        # Create a tensor of batch indices
        batch_indices_tensor = t.tensor(batch_indices_to_process, device=device)
        
        # Select only the relevant samples for this position
        current_context = context_tensor[batch_indices_tensor, :current_len]
        attn_mask = t.ones_like(current_context)

        with t.no_grad():
            generation_output = model.generate(
                inputs=current_context, 
                attention_mask=attn_mask,
                max_length=whole_samples.size(-1),
                do_sample=False,
                use_cache=True
            )
            
        # Store the generation output for this context length and these samples
        if current_len not in generated_outputs:
            generated_outputs[current_len] = {}
        for i, idx in enumerate(batch_indices_to_process):
            generated_outputs[current_len][idx] = generation_output[i]

        # Extract the generated completion (excluding the context)
        generated_completion = generation_output[:, current_len:]
        target_completion = whole_samples[batch_indices_tensor, current_len:]

        # Calculate memorization score    
        correct_tokens = (generated_completion == target_completion).sum(-1)
        memorization_score = correct_tokens / generated_completion.size(-1)

        # Calculate BLEU-4 score for the processed batch samples
        bleu_scores = []
        for i, batch_idx in enumerate(batch_indices_to_process):
            bleu_result = compute_bleu([[target_completion[i].tolist()]], [generated_completion[i].tolist()], max_order=4, smooth=True)
            (bleu, precisions, bp, ratio, translation_length, reference_length) = bleu_result
            bleu_scores.append(bleu)
        bleu_scores = t.tensor(bleu_scores)

        if benchmark_metric == 'bleu':
            benchmark_score = bleu_scores.cpu()
        elif benchmark_metric == 'memorization':
            benchmark_score = memorization_score.cpu()
        else:
            raise ValueError(f"Invalid benchmark metric: {benchmark_metric}")
        
        # Get previous scores for these specific samples
        prev_scores_batch = previous_scores[batch_indices_tensor.cpu()]
        relative_drop = (prev_scores_batch - benchmark_score) / prev_scores_batch
        
        next_token_diff_mask = (generated_completion[:,0] != target_completion[:,0]).cpu()

        # Only update min_context_len if:
        # 1. Relative drop is significant
        # 2. The generated token is different from ground truth token
        mask = (relative_drop > significant_drop_threshold) & next_token_diff_mask
        
        # Get indices of samples that meet the criteria
        indices_to_update = [batch_indices_to_process[i] for i in range(len(mask)) if mask[i]]
        
        # Store scores before and after the drop for the samples being updated
        for i, idx in enumerate(indices_to_update):
            scores_before[idx] = prev_scores_batch[i]
            scores_after[idx] = benchmark_score[i]
            min_context_len[idx] = current_len
            
        # Update previous score only for the processed samples
        for i, idx in enumerate(batch_indices_to_process):
            previous_scores[idx] = benchmark_score[i]
            
        if current_len in scores:
            scores[current_len] = (scores[current_len][0], scores[current_len][1])
        else:
            scores[current_len] = (bleu_scores.cpu(), memorization_score.cpu())

        if (~min_context_len.isnan()).all():
            break    

    # Ensure all samples have a valid min_context_len by using the full context if necessary
    min_context_len[t.isnan(min_context_len)] = context_len
    
    # Reorganize generated_outputs for convert_to_sequences
    reorganized_outputs = {}
    for current_len in generated_outputs:
        batch_size = len(batch)
        reorganized_outputs[current_len] = t.zeros((batch_size, whole_samples.size(-1)), dtype=t.long, device=device)
        for idx, output in generated_outputs[current_len].items():
            reorganized_outputs[current_len][idx] = output
    
    sequences = convert_to_sequences(context_tensor, reorganized_outputs, min_context_len, whole_samples)

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
    
    # Track unique prompt pairs to avoid duplicates
    seen_pairs = set()
    duplicate_count = 0
    
    for i in range(len(all_sequences['minimal_context'])):
        # Skip pairs where clean and corrupt are identical
        if all_sequences['next_gt_tokens'][i][0] == all_sequences['next_generated_tokens'][i][0]:
            continue
            
        clean_tokens = all_sequences['minimal_context'][i].tolist() + [all_sequences['next_gt_tokens'][i][0]]
        corrupt_tokens = all_sequences['minimal_context'][i].tolist() + [all_sequences['next_generated_tokens'][i][0]]        
        
        # Decode tokens to create the pair
        clean_text = tokenizer.decode(clean_tokens)
        corrupt_text = tokenizer.decode(corrupt_tokens)
        answer = tokenizer.decode(all_sequences['next_gt_tokens'][i][1])
        wrong_answer = tokenizer.decode(all_sequences['next_generated_tokens'][i][1])

        if len(tokenizer(clean_text)["input_ids"]) != len(tokenizer(corrupt_text)["input_ids"]):
            print(f"Skipping pair {i} because of length mismatch after decoding")
            continue
        
        # Create a unique identifier for this prompt pair
        pair_key = (clean_text, corrupt_text, answer, wrong_answer)
        
        # Skip if we've seen this pair before
        if pair_key in seen_pairs:
            duplicate_count += 1
            continue
            
        # Add to seen pairs
        seen_pairs.add(pair_key)
        
        # Create a prompt pair
        prompt_pair = {
            "clean": clean_text,
            "corrupt": corrupt_text,
            "answers": [answer],
            "wrong_answers": [wrong_answer]
        }
        
        autocircuit_data["prompts"].append(prompt_pair)
    
    # Save the dataset
    with open(path, 'w') as f:
        json.dump(autocircuit_data, f, indent=2)
    
    print(f"Saved AutoCircuit dataset with {len(autocircuit_data['prompts'])} prompt pairs to {path}")
    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate prompt pairs")


def create_contrastive_pairs(df: pd.DataFrame, 
                             tokenizer: AutoTokenizer,
                             high_threshold: float = 0.75, 
                             low_threshold: float = 0.0, 
                             max_pairs: int = 1000, batch_size=32) -> list:
    """
    Create contrastive pairs of examples with high and low memorization scores
    that are semantically similar to each other.
    
    Args:
        df: DataFrame with memorization data
        tokenizer: Tokenizer to use for encoding/decoding
        high_threshold: Minimum memorization score for "clean" examples
        low_threshold: Maximum memorization score for "corrupt" examples
        max_pairs: Maximum number of pairs to create
        batch_size: Batch size for embedding generation
        
    Returns:
        List of dictionaries with contrastive pairs
    """
    
    # Split df into high and low memorization groups
    high_mem_df = df[df['memorization_score'] >= high_threshold].reset_index()
    low_mem_df = df[df['memorization_score'] <= low_threshold].reset_index()
    
    print(f"High memorization examples: {len(high_mem_df)}")
    print(f"Low memorization examples: {len(low_mem_df)}")
    
    if len(high_mem_df) == 0 or len(low_mem_df) == 0:
        print("Not enough examples in one of the groups.")
        return []
    
    # Load sentence transformer model for semantic similarity calculation
    print("Loading sentence embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') #all-mpnet-base-v2
    
    # Create embeddings for contexts
    print("Generating embeddings for high memorization examples...")
    high_contexts = high_mem_df['decoded_context'].tolist()
    high_embeddings = model.encode(high_contexts, show_progress_bar=True, convert_to_tensor=True, batch_size=batch_size)
    
    print("Generating embeddings for low memorization examples...")
    low_contexts = low_mem_df['decoded_context'].tolist()
    low_embeddings = model.encode(low_contexts, show_progress_bar=True, convert_to_tensor=True, batch_size=batch_size)
    
    # Find most similar pairs
    print("Finding most similar pairs...")
    contrastive_pairs = []
    
    # Keep track of used low memorization examples
    available_mask = t.ones(len(low_mem_df))
    
    # For each high memorization example, find the most similar unused low memorization example
    for i in tqdm(range(min(len(high_mem_df), max_pairs))):
        
        # Calculate cosine similarity between current high embedding and all low embeddings
        similarities = t.nn.functional.cosine_similarity(high_embeddings[i].unsqueeze(0), low_embeddings).cpu()
        
        # Mask out similarities of already used low examples
        similarities_masked = similarities * available_mask
        
        # If all low examples have been used, skip this high example
        if similarities_masked.sum() == 0:
            print(f"Warning: All low memorization examples have been used. Created {len(contrastive_pairs)} pairs.")
            break
        
        # Get the index of the most similar available low example
        most_similar_idx = t.argmax(similarities_masked).item()
        similarity_score = similarities[most_similar_idx]
        
        if len(tokenizer(high_mem_df.iloc[i]['decoded_context'])["input_ids"]) != len(tokenizer(low_mem_df.iloc[most_similar_idx]['decoded_context'])["input_ids"]):
            print(f"Skipping pair {i} because of length mismatch after decoding")
            continue

        # Mark this low example as used
        available_mask[most_similar_idx] = 0
        
        # Create contrastive pair
        pair = {
            'clean': high_mem_df.iloc[i]['decoded_context'],
            "corrupt": low_mem_df.iloc[most_similar_idx]['decoded_context'],
            "answers": [tokenizer.batch_decode(high_mem_df.iloc[i]['completion'])[0]],
            "wrong_answers": [tokenizer.batch_decode(low_mem_df.iloc[most_similar_idx]['completion'])[0]],          
            'similarity_score': float(similarity_score)
        }
        
        contrastive_pairs.append(pair)
    
    # Sort by similarity score
    contrastive_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"Created {len(contrastive_pairs)} contrastive pairs")
    return {'prompts': contrastive_pairs}


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--dataset", type=str, default="timaeus/pile-wikipedia_en", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=0.75, help="Memorization score threshold")
    parser.add_argument("--metric", type=str, default="bleu", choices=["memorization", "bleu"], help="Benchmark metric to use: 'memorization' for exact memorization or 'bleu' for approximate memorization")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--prompt_tokens", type=int, default=50, help="Number of tokens to use as prompt")
    parser.add_argument("--generation_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--contrastive_mode", type=str, default="divergence", choices=["divergence", "dataset"], 
                        help="Whether to create contrastive pairs based on the token divergence position or take pairs of memorized vs non-memorized examples")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Directory to save results")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    threshold = args.threshold
    metric = args.metric
    model_name = args.model_name
    short_model_name = model_name.split('/')[-1]
    dataset = args.dataset
    short_dataset = dataset.split('/')[-1]
    output_dir = args.output_dir
    
    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens    
    contrastive_mode = args.contrastive_mode

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the memorization scores file
    mem_scores_file = Path(f"{output_dir}/mem_scores_{short_dataset}_{short_model_name}_{prompt_tokens}_{generation_tokens}.json")
    
    # Check if the file exists in the dataset folder
    if not mem_scores_file.exists():
        # Try looking in the parent results directory
        mem_scores_file = Path(f"data/results/mem_scores_{short_dataset}_{short_model_name}_{prompt_tokens}_{generation_tokens}.json")
        if not mem_scores_file.exists():
            raise FileNotFoundError(f"Could not find memorization scores file at {mem_scores_file}")

    df = pd.read_json(mem_scores_file)

    len_before = len(df)    
    df = df.drop_duplicates(subset=['decoded_context','decoded_completion','decoded_true_completion'])
    len_after = len(df)
    print(f"Removed {len_before - len_after} duplicate examples")

    # Output path for contrastive dataset
    path = Path(f"{output_dir}/contrastive_{short_dataset}_{threshold}_{short_model_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}.json")

    if path.exists():
        with open(path, 'r') as f:
            all_sequences = json.load(f)["prompts"]
        print(f"Loaded {len(all_sequences)} contrastive pairs from {path}")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if contrastive_mode == "divergence":
        mem_df = df[df['memorization_score'] > threshold]
        mem_df = mem_df.sort_values('memorization_score', ascending=False)

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if device.type == "cuda":
            model = model.half()
        # Instead of processing everything at once, let's chunk it
        batches = [mem_df.iloc[i:i+batch_size] for i in range(0, len(mem_df), batch_size)]
        all_sequences = defaultdict(list)
        for batch in tqdm(batches,desc="Processing batches"):
            sequences = find_minimum_context_for_memorization(model,batch,tokenizer,benchmark_metric=metric)
            for key in sequences:
                all_sequences[key].extend(sequences[key])
        
        save_autocircuit_ds(all_sequences,path,tokenizer)
        save_jsonl(all_sequences,path.with_name(f"{path.stem}.jsonl"),tokenizer)

    elif contrastive_mode=="dataset":
        contrastive_pairs = create_contrastive_pairs(
                df=df,
                high_threshold=threshold,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
        # Save the dataset
        with open(path, 'w') as f:
            json.dump(contrastive_pairs, f, indent=2)
        
        print(f"Saved AutoCircuit dataset with {len(contrastive_pairs['prompts'])} prompt pairs to {path}")