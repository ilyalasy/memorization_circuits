import argparse
import pandas as pd
import torch as t
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
from pandarallel import pandarallel
import json
import os
import logging

# Custom logger that prepends job ID to all messages
class JobLogger:
    _instances = {}  # Class variable to track instances by name
    
    def __init__(self, job_id=None):
        self.job_id = job_id
        self.logger = logging.getLogger("contrastive_dataset")
        
        # Clear any existing handlers to prevent duplication
        if self.logger.handlers:
            self.logger.handlers = []
            
        # Add a single handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, msg, *args, **kwargs):
        prefix = f"[Job {self.job_id}] " if self.job_id is not None else ""
        self.logger.info(f"{prefix}{msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        prefix = f"[Job {self.job_id}] " if self.job_id is not None else ""
        self.logger.warning(f"{prefix}{msg}", *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        prefix = f"[Job {self.job_id}] " if self.job_id is not None else ""
        self.logger.error(f"{prefix}{msg}", *args, **kwargs)
        
    @classmethod
    def get_logger(cls, job_id=None):
        """Get or create a logger with the specified job_id"""
        if job_id not in cls._instances:
            cls._instances[job_id] = cls(job_id)
        return cls._instances[job_id]

# Initialize logger with None job_id, will be updated later
logger = JobLogger.get_logger()

from metrics.nmt_bleu import compute_bleu
from typing import Literal

import torch as t

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from functools import partial

pandarallel.initialize(nb_workers=8,progress_bar=False)

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
        next_gt_tokens.append(whole_samples[i, min_len:].tolist())
        next_generated_tokens.append(generated_outputs[min_len][i, min_len:].tolist()) 
    
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
    whole_samples = np.concatenate((np.array(batch["context"].to_list()),np.array(batch["true_completion"].to_list())),axis=1)
    whole_samples = t.tensor(whole_samples).to(device)
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
        generated_outputs[current_len] = generation_output.cpu()

        # Extract the generated completion (excluding the context)
        generated_completion = generation_output[:, current_len:]

        # decoded_context = tokenizer.batch_decode(current_context)
        # decoded_completion = tokenizer.batch_decode(generated_completion)
        # decoded_gt = tokenizer.batch_decode(whole_samples[:, current_len:])

        # Calculate memorization score    
        correct_tokens = (generated_completion == whole_samples[:, current_len:]).sum(-1)
        memorization_score = correct_tokens / generated_completion.size(-1)

        # Calculate BLEU-4 score for each sample in the batch        
        bleu_scores = []
        for batch_i in range(generated_completion.size(0)):
            # calculate bleu score only if we haven't found the minimum context length yet
            if t.isnan(min_context_len[batch_i]):
                bleu_result = compute_bleu([[whole_samples[batch_i, current_len:].tolist()]],[generated_completion[batch_i].tolist()],max_order=4, smooth=True)
                (bleu, precisions, bp, ratio, translation_length, reference_length) = bleu_result
                bleu_scores.append(bleu)
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
    total_len = len(all_sequences['minimal_context'])
    logger.info(f"Saving decoded data to {path}")        
    with open(path, 'w') as f:
        for i in range(total_len):
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
    logger.info(f"Total skipped count: {total_skipped_count}/{total_len}")

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
        if all_sequences['next_gt_tokens'][i][1] == all_sequences['next_generated_tokens'][i][1]:
            continue
            
        clean_tokens = all_sequences['minimal_context'][i].tolist() + [all_sequences['next_gt_tokens'][i][0]]
        corrupt_tokens = all_sequences['minimal_context'][i].tolist() + [all_sequences['next_generated_tokens'][i][0]]        
        
        # Decode tokens to create the pair
        clean_text = tokenizer.decode(clean_tokens)
        corrupt_text = tokenizer.decode(corrupt_tokens)
        answer = tokenizer.decode(all_sequences['next_gt_tokens'][i][1])
        wrong_answer = tokenizer.decode(all_sequences['next_generated_tokens'][i][1])

        if len(tokenizer(clean_text)["input_ids"]) != len(tokenizer(corrupt_text)["input_ids"]):
            logger.info(f"Skipping pair {i} because of length mismatch after decoding")
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
    
    logger.info(f"Saved AutoCircuit dataset with {len(autocircuit_data['prompts'])} prompt pairs to {path}")
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate prompt pairs")


def check_matching_token(low_row:pd.Series, div_pos:int, target_token:int):
    low_context_tokens = low_row["context"]
    # Check if low example has enough tokens and matching token at div_pos
    if div_pos < len(low_context_tokens) and low_context_tokens[div_pos] == target_token:
        return True
    return False

def cut_up_to_pos(row:pd.Series, tokenizer:AutoTokenizer,div_pos:int|None=None):
    if div_pos is None:
        div_pos = int(row['diverging_position']) if not pd.isna(row.get('diverging_position')) else None

    if div_pos is None:
        return row['decoded_context']
    
    return row["context"][:div_pos], tokenizer.decode(row["context"][:div_pos])

# Function to get model embeddings through mean pooling of last layer
def get_model_embeddings(texts:list[str], tokenizer:AutoTokenizer, model:AutoModelForCausalLM, device=device, batch_size=32):
    embeddings = []
    with t.no_grad():
        for i in tqdm(range(0, len(texts), batch_size),total=len(texts)//batch_size, desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**encoded, output_hidden_states=True)
            # Get last hidden state and mean pool across sequence length
            last_hidden_state = outputs.hidden_states[-1]
            # Create attention mask that ignores padding tokens
            attention_mask = encoded.attention_mask.unsqueeze(-1)
            # Mean pooling
            mean_pooled = t.sum(last_hidden_state * attention_mask, 1) / t.sum(attention_mask, 1)
            embeddings.append(mean_pooled.cpu())
    return t.cat(embeddings, dim=0)

def create_contrastive_pairs(df: pd.DataFrame, 
                             tokenizer: AutoTokenizer,
                             model: AutoModelForCausalLM,
                             high_threshold: float = 0.75, 
                             low_threshold: float = 0.0, 
                             batch_size=32,
                             sim_metric:Literal['embedding','token_overlap']='token_overlap') -> list:
    """
    Create contrastive pairs of examples with high and low memorization scores
    that are semantically similar to each other.
    
    Args:
        df: DataFrame with memorization data
        tokenizer: Tokenizer to use for encoding/decoding
        model: The language model to use for embeddings and predictions
        high_threshold: Minimum memorization score for "clean" examples
        low_threshold: Maximum memorization score for "corrupt" examples
        batch_size: Batch size for embedding generation
        sim_metric: Similarity metric to use - 'embedding' uses model embeddings,
                   'token_overlap' calculates token-level similarity
        
    Returns:
        List of dictionaries with contrastive pairs
    """
    
    # Split df into high and low memorization groups
    high_mem_df = df[df['memorization_score'] >= high_threshold].reset_index()
    low_mem_df = df[df['memorization_score'] <= low_threshold].reset_index()
    
    logger.info(f"High memorization examples: {len(high_mem_df)}")
    logger.info(f"Low memorization examples: {len(low_mem_df)}")
    
    if len(high_mem_df) == 0 or len(low_mem_df) == 0:
        logger.info("Not enough examples in one of the groups.")
        return []
    
    # Check if diverging_position column exists
    has_diverging_position = 'diverging_position' in df.columns     

    if has_diverging_position:
        len_before = len(high_mem_df)
        high_mem_df = high_mem_df.dropna(subset=['diverging_position'])
        len_after = len(high_mem_df)
        if len_before != len_after:
            logger.info(f"Removed {len_before - len_after} examples with no diverging position")
    
    # Function to calculate token overlap similarity between two texts
    def calculate_token_overlap(tokens1, tokens2):                
        # Convert token lists to sets
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Calculate intersection and union
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Calculate Jaccard similarity (intersection over union)
        overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0
        return overlap_ratio
    
    # Process high memorization examples - cut contexts to diverging_position if available
    logger.info("Generating embeddings for high memorization examples...")
    high_mem_df['processed_context'], high_mem_df['processed_context_decoded'] = zip(*high_mem_df.apply(partial(cut_up_to_pos, tokenizer=tokenizer), axis=1))
    
    # Only generate embeddings if using embedding similarity
    high_embeddings = None
    if sim_metric == 'embedding':
        high_embeddings = get_model_embeddings(high_mem_df['processed_context_decoded'].tolist(), tokenizer=tokenizer, model=model, batch_size=batch_size)
    
    # Find most similar pairs
    logger.info("Finding most similar pairs...")
    contrastive_pairs = []
    
    # Process each high memorization example
    for i in tqdm(range(len(high_mem_df)), total=len(high_mem_df), desc="Processing high examples"):
        high_row = high_mem_df.iloc[i]
        
        # Get token at diverging_position + 1 for high memorization example
        if has_diverging_position and not pd.isna(high_row.get('diverging_position')):
            div_pos = int(high_row['diverging_position'])
            high_context_tokens = high_row["context"]
            
            if div_pos < len(high_context_tokens) - 1:
                target_token = high_context_tokens[div_pos]
                
                matching_mask = low_mem_df.parallel_apply(partial(check_matching_token, div_pos=div_pos, target_token=target_token), axis=1)
                matching_low_df = low_mem_df[matching_mask].copy()
                
                # If no matching examples found, skip this high example
                if matching_low_df.empty:
                    logger.info(f"No matching low examples found for high example {i}")
                    continue
                
                matching_low_df['processed_context'], matching_low_df['processed_context_decoded'] = zip(*matching_low_df.parallel_apply(partial(cut_up_to_pos, tokenizer=tokenizer, div_pos=div_pos), axis=1))
                
                # Calculate similarities based on selected metric
                if sim_metric == 'embedding':
                    matching_low_embeddings = get_model_embeddings(matching_low_df['processed_context_decoded'].tolist(), tokenizer=tokenizer, model=model, batch_size=batch_size)
                    similarities = t.nn.functional.cosine_similarity(high_embeddings[i].unsqueeze(0), matching_low_embeddings).cpu()
                elif sim_metric == 'token_overlap':
                    high_context = high_row['processed_context']                    
                    similarities = matching_low_df.parallel_apply(
                        lambda low_row: calculate_token_overlap(high_context, low_row['processed_context']),
                        axis=1
                    ).tolist()
                    similarities = t.tensor(similarities)
                
                # Sort similarities in descending order to try alternatives if needed
                similarity_indices = t.argsort(similarities, descending=True)
                
                # Try to find a match where the model's next token prediction matches the target token
                found_valid_match = False
                for sim_idx in similarity_indices:
                    idx = sim_idx.item()
                    best_match_row = matching_low_df.iloc[idx]
                    similarity_score = similarities[idx].item()
                    
                    # Double check that the model's prediction is the target token
                    with t.no_grad():                        
                        inputs = t.tensor(best_match_row['processed_context']).unsqueeze(0).to(device)
                        outputs = model(inputs)
                        logits = outputs.logits[0, -1, :]
                        predicted_token_id = t.argmax(logits).item()
                    
                    if predicted_token_id == target_token:
                        found_valid_match = True
                        
                        # Truncate both contexts to same length (up to diverging position)
                        high_context_truncated = high_row['processed_context_decoded']
                        low_context_truncated = best_match_row['processed_context_decoded']
                        
                        # Create contrastive pair
                        pair = {
                            'clean': high_context_truncated,
                            'corrupt': low_context_truncated,
                            'answers': [high_row['next_generated_tokens'][0]], # argmax
                            'wrong_answers': [high_row['next_gt_tokens'][0]], # memorization start
                            'similarity_score': float(similarity_score),
                            'diverging_position': div_pos
                        }
                        
                        contrastive_pairs.append(pair)
                        break
                
                if not found_valid_match:
                    logger.info(f"No valid match found for high example {i} where model predicts the target token")
                    continue
            else:
                logger.info(f"Skipping high example {i}: diverging position is at the end of context")
                continue
        else:
            # Fall back to original behavior if no diverging_position
            if sim_metric == 'embedding':
                low_contexts = low_mem_df['decoded_context'].tolist()
                low_embeddings = get_model_embeddings(low_contexts, tokenizer=tokenizer, model=model, batch_size=batch_size)
                
                # Calculate cosine similarity between current high embedding and all low embeddings
                similarities = t.nn.functional.cosine_similarity(high_embeddings[i].unsqueeze(0), low_embeddings).cpu()
            elif sim_metric == 'token_overlap':
                high_context = high_row['processed_context']
                similarities = []
                for _, low_row in low_mem_df.iterrows():
                    overlap = calculate_token_overlap(high_context, low_row['decoded_context'])
                    similarities.append(overlap)
                similarities = t.tensor(similarities)
            
            # Get the index of the most similar low example
            most_similar_idx = t.argmax(similarities).item()
            similarity_score = similarities[most_similar_idx].item()
            
            if len(tokenizer(high_row['processed_context'])['input_ids']) != len(tokenizer(low_mem_df.iloc[most_similar_idx]['decoded_context'])['input_ids']):
                logger.info(f"Skipping pair {i} because of length mismatch after decoding")
                continue
            
            # Create contrastive pair
            pair = {
                'clean': high_row['processed_context'],
                'corrupt': low_mem_df.iloc[most_similar_idx]['decoded_context'],
                'answers': [tokenizer.batch_decode(high_row['completion'])[0]],
                'wrong_answers': [tokenizer.batch_decode(low_mem_df.iloc[most_similar_idx]['completion'])[0]],
                'similarity_score': float(similarity_score)
            }
            
            contrastive_pairs.append(pair)
    
    # Sort by similarity score
    contrastive_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    logger.info(f"Created {len(contrastive_pairs)} contrastive pairs")
    return {'prompts': contrastive_pairs}

def merge_results(output_dir, dataset_name, model_name, prompt_tokens, generation_tokens, num_jobs, threshold, metric, contrastive_mode):
    """
    Merge results from multiple jobs into a single file
    """
    dataset_short_name = dataset_name.split('/')[-1]
    model_short_name = model_name.split('/')[-1]
    
    all_results = []
    
    for job_id in range(num_jobs):
        job_path = Path(f"{output_dir}/contrastive_{dataset_short_name}_{threshold}_{model_short_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}_job{job_id}.json")
        if job_path.exists():
            with open(job_path, 'r') as f:
                job_results = json.load(f)
            all_results.extend(job_results["prompts"])
            logger.info(f"Loaded job result from {job_path} ({len(job_results['prompts'])} samples)")
        else:
            logger.warning(f"Warning: Missing results file for job {job_id}: {job_path}")
    
    if not all_results:
        logger.warning("No results found to merge!") 
        return None
    
    # Combine all results
    merged_results = {"prompts": all_results}
    
    # Save merged results
    merged_path = Path(f"{output_dir}/contrastive_{dataset_short_name}_{threshold}_{model_short_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}.json")
    with open(merged_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    logger.info(f"Saved merged results to {merged_path} ({len(merged_results['prompts'])} total samples)")
    
    # Merge JSONL files if in divergence mode
    if contrastive_mode == "divergence":
        # Create output path for merged JSONL
        merged_jsonl_path = merged_path.with_name(f"{merged_path.stem}.jsonl")
        
        # Open output file
        with open(merged_jsonl_path, 'w') as outfile:
            merged_count = 0
            
            # Read and merge all JSONL files
            for job_id in range(num_jobs):
                jsonl_path = Path(f"{output_dir}/contrastive_{dataset_short_name}_{threshold}_{model_short_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}_job{job_id}.jsonl")
                if jsonl_path.exists():
                    with open(jsonl_path, 'r') as infile:
                        # Count lines for reporting
                        lines = infile.readlines()
                        merged_count += len(lines)
                        # Write all lines to the merged file
                        for line in lines:
                            outfile.write(line)
                else:
                    logger.warning(f"Warning: Missing JSONL file for job {job_id}: {jsonl_path}")
            
            logger.info(f"Saved merged JSONL results to {merged_jsonl_path} ({merged_count} total samples)")
    
    return merged_results

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--dataset", type=str, default="timaeus/pile-wikipedia_en", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=1.00, help="Memorization score threshold")
    parser.add_argument("--dataset_path", type=str, default="data/results/mem_scores_wikipedia-full_gpt-neo-125m_50_50.parquet", help="Override path to dataset")
    parser.add_argument("--metric", type=str, default="bleu", choices=["memorization", "bleu"], help="Benchmark metric to use: 'memorization' for exact memorization or 'bleu' for approximate memorization")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Model to use")
    parser.add_argument("--prompt_tokens", type=int, default=50, help="Number of tokens to use as prompt")
    parser.add_argument("--generation_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--contrastive_mode", type=str, default="dataset", choices=["divergence", "dataset"], 
                        help="Whether to create contrastive pairs based on the token divergence position or take pairs of memorized vs non-memorized examples")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Directory to save results")
    parser.add_argument("--job_id", type=int, default=None, help="Current job ID (for Slurm array jobs)")
    parser.add_argument("--num_jobs", type=int, default=None, help="Total number of jobs (for Slurm array jobs)")
    parser.add_argument("--merge_only", action="store_true", help="Only merge results from previous jobs without computing")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    threshold = args.threshold
    metric = args.metric
    model_name = args.model_name
    short_model_name = model_name.split('/')[-1]
    dataset = args.dataset
    dataset_path = args.dataset_path
    short_dataset = dataset.split('/')[-1]
    output_dir = args.output_dir
    
    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens    
    contrastive_mode = args.contrastive_mode
    
    job_id = args.job_id
    num_jobs = args.num_jobs
    merge_only = args.merge_only
    
    # Update logger with job ID
    logger = JobLogger.get_logger(job_id)
    
    # For Slurm array jobs, we can also get job_id from environment variable
    if job_id is None:
        # Handle both array and non-array job modes
        if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
            # Array job mode
            logger.info("Running in array job mode")
            array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            ntasks = int(os.environ.get("SLURM_NTASKS", 1))
            procid = int(os.environ.get("SLURM_PROCID", 0))
            job_id = array_task_id * ntasks + procid
            num_jobs = array_task_count * ntasks
            logger.info(f"Using SLURM_ARRAY_TASK_ID={array_task_id}, SLURM_NTASKS={ntasks}, SLURM_PROCID={procid}, job_id={job_id}")
        elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
            # Single job with tasks mode
            logger.info("Running in single job mode with multiple tasks")
            procid = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            job_id = procid
            num_jobs = ntasks
            logger.info(f"Using SLURM_PROCID={procid}, SLURM_NTASKS={ntasks}")
        
        # Update logger with new job ID
        logger = JobLogger.get_logger(job_id)
    
    if num_jobs is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        num_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        logger.info(f"Using SLURM_ARRAY_TASK_COUNT={num_jobs}")
    
    logger.info(f"Job ID: {job_id}, Number of jobs: {num_jobs}")

    # If only merging results
    if merge_only:
        if num_jobs is None:
            logger.error("Error: --num_jobs must be specified when using --merge_only")
            exit(1)
        
        merge_results(output_dir, dataset, model_name, prompt_tokens, generation_tokens, num_jobs, threshold, metric, contrastive_mode)
        exit(0)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine output path based on whether running as part of a job array
    if job_id is not None and num_jobs is not None:
        output_path = Path(f"{output_dir}/contrastive_{short_dataset}_{threshold}_{short_model_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}_job{job_id}.json")
    else:
        job_id = 0
        num_jobs = 1
        output_path = Path(f"{output_dir}/contrastive_{short_dataset}_{threshold}_{short_model_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}.json")

    # Get the memorization scores file
    if dataset_path is None:
        mem_scores_file = Path(f"{output_dir}/mem_scores_{short_dataset}_{short_model_name}_{prompt_tokens}_{generation_tokens}.json")
    else:
        mem_scores_file = Path(dataset_path)
    
    # Check if the file exists in the dataset folder
    if not mem_scores_file.exists():
        raise FileNotFoundError(f"Could not find memorization scores file at {mem_scores_file}")    

    # if output_path.exists():
    #     with open(output_path, 'r') as f:
    #         all_sequences = json.load(f)["prompts"]
    #     logger.info(f"Loaded {len(all_sequences)} contrastive pairs from {output_path}")
    #     exit(0)

    if mem_scores_file.suffix == ".parquet":
        df = pd.read_parquet(mem_scores_file)
    elif mem_scores_file.suffix == ".json":
        # For JSON files, read everything then shard in memory
        # WARNING: wil take a lot of time for large json
        df = pd.read_json(mem_scores_file)        
    else:
        raise ValueError(f"Unsupported file type: {mem_scores_file.suffix}")

    # Shard the dataframe
    total_rows = len(df)
    rows_per_job = total_rows // num_jobs
    start_row = job_id * rows_per_job
    end_row = start_row + rows_per_job if job_id < num_jobs - 1 else total_rows    
    logger.info(f"Job {job_id}/{num_jobs}: Processing rows {start_row} to {end_row} out of {total_rows}")
    df = df.iloc[start_row:end_row]

    len_before = len(df)
    df = df.drop_duplicates(subset=['decoded_context','decoded_completion','decoded_true_completion'])
    len_after = len(df)
    if len_before != len_after:
        logger.info(f"Removed {len_before - len_after} duplicate examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if contrastive_mode == "divergence":
        df = df[df['memorization_score'] >= threshold]
        df = df.sort_values('memorization_score', ascending=False)

        if device.type == "cuda":
            model = model.half()
        
        batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
        all_sequences = defaultdict(list)
        for batch in tqdm(batches,desc="Processing batches"):
            sequences = find_minimum_context_for_memorization(model,batch,tokenizer,benchmark_metric=metric)
            for key in sequences:
                all_sequences[key].extend(sequences[key])
        
        # Save results for this job
        autocircuit_data = save_autocircuit_ds(all_sequences, output_path, tokenizer)
        save_jsonl(all_sequences, output_path.with_name(f"{output_path.stem}.jsonl"), tokenizer)

    elif contrastive_mode=="dataset":
        # Check if the output from divergence mode exists and load it if it does
        divergence_output_path = Path(f"{output_dir}/contrastive_{short_dataset}_{threshold}_{short_model_name}_{prompt_tokens}_{generation_tokens}_{metric}_divergence.jsonl")
        if divergence_output_path.exists():
            logger.info(f"Found existing divergence output at {divergence_output_path}, loading...")
            divergence_df = pd.read_json(divergence_output_path, lines=True)
            divergence_df = divergence_df[divergence_df["score_before"] >= 0.75]
            def get_full_context(row:pd.Series):
                new_tokens_num = prompt_tokens-row["diverging_position"]
                return row["minimal_context"] + "".join(row["next_gt_tokens"][:new_tokens_num])
            divergence_df["full_context"] = divergence_df.apply(get_full_context, axis=1)
            divergence_df = divergence_df[["full_context","diverging_position","next_gt_tokens", "next_generated_tokens"]]
            df = df.merge(divergence_df, left_on="decoded_context", right_on="full_context", how="left")

        contrastive_pairs = create_contrastive_pairs(
                df=df,
                tokenizer=tokenizer,
                model=model,
                high_threshold=threshold,
                low_threshold=0.0,                
                batch_size=batch_size,
                sim_metric="token_overlap"
            )
        # Save the dataset for this job
        with open(output_path, 'w') as f:
            json.dump(contrastive_pairs, f, indent=2)
        
        logger.info(f"Saved AutoCircuit dataset with {len(contrastive_pairs['prompts'])} prompt pairs to {output_path}")