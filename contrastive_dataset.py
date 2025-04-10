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
import os

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
    print(f"Saving decoded data to {path}")        
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
    print(f"Total skipped count: {total_skipped_count}/{total_len}")

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
            print(f"Loaded job result from {job_path} ({len(job_results['prompts'])} samples)")
        else:
            print(f"Warning: Missing results file for job {job_id}: {job_path}")
    
    if not all_results:
        print("No results found to merge!") 
        return None
    
    # Combine all results
    merged_results = {"prompts": all_results}
    
    # Save merged results
    merged_path = Path(f"{output_dir}/contrastive_{dataset_short_name}_{threshold}_{model_short_name}_{prompt_tokens}_{generation_tokens}_{metric}_{contrastive_mode}.json")
    with open(merged_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    print(f"Saved merged results to {merged_path} ({len(merged_results['prompts'])} total samples)")
    
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
                    print(f"Warning: Missing JSONL file for job {job_id}: {jsonl_path}")
            
            print(f"Saved merged JSONL results to {merged_jsonl_path} ({merged_count} total samples)")
    
    return merged_results

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate contrastive dataset")
    parser.add_argument("--dataset", type=str, default="timaeus/pile-wikipedia_en", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=1.00, help="Memorization score threshold")
    parser.add_argument("--dataset_path", type=str, default="data/results/mem_scores_wikipedia-full_gpt-neo-125m_50_50_filtered100.parquet", help="Override path to dataset")
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
    
    # For Slurm array jobs, we can also get job_id from environment variable
    if job_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"Using SLURM_ARRAY_TASK_ID={job_id}")
    
    if num_jobs is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        num_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(f"Using SLURM_ARRAY_TASK_COUNT={num_jobs}")
    
    # If only merging results
    if merge_only:
        if num_jobs is None:
            print("Error: --num_jobs must be specified when using --merge_only")
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

    if output_path.exists():
        with open(output_path, 'r') as f:
            all_sequences = json.load(f)["prompts"]
        print(f"Loaded {len(all_sequences)} contrastive pairs from {output_path}")
        exit(0)

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
    print(f"Job {job_id}/{num_jobs}: Processing rows {start_row} to {end_row} out of {total_rows}")
    df = df.iloc[start_row:end_row]

    len_before = len(df)
    df = df.drop_duplicates(subset=['decoded_context','decoded_completion','decoded_true_completion'])
    len_after = len(df)
    if len_before != len_after:
        print(f"Removed {len_before - len_after} duplicate examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if contrastive_mode == "divergence":
        df = df[df['memorization_score'] >= threshold]
        df = df.sort_values('memorization_score', ascending=False)

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
        contrastive_pairs = create_contrastive_pairs(
                df=df,
                high_threshold=threshold,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
        # Save the dataset for this job
        with open(output_path, 'w') as f:
            json.dump(contrastive_pairs, f, indent=2)
        
        print(f"Saved AutoCircuit dataset with {len(contrastive_pairs['prompts'])} prompt pairs to {output_path}")