import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from functools import partial
import os
from torch.utils.data import DataLoader
# logging.get_logger("transformers").setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def tokenize_batch(batch:dict[str,list[str]], tokenizer:AutoTokenizer,):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="np")

def find_optimal_batch_size(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, dataset:Dataset, prompt_tokens:int, generation_tokens:int):
    """
    Automatically determine the maximum batch size that fits in GPU memory
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        dataset: The dataset to evaluate on
        prompt_tokens: Number of tokens to use as prompt
        generation_tokens: Number of tokens to generate
        
    Returns:
        int: Optimal batch size
    """
    print("Finding optimal batch size...")
    
    # Start with a small batch size
    low = 1
    high = 65536  # Initial upper bound guess
    optimal = low
    
    # Get a sample batch for testing
    sample_data = dataset.take(high) #next(iter())

    print(len(sample_data["input_ids"]))
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Try processing a batch of this size
            print(f"Testing batch size: {mid}")
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()                        
                        
            tokens = torch.tensor(sample_data["input_ids"])
            context = tokens[:mid, :prompt_tokens].to(device)
            
            # Test generation with this batch size
            print(f"  Running generation with batch size {mid}...")
            with torch.no_grad():
                model.generate(
                    input_ids=context,
                    max_new_tokens=generation_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    min_new_tokens=generation_tokens
                )
            
            # Track peak memory usage during generation
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)            
            print(f"  Peak CUDA memory during generation: {peak_mem:.2f} GB")            
            
            # If successful, this is our new minimum viable batch size
            optimal = mid
            low = mid + 1
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Out of memory, reduce batch size
            print(f"  Batch size {mid} too large, got error: {str(e)[:100]}...")
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"  Peak CUDA memory before OOM: {peak_mem:.2f} GB")
            high = mid - 1
            torch.cuda.empty_cache()  # Clear GPU memory after OOM
        sample_data = dataset.take(high) #next(iter())
    
    # Get final GPU stats
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Optimal batch size found: {optimal}")
    print(f"Total GPU memory: {total_gpu_memory:.2f} GB")
    print(f"Peak memory usage with batch size {optimal}: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Memory utilization: {torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.2f}%")
    
    return optimal


def calculate_memorization_score(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, dataset:Dataset, prompt_tokens=50, generation_tokens=50, batch_size=32) -> pd.DataFrame:
    """
    Calculate memorization score for a model on a dataset using batched processing.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        dataset: The dataset to evaluate on
        prompt_tokens: Number of tokens to use as prompt (n)
        generation_tokens: Number of tokens to generate (y)        
        batch_size: Number of samples to process in each batch
        
    Returns:
        pd.DataFrame: DataFrame containing memorization analysis results
    """
    results = []
    
    # Set device
    model = model.to(device).half()
    model.eval()
    
    def collate_fn(batch):        
        return torch.tensor([x["input_ids"] for x in batch])

    # Process dataset in batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16,collate_fn=collate_fn)
        
    for batch in tqdm(dataloader, desc="Processing dataset..."):
        tokens = batch.to(device)
            
        context = tokens[:, :prompt_tokens]
        true_completion = tokens[:, prompt_tokens:prompt_tokens + generation_tokens]
        
        # Generate tokens for the batch
        # with torch.no_grad():
        outputs = model.generate(
            input_ids=context,
            max_new_tokens=generation_tokens,
            do_sample=False,  # Use greedy decoding for deterministic output
            pad_token_id=tokenizer.eos_token_id,  # Handle padding properly                
            use_cache=True,
            min_new_tokens=generation_tokens
        )

        # Extract the generated tokens
        completion = outputs[:, prompt_tokens:prompt_tokens + generation_tokens]
        
        # Calculate memorization scores
        mem_scores = (completion == true_completion).sum(dim=1) / generation_tokens       
        
        # Decode texts for human-readable output
        decoded_context = tokenizer.batch_decode(context, skip_special_tokens=True)
        decoded_completion = tokenizer.batch_decode(completion, skip_special_tokens=True)
        decoded_true_completion = tokenizer.batch_decode(true_completion, skip_special_tokens=True)
        
        # Add to results
        for i in range(len(context)):
            results.append({
                'context': context[i].cpu().tolist(),
                'decoded_context': decoded_context[i],
                'completion': completion[i].cpu().tolist(),
                'decoded_completion': decoded_completion[i],
                'true_completion': true_completion[i].cpu().tolist(),
                'decoded_true_completion': decoded_true_completion[i],
                'memorization_score': mem_scores[i].item(),
            })
    
    # Convert to DataFrame
    return pd.DataFrame(results)
    
def merge_results(output_dir, dataset_name, model_name, prompt_tokens, generation_tokens, num_jobs):
    """
    Merge results from multiple jobs into a single file
    """
    dataset_short_name = dataset_name.split('/')[-1]
    model_short_name = model_name.split('/')[-1]
    
    all_results = []
    
    for job_id in range(num_jobs):
        job_path = output_dir / f"mem_scores_{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}_job{job_id}.json"
        if job_path.exists():
            job_results = pd.read_json(job_path, orient='records')
            all_results.append(job_results)
            print(f"Loaded job result from {job_path} ({len(job_results)} samples)")
        else:
            print(f"Warning: Missing results file for job {job_id}: {job_path}")
    
    if not all_results:
        print("No results found to merge!")
        return None
    
    # Combine all results
    merged_results = pd.concat(all_results, ignore_index=True)
    
    # Save merged results
    merged_path = output_dir / f"mem_scores_{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}_merged.json"
    merged_results.to_json(merged_path, orient='records', indent=2)
    print(f"Saved merged results to {merged_path} ({len(merged_results)} total samples)")
    
    return merged_results

def main():    
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate memorization scores for language models")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", 
                        help="Name of the model to use")
    parser.add_argument("--prompt_tokens", type=int, default=50, 
                        help="Number of tokens to use as prompt")
    parser.add_argument("--generation_tokens", type=int, default=50, 
                        help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4096, 
                        help="Batch size for processing (0 for auto-detect)")
    parser.add_argument("--dataset", type=str, default="/share/datasets/the-pile/wikipedia/shards", 
                        help="Dataset to use for evaluation")
    parser.add_argument("--output_dir", type=str, default="data/results", 
                        help="Directory to save results")    
    parser.add_argument("--job_id", type=int, default=None,
                        help="Current job ID (for Slurm array jobs)")
    parser.add_argument("--num_jobs", type=int, default=None,
                        help="Total number of jobs (for Slurm array jobs)")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge results from previous jobs without computing")
    
    args = parser.parse_args()
    
    # Parameters from arguments
    model_name = args.model_name
    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    batch_size = args.batch_size
    dataset_name = args.dataset
    output_dir = Path(args.output_dir)
    job_id = args.job_id
    num_jobs = args.num_jobs
    merge_only = args.merge_only
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For Slurm array jobs, we can also get job_id from environment variable
    if job_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"Using SLURM_ARRAY_TASK_ID={job_id}")
    
    if num_jobs is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        num_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(f"Using SLURM_ARRAY_TASK_COUNT={num_jobs}")
    
    dataset_short_name = dataset_name.split('/')[-1]
    model_short_name = model_name.split('/')[-1]
    
    # If only merging results
    if merge_only:
        if num_jobs is None:
            print("Error: --num_jobs must be specified when using --merge_only")
            return
        
        results_df = merge_results(output_dir, dataset_name, model_name, prompt_tokens, generation_tokens, num_jobs)
        if results_df is not None:
            print(f"Mean memorization score: {results_df['memorization_score'].mean():.4f}")
            print(f"Median memorization score: {results_df['memorization_score'].median():.4f}")
            print(f"Top 10 memorization samples:")
            top_samples = results_df.sort_values('memorization_score', ascending=False).head(10)
            print(top_samples[['memorization_score', 'decoded_context', 'decoded_completion']])
        return
    
    # Determine output path based on whether running as part of a job array
    if job_id is not None and num_jobs is not None:
        path = output_dir / f"mem_scores_{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}_job{job_id}.json"
    else:
        job_id = 0
        num_jobs = 1
        path = output_dir / f"mem_scores_{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'        
    
    if path.exists():
        results_df = pd.read_json(path, orient='records')
        print(f"Loaded existing results from {path}")
    else:        
        print(f"Loading dataset: {dataset_name}")
        # Check if dataset_name is a local path or a HuggingFace dataset
        dataset_name = Path(dataset_name) 
        if dataset_name.exists():
            if dataset_name.is_dir():
                # Handle folder containing multiple JSONL files
                jsonl_files = [str(p) for p in dataset_name.glob("*.jsonl")]
            else:
                jsonl_files = [dataset_name]

            dataset = load_dataset("json",data_files=jsonl_files, split="train") #streaming=True


            dataset = dataset.shard(num_shards=num_jobs, index=job_id, contiguous=True)
            dataset_len = dataset.num_rows
        else:
            # Load from HuggingFace Hub
            dataset = load_dataset(dataset_name, split="train") #
            dataset_len = len(dataset)
        print(f"Dataset length: {dataset_len}")


        dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer), batched=True, batch_size=1000, num_proc=32,cache_file_name=f"{dataset_name}/{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}_job{job_id}.cache")
        
        job_desc = f" (Job {job_id+1}/{num_jobs})" if job_id is not None else ""
        print(f"Calculating memorization score{job_desc} (prompt: {prompt_tokens} tokens, generate: {generation_tokens} tokens)")        

        # Load model and tokenizer in each process
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        model = model.to(device).half()
        model.eval()
        
        # Auto-detect batch size if requested
        if batch_size == 0:
            batch_size = find_optimal_batch_size(model, tokenizer, dataset, prompt_tokens, generation_tokens)
            print(f"Using auto-detected batch size: {batch_size}")
        
        results_df = calculate_memorization_score(
            model, 
            tokenizer, 
            dataset, 
            prompt_tokens=prompt_tokens, 
            generation_tokens=generation_tokens,
            batch_size=batch_size
        )
        
        # Save results
        path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_json(path, orient='records', indent=2)
        print(f"Saved results to {path}")

    # Calculate and print summary statistics for this job's results
    print(f"Mean memorization score: {results_df['memorization_score'].mean():.4f}")
    print(f"Median memorization score: {results_df['memorization_score'].median():.4f}")
    print(f"Top 10 memorization samples:")
    top_samples = results_df.sort_values('memorization_score', ascending=False).head(10)
    print(top_samples[['memorization_score', 'decoded_context', 'decoded_completion']])

if __name__ == "__main__":
    main() 