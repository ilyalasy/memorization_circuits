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
# logging.get_logger("transformers").setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
    
    # Process dataset in batches
    batches = dataset.batch(batch_size=batch_size)
        
    for batch in tqdm(batches, total=len(batches), desc="Processing dataset"):
        batch_texts = tokenizer(batch["text"], padding=True, return_tensors="pt")
        tokens = batch_texts["input_ids"].to(device)
            
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

def main():    
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate memorization scores for language models")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m-deduped", 
                        help="Name of the model to use")
    parser.add_argument("--prompt_tokens", type=int, default=32, 
                        help="Number of tokens to use as prompt")
    parser.add_argument("--generation_tokens", type=int, default=96, 
                        help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for processing")
    parser.add_argument("--dataset", type=str, default="timaeus/pile-wikipedia_en", 
                        help="Dataset to use for evaluation")
    parser.add_argument("--output_dir", type=str, default="data/results", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parameters from arguments
    model_name = args.model_name
    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    batch_size = args.batch_size
    dataset_name = args.dataset
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16) # torch_dtype=torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train") #streaming=True
    
    print(f"Calculating memorization score (prompt: {prompt_tokens} tokens, generate: {generation_tokens} tokens)")
    
    dataset_short_name = dataset_name.split('/')[-1]
    model_short_name = model_name.split('/')[-1]
    
    path = output_dir / f"mem_scores_{dataset_short_name}_{model_short_name}_{prompt_tokens}_{generation_tokens}.json"
    
    if path.exists():
        results_df = pd.read_json(path, orient='records')
        print(f"Loaded existing results from {path}")
    else:
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

    # Calculate and print summary statistics
    print(f"Mean memorization score: {results_df['memorization_score'].mean():.4f}")
    print(f"Median memorization score: {results_df['memorization_score'].median():.4f}")
    print(f"Top 10 memorization samples:")
    top_samples = results_df.sort_values('memorization_score', ascending=False).head(10)
    print(top_samples[['memorization_score', 'decoded_context', 'decoded_completion']])

if __name__ == "__main__":
    main() 