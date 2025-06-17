#!/usr/bin/env python3
from datasets import load_dataset
import json
import os
import argparse
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def filter_func(batch, pile_set_name):
    return [x.get("pile_set_name") == pile_set_name for x in batch["meta"]]    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract data from a specific pile set")
    parser.add_argument("--pile-set", type=str, default="Wikipedia (en)", 
                        help="Name of the pile set to filter (default: 'Wikipedia (en)')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output files (default: <pile_set_name>_samples)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of samples per batch when writing parquet file (default: 1000)")
    args = parser.parse_args()
    
    pile_set_name = args.pile_set
    if args.output_dir is None:
        output_dir = f"{pile_set_name.lower().replace(' ', '_')}_samples"
    else:
        output_dir = args.output_dir
    
    # Load the pile-uncopyrighted dataset
    print(f"Loading dataset...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    filtered = dataset.filter(lambda batch: filter_func(batch, pile_set_name), batched=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for specified pile set samples
    print(f"Filtering {pile_set_name} samples...")            
    # Save samples to parquet file
    output_file = f"{output_dir}/{pile_set_name.lower().replace(' ', '_')}.parquet"
    print(f"Saving to {output_file}...")
    
    len_samples = 0    
    batch = []
    schema = pa.schema([
        pa.field('text', pa.string()),
        pa.field('meta', pa.struct([
            pa.field('pile_set_name', pa.string()),            
        ]))
    ])
    writer = pq.ParquetWriter(output_file, schema)
    try:
        for sample in tqdm(filtered):
            batch.append(sample)
            len_samples += 1
            
            # When batch is full, write to parquet
            if len(batch) >= args.batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)                
                writer.write_table(table)
                batch = []
        
        # Write any remaining samples
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)            
            writer.write_table(table)
    
    finally:
        if writer:
            writer.close()
    
    print(f"Saved {len_samples} samples to {output_file}")

if __name__ == "__main__":
    main()
