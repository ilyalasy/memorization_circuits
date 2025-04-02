#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Configuration ---
# Model parameters
MODEL_NAME="EleutherAI/pythia-70m-deduped" # Model identifier (used across all scripts)
PROMPT_TOKENS=32                          # Number of prompt tokens (used across all scripts)
GENERATION_TOKENS=96                      # Number of generation tokens (used across all scripts)

# Memorization Score parameters
MEM_BATCH_SIZE=128                        # Batch size for memorization_score.py
DATASET="timaeus/pile-wikipedia_en"       # Dataset for memorization_score.py
OUTPUT_DIR="data/results"                 # Output directory for memorization_score.py

# Contrastive Dataset & Find Circuits parameters
THRESHOLD=0.5                             # Memorization score threshold
METRIC="bleu"                             # Metric ('memorization' or 'bleu')

# Contrastive Dataset parameters
CONTRASTIVE_BATCH_SIZE=8                  # Batch size for contrastive_dataset.py

# Find Circuits EAP parameters
EAP_BATCH_SIZE=32                         # Batch size for find_circuits_eap.py

# --- Pipeline Steps ---

echo "Running memorization score calculation..."
python memorization_score.py \
    --model_name "$MODEL_NAME" \
    --prompt_tokens "$PROMPT_TOKENS" \
    --generation_tokens "$GENERATION_TOKENS" \
    --batch_size "$MEM_BATCH_SIZE" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR"

echo "Running contrastive dataset generation..."
python contrastive_dataset.py \
    --model_name "$MODEL_NAME" \
    --prompt_tokens "$PROMPT_TOKENS" \
    --generation_tokens "$GENERATION_TOKENS" \
    --batch_size "$CONTRASTIVE_BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --metric "$METRIC"

echo "Running circuit finding..."
python find_circuits_eap.py \
    --model_name "$MODEL_NAME" \
    --prompt_tokens "$PROMPT_TOKENS" \
    --generation_tokens "$GENERATION_TOKENS" \
    --batch_size "$EAP_BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --metric "$METRIC"

echo "Pipeline finished."

# Note: The output file paths in contrastive_dataset.py and find_circuits_eap.py 
# are derived implicitly from the arguments (model_name, tokens, threshold, metric).
# The OUTPUT_DIR variable is only explicitly used by memorization_score.py. 