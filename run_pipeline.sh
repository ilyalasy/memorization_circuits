#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Configuration ---

GLOBAL_BATCH_SIZE=256
# Model parameters
MODEL_NAME="EleutherAI/gpt-neo-125m" # Model identifier (used across all scripts)
SHORT_MODEL_NAME=${MODEL_NAME##*/}
PROMPT_TOKENS=50                          # Number of prompt tokens (used across all scripts)
GENERATION_TOKENS=50                      # Number of generation tokens (used across all scripts)

# Memorization Score parameters
MEM_BATCH_SIZE=$GLOBAL_BATCH_SIZE                        # Batch size for memorization_score.py
DATASET="timaeus/pile-wikipedia_en"       # Dataset for memorization_score.py
OUTPUT_DIR="data/results"                 # Output directory for memorization_score.py

# Contrastive Dataset & Find Circuits parameters
THRESHOLD=0.75                             # Memorization score threshold
METRIC="bleu"                             # Metric ('memorization' or 'bleu')

# Contrastive Dataset parameters
CONTRASTIVE_BATCH_SIZE=$GLOBAL_BATCH_SIZE                 # Batch size for contrastive_dataset.py
CONTRASTIVE_MODE="divergence"                             # Mode ('divergence' or 'dataset')

# Find Circuits EAP parameters
EAP_BATCH_SIZE=64                      # Batch size for find_circuits_eap.py


# Echo all parameters
echo "--- Pipeline Parameters ---"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MODEL_NAME: $MODEL_NAME"
echo "PROMPT_TOKENS: $PROMPT_TOKENS"
echo "GENERATION_TOKENS: $GENERATION_TOKENS"
echo "DATASET: $DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "METRIC: $METRIC"
echo "THRESHOLD: $THRESHOLD"
echo "CONTRASTIVE_MODE: $CONTRASTIVE_MODE"
echo "MEM_BATCH_SIZE: $MEM_BATCH_SIZE"
echo "CONTRASTIVE_BATCH_SIZE: $CONTRASTIVE_BATCH_SIZE"
echo "EAP_BATCH_SIZE: $EAP_BATCH_SIZE"
echo "-------------------------"

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
    --metric "$METRIC" \
    --contrastive_mode "$CONTRASTIVE_MODE"

echo "Running circuit finding..."
python find_circuits_eap.py \
    --model_name "$MODEL_NAME" \
    --path "$OUTPUT_DIR/contrastive_mem_${THRESHOLD}_${SHORT_MODEL_NAME}_${PROMPT_TOKENS}_${GENERATION_TOKENS}_${METRIC}_${CONTRASTIVE_MODE}.json" \
    --batch_size "$EAP_BATCH_SIZE"

echo "Pipeline finished."

# Note: The output file paths in contrastive_dataset.py and find_circuits_eap.py 
# are derived implicitly from the arguments (model_name, tokens, threshold, metric).
# The OUTPUT_DIR variable is only explicitly used by memorization_score.py. 