#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Parse command line arguments ---
# Default values (same as before)
MODEL_NAME="EleutherAI/gpt-neo-125m"
PROMPT_TOKENS=50
GENERATION_TOKENS=50
GLOBAL_BATCH_SIZE=256
MEM_BATCH_SIZE=$GLOBAL_BATCH_SIZE
DATASET="timaeus/pile-wikipedia_en"
OUTPUT_DIR="data/results"
THRESHOLD=0.75
METRIC="bleu"
CONTRASTIVE_MODE="divergence"
CONTRASTIVE_BATCH_SIZE=$GLOBAL_BATCH_SIZE
EAP_BATCH_SIZE=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --prompt_tokens)
      PROMPT_TOKENS="$2"
      shift 2
      ;;
    --generation_tokens)
      GENERATION_TOKENS="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --metric)
      METRIC="$2"
      shift 2
      ;;
    --contrastive_mode)
      CONTRASTIVE_MODE="$2"
      shift 2
      ;;
    --mem_batch_size)
      MEM_BATCH_SIZE="$2"
      shift 2
      ;;
    --contrastive_batch_size)
      CONTRASTIVE_BATCH_SIZE="$2"
      shift 2
      ;;
    --eap_batch_size)
      EAP_BATCH_SIZE="$2"
      shift 2
      ;;
    --global_batch_size)
      GLOBAL_BATCH_SIZE="$2"
      MEM_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
      CONTRASTIVE_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
      shift 2
      ;;
    *)
      echo "Unknown parameter $1"
      exit 1
      ;;
  esac
done

# Model parameters
SHORT_MODEL_NAME=${MODEL_NAME##*/}
SHORT_DATASET=${DATASET##*/}

# Create dataset-specific directories
DATASET_RESULTS_DIR="${OUTPUT_DIR}/${SHORT_DATASET}"
CIRCUITS_DIR="data/circuits/${SHORT_DATASET}"

# Create the directories if they don't exist
mkdir -p "${DATASET_RESULTS_DIR}"
mkdir -p "${CIRCUITS_DIR}"

# Echo all parameters
echo "--- Pipeline Parameters ---"
echo "MODEL_NAME: $MODEL_NAME"
echo "PROMPT_TOKENS: $PROMPT_TOKENS"
echo "GENERATION_TOKENS: $GENERATION_TOKENS"
echo "DATASET: $DATASET"
echo "DATASET_RESULTS_DIR: $DATASET_RESULTS_DIR"
echo "CIRCUITS_DIR: $CIRCUITS_DIR"
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
    --output_dir "$DATASET_RESULTS_DIR"

echo "Running contrastive dataset generation..."
python contrastive_dataset.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --prompt_tokens "$PROMPT_TOKENS" \
    --generation_tokens "$GENERATION_TOKENS" \
    --batch_size "$CONTRASTIVE_BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --metric "$METRIC" \
    --contrastive_mode "$CONTRASTIVE_MODE" \
    --output_dir "$DATASET_RESULTS_DIR"

echo "Running circuit finding..."
python find_circuits_eap.py \
    --model_name "$MODEL_NAME" \
    --path "${DATASET_RESULTS_DIR}/contrastive_${SHORT_DATASET}_${THRESHOLD}_${SHORT_MODEL_NAME}_${PROMPT_TOKENS}_${GENERATION_TOKENS}_${METRIC}_${CONTRASTIVE_MODE}.json" \
    --batch_size "$EAP_BATCH_SIZE" \
    --output_dir "$CIRCUITS_DIR"

echo "Pipeline finished."

# Note: The output file paths in contrastive_dataset.py and find_circuits_eap.py 
# are derived implicitly from the arguments (model_name, tokens, threshold, metric).
# The OUTPUT_DIR variable is only explicitly used by memorization_score.py. 