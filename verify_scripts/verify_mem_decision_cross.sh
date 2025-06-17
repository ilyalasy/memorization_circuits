#!/bin/bash
 
#SBATCH --partition=GPU-a100        # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:a100:1
#SBATCH -J mem_circuits           # job name placeholder
#SBATCH -o logs/cross_mem_decision_%j.log         # output file with job ID and task ID

# Set up environment
export HF_HOME=<HF_HOME>
export WANDB_API_KEY=<WANDB_API_KEY>

wandb login
source $HOME/miniconda3/bin/activate
conda activate circuits

DATASET_PATH="data/results/contrastive_pile-wikipedia_en_1.0_gpt-neo-125m_50_50_bleu_divergence.json"
DATASET_SIZE=0.5

# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: logit_diff - accuracy - noising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --edge_count 141


# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: logit_diff - accuracy - denoising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --edge_count 141

# Configuration 1: neg_avg_diff - neg_logit_diff (denoising)
echo "Processing: logit - accuracy - noising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --edge_count 332

# Configuration 1: neg_avg_diff - neg_logit_diff (denoising)
echo "Processing: logit - accuracy - denoising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --edge_count 332
