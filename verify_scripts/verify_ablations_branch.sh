#!/bin/bash
 
#SBATCH --partition=GPU-a100        # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:a100:1
#SBATCH -J mem_circuits           # job name placeholder
#SBATCH -o logs/ablations_branch_%j.log         # output file with job ID and task ID

# Set up environment
export HF_HOME=<HF_HOME>
export WANDB_API_KEY=<WANDB_API_KEY>
wandb login
source $HOME/miniconda3/bin/activate
conda activate circuits

DATASET_PATH="data/results/contrastive_pile-wikipedia_en_1.0_gpt-neo-125m_50_50_bleu_divergence.json"
DATASET_SIZE=0.5

# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Branch Comparison - noising - ZERO "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --ablation_type ZERO \
  --edge_count 14

  # Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Branch Comparison - noising - TOKENWISE_MEAN_CLEAN "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --ablation_type TOKENWISE_MEAN_CLEAN \
  --edge_count 14

  # Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Branch Comparison - noising - TOKENWISE_MEAN_CORRUPT "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --ablation_type TOKENWISE_MEAN_CORRUPT \
  --edge_count 14


  # Configuration 2: neg_avg_diff - neg_logit_diff (denoising)
echo "Processing: Branch Comparison - denoising - ZERO "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --ablation_type ZERO \
  --edge_count 14

  # Configuration 3: neg_avg_diff - neg_logit_diff (denoising)
echo "Processing: Branch Comparison - denoising - TOKENWISE_MEAN_CORRUPT "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --ablation_type TOKENWISE_MEAN_CORRUPT \
  --edge_count 14


  # Configuration 4: neg_avg_diff - neg_logit_diff (denoising)
echo "Processing: Branch Comparison - denoising - TOKENWISE_MEAN_CLEAN "
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --ablation_type TOKENWISE_MEAN_CLEAN \
  --edge_count 14

echo "Done"