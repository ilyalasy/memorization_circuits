#!/bin/bash
 
#SBATCH --partition=GPU-a100        # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:a100:1
#SBATCH -J mem_circuits           # job name placeholder
#SBATCH -o logs/ablations_mem_decision_%j.log         # output file with job ID and task ID

# Set up environment
export HF_HOME=<HF_HOME>
export WANDB_API_KEY=<WANDB_API_KEY>

wandb login
source $HOME/miniconda3/bin/activate
conda activate circuits

DATASET_PATH="data/results/full-wiki_embeddings/contrastive_pile-wikipedia_en_1.0_gpt-neo-125m_50_50_bleu_dataset.json"
DATASET_SIZE=0.5

# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Memorization Decision ZERO noising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_logit_diff \
  --ablation_type ZERO \
  --edge_count 141

# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Memorization Decision TOKENWISE_MEAN_CLEAN noising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_logit_diff \
  --ablation_type TOKENWISE_MEAN_CLEAN \
  --edge_count 141


# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Memorization Decision TOKENWISE_MEAN_CORRUPT noising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_logit_diff \
  --ablation_type TOKENWISE_MEAN_CORRUPT \
  --edge_count 141

# Configuration 2: avg_val_wrong - neg_logit_diff (denoising)
echo "Processing: Memorization Decision ZERO denoising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric answer_logit \
  --ablation_type ZERO \
  --edge_count 141

# Configuration 2: avg_val_wrong - neg_logit_diff (noising)
echo "Processing: Memorization Decision TOKENWISE_MEAN_CLEAN denoising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric answer_logit \
  --ablation_type TOKENWISE_MEAN_CLEAN \
  --edge_count 141

# Configuration 2: avg_val_wrong - neg_logit_diff (noising)
echo "Processing: Memorization Decision TOKENWISE_MEAN_CORRUPT denoising"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric answer_logit \
  --ablation_type TOKENWISE_MEAN_CORRUPT \
  --edge_count 141