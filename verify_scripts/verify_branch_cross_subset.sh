#!/bin/bash
 
#SBATCH --partition=GPU-a100        # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:a100:1
#SBATCH -J mem_circuits           # job name placeholder
#SBATCH -o logs/cross_subset_branch_%j.log         # output file with job ID and task ID

# Set up environment
export HF_HOME=<HF_HOME>
export WANDB_API_KEY=<WANDB_API_KEY>

wandb login
source $HOME/miniconda3/bin/activate
conda activate circuits


DATASET_SIZE=1.0

# Configuration 1: neg_avg_diff - neg_logit_diff
echo "Running on Pile-Github (branching) (noising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-github/contrastive_pile-github_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --reverse_clean_corrupt \
  --edge_count 14

echo "Running on Pile-Github (branching) (denoising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-github/contrastive_pile-github_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --edge_count 14

# Configuration 1: neg_avg_diff - neg_logit_diff
echo "Running on Pile-Emails (branching) (noising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-enron_emails/contrastive_pile-enron_emails_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --reverse_clean_corrupt \
  --edge_count 14

# Configuration 1: neg_avg_diff - neg_logit_diff
echo "Running on Pile-Emails (branching) (denoising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-enron_emails/contrastive_pile-enron_emails_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \  
  --edge_count 14

# Configuration 1: neg_avg_diff - neg_logit_diff
echo "Running on Pile-CC (branching) (noising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-pile-cc/contrastive_pile-pile-cc_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_correct_ans_percent \
  --reverse_clean_corrupt \
  --edge_count 14

# Configuration 1: neg_avg_diff - neg_logit_diff
echo "Running on Pile-CC (branching) (denoising)"
python verify_circuit.py \
  --prune_scores_path data/paper/branch_comparison/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path data/results/pile-pile-cc/contrastive_pile-pile-cc_0.75_gpt-neo-125m_50_50_bleu_divergence.json \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_incorrect_ans_percent \
  --edge_count 14
