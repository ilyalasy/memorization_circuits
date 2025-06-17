#!/bin/bash

DATASET_PATH="data/results/full-wiki_embeddings/contrastive_pile-wikipedia_en_1.0_gpt-neo-125m_50_50_bleu_dataset.json"
DATASET_SIZE=0.5

# Configuration 1: neg_avg_diff - neg_logit_diff (noising)
echo "Processing: Memorization Decision - logit_diff - logit_diff"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_logit_diff \
  --edge_count 141

# Configuration 2: avg_val_wrong - neg_logit_diff (noising)
echo "Processing: Memorization Decision - logit - logit_diff"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric neg_logit_diff \
  --edge_count 332

# Configuration 3: neg_avg_diff - wrong_answer_logit (noising)
echo "Processing: Memorization Decision - logit_diff - logit"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric wrong_answer_logit \
  --edge_count 3769

# Configuration 4: avg_val_wrong - wrong_answer_logit (noising)
echo "Processing: Memorization Decision - logit - logit"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --reverse_clean_corrupt \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric wrong_answer_logit \
  --edge_count 1923

# Configuration 5: neg_avg_diff - answer_logit (denoising)
echo "Processing: Memorization Decision - logit_diff - logit"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_neg_avg_diff.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric answer_logit \
  --edge_count 5614

# Configuration 6: avg_val_wrong - answer_logit (denoising)
echo "Processing: Memorization Decision - logit - logit"
python verify_circuit.py \
  --prune_scores_path data/paper/mem_decision/gpt-neo-125m_minimal_circuit_ig5_logit_avg_val_wrong.pkl \
  --dataset_path "$DATASET_PATH" \
  --dataset_size $DATASET_SIZE \
  --optimize_metric answer_logit \
  --edge_count 1923