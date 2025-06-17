# Understanding Verbatim Memorization in LLMs Through Circuit Discovery

This repository implements a pipeline for discovering circuits responsible for verbatim memorization in large language models. The pipeline consists of four main stages: dataset (The Pile) collection, contrastive dataset creation, circuit discovery, and circuit verification.

## Pipeline Overview

The complete pipeline can be executed using `run_pipeline.sh`, which orchestrates the following stages:

### 1. Memorization Score Calculation (`memorization_score.py`)

Downloads a specified dataset and calculates memorization scores for each sample by:
- Using the first `n` tokens as context prompts
- Generating `y` tokens with the model
- Computing exact token match scores between generated and ground truth completions
- Saving contexts, completions, and memorization scores to JSON format

**Usage:**
```bash
python memorization_score.py \
    --model_name "EleutherAI/gpt-neo-125m" \
    --prompt_tokens 50 \
    --generation_tokens 50 \
    --dataset "timaeus/pile-wikipedia_en"
```

Note: this works directly with preprocessed huggingface dataset. Instead, you can first download some subsets of the pile by using [download_pile_subset.sh](download_pile_subset.sh) and then use [`memorization_score.py`](memorization_score.py) with downloaded path.

### 2. Contrastive Dataset Creation (`contrastive_dataset.py`)

Creates contrastive datasets for circuit analysis using two approaches:

#### Branch Decision (`--contrastive_mode divergence`)
This approach focuses on the precise moment memorization breaks down:

1. **Divergence point detection**: For each memorized sample, the algorithm progressively shortens the context until there's a significant relative drop (>30% by default) in the BLEU-4 score compared to the previous context length, AND the model's next token differs from ground truth
2. **Clean examples**: Original memorized context truncated to the divergence point + correct next token
3. **Corrupt examples**: Same truncated context + model's predicted (incorrect) token
4. **Contrastive pair format**: `(context + correct_token, context + wrong_token) → (next_correct_token, next_wrong_token)`
- **Purpose**: Understanding the moment where the model 'decides' to memorize vs. generate novel content


#### Memorization Decision (`--contrastive_mode dataset`)
This approach contrasts memorized vs. non-memorized content, with enhanced precision when divergence data is available:

**Step 1 - Load Branch Decision (optional)**: Optionaly loads results of `--contrastive_mode divergence` run
**Step 2 - Find contrastive pairs**:
- **With divergence data**: Finds low-memorization samples that have the same token at the divergence position as the high-memorization sample, then verifies the model would predict that same token, ensuring the contrast is at the exact decision point
- **Without divergence data**: Uses model embeddings or token overlap to find semantically similar pairs between high and low memorization samples
- **Similarity calculation**: Uses cosine similarity of model embeddings by default

**Contrastive pair format**: `(memorized_context, non_memorized_context) → (model_prediction, correct_answer)`
**Purpose**: Understanding what distinguishes memorizable from non-memorizable content at the neural level


**Usage:**
```bash
python contrastive_dataset.py \
    --dataset "timaeus/pile-wikipedia_en" \
    --model_name "EleutherAI/gpt-neo-125m" \
    --threshold 0.75 \
    --contrastive_mode "dataset"  # or "divergence"
```

### 3. Circuit Discovery (`find_circuits.py`)

Uses [AutoCircuit library](https://ufo-101.github.io/auto-circuit/) to discover minimal neural circuits responsible for memorization behavior:

1. **Edge Attribution**: Applies EAP-IG (Edge Attribution Patching with Integrated Gradients) to compute importance scores for each model edge
2. **Binary Search**: Finds the minimal set of edges that maintains target performance (default: 85% of baseline)

**Key Parameters:**
- `--grad_function`: Function applied to logits before gradient computation (`logit`, `prob`, `logprob`)
- `--loss_function`: Optimization target (`avg_diff`, `avg_val_wrong`, etc.)
- `--optimize_metric`: Performance metric for circuit search (`logit_diff`, `answer_logit`, etc.)

**Usage:**
```bash
python find_circuits.py \
    --model_name "EleutherAI/gpt-neo-125m" \
    --path "data/results/contrastive_dataset.json" \
    --grad_function "logit" \
    --loss_function "avg_val_wrong"
```

There was an attempt in [`find_circuits_eap.py`](find_circuits_eap.py) to try [original repo by Hanna et. al.](https://github.com/hannamw/EAP-IG) but AutoCircuit patching ended up being much faster.

### 4. Circuit Verification (`verify_circuit.py`)

Validates discovered circuits by:
- Loading pre-computed prune scores and applying specified edge counts
- Evaluating circuit performance on test datasets using [defined metrics](find_circuits.py#L93)
- Comparing against circuits containing random edges
- Computing faithfulness scores relative to full model performance

**Usage:**
```bash
python verify_circuit.py \
    --prune_scores_path "data/circuits/prune_scores.pkl" \
    --edge_count 50 \
    --dataset_path "data/results/test_dataset.json"
```

## Verification Scripts

The `verify_scripts/` directory contains reproduction scripts for various experimental configurations:

- `verify_mem_decision_*.sh`: Memorization decision experiments
- `verify_branch_*.sh`: Branch decision experiments  
- `verify_ablations_*.sh`: Experiments with different ablation methods
- `verify_*_random.sh`: Random baseline comparisons

## Requirements

See [`requirements.txt`](requirements.txt)
- PyTorch
- Transformers
- AutoCircuit ([my fork](https://github.com/ilyalasy/auto-circuit/tree/tokenization) that fixes couple bugs)
- EAP (optional) ([my fork](https://github.com/ilyalasy/EAP-IG) with some changes needed to make it all run during my experiments)
