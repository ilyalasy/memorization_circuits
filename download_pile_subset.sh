#!/bin/bash
 
#SBATCH --partition=GPU-v100       # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:v100:1
#SBATCH -J pile_extraction           # job name placeholder (will be overridden)
#SBATCH -o logs/github-extraction.log        # output file (%j = job ID)


# Set up environment
export HF_HOME=/share/ilya.lasy/huggingface
source $HOME/miniconda3/bin/activate
conda activate circuits

python pile_extraction.py --pile-set "Github" --output-dir "/share/datasets/the-pile/github"