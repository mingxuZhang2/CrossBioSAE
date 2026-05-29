#!/bin/bash
#SBATCH -J brca1_evo2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -o logs/brca1_evo2.out
#SBATCH -e logs/brca1_evo2.out
# Evo2-7b zero-shot LLR + window-token embedding deltas for BRCA1.
# Compute nodes have NO internet -> load weights from local .pt, force HF offline.
set -e
echo "=== host $(hostname)  $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python -c "import torch,flash_attn;from evo2.models import Evo2;print('imports OK')"
python scripts/extract_brca1_evo2.py \
  --variants data/variant/brca1/brca1_variants.csv \
  --chr17 data/variant/brca1/chr17_GRCh37.fna.gz \
  --local_path models/evo2_7b.pt \
  --output_dir results/variant
echo "BRCA1_EVO2_DONE_$?  $(date)"
