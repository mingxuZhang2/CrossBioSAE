#!/bin/bash
#SBATCH -J brca2_evo2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o logs/brca2_evo2.out
#SBATCH -e logs/brca2_evo2.out

set -euo pipefail
echo "=== host $(hostname)  $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/variant

python -c "import torch,flash_attn;from evo2.models import Evo2;print('imports OK')"
python -u scripts/extract_brca2_evo2.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --genome data/variant/GRCh38.fa.gz \
  --local-path models/evo2_7b.pt \
  --output-dir results/variant \
  --score-batch 8 \
  --emb-batch 2

echo "BRCA2_EVO2_DONE_$?  $(date)"
