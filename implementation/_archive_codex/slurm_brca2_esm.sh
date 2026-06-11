#!/bin/bash
#SBATCH -J brca2_esm
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/brca2_esm.out
#SBATCH -e logs/brca2_esm.out

set -euo pipefail
echo "=== host $(hostname)  $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/variant

python -u scripts/extract_brca2_esm_delta.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --protein data/variant/brca2/brca2_P51587.fasta \
  --output results/variant/brca2_esm_delta.npz \
  --device cuda \
  --batch-size 8

echo "BRCA2_ESM_DONE_$?  $(date)"
