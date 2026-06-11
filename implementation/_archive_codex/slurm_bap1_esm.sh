#!/bin/bash
#SBATCH -J bap1_esm
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/bap1_esm.out
#SBATCH -e logs/bap1_esm.out

set -euo pipefail
echo "=== host $(hostname)  $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/variant

python -u scripts/extract_brca2_esm_delta.py \
  --variants data/variant/bap1/bap1_variants.csv \
  --protein data/variant/bap1/bap1_Q96TC6.fasta \
  --output results/variant/bap1_esm_delta.npz \
  --device cuda \
  --batch-size 8

echo "BAP1_ESM_DONE_$?  $(date)"
