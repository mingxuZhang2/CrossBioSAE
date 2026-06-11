#!/bin/bash
#SBATCH -J rad51c_evo2_full
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o logs/rad51c_evo2_full_%j.out
#SBATCH -e logs/rad51c_evo2_full_%j.out

set -euo pipefail
echo "=== RAD51C full Evo2 $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/variant

if [[ -s results/variant/rad51c_evo2.npz ]]; then
  echo "results/variant/rad51c_evo2.npz already exists; exiting. $(date)"
  exit 0
fi

python -u scripts/extract_brca2_evo2.py \
  --variants data/variant/rad51c/rad51c_grch38_variants.csv \
  --genome data/variant/GRCh38.fa.gz \
  --local-path models/evo2_7b.pt \
  --output-dir results/variant \
  --output-prefix rad51c \
  --pos-col pos_hg38 \
  --score-batch 8 \
  --emb-batch 2

echo "RAD51C_EVO2_FULL_DONE $(date)"
