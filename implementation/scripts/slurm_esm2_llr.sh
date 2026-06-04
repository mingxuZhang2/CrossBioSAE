#!/bin/bash
#SBATCH -J esm2_llr
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -o logs/esm2_llr.out
#SBATCH -e logs/esm2_llr.out
set -e
echo "=== esm2_llr $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_esm2_llr.py
echo "ESM2_LLR_DONE $(date)"
