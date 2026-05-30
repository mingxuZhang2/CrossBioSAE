#!/bin/bash
#SBATCH -J vfuse_ft
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/finetune_variant.out
#SBATCH -e logs/finetune_variant.out
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/finetune_variant.py --device cuda
echo "FINETUNE_DONE_$? $(date)"
