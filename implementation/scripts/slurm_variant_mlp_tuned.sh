#!/bin/bash
#SBATCH -J var_mlp2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/variant_mlp_tuned.out
#SBATCH -e logs/variant_mlp_tuned.out
set -e
echo "=== var_mlp2 $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/train_variant_mlp_tuned.py
echo "VAR_MLP2_DONE $(date)"
