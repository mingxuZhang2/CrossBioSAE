#!/bin/bash
#SBATCH -J mlp_v3
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/mlp_v3.out
#SBATCH -e logs/mlp_v3.out

set -e
echo "=== mlp_v3 $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/train_variant_mlp_v3.py --n_seeds 3

echo "MLP_V3_DONE $(date)"
