#!/bin/bash
#SBATCH -J mlp_v5b
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/mlp_v5b.out
#SBATCH -e logs/mlp_v5b.out

set -e
echo "=== mlp_v5b $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

# 10 seeds + 100 pretrain epochs + different base_seed for diversity
python scripts/train_variant_mlp_v5.py \
    --n_seeds 10 \
    --pretrain_epochs 100 \
    --base_seed 100 \
    --out_dir results/variant_mlp_v5b

echo "MLP_V5B_DONE $(date)"
