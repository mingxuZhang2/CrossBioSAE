#!/bin/bash
#SBATCH -J mlp_v6
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/mlp_v6.out
#SBATCH -e logs/mlp_v6.out

set -e
echo "=== mlp_v6 $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/train_variant_mlp_v6.py \
    --n_seeds 20 \
    --pretrain_epochs 30 \
    --self_train_rounds 2 \
    --pseudo_threshold 0.15

echo "MLP_V6_DONE $(date)"
