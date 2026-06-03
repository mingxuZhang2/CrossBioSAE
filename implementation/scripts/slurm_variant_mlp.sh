#!/bin/bash
#SBATCH -J var_mlp
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/variant_mlp.out
#SBATCH -e logs/variant_mlp.out
# End-to-end MLP on raw edelta: prot-only, dna-only, fusion
set -e
echo "=== var_mlp $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/train_variant_mlp.py \
  --epochs 50 --batch 2048 --lr 1e-3 \
  --d_proj 512 --d_hidden 256 --dropout 0.3
echo "VAR_MLP_DONE $(date)"
