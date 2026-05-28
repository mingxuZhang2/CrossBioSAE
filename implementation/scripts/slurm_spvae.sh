#!/bin/bash
#SBATCH --job-name=spvae
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/spvae_%j.out
#SBATCH --error=logs/spvae_%j.err

exec 2>&1

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/usr/bin:/bin:/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake

echo "=== SP-VAE Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

set -e

python scripts/train_spvae.py \
    --protein_h5 data/full/protein_activations.h5 \
    --dna_h5 data/full/dna_activations.h5 \
    --labels_dir data/full/labels \
    --output_dir results/spvae \
    --shared_dim 128 \
    --private_dim 64 \
    --hidden_dim 512 \
    --n_epochs 200 \
    --batch_size 512 \
    --lr 1e-3 \
    --alpha_align 10.0 \
    --device cuda

echo "=== Done: $(date) ==="
