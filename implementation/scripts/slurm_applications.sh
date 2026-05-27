#!/bin/bash
#SBATCH --job-name=cbsae_apps
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/applications_%j.out
#SBATCH --error=logs/applications_%j.err

exec 2>&1

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake

echo "=== CrossBioSAE Applications ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(which python)"

set -e

python scripts/run_applications.py \
    --config configs/independent.yaml \
    --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
    --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt \
    --go_annotations data/full/go_annotations.json \
    --apps 1 2 3 4 \
    --device cuda

echo "=== Done: $(date) ==="
