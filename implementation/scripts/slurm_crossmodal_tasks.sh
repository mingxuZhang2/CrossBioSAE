#!/bin/bash
#SBATCH --job-name=xmodal
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/crossmodal_%j.out
#SBATCH --error=logs/crossmodal_%j.err

exec 2>&1

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/usr/bin:/bin:/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake

echo "=== Cross-Modal Tasks ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

set -e

python scripts/run_crossmodal_tasks.py \
    --protein_h5 data/full/protein_activations.h5 \
    --dna_h5 data/full/dna_activations.h5 \
    --labels_dir data/full/labels \
    --output_dir results/crossmodal_tasks \
    --tasks loeuf \
    --pca_dim 256 \
    --cca_dim 64

echo "=== Done: $(date) ==="
