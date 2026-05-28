#!/bin/bash
#SBATCH --job-name=brca1_fus
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/brca1_fusion_%j.out
#SBATCH --error=logs/brca1_fusion_%j.err

exec 2>&1
PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs results/variant

export PATH="/usr/bin:/bin:/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== BRCA1 Fusion (NT plumbing) ==="
echo "Date: $(date)"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
set -e

python -u scripts/run_brca1_fusion.py \
    --variants data/variant/brca1/brca1_variants.csv \
    --protein data/variant/brca1/brca1_P38398.fasta \
    --chr17 data/variant/brca1/chr17_GRCh37.fna.gz \
    --output_dir results/variant \
    --device cuda

echo "=== Done: $(date) ==="
