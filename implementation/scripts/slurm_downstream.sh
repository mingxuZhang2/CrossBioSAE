#!/bin/bash
#SBATCH --job-name=cbsae_downstream
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/downstream_%j.out
#SBATCH --error=logs/downstream_%j.err

# CrossBioSAE: Run all downstream tasks using trained model
# Requires: checkpoints/full/checkpoint_best.pt, data/full/*.h5

exec 2>&1

IMPL_DIR="/data/user/mzhang630/data/bioinfo/implementation"
CONFIG="full"

echo "============================================"
echo "CrossBioSAE: Downstream Tasks"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

export PATH="/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake
echo "Python: $(which python)"

set -e
cd "${IMPL_DIR}"
mkdir -p logs

CKPT="checkpoints/${CONFIG}/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then
    CKPT="checkpoints/${CONFIG}/checkpoint_final.pt"
fi
echo "Checkpoint: ${CKPT}"

# Task 1: Variant prediction
echo "============================================"
echo "=== Task 1: Variant Effect Prediction ==="
echo "============================================"
python scripts/prepare_variant_data.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --clinvar "data/full/clinvar_variants.csv" \
    --device cuda \
    --max_variants 5000
echo "Task 1 done: $(date)"

# Task 2: Function prediction (with GO annotations)
echo "============================================"
echo "=== Task 2: Function Prediction ==="
echo "============================================"
if [ -f "data/full/go_annotations.json" ]; then
    python scripts/run_downstream.py \
        --config "configs/${CONFIG}.yaml" \
        --checkpoint "${CKPT}" \
        --task function \
        --device cuda
    echo "Task 2 done: $(date)"
else
    echo "Skipping Task 2: GO annotations not found"
fi

# Task 3: Anomaly detection (re-run for completeness)
echo "============================================"
echo "=== Task 3: Anomaly Detection ==="
echo "============================================"
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --task anomaly \
    --device cuda
echo "Task 3 done: $(date)"

echo "============================================"
echo "All downstream tasks complete: $(date)"
echo "Results in: results/${CONFIG}/"
echo "============================================"
