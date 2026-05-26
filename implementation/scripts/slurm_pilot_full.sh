#!/bin/bash
#SBATCH --job-name=cbsae_pilot
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pilot_full_%j.out
#SBATCH --error=logs/pilot_full_%j.err

# CrossBioSAE: Complete pilot pipeline (all steps)
# Runs: data prep -> activation extraction -> training -> evaluation
# Usage: sbatch scripts/slurm_pilot_full.sh

set -euo pipefail

IMPL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="pilot"

echo "============================================"
echo "CrossBioSAE: Full Pilot Pipeline"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate crossbiosae 2>/dev/null || conda activate base

mkdir -p "${IMPL_DIR}/logs"
cd "${IMPL_DIR}"

# Step 1: Prepare data
echo "=== Step 1: Prepare data ==="
python scripts/prepare_data.py --config "configs/${CONFIG}.yaml" --synthetic
echo "Data preparation done: $(date)"

# Step 2a: Extract protein activations
echo "=== Step 2a: Extract protein activations ==="
python scripts/extract_activations.py --config "configs/${CONFIG}.yaml" --modality protein --device cuda
echo "Protein extraction done: $(date)"

# Step 2b: Extract DNA activations
echo "=== Step 2b: Extract DNA activations ==="
python scripts/extract_activations.py --config "configs/${CONFIG}.yaml" --modality dna --device cuda
echo "DNA extraction done: $(date)"

# Step 3: Train SAE
echo "=== Step 3: Train CrossBioSAE ==="
python scripts/train_sae.py --config "configs/${CONFIG}.yaml" --device cuda
echo "Training done: $(date)"

# Step 4: Pilot evaluation
echo "=== Step 4: Pilot evaluation ==="
python scripts/run_pilot_eval.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "checkpoints/${CONFIG}/checkpoint_best.pt" \
    --device cuda
echo "Evaluation done: $(date)"

# Step 5: Anomaly detection (quick downstream task)
echo "=== Step 5: Anomaly detection ==="
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "checkpoints/${CONFIG}/checkpoint_best.pt" \
    --task anomaly \
    --device cuda
echo "Anomaly detection done: $(date)"

echo "============================================"
echo "Full pilot pipeline complete: $(date)"
echo "Results in: results/${CONFIG}/"
echo "============================================"
