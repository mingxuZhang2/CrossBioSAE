#!/bin/bash
#SBATCH --job-name=cbsae_real
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/real_pilot_%j.out
#SBATCH --error=logs/real_pilot_%j.err

# CrossBioSAE: Real pilot with ESM-2 (protein) + Nucleotide Transformer (DNA)

# Redirect all stderr to stdout so we can see errors in one log
exec 2>&1

IMPL_DIR="/data/user/mzhang630/data/bioinfo/implementation"
CONFIG="pilot"

echo "============================================"
echo "CrossBioSAE: REAL Pilot Pipeline"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Conda setup (must happen before set -e)
export PATH="/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

set -e

cd "${IMPL_DIR}"
mkdir -p logs

# Clean SAE training outputs but keep extracted activations
rm -rf checkpoints/pilot results/pilot

# Step 1: Prepare data (skip if exists)
if [ ! -f "data/pilot/gene_pairs_pilot.json" ]; then
    echo "=== Step 1: Prepare data ==="
    python scripts/prepare_data.py --config "configs/${CONFIG}.yaml" --synthetic
    echo "Data preparation done: $(date)"
else
    echo "=== Step 1: Skipped (data exists) ==="
fi

# Step 2a: Extract protein activations (skip if exists)
if [ ! -f "data/pilot/protein_activations.h5" ]; then
    echo "=== Step 2a: Extract protein activations (ESM-2 650M) ==="
    python scripts/extract_activations.py --config "configs/${CONFIG}.yaml" --modality protein --device cuda
    echo "Protein extraction done: $(date)"
else
    echo "=== Step 2a: Skipped (protein activations exist) ==="
fi

# Step 2b: Extract DNA activations (skip if exists)
if [ ! -f "data/pilot/dna_activations.h5" ]; then
    echo "=== Step 2b: Extract DNA activations (NT v2 500M) ==="
    python scripts/extract_activations.py --config "configs/${CONFIG}.yaml" --modality dna --device cuda
    echo "DNA extraction done: $(date)"
else
    echo "=== Step 2b: Skipped (DNA activations exist) ==="
fi

# Step 3: Train CrossBioSAE
echo "=== Step 3: Train CrossBioSAE ==="
python scripts/train_sae.py --config "configs/${CONFIG}.yaml" --device cuda
echo "Training done: $(date)"

# Step 4: Pilot evaluation
echo "=== Step 4: Pilot evaluation ==="
CKPT="checkpoints/${CONFIG}/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then
    CKPT="checkpoints/${CONFIG}/checkpoint_final.pt"
fi
python scripts/run_pilot_eval.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --device cuda
echo "Evaluation done: $(date)"

# Step 5: Anomaly detection
echo "=== Step 5: Anomaly detection ==="
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --task anomaly \
    --device cuda
echo "Anomaly detection done: $(date)"

echo "============================================"
echo "REAL pilot pipeline complete: $(date)"
echo "Results in: results/${CONFIG}/"
echo "============================================"
