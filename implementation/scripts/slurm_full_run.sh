#!/bin/bash
#SBATCH --job-name=cbsae_full
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/full_run_%j.out
#SBATCH --error=logs/full_run_%j.err

# CrossBioSAE: Full-scale run with 20k genes
# ESM-2 650M (protein) + Nucleotide Transformer v2 500M (DNA)
# Includes all 3 downstream tasks

exec 2>&1

IMPL_DIR="/data/user/mzhang630/data/bioinfo/implementation"
CONFIG="full"

echo "============================================"
echo "CrossBioSAE: FULL-SCALE Run"
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

# Step 1: Check data
if [ ! -f "data/full/gene_pairs_full.json" ]; then
    echo "ERROR: data/full/gene_pairs_full.json not found."
    exit 1
fi
N_GENES=$(python -c "import json; print(len(json.load(open('data/full/gene_pairs_full.json'))))" 2>/dev/null)
echo "=== Step 1: Data ready (${N_GENES} genes) ==="

# Step 2a: Extract protein activations (skip if exists)
if [ ! -f "data/full/protein_activations.h5" ]; then
    echo "=== Step 2a: Extract protein activations (ESM-2 650M) ==="
    python scripts/extract_activations.py --config "configs/${CONFIG}.yaml" --modality protein --device cuda
    echo "Protein extraction done: $(date)"
else
    echo "=== Step 2a: Skipped (protein activations exist) ==="
fi

# Step 2b: Extract DNA activations (skip if exists)
if [ ! -f "data/full/dna_activations.h5" ]; then
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

# Step 4: Pilot-style evaluation
echo "=== Step 4: Evaluation ==="
CKPT="checkpoints/full/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then
    CKPT="checkpoints/full/checkpoint_final.pt"
fi
python scripts/run_pilot_eval.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --device cuda
echo "Evaluation done: $(date)"

# Step 5: All downstream tasks
echo "=== Step 5a: Anomaly detection ==="
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --task anomaly \
    --device cuda
echo "Anomaly detection done: $(date)"

echo "=== Step 5b: Variant prediction ==="
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --task variant \
    --device cuda
echo "Variant prediction done: $(date)"

echo "=== Step 5c: Function prediction ==="
python scripts/run_downstream.py \
    --config "configs/${CONFIG}.yaml" \
    --checkpoint "${CKPT}" \
    --task function \
    --device cuda
echo "Function prediction done: $(date)"

echo "============================================"
echo "FULL-SCALE run complete: $(date)"
echo "Results in: results/${CONFIG}/"
echo "============================================"
