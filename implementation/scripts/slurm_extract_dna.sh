#!/bin/bash
#SBATCH --job-name=cbsae_extract_dna
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/extract_dna_%j.out
#SBATCH --error=logs/extract_dna_%j.err

# CrossBioSAE: Extract DNA activations from Evo-2
# Needs more memory than protein extraction (7B model)
# Usage: sbatch scripts/slurm_extract_dna.sh [pilot|full]

set -euo pipefail

CONFIG=${1:-"pilot"}
IMPL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "CrossBioSAE: DNA Activation Extraction"
echo "Config: ${CONFIG}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

mkdir -p "${IMPL_DIR}/logs"

cd "${IMPL_DIR}"

python scripts/extract_activations.py \
    --config "configs/${CONFIG}.yaml" \
    --modality dna \
    --device cuda

echo "DNA activation extraction complete: $(date)"
