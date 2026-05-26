#!/bin/bash
#SBATCH --job-name=cbsae_extract_prot
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract_protein_%j.out
#SBATCH --error=logs/extract_protein_%j.err

# CrossBioSAE: Extract protein activations from ESM-2
# Usage: sbatch scripts/slurm_extract_protein.sh [pilot|full]

set -euo pipefail

CONFIG=${1:-"pilot"}
IMPL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "CrossBioSAE: Protein Activation Extraction"
echo "Config: ${CONFIG}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Activate conda environment
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

# Create log directory
mkdir -p "${IMPL_DIR}/logs"

cd "${IMPL_DIR}"

python scripts/extract_activations.py \
    --config "configs/${CONFIG}.yaml" \
    --modality protein \
    --device cuda

echo "Protein activation extraction complete: $(date)"
