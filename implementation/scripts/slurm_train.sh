#!/bin/bash
#SBATCH --job-name=cbsae_train
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# CrossBioSAE: Train the cross-modal SAE
# SAE training itself is not huge; 1 GPU is sufficient
# Usage: sbatch scripts/slurm_train.sh [pilot|full] [optional_checkpoint]

set -euo pipefail

CONFIG=${1:-"pilot"}
CHECKPOINT=${2:-""}
IMPL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "CrossBioSAE: SAE Training"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT:-none}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

mkdir -p "${IMPL_DIR}/logs"

cd "${IMPL_DIR}"

RESUME_FLAG=""
if [ -n "${CHECKPOINT}" ]; then
    RESUME_FLAG="--resume ${CHECKPOINT}"
fi

python scripts/train_sae.py \
    --config "configs/${CONFIG}.yaml" \
    --device cuda \
    ${RESUME_FLAG}

echo "Training complete: $(date)"
