#!/bin/bash
#SBATCH -J sae_ctrl
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/sae_controls.out
#SBATCH -e logs/sae_controls.out

set -e
echo "=== sae_controls $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/sae_interpretability_controls.py \
    --n_sae_seeds 5 \
    --expansion 8 \
    --topk 32 \
    --sae_epochs 80

echo "SAE_CONTROLS_DONE $(date)"
