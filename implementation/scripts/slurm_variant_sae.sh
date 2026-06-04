#!/bin/bash
#SBATCH -J var_sae
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/variant_sae.out
#SBATCH -e logs/variant_sae.out

set -e
echo "=== variant_sae $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/train_variant_sae.py \
    --expansion 8 \
    --topk 32 \
    --sae_epochs 80

echo "VARIANT_SAE_DONE $(date)"
