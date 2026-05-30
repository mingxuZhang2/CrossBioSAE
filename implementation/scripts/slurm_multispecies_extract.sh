#!/bin/bash
#SBATCH -J ms_esm
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/multispecies_esm.out
#SBATCH -e logs/multispecies_esm.out
# Extract ESM-2 embeddings for mouse/zebrafish/rat genes (no internet needed).
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/prepare_multispecies.py --device cuda --esm_batch 16
echo "MS_ESM_DONE_$? $(date)"
