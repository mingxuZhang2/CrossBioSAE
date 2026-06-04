#!/bin/bash
#SBATCH -J final_mlp
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/final_pipeline.out
#SBATCH -e logs/final_pipeline.out
# Step 1: extract ESM-2 LLR, Step 2: train final MLP with edelta + LLR
set -e
echo "=== final_pipeline $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

echo "--- Step 1: ESM-2 LLR extraction ---"
python scripts/extract_esm2_llr.py

echo "--- Step 2: Final MLP training ---"
python scripts/train_variant_mlp_final.py

echo "FINAL_PIPELINE_DONE $(date)"
