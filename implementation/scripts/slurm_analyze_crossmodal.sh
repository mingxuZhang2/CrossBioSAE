#!/bin/bash
#SBATCH -J xm_interp
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/analyze_crossmodal_sae.out
#SBATCH -e logs/analyze_crossmodal_sae.out
# Cross-modal SAE interpretability analysis
set -e
echo "=== xm_interp $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/analyze_crossmodal_sae.py
echo "XM_INTERP_DONE $(date)"
