#!/bin/bash
#SBATCH -J mo_valid
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/validate_multiomics.out
#SBATCH -e logs/validate_multiomics.out
# Genome-wide multi-omics fusion validation
set -e
echo "=== mo_valid $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/validate_multiomics_fusion.py
echo "MO_VALID_DONE $(date)"
