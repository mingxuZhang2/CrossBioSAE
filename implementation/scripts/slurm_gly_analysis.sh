#!/bin/bash
#SBATCH -J gly_fine
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/gly_fine_grained.out
#SBATCH -e logs/gly_fine_grained.out
set -e
echo "=== Gly fine-grained analysis $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/analyze_gly_fine_grained.py
echo "GLY_ANALYSIS_DONE $(date)"
