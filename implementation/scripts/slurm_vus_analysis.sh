#!/bin/bash
#SBATCH -J vus_mech
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/vus_analysis.out
#SBATCH -e logs/vus_analysis.out
# VUS mechanism profiling: SAE training + activation analysis
set -e
echo "=== VUS mechanism analysis $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/analyze_vus_mechanism.py
echo "VUS_ANALYSIS_DONE $(date)"
