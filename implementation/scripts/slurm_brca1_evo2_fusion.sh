#!/bin/bash
#SBATCH -J brca1_fuse
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/brca1_evo2_fusion.out
#SBATCH -e logs/brca1_evo2_fusion.out
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
PYTHONPATH=scripts python scripts/run_brca1_evo2_fusion.py
echo "BRCA1_FUSE_DONE_$?  $(date)"
