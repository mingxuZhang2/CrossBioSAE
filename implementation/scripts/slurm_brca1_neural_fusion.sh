#!/bin/bash
#SBATCH -J nfuse
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/brca1_neural_fusion.out
#SBATCH -e logs/brca1_neural_fusion.out
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/run_brca1_neural_fusion.py --device cuda --use_llr
echo "NEURAL_FUSION_DONE_$?  $(date)"
