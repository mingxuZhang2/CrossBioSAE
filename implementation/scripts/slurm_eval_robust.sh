#!/bin/bash
#SBATCH -J eval_rob
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/eval_robust.out
#SBATCH -e logs/eval_robust.out

set -e
echo "=== eval_robust $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/eval_robust.py --n_seeds 5

echo "EVAL_ROBUST_DONE $(date)"
