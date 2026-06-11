#!/bin/bash
#SBATCH -J brca2_sum
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/brca2_replication_summary.out
#SBATCH -e logs/brca2_replication_summary.out
set -euo pipefail

echo "=== BRCA2 replication summary $(hostname) $(date) ==="
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/summarize_brca2_replication.py --repo-root .

echo "BRCA2_REPLICATION_SUMMARY_DONE $(date)"
