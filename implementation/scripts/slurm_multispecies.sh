#!/bin/bash
#SBATCH -J multisp
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/multispecies.out
#SBATCH -e logs/multispecies.out
# Download mouse/zebrafish/rat gene pairs + extract ESM-2 embeddings.
# Evo2 extraction runs as a separate sharded job after this completes.
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export https_proxy=http://127.0.0.1:42685 http_proxy=http://127.0.0.1:42685
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/prepare_multispecies.py --device cuda --esm_batch 8
echo "MULTISPECIES_DONE_$? $(date)"
