#!/bin/bash
#SBATCH -J ms_evo2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -o logs/multispecies_evo2.out
#SBATCH -e logs/multispecies_evo2.out
# Extract Evo2-7b 4096-d gene embeddings for mouse/zebrafish/rat.
# All 3 species in one job (~23k genes, batch=16, CDS short → ~1-2h).
set -e
echo "=== host $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_multispecies_evo2.py --batch 16 --local_path models/evo2_7b.pt
echo "MS_EVO2_DONE_$? $(date)"
