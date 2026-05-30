#!/bin/bash
#SBATCH -J cv_emb
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH --array=0-7
#SBATCH -o logs/clinvar_evo2_emb_%a.out
#SBATCH -e logs/clinvar_evo2_emb_%a.out
# Evo2 4096-d embedding deltas for ClinVar genome-wide (for SAE).
# batch=2 for 8192bp embedding (heavier than scoring).
set -e
echo "=== host $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_clinvar_evo2_emb.py \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 2 \
  --genome data/variant/GRCh38.fa.gz \
  --local_path models/evo2_7b.pt
echo "CV_EMB_SHARD${SLURM_ARRAY_TASK_ID}_DONE_$? $(date)"
