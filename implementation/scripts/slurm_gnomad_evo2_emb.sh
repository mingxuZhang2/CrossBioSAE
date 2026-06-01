#!/bin/bash
#SBATCH -J gn_emb
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o logs/gnomad_emb_%a.out
#SBATCH -e logs/gnomad_emb_%a.out
# Evo2-7b embedding extraction for 500k variants (8 independent single-GPU jobs)
set -e
echo "=== gnomAD emb $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_clinvar_evo2_emb.py \
  --parquet data/variant/gnomad_sample/gnomad_500k.parquet \
  --output_dir results/gnomad_emb \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 4 \
  --genome data/variant/GRCh38.fa.gz \
  --local_path models/evo2_7b.pt
echo "GNOMAD_EMB_SHARD${SLURM_ARRAY_TASK_ID}_DONE $(date)"
