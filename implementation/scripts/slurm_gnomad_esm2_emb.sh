#!/bin/bash
#SBATCH -J esm_emb
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH --array=0-7
#SBATCH -o logs/esm2_emb_%a.out
#SBATCH -e logs/esm2_emb_%a.out
# ESM-2 650M embedding extraction for 500k variants (protein side)
set -e
echo "=== ESM-2 emb $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_esm2_emb.py \
  --parquet data/variant/gnomad_sample/gnomad_500k.parquet \
  --variant_summary data/variant/variant_summary.txt.gz \
  --output_dir results/gnomad_emb \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 32
echo "ESM2_EMB_SHARD${SLURM_ARRAY_TASK_ID}_DONE $(date)"
