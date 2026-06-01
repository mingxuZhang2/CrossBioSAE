#!/bin/bash
#SBATCH -J pt_evo2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o logs/pretrain_evo2_%a.out
#SBATCH -e logs/pretrain_evo2_%a.out
# Evo2-7b embeddings for 500k missense variants (cross-modal SAE pretraining)
set -e
echo "=== Evo2 pretrain emb $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_clinvar_evo2_emb.py \
  --parquet data/variant/sae_pretrain/missense_500k.parquet \
  --output_dir results/sae_pretrain_emb \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 4 \
  --genome data/variant/GRCh38.fa.gz \
  --local_path models/evo2_7b.pt
echo "PT_EVO2_SHARD${SLURM_ARRAY_TASK_ID}_DONE $(date)"
