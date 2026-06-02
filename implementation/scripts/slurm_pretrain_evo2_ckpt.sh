#!/bin/bash
#SBATCH -J pt_evo2c
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --array=0-15
#SBATCH -o logs/pretrain_evo2c_%a.out
#SBATCH -e logs/pretrain_evo2c_%a.out
# Evo2-7b embeddings — dual-modality 260k set only, checkpointed/resumable, 16 shards
set -e
echo "=== Evo2c $(hostname) shard ${SLURM_ARRAY_TASK_ID}/16 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_evo2_emb_ckpt.py \
  --parquet data/variant/sae_pretrain/missense_500k.parquet \
  --subset_idx data/variant/sae_pretrain/dual_idx.npy \
  --output_dir results/sae_pretrain_emb \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 16 \
  --batch 6 \
  --genome data/variant/GRCh38.fa.gz \
  --local_path models/evo2_7b.pt
echo "PT_EVO2C_SHARD${SLURM_ARRAY_TASK_ID}_DONE $(date)"
