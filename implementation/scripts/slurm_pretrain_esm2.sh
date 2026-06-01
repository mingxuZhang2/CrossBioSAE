#!/bin/bash
#SBATCH -J pt_esm2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 06:00:00
#SBATCH --array=0-7
#SBATCH -o logs/pretrain_esm2_%a.out
#SBATCH -e logs/pretrain_esm2_%a.out
# ESM-2 650M embeddings for 500k missense variants (cross-modal SAE pretraining)
set -e
echo "=== ESM-2 pretrain emb $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_esm2_pretrain.py \
  --annotated_csv data/variant/sae_pretrain/missense_500k_annotated.csv \
  --gene_pairs data/full/gene_pairs_full.json \
  --output_dir results/sae_pretrain_emb \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 16
echo "PT_ESM2_SHARD${SLURM_ARRAY_TASK_ID}_DONE $(date)"
