#!/bin/bash
#SBATCH -J gene_evo2
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH --array=0-7
#SBATCH -o logs/gene_evo2_%a.out
#SBATCH -e logs/gene_evo2_%a.out
# Extract Evo2-7b 4096-d gene-level embeddings for contrastive pretraining.
# 8 shards, CDS median ~1107bp so batch=16 should fit.
set -e
echo "=== host $(hostname) shard ${SLURM_ARRAY_TASK_ID}/8 $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/extract_gene_evo2.py \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 8 \
  --batch 16 \
  --local_path models/evo2_7b.pt \
  --output data/full/evo2_activations.h5
echo "GENE_EVO2_SHARD${SLURM_ARRAY_TASK_ID}_DONE_$? $(date)"
