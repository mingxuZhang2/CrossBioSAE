#!/bin/bash
#SBATCH -J xm_sae
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/train_crossmodal_sae.out
#SBATCH -e logs/train_crossmodal_sae.out
# Train TRUE cross-modal TopK SAE on z_concat=[z_prot|z_dna] (512-d)
set -e
echo "=== xm_sae $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/train_crossmodal_sae.py \
  --emb_dir results/sae_pretrain_emb \
  --clip_ckpt results/pretrain/crossmodal_clip_40k.pt \
  --out_dir results/sae_crossmodal \
  --d_hidden 4096 --k 32 \
  --epochs 200 --batch 4096 --lr 1e-3
echo "XM_SAE_DONE $(date)"
