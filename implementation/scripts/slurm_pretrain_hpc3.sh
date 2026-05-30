#!/bin/bash
#SBATCH -J xmodal_pt
#SBATCH -p acd_u
#SBATCH -A d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/pretrain_crossmodal.out
#SBATCH -e logs/pretrain_crossmodal.out
# Cross-modal contrastive pretraining on HPC3 H100.
# Full 16k gene dataset as one batch (fits easily in H100 80GB).
set -e
echo "=== host $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake
cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/pretrain logs

python scripts/pretrain_crossmodal.py \
  --prot_h5 data/full/protein_activations.h5 \
  --dna_h5 data/full/dna_activations.h5 \
  --output_dir results/pretrain \
  --d_hidden 512 --d_shared 256 --n_layers 2 \
  --batch_size 0 \
  --epochs 500 --patience 50 \
  --lr 3e-4 --wd 1e-4 \
  --temperature 0.07 \
  --device cuda

echo "PRETRAIN_DONE_$? $(date)"
