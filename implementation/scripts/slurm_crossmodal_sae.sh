#!/bin/bash
#SBATCH --job-name=xm_sae
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/crossmodal_sae/slurm_%j.log

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/crossmodal_sae

python scripts/pretrain_crossmodal_sae.py \
    --phase all \
    --emb_dir results/dms_embeddings \
    --out_dir results/crossmodal_sae \
    --pca_dim 512 \
    --expansion 4 \
    --k 32 \
    --proj_dim 256 \
    --cross_k 16 \
    --epochs 300 \
    --batch 8192 \
    --lr 5e-4 \
    --lambda_xpred 0.3 \
    --lambda_align 0.03
