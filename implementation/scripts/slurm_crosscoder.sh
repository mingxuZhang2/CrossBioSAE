#!/bin/bash
#SBATCH --job-name=cc_sae
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/crosscoder_sae/slurm_%j.log

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/crosscoder_sae

python scripts/crosscoder_sae.py \
    --phase all \
    --emb_dir results/dms_embeddings \
    --out_dir results/crosscoder_sae \
    --clinvar_esm2 results/clinvar_esm2 \
    --clinvar_evo2 results/variant \
    --clinvar_csv data/full/clinvar_variants.csv \
    --clinvar_parquet data/variant/clinvar.parquet \
    --d_prot 768 \
    --d_dna 512 \
    --n_features 4096 \
    --k 32 \
    --epochs 300 \
    --batch 8192 \
    --lr 5e-4
