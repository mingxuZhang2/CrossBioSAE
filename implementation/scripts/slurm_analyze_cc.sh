#!/bin/bash
#SBATCH --job-name=cc_anal
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=results/crosscoder_sae/analysis/slurm_%j.log

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/crosscoder_sae/analysis

python scripts/analyze_crosscoder.py \
    --checkpoint results/crosscoder_sae/crosscoder.pt \
    --emb_dir results/dms_embeddings \
    --out_dir results/crosscoder_sae/analysis \
    --clinvar_esm2 results/clinvar_esm2 \
    --clinvar_evo2 results/variant \
    --clinvar_csv data/full/clinvar_variants.csv \
    --clinvar_parquet data/variant/clinvar.parquet
