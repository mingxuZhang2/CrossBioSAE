#!/bin/bash
#SBATCH --job-name=cc_shuf
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=results/crosscoder_sae_shuffled/slurm_%j.log

# Control B: train CrossCoder on shuffled ESM-Evo pairings.
# Compare PP/DP/SH counts, DMS ablation, ClinVar AUROC with real-pair model.

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/crosscoder_sae_shuffled

python scripts/crosscoder_sae.py \
    --phase all \
    --shuffle_pairs \
    --seed 42 \
    --out_dir results/crosscoder_sae_shuffled \
    --emb_dir results/dms_embeddings \
    --clinvar_esm2 results/clinvar_esm2 \
    --clinvar_evo2 results/variant \
    --clinvar_csv data/full/clinvar_variants.csv \
    --clinvar_parquet data/variant/clinvar.parquet

echo "--- Control B done. Now run analysis on shuffled model ---"

python scripts/analyze_crosscoder.py \
    --checkpoint results/crosscoder_sae_shuffled/crosscoder.pt \
    --emb_dir results/dms_embeddings \
    --out_dir results/crosscoder_sae_shuffled/analysis \
    --clinvar_esm2 results/clinvar_esm2 \
    --clinvar_evo2 results/variant \
    --clinvar_csv data/full/clinvar_variants.csv \
    --clinvar_parquet data/variant/clinvar.parquet
