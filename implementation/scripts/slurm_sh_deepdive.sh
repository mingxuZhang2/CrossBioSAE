#!/bin/bash
#SBATCH --job-name=sh_deepdive
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/crosscoder_sae/sh_deepdive/slurm_%j.log

set -e
cd /data/user/mzhang630/data/bioinfo/implementation
source activate sake

mkdir -p results/crosscoder_sae/sh_deepdive

python scripts/analyze_sh_deepdive.py \
  --checkpoint results/crosscoder_sae/crosscoder.pt \
  --out_dir results/crosscoder_sae/sh_deepdive \
  --cat_file results/crosscoder_sae/analysis_bhfix/feature_categories.npz \
  --clinvar_fdr_csv results/crosscoder_sae/analysis_bhfix/clinvar_feature_pathogenicity_fdr.csv

echo "=== SH Deep Dive COMPLETE ==="
