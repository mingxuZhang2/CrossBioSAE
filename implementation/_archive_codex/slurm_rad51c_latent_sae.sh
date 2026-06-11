#!/bin/bash
#SBATCH -J rad51c_latent_sae
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/rad51c_latent_sae_%j.out
#SBATCH -e logs/rad51c_latent_sae_%j.out

set -euo pipefail
echo "=== RAD51C latent SAE $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/interpretability_applications

python -u scripts/run_third_gene_latent_sae_intervention.py \
  --gene-name RAD51C \
  --variant-scores results/interpretability_applications/rad51c_llr_esm_checkpoint_variant_scores.csv \
  --esm results/variant/rad51c_esm_delta.npz \
  --llr results/variant/rad51c_evo2_llr.npz \
  --output-dir results/interpretability_applications \
  --output-prefix rad51c_latent_sae_intervention \
  --group-col pos_hg38 \
  --domain-col domains \
  --device cuda \
  --epochs 400 \
  --patience 50 \
  --random-sets 100 \
  --permutation-sets 20 \
  --sae-hidden 1024 \
  --sae-k 32 \
  --top-k-features 32 \
  --dose-k-values 8,16,32,64

echo "RAD51C_LATENT_SAE_DONE $(date)"
