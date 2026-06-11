#!/bin/bash
#SBATCH -J brca2_sae
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/brca2_native_sae.out
#SBATCH -e logs/brca2_native_sae.out
set -euo pipefail

echo "=== BRCA2 native SAE $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/run_brca1_native_finetuned_sae_intervention.py \
  --repo-root . \
  --variants data/variant/brca2/brca2_variants.csv \
  --esm results/variant/brca2_esm_delta.npz \
  --evo2 results/variant/brca2_evo2.npz \
  --fold-artifacts results/brca2_gate_analysis/fold_artifacts \
  --output-dir results/interpretability_applications \
  --run-name BRCA2 \
  --output-prefix brca2_native_finetuned_sae_intervention \
  --domain-col brca2_domain \
  --device cuda \
  --epochs 800 \
  --patience 80 \
  --random-sets 100 \
  --permutation-sets 20 \
  --sae-hidden 1024 \
  --sae-k 32 \
  --top-k-features 32 \
  --dose-k-values 8,16,32,64

echo "BRCA2_NATIVE_SAE_DONE $(date)"
