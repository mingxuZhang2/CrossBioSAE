#!/bin/bash
#SBATCH -J rad51c_native_miss
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/rad51c_native_sae_missense_%j.out
#SBATCH -e logs/rad51c_native_sae_missense_%j.out

set -euo pipefail
echo "=== RAD51C missense native SAE $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/interpretability_applications

if [[ ! -d results/rad51c_gate_analysis/fold_artifacts ]]; then
  echo "Missing RAD51C gate fold artifacts; submit scripts/slurm_rad51c_gate_analysis.sh first." >&2
  exit 2
fi

python -u scripts/run_brca1_native_finetuned_sae_intervention.py \
  --repo-root . \
  --variants data/variant/rad51c/rad51c_grch38_variants.csv \
  --esm results/variant/rad51c_esm_delta.npz \
  --evo2 results/variant/rad51c_evo2.npz \
  --fold-artifacts results/rad51c_gate_analysis/fold_artifacts \
  --output-dir results/interpretability_applications \
  --run-name RAD51C_missense \
  --output-prefix rad51c_native_finetuned_sae_intervention_missense \
  --domain-col domains \
  --target-representation z_prot \
  --analysis-filter-col is_missense \
  --analysis-filter-value true \
  --device cuda \
  --epochs 800 \
  --patience 80 \
  --random-sets 200 \
  --permutation-sets 50 \
  --sae-hidden 1024 \
  --sae-k 32 \
  --top-k-features 32 \
  --dose-k-values 8,16,32,64

echo "RAD51C_NATIVE_SAE_MISSENSE_DONE $(date)"
