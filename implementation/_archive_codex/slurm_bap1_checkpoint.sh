#!/bin/bash
#SBATCH -J bap1_checkpoint
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/bap1_checkpoint_%j.out
#SBATCH -e logs/bap1_checkpoint_%j.out

set -euo pipefail
echo "=== host $(hostname)  $(date) ==="
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/interpretability_applications

exec 9>logs/bap1_checkpoint.lock
if ! flock -n 9; then
  echo "Another BAP1 checkpoint job is already running; exiting. $(date)"
  exit 0
fi

if [[ -s results/interpretability_applications/bap1_llr_esm_checkpoint_metric_summary.csv ]]; then
  echo "BAP1 checkpoint output already exists; exiting. $(date)"
  exit 0
fi

for required in results/variant/bap1_esm_delta.npz results/variant/bap1_evo2_llr.npz; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required checkpoint input: $required  $(date)"
    exit 2
  fi
done

python -u scripts/run_brca2_llr_esm_checkpoint.py \
  --variants data/variant/bap1/bap1_variants.csv \
  --esm results/variant/bap1_esm_delta.npz \
  --evo2-llr results/variant/bap1_evo2_llr.npz \
  --output-dir results/interpretability_applications \
  --output-prefix bap1_llr_esm_checkpoint \
  --group-col pos_hg38 \
  --bootstrap-sets 2000

python -u scripts/analyze_llr_esm_checkpoint_discordance.py \
  --checkpoint-scores results/interpretability_applications/bap1_llr_esm_checkpoint_variant_scores.csv \
  --output-dir results/interpretability_applications \
  --output-prefix bap1_llr_esm_discordance \
  --gene-name BAP1 \
  --domain-col domains \
  --quantile 0.20

echo "BAP1_CHECKPOINT_DONE_$?  $(date)"
