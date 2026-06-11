#!/bin/bash
#SBATCH -J rad51c_checkpoint
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/rad51c_checkpoint_%j.out
#SBATCH -e logs/rad51c_checkpoint_%j.out

set -euo pipefail
echo "=== host $(hostname)  $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/interpretability_applications

exec 9>logs/rad51c_checkpoint.lock
if ! flock -n 9; then
  echo "Another RAD51C checkpoint job is already running; exiting. $(date)"
  exit 0
fi

if [[ -s results/interpretability_applications/rad51c_llr_esm_checkpoint_metric_summary.csv ]]; then
  echo "RAD51C checkpoint output already exists; exiting. $(date)"
  exit 0
fi

for required in results/variant/rad51c_esm_delta.npz results/variant/rad51c_evo2_llr.npz; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required checkpoint input: $required  $(date)" >&2
    exit 2
  fi
done

python -u scripts/run_brca2_llr_esm_checkpoint.py \
  --variants data/variant/rad51c/rad51c_grch38_variants.csv \
  --esm results/variant/rad51c_esm_delta.npz \
  --evo2-llr results/variant/rad51c_evo2_llr.npz \
  --output-dir results/interpretability_applications \
  --output-prefix rad51c_llr_esm_checkpoint \
  --group-col pos \
  --bootstrap-sets 2000

python -u scripts/analyze_llr_esm_checkpoint_discordance.py \
  --checkpoint-scores results/interpretability_applications/rad51c_llr_esm_checkpoint_variant_scores.csv \
  --output-dir results/interpretability_applications \
  --output-prefix rad51c_llr_esm_discordance \
  --gene-name RAD51C \
  --domain-col domains \
  --quantile 0.20

echo "RAD51C_CHECKPOINT_DONE_$?  $(date)"
