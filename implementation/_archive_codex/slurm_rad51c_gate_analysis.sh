#!/bin/bash
#SBATCH -J rad51c_gate
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/rad51c_gate_%j.out
#SBATCH -e logs/rad51c_gate_%j.out

set -euo pipefail
echo "=== RAD51C gate analysis $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
mkdir -p logs results/rad51c_gate_analysis

if [[ ! -s results/variant/rad51c_evo2.npz ]]; then
  echo "Missing results/variant/rad51c_evo2.npz; submit scripts/slurm_rad51c_evo2_full.sh first." >&2
  exit 2
fi

python -u scripts/finetune_gate_analysis.py \
  --variants data/variant/rad51c/rad51c_grch38_variants.csv \
  --prot_delta results/variant/rad51c_esm_delta.npz \
  --dna_delta results/variant/rad51c_evo2.npz \
  --checkpoint results/pretrain/crossmodal_clip_40k.pt \
  --output_dir results/rad51c_gate_analysis \
  --group_col pos_hg38 \
  --run_name RAD51C \
  --output_prefix rad51c \
  --device cuda

echo "RAD51C_GATE_DONE $(date)"
