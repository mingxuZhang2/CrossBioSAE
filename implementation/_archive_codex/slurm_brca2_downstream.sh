#!/bin/bash
#SBATCH -J brca2_ds
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/brca2_downstream.out
#SBATCH -e logs/brca2_downstream.out
set -euo pipefail

echo "=== BRCA2 downstream $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python - <<'PY'
import numpy as np
import pandas as pd

variants = pd.read_csv("data/variant/brca2/brca2_variants.csv")
esm = np.load("results/variant/brca2_esm_delta.npz")
evo = np.load("results/variant/brca2_evo2.npz")
assert esm["pdelta"].shape[0] == len(variants)
assert evo["edelta"].shape[0] == len(variants)
assert evo["llr"].shape[0] == len(variants)
print("validated BRCA2 representations", esm["pdelta"].shape, evo["edelta"].shape)
print("finite llr", int(np.isfinite(evo["llr"]).sum()), "of", len(variants))
PY

python scripts/run_brca1_shared_embedding.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --esm results/variant/brca2_esm_delta.npz \
  --evo2 results/variant/brca2_evo2.npz \
  --output_dir results/interpretability_applications \
  --group_col pos_hg38 \
  --run_name BRCA2 \
  --output_prefix brca2

python scripts/finetune_variant.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --prot_delta results/variant/brca2_esm_delta.npz \
  --dna_delta results/variant/brca2_evo2.npz \
  --checkpoint results/pretrain/crossmodal_clip_40k.pt \
  --output_dir results/variant \
  --group_col pos_hg38 \
  --run_name BRCA2 \
  --output_prefix brca2 \
  --device cuda \
  --use_llr

python scripts/finetune_gate_analysis.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --prot_delta results/variant/brca2_esm_delta.npz \
  --dna_delta results/variant/brca2_evo2.npz \
  --checkpoint results/pretrain/crossmodal_clip_40k.pt \
  --output_dir results/brca2_gate_analysis \
  --group_col pos_hg38 \
  --run_name BRCA2 \
  --output_prefix brca2 \
  --device cuda

python scripts/run_sge_mechanism_validation.py \
  --variants data/variant/brca2/brca2_variants.csv \
  --gate_npz results/brca2_gate_analysis/brca2_gate_analysis.npz \
  --output_dir results/interpretability_applications \
  --output_prefix brca2 \
  --run_name BRCA2 \
  --domain_col brca2_domain

echo "BRCA2_DOWNSTREAM_DONE $(date)"
