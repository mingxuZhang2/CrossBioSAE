#!/bin/bash
#SBATCH --job-name=cc_seed
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=results/crosscoder_seed_stability/slurm_%j.log

# Seed stability: train 3 CrossCoder seeds (0, 1, 2), compare PP/DP/SH counts,
# DMS Spearman, ClinVar AUROC, feature reproducibility.
# Seed 42 is the original model — we add seeds 0, 1, 2.

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation

COMMON_ARGS="--emb_dir results/dms_embeddings \
    --clinvar_esm2 results/clinvar_esm2 \
    --clinvar_evo2 results/variant \
    --clinvar_csv data/full/clinvar_variants.csv \
    --clinvar_parquet data/variant/clinvar.parquet"

for SEED in 0 1 2; do
    OUT="results/crosscoder_seed_stability/seed_${SEED}"
    mkdir -p "$OUT"
    echo "=== Training seed $SEED ==="

    python scripts/crosscoder_sae.py \
        --phase all \
        --seed $SEED \
        --out_dir "$OUT" \
        $COMMON_ARGS

    echo "=== Analyzing seed $SEED ==="
    mkdir -p "${OUT}/analysis"
    python scripts/analyze_crosscoder.py \
        --checkpoint "${OUT}/crosscoder.pt" \
        --emb_dir results/dms_embeddings \
        --out_dir "${OUT}/analysis" \
        --clinvar_esm2 results/clinvar_esm2 \
        --clinvar_evo2 results/variant \
        --clinvar_csv data/full/clinvar_variants.csv \
        --clinvar_parquet data/variant/clinvar.parquet

    echo "=== Seed $SEED done ==="
done

echo "--- All 3 seeds complete ---"
echo "Compare: results/crosscoder_seed_stability/seed_*/analysis/analysis_summary.json"
