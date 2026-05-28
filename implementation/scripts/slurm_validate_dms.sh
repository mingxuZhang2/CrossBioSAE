#!/bin/bash
#SBATCH --job-name=val_dms
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validate_dms_%j.out
#SBATCH --error=logs/validate_dms_%j.err

exec 2>&1

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/usr/bin:/bin:/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake
export PYTHONUNBUFFERED=1

echo "=== Validate DMS Signal ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

set -e

python -u scripts/validate_dms_signal.py \
    --nt_file data/dms/nucleotide_genotypes_to_brightness.tsv \
    --ref_file data/dms/avGFP_reference.fa \
    --output_dir results/dms_fitness \
    --device cuda

echo "=== Done: $(date) ==="
