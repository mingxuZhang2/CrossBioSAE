#!/bin/bash
#SBATCH --job-name=bench_%j
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

exec 2>&1

EMB_NAME=$1
if [ -z "$EMB_NAME" ]; then
    echo "Usage: sbatch slurm_bench_single.sh <EMBEDDING_NAME>"
    exit 1
fi

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/usr/bin:/bin:/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake

echo "=== Benchmark: $EMB_NAME ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

set -e

# Gene-level: GO + OMIM
python scripts/run_gene_benchmark.py \
    --embedding_dir results/benchmark_embeddings \
    --benchmark_dir gene-embedding-benchmarks \
    --output_dir results/benchmark_results \
    --tasks go omim \
    --embeddings "$EMB_NAME"

# Gene-pair: NG, SL, TF (concat + sum)
for OP in concat sum; do
    python scripts/run_gene_benchmark.py \
        --embedding_dir results/benchmark_embeddings \
        --benchmark_dir gene-embedding-benchmarks \
        --output_dir results/benchmark_results \
        --tasks ng sl tf \
        --pair_op "$OP" \
        --embeddings "$EMB_NAME"
done

echo "=== Done $EMB_NAME: $(date) ==="
