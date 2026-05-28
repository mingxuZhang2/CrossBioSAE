#!/bin/bash
#SBATCH --job-name=cbsae_bench
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

exec 2>&1

PROJ_DIR="/data/user/mzhang630/data/bioinfo/implementation"
cd "$PROJ_DIR"
mkdir -p logs

export PATH="/data/user/mzhang630/miniconda3/bin:$PATH"
eval "$(/data/user/mzhang630/miniconda3/bin/conda shell.bash hook)"
conda activate sake

pip install mygene --quiet 2>/dev/null || true

echo "=== Gene Embedding Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"

# Clone benchmark repo if not present
if [ ! -d "gene-embedding-benchmarks" ]; then
    echo "Cloning gene-embedding-benchmarks repo..."
    git clone https://github.com/ylaboratory/gene-embedding-benchmarks.git
fi

set -e

# Step 1: Prepare embeddings (gene symbol -> Entrez ID mapping, CCA alignment)
echo ""
echo "=== Step 1: Preparing embeddings ==="
python scripts/prepare_benchmark_embeddings.py \
    --protein_h5 data/full/protein_activations.h5 \
    --dna_h5 data/full/dna_activations.h5 \
    --output_dir results/benchmark_embeddings \
    --cca_components 64 \
    --pca_dim 512

# Step 2: Run benchmarks
echo ""
echo "=== Step 2: Running gene-level benchmarks (GO, OMIM) ==="
python scripts/run_gene_benchmark.py \
    --embedding_dir results/benchmark_embeddings \
    --benchmark_dir gene-embedding-benchmarks \
    --output_dir results/benchmark_results \
    --tasks go omim \
    --embeddings ESM2-RAW NT-RAW RAW-CONCAT CCA-PROT CCA-DNA CCA-FUSED CCA-SHARED ESM2-PCA NT-PCA PCA-CONCAT

echo ""
echo "=== Step 3: Running gene-pair benchmarks (NG, SL, TF) ==="
for OP in concat sum; do
    echo "--- Operation: $OP ---"
    python scripts/run_gene_benchmark.py \
        --embedding_dir results/benchmark_embeddings \
        --benchmark_dir gene-embedding-benchmarks \
        --output_dir results/benchmark_results \
        --tasks ng sl tf \
        --pair_op "$OP" \
        --embeddings ESM2-RAW NT-RAW RAW-CONCAT CCA-FUSED CCA-SHARED
done

echo ""
echo "=== Done: $(date) ==="
