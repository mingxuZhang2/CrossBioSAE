#!/bin/bash
#SBATCH -J dms_bench
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -o logs/dms_benchmark.out
#SBATCH -e logs/dms_benchmark.out
# DMS benchmark: ESM-2 vs Evo2 vs CLIP-fusion on ProteinGym assays
set -e
echo "=== dms_bench $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/benchmark_dms.py \
  --dms_dir data/proteingym/substitutions \
  --clip_ckpt results/pretrain/crossmodal_clip_40k.pt \
  --evo2_path models/evo2_7b.pt \
  --out_dir results/dms_benchmark
echo "DMS_BENCH_DONE $(date)"
