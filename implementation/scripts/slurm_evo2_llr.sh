#!/bin/bash
#SBATCH -J evo2llr
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/evo2_llr_%a.out
#SBATCH -e logs/evo2_llr_%a.out
#SBATCH --array=0-15

set -e
echo "=== evo2_llr shard=${SLURM_ARRAY_TASK_ID} $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
export PYTHONUNBUFFERED=1
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation

python scripts/extract_evo2_llr.py \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --nshards 16 \
    --batch 6

echo "EVO2_LLR_DONE shard=${SLURM_ARRAY_TASK_ID} $(date)"
