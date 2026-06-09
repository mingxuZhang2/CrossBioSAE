#!/bin/bash
#SBATCH --job-name=cv_esm2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/clinvar_esm2/slurm_%j_%a.log
#SBATCH --array=0-3

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation
mkdir -p results/clinvar_esm2

python scripts/extract_clinvar_esm2.py \
    --clinvar_csv data/full/clinvar_variants.csv \
    --out_dir results/clinvar_esm2 \
    --shard $SLURM_ARRAY_TASK_ID \
    --nshards 4 \
    --batch_size 16
