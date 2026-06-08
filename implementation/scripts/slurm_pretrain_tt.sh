#!/bin/bash
#SBATCH --job-name=tt_pre
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/two_tower_pretrain/slurm_%j.log

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate sake

cd /data/user/mzhang630/data/bioinfo/implementation

python scripts/pretrain_two_tower.py \
    --phase evaluate \
    --sae_dir results/dms_v6 \
    --out_dir results/two_tower_pretrain \
    --checkpoint results/two_tower_pretrain/pretrained.pt \
    --head_epochs 100 \
    --head_lr 3e-3
