#!/bin/bash
#SBATCH -J gate_analysis
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -o logs/gate_analysis.out
#SBATCH -e logs/gate_analysis.out
set -e
echo "=== Gate analysis $(hostname) $(date) ==="
module load cuda/12.4
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
python scripts/finetune_gate_analysis.py \
  --checkpoint results/pretrain/crossmodal_clip_40k.pt \
  --device cuda \
  --output_dir results/gate_analysis
echo "GATE_DONE $(date)"
