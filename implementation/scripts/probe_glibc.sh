#!/bin/bash
#SBATCH -J probe
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH -t 00:03:00
#SBATCH -o logs/probe_glibc.out
#SBATCH -e logs/probe_glibc.out
echo "host: $(hostname)"
ldd --version | head -1
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
python -c "from evo2.models import Evo2; print('evo2 import OK')"
