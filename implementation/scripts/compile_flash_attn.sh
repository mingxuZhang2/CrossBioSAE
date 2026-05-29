#!/bin/bash
#SBATCH -J fa_build
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 10:00:00
#SBATCH -o logs/compile_flash_attn.out
#SBATCH -e logs/compile_flash_attn.out
# Compile flash-attn 2.8.0.post2 from source against torch 2.7.1+cu128, linked
# to the node's local glibc 2.31 (the prebuilt wheel needs glibc 2.32 -> fails).
# Only build sm_80 (A800 Ampere) to keep compile time minimal.
set -e
echo "=== host $(hostname)  $(date) ==="
ldd --version | head -1

module load cuda/12.4
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -2

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x

# build knobs: single arch (A800=sm_80), parallel nvcc capped for RAM
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=6
export NVCC_THREADS=1

# NOTE: compute nodes have NO internet. ninja/packaging/psutil already installed
# into evo2x from the login node; build offline from the local sdist tarball.
python -c "import torch, ninja, packaging, psutil; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| ninja', ninja.__version__)"

echo "=== remove any existing flash-attn ==="
pip uninstall -y flash-attn flash_attn 2>&1 | tail -2

echo "=== compile flash-attn 2.8.0.post2 from local sdist (OFFLINE) $(date) ==="
# tarball bundles cutlass; --no-index = no network, --no-build-isolation reuses installed torch
pip install --no-input --no-index --no-build-isolation --no-deps \
  flash_src/flash_attn-2.8.0.post2.tar.gz 2>&1 | tail -30

echo "=== verify $(date) ==="
python -c "import torch, flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "import flash_attn_2_cuda; print('flash_attn_2_cuda OK')"
python -c "from evo2.models import Evo2; print('evo2 import OK')"
echo "FA_BUILD_DONE_$?  $(date)"
