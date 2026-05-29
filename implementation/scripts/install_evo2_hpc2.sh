#!/bin/bash
# Robust Evo2 install on HPC2 (login node has internet via proxy, NO nvcc).
# Strategy: resumable wget for big wheels (beats proxy timeouts) + prebuilt
# flash-attn wheel (no compilation). torch 2.7.1 per Evo2 README.
set -u
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh

WD=/hpc2hdd/home/mzhang630/data/bioinfo/implementation/evo2_wheels
mkdir -p "$WD"; cd "$WD"

dl() {  # dl <url> <outfile>  -- resumable, retry until complete
  local url="$1" out="$2" n=0
  until wget -c -T 90 -q --show-progress -O "$out" "$url"; do
    n=$((n+1)); echo "  [retry $n] $out"; [ $n -ge 40 ] && { echo "GAVE UP $out"; return 1; }
    sleep 3
  done
}

GH=https://ghproxy.net  # github.com direct is blocked by the proxy; ghproxy.net works

echo "=== ensure env $(date) ==="
conda create -y -n evo2x python=3.11 2>&1 | tail -2
conda activate evo2x

echo "=== torch 2.7.1+cu128 (local whl if present) $(date) ==="
[ -s torch.whl ] || dl "https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl" torch.whl || exit 1
python -c "import torch" 2>/dev/null || pip install --no-input torch.whl 2>&1 | tail -3

ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
echo "=== torch ABI=$ABI -> flash-attn wheel via ghproxy $(date) ==="
rm -f fa.whl
dl "$GH/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl" fa.whl || exit 1
pip install --no-input --no-deps fa.whl 2>&1 | tail -3

echo "=== evo2 (+vtx, small deps via mirror) $(date) ==="
pip install --no-input evo2 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 10 2>&1 | tail -8

# evo2/vtx may have dragged torch off 2.7.1 -> flash_attn ABI would break. Pin it back.
TV=$(python -c "import torch;print(torch.__version__)" 2>/dev/null)
echo "torch after evo2 install: $TV"
case "$TV" in
  2.7.1*) : ;;
  *) echo "  torch drifted to $TV -> reinstalling local 2.7.1 whl"
     pip install --no-input --force-reinstall --no-deps torch.whl 2>&1 | tail -2 ;;
esac

echo "=== verify $(date) ==="
python -c "import torch, flash_attn; from evo2.models import Evo2; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| evo2 import OK')"
echo "EVO2X_DONE_$?  $(date)"
