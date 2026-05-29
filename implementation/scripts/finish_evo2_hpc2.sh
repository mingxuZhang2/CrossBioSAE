#!/bin/bash
# Finish Evo2 install: torch already installing/installed in evo2x.
# Real bug was wget saving wheels with non-canonical names (pip rejects) +
# empty ABI -> 404 flash-attn URL. Fix: canonical names + curl resume.
set -u
source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate evo2x
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation/evo2_wheels
GH=https://ghproxy.net

# 1) ensure torch installed (canonical name)
python -c "import torch" 2>/dev/null || {
  cp -f torch.whl "torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
  pip install --no-input "torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl" 2>&1 | tail -3
}
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
TV=$(python -c "import torch; print(torch.__version__)")
echo "=== torch=$TV ABI=$ABI $(date) ==="

# 2) flash-attn wheel via curl (resume), canonical name
FA="flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
URL="$GH/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
# if a stale fa.whl exists with possibly-wrong ABI, start fresh for the abi-correct name
n=0
until curl -fL -C - -o "$FA" "$URL"; do
  n=$((n+1)); echo "  [curl retry $n] $FA"; [ $n -ge 60 ] && { echo "GAVE UP $FA"; exit 1; }
  sleep 2
done
echo "=== downloaded $FA $(ls -la "$FA" | awk '{print $5}') bytes $(date) ==="
pip install --no-input --no-deps "$FA" 2>&1 | tail -3

# 3) evo2 (+vtx) via mirror, keep torch pinned
pip install --no-input evo2 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 10 2>&1 | tail -6
TV2=$(python -c "import torch;print(torch.__version__)" 2>/dev/null)
case "$TV2" in
  2.7.1*) : ;;
  *) echo "  torch drifted $TV2 -> reinstall local"; pip install --no-input --force-reinstall --no-deps "torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl" 2>&1 | tail -2 ;;
esac

# 4) verify
echo "=== verify $(date) ==="
python -c "import torch, flash_attn; from evo2.models import Evo2; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| evo2 import OK')"
echo "EVO2X_DONE_$?  $(date)"
