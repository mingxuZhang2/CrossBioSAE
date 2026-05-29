#!/bin/bash
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
URL="https://huggingface.co/arcinstitute/evo2_7b/resolve/main/evo2_7b.pt"
OUT=models/evo2_7b.pt
n=0
until curl -fL -C - -o "$OUT" "$URL"; do
  n=$((n+1)); echo "[retry $n] $(date)"; [ $n -ge 80 ] && { echo GAVEUP; exit 1; }
  sleep 3
done
echo "EVO2_WEIGHTS_DONE size=$(stat -c%s "$OUT") $(date)"
