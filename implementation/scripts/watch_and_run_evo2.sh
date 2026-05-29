#!/bin/bash
cd /hpc2hdd/home/mzhang630/data/bioinfo/implementation
# 1) wait for weights download to finish
for i in $(seq 1 720); do
  if grep -q "EVO2_WEIGHTS_DONE" logs/evo2_7b_download.log 2>/dev/null; then break; fi
  # also break if file reached full size (13766621200) in case marker missed
  sz=$(stat -c%s models/evo2_7b.pt 2>/dev/null || echo 0)
  [ "$sz" -ge 13766621200 ] && break
  sleep 30
done
sz=$(stat -c%s models/evo2_7b.pt 2>/dev/null || echo 0)
echo "weights size=$sz (need 13766621200)"
if [ "$sz" -lt 13766621200 ]; then echo "WEIGHTS_INCOMPLETE"; exit 1; fi
# 2) submit extraction job
JID=$(sbatch --parsable scripts/slurm_brca1_evo2.sh)
echo "submitted extract job $JID"
# 3) wait for it
for i in $(seq 1 720); do
  grep -q "BRCA1_EVO2_DONE" logs/brca1_evo2.out 2>/dev/null && break
  st=$(squeue -j "$JID" -h -o "%T" 2>/dev/null); [ -z "$st" ] && [ -f logs/brca1_evo2.out ] && break
  sleep 30
done
echo "=== brca1_evo2.out tail ==="; tail -25 logs/brca1_evo2.out 2>/dev/null
