"""
Evo2-7b zero-shot LLR for the genome-wide ClinVar / songlab GPN-MSA benchmark.

We already proved on BRCA1 that the SCALAR LLR fusion beats the 4096-d embedding
fusion, so here we compute ONLY the LLR (no embeddings) -> ~half the GPU cost.

This benchmark (data/variant/clinvar.parquet, 40,976 SNVs across 24 chromosomes)
already ships precomputed ESM-1b / NT / HyenaDNA / GPN-MSA / CADD / conservation.
The one missing piece is the strong DNA model, Evo2 -> we add it here.

Sharded for parallelism: run 8 jobs with --shard 0..7 --nshards 8, each scores
its slice and writes results/variant/clinvar_evo2_llr_shard{shard}.npz {idx, llr}.
merge_clinvar_evo2.py stitches the shards back together.

Windows: 8192bp centered on the variant (Evo2 recipe), ref-base validated.
Compute nodes are offline -> weights from local .pt, HF offline.
"""

import argparse
import gzip
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 8192


def load_genome(fa_gz, keep_chroms):
    """Parse a (gzipped) FASTA, keeping only chromosomes we need. Ensembl primary
    assembly uses bare names ('1','X') which match the benchmark's `chrom`."""
    keep = set(str(c) for c in keep_chroms)
    op = gzip.open if fa_gz.endswith(".gz") else open
    genome, cur, buf = {}, None, []
    with op(fa_gz, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and cur in keep:
                    genome[cur] = "".join(buf).upper()
                cur = line[1:].split()[0]
                buf = []
                if cur in keep:
                    logger.info(f"  reading chrom {cur} ...")
            elif cur in keep:
                buf.append(line.strip())
        if cur is not None and cur in keep:
            genome[cur] = "".join(buf).upper()
    logger.info(f"loaded {len(genome)} chroms: {sorted(genome)}")
    return genome


def parse_window(chrom_seq, pos, ref, alt):
    """1-based pos. Returns (ref_window, var_window, center) or (None,None,None)
    if ref base doesn't match (coordinate/assembly mismatch)."""
    p = pos - 1
    s = max(0, p - WINDOW // 2)
    e = min(len(chrom_seq), p + WINDOW // 2)
    ref_seq = chrom_seq[s:e]
    c = p - s
    if c < 0 or c >= len(ref_seq) or ref_seq[c] != ref:
        return None, None, None
    var_seq = ref_seq[:c] + alt + ref_seq[c + 1:]
    return ref_seq, var_seq, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/variant/clinvar.parquet")
    ap.add_argument("--genome", default="data/variant/GRCh38.fa.gz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path", default="models/evo2_7b.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)

    # shard by row (contiguous slices keep nearby positions together -> better
    # ref-window dedup within a shard)
    n = len(df)
    bounds = np.linspace(0, n, args.nshards + 1).astype(int)
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    rows = np.arange(lo, hi)
    sub = df.iloc[rows]
    logger.info(f"shard {args.shard}/{args.nshards}: rows [{lo},{hi}) = {len(sub)} variants")

    genome = load_genome(args.genome, sub["chrom"].unique())

    # build windows with ref-window dedup (multi-allelic share a ref window)
    ref_seqs, ref_key_map = [], {}
    ref_index, var_seqs, glob_rows = [], [], []
    n_bad = 0
    for gr, r in zip(rows, sub.itertuples(index=False)):
        cs = genome.get(str(r.chrom))
        if cs is None:
            n_bad += 1
            continue
        rs, vs, _ = parse_window(cs, int(r.pos), str(r.ref), str(r.alt))
        if rs is None:
            n_bad += 1
            continue
        key = (r.chrom, int(r.pos))
        if key not in ref_key_map:
            ref_key_map[key] = len(ref_seqs)
            ref_seqs.append(rs)
        ref_index.append(ref_key_map[key])
        var_seqs.append(vs)
        glob_rows.append(gr)
    logger.info(f"windows ok={len(glob_rows)} bad={n_bad}, unique ref={len(ref_seqs)}")

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    logger.info("scoring ref windows ...")
    ref_scores = np.array(model.score_sequences(ref_seqs))
    logger.info("scoring var windows ...")
    var_scores = np.array(model.score_sequences(var_seqs))
    delta = var_scores - ref_scores[np.array(ref_index)]

    out = os.path.join(args.output_dir, f"clinvar_evo2_llr_shard{args.shard}.npz")
    np.savez_compressed(out, idx=np.array(glob_rows), llr=delta.astype(np.float32))
    logger.info(f"saved {out}  ({len(glob_rows)} llr values)")


if __name__ == "__main__":
    main()
