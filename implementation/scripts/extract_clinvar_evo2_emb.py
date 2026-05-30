"""
Evo2-7b embedding deltas for genome-wide ClinVar (for SAE interpretability).
Like extract_clinvar_evo2.py but saves per-variant 4096-d embedding delta
at the variant position, not just the scalar LLR.

Sharded: --shard 0..7 --nshards 8.
Output per shard: clinvar_evo2_emb_shard{K}.npz {idx, edelta}
"""

import argparse
import gzip
import logging
import os

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 8192
LAYER = "blocks.28.mlp.l3"


def load_genome(fa_gz, keep_chroms):
    keep = set(str(c) for c in keep_chroms)
    op = gzip.open if fa_gz.endswith(".gz") else open
    genome, cur, buf = {}, None, []
    with op(fa_gz, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and cur in keep:
                    genome[cur] = "".join(buf).upper()
                cur = line[1:].split()[0]; buf = []
                if cur in keep:
                    logger.info(f"  reading chrom {cur} ...")
            elif cur in keep:
                buf.append(line.strip())
        if cur is not None and cur in keep:
            genome[cur] = "".join(buf).upper()
    logger.info(f"loaded {len(genome)} chroms")
    return genome


def parse_window(chrom_seq, pos, ref, alt):
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
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)

    n = len(df)
    bounds = np.linspace(0, n, args.nshards + 1).astype(int)
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    rows = np.arange(lo, hi)
    sub = df.iloc[rows]
    logger.info(f"shard {args.shard}/{args.nshards}: rows [{lo},{hi}) = {len(sub)} variants")

    genome = load_genome(args.genome, sub["chrom"].unique())

    ref_seqs, ref_key_map = [], {}
    ref_index, var_seqs, centers, glob_rows = [], [], [], []
    n_bad = 0
    for gr, r in zip(rows, sub.itertuples(index=False)):
        cs = genome.get(str(r.chrom))
        if cs is None:
            n_bad += 1; continue
        rs, vs, c = parse_window(cs, int(r.pos), str(r.ref), str(r.alt))
        if rs is None:
            n_bad += 1; continue
        key = (r.chrom, int(r.pos))
        if key not in ref_key_map:
            ref_key_map[key] = len(ref_seqs)
            ref_seqs.append(rs)
        ref_index.append(ref_key_map[key])
        var_seqs.append(vs)
        centers.append(c)
        glob_rows.append(gr)
    logger.info(f"windows ok={len(glob_rows)} bad={n_bad}, unique ref={len(ref_seqs)}")

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    @torch.no_grad()
    def embed_batch(seqs, ctrs):
        bs = args.batch
        out = []
        for i in range(0, len(seqs), bs):
            bseqs = seqs[i:i + bs]
            bctrs = ctrs[i:i + bs]
            while True:
                try:
                    ids = [torch.tensor(model.tokenizer.tokenize(s), dtype=torch.int) for s in bseqs]
                    maxlen = max(x.shape[0] for x in ids)
                    batch = torch.zeros(len(ids), maxlen, dtype=torch.int)
                    for j, x in enumerate(ids):
                        batch[j, :x.shape[0]] = x
                    batch = batch.to("cuda:0")
                    _, emb = model(batch, return_embeddings=True, layer_names=[LAYER])
                    e = emb[LAYER]
                    for j in range(len(bseqs)):
                        out.append(e[j, bctrs[j]].float().cpu().numpy())
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        for _ in bseqs:
                            out.append(np.zeros(4096, dtype=np.float32))
                        break
                    bs = max(1, bs // 2)
                    logger.warning(f"OOM -> batch={bs}")
            if (i // max(bs, 1)) % 50 == 0:
                logger.info(f"  emb {i + len(bseqs)}/{len(seqs)}")
        return np.array(out)

    # ref embeddings (deduped)
    logger.info("extracting ref embeddings (dedup) ...")
    ref_center = [None] * len(ref_seqs)
    for k, j in enumerate(ref_index):
        if ref_center[j] is None:
            ref_center[j] = centers[k]
    ref_emb = embed_batch(ref_seqs, ref_center)

    # var embeddings
    logger.info("extracting var embeddings ...")
    var_emb = embed_batch(var_seqs, centers)

    # delta
    edelta = np.zeros((len(glob_rows), var_emb.shape[1]), dtype=np.float32)
    for k in range(len(glob_rows)):
        edelta[k] = var_emb[k] - ref_emb[ref_index[k]]

    out = os.path.join(args.output_dir, f"clinvar_evo2_emb_shard{args.shard}.npz")
    np.savez_compressed(out, idx=np.array(glob_rows), edelta=edelta)
    logger.info(f"saved {out}: {edelta.shape}")


if __name__ == "__main__":
    main()
