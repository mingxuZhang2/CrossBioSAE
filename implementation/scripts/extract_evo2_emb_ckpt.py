"""
Evo2-7b embedding deltas — checkpointed + resumable.

Designed for the cross-modal SAE pretraining set: only the dual-modality
variants (those that already have an ESM-2 protein embedding) are extracted,
since cross-modal SAE training needs BOTH modalities present.

Key differences vs extract_clinvar_evo2_emb.py:
  - --subset_idx: only process global rows listed in this .npy (dual-modality set)
  - incremental checkpointing: edelta + done-mask flushed every FLUSH variants,
    written atomically (tmp -> rename). On restart, resumes from checkpoint.
    => robust to SLURM time-limit kills and node preemption/requeue.
  - ref/var processed inline per chunk (dedup dropped — negligible for missense).

Output per shard: {output_dir}/clinvar_evo2_emb_shard{K}.npz
  {idx: global rows, edelta: (n,4096), done: (n,) bool}
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
DIM = 4096
FLUSH = 1024  # variants between checkpoint writes


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


def atomic_save(path, idx, edelta, done):
    # tmp MUST end in .npz, else np.savez_compressed appends .npz and os.replace
    # looks for the wrong filename.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, idx=idx, edelta=edelta, done=done)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/variant/sae_pretrain/missense_500k.parquet")
    ap.add_argument("--subset_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--genome", default="data/variant/GRCh38.fa.gz")
    ap.add_argument("--output_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path", default="models/evo2_7b.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)

    # subset to dual-modality global rows, then shard over the subset
    subset = np.sort(np.load(args.subset_idx))
    bounds = np.linspace(0, len(subset), args.nshards + 1).astype(int)
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    glob_rows = subset[lo:hi]
    sub = df.iloc[glob_rows]
    n = len(glob_rows)
    logger.info(f"shard {args.shard}/{args.nshards}: subset[{lo}:{hi}] = {n} variants")

    out = os.path.join(args.output_dir, f"clinvar_evo2_emb_shard{args.shard}.npz")

    # resume from checkpoint if present
    edelta = np.zeros((n, DIM), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    if os.path.exists(out):
        try:
            ck = np.load(out)
            if len(ck["idx"]) == n and np.array_equal(ck["idx"], glob_rows):
                edelta = ck["edelta"].astype(np.float32)
                done = ck["done"]
                logger.info(f"resumed checkpoint: {done.sum()}/{n} already done")
            else:
                logger.warning("checkpoint idx mismatch -> starting fresh")
        except Exception as ex:
            logger.warning(f"failed to read checkpoint ({ex}) -> starting fresh")

    if done.all():
        logger.info("all variants already done, nothing to do")
        return

    genome = load_genome(args.genome, sub["chrom"].unique())

    # precompute windows (cheap string ops); mark bad windows done w/ zero delta
    ref_seqs = [None] * n
    var_seqs = [None] * n
    centers = [None] * n
    n_bad = 0
    for local_i, r in enumerate(sub.itertuples(index=False)):
        if done[local_i]:
            continue
        cs = genome.get(str(r.chrom))
        if cs is None:
            done[local_i] = True; n_bad += 1; continue
        rs, vs, c = parse_window(cs, int(r.pos), str(r.ref), str(r.alt))
        if rs is None:
            done[local_i] = True; n_bad += 1; continue
        ref_seqs[local_i] = rs; var_seqs[local_i] = vs; centers[local_i] = c
    todo = [i for i in range(n) if not done[i] and ref_seqs[i] is not None]
    logger.info(f"windows: todo={len(todo)} bad={n_bad} done_already={done.sum()-n_bad}")

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    @torch.no_grad()
    def embed(seqs, ctrs, bs, tag=""):
        out = []
        i = 0
        while i < len(seqs):
            if i % 256 == 0:
                logger.info(f"    [{tag}] {i}/{len(seqs)} seqs")
            bseqs = seqs[i:i + bs]; bctrs = ctrs[i:i + bs]
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
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    out.append(np.zeros(DIM, dtype=np.float32)); i += 1
                else:
                    bs = max(1, bs // 2)
                    logger.warning(f"OOM -> batch={bs}")
        return np.array(out, dtype=np.float32)

    # process todo in flush-sized chunks; ref+var each chunk, then checkpoint
    processed = 0
    for cs in range(0, len(todo), FLUSH):
        chunk = todo[cs:cs + FLUSH]
        rseqs = [ref_seqs[i] for i in chunk]
        vseqs = [var_seqs[i] for i in chunk]
        ctrs = [centers[i] for i in chunk]
        emb_ref = embed(rseqs, ctrs, args.batch, tag=f"ref {cs}")
        emb_var = embed(vseqs, ctrs, args.batch, tag=f"var {cs}")
        for k, i in enumerate(chunk):
            edelta[i] = emb_var[k] - emb_ref[k]
            done[i] = True
        processed += len(chunk)
        atomic_save(out, glob_rows, edelta, done)
        logger.info(f"  checkpoint: {done.sum()}/{n} done (+{len(chunk)} this flush)")

    atomic_save(out, glob_rows, edelta, done)
    nz = np.any(edelta != 0, axis=1).sum()
    logger.info(f"FINISHED {out}: {edelta.shape}, non-zero={nz}, done={done.sum()}/{n}")


if __name__ == "__main__":
    main()
