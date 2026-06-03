#!/usr/bin/env python3
"""
Evo2 LLR extraction — ref-only forward pass.

For each variant, runs Evo2 on the REFERENCE sequence and extracts
log P(alt_nuc | context) - log P(ref_nuc | context) at the mutation position.

Only needs ONE forward pass per variant (vs 2 for edelta).
Output: {output_dir}/evo2_llr_shard{K}.npz with {idx, llr, done}.
"""

import argparse, gzip, logging, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 8192
FLUSH = 1024


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
    return genome


def parse_window(chrom_seq, pos, ref, alt):
    p = pos - 1
    s = max(0, p - WINDOW // 2)
    e = min(len(chrom_seq), p + WINDOW // 2)
    ref_seq = chrom_seq[s:e]
    c = p - s
    if c < 0 or c >= len(ref_seq) or ref_seq[c] != ref:
        return None, None, None, None
    return ref_seq, c, ref, alt


def atomic_save(path, idx, llr, done):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, idx=idx, llr=llr, done=done)
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
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)

    subset = np.sort(np.load(args.subset_idx))
    bounds = np.linspace(0, len(subset), args.nshards + 1).astype(int)
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    glob_rows = subset[lo:hi]
    sub = df.iloc[glob_rows]
    n = len(glob_rows)
    logger.info(f"shard {args.shard}/{args.nshards}: subset[{lo}:{hi}] = {n} variants")

    out = os.path.join(args.output_dir, f"evo2_llr_shard{args.shard}.npz")

    llr = np.zeros(n, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    if os.path.exists(out):
        try:
            ck = np.load(out)
            if len(ck["idx"]) == n and np.array_equal(ck["idx"], glob_rows):
                llr = ck["llr"].astype(np.float32)
                done = ck["done"]
                logger.info(f"resumed checkpoint: {done.sum()}/{n} already done")
        except Exception as ex:
            logger.warning(f"failed to read checkpoint ({ex}) -> starting fresh")

    if done.all():
        logger.info("all variants already done")
        return

    genome = load_genome(args.genome, sub["chrom"].unique())

    ref_seqs = [None] * n
    centers = [None] * n
    ref_nucs = [None] * n
    alt_nucs = [None] * n
    n_bad = 0
    for local_i, r in enumerate(sub.itertuples(index=False)):
        if done[local_i]:
            continue
        cs = genome.get(str(r.chrom))
        if cs is None:
            done[local_i] = True; n_bad += 1; continue
        rs, c, rn, an = parse_window(cs, int(r.pos), str(r.ref), str(r.alt))
        if rs is None:
            done[local_i] = True; n_bad += 1; continue
        ref_seqs[local_i] = rs; centers[local_i] = c
        ref_nucs[local_i] = rn; alt_nucs[local_i] = an
    todo = [i for i in range(n) if not done[i] and ref_seqs[i] is not None]
    logger.info(f"windows: todo={len(todo)} bad={n_bad} done_already={done.sum()-n_bad}")

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    nuc_to_id = {}
    for nuc in "ACGT":
        tok = model.tokenizer.tokenize(nuc)
        nuc_to_id[nuc] = tok[0] if isinstance(tok, list) else int(tok)
    logger.info(f"nucleotide token IDs: {nuc_to_id}")

    @torch.no_grad()
    def extract_llr_batch(seqs, ctrs, rnucs, anucs, bs, tag=""):
        results = []
        i = 0
        while i < len(seqs):
            if i % 256 == 0:
                logger.info(f"    [{tag}] {i}/{len(seqs)} seqs")
            bseqs = seqs[i:i + bs]
            bctrs = ctrs[i:i + bs]
            bref = rnucs[i:i + bs]
            balt = anucs[i:i + bs]
            try:
                ids = [torch.tensor(model.tokenizer.tokenize(s), dtype=torch.int) for s in bseqs]
                maxlen = max(x.shape[0] for x in ids)
                batch = torch.zeros(len(ids), maxlen, dtype=torch.int)
                for j, x in enumerate(ids):
                    batch[j, :x.shape[0]] = x
                batch = batch.to("cuda:0")
                logits, _ = model(batch, return_embeddings=True, layer_names=[])
                log_probs = F.log_softmax(logits.float(), dim=-1)
                for j in range(len(bseqs)):
                    pos = bctrs[j]
                    ref_id = nuc_to_id.get(bref[j], -1)
                    alt_id = nuc_to_id.get(balt[j], -1)
                    if ref_id >= 0 and alt_id >= 0 and pos > 0 and pos < log_probs.shape[1]:
                        lp = log_probs[j, pos - 1]
                        results.append(float(lp[alt_id] - lp[ref_id]))
                    else:
                        results.append(0.0)
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    results.append(0.0); i += 1
                else:
                    bs = max(1, bs // 2)
                    logger.warning(f"OOM -> batch={bs}")
        return results

    processed = 0
    for cs in range(0, len(todo), FLUSH):
        chunk = todo[cs:cs + FLUSH]
        rseqs = [ref_seqs[i] for i in chunk]
        ctrs = [centers[i] for i in chunk]
        rnucs = [ref_nucs[i] for i in chunk]
        anucs = [alt_nucs[i] for i in chunk]
        batch_llr = extract_llr_batch(rseqs, ctrs, rnucs, anucs, args.batch, tag=f"chunk {cs}")
        for k, i in enumerate(chunk):
            llr[i] = batch_llr[k]
            done[i] = True
        processed += len(chunk)
        atomic_save(out, glob_rows, llr, done)
        logger.info(f"checkpoint: {done.sum()}/{n} done ({processed} this run)")

    logger.info(f"DONE: {done.sum()}/{n} variants")


if __name__ == "__main__":
    main()
