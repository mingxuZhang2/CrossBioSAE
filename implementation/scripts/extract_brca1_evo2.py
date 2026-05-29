"""
Evo2 features for BRCA1 variants (run on HPC2 A800 once evo2 is installed).

Produces, per variant (8192bp genomic window, Evo2 recipe):
  - evo2_llr   : zero-shot delta log-likelihood = score(alt) - score(ref).
                 This is Evo2's published variant-effect score (the baseline to beat).
  - evo2_delta : window-token embedding delta at the variant position (hidden dim,
                 4096 for evo2_7b) -> replaces the weak NT delta in the fusion.

API (from ArcInstitute/evo2 brca1 notebook + repo README):
  from evo2.models import Evo2
  model = Evo2('evo2_7b')
  scores = model.score_sequences([seq, ...])             # log-likelihoods
  _, emb = model(input_ids, return_embeddings=True, layer_names=[LAYER])

Output: results/variant/brca1_evo2.npz  {llr, edelta}
"""

import argparse
import gzip
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 8192


def load_chr(fna_gz):
    op = gzip.open if fna_gz.endswith(".gz") else open
    seq = []
    with op(fna_gz, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()


def parse_window(pos, ref, alt, chrom):
    p = pos - 1
    s = max(0, p - WINDOW // 2)
    e = min(len(chrom), p + WINDOW // 2)
    ref_seq = chrom[s:e]
    c = min(WINDOW // 2, p)
    if ref_seq[c] != ref:
        return None, None, None
    var_seq = ref_seq[:c] + alt + ref_seq[c + 1:]
    return ref_seq, var_seq, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--chr17", default="data/variant/brca1/chr17_GRCh37.fna.gz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path", default="models/evo2_7b.pt",
                    help="local Evo2 .pt weights (compute nodes have no internet)")
    ap.add_argument("--layer", default="blocks.28.mlp.l3",
                    help="Evo2 embedding layer (official README example layer for evo2_7b)")
    ap.add_argument("--emb_batch", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    chrom = load_chr(args.chr17)
    logger.info(f"variants={len(df)} chr17={len(chrom)}bp")

    # Build windows with ref dedup (many variants share a ref window)
    ref_seqs, ref_idx_map = [], {}
    ref_index, var_seqs, centers, ok_rows = [], [], [], []
    n_bad = 0
    for ridx, r in df.iterrows():
        rs, vs, c = parse_window(int(r["pos_hg19"]), str(r["ref"]), str(r["alt"]), chrom)
        if rs is None:
            n_bad += 1
            continue
        if rs not in ref_idx_map:
            ref_idx_map[rs] = len(ref_seqs)
            ref_seqs.append(rs)
        ref_index.append(ref_idx_map[rs])
        var_seqs.append(vs)
        centers.append(c)
        ok_rows.append(df.index.get_loc(ridx))
    logger.info(f"windows ok={len(ok_rows)} bad={n_bad}, unique ref seqs={len(ref_seqs)}")

    from evo2.models import Evo2
    if args.local_path and os.path.exists(args.local_path):
        logger.info(f"loading {args.model} from local {args.local_path} ...")
        model = Evo2(args.model, local_path=args.local_path)
    else:
        logger.info(f"loading {args.model} (HF cache) ...")
        model = Evo2(args.model)

    # ---- 1) zero-shot LLR baseline ----
    logger.info("scoring ref sequences ...")
    ref_scores = np.array(model.score_sequences(ref_seqs))
    logger.info("scoring var sequences ...")
    var_scores = np.array(model.score_sequences(var_seqs))
    delta = var_scores - ref_scores[np.array(ref_index)]
    llr = np.full(len(df), np.nan, dtype=np.float32)
    for k, ix in enumerate(ok_rows):
        llr[ix] = delta[k]

    # quick zero-shot sanity auROC (Evo2 baseline)
    y = df["label"].values
    m = ~np.isnan(llr)
    au = roc_auc_score(y[m], -llr[m])
    logger.info(f"Evo2 zero-shot LLR auROC (all, n={m.sum()}): {au:.4f}")
    for vt in ["coding", "noncoding"]:
        mm = m & (df["vtype"] == vt).values
        if mm.sum() > 20 and len(np.unique(y[mm])) == 2:
            logger.info(f"  {vt}: {roc_auc_score(y[mm], -llr[mm]):.4f}")

    # Save LLR immediately so the zero-shot baseline survives even if the
    # (heavier) embedding step OOMs or the layer name is off.
    np.savez_compressed(os.path.join(args.output_dir, "brca1_evo2_llr.npz"), llr=llr)
    logger.info("saved brca1_evo2_llr.npz (baseline secured)")

    # ---- 2) window-token embedding delta ----
    hidden = None
    edelta = None

    @torch.no_grad()
    def embed(seqs, centers_):
        nonlocal hidden, edelta
        out = []
        for i in range(0, len(seqs), args.emb_batch):
            bs = seqs[i:i + args.emb_batch]
            bc = centers_[i:i + args.emb_batch]
            ids = [torch.tensor(model.tokenizer.tokenize(s), dtype=torch.int) for s in bs]
            maxlen = max(x.shape[0] for x in ids)
            batch = torch.zeros(len(ids), maxlen, dtype=torch.int)
            for j, x in enumerate(ids):
                batch[j, :x.shape[0]] = x
            batch = batch.to("cuda:0")
            _, emb = model(batch, return_embeddings=True, layer_names=[args.layer])
            e = emb[args.layer]  # (B, L, H)
            for j in range(len(bs)):
                out.append(e[j, bc[j]].float().cpu().numpy())
            if (i // args.emb_batch) % 50 == 0:
                logger.info(f"  Evo2 emb {i+len(bs)}/{len(seqs)}")
        return np.array(out)

    try:
        logger.info("extracting Evo2 embeddings (ref windows, dedup) ...")
        # variant-token center for each unique ref seq = center of its first occurrence
        ref_center = [None] * len(ref_seqs)
        for k, j in enumerate(ref_index):
            if ref_center[j] is None:
                ref_center[j] = centers[k]
        ref_emb = embed(ref_seqs, ref_center)
        logger.info("extracting Evo2 embeddings (var windows) ...")
        var_emb = embed(var_seqs, centers)
        H = var_emb.shape[1]
        edelta = np.zeros((len(df), H), dtype=np.float32)
        for k, ix in enumerate(ok_rows):
            edelta[ix] = var_emb[k] - ref_emb[ref_index[k]]

        np.savez_compressed(os.path.join(args.output_dir, "brca1_evo2.npz"),
                            llr=llr, edelta=edelta)
        logger.info(f"saved brca1_evo2.npz  llr={llr.shape} edelta={edelta.shape}")
    except Exception as e:
        logger.error(f"embedding step failed ({args.layer}): {e}")
        logger.error("LLR baseline already saved to brca1_evo2_llr.npz; "
                     "inspect layer names and rerun embeddings only.")
        raise


if __name__ == "__main__":
    main()
