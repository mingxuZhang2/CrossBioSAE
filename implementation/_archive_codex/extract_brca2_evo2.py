#!/usr/bin/env python3
"""
Evo2-7b zero-shot LLR and window-token embedding deltas for BRCA2 SGE variants.

Uses GRCh38 coordinates from data/variant/brca2/brca2_variants.csv and a full
GRCh38 FASTA. Output mirrors the BRCA1 convention:

    results/variant/<output-prefix>_evo2_llr.npz  {llr}
    results/variant/<output-prefix>_evo2.npz      {llr, edelta}
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 8192


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, default=root / "data" / "variant" / "brca2" / "brca2_variants.csv")
    parser.add_argument("--genome", type=Path, default=root / "data" / "variant" / "GRCh38.fa.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "variant")
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--local-path", type=Path, default=root / "models" / "evo2_7b.pt")
    parser.add_argument("--layer", default="blocks.28.mlp.l3")
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--emb-batch", type=int, default=2)
    parser.add_argument("--pos-col", default="pos_hg38")
    parser.add_argument("--output-prefix", default="brca2")
    parser.add_argument("--llr-only", action="store_true", help="Only write LLR scores; skip embedding deltas.")
    return parser.parse_args()


def load_chrom(genome: Path, chrom: str) -> str:
    op = gzip.open if str(genome).endswith(".gz") else open
    cur = None
    buf: list[str] = []
    with op(genome, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if cur == chrom:
                    break
                cur = line[1:].split()[0]
                buf = []
            elif cur == chrom:
                buf.append(line.strip())
    if cur != chrom and not buf:
        raise RuntimeError(f"Chromosome {chrom} not found in {genome}")
    seq = "".join(buf).upper()
    logger.info("loaded chrom %s length=%d", chrom, len(seq))
    return seq


def parse_window(chrom_seq: str, pos: int, ref: str, alt: str) -> tuple[str | None, str | None, int | None]:
    p = pos - 1
    start = max(0, p - WINDOW // 2)
    end = min(len(chrom_seq), p + WINDOW // 2)
    ref_seq = chrom_seq[start:end]
    center = p - start
    if center < 0 or center >= len(ref_seq) or ref_seq[center] != ref:
        return None, None, None
    var_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]
    return ref_seq, var_seq, center


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    chroms = sorted(df["chrom"].astype(str).str.replace("chr", "", regex=False).unique())
    if len(chroms) != 1:
        raise ValueError(f"Expected exactly one chromosome in the variant table, got {chroms}")
    chrom_seq = load_chrom(args.genome, chroms[0])

    ref_seqs: list[str] = []
    ref_idx: dict[tuple[int, str], int] = {}
    ref_index: list[int] = []
    var_seqs: list[str] = []
    centers: list[int] = []
    ok_rows: list[int] = []
    bad = []
    for i, row in df.iterrows():
        rs, vs, center = parse_window(chrom_seq, int(row[args.pos_col]), str(row["ref"]), str(row["alt"]))
        if rs is None:
            bad.append((i, int(row[args.pos_col]), row["ref"], chrom_seq[int(row[args.pos_col]) - 1]))
            continue
        key = (int(row[args.pos_col]), str(row["ref"]))
        if key not in ref_idx:
            ref_idx[key] = len(ref_seqs)
            ref_seqs.append(rs)
        ref_index.append(ref_idx[key])
        var_seqs.append(vs)
        centers.append(int(center))
        ok_rows.append(i)
    logger.info("windows ok=%d bad=%d unique_ref=%d", len(ok_rows), len(bad), len(ref_seqs))
    if bad:
        logger.warning("first bad windows: %s", bad[:10])

    from evo2.models import Evo2

    if args.local_path.exists():
        logger.info("loading %s from %s", args.model, args.local_path)
        model = Evo2(args.model, local_path=str(args.local_path))
    else:
        logger.info("loading %s from cache/default", args.model)
        model = Evo2(args.model)

    def score(seqs: list[str], label: str) -> np.ndarray:
        bs = args.score_batch
        while True:
            try:
                logger.info("scoring %s n=%d batch=%d", label, len(seqs), bs)
                return np.asarray(model.score_sequences(seqs, batch_size=bs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                logger.warning("CUDA OOM, retry scoring %s with batch=%d", label, bs)

    ref_scores = score(ref_seqs, "ref")
    var_scores = score(var_seqs, "alt")
    delta = var_scores - ref_scores[np.asarray(ref_index)]
    llr = np.full(len(df), np.nan, dtype=np.float32)
    for k, row_idx in enumerate(ok_rows):
        llr[row_idx] = delta[k]
    np.savez_compressed(args.output_dir / f"{args.output_prefix}_evo2_llr.npz", llr=llr)

    ok = np.isfinite(llr)
    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
        labeled_ok = ok & np.isfinite(y)
        if labeled_ok.sum() and len(np.unique(y[labeled_ok].astype(int))) == 2:
            logger.info(
                "Evo2 zero-shot AUROC all n=%d: %.4f",
                labeled_ok.sum(),
                roc_auc_score(y[labeled_ok].astype(int), -llr[labeled_ok]),
            )
        else:
            logger.info(
                "Skipping zero-shot AUROC sanity check: labeled finite rows=%d unique_labels=%s",
                int(labeled_ok.sum()),
                sorted(pd.unique(y[labeled_ok]).tolist()) if labeled_ok.any() else [],
            )
    if args.llr_only:
        logger.info("LLR-only mode complete; skipped embeddings.")
        return

    @torch.no_grad()
    def embed(seqs: list[str], centers_: list[int], label: str) -> np.ndarray:
        rows = []
        for start in range(0, len(seqs), args.emb_batch):
            batch = seqs[start : start + args.emb_batch]
            batch_centers = centers_[start : start + args.emb_batch]
            ids = [torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int) for seq in batch]
            max_len = max(x.shape[0] for x in ids)
            toks = torch.zeros(len(ids), max_len, dtype=torch.int)
            for j, x in enumerate(ids):
                toks[j, : x.shape[0]] = x
            toks = toks.to("cuda:0")
            _, emb = model(toks, return_embeddings=True, layer_names=[args.layer])
            e = emb[args.layer]
            for j, center in enumerate(batch_centers):
                rows.append(e[j, center].float().cpu().numpy())
            if (start // args.emb_batch) % 50 == 0:
                logger.info("  Evo2 emb %s %d/%d", label, start + len(batch), len(seqs))
        return np.asarray(rows, dtype=np.float32)

    ref_centers = [0] * len(ref_seqs)
    for k, ref_i in enumerate(ref_index):
        if ref_centers[ref_i] == 0:
            ref_centers[ref_i] = centers[k]
    ref_emb = embed(ref_seqs, ref_centers, "ref")
    var_emb = embed(var_seqs, centers, "alt")
    edelta = np.zeros((len(df), var_emb.shape[1]), dtype=np.float32)
    for k, row_idx in enumerate(ok_rows):
        edelta[row_idx] = var_emb[k] - ref_emb[ref_index[k]]
    np.savez_compressed(args.output_dir / f"{args.output_prefix}_evo2.npz", llr=llr, edelta=edelta)
    logger.info("saved %s_evo2.npz llr=%s edelta=%s", args.output_prefix, llr.shape, edelta.shape)


if __name__ == "__main__":
    main()
