#!/usr/bin/env python3
"""
Extract ESM-2 per-residue protein embedding deltas for BRCA2 SGE variants.

For missense variants, the output delta is:

    ESM2(alt residue window)[center] - ESM2(ref residue window)[center]

Non-missense variants receive a zero vector and pmask=False, matching the BRCA1
variant-fusion convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PWIN = 510


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, default=root / "data" / "variant" / "brca2" / "brca2_variants.csv")
    parser.add_argument("--protein", type=Path, default=root / "data" / "variant" / "brca2" / "brca2_P51587.fasta")
    parser.add_argument("--output", type=Path, default=root / "results" / "variant" / "brca2_esm_delta.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_protein(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.variants).reset_index(drop=True)
    wt = load_protein(args.protein)

    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    dim = 1280
    pdelta = np.zeros((len(df), dim), dtype=np.float32)
    pmask = np.zeros(len(df), dtype=bool)

    items = []
    bad_ref = []
    for ridx, row in df[df["is_missense"].astype(bool)].iterrows():
        pos = row.get("aa_pos")
        if pd.isna(pos):
            continue
        p = int(pos) - 1
        if p < 0 or p >= len(wt):
            bad_ref.append((ridx, pos, row.get("aa_ref"), "out_of_range"))
            continue
        aa_ref = str(row["aa_ref"])
        aa_alt = str(row["aa_alt"])
        if wt[p] != aa_ref:
            bad_ref.append((ridx, pos, aa_ref, wt[p]))
            continue
        s = max(0, p - PWIN)
        e = min(len(wt), p + PWIN + 1)
        ref_window = wt[s:e]
        center = p - s
        alt_window = ref_window[:center] + aa_alt + ref_window[center + 1 :]
        items.append((ridx, ref_window, alt_window, center))

    print(f"BRCA2 variants={len(df)} missense={int(df['is_missense'].sum())} esm_scorable={len(items)} bad_ref={len(bad_ref)}")
    if bad_ref:
        print("First bad protein refs:", bad_ref[:10])

    def embed_center(seqs: list[str], centers: list[int]) -> np.ndarray:
        rows = []
        for start in range(0, len(seqs), args.batch_size):
            batch = [(f"s{i}", s) for i, s in enumerate(seqs[start : start + args.batch_size])]
            _, _, toks = batch_converter(batch)
            toks = toks.to(device)
            reps = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
            for j, (_, seq) in enumerate(batch):
                rows.append(reps[j, 1 + centers[start + j]].float().cpu().numpy())
            if (start // args.batch_size) % 25 == 0:
                print(f"  ESM {start + len(batch)}/{len(seqs)}")
        return np.asarray(rows, dtype=np.float32)

    if items:
        idxs = [x[0] for x in items]
        refs = [x[1] for x in items]
        alts = [x[2] for x in items]
        centers = [x[3] for x in items]
        ref_emb = embed_center(refs, centers)
        alt_emb = embed_center(alts, centers)
        for i, ridx in enumerate(idxs):
            pdelta[df.index.get_loc(ridx)] = alt_emb[i] - ref_emb[i]
            pmask[df.index.get_loc(ridx)] = True

    np.savez_compressed(args.output, pdelta=pdelta, pmask=pmask)
    print(f"Saved {args.output} pdelta={pdelta.shape} pmask_true={int(pmask.sum())}")


if __name__ == "__main__":
    main()
