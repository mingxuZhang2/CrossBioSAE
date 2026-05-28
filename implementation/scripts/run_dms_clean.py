"""
Clean decisive test: is the GFP cross-modal advantage real, or a noise/leakage artifact?

Two confounds in the original 0.507 result:
  (1) NOISE: 98% of variants have a single barcode (NaN std) = unreliable brightness.
  (2) LEAKAGE: random KFold can split synonymous variants of the same protein across
      train/test, letting the model memorize a protein's brightness from its twin.

This script removes BOTH and runs two tests:

  Test A  -- Reliability filter (>=2 barcodes) + GroupKFold by protein.
            3-way comparison protein / DNA / concat. If DNA/concat still beat
            protein under group-CV on reliable data, the gain is not leakage.

  Test B  -- SMOKING GUN: within-synonymous-group prediction.
            For proteins with >=3 reliable synonymous CDS variants, the protein
            sequence (and thus protein embedding) is IDENTICAL within a group, so a
            protein-only model has ZERO within-group predictive power by construction.
            We train with GroupKFold (test proteins entirely unseen) and measure the
            mean within-group Spearman(pred, true). If DNA-only > 0 here, the DNA LM
            captures generalizable codon-level fitness signal protein models cannot.
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_reliable(nt_file, ref_file, min_barcodes=2, max_mutations=15):
    """Load reliable (multi-barcode) variants, build CDS + protein, group by protein."""
    from Bio.Seq import Seq

    with open(ref_file) as f:
        wt_cds = "".join(l.strip() for l in f if not l.startswith(">"))
    wt_protein = str(Seq(wt_cds).translate()).rstrip("*")
    logger.info(f"WT CDS {len(wt_cds)} nt, WT protein {len(wt_protein)} aa")

    df = pd.read_csv(nt_file, sep="\t")
    df = df[df["uniqueBarcodes"] >= min_barcodes].copy()
    df = df.dropna(subset=["medianBrightness"])
    logger.info(f"Variants with >={min_barcodes} barcodes: {len(df)}")

    rows = []
    for _, row in df.iterrows():
        nt_muts = row["nMutations"]
        cds = list(wt_cds)
        if pd.isna(nt_muts) or nt_muts == "":
            n_mut = 0
        else:
            muts = str(nt_muts).split(":")
            n_mut = len(muts)
            if n_mut > max_mutations:
                continue
            for m in muts:
                mt = re.match(r"S([ACGT])(\d+)([ACGT])", m)
                if mt:
                    p = int(mt.group(2)) - 1
                    if p < len(cds):
                        cds[p] = mt.group(3)
        var_cds = "".join(cds)
        try:
            var_prot = str(Seq(var_cds).translate()).rstrip("*")
        except Exception:
            continue
        if "*" in var_prot:
            continue
        rows.append({
            "cds": var_cds, "protein": var_prot,
            "brightness": row["medianBrightness"],
            "meas_std": row["std"], "n_nt_mut": n_mut,
            "synonymous": var_prot == wt_protein,
        })
    out = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(f"Usable: {len(out)}, distinct proteins: {out['protein'].nunique()}, "
                f"synonymous: {out['synonymous'].sum()}")
    return out, wt_cds, wt_protein


@torch.no_grad()
def extract_esm2(seqs, device, batch_size=16):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch_size])]
        _, _, toks = bc(batch)
        toks = toks.to(device)
        reps = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        for j, (_, s) in enumerate(batch):
            out.append(reps[j, 1:len(s) + 1].mean(0).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info(f"  ESM-2 {i + len(batch)}/{len(seqs)}")
    del model
    torch.cuda.empty_cache()
    return np.array(out)


@torch.no_grad()
def extract_nt(seqs, device, batch_size=16):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    mid = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    local = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         f"models--{mid.replace('/', '--')}/snapshots/main")
    src = local if os.path.isdir(local) else mid
    logger.info(f"  NT from {src}")
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(src, trust_remote_code=True).to(device).eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        t = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        t = {k: v.to(device) for k, v in t.items()}
        h = model(**t, output_hidden_states=True).hidden_states[-1]
        mask = t["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)
        out.append(pooled.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info(f"  NT {i + len(batch)}/{len(seqs)}")
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, 0)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_predict(X_tr, y_tr, X_te, device, epochs=150):
    sc = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(sc.transform(X_tr), dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32, device=device)
    m = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        lf(m(Xtr), ytr).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(Xte).cpu().numpy()


def test_A_group_cv(reps, y, groups, device, n_splits=5):
    """GroupKFold by protein: pooled-prediction Spearman, no leakage."""
    logger.info("\n=== Test A: reliability filter + GroupKFold by protein ===")
    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    for name, X in reps.items():
        preds = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            preds[te] = _train_predict(X[tr], y[tr], X[te], device)
        rho, _ = spearmanr(y, preds)
        results[name] = rho
        logger.info(f"  {name:22s} ({X.shape[1]:4d}d): pooled Spearman ρ = {rho:.4f}")
    return results


def test_B_within_synonymous(reps, df, device, min_group=3, n_splits=5):
    """Smoking gun: within-synonymous-group prediction under GroupKFold.
    Protein embedding is constant within a group -> protein-only has 0 within-group
    predictive power by construction. Positive DNA-only => real codon-level signal."""
    logger.info(f"\n=== Test B: within-synonymous-group (>= {min_group} variants) ===")
    # A "synonymous group" = any protein with multiple distinct CDS variants
    # (multiple CDS -> identical protein). Within the group protein is constant.
    sizes = df.groupby("protein").size()
    keep = sizes[sizes >= min_group].index
    idx = df.index[df["protein"].isin(keep)].to_numpy()
    sub = df.loc[idx]
    groups = sub["protein"].values
    y = sub["brightness"].values
    logger.info(f"  synonymous groups: {len(keep)}, variants: {len(idx)}")
    logger.info(f"  within-group brightness std mean: "
                f"{sub.groupby('protein')['brightness'].std().mean():.4f}, "
                f"meas-noise mean: {sub.groupby('protein')['meas_std'].mean().mean():.4f}")

    n_splits = min(n_splits, len(keep))
    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    for name, Xfull in reps.items():
        X = Xfull[idx]
        preds = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            preds[te] = _train_predict(X[tr], y[tr], X[te], device)
        # within-group Spearman, averaged over groups
        per_group = []
        sub2 = sub.copy()
        sub2["pred"] = preds
        for prot, g in sub2.groupby("protein"):
            if g["pred"].std() < 1e-9 or g["brightness"].std() < 1e-9:
                continue  # constant prediction (protein-only) -> undefined
            r, _ = spearmanr(g["brightness"], g["pred"])
            if not np.isnan(r):
                per_group.append(r)
        mean_r = np.mean(per_group) if per_group else float("nan")
        results[name] = (mean_r, len(per_group))
        logger.info(f"  {name:22s}: mean within-group Spearman = {mean_r:.4f} "
                    f"(over {len(per_group)} groups with prediction variance)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt_file", required=True)
    ap.add_argument("--ref_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_barcodes", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, wt_cds, wt_prot = load_reliable(args.nt_file, args.ref_file, args.min_barcodes)

    cache = os.path.join(args.output_dir, f"clean_emb_bc{args.min_barcodes}.npz")
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if c["prot"].shape[0] == len(df):
            logger.info(f"Loaded cached embeddings {cache}")
            prot, dna = c["prot"], c["dna"]
        else:
            prot = dna = None
    else:
        prot = dna = None

    if prot is None:
        logger.info("Extracting ESM-2...")
        prot = extract_esm2(df["protein"].tolist(), args.device)
        logger.info(f"  protein {prot.shape}")
        logger.info("Extracting NT...")
        dna = extract_nt(df["cds"].tolist(), args.device)
        logger.info(f"  dna {dna.shape}")
        np.savez_compressed(cache, prot=prot, dna=dna)

    reps = {
        "Protein-only (ESM2)": prot,
        "DNA-only (NT)": dna,
        "Concat": np.hstack([prot, dna]),
    }
    y = df["brightness"].values
    groups = df["protein"].values

    resA = test_A_group_cv(reps, y, groups, args.device)
    resB = test_B_within_synonymous(reps, df, args.device)

    print("\n" + "=" * 72)
    print("CLEAN VALIDATION SUMMARY  (reliable >=%d barcodes, no leakage)" % args.min_barcodes)
    print("=" * 72)
    print("\nTest A -- GroupKFold by protein, pooled Spearman:")
    for k, v in sorted(resA.items(), key=lambda x: -x[1]):
        print(f"   {k:22s}: ρ = {v:.4f}")
    print("\nTest B -- within-synonymous-group Spearman (protein-only = 0 by construction):")
    for k, (r, n) in resB.items():
        print(f"   {k:22s}: ρ = {r:.4f}  (n={n} groups)")
    print("\nInterpretation:")
    print("  Test A: does cross-modal gain survive reliability+no-leakage?")
    print("  Test B: DNA-only > 0 => DNA LM captures codon fitness signal protein cannot.")

    pd.DataFrame([
        {"test": "A_groupcv", "rep": k, "spearman": v} for k, v in resA.items()
    ] + [
        {"test": "B_within_syn", "rep": k, "spearman": r, "n_groups": n}
        for k, (r, n) in resB.items()
    ]).to_csv(os.path.join(args.output_dir, "clean_results.csv"), index=False)


if __name__ == "__main__":
    main()
