"""
BRCA1 variant effect from the CROSS-MODAL SHARED EMBEDDING -- the project's
actual method (CrossBioSAE / CCA-aligned shared space), NOT scalar-score fusion.

For each variant we have:
  - ESM-2 per-residue delta  (1280-d protein embedding)   pdelta
  - Evo2-7b window-token delta (4096-d DNA embedding)      edelta
We learn a CCA shared space between the two modalities (fit per train fold, so
no label/position leakage), project both modalities in, and FUSE in that shared
space. The fused representation is the classification feature.

Representations compared (LogisticRegression, GroupKFold by genomic position):
  - ESM-emb         : 1280-d protein only
  - Evo2-emb        : 4096-d DNA only
  - Raw-Concat      : concat(1280, 4096)  -- naive fusion, no shared space
  - CCA-SHARED      : mean(cca_prot, cca_dna)   <-- the shared embedding
  - CCA-FUSED       : concat(cca_prot, cca_dna)
A missense-availability mask is appended (ESM delta is only meaningful at the
mutated residue).

The question: does the SHARED-EMBEDDING fusion beat each single modality (and
the naive concat)?  Bootstrap CI on the margin.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from alignment import CCAAlignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("alignment").setLevel(logging.WARNING)

N_BOOT = 2000
SEED = 0
CCA_K = 50
PCA_DIM = 256


def build_rep(kind, p_tr, d_tr, p_te, d_te):
    """Return (X_train, X_test) for a representation, fitting any CCA on TRAIN only."""
    if kind == "ESM-emb":
        return p_tr, p_te
    if kind == "Evo2-emb":
        return d_tr, d_te
    if kind == "Raw-Concat":
        return np.hstack([p_tr, d_tr]), np.hstack([p_te, d_te])
    # shared-embedding variants: fit CCA on train pairs
    cca = CCAAlignment(n_components=CCA_K, pca_dim=PCA_DIM)
    cca.fit(p_tr, d_tr)
    cp_tr, cd_tr = cca.transform(p_tr, d_tr)
    cp_te, cd_te = cca.transform(p_te, d_te)
    if kind == "CCA-SHARED":
        return 0.5 * (cp_tr + cd_tr), 0.5 * (cp_te + cd_te)
    if kind == "CCA-FUSED":
        return np.hstack([cp_tr, cd_tr]), np.hstack([cp_te, cd_te])
    raise ValueError(kind)


def oof_predictions(kind, pdelta, edelta, mask_f, y, groups, n_splits=5):
    """Out-of-fold predictions; CCA + scaler + classifier all fit per train fold."""
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for tr, te in gkf.split(pdelta, y, groups):
        Xtr, Xte = build_rep(kind, pdelta[tr], edelta[tr], pdelta[te], edelta[te])
        # append the missense-availability mask
        Xtr = np.hstack([Xtr, mask_f[tr]])
        Xte = np.hstack([Xte, mask_f[te]])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xtr), y[tr])
        pred[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return pred


def strat(y, p, df):
    out = {"all": roc_auc_score(y, p)}
    for vt in ["coding", "noncoding"]:
        m = (df["vtype"] == vt).values
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            out[vt] = roc_auc_score(y[m], p[m])
    mis = df["is_missense"].values
    if len(np.unique(y[mis])) == 2:
        out["missense"] = roc_auc_score(y[mis], p[mis])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--esm", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--evo2", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--group_col", default="pos_hg19",
                    help="variant-table column used for GroupKFold blocking")
    ap.add_argument("--run_name", default="BRCA1",
                    help="label used in console output")
    ap.add_argument("--output_prefix", default="brca1",
                    help="prefix for saved result files")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    y = df["label"].values
    if args.group_col not in df.columns:
        raise ValueError(f"group_col {args.group_col!r} not found in {args.variants}")
    groups = df[args.group_col].values

    pd_ = np.load(args.esm)
    pdelta, pmask = pd_["pdelta"].astype(np.float32), pd_["pmask"]
    edelta = np.load(args.evo2)["edelta"].astype(np.float32)
    mask_f = pmask[:, None].astype(np.float32)
    logger.info(f"ESM {pdelta.shape}  Evo2 {edelta.shape}  n={len(df)}")

    reps = ["ESM-emb", "Evo2-emb", "Raw-Concat", "CCA-SHARED", "CCA-FUSED"]
    preds = {}
    print("\n" + "=" * 76)
    print(f"{args.run_name} variant effect from cross-modal SHARED EMBEDDING (auROC, GroupKFold)")
    print("=" * 76)
    for kind in reps:
        p = oof_predictions(kind, pdelta, edelta, mask_f, y, groups)
        preds[kind] = p
        s = strat(y, p, df)
        print(f"  {kind:12s}: " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))

    # bootstrap CI: shared embedding vs each single modality
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    strata = {"all": np.ones(len(df), bool),
              "coding": (df["vtype"] == "coding").values,
              "noncoding": (df["vtype"] == "noncoding").values,
              "missense": df["is_missense"].values}
    comparisons = [("CCA-SHARED", "Evo2-emb"), ("CCA-SHARED", "ESM-emb"),
                   ("CCA-SHARED", "Raw-Concat"), ("CCA-FUSED", "Evo2-emb")]
    boot = {c: {s: [] for s in strata} for c in comparisons}
    for b in range(N_BOOT):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gs])
        yb = y[idx]
        for (a, bn) in comparisons:
            pa, pb = preds[a][idx], preds[bn][idx]
            for s, mf in strata.items():
                mb = mf[idx]
                if mb.sum() < 20 or len(np.unique(yb[mb])) < 2:
                    boot[(a, bn)][s].append(np.nan); continue
                boot[(a, bn)][s].append(
                    roc_auc_score(yb[mb], pa[mb]) - roc_auc_score(yb[mb], pb[mb]))

    print("\n" + "=" * 76)
    print(f"Delta auROC 95% CI (paired cluster bootstrap, B={N_BOOT}); '*'=excludes 0")
    print("=" * 76)
    rows = []
    for (a, bn) in comparisons:
        print(f"\n  {a}  -  {bn}")
        for s in strata:
            arr = np.array(boot[(a, bn)][s], float); arr = arr[~np.isnan(arr)]
            if len(arr) < 100:
                print(f"    {s:10s}: n/a"); continue
            lo, hi = np.percentile(arr, [2.5, 97.5]); med = np.median(arr)
            sig = "*" if lo > 0 else " "
            print(f"    {s:10s}: Δ={med:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {sig}")
            rows.append({"a": a, "b": bn, "stratum": s, "delta": med,
                         "ci_lo": lo, "ci_hi": hi, "sig": lo > 0})
    out = os.path.join(args.output_dir, f"{args.output_prefix}_shared_embedding_bootstrap.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
