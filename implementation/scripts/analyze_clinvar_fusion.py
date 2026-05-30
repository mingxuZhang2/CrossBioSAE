"""
Genome-wide test: does ESM-1b (+) Evo2 beat Evo2 alone on the songlab ClinVar
benchmark? Merges the 8 Evo2 LLR shards, fuses with the precomputed ESM-1b
scalar, and puts a bootstrap CI on the cross-modal margin -- the multi-gene
generalization of the BRCA1 result (0.913 vs 0.889, CI excluded 0).

The benchmark ships precomputed scalar scores; we add Evo2. All single-score
baselines are zero-shot (sign-oriented to AUC>0.5 once, globally). The fusion is
a tiny logistic head over [ESM-1b, Evo2-LLR, ESM-missing-mask], grouped-CV by
genomic position so no position leaks across folds.

Strata: coding/missense == ESM-1b available; noncoding == ESM-1b NaN.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT = "results/variant"
N_BOOT = 2000
SEED = 0


def oriented_auc(y, s):
    """AUC of a scalar score, oriented to its better global direction (sign is a
    fixed property of the score, not learned per-sample)."""
    m = ~np.isnan(s)
    if len(np.unique(y[m])) < 2:
        return np.nan, np.ones_like(s)
    a = roc_auc_score(y[m], s[m])
    if a < 0.5:
        return 1 - a, -s
    return a, s


def grouped_pred(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return pred


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def main():
    df = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)
    y = df["label"].astype(int).values
    groups = (df["chrom"] + ":" + df["pos"].astype(str)).values

    # --- merge Evo2 shards ---
    llr = np.full(len(df), np.nan, dtype=np.float32)
    nfound = 0
    for s in range(8):
        f = os.path.join(OUT, f"clinvar_evo2_llr_shard{s}.npz")
        if not os.path.exists(f):
            logger.warning(f"MISSING shard {s} ({f})")
            continue
        d = np.load(f)
        llr[d["idx"]] = d["llr"]
        nfound += len(d["idx"])
    logger.info(f"merged Evo2 LLR: {np.isfinite(llr).sum()} of {len(df)} variants "
                f"({nfound} from shards)")
    np.savez_compressed(os.path.join(OUT, "clinvar_evo2_llr.npz"), llr=llr)

    # only evaluate variants with an Evo2 score
    have = np.isfinite(llr)
    df, y, groups, llr = df[have].reset_index(drop=True), y[have], groups[have], llr[have]
    esm = df["ESM-1b"].values.astype(float)
    logger.info(f"evaluating on {len(df)} variants; ESM-1b present {np.isfinite(esm).sum()}")

    esm_mask = np.isfinite(esm)
    strata = {
        "all": np.ones(len(df), bool),
        "coding(ESM)": esm_mask,
        "noncoding": ~esm_mask,
    }

    # --- single-score zero-shot baselines (oriented) ---
    print("\n" + "=" * 72)
    print(f"Zero-shot single-score auROC  (n={len(df)} genome-wide ClinVar)")
    print("=" * 72)
    singles = {}
    for col in ["ESM-1b", "GPN-MSA", "CADD", "phyloP-100v", "phyloP-241m",
                "phastCons-100v", "NT", "HyenaDNA"]:
        if col not in df.columns:
            continue
        a, _ = oriented_auc(y, df[col].values.astype(float))
        singles[col] = a
    evo_a, evo_or = oriented_auc(y, llr)        # oriented Evo2 score
    singles["Evo2 (ours)"] = evo_a
    for k, v in sorted(singles.items(), key=lambda x: -(x[1] if np.isfinite(x[1]) else -1)):
        print(f"  {k:16s}: {v:.4f}")

    # --- fusion features ---
    esm_imp = np.where(esm_mask, esm, np.nanmedian(esm[esm_mask])).astype(np.float32)
    mask_f = esm_mask.astype(np.float32)[:, None]
    evo_imp = llr.astype(np.float32)
    feats = {
        "Evo2 sup": evo_imp[:, None],
        "ESM sup": np.hstack([esm_imp[:, None], mask_f]),
        "ESM+Evo2": np.hstack([esm_imp[:, None], evo_imp[:, None], mask_f]),
    }
    preds = {"Evo2 zero-shot": evo_or}
    logger.info("fitting grouped-CV fusion heads ...")
    for name, X in feats.items():
        preds[name] = grouped_pred(X, y, groups)

    print("\n" + "=" * 72)
    print("Fusion vs Evo2  (auROC by stratum)")
    print("=" * 72)
    for name, p in preds.items():
        line = f"  {name:16s}: " + "  ".join(
            f"{s}={safe_auc(y[m], p[m]):.4f}" for s, m in strata.items())
        print(line)

    # --- paired cluster bootstrap by position ---
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    comparisons = [("ESM+Evo2", "Evo2 zero-shot"),
                   ("ESM+Evo2", "Evo2 sup"),
                   ("ESM+Evo2", "ESM sup")]
    boot = {c: {s: [] for s in strata} for c in comparisons}
    logger.info(f"bootstrap B={N_BOOT} (cluster by position, {len(uniq)} positions) ...")
    for b in range(N_BOOT):
        gsamp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gsamp])
        yb = y[idx]
        for (a, bn) in comparisons:
            pa, pb = preds[a][idx], preds[bn][idx]
            for s, mfull in strata.items():
                mb = mfull[idx]
                if mb.sum() < 20 or len(np.unique(yb[mb])) < 2:
                    boot[(a, bn)][s].append(np.nan); continue
                boot[(a, bn)][s].append(
                    roc_auc_score(yb[mb], pa[mb]) - roc_auc_score(yb[mb], pb[mb]))

    print("\n" + "=" * 72)
    print(f"Delta auROC 95% CI (paired cluster bootstrap, B={N_BOOT}); '*'=excludes 0")
    print("=" * 72)
    rows = []
    for (a, bn) in comparisons:
        print(f"\n  {a}  -  {bn}")
        for s in strata:
            arr = np.array(boot[(a, bn)][s], float); arr = arr[~np.isnan(arr)]
            if len(arr) < 100:
                print(f"    {s:12s}: n/a"); continue
            lo, hi = np.percentile(arr, [2.5, 97.5]); med = np.median(arr)
            sig = "*" if lo > 0 else " "
            print(f"    {s:12s}: Δ={med:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {sig}"
                  f"  P(Δ>0)={(arr > 0).mean():.3f}")
            rows.append({"a": a, "b": bn, "stratum": s, "delta_med": med,
                         "ci_lo": lo, "ci_hi": hi, "significant": lo > 0})

    pd.DataFrame([{"score": k, "auroc": v} for k, v in singles.items()]).to_csv(
        os.path.join(OUT, "clinvar_singles.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "clinvar_fusion_bootstrap.csv"), index=False)
    print(f"\nsaved clinvar_singles.csv + clinvar_fusion_bootstrap.csv")


if __name__ == "__main__":
    main()
