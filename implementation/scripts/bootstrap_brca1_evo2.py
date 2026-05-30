"""
Does the cross-modal margin survive error bars?

We have out-of-fold predictions for the key BRCA1 configs. The question that
0.9125 vs 0.8886 cannot answer on its own: is the +0.024 real, or noise from a
single split? Here we put a 95% CI on the *difference* in auROC via a paired
CLUSTER bootstrap over genomic positions (resample groups, not rows, because
variants at the same position are non-independent and our CV blocks by position).

Compared (all + coding/noncoding/missense strata):
  ESM+Evo2-LLR  vs  Evo2 zero-shot      (the headline: do we beat Evo2)
  ESM+Evo2-LLR  vs  Evo2-LLR supervised (same paradigm, fair)
  ESM+Evo2-LLR  vs  ESM-only            (does the DNA model add anything)

A margin is "real" if its 95% CI excludes 0.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_brca1_fusion import grouped_auroc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_BOOT = 2000
SEED = 0


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def main():
    out_dir = "results/variant"
    df = pd.read_csv("data/variant/brca1/brca1_variants.csv").reset_index(drop=True)
    y = df["label"].values
    groups = df["pos_hg19"].values

    ev = np.load(os.path.join(out_dir, "brca1_evo2.npz"))
    llr = ev["llr"]
    llr_imp = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)

    c = np.load(os.path.join(out_dir, "brca1_esm_delta.npz"))
    pdelta, pmask = c["pdelta"], c["pmask"]
    pmask_f = pmask[:, None].astype(np.float32)

    # out-of-fold predictions per method (computed once on full data)
    logger.info("computing OOF predictions ...")
    preds = {
        "Evo2 zero-shot": -llr_imp,  # no training
        "Evo2-LLR sup": grouped_auroc(llr_imp[:, None], y, groups),
        "ESM-only": grouped_auroc(np.hstack([pdelta, pmask_f]), y, groups),
        "ESM+Evo2-LLR": grouped_auroc(np.hstack([pdelta, llr_imp[:, None], pmask_f]), y, groups),
    }

    strata = {
        "all": np.ones(len(df), bool),
        "coding": (df["vtype"] == "coding").values,
        "noncoding": (df["vtype"] == "noncoding").values,
        "missense": df["is_missense"].values,
    }

    # point estimates
    print("\n" + "=" * 70)
    print("Point auROC")
    print("=" * 70)
    for name, p in preds.items():
        line = f"  {name:16s}: " + "  ".join(
            f"{s}={safe_auc(y[m], p[m]):.4f}" for s, m in strata.items())
        print(line)

    # paired cluster bootstrap by genomic position
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)

    comparisons = [
        ("ESM+Evo2-LLR", "Evo2 zero-shot"),
        ("ESM+Evo2-LLR", "Evo2-LLR sup"),
        ("ESM+Evo2-LLR", "ESM-only"),
    ]
    # boot[(a,b)][stratum] = list of delta auROC
    boot = {cmp: {s: [] for s in strata} for cmp in comparisons}

    logger.info(f"bootstrap B={N_BOOT} (cluster by position, {len(uniq)} positions) ...")
    for b in range(N_BOOT):
        gsamp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gsamp])
        yb = y[idx]
        for (a, bn) in comparisons:
            pa, pb = preds[a][idx], preds[bn][idx]
            for s, mfull in strata.items():
                mb = mfull[idx]
                if mb.sum() < 10:
                    boot[(a, bn)][s].append(np.nan)
                    continue
                da = safe_auc(yb[mb], pa[mb]) - safe_auc(yb[mb], pb[mb])
                boot[(a, bn)][s].append(da)

    print("\n" + "=" * 70)
    print(f"Delta auROC with 95% CI (paired cluster bootstrap, B={N_BOOT})")
    print("'*' = CI excludes 0 (significant)")
    print("=" * 70)
    rows = []
    for (a, bn) in comparisons:
        print(f"\n  {a}  -  {bn}")
        for s in strata:
            arr = np.array(boot[(a, bn)][s], dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 100:
                print(f"    {s:10s}: n/a")
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            med = np.median(arr)
            p_gt0 = (arr > 0).mean()  # one-sided prob delta>0
            sig = "*" if lo > 0 else " "
            print(f"    {s:10s}: Δ={med:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {sig}"
                  f"  P(Δ>0)={p_gt0:.3f}")
            rows.append({"a": a, "b": bn, "stratum": s, "delta_med": med,
                         "ci_lo": lo, "ci_hi": hi, "p_gt0": p_gt0,
                         "significant": lo > 0})

    out = os.path.join(out_dir, "brca1_evo2_bootstrap.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
