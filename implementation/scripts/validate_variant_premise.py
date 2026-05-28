"""
Premise validation for cross-modal variant effect prediction, using the songlab
precomputed-score benchmark (no model runs needed).

Thesis: protein-LM (ESM) scores missense well but CANNOT score noncoding/splice;
DNA-LM (GPN-MSA / NT) covers noncoding. A fused predictor should beat either alone
across the full variant spectrum.

clinvar.parquet : coding/missense (has ESM-1b, GPN-MSA, NT, CADD, ...)
omim.parquet    : noncoding/regulatory Mendelian (has consequence, GPN-MSA, NT, CADD; NO ESM)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

DDIR = "data/variant"


def oriented_auroc(y, s):
    """auROC, auto-oriented (score sign convention unknown). NaN-safe."""
    m = ~pd.isna(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    a = roc_auc_score(y[m], s[m])
    return max(a, 1 - a), int(m.sum())


def per_score_table(df, scores, name):
    print(f"\n=== {name} (n={len(df)}, positives={int(df['label'].sum())}) ===")
    y = df["label"].astype(int).values
    for s in scores:
        if s not in df.columns:
            continue
        au, n = oriented_auroc(y, df[s].values)
        cov = 100 * n / len(df)
        print(f"  {s:14s}: auROC={au:.4f}  (scored {n}/{len(df)} = {cov:.0f}%)")


def main():
    cv = pd.read_parquet(f"{DDIR}/clinvar.parquet")
    om = pd.read_parquet(f"{DDIR}/omim.parquet")

    print("clinvar label balance:", cv["label"].value_counts().to_dict())
    print("omim label balance:", om["label"].value_counts().to_dict())
    if "consequence" in om.columns:
        print("omim consequence:", om["consequence"].value_counts().head(10).to_dict())
    print("clinvar ESM-1b coverage:", int(cv["ESM-1b"].notna().sum()), "/", len(cv))

    base = ["ESM-1b", "GPN-MSA", "NT", "CADD", "phyloP-100v", "HyenaDNA"]
    per_score_table(cv, base, "ClinVar (coding/missense)")
    per_score_table(om, ["GPN-MSA", "NT", "CADD", "phyloP-100v"], "OMIM (noncoding)")

    # ---- Complementarity on the MISSENSE subset: does protein beat DNA? ----
    miss = cv[cv["ESM-1b"].notna()].copy()
    print(f"\n=== MISSENSE-only subset (ESM scored), n={len(miss)} "
          f"pos={int(miss['label'].sum())} ===")
    y = miss["label"].astype(int).values
    for s in ["ESM-1b", "GPN-MSA", "NT", "CADD"]:
        au, _ = oriented_auroc(y, miss[s].values)
        print(f"  {s:10s}: auROC={au:.4f}")

    # orient scores so higher=pathogenic, then z-normalize for fusion
    def orient(df, s):
        v = df[s].astype(float).values
        yy = df["label"].astype(int).values
        m = ~np.isnan(v)
        if roc_auc_score(yy[m], v[m]) < 0.5:
            v = -v
        return v

    e = orient(miss, "ESM-1b")
    g = orient(miss, "GPN-MSA")
    def z(v):
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    X = np.column_stack([z(e), z(g)])
    lr = LogisticRegression(max_iter=1000)
    pf = cross_val_predict(lr, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"  Fused(ESM+GPN) 5cv: auROC={roc_auc_score(y, pf):.4f}")

    # ---- Full-spectrum: protein-only cannot cover noncoding ----
    print("\n=== FULL SPECTRUM (missense ClinVar + noncoding OMIM) ===")
    # balance omim negatives down to keep it comparable & fast
    om_pos = om[om["label"]]
    om_neg = om[~om["label"]].sample(min(len(om_pos) * 5, (~om["label"]).sum()), random_state=0)
    om_bal = pd.concat([om_pos, om_neg])
    full = pd.concat([
        miss[["label", "ESM-1b", "GPN-MSA"]].assign(kind="missense"),
        om_bal[["label", "GPN-MSA"]].assign(kind="noncoding", **{"ESM-1b": np.nan}),
    ], ignore_index=True)
    yf = full["label"].astype(int).values
    # protein-only: ESM where available, else worst-rank (cannot score)
    esm = full["ESM-1b"].astype(float).values
    esm_o = esm.copy()
    mm = ~np.isnan(esm_o)
    if roc_auc_score(yf[mm], esm_o[mm]) < 0.5:
        esm_o = -esm_o
    gpn = orient(full.assign(label=full["label"]), "GPN-MSA")
    # protein-only: impute missing (noncoding) with min => predicted benign (its blind spot)
    esm_impute = np.where(np.isnan(esm_o), np.nanmin(esm_o), esm_o)
    print(f"  Protein-only (ESM, noncoding blind): auROC={roc_auc_score(yf, esm_impute):.4f}")
    print(f"  DNA-only (GPN-MSA, all):             auROC={roc_auc_score(yf, gpn):.4f}")
    # fused: ESM on missense, GPN on noncoding (route by availability)
    fused = np.where(np.isnan(esm_o), z(gpn), z(esm_o))  # simple routed score
    # better: per-stratum z then combine
    print(f"  Fused (route ESM/GPN by type):       auROC={roc_auc_score(yf, fused):.4f}")
    print("\n  => If fused > both singles, cross-modal premise holds for variant effect.")


if __name__ == "__main__":
    main()
