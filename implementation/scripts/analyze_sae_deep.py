"""
Deep interpretability of genome-wide SAE features.

Three analyses:
1. Concept alignment — correlate each SAE feature with biological scores
2. Modality attribution — is each feature driven by protein, DNA, or both?
3. Fusion insight — what new information lives in the shared space?
"""

import numpy as np
import pandas as pd
from scipy import stats
import os, sys, warnings
warnings.filterwarnings("ignore")

RESULTS = "results/sae_genomewide"
PARQUET = "data/variant/clinvar.parquet"

# ── 1. Load data ──────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_parquet(PARQUET).reset_index(drop=True)
acts = np.load(os.path.join(RESULTS, "sae_genomewide_acts.npz"))["acts"]  # (40976, 2048)
cards = pd.read_csv(os.path.join(RESULTS, "sae_genomewide_cards.csv"))

y = df["label"].astype(int).values
is_coding = df["ESM-1b"].notna().values
print(f"  {len(df)} variants, {is_coding.sum()} coding, {(~is_coding).sum()} noncoding")
print(f"  SAE: {acts.shape}, alive features with n≥10: {len(cards)}")

# ── 2. Score direction check ─────────────────────────────────────────
# Verify which direction each score associates with pathogenicity
print(f"\n{'='*70}")
print("Score direction verification (mean for pathogenic vs benign)")
print(f"{'='*70}")
score_cols = ["ESM-1b", "GPN-MSA", "CADD", "phyloP-100v", "phyloP-241m",
              "phastCons-100v", "NT", "HyenaDNA"]
for col in score_cols:
    vals = df[col].values.astype(float)
    ok = np.isfinite(vals)
    if ok.sum() < 100:
        print(f"  {col:15s}: too few valid values ({ok.sum()})")
        continue
    m_path = np.nanmean(vals[y == 1])
    m_ben = np.nanmean(vals[y == 0])
    dirn = "↓ = pathogenic" if m_path < m_ben else "↑ = pathogenic"
    print(f"  {col:15s}: path={m_path:+.3f}  ben={m_ben:+.3f}  ({dirn})")

# ── 3. Per-feature correlation with biological scores ────────────────
print(f"\n{'='*70}")
print("Computing per-feature Spearman correlations ...")
print(f"{'='*70}")

alive_idx = cards["feature"].values
n_feat = len(alive_idx)

corr_matrix = np.full((n_feat, len(score_cols)), np.nan)
pval_matrix = np.full((n_feat, len(score_cols)), np.nan)

for j, col in enumerate(score_cols):
    vals = df[col].values.astype(float)
    ok = np.isfinite(vals)
    if ok.sum() < 100:
        continue
    for i, fi in enumerate(alive_idx):
        a = acts[:, fi]
        active = np.abs(a) > 1e-6
        both = active & ok
        if both.sum() < 20:
            continue
        rho, pv = stats.spearmanr(a[both], vals[both])
        corr_matrix[i, j] = rho
        pval_matrix[i, j] = pv

corr_df = pd.DataFrame(corr_matrix, columns=score_cols)
corr_df["feature"] = alive_idx

# ── 4. Modality attribution ─────────────────────────────────────────
# ESM-1b = protein signal proxy, NT/GPN-MSA = DNA signal proxies
print(f"\n{'='*70}")
print("Modality attribution")
print(f"{'='*70}")

esm_corr = np.abs(corr_df["ESM-1b"].values)
nt_corr = np.abs(corr_df["NT"].values)
gpn_corr = np.abs(corr_df["GPN-MSA"].values)
dna_corr = np.maximum(nt_corr, gpn_corr)  # best DNA proxy

THRESH = 0.15  # correlation threshold for "significant" association

modality = []
for i in range(n_feat):
    e = esm_corr[i] if np.isfinite(esm_corr[i]) else 0
    d = dna_corr[i] if np.isfinite(dna_corr[i]) else 0
    if e >= THRESH and d >= THRESH:
        modality.append("cross-modal")
    elif e >= THRESH and d < THRESH:
        modality.append("protein-driven")
    elif d >= THRESH and e < THRESH:
        modality.append("dna-driven")
    else:
        modality.append("novel")

cards_ext = cards.copy()
cards_ext["modality"] = modality
cards_ext["corr_ESM1b"] = corr_df["ESM-1b"].values
cards_ext["corr_NT"] = corr_df["NT"].values
cards_ext["corr_GPN"] = corr_df["GPN-MSA"].values
cards_ext["corr_CADD"] = corr_df["CADD"].values
cards_ext["corr_phyloP"] = corr_df["phyloP-100v"].values

modality_counts = pd.Series(modality).value_counts()
print("\nModality distribution:")
for m, c in modality_counts.items():
    sub = cards_ext[cards_ext["modality"] == m]
    print(f"  {m:20s}: {c:5d} features  "
          f"(mean path_rate={sub['path_rate'].mean():.3f}, "
          f"mean |rESM|={np.nanmean(np.abs(sub['corr_ESM1b'])):.3f}, "
          f"mean |rNT|={np.nanmean(np.abs(sub['corr_NT'])):.3f})")

# ── 5. Refined concept labels ───────────────────────────────────────
print(f"\n{'='*70}")
print("Refined biological concept labels")
print(f"{'='*70}")

def assign_concept(row):
    pr = row["path_rate"]
    cod = row["esm_coding_frac"]
    mod = row["modality"]
    r_esm = row["corr_ESM1b"] if np.isfinite(row["corr_ESM1b"]) else 0
    r_nt = row["corr_NT"] if np.isfinite(row["corr_NT"]) else 0
    r_gpn = row["corr_GPN"] if np.isfinite(row["corr_GPN"]) else 0

    # pathogenic features
    if pr > 0.8:
        if cod < 0.3:
            return "noncoding-pathogenic"
        if r_esm < -0.2:  # strong negative = more activation on low ESM = damaging protein
            return "protein-damaging"
        if r_gpn < -0.2:
            return "conservation-driven-pathogenic"
        return "pathogenic-general"

    # benign features
    if pr < 0.15:
        if cod < 0.3:
            return "noncoding-benign"
        if r_esm > 0.15:
            return "protein-tolerant"
        if r_gpn > 0.15:
            return "low-conservation-benign"
        return "benign-general"

    # mixed features — this is the bulk (0.15 ≤ pr ≤ 0.8)
    if mod == "cross-modal":
        return "cross-modal-mixed"
    if mod == "protein-driven":
        return "protein-signal-mixed"
    if mod == "dna-driven":
        return "dna-signal-mixed"

    # check if it captures coding/noncoding distinction
    if cod > 0.9:
        return "coding-mixed"
    if cod < 0.3:
        return "noncoding-mixed"

    # truly ambiguous
    return "unresolved"

cards_ext["concept_v2"] = cards_ext.apply(assign_concept, axis=1)
concept_counts = cards_ext["concept_v2"].value_counts()
print("\nRefined concept distribution:")
for concept, count in concept_counts.items():
    sub = cards_ext[cards_ext["concept_v2"] == concept]
    print(f"  {concept:30s}: {count:5d} feats  "
          f"path={sub['path_rate'].mean():.3f}  "
          f"cod={sub['esm_coding_frac'].mean():.2f}  "
          f"rESM={sub['corr_ESM1b'].mean():+.3f}  "
          f"rNT={sub['corr_NT'].mean():+.3f}  "
          f"rGPN={sub['corr_GPN'].mean():+.3f}")

# ── 6. Modality decomposition per concept ────────────────────────────
print(f"\n{'='*70}")
print("Modality × Concept cross-tabulation")
print(f"{'='*70}")

ct = pd.crosstab(cards_ext["concept_v2"], cards_ext["modality"])
print(ct.to_string())

# ── 7. Top features per modality ─────────────────────────────────────
print(f"\n{'='*70}")
print("Top features by modality (strongest signal)")
print(f"{'='*70}")

for mod_name in ["protein-driven", "dna-driven", "cross-modal", "novel"]:
    sub = cards_ext[cards_ext["modality"] == mod_name].copy()
    if len(sub) == 0:
        continue

    # sort by strongest signal
    if mod_name == "protein-driven":
        sub = sub.reindex(sub["corr_ESM1b"].abs().sort_values(ascending=False).index)
    elif mod_name == "dna-driven":
        sub["max_dna_corr"] = sub[["corr_NT", "corr_GPN"]].abs().max(axis=1)
        sub = sub.sort_values("max_dna_corr", ascending=False)
    elif mod_name == "cross-modal":
        sub["min_corr"] = sub[["corr_ESM1b", "corr_NT", "corr_GPN"]].abs().min(axis=1)
        sub = sub.sort_values("min_corr", ascending=False)
    else:  # novel
        sub = sub.sort_values("n_active", ascending=False)

    print(f"\n  ── {mod_name} (top 10 of {len(sub)}) ──")
    print(f"  {'feat':>5s} {'n':>6s} {'path%':>6s} {'cod%':>5s} "
          f"{'rESM':>6s} {'rNT':>6s} {'rGPN':>6s} {'rCADD':>6s} concept_v2")
    for _, r in sub.head(10).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['path_rate']:6.2f} "
              f"{r['esm_coding_frac']:5.2f} "
              f"{r['corr_ESM1b']:+6.3f} {r['corr_NT']:+6.3f} {r['corr_GPN']:+6.3f} "
              f"{r['corr_CADD']:+6.3f} {r['concept_v2']}")

# ── 8. Coding vs Noncoding feature behavior ──────────────────────────
print(f"\n{'='*70}")
print("Coding vs Noncoding variant behavior")
print(f"{'='*70}")

# for each feature, compute separate path_rate in coding vs noncoding
cod_stats = []
for i, fi in enumerate(alive_idx):
    a = acts[:, fi]
    active = np.abs(a) > 1e-6
    n_act = active.sum()
    if n_act < 20:
        continue
    n_cod = (active & is_coding).sum()
    n_noncod = (active & ~is_coding).sum()
    if n_cod > 5:
        pr_cod = y[active & is_coding].mean()
    else:
        pr_cod = np.nan
    if n_noncod > 5:
        pr_noncod = y[active & ~is_coding].mean()
    else:
        pr_noncod = np.nan
    cod_stats.append({
        "feature": fi,
        "n_coding": n_cod,
        "n_noncoding": n_noncod,
        "path_rate_coding": pr_cod,
        "path_rate_noncoding": pr_noncod,
        "coding_frac": n_cod / n_act,
    })

cod_df = pd.DataFrame(cod_stats)
both_valid = cod_df.dropna(subset=["path_rate_coding", "path_rate_noncoding"])
print(f"\n  Features active in both coding & noncoding (n≥5 each): {len(both_valid)}")

if len(both_valid) > 0:
    more_path_cod = (both_valid["path_rate_coding"] > both_valid["path_rate_noncoding"]).sum()
    more_path_noncod = (both_valid["path_rate_noncoding"] > both_valid["path_rate_coding"]).sum()
    print(f"  More pathogenic in coding: {more_path_cod}")
    print(f"  More pathogenic in noncoding: {more_path_noncod}")
    delta = both_valid["path_rate_coding"] - both_valid["path_rate_noncoding"]
    print(f"  Mean Δ(path_rate, coding - noncoding): {delta.mean():+.3f}")

# ── 9. Novel features analysis ──────────────────────────────────────
print(f"\n{'='*70}")
print("Novel features — not explained by any existing score")
print(f"{'='*70}")

novel = cards_ext[cards_ext["modality"] == "novel"].copy()
print(f"\n  {len(novel)} novel features (|r| < {THRESH} with ESM-1b, NT, and GPN-MSA)")

if len(novel) > 0:
    # check if novel features still correlate with pathogenicity
    novel_path = novel[novel["path_rate"] > 0.7]
    novel_ben = novel[novel["path_rate"] < 0.2]
    novel_mid = novel[(novel["path_rate"] >= 0.2) & (novel["path_rate"] <= 0.7)]
    print(f"  Pathogenic (pr > 0.7): {len(novel_path)}")
    print(f"  Benign (pr < 0.2): {len(novel_ben)}")
    print(f"  Middle (0.2-0.7): {len(novel_mid)}")

    if len(novel_path) > 0:
        print(f"\n  Novel pathogenic features (predict pathogenicity without known score signal):")
        for _, r in novel_path.sort_values("path_rate", ascending=False).head(5).iterrows():
            print(f"    feat {int(r['feature']):5d}: n={int(r['n_active']):5d} path={r['path_rate']:.2f} "
                  f"cod={r['esm_coding_frac']:.2f} "
                  f"rESM={r['corr_ESM1b']:+.3f} rNT={r['corr_NT']:+.3f}")

# ── 10. Correlation heatmap summary ─────────────────────────────────
print(f"\n{'='*70}")
print("Feature-score correlation summary (mean |r| across all alive features)")
print(f"{'='*70}")
for j, col in enumerate(score_cols):
    valid = np.isfinite(corr_matrix[:, j])
    if valid.sum() > 0:
        mean_abs = np.nanmean(np.abs(corr_matrix[valid, j]))
        mean_r = np.nanmean(corr_matrix[valid, j])
        strong = (np.abs(corr_matrix[valid, j]) > 0.3).sum()
        print(f"  {col:15s}: mean|r|={mean_abs:.4f}  mean_r={mean_r:+.4f}  "
              f"strong(|r|>0.3)={strong:4d}/{valid.sum()}")

# ── 11. What information does the shared space encode? ───────────────
print(f"\n{'='*70}")
print("SUMMARY: What biological information does the shared space encode?")
print(f"{'='*70}")

# features that uniquely predict pathogenicity vs existing scores
# compare: feature activation AUC vs individual score AUC
from sklearn.metrics import roc_auc_score

print("\n  Individual score AUC for pathogenicity:")
for col in score_cols:
    vals = df[col].values.astype(float)
    ok = np.isfinite(vals) & np.isfinite(y.astype(float))
    if ok.sum() < 100:
        continue
    # some scores are "lower = pathogenic" so try both directions
    auc = roc_auc_score(y[ok], vals[ok])
    auc = max(auc, 1 - auc)  # flip if needed
    print(f"    {col:15s}: AUC={auc:.4f}  (n={ok.sum()})")

# top SAE feature AUCs
print("\n  Top 10 SAE features by pathogenicity AUC:")
feat_aucs = []
for i, fi in enumerate(alive_idx):
    a = acts[:, fi]
    active = np.abs(a) > 1e-6
    if active.sum() < 50:
        continue
    auc = roc_auc_score(y[active], a[active])
    auc = max(auc, 1 - auc)
    feat_aucs.append((fi, auc, active.sum(), cards_ext.iloc[i]["modality"]))

feat_aucs.sort(key=lambda x: -x[1])
for fi, auc, n, mod in feat_aucs[:10]:
    print(f"    feature {fi:5d}: AUC={auc:.4f}  n={n:6d}  modality={mod}")

# save extended cards
out_path = os.path.join(RESULTS, "sae_genomewide_cards_deep.csv")
cards_ext.to_csv(out_path, index=False)
print(f"\nSaved extended cards to {out_path}")

# save correlation matrix
corr_out = os.path.join(RESULTS, "sae_feature_score_correlations.csv")
corr_df.to_csv(corr_out, index=False)
print(f"Saved correlation matrix to {corr_out}")
