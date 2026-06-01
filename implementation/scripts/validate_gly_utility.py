"""
Validate utility of SAE novel features for collagen Gly-sub variant classification.

Core question: does novel_path_score add value beyond ESM-1b/GPN-MSA/CADD
for distinguishing pathogenic vs benign Gly substitutions in structural proteins?

Analyses:
1. AUC comparison: novel_path_score vs existing scores on collagen Gly-sub subset
2. Orthogonality: combined score (novel + ESM-1b) vs ESM-1b alone
3. Error analysis: which variants does novel_path_score get right that ESM-1b gets wrong?
4. Leave-one-gene-out: cross-gene generalization
5. Specificity: novel features vs raw SAE total activation (is novel-specific better?)
"""

import os, re, sys, gzip
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
import warnings
warnings.filterwarnings("ignore")

OUT = "results/gly_validation"
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
print("Loading data ...")
clinvar = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
acts = np.load("results/sae_genomewide/sae_genomewide_acts.npz")["acts"]
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")
y = clinvar["label"].astype(int).values

COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
}
STRUCTURAL_GENES = COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}

# novel pathogenic features
novel_path = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
novel_feat_ids = sorted(novel_path["feature"].astype(int).values)

# all feature groups for comparison
prot_feats = cards[cards["modality"] == "protein-driven"]["feature"].astype(int).values
dna_feats = cards[cards["modality"] == "dna-driven"]["feature"].astype(int).values
cross_feats = cards[cards["modality"] == "cross-modal"]["feature"].astype(int).values

# compute scores
novel_score = np.sum(np.abs(acts[:, novel_feat_ids]), axis=1)
prot_score = np.sum(np.abs(acts[:, prot_feats]), axis=1)
dna_score = np.sum(np.abs(acts[:, dna_feats]), axis=1)
cross_score = np.sum(np.abs(acts[:, cross_feats]), axis=1)
total_score = np.sum(np.abs(acts), axis=1)

clinvar["novel_path_score"] = novel_score
clinvar["prot_score"] = prot_score
clinvar["dna_score"] = dna_score
clinvar["cross_score"] = cross_score
clinvar["total_sae_score"] = total_score

# ── Map to genes and protein changes ─────────────────────────────
print("Mapping variants to genes ...")
clinvar_info = {}
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        key = (parts[18], parts[31], parts[32], parts[33])
        if key not in clinvar_info:
            name = parts[2]
            pchange = ""
            m = re.search(r'\(p\.([A-Za-z0-9_*=]+)\)', name)
            if m:
                pchange = m.group(1)
            clinvar_info[key] = {"gene": parts[4], "pchange": pchange}

clinvar["gene"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("gene")
    for r in clinvar.itertuples()]
clinvar["pchange"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("pchange", "")
    for r in clinvar.itertuples()]

def is_gly_sub(pc):
    if not pc or pd.isna(pc):
        return False
    m = re.match(r'Gly\d+([A-Z][a-z]{2})', str(pc))
    return m is not None and m.group(1) != "Gly"

clinvar["is_gly_sub"] = clinvar["pchange"].apply(is_gly_sub)
clinvar["is_structural"] = clinvar["gene"].isin(STRUCTURAL_GENES)
clinvar["is_collagen"] = clinvar["gene"].isin(COLLAGEN_GENES)

# ── Subset: structural protein Gly-sub variants ─────────────────
struct_gly = clinvar[clinvar["is_structural"] & clinvar["is_gly_sub"]].copy()
col_gly = clinvar[clinvar["is_collagen"] & clinvar["is_gly_sub"]].copy()
all_struct = clinvar[clinvar["is_structural"]].copy()

print(f"\n  Structural Gly-sub: {len(struct_gly)} "
      f"({struct_gly['label'].sum()} P, {(~struct_gly['label']).sum()} B)")
print(f"  Collagen Gly-sub:   {len(col_gly)} "
      f"({col_gly['label'].sum()} P, {(~col_gly['label']).sum()} B)")
print(f"  All structural:     {len(all_struct)} "
      f"({all_struct['label'].sum()} P, {(~all_struct['label']).sum()} B)")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: AUC comparison on collagen Gly-sub subset
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 1: PATHOGENICITY PREDICTION ON COLLAGEN Gly-SUB VARIANTS")
print(f"{'='*70}")

def eval_score(df, score_col, label_col="label"):
    vals = df[score_col].values.astype(float)
    labels = df[label_col].astype(int).values
    ok = np.isfinite(vals)
    if ok.sum() < 20 or len(np.unique(labels[ok])) < 2:
        return {"auc": np.nan, "auprc": np.nan, "n": ok.sum()}
    auc = roc_auc_score(labels[ok], vals[ok])
    if auc < 0.5:
        auc = 1 - auc
        vals = -vals
    auprc = average_precision_score(labels[ok], vals[ok])
    return {"auc": auc, "auprc": auprc, "n": int(ok.sum())}

for subset_name, subset_df in [("Collagen Gly-sub", col_gly),
                                 ("Structural Gly-sub", struct_gly),
                                 ("All structural", all_struct),
                                 ("All ClinVar", clinvar)]:
    n_p = subset_df["label"].sum()
    n_b = (~subset_df["label"]).sum()
    print(f"\n  ── {subset_name} (n={len(subset_df)}, {n_p}P/{n_b}B) ──")

    scores = {
        "novel_path_score": "novel_path_score",
        "ESM-1b": "ESM-1b",
        "GPN-MSA": "GPN-MSA",
        "CADD": "CADD",
        "total_sae_score": "total_sae_score",
        "prot_score": "prot_score",
        "dna_score": "dna_score",
    }
    for name, col in scores.items():
        if col not in subset_df.columns:
            continue
        r = eval_score(subset_df, col)
        if np.isnan(r["auc"]):
            continue
        print(f"    {name:25s}: AUC={r['auc']:.4f}  AUPRC={r['auprc']:.4f}  (n={r['n']})")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Does novel_path_score add value BEYOND ESM-1b?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 2: ADDED VALUE — novel_path_score + ESM-1b vs ESM-1b ALONE")
print(f"{'='*70}")

for subset_name, subset_df in [("Collagen Gly-sub", col_gly),
                                 ("Structural Gly-sub", struct_gly),
                                 ("All structural", all_struct)]:
    df = subset_df.dropna(subset=["ESM-1b"]).copy()
    if len(df) < 30 or df["label"].nunique() < 2:
        continue

    y_sub = df["label"].astype(int).values
    X_esm = df[["ESM-1b"]].values
    X_novel = df[["novel_path_score"]].values
    X_both = df[["ESM-1b", "novel_path_score"]].values

    # cross-validated logistic regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe_esm = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
    pipe_novel = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
    pipe_both = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])

    pred_esm = cross_val_predict(pipe_esm, X_esm, y_sub, cv=5, method="predict_proba")[:, 1]
    pred_novel = cross_val_predict(pipe_novel, X_novel, y_sub, cv=5, method="predict_proba")[:, 1]
    pred_both = cross_val_predict(pipe_both, X_both, y_sub, cv=5, method="predict_proba")[:, 1]

    auc_esm = roc_auc_score(y_sub, pred_esm)
    auc_novel = roc_auc_score(y_sub, pred_novel)
    auc_both = roc_auc_score(y_sub, pred_both)
    delta = auc_both - auc_esm

    # bootstrap CI for delta
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(2000):
        idx = rng.choice(len(y_sub), len(y_sub), replace=True)
        if len(np.unique(y_sub[idx])) < 2:
            continue
        a_e = roc_auc_score(y_sub[idx], pred_esm[idx])
        a_b = roc_auc_score(y_sub[idx], pred_both[idx])
        deltas.append(a_b - a_e)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_val = np.mean(np.array(deltas) <= 0)

    print(f"\n  ── {subset_name} (n={len(df)}) ──")
    print(f"    ESM-1b alone:       AUC = {auc_esm:.4f}")
    print(f"    novel_score alone:  AUC = {auc_novel:.4f}")
    print(f"    ESM-1b + novel:     AUC = {auc_both:.4f}")
    print(f"    Δ(both - ESM-1b):   {delta:+.4f}  95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    P(Δ≤0):             {p_val:.4f}")

    # also try with CADD
    df2 = subset_df.dropna(subset=["ESM-1b", "CADD"]).copy()
    if len(df2) > 30 and df2["label"].nunique() == 2:
        y2 = df2["label"].astype(int).values
        X_all3 = df2[["ESM-1b", "CADD", "novel_path_score"]].values
        X_esm_cadd = df2[["ESM-1b", "CADD"]].values

        pipe3 = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
        pipe2 = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])

        pred3 = cross_val_predict(pipe3, X_all3, y2, cv=5, method="predict_proba")[:, 1]
        pred2 = cross_val_predict(pipe2, X_esm_cadd, y2, cv=5, method="predict_proba")[:, 1]

        auc3 = roc_auc_score(y2, pred3)
        auc2 = roc_auc_score(y2, pred2)
        print(f"    ESM-1b + CADD:      AUC = {auc2:.4f}")
        print(f"    ESM + CADD + novel:  AUC = {auc3:.4f}  (Δ={auc3-auc2:+.4f})")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Error analysis — what does novel_score get right that ESM-1b misses?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 3: ERROR ANALYSIS")
print(f"{'='*70}")

df_err = col_gly.dropna(subset=["ESM-1b"]).copy()
if len(df_err) > 0 and df_err["label"].nunique() == 2:
    # ESM-1b: lower = more damaging (negative LLR)
    esm_med = df_err["ESM-1b"].median()
    # novel_path_score: higher = more pathogenic
    novel_med = df_err["novel_path_score"].median()

    # use percentile-based thresholds
    esm_thresh = df_err["ESM-1b"].quantile(0.5)
    novel_thresh = 0  # any activation = positive

    df_err["esm_predicts_path"] = df_err["ESM-1b"] < esm_thresh
    df_err["novel_predicts_path"] = df_err["novel_path_score"] > 0

    # 2x2: ESM correct, novel correct
    for combo, label in [
        ((True, True), "Both predict pathogenic"),
        ((True, False), "ESM-only predicts pathogenic"),
        ((False, True), "Novel-only predicts pathogenic"),
        ((False, False), "Neither predicts pathogenic"),
    ]:
        mask = ((df_err["esm_predicts_path"] == combo[0]) &
                (df_err["novel_predicts_path"] == combo[1]))
        sub = df_err[mask]
        if len(sub) == 0:
            continue
        pr = sub["label"].mean()
        print(f"\n  {label}: n={len(sub)}, pathogenic rate={pr:.3f}")
        # show gene distribution
        genes = sub["gene"].value_counts().head(5)
        print(f"    Top genes: {dict(genes)}")

    # specifically: pathogenic variants that ESM-1b ranks as BENIGN
    # (i.e., ESM-1b false negatives)
    path_df = df_err[df_err["label"] == True]
    esm_fn = path_df[path_df["ESM-1b"] > path_df["ESM-1b"].quantile(0.75)]
    novel_catches = esm_fn[esm_fn["novel_path_score"] > 0]

    print(f"\n  ESM-1b false negatives (pathogenic but ESM-1b in top 25% least damaging):")
    print(f"    Total: {len(esm_fn)}")
    print(f"    Caught by novel features: {len(novel_catches)} ({100*len(novel_catches)/max(len(esm_fn),1):.1f}%)")
    if len(novel_catches) > 0:
        print(f"    Genes: {dict(novel_catches['gene'].value_counts().head(5))}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Leave-one-gene-out cross-validation
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 4: LEAVE-ONE-GENE-OUT VALIDATION")
print(f"{'='*70}")

# for each gene with enough variants, hold it out and predict using
# novel features learned from other genes
logo_df = struct_gly.dropna(subset=["ESM-1b"]).copy()
logo_df = logo_df[logo_df["gene"].notna()]

gene_counts = logo_df.groupby("gene").agg(
    n=("label", "count"),
    n_path=("label", "sum"),
).reset_index()
valid_genes = gene_counts[(gene_counts["n"] >= 15) &
                           (gene_counts["n_path"] >= 3) &
                           (gene_counts["n"] - gene_counts["n_path"] >= 3)]["gene"].values

print(f"\n  Genes with enough variants for LOGO: {len(valid_genes)}")
print(f"  {sorted(valid_genes)}")

logo_results = []
for test_gene in valid_genes:
    test_mask = logo_df["gene"] == test_gene
    test = logo_df[test_mask]
    y_test = test["label"].astype(int).values

    if len(np.unique(y_test)) < 2:
        continue

    # novel_path_score (already computed, no training needed — it's unsupervised)
    auc_novel = roc_auc_score(y_test, test["novel_path_score"].values)
    if auc_novel < 0.5:
        auc_novel = 1 - auc_novel

    # ESM-1b baseline
    auc_esm = roc_auc_score(y_test, -test["ESM-1b"].values)
    if auc_esm < 0.5:
        auc_esm = 1 - auc_esm

    # total SAE score (control: is novel-specific better than total?)
    auc_total = roc_auc_score(y_test, test["total_sae_score"].values)
    if auc_total < 0.5:
        auc_total = 1 - auc_total

    delta = auc_novel - auc_esm

    logo_results.append({
        "gene": test_gene,
        "n": len(test),
        "n_path": int(y_test.sum()),
        "auc_novel": auc_novel,
        "auc_esm": auc_esm,
        "auc_total_sae": auc_total,
        "delta_novel_minus_esm": delta,
    })
    print(f"  {test_gene:10s}: n={len(test):3d} ({int(y_test.sum())}P) "
          f"novel={auc_novel:.3f}  ESM={auc_esm:.3f}  total_SAE={auc_total:.3f}  "
          f"Δ(novel-ESM)={delta:+.3f}")

if logo_results:
    logo_df_out = pd.DataFrame(logo_results)
    mean_novel = logo_df_out["auc_novel"].mean()
    mean_esm = logo_df_out["auc_esm"].mean()
    mean_delta = logo_df_out["delta_novel_minus_esm"].mean()
    wins = (logo_df_out["delta_novel_minus_esm"] > 0).sum()
    print(f"\n  Mean AUC: novel={mean_novel:.3f}, ESM={mean_esm:.3f}, Δ={mean_delta:+.3f}")
    print(f"  Novel wins: {wins}/{len(logo_results)} genes")

    # sign test
    from scipy.stats import binomtest
    binom = binomtest(wins, len(logo_results), 0.5, alternative="greater")
    print(f"  Sign test p={binom.pvalue:.4f}")

    logo_df_out.to_csv(os.path.join(OUT, "logo_results.csv"), index=False)

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 5: Specificity — novel features vs total SAE vs random features
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 5: SPECIFICITY — NOVEL FEATURES vs RANDOM FEATURE SUBSETS")
print(f"{'='*70}")

rng = np.random.RandomState(42)
n_novel = len(novel_feat_ids)
all_feats = cards["feature"].astype(int).values

# on collagen Gly-sub
y_cg = col_gly["label"].astype(int).values
acts_cg = acts[col_gly.index]

auc_novel_cg = roc_auc_score(y_cg, np.sum(np.abs(acts_cg[:, novel_feat_ids]), axis=1))
if auc_novel_cg < 0.5:
    auc_novel_cg = 1 - auc_novel_cg

# random feature sets of same size
n_random = 1000
random_aucs = []
for _ in range(n_random):
    rand_feats = rng.choice(all_feats, n_novel, replace=False)
    rand_score = np.sum(np.abs(acts_cg[:, rand_feats]), axis=1)
    rauc = roc_auc_score(y_cg, rand_score)
    random_aucs.append(max(rauc, 1 - rauc))

random_aucs = np.array(random_aucs)
pval_rand = np.mean(random_aucs >= auc_novel_cg)

print(f"\n  Collagen Gly-sub (n={len(col_gly)}):")
print(f"    Novel features ({n_novel}):  AUC = {auc_novel_cg:.4f}")
print(f"    Random {n_novel} features:   AUC = {random_aucs.mean():.4f} ± {random_aucs.std():.4f}")
print(f"    P(random ≥ novel):    {pval_rand:.4f}")
print(f"    Novel z-score:        {(auc_novel_cg - random_aucs.mean()) / random_aucs.std():.2f}")

# also compare: novel features vs protein-driven vs dna-driven vs cross-modal
for name, feat_ids in [("protein-driven", prot_feats), ("dna-driven", dna_feats),
                         ("cross-modal", cross_feats)]:
    sc = np.sum(np.abs(acts_cg[:, feat_ids]), axis=1)
    auc = roc_auc_score(y_cg, sc)
    auc = max(auc, 1 - auc)
    print(f"    {name} ({len(feat_ids)}):  AUC = {auc:.4f}")

# ── Save ─────────────────────────────────────────────────────────
results_summary = {
    "novel_auc_collagen_gly": auc_novel_cg,
    "random_mean_auc": random_aucs.mean(),
    "random_std_auc": random_aucs.std(),
    "p_vs_random": pval_rand,
}
pd.DataFrame([results_summary]).to_csv(os.path.join(OUT, "validation_summary.csv"), index=False)
print(f"\nSaved to {OUT}/")
