"""
Validate utility of SAE novel features — revised.

Key insight: Gly-sub in collagen are almost ALL pathogenic in ClinVar (1020P/1B).
So P vs B classification is the wrong question.

Better questions:
1. Among ALL ClinVar variants (balanced), does novel_path_score specifically
   improve prediction for structural protein variants?
2. Specificity: novel features fire ONLY on structural variants, not randomly
3. Cross-modal specificity: novel features capture something ESM-1b alone doesn't
4. Severity ranking: do features rank known pathogenic by severity (proxied by
   number of ClinVar submitters, review stars, or functional data)?
5. VUS stratification: concordance between novel_score and ESM-1b/CADD on VUS
"""

import os, re, sys, gzip
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

OUT = "results/gly_validation"
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
print("Loading data ...")
clinvar = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
acts = np.load("results/sae_genomewide/sae_genomewide_acts.npz")["acts"]
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")
vus = pd.read_csv("results/vus_pilot/vus_mechanism_profiles.csv")
vus_acts = np.load("results/vus_pilot/vus_sae_acts.npz")["acts"]
y = clinvar["label"].astype(int).values

COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
}
STRUCTURAL_GENES = COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}

novel_path = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
novel_feat_ids = sorted(novel_path["feature"].astype(int).values)
all_feat_ids = cards["feature"].astype(int).values

# scores for ClinVar
novel_score = np.sum(np.abs(acts[:, novel_feat_ids]), axis=1)
clinvar["novel_path_score"] = novel_score

# map genes
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
            clinvar_info[key] = {
                "gene": parts[4], "pchange": pchange,
                "review": parts[24] if len(parts) > 24 else "",
                "n_submitters": parts[25] if len(parts) > 25 else "",
            }

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

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Specificity — novel features fire on structural proteins
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 1: SPECIFICITY — WHERE DO NOVEL FEATURES FIRE?")
print(f"{'='*70}")

# among pathogenic variants: activation rate by gene type
path = clinvar[clinvar["label"] == True]
ben = clinvar[clinvar["label"] == False]

for label_name, sub in [("Pathogenic", path), ("Benign", ben)]:
    struct_mask = sub["is_structural"]
    col_mask = sub["is_collagen"]
    gly_mask = sub["is_gly_sub"] & col_mask

    act_struct = (sub.loc[struct_mask, "novel_path_score"] > 0).mean()
    act_col = (sub.loc[col_mask, "novel_path_score"] > 0).mean() if col_mask.sum() > 0 else 0
    act_gly = (sub.loc[gly_mask, "novel_path_score"] > 0).mean() if gly_mask.sum() > 0 else 0
    act_other = (sub.loc[~struct_mask, "novel_path_score"] > 0).mean()

    print(f"\n  {label_name} variants:")
    print(f"    Structural protein:  {act_struct:.1%} activated "
          f"(n={struct_mask.sum()})")
    print(f"    Collagen:            {act_col:.1%} activated "
          f"(n={col_mask.sum()})")
    print(f"    Collagen Gly-sub:    {act_gly:.1%} activated "
          f"(n={gly_mask.sum()})")
    print(f"    Non-structural:      {act_other:.1%} activated "
          f"(n={(~struct_mask).sum()})")

# enrichment: structural pathogenic vs non-structural pathogenic
struct_path_act = (path.loc[path["is_structural"], "novel_path_score"] > 0).mean()
other_path_act = (path.loc[~path["is_structural"], "novel_path_score"] > 0).mean()
enrichment = struct_path_act / max(other_path_act, 1e-8)
print(f"\n  Enrichment (structural_path / other_path): {enrichment:.1f}x")

# Fisher exact test
from scipy.stats import fisher_exact
a = (path["is_structural"] & (path["novel_path_score"] > 0)).sum()
b = (path["is_structural"] & (path["novel_path_score"] == 0)).sum()
c = (~path["is_structural"] & (path["novel_path_score"] > 0)).sum()
d = (~path["is_structural"] & (path["novel_path_score"] == 0)).sum()
_, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
print(f"  Fisher exact p = {fisher_p:.4g}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: What do novel features capture that ESM-1b doesn't?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 2: NOVEL FEATURES vs ESM-1b — ORTHOGONAL INFORMATION")
print(f"{'='*70}")

# correlation between novel_path_score and ESM-1b
coding = clinvar[clinvar["ESM-1b"].notna()].copy()
rho, pval = stats.spearmanr(coding["novel_path_score"], coding["ESM-1b"])
print(f"\n  Correlation (all coding variants):")
print(f"    novel_score vs ESM-1b: Spearman rho={rho:.3f}, p={pval:.4g}")

# on structural proteins only
struct = coding[coding["is_structural"]]
if len(struct) > 10:
    rho_s, pval_s = stats.spearmanr(struct["novel_path_score"], struct["ESM-1b"])
    print(f"    structural only:      Spearman rho={rho_s:.3f}, p={pval_s:.4g}")

# KEY: among pathogenic structural variants, ESM-1b and novel_score disagreement
struct_path = coding[(coding["is_structural"]) & (coding["label"] == True)]
if len(struct_path) > 10:
    # ESM-1b thinks least damaging (top quartile, less negative)
    esm_q75 = struct_path["ESM-1b"].quantile(0.75)
    esm_mild = struct_path[struct_path["ESM-1b"] >= esm_q75]
    esm_severe = struct_path[struct_path["ESM-1b"] < struct_path["ESM-1b"].quantile(0.25)]

    print(f"\n  Among pathogenic structural variants (n={len(struct_path)}):")
    print(f"    ESM-1b 'mild' quartile (least damaging ESM score):")
    print(f"      n={len(esm_mild)}, novel activation rate="
          f"{(esm_mild['novel_path_score'] > 0).mean():.1%}, "
          f"mean novel_score={esm_mild['novel_path_score'].mean():.3f}")
    print(f"    ESM-1b 'severe' quartile (most damaging ESM score):")
    print(f"      n={len(esm_severe)}, novel activation rate="
          f"{(esm_severe['novel_path_score'] > 0).mean():.1%}, "
          f"mean novel_score={esm_severe['novel_path_score'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Specificity advantage — precision at high recall
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 3: PRECISION-SPECIFICITY ON STRUCTURAL PROTEIN DETECTION")
print(f"{'='*70}")

# task: detect structural pathogenic variants among all pathogenic variants
# this is where novel features should shine — high precision for structural
path_only = clinvar[clinvar["label"] == True].copy()
path_only["is_target"] = path_only["is_structural"].astype(int)

for score_name, score_col in [("novel_path_score", "novel_path_score"),
                                ("ESM-1b (abs)", "ESM-1b")]:
    vals = path_only[score_col].dropna()
    if len(vals) < 100:
        continue
    mask = path_only[score_col].notna()
    sub = path_only[mask]
    y_target = sub["is_target"].values

    if score_col == "ESM-1b":
        scores = -sub[score_col].values  # more negative = more damaging
    else:
        scores = sub[score_col].values

    auc = roc_auc_score(y_target, scores)

    # precision at various thresholds
    prec, rec, thresh = precision_recall_curve(y_target, scores)

    print(f"\n  {score_name} for detecting structural pathogenic:")
    print(f"    AUC = {auc:.4f}")

    # precision when only top-N variants are flagged
    sorted_idx = np.argsort(-scores)
    for topk in [50, 100, 200, 500]:
        if topk > len(sorted_idx):
            continue
        top_idx = sorted_idx[:topk]
        prec_at_k = y_target[top_idx].mean()
        print(f"    Precision@{topk}: {prec_at_k:.3f} "
              f"(baseline={y_target.mean():.3f})")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Random feature null — is novel-specific better?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 4: NOVEL FEATURES vs RANDOM FEATURE SUBSETS")
print(f"{'='*70}")

rng = np.random.RandomState(42)
n_novel = len(novel_feat_ids)

# task: detect structural pathogenic among all pathogenic
path_only = clinvar[clinvar["label"] == True].copy()
path_only["is_target"] = path_only["is_structural"].astype(int)
y_target = path_only["is_target"].values
acts_path = acts[path_only.index]

# novel features
novel_sc = np.sum(np.abs(acts_path[:, novel_feat_ids]), axis=1)
auc_novel = roc_auc_score(y_target, novel_sc)

# random null
n_perm = 1000
rand_aucs = []
for _ in range(n_perm):
    rand_feats = rng.choice(all_feat_ids, n_novel, replace=False)
    rand_sc = np.sum(np.abs(acts_path[:, rand_feats]), axis=1)
    rand_aucs.append(roc_auc_score(y_target, rand_sc))
rand_aucs = np.array(rand_aucs)

pval = np.mean(rand_aucs >= auc_novel)
zscore = (auc_novel - rand_aucs.mean()) / max(rand_aucs.std(), 1e-8)

print(f"\n  Detecting structural pathogenic among all pathogenic:")
print(f"    Novel features ({n_novel}):    AUC = {auc_novel:.4f}")
print(f"    Random {n_novel} features:     AUC = {rand_aucs.mean():.4f} ± {rand_aucs.std():.4f}")
print(f"    Z-score:                 {zscore:.2f}")
print(f"    P(random ≥ novel):       {pval:.4f}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 5: VUS stratification — agreement/disagreement with ESM-1b
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 5: VUS STRATIFICATION")
print(f"{'='*70}")

gly_vus = vus[vus["is_gly_sub"]].copy()
print(f"\n  Total Gly-sub VUS: {len(gly_vus)}")

# strata: novel-activated vs not
act = gly_vus[gly_vus["n_novel_features_active"] > 0]
noact = gly_vus[gly_vus["n_novel_features_active"] == 0]
print(f"  Novel-activated: {len(act)} ({100*len(act)/len(gly_vus):.1f}%)")
print(f"  Non-activated:   {len(noact)} ({100*len(noact)/len(gly_vus):.1f}%)")

# per-gene breakdown for activated VUS
print(f"\n  Novel-activated VUS per gene:")
for gene in sorted(act["gene"].unique()):
    g_total = gly_vus[gly_vus["gene"] == gene]
    g_act = act[act["gene"] == gene]
    rate = len(g_act) / max(len(g_total), 1)
    print(f"    {gene:10s}: {len(g_act):3d}/{len(g_total):3d} ({rate:.1%}), "
          f"mean_score={g_act['novel_path_score'].mean():.3f}")

# how many novel features activate per VUS (among activated)?
print(f"\n  Feature activation count (among activated VUS):")
for n_feat in [1, 2, 3, 4]:
    n = (act["n_novel_features_active"] == n_feat).sum()
    if n > 0:
        print(f"    {n_feat} features: {n} VUS ({100*n/len(act):.1f}%)")
n_more = (act["n_novel_features_active"] > 4).sum()
if n_more > 0:
    print(f"    5+ features: {n_more} VUS ({100*n_more/len(act):.1f}%)")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 6: Clinical context — known pathogenic Gly-sub vs ALL variants
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 6: NOVEL FEATURES AS STRUCTURAL PATHOGENICITY BIOMARKER")
print(f"{'='*70}")

# the key claim: among ALL ClinVar variants, novel_path_score > 0 is a
# highly specific biomarker for structural protein pathogenic variants
all_with_novel = clinvar.copy()
all_with_novel["novel_active"] = all_with_novel["novel_path_score"] > 0

# precision: of variants with novel_active=True, what fraction are structural pathogenic?
novel_active = all_with_novel[all_with_novel["novel_active"]]
novel_inactive = all_with_novel[~all_with_novel["novel_active"]]

n_struct_path_active = ((novel_active["is_structural"]) & (novel_active["label"] == True)).sum()
precision_struct = n_struct_path_active / max(len(novel_active), 1)
recall_struct = n_struct_path_active / max(
    ((all_with_novel["is_structural"]) & (all_with_novel["label"] == True)).sum(), 1)

print(f"\n  Novel feature activation as biomarker:")
print(f"    Activated variants: {len(novel_active)}")
print(f"    Of which structural pathogenic: {n_struct_path_active} ({precision_struct:.1%})")
print(f"    Recall of structural pathogenic: {recall_struct:.1%}")

# breakdown: what ARE the activated variants?
print(f"\n  What gets activated (novel_score > 0)?")
for (is_struct, label), grp in novel_active.groupby(["is_structural", "label"]):
    label_name = "Pathogenic" if label else "Benign"
    struct_name = "Structural" if is_struct else "Non-structural"
    print(f"    {struct_name} {label_name}: {len(grp)} "
          f"({100*len(grp)/len(novel_active):.1f}%)")

# compare with ESM-1b as biomarker (using same number of top-scoring variants)
n_active = len(novel_active)
esm_valid = all_with_novel[all_with_novel["ESM-1b"].notna()]
esm_top = esm_valid.nsmallest(n_active, "ESM-1b")  # most negative = most damaging
n_struct_path_esm = ((esm_top["is_structural"]) & (esm_top["label"] == True)).sum()
precision_esm = n_struct_path_esm / max(len(esm_top), 1)

print(f"\n  Comparison — top {n_active} variants flagged:")
print(f"    By novel features: {n_struct_path_active} structural pathogenic ({precision_struct:.1%})")
print(f"    By ESM-1b (most damaging): {n_struct_path_esm} structural pathogenic ({precision_esm:.1%})")
print(f"    Novel specificity advantage: {precision_struct/max(precision_esm, 1e-8):.1f}x")

# ── Save summary ─────────────────────────────────────────────────
summary = {
    "n_novel_features": n_novel,
    "n_novel_active_variants": len(novel_active),
    "precision_structural_path": precision_struct,
    "recall_structural_path": recall_struct,
    "auc_detect_structural_path": auc_novel,
    "random_null_auc_mean": rand_aucs.mean(),
    "random_null_pvalue": pval,
    "zscore_vs_random": zscore,
    "n_vus_activated": len(act),
    "n_vus_total_gly": len(gly_vus),
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT, "validation_summary_v2.csv"), index=False)
print(f"\nSaved to {OUT}/")
