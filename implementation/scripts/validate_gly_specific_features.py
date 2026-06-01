"""
Identify collagen-Gly-SPECIFIC SAE features and validate their utility.

Strategy: instead of using all 28 "novel pathogenic" features,
select only those with high enrichment for collagen Gly-sub variants.
Then test whether this specific subset is a better biomarker.
"""

import os, re, gzip
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

OUT = "results/gly_validation"
os.makedirs(OUT, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────
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

novel_path = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
novel_feat_ids = sorted(novel_path["feature"].astype(int).values)

# map genes
print("Mapping genes ...")
clinvar_info = {}
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        key = (parts[18], parts[31], parts[32], parts[33])
        if key not in clinvar_info:
            pchange = ""
            m = re.search(r'\(p\.([A-Za-z0-9_*=]+)\)', parts[2])
            if m:
                pchange = m.group(1)
            clinvar_info[key] = {"gene": parts[4], "pchange": pchange}

clinvar["gene"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("gene")
    for r in clinvar.itertuples()]
clinvar["pchange"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("pchange", "")
    for r in clinvar.itertuples()]

def is_gly_missense(pc):
    if not pc or pd.isna(pc):
        return False
    m = re.match(r'Gly\d+([A-Z][a-z]{2})', str(pc))
    return m is not None and m.group(1) != "Gly"

clinvar["is_gly_sub"] = clinvar["pchange"].apply(is_gly_missense)
clinvar["is_collagen"] = clinvar["gene"].isin(COLLAGEN_GENES)

# target: collagen Gly-sub pathogenic
is_target = (clinvar["is_collagen"] & clinvar["is_gly_sub"] & (y == 1)).values
is_path = (y == 1)
base_rate = is_target.sum() / max(is_path.sum(), 1)
print(f"\n  Collagen Gly-sub pathogenic: {is_target.sum()} / {is_path.sum()} = {base_rate:.4f}")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Rank features by collagen-Gly enrichment
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 1: FEATURE-LEVEL COLLAGEN-Gly ENRICHMENT")
print(f"{'='*70}")

feat_enrichment = []
for fi in novel_feat_ids:
    a = np.abs(acts[:, fi])
    active = a > 1e-6
    n_act = active.sum()
    if n_act < 10:
        continue

    n_target_active = (active & is_target).sum()
    target_rate = n_target_active / n_act
    enrichment = target_rate / max(base_rate, 1e-8)

    # also: what fraction of target variants does this feature catch?
    recall = n_target_active / max(is_target.sum(), 1)

    feat_enrichment.append({
        "feature": fi, "n_active": n_act,
        "n_collagen_gly_path": n_target_active,
        "target_rate": target_rate,
        "enrichment": enrichment,
        "recall": recall,
    })

fe = pd.DataFrame(feat_enrichment).sort_values("enrichment", ascending=False)

print(f"\n  {'feat':>5s} {'n_act':>6s} {'n_gly_p':>7s} {'tgt_rate':>8s} "
      f"{'enrich':>7s} {'recall':>6s}")
for _, r in fe.iterrows():
    print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} "
          f"{int(r['n_collagen_gly_path']):7d} {r['target_rate']:8.3%} "
          f"{r['enrichment']:7.1f}x {r['recall']:6.1%}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Select Gly-specific features (enrichment > 3x)
# ══════════════════════════════════════════════════════════════════
THRESH = 3.0
gly_specific = fe[fe["enrichment"] >= THRESH]
gly_feat_ids = sorted(gly_specific["feature"].astype(int).values)
non_gly_feats = sorted(set(novel_feat_ids) - set(gly_feat_ids))

print(f"\n  Gly-specific features (enrichment ≥ {THRESH}x): {len(gly_feat_ids)}")
print(f"  Non-specific novel features: {len(non_gly_feats)}")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Compare biomarker performance
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 2: BIOMARKER COMPARISON — DETECTING COLLAGEN Gly-SUB PATHOGENIC")
print(f"{'='*70}")
print(f"  Task: among ALL ClinVar pathogenic, identify collagen Gly-sub")

y_task = is_target[is_path]
acts_path = acts[is_path]

def auc_for_feats(feat_ids, name):
    sc = np.sum(np.abs(acts_path[:, feat_ids]), axis=1)
    auc = roc_auc_score(y_task, sc)
    # precision at various cutoffs
    sorted_idx = np.argsort(-sc)
    results = {"name": name, "n_feats": len(feat_ids), "auc": auc}
    for k in [50, 100, 200, 500]:
        if k > len(sorted_idx):
            continue
        prec = y_task[sorted_idx[:k]].mean()
        results[f"prec@{k}"] = prec
    return results

results = []
results.append(auc_for_feats(gly_feat_ids, "Gly-specific novel"))
results.append(auc_for_feats(novel_feat_ids, "All novel (28)"))
results.append(auc_for_feats(non_gly_feats, "Non-specific novel"))

# random null (same number as gly-specific)
rng = np.random.RandomState(42)
all_feats = cards["feature"].astype(int).values
rand_aucs = []
for _ in range(1000):
    rf = rng.choice(all_feats, len(gly_feat_ids), replace=False)
    sc = np.sum(np.abs(acts_path[:, rf]), axis=1)
    rand_aucs.append(roc_auc_score(y_task, sc))
rand_aucs = np.array(rand_aucs)
results.append({"name": f"Random {len(gly_feat_ids)} feats", "n_feats": len(gly_feat_ids),
                 "auc": rand_aucs.mean()})

res_df = pd.DataFrame(results)
print(f"\n  {'Method':30s} {'n':>3s} {'AUC':>7s} {'P@50':>6s} {'P@100':>6s} {'P@200':>6s}")
for _, r in res_df.iterrows():
    p50 = r.get("prec@50", np.nan)
    p100 = r.get("prec@100", np.nan)
    p200 = r.get("prec@200", np.nan)
    print(f"  {r['name']:30s} {int(r['n_feats']):3d} {r['auc']:7.4f} "
          f"{p50:6.3f} {p100:6.3f} {p200:6.3f}")
print(f"  {'Baseline (random chance)':30s}     {base_rate:7.4f}")

# significance vs random
pval = np.mean(rand_aucs >= results[0]["auc"])
zscore = (results[0]["auc"] - rand_aucs.mean()) / max(rand_aucs.std(), 1e-8)
print(f"\n  Gly-specific vs random: z={zscore:.2f}, p={pval:.4f}")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Specificity comparison — what gets flagged?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 3: WHAT GETS FLAGGED? (Gly-specific vs All-novel vs ESM-1b)")
print(f"{'='*70}")

# for each method, take top-200 pathogenic variants
n_flag = 200
# Gly-specific
gly_sc = np.sum(np.abs(acts[:, gly_feat_ids]), axis=1)
clinvar["gly_specific_score"] = gly_sc

for method_name, score_col, ascending in [
    ("Gly-specific features", "gly_specific_score", False),
    ("All novel features", "novel_path_score", False),
    ("ESM-1b", "ESM-1b", True),  # more negative = more damaging
]:
    clinvar["novel_path_score"] = np.sum(np.abs(acts[:, novel_feat_ids]), axis=1)
    sub = clinvar[clinvar["label"] == True].copy()
    if score_col not in sub.columns:
        continue
    sub = sub.dropna(subset=[score_col])
    top = sub.nsmallest(n_flag, score_col) if ascending else sub.nlargest(n_flag, score_col)

    n_col_gly = (top["is_collagen"] & top["is_gly_sub"]).sum()
    n_col = top["is_collagen"].sum()
    n_struct = top["gene"].isin(COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}).sum()

    print(f"\n  {method_name} — top {n_flag} pathogenic:")
    print(f"    Collagen Gly-sub:     {n_col_gly:4d} ({100*n_col_gly/n_flag:.1f}%)")
    print(f"    Collagen (any):       {n_col:4d} ({100*n_col/n_flag:.1f}%)")
    print(f"    Structural (any):     {n_struct:4d} ({100*n_struct/n_flag:.1f}%)")

    # gene distribution
    gene_counts = top["gene"].value_counts().head(10)
    print(f"    Top genes: {dict(gene_counts)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: VUS — Gly-specific features on VUS
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 4: VUS STRATIFICATION WITH Gly-SPECIFIC FEATURES")
print(f"{'='*70}")

gly_vus = vus[vus["is_gly_sub"]].copy()

# compute Gly-specific score for VUS
gly_specific_vus_score = np.sum(np.abs(vus_acts[:, gly_feat_ids]), axis=1)
vus["gly_specific_score"] = gly_specific_vus_score
gly_vus = vus[vus["is_gly_sub"]].copy()

gly_vus["activated_specific"] = gly_vus["gly_specific_score"] > 0
n_act_specific = gly_vus["activated_specific"].sum()
n_act_all = (gly_vus["n_novel_features_active"] > 0).sum()

print(f"\n  Gly-sub VUS: {len(gly_vus)}")
print(f"  Activated by Gly-specific features: {n_act_specific} ({100*n_act_specific/len(gly_vus):.1f}%)")
print(f"  Activated by all novel features:    {n_act_all} ({100*n_act_all/len(gly_vus):.1f}%)")

# per-gene activation rate
print(f"\n  Per-gene activation (Gly-specific features):")
print(f"  {'gene':10s} {'total':>5s} {'act':>4s} {'rate':>6s} {'mean_score':>10s}")
for gene in sorted(gly_vus["gene"].unique()):
    g = gly_vus[gly_vus["gene"] == gene]
    g_act = g[g["activated_specific"]]
    rate = len(g_act) / max(len(g), 1)
    ms = g["gly_specific_score"].mean()
    print(f"  {gene:10s} {len(g):5d} {len(g_act):4d} {rate:6.1%} {ms:10.3f}")

# top candidates with Gly-specific score
top_vus = gly_vus[gly_vus["activated_specific"]].nlargest(20, "gly_specific_score")
print(f"\n  Top 20 VUS by Gly-specific score:")
print(f"  {'gene':10s} {'chrom':>5s} {'pos':>12s} {'pchange':>15s} {'score':>8s}")
for _, r in top_vus.iterrows():
    print(f"  {r['gene']:10s} {str(r['chrom']):>5s} {int(r['pos']):12d} "
          f"{str(r['pchange']):>15s} {r['gly_specific_score']:8.3f}")

# ══════════════════════════════════════════════════════════════════
# STEP 5: ESM-1b agreement/disagreement on known variants
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 5: ORTHOGONALITY — ESM-1b vs Gly-SPECIFIC FEATURES ON KNOWN VARIANTS")
print(f"{'='*70}")

# among collagen Gly-sub pathogenic variants in ClinVar:
# do Gly-specific features fire INDEPENDENTLY of ESM-1b?
target_path = clinvar[is_target & clinvar["ESM-1b"].notna()].copy()
target_path["gly_activated"] = gly_sc[target_path.index] > 0

if len(target_path) > 20:
    # bin by ESM-1b quartiles
    target_path["esm_quartile"] = pd.qcut(target_path["ESM-1b"], q=4, labels=False)
    print(f"\n  Known pathogenic collagen Gly-sub (n={len(target_path)}):")
    print(f"  ESM-1b quartile → Gly-feature activation rate:")
    for q in range(4):
        qg = target_path[target_path["esm_quartile"] == q]
        act_rate = qg["gly_activated"].mean()
        esm_range = f"[{qg['ESM-1b'].min():.1f}, {qg['ESM-1b'].max():.1f}]"
        print(f"    Q{q} (ESM-1b {esm_range}): {len(qg)} variants, "
              f"Gly-feature activation={act_rate:.1%}")

    rho, pval = stats.spearmanr(target_path["ESM-1b"],
                                  gly_sc[target_path.index])
    print(f"\n  ESM-1b vs Gly-specific score: Spearman rho={rho:.3f}, p={pval:.4g}")
    print(f"  → {'INDEPENDENT' if abs(rho) < 0.1 else 'CORRELATED'}")

# ── Save ─────────────────────────────────────────────────────────
fe.to_csv(os.path.join(OUT, "feature_gly_enrichment.csv"), index=False)
gly_vus.to_csv(os.path.join(OUT, "vus_gly_specific_scores.csv"), index=False)
print(f"\nSaved to {OUT}/")
