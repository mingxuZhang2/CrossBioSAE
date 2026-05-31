"""
Per-variant mechanism profile from cross-modal SAE.

For each variant, decomposes the SAE activation into 4 biological signal sources:
  1. Protein-driven: features correlated with ESM-1b (protein damage)
  2. DNA-driven: features correlated with GPN-MSA/NT (DNA conservation)
  3. Cross-modal: features correlated with both modalities
  4. Novel: features not correlated with any known score (NEW biology)

Produces:
  - Per-variant mechanism profiles
  - Gene-level case studies (BRCA1, TP53, PTEN)
  - Cross-modal consistency score for VUS stratification
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os, sys, warnings
warnings.filterwarnings("ignore")

RESULTS = "results/sae_genomewide"
OUT = "results/mechanism_profiles"
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
acts = np.load(os.path.join(RESULTS, "sae_genomewide_acts.npz"))["acts"]
cards = pd.read_csv(os.path.join(RESULTS, "sae_genomewide_cards_deep.csv"))

y = df["label"].astype(int).values
is_coding = df["ESM-1b"].notna().values

# load gene mapping from crossmodal analysis
gene_map_path = "results/sae_crossmodal/variant_gene_mapping.csv"
if os.path.exists(gene_map_path):
    gm = pd.read_csv(gene_map_path)
    df["gene"] = gm["gene"].values[:len(df)] if len(gm) >= len(df) else None
    df["var_type"] = gm["var_type"].values[:len(df)] if len(gm) >= len(df) else None
else:
    df["gene"] = None

print(f"  {len(df)} variants, {len(cards)} alive features")

# ── Classify features by modality ────────────────────────────────
# Use the deep analysis modality labels
feat_idx = cards["feature"].values
modality = cards["modality"].values

prot_feats = feat_idx[modality == "protein-driven"]
dna_feats = feat_idx[modality == "dna-driven"]
cross_feats = feat_idx[modality == "cross-modal"]
novel_feats = feat_idx[modality == "novel"]

print(f"  Feature groups: protein={len(prot_feats)}, dna={len(dna_feats)}, "
      f"cross={len(cross_feats)}, novel={len(novel_feats)}")

# ── Compute per-variant mechanism profile ────────────────────────
print("\nComputing mechanism profiles ...")

def compute_profile(acts_matrix, feat_groups):
    """For each variant, sum absolute activations in each feature group."""
    profiles = {}
    for name, feat_ids in feat_groups.items():
        profiles[name] = np.abs(acts_matrix[:, feat_ids]).sum(axis=1)
    return pd.DataFrame(profiles)

feat_groups = {
    "protein_signal": prot_feats,
    "dna_signal": dna_feats,
    "crossmodal_signal": cross_feats,
    "novel_signal": novel_feats,
}

profiles = compute_profile(acts, feat_groups)
profiles["total"] = profiles.sum(axis=1)

# normalize to fractions
for col in feat_groups:
    profiles[f"{col}_frac"] = profiles[col] / (profiles["total"] + 1e-8)

profiles["label"] = y
profiles["is_coding"] = is_coding
if "gene" in df.columns:
    profiles["gene"] = df["gene"].values

# ── Cross-modal consistency score ────────────────────────────────
# High consistency = protein and DNA signals agree (both high or both low)
# Low consistency = they disagree → uncertain mechanism
prot_norm = profiles["protein_signal"] / (profiles["protein_signal"].max() + 1e-8)
dna_norm = profiles["dna_signal"] / (profiles["dna_signal"].max() + 1e-8)
profiles["consistency"] = 1 - np.abs(prot_norm - dna_norm)
profiles["dominant_modality"] = np.where(
    profiles["protein_signal_frac"] > profiles["dna_signal_frac"],
    "protein", "dna")

# ── Analysis 1: Mechanism profile by pathogenicity ───────────────
print(f"\n{'='*70}")
print("MECHANISM PROFILES BY PATHOGENICITY")
print(f"{'='*70}")

for label, name in [(1, "Pathogenic"), (0, "Benign")]:
    sub = profiles[profiles["label"] == label]
    print(f"\n  {name} (n={len(sub)}):")
    for col in feat_groups:
        frac_col = f"{col}_frac"
        print(f"    {col:20s}: mean_frac={sub[frac_col].mean():.4f} ± {sub[frac_col].std():.4f}")
    print(f"    consistency: {sub['consistency'].mean():.4f} ± {sub['consistency'].std():.4f}")

# ── Analysis 2: Predictive power of each signal ─────────────────
print(f"\n{'='*70}")
print("PER-SIGNAL PATHOGENICITY AUC")
print(f"{'='*70}")

for col in list(feat_groups.keys()) + ["total", "consistency"]:
    vals = profiles[col].values
    if len(np.unique(vals)) < 2:
        continue
    auc = roc_auc_score(y, vals)
    auc = max(auc, 1 - auc)
    print(f"  {col:25s}: AUC = {auc:.4f}")

# also test: protein_frac and dna_frac as predictors
for col in ["protein_signal_frac", "dna_signal_frac", "novel_signal_frac"]:
    auc = roc_auc_score(y, profiles[col].values)
    auc = max(auc, 1 - auc)
    print(f"  {col:25s}: AUC = {auc:.4f}")

# compare with raw scores
print(f"\n  Raw score baselines:")
for col in ["ESM-1b", "GPN-MSA", "CADD"]:
    vals = df[col].values.astype(float)
    ok = np.isfinite(vals)
    if ok.sum() < 100:
        continue
    auc = roc_auc_score(y[ok], vals[ok])
    auc = max(auc, 1 - auc)
    print(f"  {col:25s}: AUC = {auc:.4f} (n={ok.sum()})")

# ── Analysis 3: Consistency stratification ───────────────────────
print(f"\n{'='*70}")
print("CONSISTENCY-BASED STRATIFICATION")
print(f"{'='*70}")

# bin variants by consistency score
for lo, hi, label in [(0, 0.25, "very low"), (0.25, 0.5, "low"),
                       (0.5, 0.75, "medium"), (0.75, 1.01, "high")]:
    mask = (profiles["consistency"] >= lo) & (profiles["consistency"] < hi)
    n = mask.sum()
    if n < 20:
        continue
    pr = y[mask].mean()
    # AUC using total activation
    if len(np.unique(y[mask])) == 2:
        auc = roc_auc_score(y[mask], profiles["total"].values[mask])
        auc = max(auc, 1 - auc)
    else:
        auc = float('nan')
    print(f"  Consistency {label:10s} [{lo:.2f}-{hi:.2f}): n={n:6d}, "
          f"path_rate={pr:.3f}, total_AUC={auc:.4f}")

# ── Analysis 4: Case studies ─────────────────────────────────────
print(f"\n{'='*70}")
print("CASE STUDIES: PER-GENE MECHANISM PROFILES")
print(f"{'='*70}")

# gene mapping from variant_summary
import gzip
if df["gene"].isna().all():
    print("\n  Loading gene mapping ...")
    lookup_keys = set(zip(df["chrom"].astype(str), df["pos"].astype(str),
                          df["ref"].astype(str), df["alt"].astype(str)))
    gene_lookup = {}
    with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 34 or parts[16] != "GRCh38":
                continue
            key = (parts[18], parts[31], parts[32], parts[33])
            if key in lookup_keys and key not in gene_lookup:
                gene_lookup[key] = parts[4]
    df["gene"] = [gene_lookup.get((str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), None)
                  for r in df.itertuples()]
    profiles["gene"] = df["gene"].values

case_genes = ["BRCA1", "TP53", "PTEN", "MSH2", "CFTR", "SCN1A", "KCNQ2", "RB1"]

for gene in case_genes:
    mask = df["gene"].values == gene
    if mask.sum() < 10:
        continue
    sub = profiles[mask]
    sub_y = y[mask]
    n_path = sub_y.sum()
    n_ben = (1 - sub_y).sum()

    print(f"\n  ── {gene} ({mask.sum()} variants: {n_path} P, {n_ben} B) ──")
    print(f"    Mechanism profile (mean fraction):")
    for col in feat_groups:
        frac = f"{col}_frac"
        print(f"      {col:20s}: P={sub.loc[sub_y==1, frac].mean():.3f}  "
              f"B={sub.loc[sub_y==0, frac].mean():.3f}" if n_ben > 0
              else f"      {col:20s}: P={sub.loc[sub_y==1, frac].mean():.3f}")

    # which modality dominates for pathogenic variants?
    if n_path > 0:
        path_sub = sub[sub_y == 1]
        dom = path_sub["dominant_modality"].value_counts()
        print(f"    Pathogenic dominant modality: {dict(dom)}")
        print(f"    Pathogenic consistency: {path_sub['consistency'].mean():.3f}")

    # top activated features for pathogenic variants
    if n_path > 5:
        path_acts = acts[mask & (y == 1)]
        mean_act = np.abs(path_acts).mean(axis=0)
        top_feats = np.argsort(mean_act)[-5:][::-1]
        print(f"    Top features for pathogenic {gene}:")
        for fi in top_feats:
            card_row = cards[cards["feature"] == fi]
            if len(card_row) > 0:
                mod = card_row.iloc[0]["modality"]
                concept = card_row.iloc[0].get("concept_v2", card_row.iloc[0].get("concept", ""))
                print(f"      feat {fi}: act={mean_act[fi]:.3f}, modality={mod}, concept={concept}")

# ── Analysis 5: Novel pathogenic mechanism discovery ─────────────
print(f"\n{'='*70}")
print("NOVEL PATHOGENIC FEATURES — WHAT BIOLOGY?")
print(f"{'='*70}")

# for each novel pathogenic feature, analyze what genes/chroms it fires on
novel_path_feats = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
print(f"\n  {len(novel_path_feats)} novel pathogenic features (pr > 0.85)")

if len(novel_path_feats) > 0 and df["gene"].notna().any():
    for _, r in novel_path_feats.sort_values("path_rate", ascending=False).head(10).iterrows():
        fi = int(r["feature"])
        a = acts[:, fi]
        active = np.abs(a) > 1e-6
        active_genes = df.loc[active, "gene"].dropna()
        top_genes = active_genes.value_counts().head(5)
        active_y = y[active]

        print(f"\n    Feature {fi}: n={active.sum()}, path_rate={r['path_rate']:.2f}")
        print(f"      Top genes: {dict(top_genes)}")
        print(f"      Coding frac: {is_coding[active].mean():.2f}")
        # check if there's a gene family pattern
        if len(top_genes) > 0:
            gene_frac = top_genes.iloc[0] / active.sum()
            print(f"      Top gene concentration: {top_genes.index[0]} = {gene_frac:.2%}")

# ── Save ─────────────────────────────────────────────────────────
profiles.to_csv(os.path.join(OUT, "variant_mechanism_profiles.csv"), index=False)
print(f"\nSaved {len(profiles)} profiles to {OUT}/")
