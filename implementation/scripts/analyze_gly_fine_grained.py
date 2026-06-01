"""
Fine-grained analysis of collagen Gly-sub VUS: why do SAE novel features
fire on only 14% (354/2534) of Gly-sub VUS?

Three questions:
1. Position effect: C-terminal Gly subs more likely to activate? (severity gradient)
2. Substituting AA effect: Gly→bulky/charged AA vs Gly→small AA?
3. Per-feature specificity: do different novel features capture different biology?

Compare with ClinVar known pathogenic Gly subs as ground truth.
"""

import os, re, sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

OUT = "results/gly_fine_grained"
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
print("Loading data ...")
vus = pd.read_csv("results/vus_pilot/vus_mechanism_profiles.csv")
vus_acts = np.load("results/vus_pilot/vus_sae_acts.npz")["acts"]
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")

# also load known pathogenic data for comparison
clinvar = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
clinvar_acts = np.load("results/sae_genomewide/sae_genomewide_acts.npz")["acts"]

# novel pathogenic feature IDs
novel_path = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
novel_feat_ids = sorted(novel_path["feature"].astype(int).values)
print(f"  {len(novel_feat_ids)} novel pathogenic features")

# ── Parse protein change details ─────────────────────────────────
print("\nParsing protein changes ...")

AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

AA_PROPERTIES = {
    "G": "tiny", "A": "small", "S": "small", "T": "small",
    "V": "medium", "L": "large", "I": "large", "M": "large",
    "F": "aromatic", "Y": "aromatic", "W": "aromatic",
    "P": "rigid", "C": "special",
    "D": "neg_charged", "E": "neg_charged",
    "K": "pos_charged", "R": "pos_charged", "H": "pos_charged",
    "N": "polar", "Q": "polar",
}

AA_SIZE = {
    "G": 75, "A": 89, "S": 105, "T": 119, "C": 121,
    "V": 117, "L": 131, "I": 131, "M": 149,
    "P": 115, "F": 165, "Y": 181, "W": 204,
    "D": 133, "E": 147, "N": 132, "Q": 146,
    "K": 146, "R": 174, "H": 155,
}

def parse_pchange(pc):
    """Extract (from_aa_3, position, to_aa_3) from e.g. Gly123Val"""
    if not pc or pd.isna(pc) or pc == "nan":
        return None, None, None
    m = re.match(r'([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|Ter|\*)', str(pc))
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)

vus["from_aa"], vus["res_pos"], vus["to_aa"] = zip(
    *vus["pchange"].apply(parse_pchange))

gly_vus = vus[vus["is_gly_sub"] & vus["res_pos"].notna()].copy()
gly_vus["res_pos"] = gly_vus["res_pos"].astype(int)
gly_vus["to_aa_1"] = gly_vus["to_aa"].map(AA_3TO1)
gly_vus["to_aa_size"] = gly_vus["to_aa_1"].map(AA_SIZE)
gly_vus["to_aa_prop"] = gly_vus["to_aa_1"].map(AA_PROPERTIES)
gly_vus["activated"] = gly_vus["n_novel_features_active"] > 0

print(f"  Gly-sub VUS with parsed position: {len(gly_vus)}")
print(f"  Activated: {gly_vus['activated'].sum()} ({100*gly_vus['activated'].mean():.1f}%)")

# ── Also parse known pathogenic collagen Gly-subs from ClinVar ───
import gzip
COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1",
    "COL11A2",
}
STRUCTURAL_GENES = COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}

print("\nLoading ClinVar known pathogenic variants ...")
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
                "gene": parts[4], "pchange": pchange, "clin_sig": parts[6],
            }

clinvar["gene"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("gene")
    for r in clinvar.itertuples()]
clinvar["pchange_full"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("pchange", "")
    for r in clinvar.itertuples()]

clinvar["from_aa"], clinvar["res_pos_known"], clinvar["to_aa"] = zip(
    *clinvar["pchange_full"].apply(parse_pchange))

known_gly = clinvar[
    (clinvar["label"] == True) &
    clinvar["gene"].isin(STRUCTURAL_GENES) &
    (clinvar["from_aa"] == "Gly") &
    clinvar["res_pos_known"].notna()
].copy()
known_gly["res_pos_known"] = known_gly["res_pos_known"].astype(int)
known_gly["to_aa_1"] = known_gly["to_aa"].map(AA_3TO1)
known_gly["to_aa_size"] = known_gly["to_aa_1"].map(AA_SIZE)
known_gly["to_aa_prop"] = known_gly["to_aa_1"].map(AA_PROPERTIES)

# compute novel feature activations for known pathogenic
known_novel_score = np.zeros(len(clinvar))
for fi in novel_feat_ids:
    known_novel_score += np.abs(clinvar_acts[:, fi])
clinvar["novel_path_score"] = known_novel_score
known_gly["novel_path_score"] = clinvar.loc[known_gly.index, "novel_path_score"].values
known_gly["activated"] = known_gly["novel_path_score"] > 0

print(f"  Known pathogenic Gly-sub in structural genes: {len(known_gly)}")
print(f"  Activated by novel features: {known_gly['activated'].sum()} ({100*known_gly['activated'].mean():.1f}%)")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Position effect (severity gradient)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 1: POSITION EFFECT (C-terminal severity gradient)")
print(f"{'='*70}")

for gene in sorted(gly_vus["gene"].unique()):
    g = gly_vus[gly_vus["gene"] == gene]
    if len(g) < 20:
        continue
    act = g[g["activated"]]
    noact = g[~g["activated"]]
    if len(act) < 3:
        continue

    # position comparison
    pos_act = act["res_pos"].values
    pos_noact = noact["res_pos"].values
    stat, pval = stats.mannwhitneyu(pos_act, pos_noact, alternative="two-sided")

    # also: correlation between position and novel_path_score
    rho, rho_p = stats.spearmanr(g["res_pos"], g["novel_path_score"])

    print(f"\n  {gene} (n={len(g)}, {len(act)} activated):")
    print(f"    Activated pos: median={np.median(pos_act):.0f} "
          f"(range {pos_act.min()}-{pos_act.max()})")
    print(f"    Non-activated: median={np.median(pos_noact):.0f} "
          f"(range {pos_noact.min()}-{pos_noact.max()})")
    print(f"    Mann-Whitney p={pval:.4f}")
    print(f"    Position-score Spearman rho={rho:.3f}, p={rho_p:.4f}")

    # bin by position quartiles
    g_sorted = g.copy()
    g_sorted["pos_quartile"] = pd.qcut(g_sorted["res_pos"], q=4, labels=False,
                                         duplicates="drop")
    for q in sorted(g_sorted["pos_quartile"].unique()):
        qg = g_sorted[g_sorted["pos_quartile"] == q]
        lo, hi = qg["res_pos"].min(), qg["res_pos"].max()
        act_rate = qg["activated"].mean()
        mean_score = qg["novel_path_score"].mean()
        print(f"    Q{q} (pos {lo}-{hi}): {len(qg)} VUS, "
              f"activation rate={act_rate:.2%}, mean_score={mean_score:.3f}")

# also do this for known pathogenic (ground truth gradient)
print(f"\n  ── Known pathogenic Gly-sub (ground truth) ──")
for gene in ["COL1A1", "COL1A2", "COL3A1", "COL4A3", "COL4A5"]:
    kg = known_gly[known_gly["gene"] == gene]
    if len(kg) < 10:
        continue
    rho, rho_p = stats.spearmanr(kg["res_pos_known"], kg["novel_path_score"])
    print(f"  {gene}: n={len(kg)}, pos-score rho={rho:.3f}, p={rho_p:.4f}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Substituting amino acid effect
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 2: SUBSTITUTING AMINO ACID EFFECT")
print(f"{'='*70}")

# overall: activation rate by to_aa
print("\n  Activation rate by substituting amino acid (all genes):")
aa_stats = []
for aa, grp in gly_vus.groupby("to_aa"):
    if len(grp) < 10 or aa is None:
        continue
    aa1 = AA_3TO1.get(aa, "?")
    prop = AA_PROPERTIES.get(aa1, "unknown")
    size = AA_SIZE.get(aa1, 0)
    act_rate = grp["activated"].mean()
    mean_score = grp["novel_path_score"].mean()
    aa_stats.append({
        "to_aa": aa, "to_aa_1": aa1, "property": prop, "size": size,
        "n": len(grp), "act_rate": act_rate, "mean_score": mean_score,
    })

aa_df = pd.DataFrame(aa_stats).sort_values("act_rate", ascending=False)
print(f"  {'AA':>5s} {'1let':>4s} {'prop':>12s} {'size':>4s} {'n':>5s} "
      f"{'act%':>6s} {'score':>7s}")
for _, r in aa_df.iterrows():
    print(f"  {r['to_aa']:>5s} {r['to_aa_1']:>4s} {r['property']:>12s} "
          f"{int(r['size']):4d} {int(r['n']):5d} "
          f"{r['act_rate']:6.1%} {r['mean_score']:7.3f}")

# correlation: size of substituting AA vs activation
gly_with_size = gly_vus[gly_vus["to_aa_size"].notna()]
rho, pval = stats.spearmanr(gly_with_size["to_aa_size"],
                              gly_with_size["novel_path_score"])
print(f"\n  AA size vs novel_path_score: Spearman rho={rho:.3f}, p={pval:.4g}")

# property groups
print("\n  By AA property group:")
for prop, grp in gly_vus.groupby("to_aa_prop"):
    if len(grp) < 10 or prop is None:
        continue
    print(f"    {prop:>12s}: n={len(grp):4d}, act_rate={grp['activated'].mean():.1%}, "
          f"mean_score={grp['novel_path_score'].mean():.3f}")

# same analysis on known pathogenic (ground truth)
print(f"\n  ── Known pathogenic: AA effect (ground truth) ──")
for aa, grp in known_gly.groupby("to_aa"):
    if len(grp) < 5 or aa is None:
        continue
    aa1 = AA_3TO1.get(aa, "?")
    print(f"    Gly→{aa} ({aa1}): n={len(grp)}, "
          f"mean_novel_score={grp['novel_path_score'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Per-feature specificity
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 3: PER-FEATURE SPECIFICITY (what does each feature respond to?)")
print(f"{'='*70}")

feat_profiles = []
for fi in novel_feat_ids:
    a = np.abs(vus_acts[:, fi])
    active = a > 1e-6
    active_vus = gly_vus[active[gly_vus.index] if len(gly_vus) == len(vus) else
                          active[:len(gly_vus)]]
    # need to align indices properly
    vus_idx = gly_vus.index.values
    feat_active_mask = a[vus_idx] > 1e-6
    active_gly = gly_vus[feat_active_mask]

    if len(active_gly) < 3:
        continue

    # what genes does this feature prefer?
    gene_counts = active_gly["gene"].value_counts()
    top_gene = gene_counts.index[0] if len(gene_counts) > 0 else "?"
    top_gene_frac = gene_counts.iloc[0] / len(active_gly) if len(gene_counts) > 0 else 0

    # what AA substitution does it prefer?
    aa_counts = active_gly["to_aa"].value_counts()
    top_aa = aa_counts.index[0] if len(aa_counts) > 0 else "?"

    # position stats
    pos_med = active_gly["res_pos"].median()

    # AA size preference
    mean_size = active_gly["to_aa_size"].mean()

    feat_profiles.append({
        "feature": fi,
        "n_gly_active": len(active_gly),
        "top_gene": top_gene,
        "top_gene_frac": top_gene_frac,
        "n_genes": active_gly["gene"].nunique(),
        "top_aa": top_aa,
        "mean_to_aa_size": mean_size,
        "median_pos": pos_med,
    })

fp_df = pd.DataFrame(feat_profiles)
if len(fp_df) > 0:
    print(f"\n  {len(fp_df)} novel pathogenic features active on Gly-sub VUS:")
    print(f"  {'feat':>5s} {'n_gly':>5s} {'top_gene':>10s} {'gfrac':>6s} "
          f"{'n_genes':>7s} {'top_aa':>6s} {'aa_size':>7s} {'med_pos':>8s}")
    for _, r in fp_df.sort_values("n_gly_active", ascending=False).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_gly_active']):5d} "
              f"{r['top_gene']:>10s} {r['top_gene_frac']:6.1%} "
              f"{int(r['n_genes']):7d} {r['top_aa']:>6s} "
              f"{r['mean_to_aa_size']:7.0f} {r['median_pos']:8.0f}")

    # cluster features by gene preference
    print(f"\n  Feature clusters by gene preference:")
    for gene in sorted(fp_df["top_gene"].unique()):
        feats = fp_df[fp_df["top_gene"] == gene]
        print(f"    {gene}: {len(feats)} features "
              f"(feats: {sorted(feats['feature'].astype(int).tolist())})")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4: What distinguishes activated vs non-activated Gly VUS?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 4: MULTIVARIATE — WHAT PREDICTS NOVEL FEATURE ACTIVATION?")
print(f"{'='*70}")

gly_model = gly_vus[gly_vus["to_aa_size"].notna() & gly_vus["res_pos"].notna()].copy()
gly_model["is_collagen"] = gly_model["gene"].isin(COLLAGEN_GENES).astype(int)

print(f"\n  Variables predicting activation (n={len(gly_model)}):")
for var in ["res_pos", "to_aa_size", "is_collagen"]:
    vals = gly_model[var].values.astype(float)
    act = gly_model["activated"].values.astype(float)
    rho, pval = stats.spearmanr(vals, act)
    # also point-biserial
    act_vals = vals[gly_model["activated"]]
    noact_vals = vals[~gly_model["activated"]]
    t, tp = stats.mannwhitneyu(act_vals, noact_vals, alternative="two-sided")
    print(f"  {var:15s}: rho={rho:.3f} (p={pval:.4g}), "
          f"act_mean={act_vals.mean():.1f} vs noact_mean={noact_vals.mean():.1f}, "
          f"MWU p={tp:.4g}")

# interaction: position × AA size
print(f"\n  Interaction: position × AA size")
gly_model["pos_norm"] = (gly_model["res_pos"] - gly_model["res_pos"].mean()) / gly_model["res_pos"].std()
gly_model["size_norm"] = (gly_model["to_aa_size"] - gly_model["to_aa_size"].mean()) / gly_model["to_aa_size"].std()
gly_model["pos_x_size"] = gly_model["pos_norm"] * gly_model["size_norm"]
rho, pval = stats.spearmanr(gly_model["pos_x_size"], gly_model["novel_path_score"])
print(f"    pos×size vs score: rho={rho:.3f}, p={pval:.4g}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 5: Gly-X-Y triplet position
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 5: Gly-X-Y TRIPLET POSITION")
print(f"{'='*70}")

# in collagen triple helix, Gly appears every 3rd residue
# the position within the repeat (Gly at pos%3==0, X at pos%3==1, Y at pos%3==2)
# is determined by the start of the collagen domain
# approximate: residue position mod 3
gly_vus_col = gly_vus[gly_vus["gene"].isin(COLLAGEN_GENES)].copy()
gly_vus_col["pos_mod3"] = gly_vus_col["res_pos"] % 3

print(f"\n  Collagen Gly-sub VUS by position mod 3:")
for m3, grp in gly_vus_col.groupby("pos_mod3"):
    act_rate = grp["activated"].mean()
    mean_score = grp["novel_path_score"].mean()
    print(f"    pos%3 == {m3}: n={len(grp)}, act_rate={act_rate:.1%}, "
          f"mean_score={mean_score:.3f}")

# ── Save ─────────────────────────────────────────────────────────
gly_vus.to_csv(os.path.join(OUT, "gly_vus_detailed.csv"), index=False)
aa_df.to_csv(os.path.join(OUT, "aa_substitution_effects.csv"), index=False)
if len(fp_df) > 0:
    fp_df.to_csv(os.path.join(OUT, "novel_feature_profiles.csv"), index=False)
print(f"\nSaved to {OUT}/")
