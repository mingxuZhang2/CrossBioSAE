"""
Biological characterization of Gly-specific SAE features.

For each of the 4 Gly-specific features (316, 690, 1621, 805):
1. What variants activate it? (gene, domain, position, substituting AA)
2. What biological mechanism does it correspond to?
3. Do features map to different collagen domains or disease phenotypes?
4. Can we give each feature a biological name?
"""

import os, re, gzip
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

OUT = "results/gly_validation"
os.makedirs(OUT, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────
print("Loading data ...")
clinvar = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
acts = np.load("results/sae_genomewide/sae_genomewide_acts.npz")["acts"]
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")
y = clinvar["label"].astype(int).values

COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
}

# collagen domain boundaries (approximate, from UniProt)
# format: {gene: [(domain_name, start, end), ...]}
COLLAGEN_DOMAINS = {
    "COL1A1": [("signal", 1, 22), ("N-propeptide", 23, 161),
               ("triple_helix", 179, 1192), ("C-propeptide", 1193, 1464)],
    "COL1A2": [("signal", 1, 22), ("N-propeptide", 23, 79),
               ("triple_helix", 80, 1102), ("C-propeptide", 1103, 1366)],
    "COL2A1": [("signal", 1, 25), ("N-propeptide", 26, 181),
               ("triple_helix", 201, 1214), ("C-propeptide", 1215, 1487)],
    "COL3A1": [("signal", 1, 26), ("N-propeptide", 27, 168),
               ("triple_helix", 169, 1196), ("C-propeptide", 1197, 1466)],
    "COL4A3": [("7S_domain", 1, 115), ("triple_helix", 116, 1380),
               ("NC1_domain", 1381, 1670)],
    "COL4A4": [("7S_domain", 1, 115), ("triple_helix", 116, 1380),
               ("NC1_domain", 1381, 1690)],
    "COL4A5": [("7S_domain", 1, 115), ("triple_helix", 116, 1405),
               ("NC1_domain", 1406, 1685)],
    "COL5A1": [("signal", 1, 27), ("N-propeptide", 28, 505),
               ("triple_helix", 530, 1530), ("C-propeptide", 1531, 1838)],
    "COL7A1": [("NC1_domain", 1, 1253), ("triple_helix", 1254, 2784),
               ("NC2_domain", 2785, 2944)],
}

# known phenotypes by gene
GENE_DISEASES = {
    "COL1A1": "OI (osteogenesis imperfecta) + EDS",
    "COL1A2": "OI type I-IV",
    "COL2A1": "Stickler/achondrogenesis/spondyloepiphyseal",
    "COL3A1": "vEDS (vascular Ehlers-Danlos)",
    "COL4A3": "Alport syndrome / FSGS",
    "COL4A4": "Alport syndrome",
    "COL4A5": "X-linked Alport syndrome",
    "COL5A1": "classical EDS",
    "COL5A2": "classical EDS",
    "COL7A1": "dystrophic epidermolysis bullosa",
    "COL11A1": "Stickler/Marshall syndrome",
    "COL11A2": "Stickler/OSMED",
}

# map genes + pchanges
print("Mapping variants to genes and protein changes ...")
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

def parse_pchange(pc):
    if not pc or pd.isna(pc):
        return None, None, None
    m = re.match(r'([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|Ter|\*)', str(pc))
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)

clinvar["from_aa"], clinvar["res_pos"], clinvar["to_aa"] = zip(
    *clinvar["pchange"].apply(parse_pchange))

def get_domain(gene, pos):
    if gene not in COLLAGEN_DOMAINS or pos is None:
        return "unknown"
    for name, start, end in COLLAGEN_DOMAINS[gene]:
        if start <= pos <= end:
            return name
    return "other"

clinvar["domain"] = [get_domain(g, p) for g, p in
                      zip(clinvar["gene"], clinvar["res_pos"])]

# ══════════════════════════════════════════════════════════════════
# DEEP CHARACTERIZATION OF EACH Gly-SPECIFIC FEATURE
# ══════════════════════════════════════════════════════════════════
GLY_FEATS = [316, 690, 1621, 805]

for fi in GLY_FEATS:
    print(f"\n{'='*70}")
    print(f"FEATURE {fi}: BIOLOGICAL CHARACTERIZATION")
    print(f"{'='*70}")

    a = np.abs(acts[:, fi])
    active = a > 1e-6
    n_act = active.sum()

    active_df = clinvar[active].copy()
    active_df["feat_strength"] = a[active]

    # ── Basic stats ──
    n_path = (active_df["label"] == True).sum()
    n_ben = (active_df["label"] == False).sum()
    print(f"\n  Active variants: {n_act} ({n_path} P, {n_ben} B)")
    print(f"  Pathogenic rate: {n_path/n_act:.1%}")

    # ── Gene distribution ──
    print(f"\n  Gene distribution:")
    gene_counts = active_df["gene"].value_counts()
    for gene, count in gene_counts.head(15).items():
        frac = count / n_act
        path_rate = active_df[active_df["gene"] == gene]["label"].mean()
        disease = GENE_DISEASES.get(gene, "")
        print(f"    {gene:10s}: {count:4d} ({frac:5.1%}) "
              f"pathogenic_rate={path_rate:.0%}"
              f"  [{disease}]" if disease else "")

    # ── Is it collagen-specific? ──
    n_collagen = active_df["gene"].isin(COLLAGEN_GENES).sum()
    print(f"\n  Collagen genes: {n_collagen}/{n_act} ({100*n_collagen/n_act:.1f}%)")

    # ── Protein change pattern ──
    gly_subs = active_df[active_df["from_aa"] == "Gly"]
    non_gly = active_df[active_df["from_aa"] != "Gly"]
    print(f"\n  Gly substitutions: {len(gly_subs)}/{n_act} ({100*len(gly_subs)/n_act:.1f}%)")

    if len(gly_subs) > 5:
        # to_aa distribution
        print(f"  Gly→ substitutions:")
        for aa, count in gly_subs["to_aa"].value_counts().head(8).items():
            print(f"    Gly→{aa}: {count} ({100*count/len(gly_subs):.1f}%)")

    # ── Domain distribution (collagen variants only) ──
    col_active = active_df[active_df["gene"].isin(COLLAGEN_GENES)]
    if len(col_active) > 5:
        print(f"\n  Domain distribution (collagen genes, n={len(col_active)}):")
        for dom, count in col_active["domain"].value_counts().items():
            frac = count / len(col_active)
            # what's the pathogenic rate in this domain?
            dom_path = col_active[col_active["domain"] == dom]["label"].mean()
            print(f"    {dom:15s}: {count:4d} ({frac:5.1%}), pathogenic={dom_path:.0%}")

    # ── Position distribution in triple helix ──
    th_active = col_active[col_active["domain"] == "triple_helix"]
    if len(th_active) > 10 and th_active["res_pos"].notna().sum() > 10:
        positions = th_active["res_pos"].dropna().astype(int)
        print(f"\n  Triple helix positions (n={len(positions)}):")
        print(f"    Range: {positions.min()} - {positions.max()}")
        print(f"    Median: {positions.median():.0f}")
        print(f"    Mean: {positions.mean():.0f}")

        # N-terminal half vs C-terminal half
        mid = (positions.min() + positions.max()) / 2
        n_nterm = (positions < mid).sum()
        n_cterm = (positions >= mid).sum()
        print(f"    N-terminal half: {n_nterm} ({100*n_nterm/len(positions):.1f}%)")
        print(f"    C-terminal half: {n_cterm} ({100*n_cterm/len(positions):.1f}%)")

    # ── Top gene concentration ──
    if len(gene_counts) > 0:
        top_gene = gene_counts.index[0]
        top_frac = gene_counts.iloc[0] / n_act
        print(f"\n  Most concentrated gene: {top_gene} ({top_frac:.1%})")

    # ── Non-collagen variants (what else does this feature capture?) ──
    non_col = active_df[~active_df["gene"].isin(COLLAGEN_GENES)]
    if len(non_col) > 10:
        print(f"\n  Non-collagen activated variants: {len(non_col)}")
        nc_genes = non_col["gene"].value_counts().head(5)
        for gene, count in nc_genes.items():
            disease = GENE_DISEASES.get(gene, "")
            print(f"    {gene:10s}: {count:3d}"
                  f"  [{disease}]" if disease else f"    {gene:10s}: {count:3d}")

    # ── Strongest activations ──
    top_strong = active_df.nlargest(10, "feat_strength")
    print(f"\n  Top 10 strongest activations:")
    for _, r in top_strong.iterrows():
        label = "P" if r["label"] else "B"
        esm = f"ESM={r['ESM-1b']:.1f}" if pd.notna(r.get("ESM-1b")) else "ESM=NA"
        print(f"    {r['gene'] or '?':10s} {r['pchange'] or '?':15s} "
              f"strength={r['feat_strength']:.3f} {label} {esm} domain={r['domain']}")

# ══════════════════════════════════════════════════════════════════
# CROSS-FEATURE COMPARISON: do they capture different biology?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CROSS-FEATURE COMPARISON: DIFFERENT BIOLOGY?")
print(f"{'='*70}")

# overlap: how many variants are activated by multiple features?
feat_masks = {}
for fi in GLY_FEATS:
    feat_masks[fi] = np.abs(acts[:, fi]) > 1e-6

# pairwise overlap
print("\n  Pairwise Jaccard similarity:")
for i, f1 in enumerate(GLY_FEATS):
    for f2 in GLY_FEATS[i+1:]:
        inter = (feat_masks[f1] & feat_masks[f2]).sum()
        union = (feat_masks[f1] | feat_masks[f2]).sum()
        jaccard = inter / max(union, 1)
        print(f"    feat {f1} vs {f2}: Jaccard={jaccard:.3f} "
              f"(overlap={inter}, union={union})")

# gene preference comparison
print("\n  Gene preference (top 3 genes per feature):")
for fi in GLY_FEATS:
    active_df = clinvar[feat_masks[fi]]
    col_active = active_df[active_df["gene"].isin(COLLAGEN_GENES)]
    top3 = col_active["gene"].value_counts().head(3)
    genes_str = ", ".join(f"{g}({c})" for g, c in top3.items())
    print(f"    feat {fi}: {genes_str}")

# AA substitution preference
print("\n  Gly→X preference (top 3 AA per feature):")
for fi in GLY_FEATS:
    active_gly = clinvar[feat_masks[fi] & (clinvar["from_aa"] == "Gly")]
    if len(active_gly) > 5:
        top3 = active_gly["to_aa"].value_counts().head(3)
        aa_str = ", ".join(f"→{a}({c})" for a, c in top3.items())
        print(f"    feat {fi}: {aa_str}")

# domain preference
print("\n  Domain preference (collagen variants):")
for fi in GLY_FEATS:
    col_active = clinvar[feat_masks[fi] & clinvar["gene"].isin(COLLAGEN_GENES)]
    if len(col_active) > 5:
        dom_counts = col_active["domain"].value_counts()
        dom_str = ", ".join(f"{d}({c})" for d, c in dom_counts.items()
                            if d != "unknown")
        print(f"    feat {fi}: {dom_str}")

# ══════════════════════════════════════════════════════════════════
# BIOLOGICAL INTERPRETATION SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PROPOSED BIOLOGICAL NAMES")
print(f"{'='*70}")

# this will be filled based on the analysis above
# for now, compute key statistics for each feature
for fi in GLY_FEATS:
    active_df = clinvar[feat_masks[fi]].copy()
    col_df = active_df[active_df["gene"].isin(COLLAGEN_GENES)]
    gly_df = active_df[active_df["from_aa"] == "Gly"]

    n_total = len(active_df)
    n_col = len(col_df)
    n_gly = len(gly_df)
    frac_col = n_col / max(n_total, 1)
    frac_gly = n_gly / max(n_total, 1)

    top_gene = col_df["gene"].value_counts().index[0] if len(col_df) > 0 else "?"
    top_aa = gly_df["to_aa"].value_counts().index[0] if len(gly_df) > 0 else "?"
    top_dom = col_df["domain"].value_counts().index[0] if len(col_df) > 0 else "?"

    print(f"\n  Feature {fi}:")
    print(f"    Profile: {frac_col:.0%} collagen, {frac_gly:.0%} Gly-sub")
    print(f"    Top gene: {top_gene}, Top AA: Gly→{top_aa}, Top domain: {top_dom}")
    print(f"    Disease: {GENE_DISEASES.get(top_gene, 'unknown')}")

print(f"\nSaved to {OUT}/")
