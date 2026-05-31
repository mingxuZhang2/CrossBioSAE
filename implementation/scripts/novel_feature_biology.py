"""
Deep biological analysis of novel pathogenic SAE features.

Key hypothesis: novel features capture structural protein pathogenicity
(collagen Gly-X-Y substitutions, fibrillin domain variants) that existing
single-model scores (ESM-1b, GPN-MSA) miss.

Validation:
1. Map activating variants to protein changes (from variant_summary)
2. Check for Gly substitution enrichment in collagen genes
3. Compare ESM-1b/GPN-MSA scores on these vs non-structural pathogenic variants
4. Count VUS in same genes with same pattern → clinical value estimate
"""

import gzip, os, re, sys
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

OUT = "results/novel_features"
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
acts = np.load("results/sae_genomewide/sae_genomewide_acts.npz")["acts"]
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")
y = df["label"].astype(int).values

# ── Map variants to genes + protein changes ──────────────────────
print("Mapping variants to genes and protein changes ...")
lookup_keys = set(zip(df["chrom"].astype(str), df["pos"].astype(str),
                       df["ref"].astype(str), df["alt"].astype(str)))

variant_info = {}
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        key = (parts[18], parts[31], parts[32], parts[33])
        if key in lookup_keys and key not in variant_info:
            name = parts[2]
            # extract protein change: (p.Gly123Arg) or similar
            pchange = ""
            m = re.search(r'\(p\.([A-Za-z0-9_*=]+)\)', name)
            if m:
                pchange = m.group(1)
            variant_info[key] = {
                "gene": parts[4],
                "name": name,
                "pchange": pchange,
                "type": parts[1],
            }

df["gene"] = [variant_info.get((str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("gene")
              for r in df.itertuples()]
df["pchange"] = [variant_info.get((str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("pchange", "")
                 for r in df.itertuples()]

n_mapped = df["gene"].notna().sum()
n_pchange = (df["pchange"] != "").sum()
print(f"  Mapped: {n_mapped} genes, {n_pchange} protein changes")

# ── Structural protein genes ─────────────────────────────────────
COLLAGEN_GENES = {"COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
                  "COL4A5", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
                  "COL7A1", "COL9A1", "COL11A1", "COL11A2", "COL17A1"}
FIBRILLIN_GENES = {"FBN1", "FBN2"}
STRUCTURAL_GENES = COLLAGEN_GENES | FIBRILLIN_GENES | {"COMP", "FLNB", "FLNA"}

is_structural = df["gene"].isin(STRUCTURAL_GENES).values
is_collagen = df["gene"].isin(COLLAGEN_GENES).values

# check for Gly substitution
def is_gly_sub(pchange):
    if not pchange:
        return False
    return pchange.startswith("Gly") and not pchange.startswith("Gly=")

df["is_gly_sub"] = df["pchange"].apply(is_gly_sub)

print(f"\n  Structural protein variants: {is_structural.sum()}")
print(f"  Collagen variants: {is_collagen.sum()}")
print(f"  Gly substitutions (all genes): {df['is_gly_sub'].sum()}")
print(f"  Gly substitutions in collagen: {(df['is_gly_sub'] & is_collagen).sum()}")

# ── Novel pathogenic features: structural protein enrichment ─────
print(f"\n{'='*70}")
print("NOVEL FEATURE ANALYSIS: STRUCTURAL PROTEIN ENRICHMENT")
print(f"{'='*70}")

novel_path = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
print(f"\n{len(novel_path)} novel pathogenic features (pr > 0.85)")

struct_enrich = []
for _, r in novel_path.sort_values("path_rate", ascending=False).iterrows():
    fi = int(r["feature"])
    a = acts[:, fi]
    active = np.abs(a) > 1e-6
    n_act = active.sum()

    n_struct = (active & is_structural).sum()
    n_collagen = (active & is_collagen).sum()
    n_gly = (active & df["is_gly_sub"].values).sum()
    n_gly_col = (active & df["is_gly_sub"].values & is_collagen).sum()

    # enrichment vs baseline
    base_struct_rate = is_structural.mean()
    feat_struct_rate = n_struct / max(n_act, 1)
    enrichment = feat_struct_rate / max(base_struct_rate, 1e-8)

    # ESM-1b and GPN-MSA scores for structural vs non-structural active variants
    esm_struct = df.loc[active & is_structural, "ESM-1b"].dropna()
    esm_other = df.loc[active & ~is_structural, "ESM-1b"].dropna()
    gpn_struct = df.loc[active & is_structural, "GPN-MSA"].dropna()
    gpn_other = df.loc[active & ~is_structural, "GPN-MSA"].dropna()

    struct_enrich.append({
        "feature": fi, "n_active": n_act, "path_rate": r["path_rate"],
        "n_structural": n_struct, "struct_frac": feat_struct_rate,
        "struct_enrichment": enrichment,
        "n_collagen": n_collagen, "n_gly": n_gly, "n_gly_collagen": n_gly_col,
        "esm_struct": esm_struct.mean() if len(esm_struct) > 0 else np.nan,
        "esm_other": esm_other.mean() if len(esm_other) > 0 else np.nan,
        "gpn_struct": gpn_struct.mean() if len(gpn_struct) > 0 else np.nan,
        "gpn_other": gpn_other.mean() if len(gpn_other) > 0 else np.nan,
    })

se_df = pd.DataFrame(struct_enrich)

print(f"\n  {'feat':>5s} {'n':>5s} {'pr':>5s} {'struct':>6s} {'col':>4s} {'gly':>4s} "
      f"{'gly_col':>7s} {'enrich':>7s} {'ESM_s':>6s} {'ESM_o':>6s} {'GPN_s':>6s} {'GPN_o':>6s}")
for _, r in se_df.sort_values("struct_enrichment", ascending=False).head(15).iterrows():
    print(f"  {int(r['feature']):5d} {int(r['n_active']):5d} {r['path_rate']:5.2f} "
          f"{int(r['n_structural']):6d} {int(r['n_collagen']):4d} {int(r['n_gly']):4d} "
          f"{int(r['n_gly_collagen']):7d} {r['struct_enrichment']:7.1f}x "
          f"{r['esm_struct']:6.1f} {r['esm_other']:6.1f} "
          f"{r['gpn_struct']:6.1f} {r['gpn_other']:6.1f}")

# ── Gly-X-Y pattern in collagen ──────────────────────────────────
print(f"\n{'='*70}")
print("COLLAGEN Gly-X-Y ANALYSIS")
print(f"{'='*70}")

col_vars = df[is_collagen & (y == 1)].copy()
print(f"\n  Pathogenic collagen variants: {len(col_vars)}")
print(f"  With protein change: {(col_vars['pchange'] != '').sum()}")
print(f"  Gly substitutions: {col_vars['is_gly_sub'].sum()} "
      f"({100*col_vars['is_gly_sub'].mean():.1f}%)")

# amino acid substitution pattern in collagen
if len(col_vars) > 0:
    aa_from = Counter()
    for pc in col_vars["pchange"]:
        if pc and len(pc) >= 3:
            aa_from[pc[:3]] += 1
    print(f"\n  Top amino acid substitutions in pathogenic collagen:")
    for aa, count in aa_from.most_common(10):
        print(f"    {aa}: {count} ({100*count/len(col_vars):.1f}%)")

# do novel features fire more on Gly subs in collagen?
print(f"\n  Novel feature activation: Gly-sub collagen vs non-Gly collagen:")
gly_col_mask = is_collagen & df["is_gly_sub"].values & (y == 1)
nongly_col_mask = is_collagen & ~df["is_gly_sub"].values & (y == 1)
if gly_col_mask.sum() > 5 and nongly_col_mask.sum() > 5:
    for fi in se_df.sort_values("struct_enrichment", ascending=False).head(5)["feature"].astype(int):
        a_gly = np.abs(acts[gly_col_mask, fi]).mean()
        a_nongly = np.abs(acts[nongly_col_mask, fi]).mean()
        ratio = a_gly / max(a_nongly, 1e-8)
        print(f"    feat {fi}: Gly-sub={a_gly:.3f}, non-Gly={a_nongly:.3f}, ratio={ratio:.2f}x")

# ── ESM-1b / GPN-MSA blind spot analysis ────────────────────────
print(f"\n{'='*70}")
print("WHY DO EXISTING SCORES MISS STRUCTURAL PROTEINS?")
print(f"{'='*70}")

struct_path = df[is_structural & (y == 1)]
other_path = df[~is_structural & (y == 1)]

for score in ["ESM-1b", "GPN-MSA", "CADD"]:
    s1 = struct_path[score].dropna()
    s2 = other_path[score].dropna()
    if len(s1) > 10 and len(s2) > 10:
        print(f"\n  {score}:")
        print(f"    Structural pathogenic: {s1.mean():.3f} ± {s1.std():.3f} (n={len(s1)})")
        print(f"    Other pathogenic:      {s2.mean():.3f} ± {s2.std():.3f} (n={len(s2)})")
        # are structural protein variants scored LESS pathogenic?
        delta = s1.mean() - s2.mean()
        direction = "LESS damaging" if delta > 0 else "MORE damaging"
        print(f"    Δ = {delta:+.3f} ({direction} by {score})")

# ── VUS in structural protein genes ──────────────────────────────
print(f"\n{'='*70}")
print("VUS IN STRUCTURAL PROTEIN GENES (clinical value)")
print(f"{'='*70}")

import gzip
vus_struct = 0
vus_gly_col = 0
vus_by_gene = Counter()

with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        if parts[1] != "single nucleotide variant":
            continue
        if "Uncertain significance" not in parts[6]:
            continue
        gene = parts[4]
        if gene in STRUCTURAL_GENES:
            vus_struct += 1
            vus_by_gene[gene] += 1
            name = parts[2]
            m = re.search(r'\(p\.Gly\d+', name)
            if m and gene in COLLAGEN_GENES:
                vus_gly_col += 1

print(f"\n  VUS in structural protein genes: {vus_struct}")
print(f"  VUS Gly substitutions in collagen: {vus_gly_col}")
print(f"\n  Top genes by VUS count:")
for gene, count in vus_by_gene.most_common(15):
    print(f"    {gene:10s}: {count:5d} VUS")

print(f"\n  → {vus_gly_col} collagen Gly-sub VUS could be triaged by novel feature activation")

# save
se_df.to_csv(os.path.join(OUT, "novel_structural_enrichment.csv"), index=False)
print(f"\nSaved to {OUT}/")
