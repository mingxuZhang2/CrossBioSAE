"""
Concept interpretability validation + mechanism-aware VUS application.

Part 1: Validate that feature-defined mechanisms correspond to known biology
  - Gly→Asp/Glu (feat 690) vs Gly→Arg/Ser (feat 1621): severity difference?
  - In OI: Gly→Asp is more lethal than Gly→Ser (known clinical fact)
  - Does feature activation strength correlate with this?

Part 2: Application — mechanism-aware VUS annotation
  - For each VUS: which mechanism feature fires → WHY it's likely pathogenic
  - ESM-1b says "damaging" (a number), we say "damaging via charge disruption" (a mechanism)
  - Show this provides actionable information ESM-1b doesn't
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
vus = pd.read_csv("results/vus_pilot/vus_mechanism_profiles.csv")
vus_acts = np.load("results/vus_pilot/vus_sae_acts.npz")["acts"]
y = clinvar["label"].astype(int).values

COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
}

# feature definitions (from characterization results)
FEAT_ACID = 690    # Gly→Asp/Glu: charge disruption
FEAT_BASIC = 1621  # Gly→Arg/Ser: steric/polarity disruption
FEAT_MIXED = 316   # Gly→Asp/Glu/Cys: N-terminal charged
FEAT_4 = 805       # 4th feature

MECHANISM_FEATURES = {
    FEAT_ACID: {"name": "charge_disruption", "desc": "Gly→acidic (Asp/Glu): introduces negative charge into triple helix"},
    FEAT_BASIC: {"name": "steric_polarity", "desc": "Gly→basic/polar (Arg/Ser): steric clash and polarity change"},
    FEAT_MIXED: {"name": "mixed_charge", "desc": "Gly→charged/special (Asp/Glu/Cys): N-terminal triple helix"},
    FEAT_4: {"name": "type_iv_specific", "desc": "Concentrated in type IV collagen (basement membrane)"},
}

# map genes
print("Mapping variants ...")
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

def parse_gly_sub(pc):
    if not pc or pd.isna(pc):
        return None, None
    m = re.match(r'Gly(\d+)([A-Z][a-z]{2})', str(pc))
    if not m or m.group(2) == "Gly":
        return None, None
    return int(m.group(1)), m.group(2)

clinvar["gly_pos"], clinvar["gly_to_aa"] = zip(*clinvar["pchange"].apply(parse_gly_sub))

# compute per-mechanism activations
for fi, info in MECHANISM_FEATURES.items():
    clinvar[info["name"]] = np.abs(acts[:, fi])

# ══════════════════════════════════════════════════════════════════
# PART 1: CONCEPT VALIDATION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART 1: CONCEPT VALIDATION — DO FEATURES MAP TO KNOWN BIOLOGY?")
print(f"{'='*70}")

# known pathogenic collagen Gly-subs
col_gly_path = clinvar[
    (y == 1) & clinvar["gene"].isin(COLLAGEN_GENES) &
    clinvar["gly_to_aa"].notna()
].copy()
print(f"\n  Known pathogenic collagen Gly-subs: {len(col_gly_path)}")

# ── 1A: Feature activation by substituting AA ────────────────────
print(f"\n  ── 1A: Feature activation by substituting amino acid ──")

AA_GROUPS = {
    "acidic": ["Asp", "Glu"],
    "basic_polar": ["Arg", "Ser"],
    "hydrophobic": ["Val", "Ala", "Leu", "Ile"],
    "special": ["Cys", "Trp"],
}

print(f"\n  {'AA_group':15s} {'n':>5s} | {'feat690':>8s} {'feat1621':>8s} {'feat316':>8s} | {'ratio_690/1621':>14s}")
for group_name, aas in AA_GROUPS.items():
    mask = col_gly_path["gly_to_aa"].isin(aas)
    sub = col_gly_path[mask]
    if len(sub) < 5:
        continue
    act_690 = (sub["charge_disruption"] > 0).mean()
    act_1621 = (sub["steric_polarity"] > 0).mean()
    act_316 = (sub["mixed_charge"] > 0).mean()
    ratio = act_690 / max(act_1621, 1e-8)
    print(f"  {group_name:15s} {len(sub):5d} | {act_690:8.1%} {act_1621:8.1%} {act_316:8.1%} | {ratio:14.2f}")

# per individual AA
print(f"\n  Per amino acid:")
print(f"  {'Gly->':>8s} {'n':>5s} | {'f690(acid)':>10s} {'f1621(basic)':>12s} {'ratio':>8s}")
for aa in ["Asp", "Glu", "Arg", "Ser", "Val", "Cys", "Trp", "Ala"]:
    sub = col_gly_path[col_gly_path["gly_to_aa"] == aa]
    if len(sub) < 3:
        continue
    a690 = (sub["charge_disruption"] > 0).mean()
    a1621 = (sub["steric_polarity"] > 0).mean()
    ratio = a690 / max(a1621, 1e-8) if a1621 > 0 else float('inf')
    print(f"  Gly→{aa:3s} {len(sub):5d} | {a690:10.1%} {a1621:12.1%} {ratio:8.1f}")

# statistical test: feat 690 prefers acidic, feat 1621 prefers basic
acidic_mask = col_gly_path["gly_to_aa"].isin(["Asp", "Glu"])
basic_mask = col_gly_path["gly_to_aa"].isin(["Arg", "Ser"])

if acidic_mask.sum() > 10 and basic_mask.sum() > 10:
    # feat 690: acidic vs basic activation rate
    a690_acid = (col_gly_path.loc[acidic_mask, "charge_disruption"] > 0).mean()
    a690_basic = (col_gly_path.loc[basic_mask, "charge_disruption"] > 0).mean()
    from scipy.stats import fisher_exact
    # 2x2: [acid_active, acid_inactive], [basic_active, basic_inactive]
    a = (col_gly_path.loc[acidic_mask, "charge_disruption"] > 0).sum()
    b = (col_gly_path.loc[acidic_mask, "charge_disruption"] == 0).sum()
    c = (col_gly_path.loc[basic_mask, "charge_disruption"] > 0).sum()
    d = (col_gly_path.loc[basic_mask, "charge_disruption"] == 0).sum()
    _, p690 = fisher_exact([[a, b], [c, d]], alternative="greater")

    # feat 1621: basic vs acidic
    a = (col_gly_path.loc[basic_mask, "steric_polarity"] > 0).sum()
    b = (col_gly_path.loc[basic_mask, "steric_polarity"] == 0).sum()
    c = (col_gly_path.loc[acidic_mask, "steric_polarity"] > 0).sum()
    d = (col_gly_path.loc[acidic_mask, "steric_polarity"] == 0).sum()
    _, p1621 = fisher_exact([[a, b], [c, d]], alternative="greater")

    print(f"\n  Statistical validation:")
    print(f"    Feat 690 prefers acidic (Asp/Glu) over basic (Arg/Ser): "
          f"{a690_acid:.1%} vs {a690_basic:.1%}, Fisher p={p690:.4g}")
    print(f"    Feat 1621 prefers basic (Arg/Ser) over acidic (Asp/Glu): "
          f"{(col_gly_path.loc[basic_mask, 'steric_polarity'] > 0).mean():.1%} vs "
          f"{(col_gly_path.loc[acidic_mask, 'steric_polarity'] > 0).mean():.1%}, Fisher p={p1621:.4g}")

# ── 1B: Gene-level mechanism profile ─────────────────────────────
print(f"\n  ── 1B: Gene-level mechanism profile ──")
print(f"\n  {'gene':10s} {'n':>4s} | {'acid(690)':>9s} {'basic(1621)':>11s} | {'dominant':>10s} | disease")

GENE_DISEASES = {
    "COL1A1": "OI (lethal/severe)",
    "COL1A2": "OI (mild/moderate)",
    "COL2A1": "skeletal dysplasia",
    "COL3A1": "vascular EDS",
    "COL4A3": "Alport/FSGS",
    "COL4A4": "Alport",
    "COL4A5": "X-linked Alport",
    "COL5A1": "classical EDS",
    "COL7A1": "dystrophic EB",
    "COL11A1": "Stickler",
}

for gene in sorted(col_gly_path["gene"].unique()):
    g = col_gly_path[col_gly_path["gene"] == gene]
    if len(g) < 5:
        continue
    a690 = (g["charge_disruption"] > 0).mean()
    a1621 = (g["steric_polarity"] > 0).mean()
    dominant = "acid" if a690 > a1621 else "basic" if a1621 > a690 else "equal"
    disease = GENE_DISEASES.get(gene, "")
    print(f"  {gene:10s} {len(g):4d} | {a690:9.1%} {a1621:11.1%} | {dominant:>10s} | {disease}")

# ── 1C: Known severity correlation ───────────────────────────────
print(f"\n  ── 1C: Severity correlation (ESM-1b as proxy) ──")
# among known pathogenic Gly-subs, does mechanism type correlate with ESM-1b severity?
# (more negative ESM-1b = more damaging)
acid_active = col_gly_path[col_gly_path["charge_disruption"] > 0]
basic_active = col_gly_path[col_gly_path["steric_polarity"] > 0]
both_active = col_gly_path[(col_gly_path["charge_disruption"] > 0) &
                             (col_gly_path["steric_polarity"] > 0)]
neither = col_gly_path[(col_gly_path["charge_disruption"] == 0) &
                         (col_gly_path["steric_polarity"] == 0)]

for name, sub in [("Acid-feature active", acid_active),
                    ("Basic-feature active", basic_active),
                    ("Both active", both_active),
                    ("Neither active", neither)]:
    esm = sub["ESM-1b"].dropna()
    if len(esm) > 5:
        print(f"    {name:25s}: n={len(esm):4d}, ESM-1b={esm.mean():.2f} ± {esm.std():.2f}")

if len(acid_active) > 10 and len(basic_active) > 10:
    esm_acid = acid_active["ESM-1b"].dropna()
    esm_basic = basic_active["ESM-1b"].dropna()
    t, p = stats.mannwhitneyu(esm_acid, esm_basic, alternative="two-sided")
    print(f"\n    ESM-1b: acid vs basic mechanism: Mann-Whitney p={p:.4g}")
    print(f"    Acid mean={esm_acid.mean():.2f}, Basic mean={esm_basic.mean():.2f}")

# ══════════════════════════════════════════════════════════════════
# PART 2: APPLICATION — MECHANISM-AWARE VUS ANNOTATION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART 2: APPLICATION — MECHANISM-AWARE VUS ANNOTATION")
print(f"{'='*70}")

gly_vus = vus[vus["is_gly_sub"]].copy()

# compute mechanism scores for VUS
for fi, info in MECHANISM_FEATURES.items():
    gly_vus[info["name"]] = np.abs(vus_acts[gly_vus.index, fi])

# assign mechanism type
def assign_mechanism(row):
    acid = row["charge_disruption"] > 0
    basic = row["steric_polarity"] > 0
    mixed = row["mixed_charge"] > 0
    t4 = row["type_iv_specific"] > 0

    mechanisms = []
    if acid:
        mechanisms.append("charge_disruption")
    if basic:
        mechanisms.append("steric_polarity")
    if mixed:
        mechanisms.append("mixed_charge")
    if t4:
        mechanisms.append("type_iv_specific")

    if not mechanisms:
        return "no_mechanism_signal"
    return "+".join(mechanisms)

gly_vus["mechanism"] = gly_vus.apply(assign_mechanism, axis=1)

# parse to_aa for VUS
def get_to_aa(pc):
    if not pc or pd.isna(pc):
        return None
    m = re.match(r'Gly\d+([A-Z][a-z]{2})', str(pc))
    return m.group(1) if m else None

gly_vus["to_aa"] = gly_vus["pchange"].apply(get_to_aa)

# overview
print(f"\n  VUS mechanism annotation:")
mech_counts = gly_vus["mechanism"].value_counts()
for mech, count in mech_counts.items():
    frac = count / len(gly_vus)
    print(f"    {mech:35s}: {count:4d} ({frac:5.1%})")

# VUS with mechanism signal
has_mech = gly_vus[gly_vus["mechanism"] != "no_mechanism_signal"]
print(f"\n  VUS with mechanism annotation: {len(has_mech)}/{len(gly_vus)} ({100*len(has_mech)/len(gly_vus):.1f}%)")

# ── 2A: Mechanism consistency check ──────────────────────────────
print(f"\n  ── 2A: Mechanism-AA consistency (does the right feature fire?) ──")

for aa_group, expected_mech, aas in [
    ("acidic (Asp/Glu)", "charge_disruption", ["Asp", "Glu"]),
    ("basic/polar (Arg/Ser)", "steric_polarity", ["Arg", "Ser"]),
]:
    aa_vus = has_mech[has_mech["to_aa"].isin(aas)]
    if len(aa_vus) == 0:
        continue
    correct = aa_vus["mechanism"].str.contains(expected_mech).sum()
    print(f"    {aa_group}: {len(aa_vus)} VUS with mechanism, "
          f"{correct} ({100*correct/len(aa_vus):.1f}%) activate expected {expected_mech}")

# ── 2B: Example VUS annotations ─────────────────────────────────
print(f"\n  ── 2B: Example mechanism-aware VUS annotations ──")

top_annotated = has_mech.nlargest(30, "novel_path_score")
print(f"\n  {'gene':10s} {'pchange':>15s} {'mechanism':>35s} {'to_aa':>6s} {'score':>7s}")
for _, r in top_annotated.iterrows():
    gene = str(r["gene"]) if pd.notna(r["gene"]) else "?"
    pc = str(r["pchange"]) if pd.notna(r["pchange"]) else "?"
    aa = str(r["to_aa"]) if pd.notna(r["to_aa"]) else "?"
    print(f"  {gene:10s} {pc:>15s} {r['mechanism']:>35s} {aa:>6s} {r['novel_path_score']:7.3f}")

# ── 2C: What ESM-1b can NOT tell you ─────────────────────────────
print(f"\n  ── 2C: Information gain over ESM-1b ──")
print(f"\n  ESM-1b says: 'variant X has LLR = -15.3 (damaging)'")
print(f"  Our method says: 'variant X activates charge_disruption feature →")
print(f"    pathogenic via Gly→acidic substitution disrupting triple helix H-bonds'")
print(f"\n  Concrete examples of mechanism differentiation in same gene:")

# find examples where same gene has VUS with different mechanisms
for gene in ["COL4A3", "COL3A1", "COL1A1", "COL5A1", "COL4A5"]:
    gene_vus = has_mech[has_mech["gene"] == gene]
    mechs = gene_vus["mechanism"].unique()
    if len(mechs) >= 2:
        print(f"\n    {gene} — {len(gene_vus)} annotated VUS, {len(mechs)} distinct mechanisms:")
        for mech in sorted(mechs):
            m_vus = gene_vus[gene_vus["mechanism"] == mech]
            ex = m_vus.iloc[0]
            aa = str(ex["to_aa"]) if pd.notna(ex["to_aa"]) else "?"
            print(f"      {mech}: n={len(m_vus)}, e.g. {str(ex['pchange'])} (Gly→{aa})")

# ── 2D: Clinical interpretation template ─────────────────────────
print(f"\n  ── 2D: Clinical interpretation template ──")
print(f"""
  For a VUS in a collagen gene with a Gly substitution:

  1. Check which mechanism feature activates:
     - charge_disruption (feat 690): Gly→Asp/Glu
       → Introduces negative charge into tightly packed triple helix
       → Disrupts backbone hydrogen bond network
       → Associated with fibrillar collagen diseases (OI, vEDS)

     - steric_polarity (feat 1621): Gly→Arg/Ser
       → Arg: bulky guanidinium group causes steric clash
       → Ser: hydroxyl group alters local polarity
       → Associated with basement membrane collagenopathies (Alport)

     - mixed_charge (feat 316): Gly→Asp/Glu/Cys, N-terminal bias
       → N-terminal triple helix disruption
       → Cys: introduces disulfide bond potential in normally non-Cys region

  2. Combine with ESM-1b score for confidence:
     - Feature active + ESM-1b damaging → high confidence pathogenic
     - Feature active + ESM-1b mild → mechanism-specific pathogenicity
       (ESM may underestimate structural protein damage)
     - Feature inactive + ESM-1b damaging → non-structural mechanism

  3. Gene-specific interpretation:
     - COL1A1/COL1A2: OI severity correlates with substitution type
     - COL4A3/4/5: Alport vs thin basement membrane nephropathy
     - COL3A1: vEDS — any Gly-sub in triple helix is high-risk
""")

# ── Save annotated VUS ───────────────────────────────────────────
out_cols = ["chrom", "pos", "ref", "alt", "gene", "pchange", "to_aa",
            "mechanism", "charge_disruption", "steric_polarity",
            "mixed_charge", "type_iv_specific", "novel_path_score"]
has_mech[out_cols].to_csv(os.path.join(OUT, "vus_mechanism_annotated.csv"), index=False)

# summary stats
summary = {
    "total_gly_vus": len(gly_vus),
    "annotated_with_mechanism": len(has_mech),
    "annotation_rate": len(has_mech) / len(gly_vus),
    "n_charge_disruption": (gly_vus["charge_disruption"] > 0).sum(),
    "n_steric_polarity": (gly_vus["steric_polarity"] > 0).sum(),
    "feat690_acid_preference_p": p690 if 'p690' in dir() else np.nan,
    "feat1621_basic_preference_p": p1621 if 'p1621' in dir() else np.nan,
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT, "concept_validation_summary.csv"), index=False)

print(f"\nSaved {len(has_mech)} annotated VUS to {OUT}/vus_mechanism_annotated.csv")
