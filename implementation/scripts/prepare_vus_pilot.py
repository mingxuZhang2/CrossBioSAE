"""
Prepare VUS pilot dataset for structural protein genes.

Extracts VUS from ClinVar variant_summary for collagen/fibrillin genes,
creates a parquet matching the clinvar.parquet format for downstream Evo2
embedding extraction and SAE mechanism profiling.
"""

import gzip, os, re
import numpy as np
import pandas as pd

OUT = "data/variant/vus_pilot"
os.makedirs(OUT, exist_ok=True)

STRUCTURAL_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4", "COL4A5",
    "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
    "FBN1", "FBN2", "COMP", "FLNA", "FLNB",
}

print("Extracting VUS from variant_summary ...")
records = []
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34:
            continue
        if parts[16] != "GRCh38":
            continue
        if parts[1] != "single nucleotide variant":
            continue
        if "Uncertain significance" not in parts[6]:
            continue
        gene = parts[4]
        if gene not in STRUCTURAL_GENES:
            continue

        chrom = parts[18]
        pos = parts[31]
        ref = parts[32]
        alt = parts[33]
        if len(ref) != 1 or len(alt) != 1:
            continue

        name = parts[2]
        pchange = ""
        m = re.search(r'\(p\.([A-Za-z0-9_*=]+)\)', name)
        if m:
            pchange = m.group(1)

        is_gly = pchange.startswith("Gly") and not pchange.startswith("Gly=")

        records.append({
            "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
            "gene": gene, "pchange": pchange, "is_gly_sub": is_gly,
            "clin_sig": parts[6],
        })

df = pd.DataFrame(records)
df = df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
print(f"  Total structural VUS (deduped): {len(df)}")
print(f"  Gly substitutions: {df['is_gly_sub'].sum()}")
print(f"\n  By gene:")
for gene, grp in df.groupby("gene"):
    n_gly = grp["is_gly_sub"].sum()
    print(f"    {gene:10s}: {len(grp):5d} VUS ({n_gly} Gly-sub)")

# save parquet compatible with clinvar format (for Evo2 extraction)
vus_parquet = df[["chrom", "pos", "ref", "alt"]].copy()
vus_parquet["label"] = False  # placeholder
vus_parquet.to_parquet(os.path.join(OUT, "vus_structural.parquet"), index=False)

# save full annotations
df.to_csv(os.path.join(OUT, "vus_structural_annotated.csv"), index=False)
print(f"\nSaved {len(df)} VUS to {OUT}/")
print(f"  parquet: vus_structural.parquet (for Evo2 extraction)")
print(f"  csv: vus_structural_annotated.csv (with gene/pchange)")

# estimate GPU cost
n_variants = len(df)
batch_size = 2
# ~2 variants per batch, ~0.5s per batch on A800
est_time_minutes = n_variants / (batch_size * 2 * 60)
print(f"\n  Estimated Evo2 extraction time: {est_time_minutes:.0f} min "
      f"({est_time_minutes/8:.0f} min with 8 GPUs)")
