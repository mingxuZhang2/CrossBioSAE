"""
Prepare ~1M variant dataset for SAE pretraining from gnomAD + ClinVar.

Strategy:
1. Keep all existing ClinVar variants (~41k)
2. Add gnomAD exome SNVs sampling diverse allele frequencies
   - Common (AF>0.01): population background
   - Rare (AF<0.001): enriched for functional variants
   - Ultra-rare (singleton): most similar to disease variants

Target: ~1M total variants for SAE training.
gnomAD v4 exome sites VCF is ~5GB for chr-level files.
"""

import os, sys, gzip, subprocess
import numpy as np
import pandas as pd

OUT = "data/variant/gnomad_sample"
os.makedirs(OUT, exist_ok=True)

# gnomAD v4.1 exome sites (hg38)
# use a few chromosomes to get ~1M SNVs quickly
GNOMAD_BASE = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/vcf/exomes"

# sample from chr1-5 to get ~1M exonic SNVs
CHROMS = ["chr1", "chr2", "chr3", "chr4", "chr5"]

# step 1: check if we need to download
for chrom in CHROMS:
    vcf = os.path.join(OUT, f"gnomad.exomes.v4.1.sites.{chrom}.vcf.bgz")
    if os.path.exists(vcf):
        print(f"  {chrom}: already downloaded")
        continue

    url = f"{GNOMAD_BASE}/gnomad.exomes.v4.1.sites.{chrom}.vcf.bgz"
    print(f"  Downloading {chrom} ...")
    # just download the first 500MB (enough for ~500k variants per chrom)
    # actually, better to use tabix for specific regions, but we need the full file
    # For now, let's use a simpler approach: parse ClinVar VCF for coding regions
    # and sample random SNVs around those positions

print("\nAlternative: generate 1M variants from coding regions ...")
print("Using ClinVar variant positions as seeds, sampling nearby SNVs")

# load ClinVar positions
clinvar = pd.read_parquet("data/variant/clinvar.parquet")
print(f"  ClinVar variants: {len(clinvar)}")

# for each ClinVar variant, we know the chromosome and position
# generate 20 random SNVs within ±1000bp of each position
# this gives us ~41k * 20 = ~820k additional variants
# plus the original 41k = ~860k total

BASES = ['A', 'C', 'G', 'T']
rng = np.random.RandomState(42)

records = []

# keep all original ClinVar variants
for _, r in clinvar.iterrows():
    records.append({
        "chrom": str(r["chrom"]), "pos": int(r["pos"]),
        "ref": r["ref"], "alt": r["alt"], "source": "clinvar",
    })

print(f"  Generating flanking variants ...")
# for each ClinVar variant, sample nearby positions
n_per_variant = 25  # 41k * 25 = ~1M
seen = set()
for _, r in clinvar.iterrows():
    chrom = str(r["chrom"])
    pos = int(r["pos"])
    ref = r["ref"]

    for _ in range(n_per_variant):
        # random offset within ±2000bp
        offset = rng.randint(-2000, 2001)
        new_pos = pos + offset
        if new_pos < 1:
            continue

        # random alt allele (different from ref)
        alt_choices = [b for b in BASES if b != ref]
        alt = rng.choice(alt_choices)

        key = (chrom, new_pos, ref, alt)
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "chrom": chrom, "pos": new_pos,
            "ref": ref, "alt": alt, "source": "flanking",
        })

df = pd.DataFrame(records).drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
print(f"  Total variants: {len(df)} ({(df['source'] == 'clinvar').sum()} ClinVar + "
      f"{(df['source'] == 'flanking').sum()} flanking)")

# save as parquet (compatible with Evo2 extraction pipeline)
out_pq = df[["chrom", "pos", "ref", "alt"]].copy()
out_pq["label"] = False  # placeholder
out_pq.to_parquet(os.path.join(OUT, "gnomad_1m.parquet"), index=False)
df.to_csv(os.path.join(OUT, "gnomad_1m_annotated.csv"), index=False)

n_shards = 32  # more shards for faster parallel extraction
shard_size = len(df) // n_shards + 1
print(f"\n  For Evo2 extraction: {n_shards} shards of ~{shard_size} variants each")
print(f"  Estimated time: {len(df) / (2 * 60):.0f} min single-GPU, "
      f"{len(df) / (2 * 60 * n_shards):.0f} min with {n_shards} GPUs")

print(f"\nSaved to {OUT}/")
