"""
Sample 500k missense SNVs from ClinVar for cross-modal SAE pretraining.
All missense variants have both genomic coordinates (for Evo2) and
protein change annotations (for ESM-2) → 100% dual-modality coverage.
"""

import gzip, re, os
import numpy as np
import pandas as pd

OUT = "data/variant/sae_pretrain"
os.makedirs(OUT, exist_ok=True)

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

print("Extracting all missense SNVs from ClinVar ...")
records = []
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        if parts[1] != "single nucleotide variant":
            continue

        chrom = parts[18]
        pos = parts[31]
        ref = parts[32]
        alt = parts[33]
        if len(ref) != 1 or len(alt) != 1:
            continue

        name = parts[2]
        gene = parts[4]
        sig = parts[6]

        m = re.search(r'\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)', name)
        if not m:
            continue
        from_aa3 = m.group(1)
        prot_pos = int(m.group(2))
        to_aa3 = m.group(3)
        from_aa = AA_3TO1.get(from_aa3)
        to_aa = AA_3TO1.get(to_aa3)
        if not from_aa or not to_aa or from_aa == to_aa:
            continue

        records.append({
            "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
            "gene": gene, "from_aa": from_aa, "prot_pos": prot_pos,
            "to_aa": to_aa, "clin_sig": sig,
        })

df = pd.DataFrame(records)
df = df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
print(f"  Total unique missense SNVs: {len(df)}")
print(f"  Genes: {df['gene'].nunique()}")

# sample 500k, stratified to keep all P/LP/B/LB and fill rest with VUS
is_labeled = df["clin_sig"].str.contains(
    "Pathogenic|Benign", case=False, na=False
) & ~df["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)

labeled = df[is_labeled]
unlabeled = df[~is_labeled]
print(f"  Labeled (P/LP/B/LB): {len(labeled)}")
print(f"  Unlabeled (VUS etc): {len(unlabeled)}")

TARGET = 500000
n_sample_unlabeled = min(TARGET - len(labeled), len(unlabeled))

rng = np.random.RandomState(42)
sampled_unlabeled = unlabeled.sample(n=n_sample_unlabeled, random_state=rng)
final = pd.concat([labeled, sampled_unlabeled], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"\n  Final dataset: {len(final)}")
print(f"  Labeled: {is_labeled.sum()} ({100*len(labeled)/len(final):.1f}%)")

# save parquet for Evo2 extraction (genomic coords)
evo2_pq = final[["chrom", "pos", "ref", "alt"]].copy()
evo2_pq["label"] = False  # placeholder
evo2_pq.to_parquet(os.path.join(OUT, "missense_500k.parquet"), index=False)

# save full annotations (for ESM-2 mapping)
final.to_csv(os.path.join(OUT, "missense_500k_annotated.csv"), index=False)

# stats
print(f"\n  By clinical significance:")
for sig, n in final["clin_sig"].value_counts().head(10).items():
    print(f"    {sig[:50]:50s} {n:6d}")

print(f"\n  Top genes:")
for gene, n in final["gene"].value_counts().head(10).items():
    print(f"    {gene:10s} {n:5d}")

print(f"\nSaved to {OUT}/")
