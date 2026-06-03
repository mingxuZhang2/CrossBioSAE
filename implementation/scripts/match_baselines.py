"""
Match external baseline scores (AlphaMissense, REVEL, CADD, EVE) to our
ClinVar 260k dual-modality variant set.

Produces: results/multiomics_validation/baseline_scores.csv
  columns: chrom, pos, ref, alt, am_score, revel_score, ...
  indexed by global dual_idx for direct join with our CLIP/fusion results.
"""

import gzip
import os
import sys

import numpy as np
import pandas as pd

OUT = "results/multiomics_validation"
os.makedirs(OUT, exist_ok=True)


def load_our_variants():
    """Load the 260k dual-modality variants with positions."""
    df = pd.read_csv("data/variant/sae_pretrain/missense_500k_annotated.csv", low_memory=False)
    dual_idx = np.load("data/variant/sae_pretrain/dual_idx.npy")
    sub = df.iloc[dual_idx].reset_index(drop=True)
    sub["key"] = sub["chrom"].astype(str) + ":" + sub["pos"].astype(str) + ":" + \
                 sub["ref"].astype(str) + ":" + sub["alt"].astype(str)
    return sub


def match_alphamissense(variants, am_path="data/baselines/AlphaMissense_hg38.tsv.gz"):
    """Stream-match AlphaMissense scores to our variants."""
    if not os.path.exists(am_path):
        print("  AlphaMissense file not found, skipping")
        return pd.Series(np.nan, index=variants.index, name="am_score")

    target = set(variants["key"].values)
    scores = {}
    print(f"  matching {len(target)} variants against AlphaMissense ...", flush=True)

    with gzip.open(am_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("CHROM"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            # format: CHROM POS REF ALT genome uniprot_id transcript_id protein_variant am_pathogenicity am_class
            chrom = parts[0].replace("chr", "")
            key = f"{chrom}:{parts[1]}:{parts[2]}:{parts[3]}"
            if key in target:
                try:
                    scores[key] = float(parts[8])  # am_pathogenicity is col 8
                except (ValueError, IndexError):
                    pass

    print(f"  matched: {len(scores)}/{len(target)} ({100*len(scores)/len(target):.1f}%)", flush=True)
    return variants["key"].map(scores).rename("am_score")


def match_revel(variants, revel_path="data/baselines/revel_all_chromosomes.csv"):
    """Match REVEL scores."""
    if not os.path.exists(revel_path) and not os.path.exists(revel_path + ".zip"):
        print("  REVEL file not found, skipping")
        return pd.Series(np.nan, index=variants.index, name="revel_score")

    # unzip if needed
    if not os.path.exists(revel_path) and os.path.exists(revel_path + ".zip"):
        import zipfile
        print("  unzipping REVEL ...", flush=True)
        with zipfile.ZipFile(revel_path + ".zip") as z:
            z.extractall(os.path.dirname(revel_path))

    if not os.path.exists(revel_path):
        import glob
        candidates = glob.glob(os.path.join(os.path.dirname(revel_path), "revel*"))
        candidates = [c for c in candidates if not c.endswith('.zip')]
        if candidates:
            revel_path = candidates[0]
        else:
            print("  REVEL file not found after unzip")
            return pd.Series(np.nan, index=variants.index, name="revel_score")

    target = set(variants["key"].values)
    scores = {}
    print(f"  matching against REVEL ({revel_path}) ...", flush=True)

    with open(revel_path) as f:
        header = f.readline().strip().split(",")
        chr_i = header.index("chr") if "chr" in header else 0
        pos_i = header.index("hg38_pos") if "hg38_pos" in header else (header.index("grch38_pos") if "grch38_pos" in header else 1)
        ref_i = header.index("ref") if "ref" in header else 2
        alt_i = header.index("alt") if "alt" in header else 3
        score_i = header.index("REVEL") if "REVEL" in header else (header.index("REVEL_score") if "REVEL_score" in header else -1)
        print(f"  header: {header[:8]}, score col={score_i}", flush=True)

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= score_i:
                continue
            chrom = parts[chr_i].replace("chr", "")
            key = f"{chrom}:{parts[pos_i]}:{parts[ref_i]}:{parts[alt_i]}"
            if key in target:
                try:
                    scores[key] = float(parts[score_i])
                except (ValueError, IndexError):
                    pass

    print(f"  matched: {len(scores)}/{len(target)} ({100*len(scores)/len(target):.1f}%)", flush=True)
    return variants["key"].map(scores).rename("revel_score")


def main():
    print("=== Loading our variants ===", flush=True)
    variants = load_our_variants()
    print(f"  {len(variants)} dual-modality variants", flush=True)

    print("\n=== AlphaMissense ===", flush=True)
    am = match_alphamissense(variants)

    print("\n=== REVEL ===", flush=True)
    revel = match_revel(variants)

    # combine
    result = variants[["chrom", "pos", "ref", "alt", "gene", "from_aa", "to_aa", "clin_sig"]].copy()
    result["am_score"] = am.values
    result["revel_score"] = revel.values

    out = os.path.join(OUT, "baseline_scores.csv")
    result.to_csv(out, index=False)

    print(f"\n=== Coverage ===", flush=True)
    for col in ["am_score", "revel_score"]:
        n = result[col].notna().sum()
        print(f"  {col}: {n}/{len(result)} ({100*n/len(result):.1f}%)", flush=True)

    # quick AUC comparison on labeled variants
    is_path = result["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~result["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = result["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~result["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = is_path | is_ben
    y = np.where(is_path, 1, 0)[labeled.values]

    from sklearn.metrics import roc_auc_score
    print(f"\n=== Quick AUC on {labeled.sum()} labeled variants ===", flush=True)
    for col in ["am_score", "revel_score"]:
        vals = result.loc[labeled, col]
        valid = vals.notna()
        if valid.sum() > 100:
            auc = roc_auc_score(y[valid.values], vals[valid].values)
            print(f"  {col}: AUC={auc:.4f} (n={valid.sum()})", flush=True)

    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
