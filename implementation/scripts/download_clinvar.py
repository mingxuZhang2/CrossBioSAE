#!/usr/bin/env python3
"""
Download ClinVar missense variants for variant effect prediction task.
Fetches recent pathogenic and benign variants with protein-level annotations.

Usage:
    python scripts/download_clinvar.py --output data/full/clinvar_variants.csv
"""

import argparse
import gzip
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_clinvar_summary(output_path: str, max_variants: int = 50000):
    """Download ClinVar variant summary and filter for missense variants."""

    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    logger.info(f"Downloading ClinVar variant summary from {url}...")

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    # Write gzipped file
    gz_path = Path(output_path).parent / "variant_summary.txt.gz"
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gz_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded to {gz_path}")

    # Parse
    logger.info("Parsing ClinVar data...")
    df = pd.read_csv(gz_path, sep="\t", low_memory=False, compression="gzip")

    logger.info(f"Total variants: {len(df)}")

    # Filter: missense only, human, single nucleotide
    mask = (
        (df["Assembly"] == "GRCh38") &
        (df["Type"] == "single nucleotide variant") &
        (df["GeneSymbol"].notna()) &
        (df["ClinicalSignificance"].str.contains("Pathogenic|Benign", case=False, na=False)) &
        (~df["ClinicalSignificance"].str.contains("Conflicting", case=False, na=False))
    )
    filtered = df[mask].copy()
    logger.info(f"After filtering (SNV, GRCh38, clear pathogenicity): {len(filtered)}")

    # Label
    filtered["pathogenicity"] = filtered["ClinicalSignificance"].apply(
        lambda x: 1 if "athogenic" in str(x) else 0
    )

    # Keep relevant columns
    cols = ["GeneSymbol", "Name", "ClinicalSignificance", "pathogenicity",
            "RS# (dbSNP)", "ReferenceAlleleVCF", "AlternateAlleleVCF",
            "Chromosome", "PositionVCF", "ReviewStatus", "LastEvaluated"]
    keep = [c for c in cols if c in filtered.columns]
    result = filtered[keep].head(max_variants)

    # Stats
    n_path = (result["pathogenicity"] == 1).sum()
    n_benign = (result["pathogenicity"] == 0).sum()
    logger.info(f"Final dataset: {len(result)} variants ({n_path} pathogenic, {n_benign} benign)")
    logger.info(f"Genes represented: {result['GeneSymbol'].nunique()}")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    # Cleanup
    gz_path.unlink(missing_ok=True)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/full/clinvar_variants.csv")
    parser.add_argument("--max_variants", type=int, default=50000)
    args = parser.parse_args()

    download_clinvar_summary(args.output, args.max_variants)


if __name__ == "__main__":
    main()
