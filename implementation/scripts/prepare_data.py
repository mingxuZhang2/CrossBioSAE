#!/usr/bin/env python3
"""
Step 1: Download gene pairs and prepare data for CrossBioSAE.

Usage:
    python scripts/prepare_data.py --config configs/pilot.yaml
    python scripts/prepare_data.py --config configs/full.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import GenePairDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare gene pair data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (for testing pipeline without API access)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    data_dir = data_config["data_dir"]

    logger.info(f"Preparing data for split: {data_config['split']}")
    logger.info(f"Data directory: {data_dir}")

    # Create dataset
    dataset = GenePairDataset(
        data_dir=data_dir,
        split=data_config["split"],
    )

    if args.synthetic:
        logger.info("Using synthetic data for pipeline testing")
        pairs = dataset._generate_synthetic_pairs(
            n_genes=data_config.get("n_genes", 1000),
            max_protein_length=data_config.get("max_protein_length", 1000),
            max_cds_length=data_config.get("max_cds_length", 3000),
        )
    else:
        pairs = dataset.download_gene_pairs(
            organism=data_config.get("organism", "Homo sapiens"),
            max_genes=data_config.get("n_genes"),
            max_protein_length=data_config.get("max_protein_length", 1000),
            max_cds_length=data_config.get("max_cds_length", 3000),
        )

    logger.info(f"Downloaded {len(pairs)} gene pairs")

    # Summary statistics
    protein_lengths = [p["protein_length"] for p in pairs]
    cds_lengths = [len(p["cds_seq"]) for p in pairs]

    logger.info(f"Protein length: min={min(protein_lengths)}, max={max(protein_lengths)}, "
                f"mean={sum(protein_lengths)/len(protein_lengths):.0f}")
    logger.info(f"CDS length: min={min(cds_lengths)}, max={max(cds_lengths)}, "
                f"mean={sum(cds_lengths)/len(cds_lengths):.0f}")
    logger.info(f"Data saved to: {dataset.pairs_file}")


if __name__ == "__main__":
    main()
