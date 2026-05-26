#!/usr/bin/env python3
"""
Step 2: Extract activations from pre-trained protein and DNA models.

This is the compute-heavy step that requires GPU.
Run on HPC with GPU allocation.

Usage:
    # Extract protein activations (ESM-2)
    python scripts/extract_activations.py --config configs/pilot.yaml --modality protein

    # Extract DNA activations (Evo-2)
    python scripts/extract_activations.py --config configs/pilot.yaml --modality dna

    # Extract both
    python scripts/extract_activations.py --config configs/pilot.yaml --modality both
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import GenePairDataset, ActivationExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_protein_activations(config: dict, device: str):
    """Extract protein model activations."""
    prot_config = config["protein_model"]
    data_config = config["data"]
    paths = config["paths"]

    logger.info(f"Loading gene pairs from {data_config['data_dir']}")
    dataset = GenePairDataset(
        data_dir=data_config["data_dir"],
        split=data_config["split"],
    )
    sequences = dataset.get_protein_sequences()
    logger.info(f"Loaded {len(sequences)} protein sequences")

    extractor = ActivationExtractor(
        model_name=prot_config["name"],
        layer=prot_config.get("layer"),
        device=device,
        batch_size=prot_config.get("batch_size", 8),
        max_length=prot_config.get("max_length", 1024),
    )

    output_file = paths["protein_activations"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting protein activations to {output_file}")
    activations = extractor.extract(sequences, output_file)
    logger.info(f"Protein activations shape: {activations.shape}")


def extract_dna_activations(config: dict, device: str):
    """Extract DNA model activations."""
    dna_config = config["dna_model"]
    data_config = config["data"]
    paths = config["paths"]

    logger.info(f"Loading gene pairs from {data_config['data_dir']}")
    dataset = GenePairDataset(
        data_dir=data_config["data_dir"],
        split=data_config["split"],
    )
    sequences = dataset.get_dna_sequences()
    logger.info(f"Loaded {len(sequences)} DNA sequences")

    extractor = ActivationExtractor(
        model_name=dna_config["name"],
        layer=dna_config.get("layer"),
        device=device,
        batch_size=dna_config.get("batch_size", 4),
        max_length=dna_config.get("max_length", 3072),
    )

    output_file = paths["dna_activations"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting DNA activations to {output_file}")
    activations = extractor.extract(sequences, output_file)
    logger.info(f"DNA activations shape: {activations.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract model activations")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--modality", type=str, default="both",
        choices=["protein", "dna", "both"],
        help="Which modality to extract"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    if args.modality in ("protein", "both"):
        extract_protein_activations(config, device)

    if args.modality in ("dna", "both"):
        extract_dna_activations(config, device)

    logger.info("Activation extraction complete")


if __name__ == "__main__":
    main()
