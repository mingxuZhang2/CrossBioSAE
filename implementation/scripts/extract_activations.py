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


def generate_synthetic_activations(config: dict, modality: str):
    """Generate random activations for pipeline testing without real models."""
    import numpy as np

    data_config = config["data"]
    paths = config["paths"]
    mod_config = config["protein_model"] if modality == "protein" else config["dna_model"]

    dataset = GenePairDataset(data_dir=data_config["data_dir"], split=data_config["split"])
    pairs = dataset.load_pairs()
    gene_names = [p["gene_name"] for p in pairs]
    hidden_dim = mod_config["hidden_dim"]

    logger.info(f"Generating synthetic {modality} activations: {len(gene_names)} genes, dim={hidden_dim}")
    activations = np.random.randn(len(gene_names), hidden_dim).astype(np.float32)

    output_file = paths["protein_activations"] if modality == "protein" else paths["dna_activations"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    import h5py
    with h5py.File(output_file, "w") as f:
        f.create_dataset("activations", data=activations, compression="gzip")
        f.create_dataset("gene_names", data=np.array(gene_names, dtype=h5py.string_dtype()))
        f.attrs["model"] = "synthetic"
        f.attrs["layer"] = mod_config.get("layer", 16)
        f.attrs["pooling"] = "mean"
        f.attrs["n_sequences"] = len(gene_names)

    logger.info(f"Saved synthetic activations to {output_file}")


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
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate random activations instead of running real models (for pipeline testing)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    if args.synthetic:
        logger.info("Generating SYNTHETIC activations (no real models loaded)")
        if args.modality in ("protein", "both"):
            generate_synthetic_activations(config, "protein")
        if args.modality in ("dna", "both"):
            generate_synthetic_activations(config, "dna")
    else:
        if args.modality in ("protein", "both"):
            extract_protein_activations(config, device)
        if args.modality in ("dna", "both"):
            extract_dna_activations(config, device)

    logger.info("Activation extraction complete")


if __name__ == "__main__":
    main()
