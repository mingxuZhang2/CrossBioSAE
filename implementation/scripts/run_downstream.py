#!/usr/bin/env python3
"""
Step 5: Run downstream tasks after successful training.

Tasks:
  1. Cross-modal anomaly detection (simplest, run first)
  2. Function prediction
  3. Variant prediction (requires ClinVar data)

Usage:
    python scripts/run_downstream.py --config configs/pilot.yaml \
        --checkpoint checkpoints/pilot/checkpoint_best.pt \
        --task anomaly

    python scripts/run_downstream.py --config configs/full.yaml \
        --checkpoint checkpoints/full/checkpoint_best.pt \
        --task all
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CrossBioSAE, CrossBioSAEConfig
from src.evaluation import AnomalyDetector, FunctionPredictor, VariantPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(config: dict, checkpoint_path: str, device: str) -> CrossBioSAE:
    """Load trained CrossBioSAE model from checkpoint."""
    model_cfg = config["model"]
    train_cfg = config["training"]

    sae_config = CrossBioSAEConfig(
        dim_dna=model_cfg["dim_dna"],
        dim_protein=model_cfg["dim_protein"],
        dim_shared=model_cfg["dim_shared"],
        expansion_factor=model_cfg["expansion_factor"],
        sparsity_type=model_cfg.get("sparsity_type", "topk"),
        topk_k=model_cfg.get("topk_k", 64),
        normalize_inputs=model_cfg.get("normalize_inputs", True),
        tied_decoder=model_cfg.get("tied_decoder", False),
        bias=model_cfg.get("bias", True),
        crossmodal_weight=train_cfg.get("crossmodal_weight", 0.1),
        crossmodal_type=model_cfg.get("crossmodal_type", "cosine"),
    )

    model = CrossBioSAE(sae_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def run_anomaly_detection(model, config, device):
    """Task 3: Cross-modal anomaly detection."""
    paths = config["paths"]
    output_dir = Path(paths["output_dir"]) / "task3_anomaly"

    detector = AnomalyDetector(model=model, device=device)

    anomaly_df = detector.compute_anomaly_scores(
        dna_h5_path=paths["dna_activations"],
        protein_h5_path=paths["protein_activations"],
        output_path=str(output_dir),
    )

    # Print top anomalous genes
    print("\nTop 20 most anomalous genes (lowest cross-modal consistency):")
    print(anomaly_df.head(20)[["gene_name", "consistency_score", "jsd_score",
                                "n_shared_active", "n_dna_only", "n_protein_only"]].to_string())

    # Print most consistent genes for comparison
    print("\nTop 20 most consistent genes:")
    print(anomaly_df.tail(20)[["gene_name", "consistency_score", "jsd_score",
                                "n_shared_active", "n_dna_only", "n_protein_only"]].to_string())

    return anomaly_df


def run_function_prediction(model, config, device):
    """Task 2: Function prediction for uncharacterized genes."""
    paths = config["paths"]
    output_dir = Path(paths["output_dir"]) / "task2_function"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = FunctionPredictor(model=model, device=device)

    # NOTE: GO annotation data needs to be prepared separately
    # For now, just compute features without GO labeling
    logger.info("Function prediction requires GO annotation data. "
                "See README for data preparation instructions.")
    logger.info("Computing features for all genes...")

    predictions = predictor.predict_function(
        dna_h5_path=paths["dna_activations"],
        protein_h5_path=paths["protein_activations"],
        output_path=str(output_dir / "predictions.csv"),
    )

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run downstream tasks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--task", type=str, default="anomaly",
        choices=["anomaly", "function", "variant", "all"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(config, args.checkpoint, device)

    if args.task in ("anomaly", "all"):
        logger.info("Running Task 3: Cross-modal anomaly detection")
        run_anomaly_detection(model, config, device)

    if args.task in ("function", "all"):
        logger.info("Running Task 2: Function prediction")
        run_function_prediction(model, config, device)

    if args.task in ("variant", "all"):
        logger.info("Task 1: Variant prediction requires ClinVar data preparation")
        logger.info("See README for instructions on preparing ClinVar holdout data")

    logger.info("Downstream tasks complete")


if __name__ == "__main__":
    main()
