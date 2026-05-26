#!/usr/bin/env python3
"""
Step 4: Run pilot evaluation after training.

This is the critical feasibility check before scaling up.

Usage:
    python scripts/run_pilot_eval.py --config configs/pilot.yaml --checkpoint checkpoints/pilot/checkpoint_best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CrossBioSAE, CrossBioSAEConfig
from src.evaluation import run_pilot_evaluation, compute_feature_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run pilot evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    logger.info(f"Training step: {checkpoint.get('global_step', 'unknown')}")

    paths = config["paths"]
    output_dir = paths["output_dir"]

    # Run pilot evaluation
    results = run_pilot_evaluation(
        model=model,
        dna_h5_path=paths["dna_activations"],
        protein_h5_path=paths["protein_activations"],
        output_dir=output_dir,
        device=device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PILOT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Genes tested:           {results['n_genes']}")
    print(f"  SAE features:           {results['n_features']}")
    print(f"  Shared features:        {results['n_shared_features']} ({results['shared_fraction']:.1%})")
    print(f"  DNA-specific features:  {results['n_dna_specific_features']}")
    print(f"  Protein-specific:       {results['n_protein_specific_features']}")
    print(f"  Dead features:          {results['n_dead_features']}")
    print(f"  Cross-modal sim:        {results['mean_crossmodal_sim']:.4f}")
    print(f"  Permutation baseline:   {results['permutation_baseline_mean']:.4f} +/- {results['permutation_baseline_std']:.4f}")
    print(f"  Z-score:                {results['z_score']:.2f}")
    print(f"  P-value:                {results['p_value']:.2e}")
    print(f"  FEASIBLE:               {'YES' if results['feasible'] else 'NO / MARGINAL'}")
    print("=" * 60)

    if results["feasible"]:
        print("\nPilot passed. Safe to proceed to full-scale training.")
    else:
        print("\nPilot did not clearly pass. Consider:")
        print("  - Increasing crossmodal_weight")
        print("  - Trying different layer for activation extraction")
        print("  - Using different sparsity type (JumpReLU vs TopK)")
        print("  - Checking if activation extraction worked correctly")


if __name__ == "__main__":
    main()
