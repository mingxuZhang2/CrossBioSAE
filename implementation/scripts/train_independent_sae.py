#!/usr/bin/env python3
"""
Train an independent (single-modality) StandardSAE.

This trains one SAE on either protein or DNA activations, with NO cross-modal
training signal. The resulting features are used for post-hoc alignment
evaluation, addressing the reviewer concern about circularity in joint training.

Usage:
    # Train protein SAE
    python scripts/train_independent_sae.py --config configs/full.yaml --modality protein --device cuda

    # Train DNA SAE
    python scripts/train_independent_sae.py --config configs/full.yaml --modality dna --device cuda

    # With independent-specific config overrides
    python scripts/train_independent_sae.py --config configs/independent.yaml --modality protein --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import StandardSAEConfig
from src.data import create_single_modality_dataloaders
from src.trainer import StandardSAETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train independent (single-modality) StandardSAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--modality", type=str, required=True, choices=["protein", "dna"],
                        help="Which modality to train on")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Training independent SAE for modality: {args.modality}")

    # Config sections
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths = config["paths"]
    wandb_cfg = config.get("wandb", {})

    # Independent SAE config may have its own section, or fall back to model config
    independent_cfg = config.get("independent_sae", {})

    # Determine input dimension based on modality
    if args.modality == "protein":
        dim_input = model_cfg["dim_protein"]
        h5_path = paths["protein_activations"]
    elif args.modality == "dna":
        dim_input = model_cfg["dim_dna"]
        h5_path = paths["dna_activations"]
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    # Build StandardSAEConfig
    sae_config = StandardSAEConfig(
        dim_input=dim_input,
        expansion_factor=independent_cfg.get("expansion_factor", model_cfg["expansion_factor"]),
        sparsity_type=independent_cfg.get("sparsity_type", model_cfg.get("sparsity_type", "topk")),
        topk_k=independent_cfg.get("topk_k", model_cfg.get("topk_k", 64)),
        l1_coeff=independent_cfg.get("l1_coeff", model_cfg.get("l1_coeff", 1e-3)),
        normalize_inputs=independent_cfg.get("normalize_inputs", model_cfg.get("normalize_inputs", True)),
        tied_decoder=independent_cfg.get("tied_decoder", model_cfg.get("tied_decoder", False)),
        bias=independent_cfg.get("bias", model_cfg.get("bias", True)),
        recon_weight=train_cfg.get("recon_weight", 1.0),
        sparsity_weight=train_cfg.get("sparsity_weight", 1.0),
    )

    logger.info(f"StandardSAE config: {sae_config}")
    logger.info(f"Number of features: {sae_config.n_features}")
    logger.info(f"Input dim: {sae_config.dim_input}, Expansion: {sae_config.expansion_factor}")

    # Data
    logger.info(f"Loading {args.modality} activations from {h5_path}...")
    train_loader, val_loader = create_single_modality_dataloaders(
        h5_path=h5_path,
        modality=args.modality,
        batch_size=train_cfg["batch_size"],
        train_ratio=0.9,
        num_workers=train_cfg.get("num_workers", 4),
        seed=train_cfg.get("seed", 42),
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Checkpoint dir: per-modality subdirectory
    save_dir = Path(paths["checkpoint_dir"]) / f"{args.modality}_sae"

    # Trainer
    trainer = StandardSAETrainer(
        config=sae_config,
        modality=args.modality,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=independent_cfg.get("learning_rate", train_cfg["learning_rate"]),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        lr_schedule=train_cfg.get("lr_schedule", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        max_steps=independent_cfg.get("max_steps", train_cfg.get("max_steps")),
        max_epochs=independent_cfg.get("max_epochs", train_cfg.get("max_epochs", 100)),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        log_interval=train_cfg.get("log_interval", 100),
        eval_interval=train_cfg.get("eval_interval", 1000),
        save_interval=train_cfg.get("save_interval", 5000),
        save_dir=str(save_dir),
        device=device,
        use_wandb=wandb_cfg.get("enabled", False) and not args.no_wandb,
        wandb_project=wandb_cfg.get("project", "crossbiosae"),
        wandb_run_name=wandb_cfg.get("run_name", f"independent_{args.modality}"),
        normalize_decoder_interval=train_cfg.get("normalize_decoder_interval", 100),
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    # Train
    logger.info(f"Starting independent SAE training for {args.modality}...")
    trainer.train()
    logger.info(f"Independent SAE training for {args.modality} complete")


if __name__ == "__main__":
    main()
