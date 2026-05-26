#!/usr/bin/env python3
"""
Step 3: Train the CrossBioSAE model.

Usage:
    python scripts/train_sae.py --config configs/pilot.yaml
    python scripts/train_sae.py --config configs/pilot.yaml --resume checkpoints/pilot/checkpoint_latest.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CrossBioSAE, CrossBioSAEConfig
from src.data import create_dataloaders
from src.trainer import CrossBioSAETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CrossBioSAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
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

    # Model config
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths = config["paths"]
    wandb_cfg = config.get("wandb", {})

    sae_config = CrossBioSAEConfig(
        dim_dna=model_cfg["dim_dna"],
        dim_protein=model_cfg["dim_protein"],
        dim_shared=model_cfg["dim_shared"],
        expansion_factor=model_cfg["expansion_factor"],
        sparsity_type=model_cfg.get("sparsity_type", "topk"),
        topk_k=model_cfg.get("topk_k", 64),
        l1_coeff=model_cfg.get("l1_coeff", 1e-3),
        normalize_inputs=model_cfg.get("normalize_inputs", True),
        tied_decoder=model_cfg.get("tied_decoder", False),
        bias=model_cfg.get("bias", True),
        recon_weight=train_cfg.get("recon_weight", 1.0),
        sparsity_weight=train_cfg.get("sparsity_weight", 1.0),
        crossmodal_weight=train_cfg.get("crossmodal_weight", 0.1),
        crossmodal_type=model_cfg.get("crossmodal_type", "cosine"),
    )

    logger.info(f"Model config: {sae_config}")
    logger.info(f"Number of features: {sae_config.n_features}")

    # Data
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        dna_h5_path=paths["dna_activations"],
        protein_h5_path=paths["protein_activations"],
        batch_size=train_cfg["batch_size"],
        train_ratio=0.9,
        num_workers=train_cfg.get("num_workers", 4),
        seed=train_cfg.get("seed", 42),
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Trainer
    trainer = CrossBioSAETrainer(
        config=sae_config,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        lr_schedule=train_cfg.get("lr_schedule", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        max_steps=train_cfg.get("max_steps"),
        max_epochs=train_cfg.get("max_epochs", 100),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        log_interval=train_cfg.get("log_interval", 100),
        eval_interval=train_cfg.get("eval_interval", 1000),
        save_interval=train_cfg.get("save_interval", 5000),
        save_dir=paths["checkpoint_dir"],
        device=device,
        use_wandb=wandb_cfg.get("enabled", False) and not args.no_wandb,
        wandb_project=wandb_cfg.get("project", "crossbiosae"),
        wandb_run_name=wandb_cfg.get("run_name"),
        normalize_decoder_interval=train_cfg.get("normalize_decoder_interval", 100),
        crossmodal_warmup_steps=train_cfg.get("crossmodal_warmup_steps", 2000),
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
