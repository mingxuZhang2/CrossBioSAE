"""
Trainer for CrossBioSAE.

Supports:
- Mixed training: both modalities + cross-modal consistency loss
- Single-modality warmup: pre-train adapters on each modality separately
- Wandb logging
- Periodic decoder normalization
- Checkpoint saving/loading
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .model import CrossBioSAE, CrossBioSAEConfig, CrossBioSAELoss, count_parameters

logger = logging.getLogger(__name__)


class CrossBioSAETrainer:
    """Training loop for CrossBioSAE."""

    def __init__(
        self,
        config: CrossBioSAEConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_schedule: str = "cosine",
        warmup_steps: int = 1000,
        max_steps: Optional[int] = None,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: str = "checkpoints",
        device: str = "cuda",
        use_wandb: bool = False,
        wandb_project: str = "crossbiosae",
        wandb_run_name: Optional[str] = None,
        normalize_decoder_interval: int = 100,
        crossmodal_warmup_steps: int = 2000,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.normalize_decoder_interval = normalize_decoder_interval
        self.crossmodal_warmup_steps = crossmodal_warmup_steps
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb and HAS_WANDB

        # Model
        self.model = CrossBioSAE(config).to(device)
        self.loss_fn = CrossBioSAELoss(config)

        # Log parameter counts
        param_counts = count_parameters(self.model)
        logger.info("Model parameter counts:")
        for name, count in param_counts.items():
            logger.info(f"  {name}: {count:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler (T_max excludes warmup steps)
        total_steps = self._estimate_total_steps()
        decay_steps = max(total_steps - warmup_steps, 1)
        if lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=decay_steps, eta_min=learning_rate * 0.01
            )
        elif lr_schedule == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=decay_steps
            )
        else:
            self.scheduler = None

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model": vars(config),
                    "training": {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "lr_schedule": lr_schedule,
                        "warmup_steps": warmup_steps,
                        "max_steps": max_steps,
                        "max_epochs": max_epochs,
                        "grad_clip": grad_clip,
                        "crossmodal_warmup_steps": crossmodal_warmup_steps,
                    },
                    "data": {
                        "train_size": len(train_loader.dataset),
                        "val_size": len(val_loader.dataset) if val_loader else 0,
                        "batch_size": train_loader.batch_size,
                    },
                },
            )

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps."""
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader)
        return steps_per_epoch * self.max_epochs

    def _get_crossmodal_weight(self) -> float:
        """
        Ramp up cross-modal consistency loss over warmup period.
        This prevents the cross-modal loss from dominating early training
        before the adapters have learned basic reconstruction.
        """
        if self.global_step < self.crossmodal_warmup_steps:
            return self.config.crossmodal_weight * (
                self.global_step / self.crossmodal_warmup_steps
            )
        return self.config.crossmodal_weight

    def _warmup_lr(self):
        """Linear warmup for learning rate."""
        if self.global_step < self.warmup_steps:
            lr_scale = self.global_step / max(1, self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * lr_scale

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step."""
        self.model.train()

        dna_acts = batch.get("dna_acts")
        protein_acts = batch.get("protein_acts")

        if dna_acts is not None:
            dna_acts = dna_acts.to(self.device)
        if protein_acts is not None:
            protein_acts = protein_acts.to(self.device)

        # Update running mean (for input normalization)
        if self.config.normalize_inputs:
            self.model.update_running_mean(dna_acts, protein_acts)

        # Forward pass
        output = self.model(dna_acts=dna_acts, protein_acts=protein_acts)

        # Compute losses with dynamic cross-modal weight
        original_cm_weight = self.config.crossmodal_weight
        self.config.crossmodal_weight = self._get_crossmodal_weight()

        losses = self.loss_fn(
            dna_acts=dna_acts,
            protein_acts=protein_acts,
            model_output=output,
            paired=(dna_acts is not None and protein_acts is not None),
        )

        self.config.crossmodal_weight = original_cm_weight

        # Backward pass
        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        if self.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        self.optimizer.step()

        # LR warmup
        self._warmup_lr()

        # LR schedule step
        if self.scheduler and self.global_step >= self.warmup_steps:
            self.scheduler.step()

        # Periodically normalize decoder weights
        if self.global_step % self.normalize_decoder_interval == 0:
            self.model.sae._normalize_decoder()

        # Convert losses to float for logging
        metrics = {k: v.item() for k, v in losses.items()}
        metrics["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["crossmodal_weight_effective"] = self._get_crossmodal_weight()

        # Sparsity metrics
        if "features_dna" in output:
            features_dna = output["features_dna"]
            metrics["l0_dna"] = (features_dna > 0).float().sum(dim=-1).mean().item()
            metrics["feature_density_dna"] = (features_dna > 0).float().mean().item()
        if "features_protein" in output:
            features_protein = output["features_protein"]
            metrics["l0_protein"] = (features_protein > 0).float().sum(dim=-1).mean().item()
            metrics["feature_density_protein"] = (features_protein > 0).float().mean().item()

        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_metrics = {}
        n_batches = 0

        for batch in self.val_loader:
            dna_acts = batch.get("dna_acts")
            protein_acts = batch.get("protein_acts")

            if dna_acts is not None:
                dna_acts = dna_acts.to(self.device)
            if protein_acts is not None:
                protein_acts = protein_acts.to(self.device)

            output = self.model(dna_acts=dna_acts, protein_acts=protein_acts)
            losses = self.loss_fn(
                dna_acts=dna_acts,
                protein_acts=protein_acts,
                model_output=output,
                paired=(dna_acts is not None and protein_acts is not None),
            )

            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            # Cross-modal alignment metric
            if "features_dna" in output and "features_protein" in output:
                cos_sim = torch.nn.functional.cosine_similarity(
                    output["features_dna"], output["features_protein"], dim=-1
                ).mean()
                if "crossmodal_cosine_sim" not in total_metrics:
                    total_metrics["crossmodal_cosine_sim"] = 0.0
                total_metrics["crossmodal_cosine_sim"] += cos_sim.item()

            n_batches += 1

        # Average
        metrics = {f"val/{k}": v / n_batches for k, v in total_metrics.items()}
        return metrics

    def save_checkpoint(self, tag: str = "latest"):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self):
        """Main training loop."""
        logger.info(
            f"Starting training: {self._estimate_total_steps()} estimated steps, "
            f"{len(self.train_loader)} batches/epoch"
        )

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    metrics_str = " | ".join(
                        f"{k}: {v:.4f}" for k, v in sorted(metrics.items())
                    )
                    logger.info(f"Step {self.global_step} | {metrics_str}")
                    if self.use_wandb:
                        wandb.log(
                            {f"train/{k}": v for k, v in metrics.items()},
                            step=self.global_step,
                        )

                # Evaluation
                if self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        val_str = " | ".join(
                            f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())
                        )
                        logger.info(f"Validation @ step {self.global_step} | {val_str}")
                        if self.use_wandb:
                            wandb.log(val_metrics, step=self.global_step)

                        # Save best model
                        val_total = val_metrics.get("val/total", float("inf"))
                        if val_total < self.best_val_loss:
                            self.best_val_loss = val_total
                            self.save_checkpoint("best")

                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint("latest")
                    self.save_checkpoint(f"step_{self.global_step}")

                # Max steps check
                if self.max_steps and self.global_step >= self.max_steps:
                    logger.info(f"Reached max steps ({self.max_steps})")
                    self.save_checkpoint("final")
                    return

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        self.save_checkpoint("final")
        logger.info("Training complete")

        if self.use_wandb:
            wandb.finish()
