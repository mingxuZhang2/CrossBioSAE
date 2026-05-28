"""
Shared-Private Autoencoder for cross-modal gene representation learning.

Deterministic AE (no KL divergence) that decomposes protein LM and DNA LM
embeddings into:
- z_shared: what both modalities agree on (cross-modal biological concepts)
- z_prot_private: protein-specific information
- z_dna_private: DNA-specific information

Losses:
1. Reconstruction (must reconstruct input from shared + private)
2. Alignment (shared representations match across modalities)
3. Disentanglement (shared independent of private)
4. Cross-reconstruction (reconstruct protein from shared + DNA private, and vice versa)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SPAEConfig:
    prot_dim: int = 1280
    dna_dim: int = 1024
    hidden_dim: int = 1024
    shared_dim: int = 128
    private_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    alpha_align: float = 10.0
    alpha_disentangle: float = 1.0
    alpha_cross_recon: float = 0.5


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int,
                 private_dim: int, n_layers: int, dropout: float):
        super().__init__()

        dims = [input_dim] + [hidden_dim] * n_layers
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*layers)
        self.shared_head = nn.Linear(hidden_dim, shared_dim)
        self.private_head = nn.Linear(hidden_dim, private_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.shared_head(h), self.private_head(h)


class Decoder(nn.Module):
    def __init__(self, shared_dim: int, private_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int, dropout: float):
        super().__init__()

        input_dim = shared_dim + private_dim
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z_shared, z_private):
        return self.net(torch.cat([z_shared, z_private], dim=-1))


class SharedPrivateAE(nn.Module):
    def __init__(self, config: SPAEConfig):
        super().__init__()
        self.config = config

        self.prot_encoder = Encoder(
            config.prot_dim, config.hidden_dim, config.shared_dim,
            config.private_dim, config.n_layers, config.dropout,
        )
        self.dna_encoder = Encoder(
            config.dna_dim, config.hidden_dim, config.shared_dim,
            config.private_dim, config.n_layers, config.dropout,
        )
        self.prot_decoder = Decoder(
            config.shared_dim, config.private_dim, config.hidden_dim,
            config.prot_dim, config.n_layers, config.dropout,
        )
        self.dna_decoder = Decoder(
            config.shared_dim, config.private_dim, config.hidden_dim,
            config.dna_dim, config.n_layers, config.dropout,
        )

    def forward(self, prot, dna):
        z_p_shared, z_p_priv = self.prot_encoder(prot)
        z_d_shared, z_d_priv = self.dna_encoder(dna)

        z_shared = (z_p_shared + z_d_shared) / 2.0

        prot_recon = self.prot_decoder(z_shared, z_p_priv)
        dna_recon = self.dna_decoder(z_shared, z_d_priv)

        # Cross-reconstruction: use shared from one modality + private from the other
        prot_cross = self.prot_decoder(z_d_shared, z_p_priv)
        dna_cross = self.dna_decoder(z_p_shared, z_d_priv)

        return {
            "prot_recon": prot_recon, "dna_recon": dna_recon,
            "prot_cross": prot_cross, "dna_cross": dna_cross,
            "z_shared": z_shared,
            "z_p_shared": z_p_shared, "z_d_shared": z_d_shared,
            "z_p_priv": z_p_priv, "z_d_priv": z_d_priv,
        }

    @torch.no_grad()
    def get_representations(self, prot, dna):
        self.eval()
        out = self.forward(prot, dna)
        return {
            "shared": out["z_shared"].cpu().numpy(),
            "prot_private": out["z_p_priv"].cpu().numpy(),
            "dna_private": out["z_d_priv"].cpu().numpy(),
        }


class SPAELoss(nn.Module):
    def __init__(self, config: SPAEConfig):
        super().__init__()
        self.config = config

    def forward(self, prot, dna, out):
        cfg = self.config

        recon_prot = F.mse_loss(out["prot_recon"], prot)
        recon_dna = F.mse_loss(out["dna_recon"], dna)
        recon_loss = recon_prot + recon_dna

        cross_prot = F.mse_loss(out["prot_cross"], prot)
        cross_dna = F.mse_loss(out["dna_cross"], dna)
        cross_loss = cfg.alpha_cross_recon * (cross_prot + cross_dna)

        align_loss = cfg.alpha_align * F.mse_loss(out["z_p_shared"], out["z_d_shared"])

        def cross_cov_penalty(z1, z2):
            z1_c = z1 - z1.mean(dim=0)
            z2_c = z2 - z2.mean(dim=0)
            cov = (z1_c.T @ z2_c) / (z1.shape[0] - 1)
            return (cov ** 2).mean()

        disentangle = cfg.alpha_disentangle * (
            cross_cov_penalty(out["z_p_shared"], out["z_p_priv"]) +
            cross_cov_penalty(out["z_d_shared"], out["z_d_priv"])
        )

        total = recon_loss + cross_loss + align_loss + disentangle

        return {
            "loss": total,
            "recon_prot": recon_prot.item(),
            "recon_dna": recon_dna.item(),
            "cross_prot": cross_prot.item(),
            "cross_dna": cross_dna.item(),
            "align": align_loss.item(),
            "disentangle": disentangle.item(),
        }


class SPAETrainer:
    def __init__(self, model, config, lr=1e-3, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = SPAELoss(config)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self, prot_data, dna_data, batch_size=512):
        self.model.train()
        n = prot_data.shape[0]
        perm = torch.randperm(n)
        total_losses = {}
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            prot = prot_data[idx].to(self.device)
            dna = dna_data[idx].to(self.device)

            out = self.model(prot, dna)
            losses = self.loss_fn(prot, dna, out)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, prot_data, dna_data, batch_size=512):
        self.model.eval()
        n = prot_data.shape[0]
        total_losses = {}
        n_batches = 0

        for i in range(0, n, batch_size):
            prot = prot_data[i:i + batch_size].to(self.device)
            dna = dna_data[i:i + batch_size].to(self.device)

            out = self.model(prot, dna)
            losses = self.loss_fn(prot, dna, out)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def fit(self, prot_train, dna_train, prot_val=None, dna_val=None,
            n_epochs=300, batch_size=512, patience=30, save_path=None):
        best_val_loss = float("inf")
        wait = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-5
        )

        for epoch in range(n_epochs):
            train_losses = self.train_epoch(prot_train, dna_train, batch_size)
            scheduler.step()

            log = f"Epoch {epoch+1:3d} | loss={train_losses['loss']:.4f}"
            log += f" recon_p={train_losses['recon_prot']:.4f} recon_d={train_losses['recon_dna']:.4f}"
            log += f" cross_p={train_losses['cross_prot']:.4f} cross_d={train_losses['cross_dna']:.4f}"
            log += f" align={train_losses['align']:.4f}"

            if prot_val is not None:
                val_losses = self.evaluate(prot_val, dna_val, batch_size)
                log += f" | val={val_losses['loss']:.4f} recon={val_losses['recon_prot']+val_losses['recon_dna']:.4f}"

                if val_losses["loss"] < best_val_loss:
                    best_val_loss = val_losses["loss"]
                    wait = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(log, flush=True)

        if save_path and prot_val is not None:
            self.model.load_state_dict(torch.load(save_path, weights_only=True))

    @torch.no_grad()
    def extract_all(self, prot_data, dna_data, batch_size=512):
        self.model.eval()
        all_shared, all_p_priv, all_d_priv = [], [], []

        for i in range(0, prot_data.shape[0], batch_size):
            prot = prot_data[i:i + batch_size].to(self.device)
            dna = dna_data[i:i + batch_size].to(self.device)
            reps = self.model.get_representations(prot, dna)
            all_shared.append(reps["shared"])
            all_p_priv.append(reps["prot_private"])
            all_d_priv.append(reps["dna_private"])

        import numpy as np
        return {
            "shared": np.concatenate(all_shared),
            "prot_private": np.concatenate(all_p_priv),
            "dna_private": np.concatenate(all_d_priv),
        }
