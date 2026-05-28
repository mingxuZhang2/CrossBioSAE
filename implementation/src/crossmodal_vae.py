"""
Shared-Private VAE for cross-modal gene representation learning.

Learns to decompose protein LM and DNA LM embeddings into:
- z_shared: what both modalities agree on (cross-modal biological concepts)
- z_prot_private: protein-specific information (structure, domains)
- z_dna_private: DNA-specific information (codon usage, regulatory signals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SPVAEConfig:
    prot_dim: int = 1280
    dna_dim: int = 1024
    hidden_dim: int = 512
    shared_dim: int = 128
    private_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    beta_shared: float = 1.0
    beta_private: float = 1.0
    alpha_align: float = 10.0
    alpha_disentangle: float = 1.0


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int,
                 private_dim: int, n_layers: int, dropout: float):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)

        self.shared_mu = nn.Linear(hidden_dim, shared_dim)
        self.shared_logvar = nn.Linear(hidden_dim, shared_dim)
        self.private_mu = nn.Linear(hidden_dim, private_dim)
        self.private_logvar = nn.Linear(hidden_dim, private_dim)

    def forward(self, x):
        h = self.backbone(x)
        return (
            self.shared_mu(h), self.shared_logvar(h),
            self.private_mu(h), self.private_logvar(h),
        )


class Decoder(nn.Module):
    def __init__(self, shared_dim: int, private_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int, dropout: float):
        super().__init__()

        input_dim = shared_dim + private_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_shared, z_private):
        return self.net(torch.cat([z_shared, z_private], dim=-1))


class SharedPrivateVAE(nn.Module):
    def __init__(self, config: SPVAEConfig):
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

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, prot, dna):
        """Encode both modalities. Returns all latent parameters."""
        p_shared_mu, p_shared_logvar, p_priv_mu, p_priv_logvar = self.prot_encoder(prot)
        d_shared_mu, d_shared_logvar, d_priv_mu, d_priv_logvar = self.dna_encoder(dna)
        return {
            "p_shared_mu": p_shared_mu, "p_shared_logvar": p_shared_logvar,
            "p_priv_mu": p_priv_mu, "p_priv_logvar": p_priv_logvar,
            "d_shared_mu": d_shared_mu, "d_shared_logvar": d_shared_logvar,
            "d_priv_mu": d_priv_mu, "d_priv_logvar": d_priv_logvar,
        }

    def decode(self, z_shared, z_prot_priv, z_dna_priv):
        """Decode from latent variables."""
        prot_recon = self.prot_decoder(z_shared, z_prot_priv)
        dna_recon = self.dna_decoder(z_shared, z_dna_priv)
        return prot_recon, dna_recon

    def forward(self, prot, dna):
        params = self.encode(prot, dna)

        # Sample latent variables
        z_p_shared = self.reparameterize(params["p_shared_mu"], params["p_shared_logvar"])
        z_d_shared = self.reparameterize(params["d_shared_mu"], params["d_shared_logvar"])
        z_p_priv = self.reparameterize(params["p_priv_mu"], params["p_priv_logvar"])
        z_d_priv = self.reparameterize(params["d_priv_mu"], params["d_priv_logvar"])

        # Use averaged shared for decoding (product-of-experts approximation)
        z_shared = (z_p_shared + z_d_shared) / 2.0

        prot_recon, dna_recon = self.decode(z_shared, z_p_priv, z_d_priv)

        return {
            "prot_recon": prot_recon, "dna_recon": dna_recon,
            "z_shared": z_shared,
            "z_p_shared": z_p_shared, "z_d_shared": z_d_shared,
            "z_p_priv": z_p_priv, "z_d_priv": z_d_priv,
            "params": params,
        }

    def get_representations(self, prot, dna):
        """Get all representations for downstream use (no grad)."""
        with torch.no_grad():
            out = self.forward(prot, dna)
        return {
            "shared": out["z_shared"].cpu().numpy(),
            "prot_private": out["z_p_priv"].cpu().numpy(),
            "dna_private": out["z_d_priv"].cpu().numpy(),
            "prot_shared": out["z_p_shared"].cpu().numpy(),
            "dna_shared": out["z_d_shared"].cpu().numpy(),
        }


class SPVAELoss(nn.Module):
    def __init__(self, config: SPVAEConfig):
        super().__init__()
        self.config = config

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(self, prot, dna, model_output):
        cfg = self.config
        params = model_output["params"]

        # 1. Reconstruction loss
        recon_prot = F.mse_loss(model_output["prot_recon"], prot)
        recon_dna = F.mse_loss(model_output["dna_recon"], dna)
        recon_loss = recon_prot + recon_dna

        # 2. KL divergence for shared latents
        kl_p_shared = self.kl_divergence(params["p_shared_mu"], params["p_shared_logvar"])
        kl_d_shared = self.kl_divergence(params["d_shared_mu"], params["d_shared_logvar"])
        kl_shared = cfg.beta_shared * (kl_p_shared + kl_d_shared)

        # 3. KL divergence for private latents
        kl_p_priv = self.kl_divergence(params["p_priv_mu"], params["p_priv_logvar"])
        kl_d_priv = self.kl_divergence(params["d_priv_mu"], params["d_priv_logvar"])
        kl_private = cfg.beta_private * (kl_p_priv + kl_d_priv)

        # 4. Alignment loss: shared representations should match across modalities
        align_loss = cfg.alpha_align * F.mse_loss(
            params["p_shared_mu"], params["d_shared_mu"]
        )

        # 5. Disentanglement: shared and private should be independent
        # Use cosine similarity penalty
        cos_p = F.cosine_similarity(
            model_output["z_p_shared"], model_output["z_p_priv"], dim=-1
        ).abs().mean()
        cos_d = F.cosine_similarity(
            model_output["z_d_shared"], model_output["z_d_priv"], dim=-1
        ).abs().mean()
        disentangle_loss = cfg.alpha_disentangle * (cos_p + cos_d)

        total = recon_loss + kl_shared + kl_private + align_loss + disentangle_loss

        return {
            "loss": total,
            "recon_prot": recon_prot.item(),
            "recon_dna": recon_dna.item(),
            "kl_shared": kl_shared.item(),
            "kl_private": kl_private.item(),
            "align": align_loss.item(),
            "disentangle": disentangle_loss.item(),
        }


class SPVAETrainer:
    def __init__(self, model: SharedPrivateVAE, config: SPVAEConfig,
                 lr: float = 1e-3, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = SPVAELoss(config)
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
            n_epochs=200, batch_size=512, patience=20, save_path=None):
        best_val_loss = float("inf")
        wait = 0
        history = []

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-5
        )

        for epoch in range(n_epochs):
            train_losses = self.train_epoch(prot_train, dna_train, batch_size)
            scheduler.step()

            log = f"Epoch {epoch+1:3d} | train loss={train_losses['loss']:.4f}"
            log += f" recon_p={train_losses['recon_prot']:.4f} recon_d={train_losses['recon_dna']:.4f}"
            log += f" align={train_losses['align']:.4f} disentangle={train_losses['disentangle']:.4f}"

            if prot_val is not None:
                val_losses = self.evaluate(prot_val, dna_val, batch_size)
                log += f" | val loss={val_losses['loss']:.4f}"

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

                history.append({**{f"train_{k}": v for k, v in train_losses.items()},
                                **{f"val_{k}": v for k, v in val_losses.items()}})
            else:
                history.append({f"train_{k}": v for k, v in train_losses.items()})

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(log)

        if save_path and prot_val is not None:
            self.model.load_state_dict(torch.load(save_path, weights_only=True))

        return history

    @torch.no_grad()
    def extract_all(self, prot_data, dna_data, batch_size=512):
        """Extract all representations for all genes."""
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
