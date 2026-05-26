"""
CrossBioSAE: Cross-modal Sparse Autoencoder for Protein-DNA Feature Alignment.

Architecture:
  DNA activation (dim_dna)     -> Adapter_DNA (linear) ─────────────┐
                                                                     ├─> Shared SAE encoder -> sparse features -> SAE decoder ─┬─> Adapter_DNA_inv -> reconstruct DNA
  Protein activation (dim_prot) -> Adapter_Prot (linear) ───────────┘                                                          └─> Adapter_Prot_inv -> reconstruct Protein

Three training losses:
  1. Reconstruction loss: MSE on reconstructed activations for both modalities
  2. Sparsity loss: L1 or TopK or JumpReLU
  3. Cross-modal consistency loss: cosine similarity of feature patterns for matched gene pairs
"""

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossBioSAEConfig:
    """Configuration for CrossBioSAE model."""
    # Input dimensions (from each modality)
    dim_dna: int = 4096            # Evo-2 hidden dim
    dim_protein: int = 1280        # ESM-2 650M hidden dim (2560 for ESM-C 600M)
    dim_shared: int = 2048         # Shared adapter output dim
    expansion_factor: int = 8      # SAE expansion factor (8x for pilot, 32x for full)

    # Sparsity
    sparsity_type: Literal["topk", "l1", "jumprelu"] = "topk"
    topk_k: int = 64              # Number of active features (for topk)
    l1_coeff: float = 1e-3        # L1 penalty coefficient (for l1)
    jumprelu_threshold: float = 0.01  # JumpReLU threshold (for jumprelu)
    jumprelu_bandwidth: float = 0.001  # JumpReLU STE bandwidth

    # Loss weights
    recon_weight: float = 1.0     # Reconstruction loss weight
    sparsity_weight: float = 1.0  # Sparsity loss weight
    crossmodal_weight: float = 0.1  # Cross-modal consistency loss weight
    crossmodal_type: Literal["cosine", "kl"] = "cosine"  # Cross-modal loss type

    # Architecture options
    normalize_inputs: bool = True  # Subtract mean from activations
    tied_decoder: bool = False     # Tie encoder/decoder weights (not recommended for cross-modal)
    bias: bool = True              # Use bias in encoder/decoder

    @property
    def n_features(self) -> int:
        return self.dim_shared * self.expansion_factor


class ModalityAdapter(nn.Module):
    """Linear adapter that projects modality-specific activations to shared space."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        # Initialize close to identity if dims match, else Xavier
        if dim_in == dim_out:
            nn.init.eye_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class InverseAdapter(nn.Module):
    """Linear adapter that projects from shared space back to modality-specific space."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TopKActivation(nn.Module):
    """TopK sparsity: keep only top-k activations, zero out the rest."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, topk_values)
        return result


class JumpReLU(nn.Module):
    """JumpReLU activation: x * (x > threshold), with STE for gradients."""

    def __init__(self, threshold: float = 0.01, bandwidth: float = 0.001):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward: hard threshold
        mask = (x > self.threshold).float()
        # STE: use sigmoid approximation for gradient
        if self.training:
            soft_mask = torch.sigmoid((x - self.threshold) / self.bandwidth)
            # Straight-through estimator
            mask = mask + (soft_mask - soft_mask.detach())
        return x * mask


class SharedSAE(nn.Module):
    """
    Shared Sparse Autoencoder operating in the adapter-projected shared space.

    Encoder: dim_shared -> n_features (with sparsity)
    Decoder: n_features -> dim_shared
    """

    def __init__(self, config: CrossBioSAEConfig):
        super().__init__()
        self.config = config
        n_features = config.n_features

        # Encoder
        self.encoder = nn.Linear(config.dim_shared, n_features, bias=config.bias)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        if config.bias:
            nn.init.zeros_(self.encoder.bias)

        # Decoder
        if config.tied_decoder:
            self.decoder_weight = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(n_features, config.dim_shared, bias=config.bias)
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            if config.bias:
                nn.init.zeros_(self.decoder.bias)

        # Normalize decoder columns (unit norm)
        self._normalize_decoder()

        # Sparsity activation
        if config.sparsity_type == "topk":
            self.activation = TopKActivation(config.topk_k)
        elif config.sparsity_type == "jumprelu":
            self.activation = JumpReLU(config.jumprelu_threshold, config.jumprelu_bandwidth)
        elif config.sparsity_type == "l1":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown sparsity type: {config.sparsity_type}")

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        if self.config.tied_decoder:
            return
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode shared-space activations to sparse features."""
        pre_activation = self.encoder(x)
        features = self.activation(pre_activation)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to shared space."""
        if self.config.tied_decoder:
            return F.linear(features, self.encoder.weight.t(),
                           self.encoder.bias if self.config.bias else None)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, dim_shared) shared-space activations
        Returns:
            reconstructed: (batch, dim_shared) reconstructed activations
            features: (batch, n_features) sparse feature activations
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features


class CrossBioSAE(nn.Module):
    """
    Full CrossBioSAE model with modality adapters and shared SAE.

    Forward pass:
        1. Project each modality to shared space via adapters
        2. Encode through shared SAE to get sparse features
        3. Decode and project back to original modality space
    """

    def __init__(self, config: CrossBioSAEConfig):
        super().__init__()
        self.config = config

        # Modality adapters: project to shared space
        self.adapter_dna = ModalityAdapter(config.dim_dna, config.dim_shared, bias=config.bias)
        self.adapter_protein = ModalityAdapter(config.dim_protein, config.dim_shared, bias=config.bias)

        # Shared SAE in the projected space
        self.sae = SharedSAE(config)

        # Inverse adapters: project back to original space
        self.inv_adapter_dna = InverseAdapter(config.dim_shared, config.dim_dna, bias=config.bias)
        self.inv_adapter_protein = InverseAdapter(config.dim_shared, config.dim_protein, bias=config.bias)

        # Running mean for input normalization (per modality)
        if config.normalize_inputs:
            self.register_buffer("dna_mean", torch.zeros(config.dim_dna))
            self.register_buffer("protein_mean", torch.zeros(config.dim_protein))
            self.register_buffer("n_dna_seen", torch.tensor(0, dtype=torch.long))
            self.register_buffer("n_protein_seen", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_running_mean(
        self,
        dna_acts: Optional[torch.Tensor] = None,
        protein_acts: Optional[torch.Tensor] = None,
    ):
        """Update running mean for input normalization (separate counters per modality)."""
        if not self.config.normalize_inputs:
            return
        if dna_acts is not None:
            n = self.n_dna_seen.item()
            batch_mean = dna_acts.mean(dim=0)
            batch_size = dna_acts.shape[0]
            self.dna_mean = (self.dna_mean * n + batch_mean * batch_size) / (n + batch_size)
            self.n_dna_seen += batch_size
        if protein_acts is not None:
            n = self.n_protein_seen.item()
            batch_mean = protein_acts.mean(dim=0)
            batch_size = protein_acts.shape[0]
            self.protein_mean = (self.protein_mean * n + batch_mean * batch_size) / (n + batch_size)
            self.n_protein_seen += batch_size

    def _normalize(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Subtract running mean from activations."""
        if not self.config.normalize_inputs:
            return x
        if modality == "dna":
            return x - self.dna_mean.to(x.device)
        elif modality == "protein":
            return x - self.protein_mean.to(x.device)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _denormalize(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Add running mean back to reconstructed activations."""
        if not self.config.normalize_inputs:
            return x
        if modality == "dna":
            return x + self.dna_mean.to(x.device)
        elif modality == "protein":
            return x + self.protein_mean.to(x.device)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def encode_dna(self, dna_acts: torch.Tensor) -> torch.Tensor:
        """Encode DNA activations to sparse features."""
        x = self._normalize(dna_acts, "dna")
        shared = self.adapter_dna(x)
        features = self.sae.encode(shared)
        return features

    def encode_protein(self, protein_acts: torch.Tensor) -> torch.Tensor:
        """Encode protein activations to sparse features."""
        x = self._normalize(protein_acts, "protein")
        shared = self.adapter_protein(x)
        features = self.sae.encode(shared)
        return features

    def forward_dna(
        self, dna_acts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for DNA modality.
        Returns: (reconstructed_dna, features, shared_repr)
        """
        x = self._normalize(dna_acts, "dna")
        shared = self.adapter_dna(x)
        recon_shared, features = self.sae(shared)
        recon_dna = self.inv_adapter_dna(recon_shared)
        recon_dna = self._denormalize(recon_dna, "dna")
        return recon_dna, features, shared

    def forward_protein(
        self, protein_acts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for protein modality.
        Returns: (reconstructed_protein, features, shared_repr)
        """
        x = self._normalize(protein_acts, "protein")
        shared = self.adapter_protein(x)
        recon_shared, features = self.sae(shared)
        recon_protein = self.inv_adapter_protein(recon_shared)
        recon_protein = self._denormalize(recon_protein, "protein")
        return recon_protein, features, shared

    def forward(
        self,
        dna_acts: Optional[torch.Tensor] = None,
        protein_acts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass for one or both modalities.

        Args:
            dna_acts: (batch, dim_dna) or None
            protein_acts: (batch, dim_protein) or None

        Returns:
            Dict with keys depending on which inputs were provided:
            - recon_dna, features_dna, shared_dna (if dna_acts given)
            - recon_protein, features_protein, shared_protein (if protein_acts given)
        """
        result = {}

        if dna_acts is not None:
            recon_dna, features_dna, shared_dna = self.forward_dna(dna_acts)
            result["recon_dna"] = recon_dna
            result["features_dna"] = features_dna
            result["shared_dna"] = shared_dna

        if protein_acts is not None:
            recon_protein, features_protein, shared_protein = self.forward_protein(protein_acts)
            result["recon_protein"] = recon_protein
            result["features_protein"] = features_protein
            result["shared_protein"] = shared_protein

        return result


class CrossBioSAELoss(nn.Module):
    """
    Combined loss for CrossBioSAE training.

    L = w_recon * L_recon + w_sparsity * L_sparsity + w_crossmodal * L_crossmodal

    Where:
      L_recon = MSE(input, reconstructed) for each modality
      L_sparsity = L1 norm of features (or implicitly handled by TopK/JumpReLU)
      L_crossmodal = 1 - cosine_sim(features_dna, features_protein) for matched pairs
    """

    def __init__(self, config: CrossBioSAEConfig):
        super().__init__()
        self.config = config

    def reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction loss, normalized by input dimension."""
        return F.mse_loss(reconstructed, original)

    def sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sparsity loss.
        For TopK: L0 (number of active features) — informational only, not differentiable.
        For L1: L1 norm of feature activations.
        For JumpReLU: L0 approximation via sigmoid.
        """
        if self.config.sparsity_type == "topk":
            # TopK enforces sparsity structurally; return L1 for monitoring
            return features.abs().sum(dim=-1).mean()
        elif self.config.sparsity_type == "l1":
            return self.config.l1_coeff * features.abs().sum(dim=-1).mean()
        elif self.config.sparsity_type == "jumprelu":
            # L0 via sigmoid approximation
            return (torch.sigmoid(
                (features - self.config.jumprelu_threshold) / self.config.jumprelu_bandwidth
            )).sum(dim=-1).mean()
        return torch.tensor(0.0, device=features.device)

    def crossmodal_consistency_loss(
        self,
        features_dna: torch.Tensor,
        features_protein: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-modal consistency loss for matched gene pairs.
        Encourages same-gene DNA and protein feature patterns to be similar.
        """
        if self.config.crossmodal_type == "cosine":
            # 1 - cosine_similarity, so loss is 0 when perfectly aligned
            cos_sim = F.cosine_similarity(features_dna, features_protein, dim=-1)
            return (1.0 - cos_sim).mean()
        elif self.config.crossmodal_type == "kl":
            # Symmetrized KL divergence on normalized feature distributions
            p = F.softmax(features_dna, dim=-1) + 1e-8
            q = F.softmax(features_protein, dim=-1) + 1e-8
            kl_pq = F.kl_div(q.log(), p, reduction="batchmean")
            kl_qp = F.kl_div(p.log(), q, reduction="batchmean")
            return (kl_pq + kl_qp) / 2.0
        else:
            raise ValueError(f"Unknown crossmodal loss type: {self.config.crossmodal_type}")

    def forward(
        self,
        dna_acts: Optional[torch.Tensor] = None,
        protein_acts: Optional[torch.Tensor] = None,
        model_output: dict[str, torch.Tensor] = None,
        paired: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            dna_acts: Original DNA activations
            protein_acts: Original protein activations
            model_output: Output dict from CrossBioSAE.forward()
            paired: If True, compute cross-modal consistency loss (same gene pairs)

        Returns:
            Dict with loss components and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(model_output.values())).device)

        # DNA reconstruction
        if dna_acts is not None and "recon_dna" in model_output:
            recon_loss_dna = self.reconstruction_loss(dna_acts, model_output["recon_dna"])
            losses["recon_dna"] = recon_loss_dna
            total_loss = total_loss + self.config.recon_weight * recon_loss_dna

            if self.config.sparsity_type != "topk":
                sparse_loss_dna = self.sparsity_loss(model_output["features_dna"])
                losses["sparsity_dna"] = sparse_loss_dna
                total_loss = total_loss + self.config.sparsity_weight * sparse_loss_dna
            else:
                # For monitoring only
                losses["l0_dna"] = (model_output["features_dna"] > 0).float().sum(dim=-1).mean()

        # Protein reconstruction
        if protein_acts is not None and "recon_protein" in model_output:
            recon_loss_prot = self.reconstruction_loss(protein_acts, model_output["recon_protein"])
            losses["recon_protein"] = recon_loss_prot
            total_loss = total_loss + self.config.recon_weight * recon_loss_prot

            if self.config.sparsity_type != "topk":
                sparse_loss_prot = self.sparsity_loss(model_output["features_protein"])
                losses["sparsity_protein"] = sparse_loss_prot
                total_loss = total_loss + self.config.sparsity_weight * sparse_loss_prot
            else:
                losses["l0_protein"] = (model_output["features_protein"] > 0).float().sum(dim=-1).mean()

        # Cross-modal consistency (only for paired data)
        if paired and "features_dna" in model_output and "features_protein" in model_output:
            cm_loss = self.crossmodal_consistency_loss(
                model_output["features_dna"],
                model_output["features_protein"],
            )
            losses["crossmodal"] = cm_loss
            total_loss = total_loss + self.config.crossmodal_weight * cm_loss

        losses["total"] = total_loss
        return losses


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable parameters by component."""
    counts = {}
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()
    counts["total"] = sum(counts.values())
    return counts
