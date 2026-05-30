"""
CLIP-style contrastive pretraining for cross-modal gene representation.

Learns a shared embedding space where the same gene's protein and DNA
representations are close, different genes are far apart. Self-supervised —
the pairing itself is the signal, no labels needed.

After pretraining, the dual encoders define a shared space that can be used
for downstream tasks (variant effect prediction, function transfer, etc.).
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GeneEncoder(nn.Module):
    """MLP encoder: raw activation → shared space."""
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, p_drop=0.1):
        super().__init__()
        layers = []
        d = d_in
        for i in range(n_layers):
            layers += [nn.Linear(d, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden),
                       nn.Dropout(p_drop)]
            d = d_hidden
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class CrossModalCLIP(nn.Module):
    """Dual-encoder contrastive model for protein × DNA gene representations."""
    def __init__(self, d_prot=1280, d_dna=1024, d_hidden=512, d_shared=256,
                 n_layers=2, p_drop=0.1, temperature=0.07):
        super().__init__()
        self.enc_prot = GeneEncoder(d_prot, d_hidden, d_shared, n_layers, p_drop)
        self.enc_dna = GeneEncoder(d_dna, d_hidden, d_shared, n_layers, p_drop)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, xp, xd):
        zp = self.enc_prot(xp)
        zd = self.enc_dna(xd)
        return zp, zd

    def contrastive_loss(self, zp, zd):
        """Symmetric InfoNCE (CLIP-style)."""
        temp = self.log_temp.exp().clamp(max=100.0)
        logits = zp @ zd.T * temp  # (B, B)
        labels = torch.arange(len(zp), device=zp.device)
        loss_p2d = F.cross_entropy(logits, labels)
        loss_d2p = F.cross_entropy(logits.T, labels)
        return (loss_p2d + loss_d2p) / 2

    @torch.no_grad()
    def retrieval_accuracy(self, zp, zd):
        """Top-1 retrieval accuracy (protein→DNA and DNA→protein)."""
        sim = zp @ zd.T
        p2d = (sim.argmax(dim=1) == torch.arange(len(zp), device=zp.device)).float().mean()
        d2p = (sim.argmax(dim=0) == torch.arange(len(zd), device=zd.device)).float().mean()
        return {"p2d_acc": p2d.item(), "d2p_acc": d2p.item()}


class VariantFusionHead(nn.Module):
    """Fine-tuning head for variant effect prediction from pretrained encoders.

    Takes per-variant embedding deltas (ref→alt), projects through the frozen
    pretrained encoders into the shared space, fuses there, and classifies.
    """
    def __init__(self, pretrained_clip, d_shared=256, d_hidden=128, p_drop=0.3,
                 freeze_encoders=True):
        super().__init__()
        self.enc_prot = pretrained_clip.enc_prot
        self.enc_dna = pretrained_clip.enc_dna
        if freeze_encoders:
            for p in self.enc_prot.parameters():
                p.requires_grad = False
            for p in self.enc_dna.parameters():
                p.requires_grad = False
        # gate over modalities in shared space
        self.gate = nn.Sequential(
            nn.Linear(d_shared * 2, d_hidden), nn.GELU(), nn.Linear(d_hidden, 2))
        self.head = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, d_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, xp, xd, mask):
        """xp: protein delta, xd: DNA delta, mask: 1 if protein delta valid."""
        zp = self.enc_prot(xp) * mask
        zd = self.enc_dna(xd)
        g = F.softmax(self.gate(torch.cat([zp, zd], dim=-1)), dim=-1)
        z = g[:, 0:1] * zp + g[:, 1:2] * zd
        return self.head(z).squeeze(-1), zp, zd
