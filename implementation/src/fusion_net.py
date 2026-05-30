"""
Learned (neural) cross-modal fusion for variant effect -- the AI method for a
SHARED EMBEDDING, replacing the linear CCA baseline.

Each variant carries two frozen embeddings:
  - ESM-2 protein delta (d_prot, e.g. 1280)
  - Evo2  DNA     delta (d_dna,  e.g. 4096)
Two encoders map BOTH modalities into one shared space (d_shared). A gate fuses
them there into a single shared embedding z, and a head classifies z. A
cross-modal alignment loss pulls the two modalities' shared reps of the SAME
variant together, so the space is genuinely shared (echoes CrossBioSAE's
cross-modal consistency loss) -- but here the whole thing is learned end-to-end
on GPU and supervised by the variant label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusionNet(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_hidden=512, d_shared=256, p_drop=0.3):
        super().__init__()
        self.enc_p = nn.Sequential(
            nn.Linear(d_prot, d_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_shared),
        )
        self.enc_d = nn.Sequential(
            nn.Linear(d_dna, d_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_shared),
        )
        # soft gate over the two modalities (in shared space)
        self.gate = nn.Linear(2 * d_shared, 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_shared), nn.Linear(d_shared, d_hidden), nn.GELU(),
            nn.Dropout(p_drop), nn.Linear(d_hidden, 1),
        )

    def forward(self, xp, xd, mask):
        """xp:(B,d_prot) xd:(B,d_dna) mask:(B,1)=1 when protein delta is valid."""
        zp = self.enc_p(xp) * mask          # protein shared rep (zeroed if invalid)
        zd = self.enc_d(xd)                  # DNA shared rep
        g = F.softmax(self.gate(torch.cat([zp, zd], dim=-1)), dim=-1)
        z = g[:, 0:1] * zp + g[:, 1:2] * zd  # FUSED SHARED EMBEDDING
        logit = self.head(z).squeeze(-1)
        return logit, zp, zd, z


def fusion_loss(logit, y, zp, zd, mask, lambda_align=0.1):
    """BCE on the variant label + cross-modal alignment (cosine pull) on the
    variants where both modalities are valid (missense)."""
    bce = F.binary_cross_entropy_with_logits(logit, y)
    m = mask.squeeze(-1) > 0.5
    if m.any():
        cos = F.cosine_similarity(zp[m], zd[m], dim=-1)
        align = (1.0 - cos).mean()
    else:
        align = torch.zeros((), device=logit.device)
    return bce + lambda_align * align, bce.item(), align.item()
