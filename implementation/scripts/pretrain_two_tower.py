"""
Proper Two-Tower SAE pretrain + per-assay CV evaluation pipeline.

Phase A: Unsupervised reconstruction pretrain on ALL 241k variants
  - SAE_prot(768 → expansion*768, k=tower_k)
  - SAE_dna(768 → expansion*768, k=tower_k)
  - SAE_cross(cross_dim → cross_expansion*cross_dim, k=cross_k)
  - Cosine LR with warmup, gradient clipping, decoder norm constraint
  - Checkpoint saved after Phase A

Evaluation: Per-assay 5-fold CV (Phase B per fold, frozen SAEs)
  - Fair comparison: SAEs see all data (unsupervised), head sees only train fold
  - Also runs MLP baselines on raw features

Usage:
  python pretrain_two_tower.py --phase pretrain  [--epochs 500 --expansion 8 --tower_k 32]
  python pretrain_two_tower.py --phase evaluate  [--checkpoint results/.../pretrained.pt]
  python pretrain_two_tower.py --phase all       (both)
"""

import argparse, glob, os, re, time, json, copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ── AA utilities ──

AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
HYDROPHOBIC = set('AVILMFWP')
CHARGE_MAP = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def parse_mutant(s):
    s = str(s).strip().replace('p.', '')
    if ':' in s: s = s.split(':')[0]
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', s)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def compute_human_features(ref, alt):
    f = np.zeros(9, dtype=np.float32)
    rc, ac = CHARGE_MAP.get(ref, 0), CHARGE_MAP.get(alt, 0)
    f[0] = ac - rc
    f[1] = AA_VOLUME.get(alt, 140) - AA_VOLUME.get(ref, 140)
    f[2] = AA_HYDRO.get(alt, 0) - AA_HYDRO.get(ref, 0)
    f[3] = float(ref == 'C')
    f[4] = float(alt == 'P')
    f[5] = float(ref == 'P')
    f[6] = float(ref == 'G')
    f[7] = float(ref == 'W')
    f[8] = float(rc * ac < 0)
    return f


# ── Model ──

@dataclass
class TwoTowerConfig:
    d_prot: int = 768
    d_dna: int = 768
    expansion: int = 8
    tower_k: int = 32
    cross_proj_dim: int = 256
    cross_expansion: int = 4
    cross_k: int = 8
    n_human: int = 9
    # Phase A
    phase_a_epochs: int = 500
    phase_a_lr: float = 1e-3
    phase_a_batch: int = 4096
    phase_a_warmup: int = 20
    # Phase B (per-assay head training)
    phase_b_epochs: int = 100
    phase_b_lr: float = 3e-3
    phase_b_batch: int = 512
    phase_b_weight_decay: float = 1e-3


class TopKSAE(nn.Module):
    def __init__(self, d_input, n_features, k):
        super().__init__()
        self.k = k
        self.d_input = d_input
        self.n_features = n_features
        self.encoder = nn.Linear(d_input, n_features)
        self.decoder = nn.Linear(n_features, d_input, bias=False)

    def encode(self, x):
        z = F.relu(self.encoder(x))
        if self.k < self.n_features:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            mask = torch.zeros_like(z)
            mask.scatter_(1, topk_idx, 1.0)
            z = z * mask
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        n_prot = cfg.d_prot * cfg.expansion
        n_dna = cfg.d_dna * cfg.expansion

        self.sae_prot = TopKSAE(cfg.d_prot, n_prot, cfg.tower_k)
        self.sae_dna = TopKSAE(cfg.d_dna, n_dna, cfg.tower_k)

        cross_in = n_prot + n_dna
        self.cross_proj = nn.Sequential(
            nn.Linear(cross_in, cfg.cross_proj_dim),
            nn.LayerNorm(cfg.cross_proj_dim),
            nn.GELU(),
        )
        n_cross = cfg.cross_proj_dim * cfg.cross_expansion
        self.sae_cross = TopKSAE(cfg.cross_proj_dim, n_cross, cfg.cross_k)

        self.w_prot = nn.Linear(n_prot, 1, bias=False)
        self.w_dna = nn.Linear(n_dna, 1, bias=False)
        self.w_cross = nn.Linear(n_cross, 1, bias=False)
        self.w_human = nn.Linear(cfg.n_human, 1, bias=False)

        self.gate_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        self.pred_bias = nn.Parameter(torch.zeros(1))

    def encode_towers(self, prot_reps, dna_reps):
        prot_hat, prot_z = self.sae_prot(prot_reps)
        dna_hat, dna_z = self.sae_dna(dna_reps)
        return prot_z, dna_z, prot_hat, dna_hat

    def encode_cross(self, prot_z, dna_z):
        combined = torch.cat([prot_z, dna_z], dim=1)
        proj = self.cross_proj(combined)
        cross_hat, cross_z = self.sae_cross(proj)
        return cross_z, cross_hat, proj

    def compute_gate(self, prot_z, dna_z, cross_z):
        prot_signal = prot_z.abs().sum(dim=1, keepdim=True)
        dna_signal = dna_z.abs().sum(dim=1, keepdim=True)
        cross_signal = cross_z.abs().sum(dim=1, keepdim=True)
        gate_input = torch.cat([prot_signal, dna_signal, cross_signal], dim=1)
        gate_logits = self.gate_net(gate_input)
        return F.softmax(gate_logits, dim=1)

    def predict(self, prot_z, dna_z, cross_z, human_feats, gate_weights):
        s_prot = self.w_prot(prot_z).squeeze(-1)
        s_dna = self.w_dna(dna_z).squeeze(-1)
        s_cross = self.w_cross(cross_z).squeeze(-1)
        s_human = self.w_human(human_feats).squeeze(-1)
        return (gate_weights[:, 0] * s_prot +
                gate_weights[:, 1] * s_dna +
                gate_weights[:, 2] * s_cross +
                s_human + self.pred_bias)

    def forward(self, prot_reps, dna_reps, human_feats):
        prot_z, dna_z, prot_hat, dna_hat = self.encode_towers(prot_reps, dna_reps)
        cross_z, cross_hat, cross_input = self.encode_cross(prot_z, dna_z)
        gate_weights = self.compute_gate(prot_z, dna_z, cross_z)
        score = self.predict(prot_z, dna_z, cross_z, human_feats, gate_weights)
        return {
            'score': score,
            'prot_z': prot_z, 'dna_z': dna_z, 'cross_z': cross_z,
            'prot_hat': prot_hat, 'dna_hat': dna_hat,
            'cross_hat': cross_hat, 'cross_input': cross_input,
            'gate_weights': gate_weights,
        }

    def get_concept_features(self, prot_reps, dna_reps, human_feats):
        """Extract concept activations for prediction head training."""
        with torch.no_grad():
            prot_z, dna_z, _, _ = self.encode_towers(prot_reps, dna_reps)
            cross_z, _, _ = self.encode_cross(prot_z, dna_z)
        return torch.cat([prot_z, dna_z, cross_z, human_feats], dim=1)

    @property
    def concept_dim(self):
        cfg = self.cfg
        return (cfg.d_prot * cfg.expansion +
                cfg.d_dna * cfg.expansion +
                cfg.cross_proj_dim * cfg.cross_expansion +
                cfg.n_human)


# ── Data ──

def load_assay_data(sae_dir):
    sae_files = sorted(glob.glob(os.path.join(sae_dir, "*_sae.npz")))
    data = []
    for sf in sae_files:
        name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        y = d["y_score"]
        mut = d.get("mutants", np.array([]))
        pr = d.get("prot_reps")
        dr = d.get("dna_reps")
        if pr is None or dr is None or len(mut) == 0 or len(mut) != len(y) or len(y) < 50:
            continue
        n = len(mut)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            r, p, a = parse_mutant(mut[i])
            if r and a and r in AA_TO_IDX and a in AA_TO_IDX:
                hf[i] = compute_human_features(r, a)
        data.append({'name': name, 'prot': pr.astype(np.float32),
                     'dna': dr.astype(np.float32), 'y': y.astype(np.float32),
                     'hf': hf, 'mutants': mut})
    return data


# ── Phase A: Pretrain ──

def get_cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_single_sae(sae, data_tensor, cfg, label, device):
    """Train one SAE tower with proper schedule."""
    loader = DataLoader(TensorDataset(data_tensor), batch_size=cfg.phase_a_batch, shuffle=True)
    steps_per_epoch = len(loader)
    total_steps = cfg.phase_a_epochs * steps_per_epoch
    warmup_steps = cfg.phase_a_warmup * steps_per_epoch

    opt = optim.AdamW(sae.parameters(), lr=cfg.phase_a_lr, weight_decay=1e-5)
    best_loss = float('inf')

    for ep in range(cfg.phase_a_epochs):
        ep_loss = 0.0
        n_batch = 0
        for (xb,) in loader:
            step = ep * steps_per_epoch + n_batch
            lr = get_cosine_lr(step, total_steps, warmup_steps, cfg.phase_a_lr)
            for pg in opt.param_groups:
                pg['lr'] = lr

            x_hat, z = sae(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder()

            ep_loss += loss.item()
            n_batch += 1

        avg_loss = ep_loss / n_batch
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                sample = data_tensor[:5000]
                z_sample = sae.encode(sample)
                alive = ((z_sample > 0).float().mean(dim=0) > 0.01).sum().item()
                active_per = (z_sample > 0).float().sum(dim=1).mean().item()
            print("  [%s] Epoch %d/%d: loss=%.6f, alive=%d/%d, active/sample=%.1f, lr=%.2e" % (
                label, ep + 1, cfg.phase_a_epochs, avg_loss,
                int(alive), sae.n_features, active_per, lr), flush=True)

    return best_loss


def train_cross_sae(model, prot_data, dna_data, cfg, device):
    """Train cross SAE with frozen towers."""
    for p in model.sae_prot.parameters():
        p.requires_grad = False
    for p in model.sae_dna.parameters():
        p.requires_grad = False

    cross_params = list(model.cross_proj.parameters()) + list(model.sae_cross.parameters())
    total_steps = cfg.phase_a_epochs * (len(prot_data) // cfg.phase_a_batch + 1)
    warmup_steps = cfg.phase_a_warmup * (len(prot_data) // cfg.phase_a_batch + 1)
    opt = optim.AdamW(cross_params, lr=cfg.phase_a_lr, weight_decay=1e-5)

    for ep in range(cfg.phase_a_epochs):
        ep_loss = 0.0
        n_batch = 0
        indices = torch.randperm(len(prot_data))
        for start in range(0, len(prot_data), cfg.phase_a_batch):
            idx = indices[start:start + cfg.phase_a_batch]
            step = ep * (len(prot_data) // cfg.phase_a_batch + 1) + n_batch
            lr = get_cosine_lr(step, total_steps, warmup_steps, cfg.phase_a_lr)
            for pg in opt.param_groups:
                pg['lr'] = lr

            with torch.no_grad():
                prot_z = model.sae_prot.encode(prot_data[idx])
                dna_z = model.sae_dna.encode(dna_data[idx])

            combined = torch.cat([prot_z, dna_z], dim=1)
            proj = model.cross_proj(combined)
            cross_hat, cross_z = model.sae_cross(proj)
            loss = F.mse_loss(cross_hat, proj.detach())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cross_params, 1.0)
            opt.step()
            model.sae_cross.normalize_decoder()

            ep_loss += loss.item()
            n_batch += 1

        if (ep + 1) % 50 == 0 or ep == 0:
            print("  [CROSS] Epoch %d/%d: loss=%.6f" % (
                ep + 1, cfg.phase_a_epochs, ep_loss / n_batch), flush=True)

    for p in model.sae_prot.parameters():
        p.requires_grad = True
    for p in model.sae_dna.parameters():
        p.requires_grad = True


def phase_a_pretrain(model, all_prot, all_dna, cfg, device, out_dir):
    """Full Phase A: pretrain all three SAEs."""
    print("=" * 70, flush=True)
    print("PHASE A: Reconstruction Pretrain (%d variants)" % len(all_prot), flush=True)
    print("  Config: expansion=%d, tower_k=%d, cross_k=%d, epochs=%d" % (
        cfg.expansion, cfg.tower_k, cfg.cross_k, cfg.phase_a_epochs), flush=True)
    print("  SAE dims: prot=%d→%d, dna=%d→%d, cross=%d→%d" % (
        cfg.d_prot, cfg.d_prot * cfg.expansion,
        cfg.d_dna, cfg.d_dna * cfg.expansion,
        cfg.cross_proj_dim, cfg.cross_proj_dim * cfg.cross_expansion), flush=True)
    print("=" * 70, flush=True)

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)

    t0 = time.time()

    print("\n--- SAE_prot ---", flush=True)
    train_single_sae(model.sae_prot, prot_t, cfg, "PROT", device)

    print("\n--- SAE_dna ---", flush=True)
    train_single_sae(model.sae_dna, dna_t, cfg, "DNA", device)

    print("\n--- SAE_cross ---", flush=True)
    train_cross_sae(model, prot_t, dna_t, cfg, device)

    elapsed = time.time() - t0
    print("\nPhase A complete in %.1f min" % (elapsed / 60), flush=True)

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, "pretrained.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
    }, ckpt_path)
    print("Saved checkpoint: %s" % ckpt_path, flush=True)

    # Report alive features
    model.eval()
    with torch.no_grad():
        sample_p = prot_t[:5000]
        sample_d = dna_t[:5000]
        pz = model.sae_prot.encode(sample_p)
        dz = model.sae_dna.encode(sample_d)
        cz, _, _ = model.encode_cross(pz, dz)

        alive_p = ((pz > 0).float().mean(dim=0) > 0.01).sum().item()
        alive_d = ((dz > 0).float().mean(dim=0) > 0.01).sum().item()
        alive_c = ((cz > 0).float().mean(dim=0) > 0.01).sum().item()

        recon_p = F.mse_loss(model.sae_prot.decode(pz), sample_p).item()
        recon_d = F.mse_loss(model.sae_dna.decode(dz), sample_d).item()

    print("\nPretrain summary:", flush=True)
    print("  PROT alive: %d/%d, recon MSE: %.6f" % (
        int(alive_p), model.sae_prot.n_features, recon_p), flush=True)
    print("  DNA  alive: %d/%d, recon MSE: %.6f" % (
        int(alive_d), model.sae_dna.n_features, recon_d), flush=True)
    print("  CROSS alive: %d/%d" % (int(alive_c), model.sae_cross.n_features), flush=True)


# ── Evaluation: per-assay 5-fold CV ──

class GatedHead(nn.Module):
    """Gated prediction head trained per fold. Produces score + gate weights."""
    def __init__(self, n_prot, n_dna, n_cross, n_human, gate_temp=2.0):
        super().__init__()
        self.w_prot = nn.Linear(n_prot, 1, bias=False)
        self.w_dna = nn.Linear(n_dna, 1, bias=False)
        self.w_cross = nn.Linear(n_cross, 1, bias=False)
        self.w_human = nn.Linear(n_human, 1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        self.pred_bias = nn.Parameter(torch.zeros(1))
        self.gate_temp = gate_temp

    def forward(self, prot_z, dna_z, cross_z, human_feats):
        # L2-normalize gate inputs to prevent magnitude-driven collapse
        prot_signal = prot_z.norm(dim=1, keepdim=True)
        dna_signal = dna_z.norm(dim=1, keepdim=True)
        cross_signal = cross_z.norm(dim=1, keepdim=True)
        gate_input = torch.cat([prot_signal, dna_signal, cross_signal], dim=1)
        gate_input = gate_input / (gate_input.sum(dim=1, keepdim=True) + 1e-8)

        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits / self.gate_temp, dim=1)

        s_prot = self.w_prot(prot_z).squeeze(-1)
        s_dna = self.w_dna(dna_z).squeeze(-1)
        s_cross = self.w_cross(cross_z).squeeze(-1)
        s_human = self.w_human(human_feats).squeeze(-1)

        score = (gate_weights[:, 0] * s_prot +
                 gate_weights[:, 1] * s_dna +
                 gate_weights[:, 2] * s_cross +
                 s_human + self.pred_bias)
        return score, gate_weights


def train_gated_head_on_fold(model, prot_tr, dna_tr, hf_tr, y_tr, cfg, device,
                              entropy_weight=0.05):
    """Train a GatedHead on one fold with entropy regularization."""
    model.eval()
    prot_t = torch.FloatTensor(prot_tr).to(device)
    dna_t = torch.FloatTensor(dna_tr).to(device)
    hf_t = torch.FloatTensor(hf_tr).to(device)
    y_t = torch.FloatTensor(y_tr).to(device)

    with torch.no_grad():
        prot_z, dna_z, _, _ = model.encode_towers(prot_t, dna_t)
        cross_z, _, _ = model.encode_cross(prot_z, dna_z)

    n_prot = prot_z.shape[1]
    n_dna = dna_z.shape[1]
    n_cross = cross_z.shape[1]
    head = GatedHead(n_prot, n_dna, n_cross, cfg.n_human).to(device)

    opt = optim.AdamW(head.parameters(), lr=cfg.phase_b_lr, weight_decay=cfg.phase_b_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase_b_epochs)

    ds = TensorDataset(prot_z, dna_z, cross_z, hf_t, y_t)
    bs = min(cfg.phase_b_batch, len(y_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    head.train()
    for ep in range(cfg.phase_b_epochs):
        for pz, dz, cz, hf, yb in loader:
            pred, gw = head(pz, dz, cz, hf)
            pred_loss = F.mse_loss(pred, yb)
            # Entropy regularization: maximize gate entropy to prevent collapse
            gate_entropy = -(gw * torch.log(gw + 1e-8)).sum(dim=1).mean()
            loss = pred_loss - entropy_weight * gate_entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    return head, (prot_z, dna_z, cross_z)


def predict_with_gated_head(model, head, prot_te, dna_te, hf_te, device):
    model.eval()
    head.eval()
    prot_t = torch.FloatTensor(prot_te).to(device)
    dna_t = torch.FloatTensor(dna_te).to(device)
    hf_t = torch.FloatTensor(hf_te).to(device)
    with torch.no_grad():
        prot_z, dna_z, _, _ = model.encode_towers(prot_t, dna_t)
        cross_z, _, _ = model.encode_cross(prot_z, dna_z)
        scores, gate_weights = head(prot_z, dna_z, cross_z, hf_t)
    return scores.cpu().numpy(), gate_weights.cpu().numpy()


def evaluate_two_tower_cv(model, assay_data, cfg, device):
    """Per-assay 5-fold CV with gated prediction head."""
    results = []
    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_rhos = []
        fold_gates = []
        for tr_idx, te_idx in kf.split(ad['prot']):
            head, _ = train_gated_head_on_fold(
                model,
                ad['prot'][tr_idx], ad['dna'][tr_idx], ad['hf'][tr_idx], ad['y'][tr_idx],
                cfg, device)
            preds, gw = predict_with_gated_head(
                model, head,
                ad['prot'][te_idx], ad['dna'][te_idx], ad['hf'][te_idx], device)
            y_te = ad['y'][te_idx]
            if np.std(preds) > 1e-8 and np.std(y_te) > 1e-8:
                rho = stats.spearmanr(preds, y_te).statistic
                if not np.isnan(rho):
                    fold_rhos.append(rho)
                    fold_gates.append(gw.mean(axis=0))

        if fold_rhos:
            mean_rho = np.mean(fold_rhos)
            mean_gate = np.mean(fold_gates, axis=0)
            results.append({
                'assay': ad['name'], 'n': n,
                'spearman': mean_rho,
                'gate_prot': mean_gate[0], 'gate_dna': mean_gate[1], 'gate_cross': mean_gate[2],
            })

        if (i + 1) % 20 == 0:
            rhos_so_far = [r['spearman'] for r in results]
            print("  %d/%d assays ... mean=%.4f" % (
                i + 1, len(assay_data), np.mean(rhos_so_far)), flush=True)

    return pd.DataFrame(results)


def evaluate_two_tower_linear(model, assay_data, cfg, device):
    """Per-assay 5-fold CV with Ridge on concept features (interpretable baseline)."""
    results = []
    for ad in assay_data:
        n = len(ad['y'])
        if n < 50:
            continue

        model.eval()
        prot_t = torch.FloatTensor(ad['prot']).to(device)
        dna_t = torch.FloatTensor(ad['dna']).to(device)
        hf_t = torch.FloatTensor(ad['hf']).to(device)
        with torch.no_grad():
            concepts = model.get_concept_features(prot_t, dna_t, hf_t).cpu().numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos = []
        for tr, te in kf.split(concepts):
            m = Ridge(alpha=1.0)
            m.fit(concepts[tr], ad['y'][tr])
            yp = m.predict(concepts[te])
            if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                r = stats.spearmanr(yp, ad['y'][te]).statistic
                if not np.isnan(r):
                    rhos.append(r)
        if rhos:
            results.append({'assay': ad['name'], 'n': n, 'tt_ridge': np.mean(rhos)})

    return pd.DataFrame(results)


def evaluate_baselines(assay_data):
    """Baselines: Ridge + MLP on raw features."""
    results = []
    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue
        row = {'assay': ad['name'], 'n': n}

        feature_sets = {
            'prot_ridge': ad['prot'],
            'dna_ridge': ad['dna'],
            'concat_ridge': np.hstack([ad['prot'], ad['dna']]),
            'concat+hf_ridge': np.hstack([ad['prot'], ad['dna'], ad['hf']]),
            'human_ridge': ad['hf'],
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for cname, X in feature_sets.items():
            rhos = []
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0)
                m.fit(X[tr], ad['y'][tr])
                yp = m.predict(X[te])
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        rhos.append(r)
            row[cname] = np.mean(rhos) if rhos else 0

        # MLP baselines (3 random seeds, averaged)
        mlp_sets = {
            'prot_mlp': ad['prot'],
            'concat_mlp': np.hstack([ad['prot'], ad['dna']]),
            'concat+hf_mlp': np.hstack([ad['prot'], ad['dna'], ad['hf']]),
        }
        for cname, X in mlp_sets.items():
            rhos = []
            for tr, te in kf.split(X):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr])
                X_te = scaler.transform(X[te])
                seed_preds = []
                for seed in [42, 123, 456]:
                    m = MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=500,
                                     early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=20,
                                     random_state=seed, learning_rate_init=5e-4,
                                     alpha=1e-3, batch_size=min(256, len(X_tr)))
                    m.fit(X_tr, ad['y'][tr])
                    seed_preds.append(m.predict(X_te))
                yp = np.mean(seed_preds, axis=0)
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        rhos.append(r)
            row[cname] = np.mean(rhos) if rhos else 0

        results.append(row)

        if (i + 1) % 20 == 0:
            rhos_so_far = [r['prot_mlp'] for r in results]
            print("  Baselines: %d/%d assays ... prot_mlp mean=%.4f" % (
                i + 1, len(assay_data), np.mean(rhos_so_far)), flush=True)

    return pd.DataFrame(results)


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["pretrain", "evaluate", "all"])
    ap.add_argument("--sae_dir", default="results/dms_v6")
    ap.add_argument("--out_dir", default="results/two_tower_pretrain")
    ap.add_argument("--checkpoint", default=None, help="Path to pretrained.pt for evaluate-only")
    # Architecture
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--tower_k", type=int, default=32)
    ap.add_argument("--cross_k", type=int, default=8)
    ap.add_argument("--cross_proj_dim", type=int, default=256)
    # Training
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head_epochs", type=int, default=100)
    ap.add_argument("--head_lr", type=float, default=3e-3)
    # Evaluation
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    cfg = TwoTowerConfig(
        expansion=args.expansion,
        tower_k=args.tower_k,
        cross_k=args.cross_k,
        cross_proj_dim=args.cross_proj_dim,
        phase_a_epochs=args.epochs,
        phase_a_batch=args.batch_size,
        phase_a_lr=args.lr,
        phase_b_epochs=args.head_epochs,
        phase_b_lr=args.head_lr,
    )

    # Load data
    print("Loading data from %s ..." % args.sae_dir, flush=True)
    assay_data = load_assay_data(args.sae_dir)
    print("Loaded %d assays" % len(assay_data), flush=True)

    all_prot = np.vstack([ad['prot'] for ad in assay_data])
    all_dna = np.vstack([ad['dna'] for ad in assay_data])
    total_n = len(all_prot)
    print("Total: %d variants" % total_n, flush=True)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Build model
    model = TwoTowerModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: {:,}".format(n_params), flush=True)

    # Phase A
    if args.phase in ("pretrain", "all"):
        phase_a_pretrain(model, all_prot, all_dna, cfg, device, args.out_dir)

    # Load checkpoint for evaluate-only
    if args.phase == "evaluate":
        ckpt = args.checkpoint or os.path.join(args.out_dir, "pretrained.pt")
        print("Loading checkpoint: %s" % ckpt, flush=True)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        # Rebuild config from checkpoint if needed
        if 'config' in state and isinstance(state['config'], dict):
            cfg = TwoTowerConfig(**state['config'])
            model = TwoTowerModel(cfg).to(device)
        model.load_state_dict(state['state_dict'])

    # Evaluation
    if args.phase in ("evaluate", "all"):
        print("\n" + "=" * 70, flush=True)
        print("EVALUATION: Per-assay 5-fold CV (pretrained SAE + per-fold head)", flush=True)
        print("=" * 70, flush=True)

        # Two-Tower with MLP head
        print("\n--- Two-Tower SAE + MLP head ---", flush=True)
        tt_results = evaluate_two_tower_cv(model, assay_data, cfg, device)
        tt_results.to_csv(os.path.join(args.out_dir, "tt_mlp_results.csv"), index=False)

        # Two-Tower with Ridge (interpretable)
        print("\n--- Two-Tower SAE + Ridge (fully interpretable) ---", flush=True)
        tt_ridge = evaluate_two_tower_linear(model, assay_data, cfg, device)

        # Baselines
        if not args.skip_baselines:
            print("\n--- Baselines (Ridge + MLP) ---", flush=True)
            bl_results = evaluate_baselines(assay_data)
            bl_results.to_csv(os.path.join(args.out_dir, "baseline_results.csv"), index=False)

            # Merge
            merged = tt_results[['assay', 'spearman', 'gate_prot', 'gate_dna', 'gate_cross']].rename(
                columns={'spearman': 'tt_mlp'})
            merged = merged.merge(tt_ridge[['assay', 'tt_ridge']], on='assay', how='left')
            merged = merged.merge(
                bl_results.drop(columns=['n'], errors='ignore'),
                on='assay', how='inner')
            merged.to_csv(os.path.join(args.out_dir, "comparison.csv"), index=False)
        else:
            merged = tt_results[['assay', 'spearman']].rename(columns={'spearman': 'tt_mlp'})
            merged = merged.merge(tt_ridge[['assay', 'tt_ridge']], on='assay', how='left')

        # Report
        print("\n" + "=" * 70, flush=True)
        print("RESULTS SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print("%-25s %8s %8s  %s" % ("Method", "Mean", "Median", "Interpretable"), flush=True)
        print("-" * 65, flush=True)

        methods = []
        if 'human_ridge' in merged.columns:
            methods += [('human_ridge', 'YES'), ('dna_ridge', 'no'), ('prot_ridge', 'no'),
                        ('concat_ridge', 'no'), ('concat+hf_ridge', 'no'),
                        ('prot_mlp', 'no'), ('concat_mlp', 'no'), ('concat+hf_mlp', 'no')]
        methods += [('tt_ridge', 'YES (ours)'), ('tt_mlp', 'YES (ours)')]

        for col, interp in methods:
            if col in merged.columns:
                vals = merged[col].dropna()
                print("%-25s %8.4f %8.4f  %s" % (col, vals.mean(), vals.median(), interp), flush=True)

        # Win rates
        if 'concat+hf_mlp' in merged.columns:
            print("\nWin rates:", flush=True)
            for baseline in ['prot_ridge', 'prot_mlp', 'concat_ridge', 'concat_mlp',
                             'concat+hf_ridge', 'concat+hf_mlp']:
                if baseline in merged.columns:
                    valid = merged.dropna(subset=['tt_mlp', baseline])
                    wins = (valid['tt_mlp'] > valid[baseline]).sum()
                    print("  TT_MLP vs %-20s: %d/%d (%.0f%%)" % (
                        baseline, wins, len(valid), 100 * wins / max(len(valid), 1)), flush=True)

        # Gate analysis
        print("\nGate weights (mean):", flush=True)
        for gc in ['gate_prot', 'gate_dna', 'gate_cross']:
            if gc in merged.columns:
                print("  %s: %.1f%%" % (gc, merged[gc].mean() * 100), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
