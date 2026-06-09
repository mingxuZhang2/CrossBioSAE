"""
Cross-Modal SAE Pretraining on Raw ESM2(1280) + Evo2(4096) Embeddings.

Joint cross-predictive training:
  Phase A:  Both towers learn reconstruction + cross-prediction jointly
  Phase A2: Cross SAE trained on residual (frozen towers)
  Eval:     Per-assay 5-fold CV + ablation attribution

Supports multi-GPU via DataParallel.

Usage:
  python pretrain_crossmodal_sae.py --phase pretrain
  python pretrain_crossmodal_sae.py --phase evaluate --checkpoint results/.../pretrained.pt
  python pretrain_crossmodal_sae.py --phase all
"""

import argparse, glob, os, re, time, json
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
from sklearn.model_selection import KFold


# ── Config ──

@dataclass
class CrossModalConfig:
    d_prot: int = 1280
    d_dna: int = 4096
    prot_expansion: int = 4
    dna_expansion: int = 2
    prot_k: int = 32
    dna_k: int = 32
    proj_dim: int = 512
    cross_expansion: int = 4
    cross_k: int = 16
    n_human: int = 9
    # Phase A
    phase_a_epochs: int = 300
    phase_a_lr: float = 5e-4
    phase_a_batch: int = 8192
    phase_a_warmup: int = 30
    lambda_xpred: float = 0.3
    lambda_align: float = 0.03
    # Phase A2
    phase_a2_epochs: int = 100
    phase_a2_lr: float = 1e-3
    # Eval
    eval_head_epochs: int = 100
    eval_head_lr: float = 3e-3
    eval_batch: int = 512


# ── AA utilities ──

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
    f[3] = float(ref == 'C'); f[4] = float(alt == 'P')
    f[5] = float(ref == 'P'); f[6] = float(ref == 'G')
    f[7] = float(ref == 'W'); f[8] = float(rc * ac < 0)
    return f


# ── Model ──

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


class CrossModalSAE(nn.Module):
    def __init__(self, cfg: CrossModalConfig):
        super().__init__()
        self.cfg = cfg
        n_prot = cfg.d_prot * cfg.prot_expansion
        n_dna = cfg.d_dna * cfg.dna_expansion
        n_cross = cfg.proj_dim * cfg.cross_expansion

        self.sae_prot = TopKSAE(cfg.d_prot, n_prot, cfg.prot_k)
        self.sae_dna = TopKSAE(cfg.d_dna, n_dna, cfg.dna_k)

        self.proj_prot = nn.Sequential(
            nn.Linear(n_prot, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
        )
        self.proj_dna = nn.Sequential(
            nn.Linear(n_dna, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
        )

        # Cross-prediction decoders (scaffolding, discarded after Phase A)
        self.xpred_p2d = nn.Linear(n_prot, cfg.d_dna, bias=False)
        self.xpred_d2p = nn.Linear(n_dna, cfg.d_prot, bias=False)

        self.sae_cross = TopKSAE(cfg.proj_dim, n_cross, cfg.cross_k)

    def encode_towers(self, prot_in, dna_in):
        prot_hat, prot_z = self.sae_prot(prot_in)
        dna_hat, dna_z = self.sae_dna(dna_in)
        return prot_z, dna_z, prot_hat, dna_hat

    def encode_cross(self, prot_z, dna_z):
        prot_c = self.proj_prot(prot_z)
        dna_c = self.proj_dna(dna_z)
        residual = prot_c - dna_c
        cross_hat, cross_z = self.sae_cross(residual)
        return cross_z, cross_hat, residual, prot_c, dna_c

    def get_all_features(self, prot_in, dna_in, human_feats=None):
        with torch.no_grad():
            prot_z, dna_z, _, _ = self.encode_towers(prot_in, dna_in)
            cross_z, _, _, _, _ = self.encode_cross(prot_z, dna_z)
        parts = [prot_z, dna_z, cross_z]
        if human_feats is not None:
            parts.append(human_feats)
        return torch.cat(parts, dim=1)

    @property
    def n_prot_features(self):
        return self.cfg.d_prot * self.cfg.prot_expansion

    @property
    def n_dna_features(self):
        return self.cfg.d_dna * self.cfg.dna_expansion

    @property
    def n_cross_features(self):
        return self.cfg.proj_dim * self.cfg.cross_expansion

    @property
    def concept_dim(self):
        return self.n_prot_features + self.n_dna_features + self.n_cross_features + self.cfg.n_human


# ── Data ──

def load_raw_embeddings(emb_dir):
    """Load matched ESM2 + Evo2 edelta from dms_embeddings/."""
    esm2_files = sorted(glob.glob(os.path.join(emb_dir, "*_esm2.npz")))
    all_prot, all_dna = [], []
    assay_data = []
    skipped = 0

    for ef in esm2_files:
        base = ef.replace("_esm2.npz", "")
        evo2_file = base + "_evo2.npz"
        if not os.path.exists(evo2_file):
            skipped += 1
            continue

        esm2 = np.load(ef, allow_pickle=True)
        evo2 = np.load(evo2_file, allow_pickle=True)

        valid = evo2['evo2_valid']
        if valid.sum() < 20:
            skipped += 1
            continue

        prot = esm2['esm2_edelta'][valid].astype(np.float32)
        dna = evo2['evo2_edelta'][valid].astype(np.float32)
        y = esm2['y_score'][valid].astype(np.float32)
        muts = esm2['mutants'][valid]
        name = os.path.basename(base)

        # Human features
        n = len(muts)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            r, p, a = parse_mutant(muts[i])
            if r and a:
                hf[i] = compute_human_features(r, a)

        all_prot.append(prot)
        all_dna.append(dna)
        assay_data.append({
            'name': name, 'prot': prot, 'dna': dna,
            'y': y, 'hf': hf, 'mutants': muts,
        })

    all_prot = np.vstack(all_prot)
    all_dna = np.vstack(all_dna)

    print("Loaded %d assays, %d variants (skipped %d)" % (
        len(assay_data), len(all_prot), skipped), flush=True)
    print("  Prot: %s, DNA: %s" % (all_prot.shape, all_dna.shape), flush=True)

    return all_prot, all_dna, assay_data


def standardize(all_prot, all_dna):
    """Standardize embeddings: zero mean + unit variance per dimension."""
    prot_mean = all_prot.mean(axis=0)
    prot_std = all_prot.std(axis=0) + 1e-8
    dna_mean = all_dna.mean(axis=0)
    dna_std = all_dna.std(axis=0) + 1e-8
    all_prot = (all_prot - prot_mean) / prot_std
    all_dna = (all_dna - dna_mean) / dna_std
    print("  Standardized: prot norm %.2f, dna norm %.2f (per-sample mean)" % (
        np.linalg.norm(all_prot, axis=1).mean(),
        np.linalg.norm(all_dna, axis=1).mean()), flush=True)
    return all_prot, all_dna, {'prot_mean': prot_mean, 'prot_std': prot_std,
                                'dna_mean': dna_mean, 'dna_std': dna_std}


# ── Phase A: Joint Cross-Predictive Pretraining ──

def info_nce_loss(prot_c, dna_c, temperature=0.07):
    """Contrastive alignment: matched pairs close, unmatched pairs far."""
    prot_c = F.normalize(prot_c, dim=1)
    dna_c = F.normalize(dna_c, dim=1)
    logits = prot_c @ dna_c.T / temperature
    labels = torch.arange(len(prot_c), device=prot_c.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def get_cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def phase_a_pretrain(model, all_prot, all_dna, cfg, device, out_dir):
    """Joint cross-predictive pretraining of both towers."""
    print("=" * 70, flush=True)
    print("PHASE A: Cross-Predictive Pretraining (%d variants)" % len(all_prot), flush=True)
    print("  SAE_prot: %d→%d (k=%d)" % (cfg.d_prot, cfg.d_prot * cfg.prot_expansion, cfg.prot_k), flush=True)
    print("  SAE_dna:  %d→%d (k=%d)" % (cfg.d_dna, cfg.d_dna * cfg.dna_expansion, cfg.dna_k), flush=True)
    print("  Cross:    %d→%d (k=%d), residual-based" % (
        cfg.proj_dim, cfg.proj_dim * cfg.cross_expansion, cfg.cross_k), flush=True)
    print("  lambda_xpred=%.2f, lambda_align=%.3f" % (cfg.lambda_xpred, cfg.lambda_align), flush=True)
    print("  batch=%d, epochs=%d, lr=%.1e" % (cfg.phase_a_batch, cfg.phase_a_epochs, cfg.phase_a_lr), flush=True)
    print("=" * 70, flush=True)

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)

    # Phase A trains everything except sae_cross
    phase_a_params = (
        list(model.sae_prot.parameters()) +
        list(model.sae_dna.parameters()) +
        list(model.proj_prot.parameters()) +
        list(model.proj_dna.parameters()) +
        list(model.xpred_p2d.parameters()) +
        list(model.xpred_d2p.parameters())
    )

    opt = optim.AdamW(phase_a_params, lr=cfg.phase_a_lr, weight_decay=1e-5)
    N = len(prot_t)
    steps_per_epoch = N // cfg.phase_a_batch + 1
    total_steps = cfg.phase_a_epochs * steps_per_epoch
    warmup_steps = cfg.phase_a_warmup * steps_per_epoch

    t0 = time.time()
    best_loss = float('inf')

    for ep in range(cfg.phase_a_epochs):
        indices = torch.randperm(N, device=device)
        ep_recon = ep_xpred = ep_align = 0.0
        n_batch = 0

        for start in range(0, N, cfg.phase_a_batch):
            step = ep * steps_per_epoch + n_batch
            lr = get_cosine_lr(step, total_steps, warmup_steps, cfg.phase_a_lr)
            for pg in opt.param_groups:
                pg['lr'] = lr

            idx = indices[start:start + cfg.phase_a_batch]
            p_in, d_in = prot_t[idx], dna_t[idx]

            # Forward
            p_hat, p_z = model.sae_prot(p_in)
            d_hat, d_z = model.sae_dna(d_in)

            L_recon = F.mse_loss(p_hat, p_in) + F.mse_loss(d_hat, d_in)

            d_pred = model.xpred_p2d(p_z)
            p_pred = model.xpred_d2p(d_z)
            L_xpred = F.mse_loss(d_pred, d_in) + F.mse_loss(p_pred, p_in)

            p_c = model.proj_prot(p_z)
            d_c = model.proj_dna(d_z)
            L_align = info_nce_loss(p_c, d_c)

            loss = L_recon + cfg.lambda_xpred * L_xpred + cfg.lambda_align * L_align

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase_a_params, 1.0)
            opt.step()
            model.sae_prot.normalize_decoder()
            model.sae_dna.normalize_decoder()

            ep_recon += L_recon.item()
            ep_xpred += L_xpred.item()
            ep_align += L_align.item()
            n_batch += 1

        avg_loss = (ep_recon + cfg.lambda_xpred * ep_xpred + cfg.lambda_align * ep_align) / n_batch
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                s = min(5000, N)
                pz = model.sae_prot.encode(prot_t[:s])
                dz = model.sae_dna.encode(dna_t[:s])
                alive_p = ((pz > 0).float().mean(dim=0) > 0.01).sum().item()
                alive_d = ((dz > 0).float().mean(dim=0) > 0.01).sum().item()
                # Cross-prediction R^2
                d_pred_s = model.xpred_p2d(pz)
                r2_p2d = 1 - F.mse_loss(d_pred_s, dna_t[:s]).item() / dna_t[:s].var().item()
                p_pred_s = model.xpred_d2p(dz)
                r2_d2p = 1 - F.mse_loss(p_pred_s, prot_t[:s]).item() / prot_t[:s].var().item()
                # Alignment: retrieval accuracy (top-1)
                pc = F.normalize(model.proj_prot(pz), dim=1)
                dc = F.normalize(model.proj_dna(dz), dim=1)
                sim = pc @ dc.T
                retrieval_acc = (sim.argmax(dim=1) == torch.arange(s, device=sim.device)).float().mean().item()

            elapsed = (time.time() - t0) / 60
            print("  Ep %d/%d (%.1fmin): recon=%.5f xpred=%.5f align=%.4f | "
                  "alive P=%d/%d D=%d/%d | R2 p→d=%.4f d→p=%.4f | R@1=%.4f" % (
                      ep + 1, cfg.phase_a_epochs, elapsed,
                      ep_recon / n_batch, ep_xpred / n_batch, ep_align / n_batch,
                      int(alive_p), model.n_prot_features,
                      int(alive_d), model.n_dna_features,
                      r2_p2d, r2_d2p, retrieval_acc), flush=True)

    print("\nPhase A done in %.1f min" % ((time.time() - t0) / 60), flush=True)


def phase_a2_cross(model, all_prot, all_dna, cfg, device):
    """Train cross SAE on residual = proj_prot(prot_z) - proj_dna(dna_z)."""
    print("\n" + "=" * 70, flush=True)
    print("PHASE A2: Cross SAE on Residual (%d epochs)" % cfg.phase_a2_epochs, flush=True)
    print("=" * 70, flush=True)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.sae_cross.parameters():
        p.requires_grad = True

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)
    N = len(prot_t)

    opt = optim.AdamW(model.sae_cross.parameters(), lr=cfg.phase_a2_lr, weight_decay=1e-5)

    for ep in range(cfg.phase_a2_epochs):
        indices = torch.randperm(N, device=device)
        ep_loss = 0.0
        n_batch = 0
        for start in range(0, N, cfg.phase_a_batch):
            idx = indices[start:start + cfg.phase_a_batch]
            with torch.no_grad():
                pz = model.sae_prot.encode(prot_t[idx])
                dz = model.sae_dna.encode(dna_t[idx])
                pc = model.proj_prot(pz)
                dc = model.proj_dna(dz)
                residual = pc - dc

            cross_hat, cross_z = model.sae_cross(residual)
            loss = F.mse_loss(cross_hat, residual)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.sae_cross.normalize_decoder()

            ep_loss += loss.item()
            n_batch += 1

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                s = min(5000, N)
                pz = model.sae_prot.encode(prot_t[:s])
                dz = model.sae_dna.encode(dna_t[:s])
                pc = model.proj_prot(pz)
                dc = model.proj_dna(dz)
                res = pc - dc
                cz = model.sae_cross.encode(res)
                alive_c = ((cz > 0).float().mean(dim=0) > 0.01).sum().item()
            print("  Ep %d/%d: loss=%.6f, alive=%d/%d" % (
                ep + 1, cfg.phase_a2_epochs, ep_loss / n_batch,
                int(alive_c), model.n_cross_features), flush=True)

    for p in model.parameters():
        p.requires_grad = True

    print("Phase A2 done.", flush=True)


def save_checkpoint(model, cfg, norm_stats, out_dir):
    ckpt_path = os.path.join(out_dir, "pretrained.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
        'norm_stats': norm_stats,
    }, ckpt_path)
    print("Saved: %s" % ckpt_path, flush=True)

    # Report alive features
    print("\nPretrain summary:", flush=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy_p = torch.zeros(1, cfg.d_prot, device=device)
        dummy_d = torch.zeros(1, cfg.d_dna, device=device)
        pz = model.sae_prot.encode(dummy_p)
        dz = model.sae_dna.encode(dummy_d)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Total params: {:,}".format(n_params), flush=True)
    print("  Features: prot=%d, dna=%d, cross=%d, total=%d" % (
        model.n_prot_features, model.n_dna_features,
        model.n_cross_features,
        model.n_prot_features + model.n_dna_features + model.n_cross_features), flush=True)


# ── Evaluation ──

class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_head(X_tr, y_tr, input_dim, cfg, device):
    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_tr).to(device)
    head = PredictionHead(input_dim, hidden=256).to(device)
    opt = optim.AdamW(head.parameters(), lr=cfg.eval_head_lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.eval_head_epochs)
    ds = TensorDataset(X_t, y_t)
    bs = min(cfg.eval_batch, len(y_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    head.train()
    for _ in range(cfg.eval_head_epochs):
        for xb, yb in loader:
            pred = head(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head


def apply_norm(data, mean, std):
    return ((data - mean) / std).astype(np.float32)


def evaluate_ablation(model, assay_data, cfg, device, norm_stats):
    """Per-assay ablation: Ridge CV on feature subsets for modality attribution."""
    MODES = [
        ('full', ['prot', 'dna', 'cross', 'human']),
        ('prot_only', ['prot', 'human']),
        ('dna_only', ['dna', 'human']),
        ('cross_only', ['cross', 'human']),
        ('no_prot', ['dna', 'cross', 'human']),
        ('no_dna', ['prot', 'cross', 'human']),
        ('no_cross', ['prot', 'dna', 'human']),
    ]

    print("\n--- Ablation: modality attribution (Ridge CV) ---", flush=True)
    results = []

    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        prot_mc = apply_norm(ad['prot'], norm_stats['prot_mean'], norm_stats['prot_std'])
        dna_mc = apply_norm(ad['dna'], norm_stats['dna_mean'], norm_stats['dna_std'])
        model.eval()
        pt = torch.FloatTensor(prot_mc).to(device)
        dt = torch.FloatTensor(dna_mc).to(device)
        with torch.no_grad():
            pz, dz, _, _ = model.encode_towers(pt, dt)
            cz, _, _, _, _ = model.encode_cross(pz, dz)

        feats = {
            'prot': pz.cpu().numpy(),
            'dna': dz.cpu().numpy(),
            'cross': cz.cpu().numpy(),
            'human': ad['hf'],
        }

        row = {'assay': ad['name'], 'n': n}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for mode_name, mode_keys in MODES:
            X = np.hstack([feats[k] for k in mode_keys])
            rhos = []
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0)
                m.fit(X[tr], ad['y'][tr])
                yp = m.predict(X[te])
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        rhos.append(r)
            row[mode_name] = np.mean(rhos) if rhos else 0

        row['imp_prot'] = row['full'] - row['no_prot']
        row['imp_dna'] = row['full'] - row['no_dna']
        row['imp_cross'] = row['full'] - row['no_cross']
        results.append(row)

        if (i + 1) % 50 == 0:
            df_tmp = pd.DataFrame(results)
            print("  %d/%d ... full=%.4f imp: P=%.4f D=%.4f C=%.4f" % (
                i + 1, len(assay_data), df_tmp['full'].mean(),
                df_tmp['imp_prot'].mean(), df_tmp['imp_dna'].mean(),
                df_tmp['imp_cross'].mean()), flush=True)

    return pd.DataFrame(results)


def evaluate_mlp_cv(model, assay_data, cfg, device, norm_stats):
    """Per-assay 5-fold CV with MLP head on SAE features."""
    print("\n--- Two-Tower MLP CV ---", flush=True)
    results = []

    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        prot_mc = apply_norm(ad['prot'], norm_stats['prot_mean'], norm_stats['prot_std'])
        dna_mc = apply_norm(ad['dna'], norm_stats['dna_mean'], norm_stats['dna_std'])
        model.eval()
        pt = torch.FloatTensor(prot_mc).to(device)
        dt = torch.FloatTensor(dna_mc).to(device)
        ht = torch.FloatTensor(ad['hf']).to(device)

        with torch.no_grad():
            concepts = model.get_all_features(pt, dt, ht).cpu().numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos = []
        for tr, te in kf.split(concepts):
            head = train_mlp_head(concepts[tr], ad['y'][tr], concepts.shape[1], cfg, device)
            head.eval()
            with torch.no_grad():
                preds = head(torch.FloatTensor(concepts[te]).to(device)).cpu().numpy()
            if np.std(preds) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                r = stats.spearmanr(preds, ad['y'][te]).statistic
                if not np.isnan(r):
                    rhos.append(r)

        if rhos:
            results.append({'assay': ad['name'], 'n': n, 'tt_mlp': np.mean(rhos)})

        if (i + 1) % 20 == 0:
            rhos_so_far = [r['tt_mlp'] for r in results]
            print("  %d/%d ... mean=%.4f" % (
                i + 1, len(assay_data), np.mean(rhos_so_far)), flush=True)

    return pd.DataFrame(results)


def evaluate_baselines(assay_data, norm_stats):
    """Raw feature baselines."""
    print("\n--- Baselines (Ridge) ---", flush=True)
    results = []

    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        prot_mc = apply_norm(ad['prot'], norm_stats['prot_mean'], norm_stats['prot_std'])
        dna_mc = apply_norm(ad['dna'], norm_stats['dna_mean'], norm_stats['dna_std'])

        feature_sets = {
            'prot_ridge': prot_mc,
            'dna_ridge': dna_mc,
            'concat_ridge': np.hstack([prot_mc, dna_mc]),
            'human_ridge': ad['hf'],
        }

        row = {'assay': ad['name'], 'n': n}
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

        results.append(row)

        if (i + 1) % 50 == 0:
            print("  %d/%d ... prot=%.4f" % (
                i + 1, len(assay_data), np.mean([r['prot_ridge'] for r in results])), flush=True)

    return pd.DataFrame(results)


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["pretrain", "evaluate", "all"])
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--out_dir", default="results/crossmodal_sae")
    ap.add_argument("--checkpoint", default=None)
    # Architecture
    ap.add_argument("--prot_expansion", type=int, default=4)
    ap.add_argument("--dna_expansion", type=int, default=2)
    ap.add_argument("--prot_k", type=int, default=32)
    ap.add_argument("--dna_k", type=int, default=32)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--cross_k", type=int, default=16)
    # Training
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_xpred", type=float, default=0.3)
    ap.add_argument("--lambda_align", type=float, default=0.03)
    # Eval
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--skip_mlp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    print("Device: %s (%d GPUs)" % (device, n_gpu), flush=True)

    cfg = CrossModalConfig(
        prot_expansion=args.prot_expansion,
        dna_expansion=args.dna_expansion,
        prot_k=args.prot_k,
        dna_k=args.dna_k,
        proj_dim=args.proj_dim,
        cross_k=args.cross_k,
        phase_a_epochs=args.epochs,
        phase_a_batch=args.batch,
        phase_a_lr=args.lr,
        lambda_xpred=args.lambda_xpred,
        lambda_align=args.lambda_align,
    )

    # Load data
    print("Loading raw embeddings from %s ..." % args.emb_dir, flush=True)
    all_prot, all_dna, assay_data = load_raw_embeddings(args.emb_dir)

    # Standardize
    print("Standardizing ...", flush=True)
    all_prot, all_dna, norm_stats = standardize(all_prot, all_dna)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Build model
    model = CrossModalSAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model: {:,} parameters".format(n_params), flush=True)
    print("  SAE_prot: %d→%d (k=%d)" % (cfg.d_prot, model.n_prot_features, cfg.prot_k), flush=True)
    print("  SAE_dna:  %d→%d (k=%d)" % (cfg.d_dna, model.n_dna_features, cfg.dna_k), flush=True)
    print("  SAE_cross: %d→%d (k=%d)" % (cfg.proj_dim, model.n_cross_features, cfg.cross_k), flush=True)

    # Pretrain
    if args.phase in ("pretrain", "all"):
        phase_a_pretrain(model, all_prot, all_dna, cfg, device, args.out_dir)
        phase_a2_cross(model, all_prot, all_dna, cfg, device)
        save_checkpoint(model, cfg, norm_stats, args.out_dir)

    # Load checkpoint
    if args.phase == "evaluate":
        ckpt = args.checkpoint or os.path.join(args.out_dir, "pretrained.pt")
        print("Loading: %s" % ckpt, flush=True)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if 'config' in state:
            cfg = CrossModalConfig(**state['config'])
            model = CrossModalSAE(cfg).to(device)
        model.load_state_dict(state['state_dict'])
        norm_stats = state['norm_stats']

    # Evaluate
    if args.phase in ("evaluate", "all"):
        print("\n" + "=" * 70, flush=True)
        print("EVALUATION", flush=True)
        print("=" * 70, flush=True)

        # Ablation
        abl = evaluate_ablation(model, assay_data, cfg, device, norm_stats)
        abl.to_csv(os.path.join(args.out_dir, "ablation.csv"), index=False)

        # MLP CV
        if not args.skip_mlp:
            mlp = evaluate_mlp_cv(model, assay_data, cfg, device, norm_stats)
            mlp.to_csv(os.path.join(args.out_dir, "mlp_results.csv"), index=False)

        # Baselines
        if not args.skip_baselines:
            bl = evaluate_baselines(assay_data, norm_stats)
            bl.to_csv(os.path.join(args.out_dir, "baselines.csv"), index=False)

        # Report
        print("\n" + "=" * 70, flush=True)
        print("RESULTS SUMMARY", flush=True)
        print("=" * 70, flush=True)

        # Ablation
        print("\nAblation (Ridge CV):", flush=True)
        print("%-20s %8s %8s" % ("Mode", "Mean", "Median"), flush=True)
        print("-" * 40, flush=True)
        for col in ['full', 'prot_only', 'dna_only', 'cross_only']:
            if col in abl.columns:
                v = abl[col].dropna()
                print("%-20s %8.4f %8.4f" % (col, v.mean(), v.median()), flush=True)

        print("\nModality importance (Δ Spearman):", flush=True)
        print("%-20s %8s %8s %8s" % ("", "PROT", "DNA", "CROSS"), flush=True)
        print("-" * 50, flush=True)
        for stat, fn in [("mean", np.mean), ("median", np.median), (">0 count", lambda x: (x > 0).sum())]:
            vals = [fn(abl[c].dropna().values) for c in ['imp_prot', 'imp_dna', 'imp_cross']]
            if stat == ">0 count":
                print("%-20s %8d %8d %8d" % (stat, *vals), flush=True)
            else:
                print("%-20s %8.4f %8.4f %8.4f" % (stat, *vals), flush=True)

        # MLP
        if not args.skip_mlp and len(mlp):
            print("\nMLP CV: mean=%.4f median=%.4f" % (mlp['tt_mlp'].mean(), mlp['tt_mlp'].median()), flush=True)

        # Baselines
        if not args.skip_baselines and len(bl):
            print("\nBaselines:", flush=True)
            for col in ['human_ridge', 'dna_ridge', 'prot_ridge', 'concat_ridge']:
                if col in bl.columns:
                    v = bl[col].dropna()
                    print("  %-20s mean=%.4f" % (col, v.mean()), flush=True)

        # Top assays
        print("\nTop protein-driven (by imp_prot):", flush=True)
        for _, r in abl.nlargest(5, 'imp_prot').iterrows():
            print("  %-40s P=%+.3f D=%+.3f C=%+.3f" % (
                r['assay'][:40], r['imp_prot'], r['imp_dna'], r['imp_cross']), flush=True)
        print("\nTop DNA-driven (by imp_dna):", flush=True)
        for _, r in abl.nlargest(5, 'imp_dna').iterrows():
            print("  %-40s P=%+.3f D=%+.3f C=%+.3f" % (
                r['assay'][:40], r['imp_prot'], r['imp_dna'], r['imp_cross']), flush=True)
        print("\nTop cross-modal (by imp_cross):", flush=True)
        for _, r in abl.nlargest(5, 'imp_cross').iterrows():
            print("  %-40s P=%+.3f D=%+.3f C=%+.3f" % (
                r['assay'][:40], r['imp_prot'], r['imp_dna'], r['imp_cross']), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
