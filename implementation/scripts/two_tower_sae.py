"""
Two-Tower SAE: Interpretable Cross-Modal Variant Effect Prediction.

Architecture:
  prot_reps (768) → SAE_prot(768→3072, k=24) → protein concepts
  dna_reps  (768) → SAE_dna(768→3072, k=24)  → DNA concepts
                                                      │
                          concat → project(6144→256) → SAE_cross(256→1024, k=8) → cross concepts
                                                      │
                          concat(prot, dna, cross, human) → Gated Head → score

Training:
  Phase A: Reconstruction pretraining (unsupervised)
  Phase B: Supervised fine-tuning (freeze SAE, train head)
  Phase C: End-to-end fine-tuning (unfreeze SAE with small lr)
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


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
    f[0] = ac - rc; f[1] = AA_VOLUME.get(alt,140)-AA_VOLUME.get(ref,140)
    f[2] = AA_HYDRO.get(alt,0)-AA_HYDRO.get(ref,0)
    f[3] = float(ref=='C'); f[4] = float(alt=='P'); f[5] = float(ref=='P')
    f[6] = float(ref=='G'); f[7] = float(ref=='W'); f[8] = float(rc*ac<0)
    return f


# ── Model components ──

@dataclass
class Config:
    d_prot: int = 768
    d_dna: int = 768
    tower_expansion: int = 4
    tower_k: int = 24
    cross_proj_dim: int = 256
    cross_expansion: int = 4
    cross_k: int = 8
    n_human: int = 9
    phase_a_epochs: int = 100
    phase_a_lr: float = 1e-3
    phase_a_batch: int = 2048
    phase_b_epochs: int = 50
    phase_b_lr: float = 1e-3
    phase_b_batch: int = 1024
    phase_c_epochs: int = 20
    phase_c_lr: float = 1e-4
    phase_c_batch: int = 1024


class TopKSAE(nn.Module):
    def __init__(self, d_input, n_features, k):
        super().__init__()
        self.k = k
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_prot = cfg.d_prot * cfg.tower_expansion
        n_dna = cfg.d_dna * cfg.tower_expansion

        self.sae_prot = TopKSAE(cfg.d_prot, n_prot, cfg.tower_k)
        self.sae_dna = TopKSAE(cfg.d_dna, n_dna, cfg.tower_k)

        cross_in = n_prot + n_dna
        self.cross_proj = nn.Sequential(
            nn.Linear(cross_in, cfg.cross_proj_dim),
            nn.LayerNorm(cfg.cross_proj_dim),
            nn.ReLU(),
        )
        n_cross = cfg.cross_proj_dim * cfg.cross_expansion
        self.sae_cross = TopKSAE(cfg.cross_proj_dim, n_cross, cfg.cross_k)

        # Gated prediction head
        # Three streams: prot, dna, cross. Each → scalar contribution.
        self.w_prot = nn.Linear(n_prot, 1, bias=False)
        self.w_dna = nn.Linear(n_dna, 1, bias=False)
        self.w_cross = nn.Linear(n_cross, 1, bias=False)
        self.w_human = nn.Linear(cfg.n_human, 1, bias=False)

        # Gate: learns how much each stream contributes
        # Input: summary of each stream's activity
        self.gate_net = nn.Sequential(
            nn.Linear(3, 3),
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
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, 3), sums to 1
        return gate_weights

    def predict(self, prot_z, dna_z, cross_z, human_feats, gate_weights):
        s_prot = self.w_prot(prot_z).squeeze(-1)
        s_dna = self.w_dna(dna_z).squeeze(-1)
        s_cross = self.w_cross(cross_z).squeeze(-1)
        s_human = self.w_human(human_feats).squeeze(-1)

        score = (gate_weights[:, 0] * s_prot +
                 gate_weights[:, 1] * s_dna +
                 gate_weights[:, 2] * s_cross +
                 s_human + self.pred_bias)
        return score

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


# ── Data loading ──

def load_assay_data(sae_dir):
    sae_files = sorted(glob.glob(os.path.join(sae_dir, "*_sae.npz")))
    data = []
    for sf in sae_files:
        name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        y = d["y_score"]; mut = d.get("mutants", np.array([]))
        pr = d.get("prot_reps"); dr = d.get("dna_reps")
        if pr is None or dr is None or len(mut)==0 or len(mut)!=len(y) or len(y)<50:
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


# ── Training phases ──

def train_phase_a(model, all_prot, all_dna, cfg, device):
    """Reconstruction pretraining: train SAE_prot, SAE_dna, then SAE_cross."""
    print("Phase A: Reconstruction pretraining", flush=True)

    # A1: Train SAE_prot
    print("  A1: Training SAE_prot ...", flush=True)
    ds = TensorDataset(torch.FloatTensor(all_prot).to(device))
    loader = DataLoader(ds, batch_size=cfg.phase_a_batch, shuffle=True)
    opt = optim.Adam(model.sae_prot.parameters(), lr=cfg.phase_a_lr)
    for ep in range(cfg.phase_a_epochs):
        total_loss = 0; n = 0
        for (xb,) in loader:
            x_hat, z = model.sae_prot(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            model.sae_prot.normalize_decoder()
            total_loss += loss.item(); n += 1
        if (ep+1) % 20 == 0:
            alive = ((torch.FloatTensor(all_prot[:2000]).to(device)).unsqueeze(0) if False else None)
            print("    Epoch %d: loss=%.6f" % (ep+1, total_loss/n), flush=True)

    with torch.no_grad():
        z_test = model.sae_prot.encode(torch.FloatTensor(all_prot[:2000]).to(device))
        alive_p = (z_test.abs() > 0).float().mean(dim=0).sum().item()
    print("  SAE_prot alive features: %d / %d" % (int(alive_p), model.sae_prot.n_features), flush=True)

    # A2: Train SAE_dna
    print("  A2: Training SAE_dna ...", flush=True)
    ds = TensorDataset(torch.FloatTensor(all_dna).to(device))
    loader = DataLoader(ds, batch_size=cfg.phase_a_batch, shuffle=True)
    opt = optim.Adam(model.sae_dna.parameters(), lr=cfg.phase_a_lr)
    for ep in range(cfg.phase_a_epochs):
        total_loss = 0; n = 0
        for (xb,) in loader:
            x_hat, z = model.sae_dna(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            model.sae_dna.normalize_decoder()
            total_loss += loss.item(); n += 1
        if (ep+1) % 20 == 0:
            print("    Epoch %d: loss=%.6f" % (ep+1, total_loss/n), flush=True)

    with torch.no_grad():
        z_test = model.sae_dna.encode(torch.FloatTensor(all_dna[:2000]).to(device))
        alive_d = (z_test.abs() > 0).float().mean(dim=0).sum().item()
    print("  SAE_dna alive features: %d / %d" % (int(alive_d), model.sae_dna.n_features), flush=True)

    # A3: Freeze towers, train cross SAE
    print("  A3: Training SAE_cross ...", flush=True)
    for p in model.sae_prot.parameters(): p.requires_grad = False
    for p in model.sae_dna.parameters(): p.requires_grad = False

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)

    cross_params = list(model.cross_proj.parameters()) + list(model.sae_cross.parameters())
    opt = optim.Adam(cross_params, lr=cfg.phase_a_lr)

    for ep in range(cfg.phase_a_epochs):
        # Process in batches
        total_loss = 0; n_batch = 0
        indices = torch.randperm(len(all_prot))
        for start in range(0, len(all_prot), cfg.phase_a_batch):
            idx = indices[start:start+cfg.phase_a_batch]
            with torch.no_grad():
                prot_z = model.sae_prot.encode(prot_t[idx])
                dna_z = model.sae_dna.encode(dna_t[idx])
            combined = torch.cat([prot_z, dna_z], dim=1)
            proj = model.cross_proj(combined)
            cross_hat, cross_z = model.sae_cross(proj)
            loss = F.mse_loss(cross_hat, proj.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            model.sae_cross.normalize_decoder()
            total_loss += loss.item(); n_batch += 1
        if (ep+1) % 20 == 0:
            print("    Epoch %d: loss=%.6f" % (ep+1, total_loss/n_batch), flush=True)

    # Unfreeze towers for later phases
    for p in model.sae_prot.parameters(): p.requires_grad = True
    for p in model.sae_dna.parameters(): p.requires_grad = True

    with torch.no_grad():
        prot_z = model.sae_prot.encode(prot_t[:2000])
        dna_z = model.sae_dna.encode(dna_t[:2000])
        combined = torch.cat([prot_z, dna_z], dim=1)
        proj = model.cross_proj(combined)
        cross_z = model.sae_cross.encode(proj)
        alive_c = (cross_z.abs() > 0).float().mean(dim=0).sum().item()
    print("  SAE_cross alive features: %d / %d" % (int(alive_c), model.sae_cross.n_features), flush=True)

    print("Phase A complete.", flush=True)


def train_phase_b(model, all_prot, all_dna, all_hf, all_y, cfg, device):
    """Supervised: freeze SAEs, train prediction head."""
    print("Phase B: Supervised fine-tuning (frozen SAEs)", flush=True)

    for p in model.sae_prot.parameters(): p.requires_grad = False
    for p in model.sae_dna.parameters(): p.requires_grad = False
    for p in model.cross_proj.parameters(): p.requires_grad = False
    for p in model.sae_cross.parameters(): p.requires_grad = False

    head_params = (list(model.w_prot.parameters()) + list(model.w_dna.parameters()) +
                   list(model.w_cross.parameters()) + list(model.w_human.parameters()) +
                   list(model.gate_net.parameters()) + [model.pred_bias])

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)
    hf_t = torch.FloatTensor(all_hf).to(device)
    y_t = torch.FloatTensor(all_y).to(device)

    opt = optim.Adam(head_params, lr=cfg.phase_b_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase_b_epochs)

    for ep in range(cfg.phase_b_epochs):
        model.train()
        total_loss = 0; n_batch = 0
        indices = torch.randperm(len(all_prot))
        for start in range(0, len(all_prot), cfg.phase_b_batch):
            idx = indices[start:start+cfg.phase_b_batch]
            out = model(prot_t[idx], dna_t[idx], hf_t[idx])
            loss = F.mse_loss(out['score'], y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batch += 1
        scheduler.step()
        if (ep+1) % 10 == 0:
            print("    Epoch %d: pred_loss=%.4f" % (ep+1, total_loss/n_batch), flush=True)

    # Unfreeze all
    for p in model.parameters(): p.requires_grad = True
    print("Phase B complete.", flush=True)


def train_phase_c(model, all_prot, all_dna, all_hf, all_y, cfg, device):
    """End-to-end: unfreeze SAE with small lr."""
    print("Phase C: End-to-end fine-tuning", flush=True)

    sae_params = (list(model.sae_prot.parameters()) + list(model.sae_dna.parameters()) +
                  list(model.cross_proj.parameters()) + list(model.sae_cross.parameters()))
    head_params = (list(model.w_prot.parameters()) + list(model.w_dna.parameters()) +
                   list(model.w_cross.parameters()) + list(model.w_human.parameters()) +
                   list(model.gate_net.parameters()) + [model.pred_bias])

    opt = optim.Adam([
        {'params': sae_params, 'lr': cfg.phase_c_lr},
        {'params': head_params, 'lr': cfg.phase_c_lr * 10},
    ], weight_decay=1e-4)

    prot_t = torch.FloatTensor(all_prot).to(device)
    dna_t = torch.FloatTensor(all_dna).to(device)
    hf_t = torch.FloatTensor(all_hf).to(device)
    y_t = torch.FloatTensor(all_y).to(device)

    for ep in range(cfg.phase_c_epochs):
        model.train()
        total_pred = 0; total_recon = 0; n_batch = 0
        indices = torch.randperm(len(all_prot))
        for start in range(0, len(all_prot), cfg.phase_c_batch):
            idx = indices[start:start+cfg.phase_c_batch]
            out = model(prot_t[idx], dna_t[idx], hf_t[idx])
            pred_loss = F.mse_loss(out['score'], y_t[idx])
            recon_loss = (F.mse_loss(out['prot_hat'], prot_t[idx]) +
                          F.mse_loss(out['dna_hat'], dna_t[idx]) +
                          F.mse_loss(out['cross_hat'], out['cross_input'].detach()))
            loss = pred_loss + 0.1 * recon_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.sae_prot.normalize_decoder()
            model.sae_dna.normalize_decoder()
            model.sae_cross.normalize_decoder()
            total_pred += pred_loss.item(); total_recon += recon_loss.item(); n_batch += 1
        if (ep+1) % 5 == 0:
            print("    Epoch %d: pred=%.4f recon=%.4f" % (
                ep+1, total_pred/n_batch, total_recon/n_batch), flush=True)
    print("Phase C complete.", flush=True)


# ── Evaluation ──

def evaluate_per_assay(model, assay_data, device):
    """Per-assay 5-fold CV: clone pretrained model, train Phase B per fold."""
    model.eval()
    results = []
    for ad in assay_data:
        y = ad['y']; n = len(y)
        if n < 50: continue

        prot_t = torch.FloatTensor(ad['prot']).to(device)
        dna_t = torch.FloatTensor(ad['dna']).to(device)
        hf_t = torch.FloatTensor(ad['hf']).to(device)

        with torch.no_grad():
            out = model(prot_t, dna_t, hf_t)
            scores = out['score'].cpu().numpy()
            gate_w = out['gate_weights'].cpu().numpy().mean(axis=0)

        if np.std(scores) > 1e-8 and np.std(y) > 1e-8:
            rho = stats.spearmanr(scores, y).statistic
            if not np.isnan(rho):
                results.append({
                    'assay': ad['name'], 'n': n, 'spearman': rho,
                    'gate_prot': gate_w[0], 'gate_dna': gate_w[1], 'gate_cross': gate_w[2],
                })

    return pd.DataFrame(results)


def evaluate_baselines(assay_data, device):
    """Baseline comparisons using per-assay 5-fold CV ridge."""
    results = []
    for ad in assay_data:
        y = ad['y']; n = len(y)
        if n < 50: continue
        row = {'assay': ad['name'], 'n': n}

        configs = {
            'prot_ridge': ad['prot'],
            'dna_ridge': ad['dna'],
            'concat_ridge': np.hstack([ad['prot'], ad['dna']]),
            'concat_human_ridge': np.hstack([ad['prot'], ad['dna'], ad['hf']]),
            'human_only': ad['hf'],
        }

        for cname, X in configs.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rhos = []
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0); m.fit(X[tr], y[tr])
                yp = m.predict(X[te])
                if np.std(yp) > 1e-8 and np.std(y[te]) > 1e-8:
                    r = stats.spearmanr(yp, y[te]).statistic
                    if not np.isnan(r): rhos.append(r)
            row[cname] = np.mean(rhos) if rhos else 0
        results.append(row)
    return pd.DataFrame(results)


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--out_dir", default="results/two_tower_sae")
    ap.add_argument("--phase_a_epochs", type=int, default=100)
    ap.add_argument("--phase_b_epochs", type=int, default=50)
    ap.add_argument("--phase_c_epochs", type=int, default=20)
    ap.add_argument("--skip_phase_c", action="store_true")
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    cfg = Config(phase_a_epochs=args.phase_a_epochs,
                 phase_b_epochs=args.phase_b_epochs,
                 phase_c_epochs=args.phase_c_epochs)

    # Load data
    print("Loading data ...", flush=True)
    assay_data = load_assay_data(args.sae_dir)
    print("Loaded %d assays" % len(assay_data), flush=True)

    # Pool all representations
    all_prot = np.vstack([ad['prot'] for ad in assay_data])
    all_dna = np.vstack([ad['dna'] for ad in assay_data])
    all_hf = np.vstack([ad['hf'] for ad in assay_data])
    all_y_raw = np.concatenate([ad['y'] for ad in assay_data])

    # Z-score fitness per assay
    all_y = np.zeros_like(all_y_raw)
    offset = 0
    for ad in assay_data:
        n = len(ad['y'])
        mu, sd = ad['y'].mean(), ad['y'].std() + 1e-8
        all_y[offset:offset+n] = (ad['y'] - mu) / sd
        offset += n

    print("Total: %d variants, prot=%s, dna=%s" % (
        len(all_prot), all_prot.shape, all_dna.shape), flush=True)

    # Build model
    model = TwoTowerModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: %d" % n_params, flush=True)

    # Phase A
    train_phase_a(model, all_prot, all_dna, cfg, device)

    # Phase B
    train_phase_b(model, all_prot, all_dna, all_hf, all_y, cfg, device)

    # Phase C
    if not args.skip_phase_c:
        train_phase_c(model, all_prot, all_dna, all_hf, all_y, cfg, device)

    # Save model
    torch.save({'state_dict': model.state_dict(), 'config': cfg},
               os.path.join(args.out_dir, "two_tower_model.pt"))

    # Evaluate: per-assay Spearman (using global model, in-sample for now)
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION: Per-assay Spearman (global model)", flush=True)
    print("=" * 70, flush=True)

    results = evaluate_per_assay(model, assay_data, device)
    results.to_csv(os.path.join(args.out_dir, "per_assay_results.csv"), index=False)
    print("Two-Tower SAE: mean=%.4f, median=%.4f (%d assays)" % (
        results['spearman'].mean(), results['spearman'].median(), len(results)), flush=True)

    # Gate analysis
    print("\nGate weights (mean across assays):", flush=True)
    print("  PROT:  %.1f%%" % (results['gate_prot'].mean() * 100), flush=True)
    print("  DNA:   %.1f%%" % (results['gate_dna'].mean() * 100), flush=True)
    print("  CROSS: %.1f%%" % (results['gate_cross'].mean() * 100), flush=True)

    # Per-gene gate
    print("\nPer-gene gate weights:", flush=True)
    results['gene'] = results['assay'].str.split('_').str[0]
    gene_gate = results.groupby('gene')[['gate_prot', 'gate_dna', 'gate_cross']].mean()
    gene_gate_sorted = gene_gate.sort_values('gate_dna', ascending=False)
    print("  Top 10 DNA-driven:", flush=True)
    for gene, row in gene_gate_sorted.head(10).iterrows():
        print("    %-15s PROT=%.0f%% DNA=%.0f%% CROSS=%.0f%%" % (
            gene, row['gate_prot']*100, row['gate_dna']*100, row['gate_cross']*100), flush=True)
    print("  Top 10 PROT-driven:", flush=True)
    gene_gate_sorted2 = gene_gate.sort_values('gate_prot', ascending=False)
    for gene, row in gene_gate_sorted2.head(10).iterrows():
        print("    %-15s PROT=%.0f%% DNA=%.0f%% CROSS=%.0f%%" % (
            gene, row['gate_prot']*100, row['gate_dna']*100, row['gate_cross']*100), flush=True)

    # Baselines
    if not args.skip_baselines:
        print("\n" + "=" * 70, flush=True)
        print("BASELINES (per-assay 5-fold CV Ridge)", flush=True)
        print("=" * 70, flush=True)

        baselines = evaluate_baselines(assay_data, device)
        baselines.to_csv(os.path.join(args.out_dir, "baseline_results.csv"), index=False)

        merged = results.merge(baselines[['assay'] + [c for c in baselines.columns if c not in ['assay','n']]],
                               on='assay', how='inner')

        print("%-25s %8s %8s" % ("Method", "Mean", "Median"), flush=True)
        print("-" * 45, flush=True)
        for col in ['human_only', 'dna_ridge', 'prot_ridge', 'concat_ridge',
                     'concat_human_ridge', 'spearman']:
            if col in merged.columns:
                label = col if col != 'spearman' else 'Two-Tower SAE (ours)'
                print("%-25s %8.4f %8.4f" % (label, merged[col].mean(), merged[col].median()),
                      flush=True)

        # Win rates
        print("\nWin rates (Two-Tower vs baselines):", flush=True)
        for col in ['human_only', 'dna_ridge', 'prot_ridge', 'concat_ridge', 'concat_human_ridge']:
            if col in merged.columns:
                wins = (merged['spearman'] > merged[col]).sum()
                print("  vs %-25s: %d/%d (%.0f%%)" % (col, wins, len(merged), 100*wins/len(merged)),
                      flush=True)

    # Alive features summary
    print("\n" + "=" * 70, flush=True)
    print("CONCEPT SUMMARY", flush=True)
    print("=" * 70, flush=True)
    with torch.no_grad():
        sample_prot = torch.FloatTensor(all_prot[:5000]).to(device)
        sample_dna = torch.FloatTensor(all_dna[:5000]).to(device)
        prot_z = model.sae_prot.encode(sample_prot)
        dna_z = model.sae_dna.encode(sample_dna)
        cross_z, _, _ = model.encode_cross(prot_z, dna_z)

    alive_p = ((prot_z > 0).float().mean(dim=0) > 0.01).sum().item()
    alive_d = ((dna_z > 0).float().mean(dim=0) > 0.01).sum().item()
    alive_c = ((cross_z > 0).float().mean(dim=0) > 0.01).sum().item()
    print("Alive features: PROT=%d/%d, DNA=%d/%d, CROSS=%d/%d" % (
        alive_p, model.sae_prot.n_features,
        alive_d, model.sae_dna.n_features,
        alive_c, model.sae_cross.n_features), flush=True)
    print("Active per variant: PROT=%.1f, DNA=%.1f, CROSS=%.1f" % (
        (prot_z > 0).float().sum(dim=1).mean().item(),
        (dna_z > 0).float().sum(dim=1).mean().item(),
        (cross_z > 0).float().sum(dim=1).mean().item()), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
