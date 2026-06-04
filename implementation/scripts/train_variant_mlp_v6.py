#!/usr/bin/env python3
"""
Variant MLP v6: Pretrain + AA substitution features + self-training.

Builds on v5 (pretrain-then-finetune) with:
1. Amino acid substitution features (~15-d): BLOSUM62, Grantham distance,
   physicochemical property changes (hydrophobicity, charge, volume, etc.)
2. Self-training: after initial finetune, pseudo-label high-confidence VUS
   from model predictions, retrain with expanded set
3. 20 seeds for a strong ensemble
"""

import argparse, glob, gzip, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Amino acid features ──────────────────────────────────────────────

BLOSUM62 = {
    'A': {'A': 4,'R':-1,'N':-2,'D':-2,'C': 0,'Q':-1,'E':-1,'G': 0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 0,'W':-3,'Y':-2,'V': 0},
    'R': {'A':-1,'R': 5,'N': 0,'D':-2,'C':-3,'Q': 1,'E': 0,'G':-2,'H': 0,'I':-3,'L':-2,'K': 2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
    'N': {'A':-2,'R': 0,'N': 6,'D': 1,'C':-3,'Q': 0,'E': 0,'G': 0,'H': 1,'I':-3,'L':-3,'K': 0,'M':-2,'F':-3,'P':-2,'S': 1,'T': 0,'W':-4,'Y':-2,'V':-3},
    'D': {'A':-2,'R':-2,'N': 1,'D': 6,'C':-3,'Q': 0,'E': 2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S': 0,'T':-1,'W':-4,'Y':-3,'V':-3},
    'C': {'A': 0,'R':-3,'N':-3,'D':-3,'C': 9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
    'Q': {'A':-1,'R': 1,'N': 0,'D': 0,'C':-3,'Q': 5,'E': 2,'G':-2,'H': 0,'I':-3,'L':-2,'K': 1,'M': 0,'F':-3,'P':-1,'S': 0,'T':-1,'W':-2,'Y':-1,'V':-2},
    'E': {'A':-1,'R': 0,'N': 0,'D': 2,'C':-4,'Q': 2,'E': 5,'G':-2,'H': 0,'I':-3,'L':-3,'K': 1,'M':-2,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'G': {'A': 0,'R':-2,'N': 0,'D':-1,'C':-3,'Q':-2,'E':-2,'G': 6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S': 0,'T':-2,'W':-2,'Y':-3,'V':-3},
    'H': {'A':-2,'R': 0,'N': 1,'D':-1,'C':-3,'Q': 0,'E': 0,'G':-2,'H': 8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y': 2,'V':-3},
    'I': {'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I': 4,'L': 2,'K':-3,'M': 1,'F': 0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V': 3},
    'L': {'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I': 2,'L': 4,'K':-2,'M': 2,'F': 0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V': 1},
    'K': {'A':-1,'R': 2,'N': 0,'D':-1,'C':-3,'Q': 1,'E': 1,'G':-2,'H':-1,'I':-3,'L':-2,'K': 5,'M':-1,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'M': {'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q': 0,'E':-2,'G':-3,'H':-2,'I': 1,'L': 2,'K':-1,'M': 5,'F': 0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V': 1},
    'F': {'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I': 0,'L': 0,'K':-3,'M': 0,'F': 6,'P':-4,'S':-2,'T':-2,'W': 1,'Y': 3,'V':-1},
    'P': {'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P': 7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
    'S': {'A': 1,'R':-1,'N': 1,'D': 0,'C':-1,'Q': 0,'E': 0,'G': 0,'H':-1,'I':-2,'L':-2,'K': 0,'M':-1,'F':-2,'P':-1,'S': 4,'T': 1,'W':-3,'Y':-2,'V':-2},
    'T': {'A': 0,'R':-1,'N': 0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 5,'W':-2,'Y':-2,'V': 0},
    'W': {'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F': 1,'P':-4,'S':-3,'T':-2,'W':11,'Y': 2,'V':-3},
    'Y': {'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H': 2,'I':-1,'L':-1,'K':-2,'M':-1,'F': 3,'P':-3,'S':-2,'T':-2,'W': 2,'Y': 7,'V':-1},
    'V': {'A': 0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I': 3,'L': 1,'K':-2,'M': 1,'F':-1,'P':-2,'S':-2,'T': 0,'W':-3,'Y':-1,'V': 4},
}

GRANTHAM = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,('A','E'):107,('A','G'):60,('A','H'):86,('A','I'):94,
    ('A','L'):96,('A','K'):106,('A','M'):84,('A','F'):113,('A','P'):27,('A','S'):99,('A','T'):58,('A','W'):148,('A','Y'):112,('A','V'):64,
    ('R','N'):86,('R','D'):96,('R','C'):180,('R','Q'):43,('R','E'):54,('R','G'):125,('R','H'):29,('R','I'):97,('R','L'):102,
    ('R','K'):26,('R','M'):91,('R','F'):97,('R','P'):103,('R','S'):110,('R','T'):71,('R','W'):101,('R','Y'):77,('R','V'):96,
    ('N','D'):23,('N','C'):139,('N','Q'):46,('N','E'):42,('N','G'):80,('N','H'):68,('N','I'):149,('N','L'):153,
    ('N','K'):94,('N','M'):142,('N','F'):158,('N','P'):91,('N','S'):46,('N','T'):65,('N','W'):174,('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','Q'):61,('D','E'):45,('D','G'):94,('D','H'):81,('D','I'):168,('D','L'):172,
    ('D','K'):101,('D','M'):160,('D','F'):177,('D','P'):108,('D','S'):65,('D','T'):85,('D','W'):181,('D','Y'):160,('D','V'):152,
    ('C','Q'):154,('C','E'):170,('C','G'):159,('C','H'):174,('C','I'):198,('C','L'):198,
    ('C','K'):202,('C','M'):196,('C','F'):205,('C','P'):169,('C','S'):112,('C','T'):149,('C','W'):215,('C','Y'):194,('C','V'):192,
    ('Q','E'):29,('Q','G'):87,('Q','H'):24,('Q','I'):109,('Q','L'):113,
    ('Q','K'):53,('Q','M'):101,('Q','F'):116,('Q','P'):76,('Q','S'):68,('Q','T'):42,('Q','W'):130,('Q','Y'):99,('Q','V'):96,
    ('E','G'):98,('E','H'):40,('E','I'):134,('E','L'):138,
    ('E','K'):56,('E','M'):126,('E','F'):140,('E','P'):93,('E','S'):80,('E','T'):65,('E','W'):152,('E','Y'):122,('E','V'):121,
    ('G','H'):98,('G','I'):135,('G','L'):138,
    ('G','K'):127,('G','M'):127,('G','F'):153,('G','P'):42,('G','S'):56,('G','T'):59,('G','W'):184,('G','Y'):147,('G','V'):109,
    ('H','I'):94,('H','L'):99,
    ('H','K'):32,('H','M'):87,('H','F'):100,('H','P'):77,('H','S'):89,('H','T'):47,('H','W'):115,('H','Y'):83,('H','V'):84,
    ('I','L'):5,
    ('I','K'):102,('I','M'):10,('I','F'):21,('I','P'):95,('I','S'):142,('I','T'):89,('I','W'):61,('I','Y'):33,('I','V'):29,
    ('L','K'):107,('L','M'):15,('L','F'):22,('L','P'):98,('L','S'):145,('L','T'):92,('L','W'):61,('L','Y'):36,('L','V'):32,
    ('K','M'):95,('K','F'):102,('K','P'):103,('K','S'):121,('K','T'):78,('K','W'):110,('K','Y'):85,('K','V'):97,
    ('M','F'):28,('M','P'):87,('M','S'):135,('M','T'):81,('M','W'):67,('M','Y'):36,('M','V'):21,
    ('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40,('F','Y'):22,('F','V'):50,
    ('P','S'):74,('P','T'):38,('P','W'):147,('P','Y'):110,('P','V'):68,
    ('S','T'):58,('S','W'):177,('S','Y'):144,('S','V'):124,
    ('T','W'):128,('T','Y'):92,('T','V'):69,
    ('W','Y'):37,('W','V'):88,
    ('Y','V'):55,
}

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

CHARGE = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}
POLAR = set('STNQYHKRDE')
AROMATIC = set('FWY')
SMALL = set('AGSTPV')
PROLINE = 'P'
GLYCINE = 'G'
CYSTEINE = 'C'


def get_grantham(a, b):
    if a == b:
        return 0
    return GRANTHAM.get((a, b), GRANTHAM.get((b, a), 100))


def compute_aa_features(from_aa, to_aa):
    """Compute amino acid substitution features for arrays of from/to AA."""
    n = len(from_aa)
    feats = np.zeros((n, 15), dtype=np.float32)

    for i in range(n):
        fa, ta = from_aa[i], to_aa[i]
        if fa not in HYDROPHOBICITY or ta not in HYDROPHOBICITY:
            continue

        feats[i, 0] = BLOSUM62.get(fa, {}).get(ta, -4)
        feats[i, 1] = get_grantham(fa, ta) / 215.0
        feats[i, 2] = HYDROPHOBICITY[ta] - HYDROPHOBICITY[fa]
        feats[i, 3] = abs(feats[i, 2])
        feats[i, 4] = (VOLUME[ta] - VOLUME[fa]) / 100.0
        feats[i, 5] = abs(feats[i, 4])
        feats[i, 6] = CHARGE.get(ta, 0) - CHARGE.get(fa, 0)
        feats[i, 7] = float(abs(feats[i, 6]) > 0.5)
        feats[i, 8] = float((fa in POLAR) != (ta in POLAR))
        feats[i, 9] = float((fa in AROMATIC) != (ta in AROMATIC))
        feats[i, 10] = float(ta == PROLINE and fa != PROLINE)
        feats[i, 11] = float(fa == PROLINE and ta != PROLINE)
        feats[i, 12] = float(fa == GLYCINE and ta != GLYCINE)
        feats[i, 13] = float(fa == CYSTEINE or ta == CYSTEINE)
        feats[i, 14] = float((fa in SMALL) != (ta in SMALL))

    return feats


# ── Model ─────────────────────────────────────────────────────────────

class VariantMLPv6(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_extra=23,
                 d_proj=768, d_hidden=512, dropout=0.2):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        d_in = d_proj * 2 + d_extra
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, x_prot, x_dna, x_extra):
        hp = self.proj_prot(x_prot)
        hd = self.proj_dna(x_dna)
        return self.head(torch.cat([hp, hd, x_extra], dim=-1)).squeeze(-1)

    def encode(self, x_prot, x_dna):
        return torch.cat([self.proj_prot(x_prot), self.proj_dna(x_dna)], dim=-1)


class PretrainHead(nn.Module):
    def __init__(self, d_in=1536, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ── Training helpers ──────────────────────────────────────────────────

def cosine_lr(opt, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def pretrain(proj_prot, proj_dna, pt_head, Xp, Xd, y_llr,
             device, epochs=30, batch=4096, lr=1e-3):
    params = list(proj_prot.parameters()) + list(proj_dna.parameters()) + list(pt_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    yt = torch.tensor(y_llr, dtype=torch.float32)
    n = len(y_llr)
    spe = max(n // batch, 1)
    total_steps = epochs * spe
    warmup_steps = 2 * spe

    for ep in range(epochs):
        proj_prot.train(); proj_dna.train(); pt_head.train()
        perm = torch.randperm(n)
        ep_loss = 0; nb = 0
        for b in range(0, n, batch):
            idx = perm[b:b + batch].numpy()
            xp = torch.tensor(Xp[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(Xd[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)
            step = ep * spe + nb
            cosine_lr(opt, step, warmup_steps, total_steps, lr)
            h = torch.cat([proj_prot(xp), proj_dna(xd)], dim=-1)
            loss = loss_fn(pt_head(h), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"    pretrain ep {ep+1}: loss={ep_loss/nb:.6f}", flush=True)


def finetune_fold(model, Xp_tr, Xd_tr, Xe_tr, y_tr,
                  Xp_te, Xd_te, Xe_te, y_te,
                  device, epochs=100, batch=2048, lr=3e-4,
                  proj_lr_scale=0.1):
    proj_params = list(model.proj_prot.parameters()) + list(model.proj_dna.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": proj_params, "lr": lr * proj_lr_scale},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_tr, dtype=torch.float32)

    Xp_v = torch.tensor(Xp_te, dtype=torch.float32, device=device)
    Xd_v = torch.tensor(Xd_te, dtype=torch.float32, device=device)
    Xe_v = torch.tensor(Xe_te, dtype=torch.float32, device=device)

    n = len(y_tr)
    spe = max(n // batch, 1)
    total_steps = epochs * spe
    warmup_steps = 5 * spe

    best_auc = 0; best_state = None; wait = 0; step = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch].numpy()
            xp = torch.tensor(Xp_tr[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(Xd_tr[idx], dtype=torch.float32, device=device)
            xe = torch.tensor(Xe_tr[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)

            for pg in opt.param_groups:
                base = pg.get("initial_lr", pg["lr"])
                if step == 0:
                    pg["initial_lr"] = pg["lr"]
                    base = pg["lr"]
                if step < warmup_steps:
                    pg["lr"] = base * step / max(warmup_steps, 1)
                else:
                    pg["lr"] = base * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))

            opt.zero_grad()
            loss = loss_fn(model(xp, xd, xe), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            vauc = roc_auc_score(y_te, torch.sigmoid(model(Xp_v, Xd_v, Xe_v)).cpu().numpy())
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(Xp_v, Xd_v, Xe_v)).cpu().numpy()
    return preds, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--constraint", default="data/full/labels/gnomad_constraint.txt.bgz")
    ap.add_argument("--out_dir", default="results/variant_mlp_v6")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--self_train_rounds", type=int, default=2)
    ap.add_argument("--pseudo_threshold", type=float, default=0.15,
                    help="pseudo-label VUS with pred < thresh (benign) or > 1-thresh (path)")
    ap.add_argument("--base_seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    print("Loading data ...", flush=True)
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx)
    sub = df.iloc[dual_idx].reset_index(drop=True)
    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    n = len(dual_idx)

    prot = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l:
                prot[g2l[int(gi)]] = row

    dna = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l:
                dna[g2l[int(gi)]] = row

    llr_all = np.load(args.llr_path)["llr"].astype(np.float32)
    print(f"  total: {n}, prot: {prot.shape}, dna: {dna.shape}", flush=True)

    # z-score normalize
    prot_mean, prot_std = prot.mean(0), prot.std(0) + 1e-8
    dna_mean, dna_std = dna.mean(0), dna.std(0) + 1e-8
    prot_z = (prot - prot_mean) / prot_std
    dna_z = (dna - dna_mean) / dna_std
    llr_mean, llr_std = llr_all.mean(), llr_all.std() + 1e-8
    llr_z = (llr_all - llr_mean) / llr_std

    # AA substitution features (all 260k)
    print("Computing AA substitution features ...", flush=True)
    aa_feats = compute_aa_features(sub["from_aa"].values, sub["to_aa"].values)
    print(f"  AA features: {aa_feats.shape} (mean abs: {np.abs(aa_feats).mean(0).round(3).tolist()})", flush=True)

    # Gene-level constraint scores from gnomAD
    print("Loading gene constraint scores ...", flush=True)
    gc_map = {}
    if os.path.exists(args.constraint):
        with gzip.open(args.constraint, "rt") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                vals = line.strip().split("\t")
                rec = dict(zip(header, vals))
                g = rec["gene"]
                try:
                    gc_map[g] = {
                        "pLI": float(rec["pLI"]) if rec["pLI"] not in ("NA", "") else np.nan,
                        "oe_mis": float(rec["oe_mis"]) if rec["oe_mis"] not in ("NA", "") else np.nan,
                        "mis_z": float(rec["mis_z"]) if rec["mis_z"] not in ("NA", "") else np.nan,
                    }
                except (ValueError, KeyError):
                    pass
    gene_names = sub["gene"].values
    pli_arr = np.array([gc_map.get(g, {}).get("pLI", np.nan) for g in gene_names], dtype=np.float32)
    oe_mis_arr = np.array([gc_map.get(g, {}).get("oe_mis", np.nan) for g in gene_names], dtype=np.float32)
    mis_z_arr = np.array([gc_map.get(g, {}).get("mis_z", np.nan) for g in gene_names], dtype=np.float32)
    # Fill NaN with median
    for arr in [pli_arr, oe_mis_arr, mis_z_arr]:
        med = np.nanmedian(arr)
        arr[np.isnan(arr)] = med
    gene_feats = np.column_stack([pli_arr, oe_mis_arr, mis_z_arr])
    matched = sum(1 for g in gene_names if g in gc_map)
    print(f"  Gene constraint: {len(gc_map)} genes in DB, {matched}/{n} variants matched ({matched/n*100:.1f}%)", flush=True)

    # Extra features: LLR + norms + AA features + gene constraint
    prot_norm = np.linalg.norm(prot, axis=1, keepdims=True)
    dna_norm = np.linalg.norm(dna, axis=1, keepdims=True)
    prot_max = np.abs(prot).max(axis=1, keepdims=True)
    dna_max = np.abs(dna).max(axis=1, keepdims=True)
    prot_sd = prot.std(axis=1, keepdims=True)
    dna_sd = dna.std(axis=1, keepdims=True)
    norm_ratio = prot_norm / (dna_norm + 1e-8)

    extra = np.concatenate([
        prot_norm, dna_norm, llr_all[:, None],
        prot_max, dna_max, prot_sd, dna_sd, norm_ratio,
        aa_feats,
        gene_feats,
    ], axis=1).astype(np.float32)
    d_extra = extra.shape[1]
    print(f"  Extra features: {d_extra}-d (8 edelta + {aa_feats.shape[1]} AA + 3 gene constraint)", flush=True)

    # Labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled_mask = (is_path | is_ben).values
    y_full = np.where(is_path.values, 1.0, 0.0)
    y = y_full[labeled_mask]
    genes = sub["gene"].values
    genes_lab = genes[labeled_mask]

    prot_lab = prot_z[labeled_mask]; dna_lab = dna_z[labeled_mask]; extra_lab = extra[labeled_mask]
    n_lab = labeled_mask.sum()
    print(f"  Labeled: {n_lab} (P:{(y==1).sum()}, B:{(y==0).sum()})", flush=True)

    # Unlabeled (VUS) indices for self-training
    vus_mask = ~labeled_mask
    n_vus = vus_mask.sum()
    print(f"  VUS (unlabeled): {n_vus}", flush=True)

    # ── Multi-seed training ──
    all_seed_aucs = []
    all_oof_preds = np.zeros(n_lab, dtype=np.float64)
    seed_oof = []

    for si in range(args.n_seeds):
        seed = args.base_seed + si
        print(f"\n{'='*60}\nSeed {si} (random_state={seed})\n{'='*60}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Step 1: Pretrain on ALL 260k
        print("  [1] Pretraining on 260k variants ...", flush=True)
        model = VariantMLPv6(d_extra=d_extra).to(device)
        pt_head = PretrainHead().to(device)
        pretrain(model.proj_prot, model.proj_dna, pt_head, prot_z, dna_z, llr_z,
                 device, epochs=args.pretrain_epochs)
        del pt_head
        pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                           if "proj_" in k}

        # Step 2: 5-fold CV fine-tuning
        print("  [2] Fine-tuning (5-fold CV) ...", flush=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(n_lab, dtype=np.float32)

        for fi, (tr, te) in enumerate(skf.split(prot_lab, y)):
            # Per-fold normalization for extra features
            em, es = extra_lab[tr].mean(0), extra_lab[tr].std(0) + 1e-8

            model_fold = VariantMLPv6(d_extra=d_extra).to(device)
            # Load pretrained projections
            sd = model_fold.state_dict()
            for k, v in pretrained_state.items():
                if k in sd:
                    sd[k] = v.clone()
            model_fold.load_state_dict(sd)

            preds, bauc = finetune_fold(
                model_fold,
                prot_lab[tr], dna_lab[tr], (extra_lab[tr] - em) / es, y[tr],
                prot_lab[te], dna_lab[te], (extra_lab[te] - em) / es, y[te],
                device,
            )
            oof[te] = preds
            print(f"    fold {fi}: val_auc={bauc:.4f}", flush=True)

        seed_auc = roc_auc_score(y, oof)
        all_seed_aucs.append(seed_auc)
        all_oof_preds += oof
        seed_oof.append(oof.copy())
        print(f"  seed {si} genome-wide AUC = {seed_auc:.4f}", flush=True)

        # Step 3: Self-training rounds
        if args.self_train_rounds > 0 and si == 0:
            print(f"\n  [3] Self-training ({args.self_train_rounds} rounds) ...", flush=True)

            # Train a full model on all labeled data to predict VUS
            model_full = VariantMLPv6(d_extra=d_extra).to(device)
            sd = model_full.state_dict()
            for k, v in pretrained_state.items():
                if k in sd:
                    sd[k] = v.clone()
            model_full.load_state_dict(sd)

            # Normalize extra features using labeled stats
            em_full, es_full = extra_lab.mean(0), extra_lab.std(0) + 1e-8
            extra_lab_n = (extra_lab - em_full) / es_full

            # Quick train on all labeled
            proj_params = list(model_full.proj_prot.parameters()) + list(model_full.proj_dna.parameters())
            head_params = list(model_full.head.parameters())
            opt = torch.optim.AdamW([
                {"params": proj_params, "lr": 3e-5},
                {"params": head_params, "lr": 3e-4},
            ], weight_decay=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()

            Xp_all_lab = torch.tensor(prot_lab, dtype=torch.float32, device=device)
            Xd_all_lab = torch.tensor(dna_lab, dtype=torch.float32, device=device)
            Xe_all_lab = torch.tensor(extra_lab_n, dtype=torch.float32, device=device)
            y_lab_t = torch.tensor(y, dtype=torch.float32, device=device)

            model_full.train()
            for ep in range(50):
                perm = torch.randperm(n_lab)
                for b in range(0, n_lab, 2048):
                    idx = perm[b:b+2048]
                    opt.zero_grad()
                    loss = loss_fn(model_full(Xp_all_lab[idx], Xd_all_lab[idx], Xe_all_lab[idx]), y_lab_t[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
                    opt.step()

            # Predict VUS (normalize extra with same stats)
            model_full.eval()
            vus_indices = np.where(vus_mask)[0]
            extra_vus_n = (extra[vus_mask] - em_full) / es_full
            with torch.no_grad():
                vus_preds = torch.sigmoid(model_full(
                    torch.tensor(prot_z[vus_mask], dtype=torch.float32, device=device),
                    torch.tensor(dna_z[vus_mask], dtype=torch.float32, device=device),
                    torch.tensor(extra_vus_n, dtype=torch.float32, device=device),
                )).cpu().numpy()

            # Select high-confidence pseudo-labels
            thresh = args.pseudo_threshold
            pseudo_path = vus_preds > (1 - thresh)
            pseudo_ben = vus_preds < thresh
            n_pseudo_p = pseudo_path.sum()
            n_pseudo_b = pseudo_ben.sum()
            print(f"    VUS pseudo-labels: {n_pseudo_p} path (>{1-thresh:.2f}), {n_pseudo_b} benign (<{thresh:.2f})", flush=True)

            if n_pseudo_p + n_pseudo_b > 100:
                # Merge pseudo-labeled with original labeled
                pseudo_idx = vus_indices[pseudo_path | pseudo_ben]
                pseudo_y = np.where(vus_preds[pseudo_path | pseudo_ben] > 0.5, 1.0, 0.0)

                prot_aug = np.concatenate([prot_lab, prot_z[pseudo_idx]])
                dna_aug = np.concatenate([dna_lab, dna_z[pseudo_idx]])
                extra_aug = np.concatenate([extra_lab, extra[pseudo_idx]])
                y_aug = np.concatenate([y, pseudo_y])
                genes_aug = np.concatenate([genes_lab, genes[pseudo_idx]])

                print(f"    Augmented training set: {len(y_aug)} ({len(y)} original + {len(pseudo_y)} pseudo)", flush=True)

                # Retrain with augmented data (only use original labeled for eval)
                for st_round in range(args.self_train_rounds):
                    print(f"    Self-train round {st_round+1}/{args.self_train_rounds}", flush=True)
                    np.random.seed(seed + 1000 + st_round)
                    torch.manual_seed(seed + 1000 + st_round)

                    model_st = VariantMLPv6(d_extra=d_extra).to(device)
                    pt_head_st = PretrainHead().to(device)
                    pretrain(model_st.proj_prot, model_st.proj_dna, pt_head_st,
                             prot_z, dna_z, llr_z, device, epochs=args.pretrain_epochs)
                    del pt_head_st
                    pt_state_st = {k: v.cpu().clone() for k, v in model_st.state_dict().items()
                                   if "proj_" in k}

                    oof_st = np.zeros(n_lab, dtype=np.float32)
                    skf_st = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1000 + st_round)
                    for fi, (tr, te) in enumerate(skf_st.split(prot_lab, y)):
                        # Augment training fold with pseudo-labeled
                        Xp_tr_aug = np.concatenate([prot_lab[tr], prot_z[pseudo_idx]])
                        Xd_tr_aug = np.concatenate([dna_lab[tr], dna_z[pseudo_idx]])
                        Xe_tr_raw = np.concatenate([extra_lab[tr], extra[pseudo_idx]])
                        y_tr_aug = np.concatenate([y[tr], pseudo_y])

                        # Per-fold normalization (fit on augmented train)
                        em_st, es_st = Xe_tr_raw.mean(0), Xe_tr_raw.std(0) + 1e-8

                        model_f = VariantMLPv6(d_extra=d_extra).to(device)
                        sd = model_f.state_dict()
                        for k, v in pt_state_st.items():
                            if k in sd:
                                sd[k] = v.clone()
                        model_f.load_state_dict(sd)

                        preds, bauc = finetune_fold(
                            model_f,
                            Xp_tr_aug, Xd_tr_aug, (Xe_tr_raw - em_st) / es_st, y_tr_aug,
                            prot_lab[te], dna_lab[te], (extra_lab[te] - em_st) / es_st, y[te],
                            device,
                        )
                        oof_st[te] = preds
                        print(f"      fold {fi}: val_auc={bauc:.4f}", flush=True)

                    st_auc = roc_auc_score(y, oof_st)
                    print(f"    self-train round {st_round+1} AUC = {st_auc:.4f}", flush=True)
                    all_seed_aucs.append(st_auc)
                    all_oof_preds += oof_st
                    seed_oof.append(oof_st.copy())

    # ── Results ──
    n_total_models = len(all_seed_aucs)
    all_oof_preds /= n_total_models

    print(f"\n{'='*60}\n=== RESULTS ===\n{'='*60}", flush=True)
    for i, a in enumerate(all_seed_aucs):
        tag = "self-train" if i >= args.n_seeds else f"seed {i}"
        print(f"  {tag}: {a:.4f}", flush=True)
    print(f"  mean single:    {np.mean(all_seed_aucs):.4f} ± {np.std(all_seed_aucs):.4f}", flush=True)

    # Progressive ensemble
    ranked = np.argsort(all_seed_aucs)[::-1]
    for k in [2, 3, 5, 10, min(15, n_total_models), n_total_models]:
        if k > n_total_models:
            continue
        ens = np.mean([seed_oof[ranked[i]] for i in range(k)], axis=0)
        auc = roc_auc_score(y, ens)
        print(f"  ensemble({k}): {auc:.4f}", flush=True)

    # Final ensemble (all)
    auc_all = roc_auc_score(y, all_oof_preds)
    ci = []
    for _ in range(1000):
        idx = np.random.choice(n_lab, n_lab, replace=True)
        ci.append(roc_auc_score(y[idx], all_oof_preds[idx]))
    ci = np.percentile(ci, [2.5, 97.5])
    print(f"\n  FINAL ensemble ({n_total_models}): {auc_all:.4f} 95%CI [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
    print(f"  (AlphaMissense:       0.9638)", flush=True)
    print(f"  (REVEL:               0.9689)", flush=True)

    # Per-gene AUC
    gene_aucs = []
    for g in np.unique(genes_lab):
        mask = genes_lab == g
        yg = y[mask]
        if (yg == 1).sum() >= 20 and (yg == 0).sum() >= 20:
            gene_aucs.append({"gene": g, "auc": roc_auc_score(yg, all_oof_preds[mask])})
    gdf = pd.DataFrame(gene_aucs).sort_values("auc", ascending=False)
    gdf.to_csv(os.path.join(args.out_dir, "per_gene_auc.csv"), index=False)
    print(f"  per-gene: mean={gdf['auc'].mean():.3f} median={gdf['auc'].median():.3f} (n={len(gdf)})", flush=True)

    # Save
    np.savez_compressed(os.path.join(args.out_dir, "preds.npz"),
                        ensemble=all_oof_preds, y=y, genes=genes_lab)
    results = {
        "seed_aucs": all_seed_aucs,
        "mean_single": float(np.mean(all_seed_aucs)),
        "ensemble_auc": float(auc_all),
        "ci_95": [float(ci[0]), float(ci[1])],
        "n_seeds": args.n_seeds,
        "self_train_rounds": args.self_train_rounds,
        "aa_features": True,
        "d_extra": d_extra,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
