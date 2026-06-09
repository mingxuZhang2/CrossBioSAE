"""
CrossCoder SAE for Variant Effect Decomposition.

Single shared sparse dictionary with modality-specific decoders.
Each feature is naturally protein-private, DNA-private, or shared
based on decoder norms — no auxiliary loss needed.

Architecture:
  ESM-2 edelta (1280) → standardize → PCA(768) → prot_repr
  Evo-2 edelta (4096) → standardize → PCA(512, whiten) → dna_repr
  CrossCoder: encoder(concat) → TopK sparse z
              decoder_prot(z) → reconstruct prot_repr
              decoder_dna(z)  → reconstruct dna_repr

Usage:
  python crosscoder_sae.py --phase all
  python crosscoder_sae.py --phase evaluate --checkpoint results/.../crosscoder.pt
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
from sklearn.decomposition import PCA


# ── Config ──

@dataclass
class CrossCoderConfig:
    d_prot_raw: int = 1280
    d_dna_raw: int = 4096
    d_prot: int = 768
    d_dna: int = 512
    whiten_dna: bool = True
    n_features: int = 4096
    k: int = 32
    n_human: int = 9
    epochs: int = 300
    lr: float = 5e-4
    batch: int = 8192
    warmup: int = 30


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

class CrossCoderSAE(nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_prot + cfg.d_dna
        self.encoder = nn.Linear(d_in, cfg.n_features)
        self.decoder_prot = nn.Linear(cfg.n_features, cfg.d_prot, bias=False)
        self.decoder_dna = nn.Linear(cfg.n_features, cfg.d_dna, bias=False)
        self.k = cfg.k

    def encode(self, prot, dna):
        x = torch.cat([prot, dna], dim=1)
        z = F.relu(self.encoder(x))
        if self.k < self.cfg.n_features:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            mask = torch.zeros_like(z)
            mask.scatter_(1, topk_idx, 1.0)
            z = z * mask
        return z

    def forward(self, prot, dna):
        z = self.encode(prot, dna)
        prot_hat = self.decoder_prot(z)
        dna_hat = self.decoder_dna(z)
        return prot_hat, dna_hat, z

    def normalize_decoder(self):
        with torch.no_grad():
            full = torch.cat([self.decoder_prot.weight.data,
                              self.decoder_dna.weight.data], dim=0)
            full = F.normalize(full, dim=0)
            dp = self.cfg.d_prot
            self.decoder_prot.weight.data = full[:dp]
            self.decoder_dna.weight.data = full[dp:]

    def get_feature_modality(self):
        """Return per-feature decoder norms for modality classification."""
        with torch.no_grad():
            pn = self.decoder_prot.weight.data.norm(dim=0).cpu().numpy()
            dn = self.decoder_dna.weight.data.norm(dim=0).cpu().numpy()
        return pn, dn

    def classify_features(self, threshold=0.7):
        """Classify features as protein-private, dna-private, shared, or mixed."""
        pn, dn = self.get_feature_modality()
        total = pn + dn + 1e-8
        prot_frac = pn / total
        categories = np.empty(len(pn), dtype='<U12')
        categories[prot_frac > threshold] = 'prot-private'
        categories[prot_frac < (1 - threshold)] = 'dna-private'
        categories[(prot_frac >= (1 - threshold)) & (prot_frac <= threshold)] = 'shared'
        alive_mask = (pn + dn) > 0.01
        categories[~alive_mask] = 'dead'
        return categories, pn, dn


# ── Data ──

def load_raw_embeddings(emb_dir):
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
        n = len(muts)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            r, p, a = parse_mutant(muts[i])
            if r and a:
                hf[i] = compute_human_features(r, a)
        all_prot.append(prot)
        all_dna.append(dna)
        assay_data.append({'name': name, 'prot': prot, 'dna': dna,
                           'y': y, 'hf': hf, 'mutants': muts})

    all_prot = np.vstack(all_prot)
    all_dna = np.vstack(all_dna)
    print("Loaded %d DMS assays, %d variants (skipped %d)" % (
        len(assay_data), len(all_prot), skipped), flush=True)
    return all_prot, all_dna, assay_data


def load_clinvar_matched(esm2_dir, evo2_dir, clinvar_csv, clinvar_parquet):
    """Load ClinVar variants with both ESM-2 and Evo-2 embeddings."""
    import json as _json
    cv_csv = pd.read_csv(clinvar_csv)
    cv_par = pd.read_parquet(clinvar_parquet)

    cv_csv['key'] = (cv_csv['Chromosome'].astype(str) + ':' +
                     cv_csv['PositionVCF'].astype(str) + ':' +
                     cv_csv['ReferenceAlleleVCF'].astype(str) + ':' +
                     cv_csv['AlternateAlleleVCF'].astype(str))
    cv_par['key'] = (cv_par['chrom'].astype(str) + ':' +
                     cv_par['pos'].astype(str) + ':' +
                     cv_par['ref'].astype(str) + ':' +
                     cv_par['alt'].astype(str))
    par_key2idx = dict(zip(cv_par['key'], range(len(cv_par))))

    # Load Evo-2 embeddings into array indexed by parquet row
    n_par = len(cv_par)
    evo2_arr = np.zeros((n_par, 4096), dtype=np.float32)
    evo2_mask = np.zeros(n_par, dtype=bool)
    for f in sorted(glob.glob(os.path.join(evo2_dir, 'clinvar_evo2_emb_shard*.npz'))):
        d = np.load(f, allow_pickle=True)
        idxs = d['idx']
        valid = idxs < n_par
        evo2_arr[idxs[valid]] = d['edelta'][valid]
        evo2_mask[idxs[valid]] = True
    print("  Evo2: %d / %d parquet rows loaded" % (evo2_mask.sum(), n_par), flush=True)

    # Load ESM-2 embeddings
    esm2_parts = []
    for f in sorted(glob.glob(os.path.join(esm2_dir, 'shard_*.npz'))):
        d = np.load(f, allow_pickle=True)
        esm2_parts.append({
            'edelta': d['esm2_edelta'],
            'idx': d['idx'].astype(int),
            'gene': d['gene'].astype(str),
            'mutant': d['mutant'].astype(str),
            'label': d['label'].astype(int),
        })
    esm2_edelta = np.vstack([p['edelta'] for p in esm2_parts])
    esm2_idx = np.concatenate([p['idx'] for p in esm2_parts])
    esm2_gene = np.concatenate([p['gene'] for p in esm2_parts])
    esm2_mutant = np.concatenate([p['mutant'] for p in esm2_parts])
    esm2_label = np.concatenate([p['label'] for p in esm2_parts])
    print("  ESM2: %d variants loaded" % len(esm2_idx), flush=True)

    # Match: ESM2 csv_idx -> csv key -> parquet key -> parquet idx -> Evo2
    csv_keys = cv_csv['key'].values
    matched = []
    for i in range(len(esm2_idx)):
        ci = esm2_idx[i]
        if ci >= len(csv_keys):
            continue
        key = csv_keys[ci]
        if key not in par_key2idx:
            continue
        pidx = par_key2idx[key]
        if not evo2_mask[pidx]:
            continue
        matched.append((i, pidx))

    if not matched:
        print("WARNING: No matched ClinVar variants found", flush=True)
        return None, None, None

    esm2_sel = np.array([m[0] for m in matched])
    evo2_sel = np.array([m[1] for m in matched])
    prot_arr = esm2_edelta[esm2_sel]
    dna_arr = evo2_arr[evo2_sel]
    labels = esm2_label[esm2_sel]
    genes = esm2_gene[esm2_sel]
    mutants = esm2_mutant[esm2_sel]

    clinvar_data = {
        'prot': prot_arr, 'dna': dna_arr,
        'label': labels,
        'gene': genes,
        'mutant': mutants,
    }
    n_path = (labels == 1).sum()
    n_ben = (labels == 0).sum()
    print("Matched %d ClinVar variants (%d pathogenic, %d benign, %d genes)" % (
        len(prot_arr), n_path, n_ben, len(set(genes))), flush=True)
    return prot_arr, dna_arr, clinvar_data


def preprocess(all_prot, all_dna, cfg):
    """Standardize + asymmetric PCA (whiten DNA)."""
    prot_mean, prot_std = all_prot.mean(0), all_prot.std(0) + 1e-8
    dna_mean, dna_std = all_dna.mean(0), all_dna.std(0) + 1e-8
    prot_normed = (all_prot - prot_mean) / prot_std
    dna_normed = (all_dna - dna_mean) / dna_std

    pca_prot = PCA(n_components=cfg.d_prot, whiten=False, random_state=42)
    pca_dna = PCA(n_components=cfg.d_dna, whiten=cfg.whiten_dna, random_state=42)
    prot_pca = pca_prot.fit_transform(prot_normed).astype(np.float32)
    dna_pca = pca_dna.fit_transform(dna_normed).astype(np.float32)

    print("  PCA prot: %d→%d, var=%.1f%%, norm=%.2f" % (
        all_prot.shape[1], cfg.d_prot,
        pca_prot.explained_variance_ratio_.sum() * 100,
        np.linalg.norm(prot_pca, axis=1).mean()), flush=True)
    print("  PCA dna:  %d→%d, var=%.1f%%, whiten=%s, norm=%.2f" % (
        all_dna.shape[1], cfg.d_dna,
        pca_dna.explained_variance_ratio_.sum() * 100,
        cfg.whiten_dna,
        np.linalg.norm(dna_pca, axis=1).mean()), flush=True)

    prep = {
        'prot_mean': prot_mean, 'prot_std': prot_std,
        'dna_mean': dna_mean, 'dna_std': dna_std,
        'pca_prot_components': pca_prot.components_,
        'pca_prot_mean': pca_prot.mean_,
        'pca_dna_components': pca_dna.components_,
        'pca_dna_mean': pca_dna.mean_,
    }
    if cfg.whiten_dna:
        prep['pca_dna_var'] = pca_dna.explained_variance_

    return prot_pca, dna_pca, prep


def apply_prep(raw, mean, std, pca_components, pca_mean, pca_var=None):
    normed = (raw - mean) / std
    pca_out = (normed - pca_mean) @ pca_components.T
    if pca_var is not None:
        pca_out = pca_out / np.sqrt(pca_var + 1e-8)
    return pca_out.astype(np.float32)


def prep_assay(ad, prep, cfg):
    prot = apply_prep(ad['prot'], prep['prot_mean'], prep['prot_std'],
                      prep['pca_prot_components'], prep['pca_prot_mean'])
    dna = apply_prep(ad['dna'], prep['dna_mean'], prep['dna_std'],
                     prep['pca_dna_components'], prep['pca_dna_mean'],
                     prep.get('pca_dna_var') if cfg.whiten_dna else None)
    return prot, dna


# ── Training ──

def get_cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_crosscoder(model, prot_data, dna_data, cfg, device):
    print("=" * 70, flush=True)
    print("TRAINING CrossCoder (%d variants, %d features, k=%d)" % (
        len(prot_data), cfg.n_features, cfg.k), flush=True)
    print("  Input: prot(%d) + dna(%d) = %d" % (cfg.d_prot, cfg.d_dna, cfg.d_prot + cfg.d_dna), flush=True)
    print("  Epochs=%d, batch=%d, lr=%.1e" % (cfg.epochs, cfg.batch, cfg.lr), flush=True)
    print("=" * 70, flush=True)

    prot_t = torch.FloatTensor(prot_data).to(device)
    dna_t = torch.FloatTensor(dna_data).to(device)
    N = len(prot_t)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    steps_per_ep = N // cfg.batch + 1
    total_steps = cfg.epochs * steps_per_ep
    warmup_steps = cfg.warmup * steps_per_ep

    t0 = time.time()
    for ep in range(cfg.epochs):
        idx = torch.randperm(N, device=device)
        ep_loss_p = ep_loss_d = 0.0
        nb = 0
        for start in range(0, N, cfg.batch):
            step = ep * steps_per_ep + nb
            lr = get_cosine_lr(step, total_steps, warmup_steps, cfg.lr)
            for pg in opt.param_groups:
                pg['lr'] = lr

            bi = idx[start:start + cfg.batch]
            p_hat, d_hat, z = model(prot_t[bi], dna_t[bi])
            loss_p = F.mse_loss(p_hat, prot_t[bi])
            loss_d = F.mse_loss(d_hat, dna_t[bi])
            loss = loss_p + loss_d

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.normalize_decoder()

            ep_loss_p += loss_p.item()
            ep_loss_d += loss_d.item()
            nb += 1

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                s = min(5000, N)
                z_s = model.encode(prot_t[:s], dna_t[:s])
                alive = ((z_s > 0).float().mean(dim=0) > 0.01).sum().item()
                active_per = (z_s > 0).float().sum(dim=1).mean().item()

            cats, pn, dn = model.classify_features()
            n_pp = (cats == 'prot-private').sum()
            n_dp = (cats == 'dna-private').sum()
            n_sh = (cats == 'shared').sum()
            n_dead = (cats == 'dead').sum()

            elapsed = (time.time() - t0) / 60
            print("  Ep %d/%d (%.1fmin): prot=%.5f dna=%.5f | "
                  "alive=%d/%d active/sample=%.0f | "
                  "PP=%d DP=%d SH=%d dead=%d" % (
                      ep + 1, cfg.epochs, elapsed,
                      ep_loss_p / nb, ep_loss_d / nb,
                      int(alive), cfg.n_features, active_per,
                      n_pp, n_dp, n_sh, n_dead), flush=True)

    print("\nTraining done in %.1f min" % ((time.time() - t0) / 60), flush=True)


# ── Evaluation ──

def ridge_cv(X, y, alpha=1.0, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rhos = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        if np.std(yp) > 1e-8 and np.std(y[te]) > 1e-8:
            r = stats.spearmanr(yp, y[te]).statistic
            if not np.isnan(r):
                rhos.append(r)
    return np.mean(rhos) if rhos else 0.0


def bottleneck_ladder(assay_data, prep, cfg, model, device):
    """Diagnostic: where does information get lost?"""
    print("\n--- Bottleneck Ladder ---", flush=True)
    methods = ['raw_esm', 'pca_esm', 'pca_evo', 'concat_pca', 'crosscoder_z', 'cc_z+human']
    results = {m: [] for m in methods}

    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        # Raw standardized
        prot_std = ((ad['prot'] - prep['prot_mean']) / prep['prot_std']).astype(np.float32)
        # PCA
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)

        results['raw_esm'].append(ridge_cv(prot_std, ad['y']))
        results['pca_esm'].append(ridge_cv(prot_pca, ad['y']))
        results['pca_evo'].append(ridge_cv(dna_pca, ad['y']))
        results['concat_pca'].append(ridge_cv(
            np.hstack([prot_pca, dna_pca]), ad['y']))

        # CrossCoder sparse z
        model.eval()
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()
        results['crosscoder_z'].append(ridge_cv(z, ad['y']))
        results['cc_z+human'].append(ridge_cv(
            np.hstack([z, ad['hf']]), ad['y']))

        if (i + 1) % 50 == 0:
            print("  %d/%d ..." % (i + 1, len(assay_data)), flush=True)

    print("\n%-20s %8s %8s" % ("Method", "Mean", "Median"), flush=True)
    print("-" * 40, flush=True)
    for m in methods:
        v = np.array(results[m])
        print("%-20s %8.4f %8.4f" % (m, v.mean(), np.median(v)), flush=True)

    return {m: np.array(v) for m, v in results.items()}


def feature_ablation(assay_data, prep, cfg, model, device):
    """Ablation by feature modality type."""
    print("\n--- Feature Ablation by Modality Type ---", flush=True)

    cats, pn, dn = model.classify_features()
    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'
    sh_mask = cats == 'shared'
    alive_mask = cats != 'dead'

    print("  Feature counts: PP=%d DP=%d SH=%d dead=%d alive=%d" % (
        pp_mask.sum(), dp_mask.sum(), sh_mask.sum(),
        (cats == 'dead').sum(), alive_mask.sum()), flush=True)

    modes = [
        ('all', alive_mask),
        ('prot-private', pp_mask),
        ('dna-private', dp_mask),
        ('shared', sh_mask),
        ('no-prot-priv', alive_mask & ~pp_mask),
        ('no-dna-priv', alive_mask & ~dp_mask),
        ('no-shared', alive_mask & ~sh_mask),
    ]

    results = {name: [] for name, _ in modes}
    assay_results = []

    for i, ad in enumerate(assay_data):
        n = len(ad['y'])
        if n < 50:
            continue

        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        model.eval()
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()

        row = {'assay': ad['name'], 'n': n}
        for name, mask in modes:
            z_sub = z[:, mask]
            if z_sub.shape[1] == 0:
                row[name] = 0.0
            else:
                X = np.hstack([z_sub, ad['hf']])
                row[name] = ridge_cv(X, ad['y'])
            results[name].append(row[name])

        # Modality importance
        row['imp_prot'] = row['all'] - row['no-prot-priv']
        row['imp_dna'] = row['all'] - row['no-dna-priv']
        row['imp_shared'] = row['all'] - row['no-shared']
        assay_results.append(row)

        if (i + 1) % 50 == 0:
            df_tmp = pd.DataFrame(assay_results)
            print("  %d/%d ... all=%.4f imp: PP=%+.4f DP=%+.4f SH=%+.4f" % (
                i + 1, len(assay_data), df_tmp['all'].mean(),
                df_tmp['imp_prot'].mean(), df_tmp['imp_dna'].mean(),
                df_tmp['imp_shared'].mean()), flush=True)

    df = pd.DataFrame(assay_results)

    # Summary
    print("\n%-20s %8s %8s" % ("Mode", "Mean", "Median"), flush=True)
    print("-" * 40, flush=True)
    for name, _ in modes:
        v = np.array(results[name])
        print("%-20s %8.4f %8.4f" % (name, v.mean(), np.median(v)), flush=True)

    print("\nModality importance (Δ = all - no_type):", flush=True)
    print("%-20s %8s %8s %8s" % ("", "PP", "DP", "Shared"), flush=True)
    print("-" * 50, flush=True)
    for stat, fn in [("mean", np.mean), ("median", np.median),
                     (">0 count", lambda x: (x > 0).sum())]:
        vals = [fn(df[c].dropna().values) for c in ['imp_prot', 'imp_dna', 'imp_shared']]
        if stat == ">0 count":
            print("%-20s %8d %8d %8d" % (stat, *vals), flush=True)
        else:
            print("%-20s %8.4f %8.4f %8.4f" % (stat, *vals), flush=True)

    # Top assays by modality
    for label, col in [("protein-private driven", "imp_prot"),
                       ("DNA-private driven", "imp_dna"),
                       ("shared-driven", "imp_shared")]:
        print("\nTop %s:" % label, flush=True)
        for _, r in df.nlargest(5, col).iterrows():
            print("  %-40s all=%.3f PP=%+.3f DP=%+.3f SH=%+.3f" % (
                r['assay'][:40], r['all'],
                r['imp_prot'], r['imp_dna'], r['imp_shared']), flush=True)

    return df


def evaluate_clinvar(clinvar_data, prep, cfg, model, device):
    """ClinVar pathogenicity classification (AUROC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n--- ClinVar Pathogenicity (AUROC) ---", flush=True)
    prot_pca = apply_prep(clinvar_data['prot'], prep['prot_mean'], prep['prot_std'],
                          prep['pca_prot_components'], prep['pca_prot_mean'])
    dna_pca = apply_prep(clinvar_data['dna'], prep['dna_mean'], prep['dna_std'],
                         prep['pca_dna_components'], prep['pca_dna_mean'],
                         prep.get('pca_dna_var') if cfg.whiten_dna else None)
    y = clinvar_data['label']

    model.eval()
    pt = torch.FloatTensor(prot_pca).to(device)
    dt = torch.FloatTensor(dna_pca).to(device)
    with torch.no_grad():
        z = model.encode(pt, dt).cpu().numpy()

    cats, _, _ = model.classify_features()
    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'
    sh_mask = cats == 'shared'
    alive_mask = cats != 'dead'

    feature_sets = {
        'pca_prot': prot_pca,
        'pca_dna': dna_pca,
        'pca_concat': np.hstack([prot_pca, dna_pca]),
        'cc_all': z[:, alive_mask],
        'cc_prot_priv': z[:, pp_mask],
        'cc_dna_priv': z[:, dp_mask],
        'cc_shared': z[:, sh_mask],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("%-20s %8s" % ("Features", "AUROC"), flush=True)
    print("-" * 30, flush=True)

    results = {}
    for name, X in feature_sets.items():
        if X.shape[1] == 0:
            results[name] = 0.0
            continue
        aucs = []
        for tr, te in kf.split(X):
            clf = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        results[name] = np.mean(aucs)
        print("%-20s %8.4f" % (name, results[name]), flush=True)

    # Per-gene analysis for key genes
    key_genes = ['BRCA1', 'BRCA2', 'TP53', 'MSH2', 'MLH1', 'SCN1A', 'CFTR', 'LDLR']
    print("\nPer-gene AUROC (cc_all, genes with >=20 variants):", flush=True)
    gene_results = []
    for gene in key_genes:
        mask = clinvar_data['gene'] == gene
        if mask.sum() < 20:
            continue
        Xg = z[mask][:, alive_mask]
        yg = y[mask]
        if len(set(yg)) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
            clf.fit(Xg, yg)
            prob = clf.predict_proba(Xg)[:, 1]
            auc = roc_auc_score(yg, prob)
            gene_results.append({'gene': gene, 'n': int(mask.sum()),
                                 'n_path': int((yg == 1).sum()),
                                 'auroc': auc})
            print("  %-10s n=%4d (path=%d) AUROC=%.4f" % (gene, mask.sum(), (yg == 1).sum(), auc), flush=True)
        except:
            pass

    return results, gene_results


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["pretrain", "evaluate", "all"])
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--out_dir", default="results/crosscoder_sae")
    ap.add_argument("--checkpoint", default=None)
    # ClinVar
    ap.add_argument("--clinvar_esm2", default="results/clinvar_esm2")
    ap.add_argument("--clinvar_evo2", default="results/variant")
    ap.add_argument("--clinvar_csv", default="data/full/clinvar_variants.csv")
    ap.add_argument("--clinvar_parquet", default="data/variant/clinvar.parquet")
    # Architecture
    ap.add_argument("--d_prot", type=int, default=768)
    ap.add_argument("--d_dna", type=int, default=512)
    ap.add_argument("--n_features", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no_whiten", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    cfg = CrossCoderConfig(
        d_prot=args.d_prot, d_dna=args.d_dna,
        n_features=args.n_features, k=args.k,
        whiten_dna=not args.no_whiten,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
    )

    # Load DMS data
    print("Loading DMS from %s ..." % args.emb_dir, flush=True)
    all_prot, all_dna, assay_data = load_raw_embeddings(args.emb_dir)

    # Load ClinVar data
    clinvar_data = None
    if os.path.exists(args.clinvar_esm2) and os.path.exists(args.clinvar_csv):
        print("Loading ClinVar ...", flush=True)
        cv_prot, cv_dna, clinvar_data = load_clinvar_matched(
            args.clinvar_esm2, args.clinvar_evo2, args.clinvar_csv, args.clinvar_parquet)
        if cv_prot is not None:
            all_prot = np.vstack([all_prot, cv_prot])
            all_dna = np.vstack([all_dna, cv_dna])
            print("Merged: %d total variants (DMS + ClinVar)" % len(all_prot), flush=True)

    # Preprocess
    print("Preprocessing ...", flush=True)
    prot_pca, dna_pca, prep = preprocess(all_prot, all_dna, cfg)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Model
    model = CrossCoderSAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("CrossCoder: {:,} params, {} features, k={}".format(
        n_params, cfg.n_features, cfg.k), flush=True)

    # Train
    if args.phase in ("pretrain", "all"):
        train_crosscoder(model, prot_pca, dna_pca, cfg, device)
        torch.save({
            'state_dict': model.state_dict(),
            'config': asdict(cfg),
            'prep': prep,
        }, os.path.join(args.out_dir, "crosscoder.pt"))
        print("Saved checkpoint.", flush=True)

    # Load
    if args.phase == "evaluate":
        ckpt_path = args.checkpoint or os.path.join(args.out_dir, "crosscoder.pt")
        print("Loading: %s" % ckpt_path, flush=True)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'config' in state:
            cfg = CrossCoderConfig(**state['config'])
            model = CrossCoderSAE(cfg).to(device)
        model.load_state_dict(state['state_dict'])
        prep = state['prep']

    # Evaluate
    if args.phase in ("evaluate", "all"):
        print("\n" + "=" * 70, flush=True)
        print("EVALUATION", flush=True)
        print("=" * 70, flush=True)

        # Feature classification
        cats, pn, dn = model.classify_features()
        print("\nFeature modality profile:", flush=True)
        for cat in ['prot-private', 'dna-private', 'shared', 'dead']:
            print("  %-15s %d" % (cat, (cats == cat).sum()), flush=True)

        # Bottleneck ladder
        ladder = bottleneck_ladder(assay_data, prep, cfg, model, device)

        # Feature ablation (DMS)
        abl_df = feature_ablation(assay_data, prep, cfg, model, device)
        abl_df.to_csv(os.path.join(args.out_dir, "ablation.csv"), index=False)

        # ClinVar pathogenicity
        if clinvar_data is not None:
            cv_results, cv_genes = evaluate_clinvar(clinvar_data, prep, cfg, model, device)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
