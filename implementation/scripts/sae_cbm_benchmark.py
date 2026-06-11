"""
Concept Bottleneck Model benchmark:
  SAE concepts (interpretable) → MLP head → prediction

The concepts remain interpretable (you can inspect which SAE features fire),
but the combination rule is nonlinear (small MLP).
This is the standard CBM architecture from medical imaging literature.

Benchmark: per-assay 5-fold CV, compare ridge vs MLP on SAE features
at different TopK values. Also compare with raw embedding baselines.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


HYDROPHOBIC = set('AVILMFWP')
CHARGE_MAP = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


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


def topk_np(z, k):
    if k >= z.shape[1]:
        return z.copy()
    out = np.zeros_like(z)
    for i in range(z.shape[0]):
        idx = np.argpartition(z[i], -k)[-k:]
        out[i, idx] = z[i, idx]
    return out


class SmallMLP(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_test, y_test, d_in, n_epochs=50, lr=1e-3, bs=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SmallMLP(d_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_ds = TensorDataset(torch.FloatTensor(X_train).to(device),
                              torch.FloatTensor(y_train).to(device))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    return y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_cbm_benchmark")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    # Load SAE encoder
    print("Loading SAE ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()
    enc_b = sd["encoder.bias"].numpy()

    # Load data
    print("Loading data ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    assay_data = []
    for sf in sae_files:
        name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        y = d["y_score"]; mut = d.get("mutants", np.array([]))
        pr = d.get("prot_reps"); dr = d.get("dna_reps")
        if pr is None or dr is None or len(mut) == 0 or len(mut) != len(y):
            continue
        if len(y) < 100:
            continue

        x = np.concatenate([pr, dr], axis=1).astype(np.float32)
        z_full = np.maximum(0, x @ enc_w.T + enc_b)  # pre-TopK

        n = len(mut)
        hf = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            r, p, a = parse_mutant(mut[i])
            if r and a and r in AA_TO_IDX and a in AA_TO_IDX:
                hf[i] = compute_human_features(r, a)

        assay_data.append({
            'name': name, 'z_full': z_full, 'hf': hf,
            'y': y.astype(np.float32),
            'x_raw': x, 'prot_reps': pr, 'esm_raw': d.get("prot_reps"),
        })

    print("Loaded %d assays" % len(assay_data), flush=True)

    # Benchmark configs
    configs = [
        # (name, feature_fn, use_mlp)
        ("SAE k=128 ridge",    lambda ad: np.hstack([topk_np(ad['z_full'], 128), ad['hf']]), False),
        ("SAE k=128 MLP",      lambda ad: np.hstack([topk_np(ad['z_full'], 128), ad['hf']]), True),
        ("SAE k=256 ridge",    lambda ad: np.hstack([topk_np(ad['z_full'], 256), ad['hf']]), False),
        ("SAE k=256 MLP",      lambda ad: np.hstack([topk_np(ad['z_full'], 256), ad['hf']]), True),
        ("SAE k=all ridge",    lambda ad: np.hstack([ad['z_full'], ad['hf']]), False),
        ("SAE k=all MLP",      lambda ad: np.hstack([ad['z_full'], ad['hf']]), True),
        ("raw concat ridge",   lambda ad: np.hstack([ad['x_raw'], ad['hf']]), False),
        ("raw concat MLP",     lambda ad: np.hstack([ad['x_raw'], ad['hf']]), True),
    ]

    results = {c[0]: [] for c in configs}
    assay_names = []

    for ad_idx, ad in enumerate(assay_data):
        y = ad['y']
        assay_names.append(ad['name'])

        for cname, feat_fn, use_mlp in configs:
            X = feat_fn(ad)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rhos = []
            for train_idx, test_idx in kf.split(X):
                if use_mlp:
                    y_pred = train_mlp(X[train_idx], y[train_idx],
                                       X[test_idx], y[test_idx],
                                       d_in=X.shape[1], n_epochs=40)
                else:
                    model = Ridge(alpha=1.0)
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])

                if np.std(y_pred) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                    rho = stats.spearmanr(y_pred, y[test_idx]).statistic
                    if not np.isnan(rho):
                        rhos.append(rho)

            results[cname].append(np.mean(rhos) if rhos else 0)

        if (ad_idx + 1) % 30 == 0:
            print("  %d/%d assays ..." % (ad_idx + 1, len(assay_data)), flush=True)
            for cname, _, _ in configs:
                vals = results[cname]
                if vals:
                    print("    %-25s mean=%.4f" % (cname, np.mean(vals)), flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("CBM BENCHMARK SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print("%-25s %8s %8s %12s" % ("Method", "Mean", "Median", "Interpretable"), flush=True)
    print("-" * 60, flush=True)
    for cname, _, _ in configs:
        vals = np.array(results[cname])
        interp = "YES" if "SAE" in cname else "no"
        print("%-25s %8.4f %8.4f %12s" % (cname, vals.mean(), np.median(vals), interp),
              flush=True)

    # Save
    rows = []
    for i, name in enumerate(assay_names):
        row = {'assay': name}
        for cname, _, _ in configs:
            row[cname.replace(' ', '_')] = results[cname][i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "cbm_results.csv"), index=False)

    # Win rates
    print("\nWin rates (SAE k=128 MLP vs others):", flush=True)
    ours = np.array(results["SAE k=128 MLP"])
    for cname, _, _ in configs:
        if cname == "SAE k=128 MLP":
            continue
        other = np.array(results[cname])
        wins = (ours > other).sum()
        print("  vs %-25s: %d/%d (%.0f%%)" % (cname, wins, len(ours), 100*wins/len(ours)),
              flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
