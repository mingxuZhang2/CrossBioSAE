"""
Fine-tune projection layers + MLP on DMS data (same flow as ClinVar v6).
1. Load all ESM-2 + Evo2 edelta from saved .npz
2. Initialize projections from CLIP (via v6 checkpoint)
3. Pool all assays (rank-normalize fitness per assay)
4. Train projections + MLP end-to-end
5. Evaluate per-assay Spearman
"""

import argparse, glob, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from train_variant_sae import ProjectionLayers, TopKSAE


class FusionMLP(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_proj=256, d_hidden=256, dropout=0.2):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        d_in = d_proj * 2
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, xp, xd):
        hp = self.proj_prot(xp)
        hd = self.proj_dna(xd)
        return self.head(torch.cat([hp, hd], dim=-1)).squeeze(-1)

    def encode(self, xp, xd):
        return torch.cat([self.proj_prot(xp), self.proj_dna(xd)], dim=-1)


class ESM2MLP(nn.Module):
    def __init__(self, d_in=1280, d_hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rank_normalize(y):
    """Rank-normalize to [0, 1]."""
    ranks = stats.rankdata(y)
    return (ranks - 1) / (len(ranks) - 1)


def cosine_lr(opt, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def load_all_data(emb_dir):
    """Load all assays, return pooled arrays + per-assay indices."""
    esm2_files = sorted(glob.glob(os.path.join(emb_dir, "*_esm2.npz")))

    all_esm, all_evo, all_y, all_assay_idx = [], [], [], []
    assay_names = []
    assay_ranges = []

    for esm2_path in esm2_files:
        dms_id = os.path.basename(esm2_path).replace("_esm2.npz", "")
        evo2_path = os.path.join(emb_dir, f"{dms_id}_evo2.npz")
        if not os.path.exists(evo2_path):
            continue

        d1 = np.load(esm2_path, allow_pickle=True)
        d2 = np.load(evo2_path, allow_pickle=True)

        esm_d = d1["esm2_edelta"]
        evo_d = d2["evo2_edelta"]
        evo_valid = d2["evo2_valid"].astype(bool)
        y = d1["y_score"]

        mask = evo_valid
        if mask.sum() < 50:
            continue

        esm_sub = esm_d[mask]
        evo_sub = evo_d[mask]
        y_sub = rank_normalize(y[mask])

        start = len(all_y) if not all_y else sum(len(a) for a in all_y)
        all_esm.append(esm_sub)
        all_evo.append(evo_sub)
        all_y.append(y_sub)
        assay_names.append(dms_id)
        assay_ranges.append((start, start + len(y_sub)))

    Xp = np.concatenate(all_esm, axis=0).astype(np.float32)
    Xd = np.concatenate(all_evo, axis=0).astype(np.float32)
    Y = np.concatenate(all_y, axis=0).astype(np.float32)

    # Z-score normalize features
    prot_mean, prot_std = Xp.mean(0), Xp.std(0) + 1e-8
    dna_mean, dna_std = Xd.mean(0), Xd.std(0) + 1e-8
    Xp = (Xp - prot_mean) / prot_std
    Xd = (Xd - dna_mean) / dna_std

    print(f"Loaded {len(assay_names)} assays, {len(Y)} total variants", flush=True)
    return Xp, Xd, Y, assay_names, assay_ranges, (prot_mean, prot_std, dna_mean, dna_std)


def train_model(model, Xp_train, Xd_train, y_train, device,
                epochs=50, batch=2048, lr=1e-3, is_fusion=True):
    """Train a model end-to-end."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    n = len(y_train)
    spe = max(n // batch, 1)
    total_steps = epochs * spe

    Xp_t = torch.tensor(Xp_train, dtype=torch.float32, device=device)
    Xd_t = torch.tensor(Xd_train, dtype=torch.float32, device=device) if Xd_train is not None else None
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    model.train()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0
        nb = 0
        for b in range(0, n, batch):
            idx = perm[b:b+batch]
            cosine_lr(opt, step, warmup=spe * 3, total=total_steps, base_lr=lr)
            opt.zero_grad()
            if is_fusion:
                pred = model(Xp_t[idx], Xd_t[idx])
            else:
                pred = model(Xp_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            step += 1

    return model


def evaluate_per_assay(model, Xp, Xd, Y, assay_names, assay_ranges,
                       device, is_fusion=True):
    """Get per-assay Spearman on the full dataset (in-sample, for pooled model)."""
    model.eval()
    results = []
    with torch.no_grad():
        Xp_t = torch.tensor(Xp, dtype=torch.float32, device=device)
        Xd_t = torch.tensor(Xd, dtype=torch.float32, device=device) if Xd is not None else None
        if is_fusion:
            preds = model(Xp_t, Xd_t).cpu().numpy()
        else:
            preds = model(Xp_t).cpu().numpy()

    for name, (s, e) in zip(assay_names, assay_ranges):
        y_true = Y[s:e]
        y_pred = preds[s:e]
        if len(y_true) < 30 or np.std(y_true) < 1e-8:
            continue
        rho = stats.spearmanr(y_true, y_pred).statistic
        results.append({"assay": name, "n": e - s, "rho": rho})
    return results


def cv_by_assay(Xp, Xd, Y, assay_names, assay_ranges, device,
                n_splits=5, epochs=30, batch=2048, lr=1e-3,
                is_fusion=True, d_proj=256):
    """K-fold CV where folds split ASSAYS (not variants) to avoid leakage."""
    n_assays = len(assay_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = np.zeros(len(Y))

    for fold, (train_assay_idx, test_assay_idx) in enumerate(kf.split(range(n_assays))):
        # Build train/test index masks
        train_mask = np.zeros(len(Y), dtype=bool)
        test_mask = np.zeros(len(Y), dtype=bool)
        for ai in train_assay_idx:
            s, e = assay_ranges[ai]
            train_mask[s:e] = True
        for ai in test_assay_idx:
            s, e = assay_ranges[ai]
            test_mask[s:e] = True

        if is_fusion:
            model = FusionMLP(d_proj=d_proj).to(device)
        else:
            model = ESM2MLP().to(device)

        Xd_tr = Xd[train_mask] if Xd is not None else None
        train_model(model, Xp[train_mask], Xd_tr, Y[train_mask],
                    device, epochs=epochs, batch=batch, lr=lr, is_fusion=is_fusion)

        model.eval()
        with torch.no_grad():
            xp = torch.tensor(Xp[test_mask], dtype=torch.float32, device=device)
            if is_fusion:
                xd = torch.tensor(Xd[test_mask], dtype=torch.float32, device=device)
                all_preds[test_mask] = model(xp, xd).cpu().numpy()
            else:
                all_preds[test_mask] = model(xp).cpu().numpy()

        n_test = sum(1 for ai in test_assay_idx
                     if assay_ranges[ai][1] - assay_ranges[ai][0] >= 30)
        print(f"  Fold {fold+1}/{n_splits}: {n_test} test assays", flush=True)

    # Per-assay Spearman from held-out predictions
    results = []
    for name, (s, e) in zip(assay_names, assay_ranges):
        y_true = Y[s:e]
        y_pred = all_preds[s:e]
        if len(y_true) < 30 or np.std(y_true) < 1e-8:
            continue
        rho = stats.spearmanr(y_true, y_pred).statistic
        results.append({"assay": name, "n": e - s, "rho": rho})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--sae_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/dms_v6_finetune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_proj", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading all DMS data ...", flush=True)
    Xp, Xd, Y, assay_names, assay_ranges, norms = load_all_data(args.emb_dir)

    # ── 1. ESM-2 only MLP (baseline) ──
    print("\n=== ESM-2 MLP (leave-assay-out CV) ===", flush=True)
    esm2_results = cv_by_assay(
        Xp, None, Y, assay_names, assay_ranges, device,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        is_fusion=False)
    esm2_rhos = [r["rho"] for r in esm2_results]
    print(f"  ESM-2 MLP: mean={np.nanmean(esm2_rhos):.4f}  "
          f"median={np.nanmedian(esm2_rhos):.4f}  n={len(esm2_rhos)}", flush=True)

    # ── 2. Fusion MLP (end-to-end projection + head) ──
    print("\n=== Fusion MLP (leave-assay-out CV) ===", flush=True)
    fusion_results = cv_by_assay(
        Xp, Xd, Y, assay_names, assay_ranges, device,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        is_fusion=True, d_proj=args.d_proj)
    fusion_rhos = [r["rho"] for r in fusion_results]
    print(f"  Fusion MLP: mean={np.nanmean(fusion_rhos):.4f}  "
          f"median={np.nanmedian(fusion_rhos):.4f}  n={len(fusion_rhos)}", flush=True)

    # ── 3. Merge and save ──
    esm2_df = pd.DataFrame(esm2_results).rename(columns={"rho": "rho_esm2_mlp_cv"})
    fusion_df = pd.DataFrame(fusion_results).rename(columns={"rho": "rho_fusion_mlp_cv"})
    merged = esm2_df.merge(fusion_df[["assay", "rho_fusion_mlp_cv"]], on="assay", how="outer")

    # Per-assay: fusion > esm2?
    both = merged.dropna(subset=["rho_esm2_mlp_cv", "rho_fusion_mlp_cv"])
    n_fusion_wins = (both["rho_fusion_mlp_cv"] > both["rho_esm2_mlp_cv"]).sum()
    print(f"\n  Fusion > ESM2: {n_fusion_wins}/{len(both)} assays "
          f"({100*n_fusion_wins/len(both):.1f}%)", flush=True)

    merged.to_csv(os.path.join(args.out_dir, "dms_finetune_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("DMS FINE-TUNE SUMMARY")
    print("=" * 80)
    for col in ["rho_esm2_mlp_cv", "rho_fusion_mlp_cv"]:
        vals = merged[col].dropna()
        if len(vals) > 0:
            print(f"  {col:30s}: mean={vals.mean():.4f}  median={vals.median():.4f}  n={len(vals)}")

    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
