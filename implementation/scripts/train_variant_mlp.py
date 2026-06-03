"""
End-to-end MLP for variant pathogenicity prediction on raw edelta embeddings.

Architecture (dual-encoder fusion):
  edelta_prot (1280) → proj_prot (1280→512) → ReLU → Dropout
                                                                ↘
                                                                 concat (1024) → hidden (1024→256) → ReLU → Dropout → out (256→1)
                                                                ↗
  edelta_dna  (4096) → proj_dna  (4096→512) → ReLU → Dropout

Also trains prot-only and dna-only variants for ablation.
5-fold stratified CV, reports genome-wide AUC + per-gene AUC.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats


class VariantMLP(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_proj=512, d_hidden=256, dropout=0.3,
                 mode="fusion"):
        super().__init__()
        self.mode = mode
        if mode in ("fusion", "prot"):
            self.proj_prot = nn.Sequential(
                nn.Linear(d_prot, d_proj), nn.ReLU(), nn.Dropout(dropout),
            )
        if mode in ("fusion", "dna"):
            self.proj_dna = nn.Sequential(
                nn.Linear(d_dna, d_proj), nn.ReLU(), nn.Dropout(dropout),
            )
        d_in = {"fusion": d_proj * 2, "prot": d_proj, "dna": d_proj}[mode]
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x_prot=None, x_dna=None):
        parts = []
        if self.mode in ("fusion", "prot") and x_prot is not None:
            parts.append(self.proj_prot(x_prot))
        if self.mode in ("fusion", "dna") and x_dna is not None:
            parts.append(self.proj_dna(x_dna))
        h = torch.cat(parts, dim=-1)
        return self.head(h).squeeze(-1)


def load_data(emb_dir, annotated_csv, dual_idx_path):
    """Load raw edelta for dual-modality variants + labels."""
    df = pd.read_csv(annotated_csv, low_memory=False)
    dual_idx = np.load(dual_idx_path)
    sub = df.iloc[dual_idx].reset_index(drop=True)

    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    n = len(dual_idx)

    prot = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l:
                prot[g2l[int(gi)]] = row

    dna = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l:
                dna[g2l[int(gi)]] = row

    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)

    return prot, dna, y, labeled, sub["gene"].values


def train_fold(model, X_prot_tr, X_dna_tr, y_tr, X_prot_val, X_dna_val, y_val,
               device, epochs=50, batch=2048, lr=1e-3, patience=8):
    """Train one fold with early stopping."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = torch.tensor(y_tr, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    Xp_v = torch.tensor(X_prot_val, dtype=torch.float32, device=device) if X_prot_val is not None else None
    Xd_v = torch.tensor(X_dna_val, dtype=torch.float32, device=device) if X_dna_val is not None else None

    best_auc = 0
    best_state = None
    wait = 0
    n = len(y_tr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            xp = torch.tensor(X_prot_tr[idx], dtype=torch.float32, device=device) if X_prot_tr is not None else None
            xd = torch.tensor(X_dna_tr[idx], dtype=torch.float32, device=device) if X_dna_tr is not None else None
            yb = yt[idx].to(device)
            logits = model(xp, xd)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xp_v, Xd_v)
            val_loss = loss_fn(val_logits, yv).item()
            val_preds = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_preds)

        scheduler.step(val_loss)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(Xp_v, Xd_v)).cpu().numpy()
    return preds, best_auc


def per_gene_auc(preds, y, genes, min_pos=20, min_neg=20):
    rows = []
    for g in np.unique(genes):
        mask = genes == g
        yg = y[mask]
        if (yg == 1).sum() >= min_pos and (yg == 0).sum() >= min_neg:
            rows.append({"gene": g, "auc": roc_auc_score(yg, preds[mask]),
                         "n": int(mask.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--out_dir", default="results/variant_mlp")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_proj", type=int, default=512)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("loading data ...", flush=True)
    prot, dna, y, labeled, genes = load_data(args.emb_dir, args.annotated_csv, args.dual_idx)

    prot_lab = prot[labeled]
    dna_lab = dna[labeled]
    y_lab = y[labeled]
    genes_lab = genes[labeled]
    print(f"labeled: {len(y_lab)} (path={int(y_lab.sum())}, ben={int((y_lab==0).sum())})", flush=True)

    # standardize per fold (mean/std from train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(skf.split(prot_lab, y_lab))

    results = {}
    for mode in ["prot", "dna", "fusion"]:
        print(f"\n=== Training {mode} MLP ===", flush=True)
        all_preds = np.zeros(len(y_lab))

        for fi, (tr_idx, te_idx) in enumerate(folds):
            # normalize
            p_mean = prot_lab[tr_idx].mean(0); p_std = prot_lab[tr_idx].std(0) + 1e-8
            d_mean = dna_lab[tr_idx].mean(0); d_std = dna_lab[tr_idx].std(0) + 1e-8

            Xp_tr = (prot_lab[tr_idx] - p_mean) / p_std
            Xp_te = (prot_lab[te_idx] - p_mean) / p_std
            Xd_tr = (dna_lab[tr_idx] - d_mean) / d_std
            Xd_te = (dna_lab[te_idx] - d_mean) / d_std

            if mode == "prot":
                Xd_tr = Xd_te = None
            elif mode == "dna":
                Xp_tr = Xp_te = None

            model = VariantMLP(
                d_proj=args.d_proj, d_hidden=args.d_hidden,
                dropout=args.dropout, mode=mode,
            )
            preds, best_auc = train_fold(
                model, Xp_tr, Xd_tr, y_lab[tr_idx],
                Xp_te, Xd_te, y_lab[te_idx],
                device, epochs=args.epochs, batch=args.batch, lr=args.lr,
            )
            all_preds[te_idx] = preds
            print(f"  fold {fi}: val_auc={best_auc:.4f}", flush=True)

        auc = roc_auc_score(y_lab, all_preds)
        results[mode] = {"auc": auc, "preds": all_preds}
        print(f"  {mode} genome-wide AUC = {auc:.4f}", flush=True)

    # comparison table
    print("\n=== Genome-wide AUC comparison ===", flush=True)
    print(f"  MLP-prot:   {results['prot']['auc']:.4f}", flush=True)
    print(f"  MLP-dna:    {results['dna']['auc']:.4f}", flush=True)
    print(f"  MLP-fusion: {results['fusion']['auc']:.4f}", flush=True)
    print(f"  (prev LR CLIP-prot:   0.8869)", flush=True)
    print(f"  (prev LR CLIP-fusion: 0.9227)", flush=True)
    print(f"  (AlphaMissense:       0.9638)", flush=True)
    print(f"  (REVEL:               0.9689)", flush=True)

    # per-gene AUC
    print("\n=== Per-gene AUC ===", flush=True)
    gene_results = {}
    for mode in ["prot", "dna", "fusion"]:
        gdf = per_gene_auc(results[mode]["preds"], y_lab, genes_lab)
        gene_results[mode] = gdf
        print(f"  {mode}: mean={gdf['auc'].mean():.3f} median={gdf['auc'].median():.3f} (n={len(gdf)} genes)", flush=True)

    # fusion vs best single
    fusion_g = gene_results["fusion"].set_index("gene")["auc"]
    prot_g = gene_results["prot"].set_index("gene")["auc"]
    dna_g = gene_results["dna"].set_index("gene")["auc"]
    common = fusion_g.index.intersection(prot_g.index).intersection(dna_g.index)
    best_single = pd.concat([prot_g[common], dna_g[common]], axis=1).max(axis=1)
    delta = fusion_g[common] - best_single
    n_better = (delta > 0).sum()
    w = stats.wilcoxon(fusion_g[common], best_single, alternative="greater")
    print(f"\n  fusion > best_single: {n_better}/{len(common)} genes", flush=True)
    print(f"  Wilcoxon p={w.pvalue:.2e}", flush=True)
    print(f"  mean delta-AUC: {delta.mean():.4f}", flush=True)

    # save
    out = {"genome_wide": {m: results[m]["auc"] for m in results},
           "per_gene": {m: gene_results[m].to_dict() for m in gene_results}}
    import json
    with open(os.path.join(args.out_dir, "mlp_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # save best fusion model from last fold
    torch.save(model.state_dict(), os.path.join(args.out_dir, "fusion_mlp_last_fold.pt"))

    # save all predictions for downstream analysis
    np.savez_compressed(os.path.join(args.out_dir, "mlp_preds.npz"),
                        prot=results["prot"]["preds"],
                        dna=results["dna"]["preds"],
                        fusion=results["fusion"]["preds"],
                        y=y_lab, genes=genes_lab)

    print(f"\nsaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
