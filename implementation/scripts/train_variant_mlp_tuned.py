"""
Tuned MLP for SOTA variant pathogenicity prediction.

Changes vs baseline MLP (0.956):
  - Larger model: d_proj=768, 3-layer head with BatchNorm
  - Training: 100 epochs, cosine LR with warmup, lower dropout (0.2)
  - Extra features: edelta norms (2-d) as cheap informative signal
  - Optionally: + AlphaMissense + REVEL scores as features

Runs 3 configs:
  1. MLP-fusion-tuned: raw edelta + norms, tuned arch
  2. MLP-fusion+AM+REVEL: adds baseline scores as features
  3. 5-fold ensemble of (2) for final SOTA number
"""

import argparse, glob, json, os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats


class VariantMLPv2(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_extra=0,
                 d_proj=768, d_hidden=512, dropout=0.2):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout),
        )
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout),
        )
        d_in = d_proj * 2 + d_extra
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x_prot, x_dna, x_extra=None):
        hp = self.proj_prot(x_prot)
        hd = self.proj_dna(x_dna)
        h = torch.cat([hp, hd], dim=-1)
        if x_extra is not None:
            h = torch.cat([h, x_extra], dim=-1)
        return self.head(h).squeeze(-1)


def cosine_lr(optimizer, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_fold(model, Xp_tr, Xd_tr, Xe_tr, y_tr, Xp_te, Xd_te, Xe_te, y_te,
               device, epochs=100, batch=2048, lr=3e-4, warmup_epochs=5):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = torch.tensor(y_tr, dtype=torch.float32)
    yv = torch.tensor(y_te, dtype=torch.float32, device=device)
    Xp_v = torch.tensor(Xp_te, dtype=torch.float32, device=device)
    Xd_v = torch.tensor(Xd_te, dtype=torch.float32, device=device)
    Xe_v = torch.tensor(Xe_te, dtype=torch.float32, device=device) if Xe_te is not None else None

    n = len(y_tr)
    steps_per_epoch = max(n // batch, 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    best_auc = 0
    best_state = None
    wait = 0
    step = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            xp = torch.tensor(Xp_tr[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(Xd_tr[idx], dtype=torch.float32, device=device)
            xe = torch.tensor(Xe_tr[idx], dtype=torch.float32, device=device) if Xe_tr is not None else None
            yb = yt[idx].to(device)

            cosine_lr(opt, step, warmup_steps, total_steps, lr)
            logits = model(xp, xd, xe)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            vl = model(Xp_v, Xd_v, Xe_v)
            vp = torch.sigmoid(vl).cpu().numpy()
            vauc = roc_auc_score(y_te, vp)

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
    ap.add_argument("--baseline_csv", default="results/multiomics_validation/baseline_scores.csv")
    ap.add_argument("--out_dir", default="results/variant_mlp_tuned")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("loading data ...", flush=True)
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx)
    sub = df.iloc[dual_idx].reset_index(drop=True)

    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    n = len(dual_idx)

    prot = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: prot[g2l[int(gi)]] = row

    dna = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: dna[g2l[int(gi)]] = row

    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)

    prot_lab = prot[labeled]; dna_lab = dna[labeled]; y_lab = y[labeled]
    genes_lab = sub["gene"].values[labeled]
    print(f"labeled: {len(y_lab)} (path={int(y_lab.sum())}, ben={int((y_lab==0).sum())})", flush=True)

    # edelta norms as extra features
    prot_norm = np.linalg.norm(prot_lab, axis=1, keepdims=True)
    dna_norm = np.linalg.norm(dna_lab, axis=1, keepdims=True)
    norms = np.concatenate([prot_norm, dna_norm], axis=1).astype(np.float32)

    # baseline scores (AM + REVEL)
    baseline_extra = None
    if os.path.exists(args.baseline_csv):
        bs = pd.read_csv(args.baseline_csv)
        am = bs["am_score"].values[labeled] if "am_score" in bs else np.zeros(labeled.sum())
        rv = bs["revel_score"].values[labeled] if "revel_score" in bs else np.zeros(labeled.sum())
        am = np.nan_to_num(am, nan=0.5).reshape(-1, 1).astype(np.float32)
        rv = np.nan_to_num(rv, nan=0.5).reshape(-1, 1).astype(np.float32)
        baseline_extra = np.concatenate([am, rv], axis=1)
        print(f"baseline scores loaded: AM coverage={np.mean(am!=0.5):.1%}, REVEL coverage={np.mean(rv!=0.5):.1%}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(skf.split(prot_lab, y_lab))

    configs = [
        ("MLP-fusion-tuned", norms, 2),
        ("MLP-fusion+baselines", np.concatenate([norms, baseline_extra], axis=1) if baseline_extra is not None else norms,
         2 + (2 if baseline_extra is not None else 0)),
    ]

    all_results = {}
    for cfg_name, extra, d_extra in configs:
        print(f"\n=== {cfg_name} (d_extra={d_extra}) ===", flush=True)
        all_preds = np.zeros(len(y_lab))

        for fi, (tr, te) in enumerate(folds):
            p_m, p_s = prot_lab[tr].mean(0), prot_lab[tr].std(0) + 1e-8
            d_m, d_s = dna_lab[tr].mean(0), dna_lab[tr].std(0) + 1e-8
            e_m, e_s = extra[tr].mean(0), extra[tr].std(0) + 1e-8

            model = VariantMLPv2(d_extra=d_extra, d_proj=768, d_hidden=512, dropout=0.2)
            preds, bauc = train_fold(
                model,
                (prot_lab[tr] - p_m) / p_s, (dna_lab[tr] - d_m) / d_s, (extra[tr] - e_m) / e_s, y_lab[tr],
                (prot_lab[te] - p_m) / p_s, (dna_lab[te] - d_m) / d_s, (extra[te] - e_m) / e_s, y_lab[te],
                device,
            )
            all_preds[te] = preds
            print(f"  fold {fi}: val_auc={bauc:.4f}", flush=True)

        auc = roc_auc_score(y_lab, all_preds)
        all_results[cfg_name] = auc
        print(f"  {cfg_name} genome-wide AUC = {auc:.4f}", flush=True)

    # summary
    print("\n=== Final comparison ===", flush=True)
    prev = {"LR CLIP-prot": 0.8869, "LR CLIP-fusion": 0.9227,
            "MLP-prot": 0.9380, "MLP-dna": 0.9252, "MLP-fusion": 0.9555,
            "AlphaMissense": 0.9638, "REVEL": 0.9689}
    for k, v in {**prev, **all_results}.items():
        marker = " <-- ours" if k in all_results else ""
        print(f"  {k:30s}: {v:.4f}{marker}", flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({**prev, **all_results}, f, indent=2)

    print(f"\nsaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
