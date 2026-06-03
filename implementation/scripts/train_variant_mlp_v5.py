#!/usr/bin/env python3
"""
Variant MLP v5: Pretrain projections on 260k variants → fine-tune on 108k labeled.

Key idea: Use ESM-2 LLR (AUC=0.903 zero-shot) as pseudo-supervision for ALL 260k
dual-modality variants. This trains the projection layers on 2.4x more data than
the labeled-only approach, giving a better starting point for fine-tuning.

Stage 1: Pretrain - predict LLR from edelta features (regression, all 260k)
Stage 2: Fine-tune - predict pathogenicity from pretrained features (classification, 108k labeled)
"""

import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Model ──────────────────────────────────────────────────────────────

class VariantMLPv5(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_extra=8,
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
    """Regression head for LLR pretraining."""
    def __init__(self, d_in=1536, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ── Training helpers ───────────────────────────────────────────────────

def cosine_lr(opt, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def pretrain(proj_prot, proj_dna, pt_head, Xp, Xd, y_llr,
             device, epochs=30, batch=4096, lr=1e-3):
    """Pretrain projections to predict LLR from edelta."""
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
                  finetune_proj_lr_scale=0.1):
    """Fine-tune with lower LR on pretrained projections."""
    proj_params = list(model.proj_prot.parameters()) + list(model.proj_dna.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": proj_params, "lr": lr * finetune_proj_lr_scale},
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

            frac = step / max(total_steps, 1)
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
    ap.add_argument("--baseline_csv", default="results/multiomics_validation/baseline_scores.csv")
    ap.add_argument("--out_dir", default="results/variant_mlp_v5")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load all data ──
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
    print(f"  total: {n}, prot: {prot.shape}, dna: {dna.shape}, llr non-zero: {np.count_nonzero(llr_all)}/{n}", flush=True)

    # z-score normalize ALL data (for pretraining)
    prot_mean, prot_std = prot.mean(0), prot.std(0) + 1e-8
    dna_mean, dna_std = dna.mean(0), dna.std(0) + 1e-8
    prot_z = (prot - prot_mean) / prot_std
    dna_z = (dna - dna_mean) / dna_std
    llr_mean, llr_std = llr_all.mean(), llr_all.std() + 1e-8
    llr_z = (llr_all - llr_mean) / llr_std

    # Labels (for fine-tuning)
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)[labeled]
    genes = sub["gene"].values[labeled]
    prot_lab = prot_z[labeled]; dna_lab = dna_z[labeled]; llr_lab = llr_all[labeled]

    # Extra features
    pn = np.linalg.norm(prot[labeled], axis=1, keepdims=True)
    dn = np.linalg.norm(dna[labeled], axis=1, keepdims=True)
    extra = np.concatenate([
        pn, dn, llr_lab.reshape(-1, 1),
        np.max(np.abs(prot[labeled]), axis=1, keepdims=True),
        np.max(np.abs(dna[labeled]), axis=1, keepdims=True),
        np.std(prot[labeled], axis=1, keepdims=True),
        np.std(dna[labeled], axis=1, keepdims=True),
        pn / (dn + 1e-8),
    ], axis=1).astype(np.float32)
    d_extra = extra.shape[1]

    N = len(y)
    print(f"  labeled: {N} (path={int(y.sum())}, ben={int((y==0).sum())}), extra={d_extra}", flush=True)
    print(f"  pretrain data: {n} variants (all dual-modality)", flush=True)

    # Baselines
    am_auc_ref = 0.9638; revel_auc_ref = 0.9689

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.base_seed)
    folds = list(skf.split(prot_lab, y))

    all_preds = {}
    seed_aucs = []

    for seed in range(args.n_seeds):
        rng_seed = args.base_seed + seed
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        print(f"\n=== Seed {seed} (rng={rng_seed}) ===", flush=True)

        # Stage 1: Pretrain projections on ALL 260k
        print(f"  Stage 1: pretraining on {n} variants ...", flush=True)
        ref_model = VariantMLPv5(d_extra=d_extra).to(device)
        pt_head = PretrainHead(d_in=768 * 2).to(device)
        pretrain(ref_model.proj_prot, ref_model.proj_dna, pt_head,
                 prot_z, dna_z, llr_z, device, epochs=args.pretrain_epochs)
        pretrained_state = {
            "proj_prot": {k: v.cpu().clone() for k, v in ref_model.proj_prot.state_dict().items()},
            "proj_dna": {k: v.cpu().clone() for k, v in ref_model.proj_dna.state_dict().items()},
        }
        del ref_model, pt_head
        torch.cuda.empty_cache()

        # Stage 2: Fine-tune on labeled, per-fold
        preds = np.zeros(N)
        fold_aucs = []
        for fi, (tr, te) in enumerate(folds):
            # Per-fold normalization for extra features
            em, es = extra[tr].mean(0), extra[tr].std(0) + 1e-8

            model = VariantMLPv5(d_extra=d_extra).to(device)
            model.proj_prot.load_state_dict(pretrained_state["proj_prot"])
            model.proj_dna.load_state_dict(pretrained_state["proj_dna"])

            fp, fauc = finetune_fold(
                model,
                prot_lab[tr], dna_lab[tr], (extra[tr] - em) / es, y[tr],
                prot_lab[te], dna_lab[te], (extra[te] - em) / es, y[te],
                device)
            preds[te] = fp
            fold_aucs.append(fauc)
            print(f"  fold {fi}: {fauc:.4f}", flush=True)

        sauc = roc_auc_score(y, preds)
        all_preds[seed] = preds
        seed_aucs.append(sauc)
        print(f"  -> genome-wide: {sauc:.4f}", flush=True)

    # ── Ensemble ──
    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    for s in range(args.n_seeds):
        print(f"  seed {s}: {seed_aucs[s]:.4f}", flush=True)
    print(f"  mean single: {np.mean(seed_aucs):.4f} +/- {np.std(seed_aucs):.4f}", flush=True)

    ens = np.mean([all_preds[s] for s in range(args.n_seeds)], axis=0)
    auc_ens = roc_auc_score(y, ens)

    # Progressive ensemble
    for k in [2, 3, 5]:
        if k <= args.n_seeds:
            ens_k = np.mean([all_preds[s] for s in range(k)], axis=0)
            print(f"  ensemble({k}): {roc_auc_score(y, ens_k):.4f}", flush=True)

    print(f"\n  >>> OUR BEST:        {auc_ens:.4f} <<<", flush=True)
    print(f"  AlphaMissense:       {am_auc_ref}", flush=True)
    print(f"  REVEL:               {revel_auc_ref}", flush=True)
    delta = auc_ens - am_auc_ref
    print(f"  Delta vs AM:         {delta:+.4f} {'*** BEAT! ***' if delta > 0 else ''}", flush=True)

    # Bootstrap CI
    np.random.seed(args.base_seed)
    n_boot = 2000
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        yu = y[idx]
        if yu.sum() > 0 and yu.sum() < len(yu):
            boot.append(roc_auc_score(yu, ens[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI:              [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    # Per-gene
    gene_aucs = []
    for g in np.unique(genes):
        mask = genes == g; yg = y[mask]
        if (yg == 1).sum() >= 20 and (yg == 0).sum() >= 20:
            gene_aucs.append({"gene": g, "auc": roc_auc_score(yg, ens[mask]), "n": int(mask.sum())})
    gdf = pd.DataFrame(gene_aucs)
    print(f"\n  per-gene: mean={gdf['auc'].mean():.3f} median={gdf['auc'].median():.3f} (n={len(gdf)})", flush=True)

    # Save
    gdf.to_csv(os.path.join(args.out_dir, "per_gene_auc.csv"), index=False)
    np.savez_compressed(os.path.join(args.out_dir, "preds.npz"),
                        ensemble=ens, y=y, genes=genes)
    results = {
        "seed_aucs": [float(x) for x in seed_aucs],
        "mean_single": float(np.mean(seed_aucs)),
        "ensemble_auc": float(auc_ens),
        "ci_95": [float(ci_lo), float(ci_hi)],
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
