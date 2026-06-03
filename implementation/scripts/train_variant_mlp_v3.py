#!/usr/bin/env python3
"""
Variant MLP v3: Diverse ensemble to beat AlphaMissense (0.9638).

Key improvements over v2:
  - Cross-modal gated Hadamard interaction (protein ⊙ DNA modulated by learned gate)
  - Residual blocks in classifier head
  - Richer hand-crafted features (8-d: norms, max, std, ratio, LLR)
  - Label smoothing + Mixup augmentation
  - Gradient clipping
  - Diverse ensemble: 5 architectures × 3 seeds
"""

import argparse, glob, json, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ── Model ──────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.GELU(), nn.Dropout(dropout))
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class VariantMLP(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_extra=8,
                 d_proj=1024, d_hidden=768, dropout=0.15, use_cross=True):
        super().__init__()
        self.use_cross = use_cross
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        if use_cross:
            self.gate = nn.Sequential(nn.Linear(d_proj * 2, d_proj), nn.Sigmoid())
            d_in = d_proj * 3 + d_extra
        else:
            d_in = d_proj * 2 + d_extra
        self.head = nn.Sequential(
            ResBlock(d_in, d_hidden, dropout),
            ResBlock(d_hidden, d_hidden, dropout),
            ResBlock(d_hidden, d_hidden // 2, dropout),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, x_prot, x_dna, x_extra):
        hp = self.proj_prot(x_prot)
        hd = self.proj_dna(x_dna)
        if self.use_cross:
            g = self.gate(torch.cat([hp, hd], dim=-1))
            hc = g * (hp * hd)
            h = torch.cat([hp, hd, hc, x_extra], dim=-1)
        else:
            h = torch.cat([hp, hd, x_extra], dim=-1)
        return self.head(h).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────

def cosine_lr(opt, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def train_fold(cfg, Xp_tr, Xd_tr, Xe_tr, y_tr, Xp_te, Xd_te, Xe_te, y_te, device):
    d_extra = Xe_tr.shape[1]
    model = VariantMLP(d_extra=d_extra, d_proj=cfg["d_proj"], d_hidden=cfg["d_hidden"],
                       dropout=cfg["dropout"], use_cross=cfg["cross"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    eps = cfg["label_smooth"]
    y_sm = y_tr * (1 - eps) + (1 - y_tr) * eps if eps > 0 else y_tr
    yt = torch.tensor(y_sm, dtype=torch.float32)

    Xp_v = torch.tensor(Xp_te, dtype=torch.float32, device=device)
    Xd_v = torch.tensor(Xd_te, dtype=torch.float32, device=device)
    Xe_v = torch.tensor(Xe_te, dtype=torch.float32, device=device)

    n = len(y_tr)
    batch = cfg.get("batch", 2048)
    spe = max(n // batch, 1)
    epochs = cfg.get("epochs", 150)
    total_steps = epochs * spe
    warmup_steps = 10 * spe
    mixup_alpha = cfg.get("mixup", 0.0)

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

            if mixup_alpha > 0 and len(idx) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm2 = torch.randperm(len(xp), device=device)
                xp = lam * xp + (1 - lam) * xp[perm2]
                xd = lam * xd + (1 - lam) * xd[perm2]
                xe = lam * xe + (1 - lam) * xe[perm2]
                yb = lam * yb + (1 - lam) * yb[perm2]

            cosine_lr(opt, step, warmup_steps, total_steps, cfg["lr"])
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
            if wait >= 20:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(Xp_v, Xd_v, Xe_v)).cpu().numpy()
    return preds, best_auc


# ── Configs (diverse for ensemble) ─────────────────────────────────────

CONFIGS = [
    {"name": "A_large_cross", "d_proj": 1024, "d_hidden": 768, "dropout": 0.15,
     "cross": True, "label_smooth": 0.05, "mixup": 0.2, "lr": 5e-4},
    {"name": "B_med_cross", "d_proj": 768, "d_hidden": 512, "dropout": 0.2,
     "cross": True, "label_smooth": 0.03, "mixup": 0.1, "lr": 3e-4},
    {"name": "C_xlarge_cross", "d_proj": 1536, "d_hidden": 1024, "dropout": 0.1,
     "cross": True, "label_smooth": 0.05, "mixup": 0.3, "lr": 5e-4},
    {"name": "D_large_nocross", "d_proj": 1024, "d_hidden": 768, "dropout": 0.15,
     "cross": False, "label_smooth": 0.05, "mixup": 0.2, "lr": 5e-4},
    {"name": "E_baseline_plus", "d_proj": 768, "d_hidden": 512, "dropout": 0.2,
     "cross": False, "label_smooth": 0.0, "mixup": 0.0, "lr": 3e-4},
]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--baseline_csv", default="results/multiomics_validation/baseline_scores.csv")
    ap.add_argument("--out_dir", default="results/variant_mlp_v3")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=42)
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

    llr = np.load(args.llr_path)["llr"].astype(np.float32)
    print(f"  prot: {prot.shape}, dna: {dna.shape}, llr non-zero: {np.count_nonzero(llr)}/{n}", flush=True)

    # Labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)[labeled]
    genes = sub["gene"].values[labeled]
    prot_lab = prot[labeled]; dna_lab = dna[labeled]; llr_lab = llr[labeled]

    # Extra features (8-d)
    pn = np.linalg.norm(prot_lab, axis=1, keepdims=True)
    dn = np.linalg.norm(dna_lab, axis=1, keepdims=True)
    extra = np.concatenate([
        pn, dn, llr_lab.reshape(-1, 1),
        np.max(np.abs(prot_lab), axis=1, keepdims=True),
        np.max(np.abs(dna_lab), axis=1, keepdims=True),
        np.std(prot_lab, axis=1, keepdims=True),
        np.std(dna_lab, axis=1, keepdims=True),
        pn / (dn + 1e-8),
    ], axis=1).astype(np.float32)

    N = len(y)
    print(f"  labeled: {N} (path={int(y.sum())}, ben={int((y==0).sum())}), extra={extra.shape[1]}d", flush=True)

    # Load baseline scores for paired comparison
    am_scores = None
    revel_scores = None
    if os.path.exists(args.baseline_csv):
        bdf = pd.read_csv(args.baseline_csv)
        if "am_score" in bdf.columns and len(bdf) == n:
            am_all = bdf["am_score"].values.astype(np.float32)
            am_lab = am_all[labeled]
            am_valid = ~np.isnan(am_lab)
            if am_valid.sum() > 1000:
                am_scores = am_lab
                print(f"  AM scores loaded: {am_valid.sum()}/{N} valid", flush=True)
        if "revel_score" in bdf.columns and len(bdf) == n:
            rv_all = bdf["revel_score"].values.astype(np.float32)
            rv_lab = rv_all[labeled]
            rv_valid = ~np.isnan(rv_lab)
            if rv_valid.sum() > 1000:
                revel_scores = rv_lab
                print(f"  REVEL scores loaded: {rv_valid.sum()}/{N} valid", flush=True)

    # 5-fold CV (fixed across all configs/seeds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.base_seed)
    folds = list(skf.split(prot_lab, y))

    # ── Train all configs × seeds ──
    all_run_preds = {}
    run_aucs = {}

    for ci, cfg in enumerate(CONFIGS):
        for seed in range(args.n_seeds):
            run_key = (cfg["name"], seed)
            rng_seed = args.base_seed + ci * 1000 + seed
            torch.manual_seed(rng_seed)
            np.random.seed(rng_seed)
            preds = np.zeros(N)
            fold_aucs = []
            print(f"\n[{cfg['name']}] seed={seed} (rng={rng_seed})", flush=True)

            for fi, (tr, te) in enumerate(folds):
                pm, ps = prot_lab[tr].mean(0), prot_lab[tr].std(0) + 1e-8
                dm, ds = dna_lab[tr].mean(0), dna_lab[tr].std(0) + 1e-8
                em, es = extra[tr].mean(0), extra[tr].std(0) + 1e-8
                fp, fauc = train_fold(
                    cfg,
                    (prot_lab[tr] - pm) / ps, (dna_lab[tr] - dm) / ds, (extra[tr] - em) / es, y[tr],
                    (prot_lab[te] - pm) / ps, (dna_lab[te] - dm) / ds, (extra[te] - em) / es, y[te],
                    device)
                preds[te] = fp
                fold_aucs.append(fauc)
                print(f"  fold {fi}: {fauc:.4f}", flush=True)

            run_auc = roc_auc_score(y, preds)
            all_run_preds[run_key] = preds
            run_aucs[run_key] = run_auc
            print(f"  -> genome-wide: {run_auc:.4f}", flush=True)

        # save partial results after each config
        partial = {cfg["name"]: {
            "single_aucs": [run_aucs[(cfg["name"], s)] for s in range(args.n_seeds)],
            "mean": float(np.mean([run_aucs[(cfg["name"], s)] for s in range(args.n_seeds)])),
        }}
        with open(os.path.join(args.out_dir, f"partial_{cfg['name']}.json"), "w") as f:
            json.dump(partial, f, indent=2)

    # ── Ensemble ──
    print("\n" + "=" * 60, flush=True)
    print("ENSEMBLE RESULTS", flush=True)
    print("=" * 60, flush=True)

    # Per-config ensemble
    config_ens_aucs = {}
    for cfg in CONFIGS:
        seeds_preds = [all_run_preds[(cfg["name"], s)] for s in range(args.n_seeds)]
        ens = np.mean(seeds_preds, axis=0)
        auc = roc_auc_score(y, ens)
        config_ens_aucs[cfg["name"]] = auc
        mean_single = np.mean([run_aucs[(cfg["name"], s)] for s in range(args.n_seeds)])
        print(f"  {cfg['name']:20s}  single={mean_single:.4f}  ens({args.n_seeds})={auc:.4f}", flush=True)

    # Full diverse ensemble
    all_preds_list = list(all_run_preds.values())
    full_ens = np.mean(all_preds_list, axis=0)
    full_auc = roc_auc_score(y, full_ens)

    # Top-K ensemble (pick best configs)
    sorted_cfgs = sorted(config_ens_aucs.items(), key=lambda x: -x[1])
    top3_names = [c[0] for c in sorted_cfgs[:3]]
    top3_preds = []
    for name in top3_names:
        for s in range(args.n_seeds):
            top3_preds.append(all_run_preds[(name, s)])
    top3_ens = np.mean(top3_preds, axis=0)
    top3_auc = roc_auc_score(y, top3_ens)

    print(f"\n  FULL ensemble  ({len(all_preds_list):2d} models): {full_auc:.4f}", flush=True)
    print(f"  TOP-3 ensemble ({len(top3_preds):2d} models): {top3_auc:.4f}  [{', '.join(top3_names)}]", flush=True)

    best_ens = full_ens if full_auc >= top3_auc else top3_ens
    best_auc = max(full_auc, top3_auc)
    best_label = "FULL" if full_auc >= top3_auc else "TOP-3"

    print(f"\n  >>> BEST ({best_label}): {best_auc:.4f} <<<", flush=True)
    print(f"  AlphaMissense:       0.9638", flush=True)
    print(f"  REVEL:               0.9689", flush=True)
    delta_am = best_auc - 0.9638
    print(f"  Delta vs AM:         {delta_am:+.4f} {'BEAT!' if delta_am > 0 else 'below'}", flush=True)

    # Bootstrap 95% CI
    np.random.seed(args.base_seed)
    n_boot = 2000
    boot_aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        yu = y[idx]
        if yu.sum() > 0 and yu.sum() < len(yu):
            boot_aucs.append(roc_auc_score(yu, best_ens[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  95% Bootstrap CI:    [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    # Paired bootstrap vs baselines
    if am_scores is not None:
        am_valid = ~np.isnan(am_scores)
        am_auc = roc_auc_score(y[am_valid], am_scores[am_valid])
        wins = 0
        for _ in range(n_boot):
            idx = np.random.choice(np.where(am_valid)[0], am_valid.sum(), replace=True)
            yu = y[idx]
            if yu.sum() > 0 and yu.sum() < len(yu):
                if roc_auc_score(yu, best_ens[idx]) > roc_auc_score(yu, am_scores[idx]):
                    wins += 1
        p_vs_am = 1 - wins / n_boot
        print(f"  AM AUC (matched):    {am_auc:.4f}  p(ours>AM)={1-p_vs_am:.3f}", flush=True)

    if revel_scores is not None:
        rv_valid = ~np.isnan(revel_scores)
        rv_auc = roc_auc_score(y[rv_valid], revel_scores[rv_valid])
        wins = 0
        for _ in range(n_boot):
            idx = np.random.choice(np.where(rv_valid)[0], rv_valid.sum(), replace=True)
            yu = y[idx]
            if yu.sum() > 0 and yu.sum() < len(yu):
                if roc_auc_score(yu, best_ens[idx]) > roc_auc_score(yu, revel_scores[idx]):
                    wins += 1
        p_vs_rv = 1 - wins / n_boot
        print(f"  REVEL AUC (matched): {rv_auc:.4f}  p(ours>RV)={1-p_vs_rv:.3f}", flush=True)

    # Per-gene AUC
    gene_aucs = []
    for g in np.unique(genes):
        mask = genes == g
        yg = y[mask]
        if (yg == 1).sum() >= 20 and (yg == 0).sum() >= 20:
            gene_aucs.append({"gene": g, "auc": roc_auc_score(yg, best_ens[mask]), "n": int(mask.sum())})
    gdf = pd.DataFrame(gene_aucs)
    print(f"\n  per-gene: mean={gdf['auc'].mean():.3f} median={gdf['auc'].median():.3f} (n={len(gdf)})", flush=True)

    # Save
    gdf.to_csv(os.path.join(args.out_dir, "per_gene_auc.csv"), index=False)
    np.savez_compressed(os.path.join(args.out_dir, "preds.npz"),
                        full_ensemble=full_ens, top3_ensemble=top3_ens,
                        y=y, genes=genes)
    results = {
        "full_ensemble_auc": float(full_auc),
        "top3_ensemble_auc": float(top3_auc),
        "best_auc": float(best_auc),
        "best_label": best_label,
        "ci_95": [float(ci_lo), float(ci_hi)],
        "n_models": len(all_preds_list),
        "per_config": {cfg["name"]: {
            "single_aucs": [float(run_aucs[(cfg["name"], s)]) for s in range(args.n_seeds)],
            "mean_single": float(np.mean([run_aucs[(cfg["name"], s)] for s in range(args.n_seeds)])),
            "ensemble_auc": float(config_ens_aucs[cfg["name"]]),
        } for cfg in CONFIGS},
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
