#!/usr/bin/env python3
"""
Variant MLP v4: Add pretrained CLIP features to v2-sized model.

Instead of learning cross-modal interaction from scratch (v3), leverage the
pretrained CLIP encoders to project edelta into the cross-modal aligned space.

Features:
  - raw edelta_prot (1280-d) + raw edelta_dna (4096-d)  [v2 baseline]
  - CLIP-projected: clip_prot (256-d) + clip_dna (256-d)  [pretrained alignment]
  - CLIP interaction: clip_prot * clip_dna (256-d)  [cross-modal signal]
  - Extra hand-crafted (8-d): norms, max, std, ratio, LLR

Total input: 1280 + 4096 + 256 + 256 + 256 + 8 = 6152-d
Model: v2-sized (d_proj=768, d_hidden=512) proven to fit the data.
Ensemble: 10 seeds for diversity.
"""

import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP


# ── Model ──────────────────────────────────────────────────────────────

class VariantMLPv4(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_clip=256, d_extra=8,
                 d_proj=768, d_hidden=512, dropout=0.2):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(d_prot, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_dna = nn.Sequential(
            nn.Linear(d_dna, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        self.proj_clip = nn.Sequential(
            nn.Linear(d_clip, d_proj), nn.BatchNorm1d(d_proj), nn.GELU(), nn.Dropout(dropout))
        d_in = d_proj * 3 + d_extra
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, x_prot, x_dna, x_clip, x_extra):
        hp = self.proj_prot(x_prot)
        hd = self.proj_dna(x_dna)
        hc = self.proj_clip(x_clip)
        h = torch.cat([hp, hd, hc, x_extra], dim=-1)
        return self.head(h).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────

def cosine_lr(opt, step, warmup, total, base_lr):
    if step < warmup:
        lr = base_lr * step / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def train_fold(Xp_tr, Xd_tr, Xc_tr, Xe_tr, y_tr,
               Xp_te, Xd_te, Xc_te, Xe_te, y_te,
               device, d_clip=256, d_extra=8,
               epochs=100, batch=2048, lr=3e-4):
    model = VariantMLPv4(d_clip=d_clip, d_extra=d_extra).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xp_v = torch.tensor(Xp_te, dtype=torch.float32, device=device)
    Xd_v = torch.tensor(Xd_te, dtype=torch.float32, device=device)
    Xc_v = torch.tensor(Xc_te, dtype=torch.float32, device=device)
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
            xc = torch.tensor(Xc_tr[idx], dtype=torch.float32, device=device)
            xe = torch.tensor(Xe_tr[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)

            cosine_lr(opt, step, warmup_steps, total_steps, lr)
            opt.zero_grad()
            loss = loss_fn(model(xp, xd, xc, xe), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            vauc = roc_auc_score(y_te, torch.sigmoid(model(Xp_v, Xd_v, Xc_v, Xe_v)).cpu().numpy())
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
        preds = torch.sigmoid(model(Xp_v, Xd_v, Xc_v, Xe_v)).cpu().numpy()
    return preds, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--clip_ckpt", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--baseline_csv", default="results/multiomics_validation/baseline_scores.csv")
    ap.add_argument("--out_dir", default="results/variant_mlp_v4")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load CLIP model ──
    print("Loading CLIP model ...", flush=True)
    ck = torch.load(args.clip_ckpt, map_location="cpu", weights_only=False)
    clip_cfg = ck["config"]
    clip_model = CrossModalCLIP(**clip_cfg)
    clip_model.load_state_dict(ck["model_state"])
    clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False
    d_shared = clip_cfg["d_shared"]
    print(f"  CLIP loaded: d_shared={d_shared}", flush=True)

    # ── Load embeddings ──
    print("Loading embeddings ...", flush=True)
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

    # ── Project through CLIP (batch to avoid OOM) ──
    print("Projecting through CLIP ...", flush=True)
    clip_prot = np.zeros((n, d_shared), dtype=np.float32)
    clip_dna = np.zeros((n, d_shared), dtype=np.float32)
    bs = 4096
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            xp = torch.tensor(prot[i:j], dtype=torch.float32, device=device)
            xd = torch.tensor(dna[i:j], dtype=torch.float32, device=device)
            clip_prot[i:j] = clip_model.enc_prot(xp).cpu().numpy()
            clip_dna[i:j] = clip_model.enc_dna(xd).cpu().numpy()
    clip_cross = clip_prot * clip_dna
    clip_feats = np.concatenate([clip_prot, clip_dna, clip_cross], axis=1).astype(np.float32)
    d_clip = clip_feats.shape[1]
    print(f"  CLIP features: {clip_feats.shape} (prot={d_shared} + dna={d_shared} + cross={d_shared})", flush=True)

    del clip_model
    torch.cuda.empty_cache()

    # Labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)[labeled]
    genes = sub["gene"].values[labeled]
    prot_lab = prot[labeled]; dna_lab = dna[labeled]; llr_lab = llr[labeled]
    clip_lab = clip_feats[labeled]

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
    d_extra = extra.shape[1]

    N = len(y)
    print(f"  labeled: {N} (path={int(y.sum())}, ben={int((y==0).sum())})", flush=True)
    print(f"  features: prot=1280, dna=4096, clip={d_clip}, extra={d_extra}", flush=True)

    # Load baselines
    am_scores = None; revel_scores = None
    if os.path.exists(args.baseline_csv):
        bdf = pd.read_csv(args.baseline_csv)
        if "am_score" in bdf.columns and len(bdf) == n:
            am_all = bdf["am_score"].values.astype(np.float32)
            am_lab = am_all[labeled]
            am_valid = ~np.isnan(am_lab)
            if am_valid.sum() > 1000:
                am_scores = am_lab
                print(f"  AM: {am_valid.sum()}/{N} valid, AUC={roc_auc_score(y[am_valid], am_lab[am_valid]):.4f}", flush=True)
        if "revel_score" in bdf.columns and len(bdf) == n:
            rv_all = bdf["revel_score"].values.astype(np.float32)
            rv_lab = rv_all[labeled]
            rv_valid = ~np.isnan(rv_lab)
            if rv_valid.sum() > 1000:
                revel_scores = rv_lab
                print(f"  REVEL: {rv_valid.sum()}/{N} valid, AUC={roc_auc_score(y[rv_valid], rv_lab[rv_valid]):.4f}", flush=True)

    # 5-fold CV (same folds as v2)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.base_seed)
    folds = list(skf.split(prot_lab, y))

    # ── Train N seeds ──
    all_preds = {}
    seed_aucs = []

    for seed in range(args.n_seeds):
        rng_seed = args.base_seed + seed
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        preds = np.zeros(N)
        fold_aucs = []
        print(f"\n=== Seed {seed} (rng={rng_seed}) ===", flush=True)

        for fi, (tr, te) in enumerate(folds):
            pm, ps = prot_lab[tr].mean(0), prot_lab[tr].std(0) + 1e-8
            dm, ds = dna_lab[tr].mean(0), dna_lab[tr].std(0) + 1e-8
            cm, cs = clip_lab[tr].mean(0), clip_lab[tr].std(0) + 1e-8
            em, es = extra[tr].mean(0), extra[tr].std(0) + 1e-8

            fp, fauc = train_fold(
                (prot_lab[tr] - pm) / ps, (dna_lab[tr] - dm) / ds,
                (clip_lab[tr] - cm) / cs, (extra[tr] - em) / es, y[tr],
                (prot_lab[te] - pm) / ps, (dna_lab[te] - dm) / ds,
                (clip_lab[te] - cm) / cs, (extra[te] - em) / es, y[te],
                device, d_clip=d_clip, d_extra=d_extra)
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
    print(f"  mean single:  {np.mean(seed_aucs):.4f} +/- {np.std(seed_aucs):.4f}", flush=True)

    # Progressive ensemble (2, 3, 5, 7, 10 seeds)
    for k in [2, 3, 5, 7, 10]:
        if k <= args.n_seeds:
            ens_k = np.mean([all_preds[s] for s in range(k)], axis=0)
            auc_k = roc_auc_score(y, ens_k)
            print(f"  ensemble({k:2d}):  {auc_k:.4f}", flush=True)

    # Best ensemble
    ens = np.mean([all_preds[s] for s in range(args.n_seeds)], axis=0)
    auc_ens = roc_auc_score(y, ens)

    print(f"\n  >>> OUR BEST:        {auc_ens:.4f} <<<", flush=True)
    print(f"  AlphaMissense:       0.9638", flush=True)
    print(f"  REVEL:               0.9689", flush=True)
    delta_am = auc_ens - 0.9638
    print(f"  Delta vs AM:         {delta_am:+.4f} {'*** BEAT! ***' if delta_am > 0 else ''}", flush=True)

    # Bootstrap 95% CI
    np.random.seed(args.base_seed)
    n_boot = 2000
    boot_aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        yu = y[idx]
        if yu.sum() > 0 and yu.sum() < len(yu):
            boot_aucs.append(roc_auc_score(yu, ens[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  95% CI:              [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    # Paired bootstrap vs AM
    if am_scores is not None:
        am_valid = ~np.isnan(am_scores)
        wins = 0
        for _ in range(n_boot):
            idx = np.random.choice(np.where(am_valid)[0], am_valid.sum(), replace=True)
            yu = y[idx]
            if yu.sum() > 0 and yu.sum() < len(yu):
                if roc_auc_score(yu, ens[idx]) > roc_auc_score(yu, am_scores[idx]):
                    wins += 1
        print(f"  p(ours > AM):        {wins/n_boot:.3f}", flush=True)

    if revel_scores is not None:
        rv_valid = ~np.isnan(revel_scores)
        wins = 0
        for _ in range(n_boot):
            idx = np.random.choice(np.where(rv_valid)[0], rv_valid.sum(), replace=True)
            yu = y[idx]
            if yu.sum() > 0 and yu.sum() < len(yu):
                if roc_auc_score(yu, ens[idx]) > roc_auc_score(yu, revel_scores[idx]):
                    wins += 1
        print(f"  p(ours > REVEL):     {wins/n_boot:.3f}", flush=True)

    # Per-gene AUC
    gene_aucs = []
    for g in np.unique(genes):
        mask = genes == g
        yg = y[mask]
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
        "n_seeds": args.n_seeds,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
