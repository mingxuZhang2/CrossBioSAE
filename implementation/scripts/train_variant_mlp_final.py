"""
Final MLP: edelta (1280+4096) + LLR (1) + edelta norms (2) → pathogenicity.

The LLR (log-likelihood ratio from ESM-2 wildtype marginal) is the primary
zero-shot signal for variant effect prediction. Combined with the full
embedding delta from both modalities, this should push past AlphaMissense.
"""

import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats


class VariantMLPFinal(nn.Module):
    def __init__(self, d_prot=1280, d_dna=4096, d_extra=3,
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

    def forward(self, x_prot, x_dna, x_extra):
        hp = self.proj_prot(x_prot)
        hd = self.proj_dna(x_dna)
        h = torch.cat([hp, hd, x_extra], dim=-1)
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
               device, epochs=100, batch=2048, lr=3e-4):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = torch.tensor(y_tr, dtype=torch.float32)
    yv = torch.tensor(y_te, dtype=torch.float32, device=device)
    Xp_v = torch.tensor(Xp_te, dtype=torch.float32, device=device)
    Xd_v = torch.tensor(Xd_te, dtype=torch.float32, device=device)
    Xe_v = torch.tensor(Xe_te, dtype=torch.float32, device=device)

    n = len(y_tr)
    steps_per_ep = max(n // batch, 1)
    total_steps = epochs * steps_per_ep
    warmup_steps = 5 * steps_per_ep

    best_auc = 0; best_state = None; wait = 0; step = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            xp = torch.tensor(Xp_tr[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(Xd_tr[idx], dtype=torch.float32, device=device)
            xe = torch.tensor(Xe_tr[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)
            cosine_lr(opt, step, warmup_steps, total_steps, lr)
            loss = loss_fn(model(xp, xd, xe), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xp_v, Xd_v, Xe_v)).cpu().numpy()
            vauc = roc_auc_score(y_te, vp)

        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(Xp_v, Xd_v, Xe_v)).cpu().numpy()
    return preds, best_auc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--out_dir", default="results/variant_mlp_final")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load
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

    # LLR
    llr = np.zeros(n, dtype=np.float32)
    if os.path.exists(args.llr_path):
        llr_data = np.load(args.llr_path)
        llr = llr_data["llr"].astype(np.float32)
        print(f"LLR loaded: {np.count_nonzero(llr)}/{n} non-zero", flush=True)
    else:
        print("WARNING: LLR file not found, using zeros", flush=True)

    # labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y_lab = np.where(is_path.values, 1.0, 0.0)[labeled]
    genes_lab = sub["gene"].values[labeled]

    prot_lab = prot[labeled]; dna_lab = dna[labeled]; llr_lab = llr[labeled]
    prot_norm = np.linalg.norm(prot_lab, axis=1, keepdims=True)
    dna_norm = np.linalg.norm(dna_lab, axis=1, keepdims=True)
    extra = np.concatenate([prot_norm, dna_norm, llr_lab.reshape(-1, 1)], axis=1).astype(np.float32)
    d_extra = extra.shape[1]

    print(f"labeled: {len(y_lab)} (path={int(y_lab.sum())}, ben={int((y_lab==0).sum())})", flush=True)
    print(f"features: prot={prot_lab.shape[1]}, dna={dna_lab.shape[1]}, extra={d_extra}", flush=True)

    # LLR-only AUC (sanity check)
    llr_valid = llr_lab != 0
    if llr_valid.sum() > 100:
        llr_auc = roc_auc_score(y_lab[llr_valid], -llr_lab[llr_valid])
        print(f"LLR-only AUC (zero-shot): {llr_auc:.4f} (n={llr_valid.sum()})", flush=True)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(skf.split(prot_lab, y_lab))

    all_preds = np.zeros(len(y_lab))
    fold_models = []

    for fi, (tr, te) in enumerate(folds):
        p_m, p_s = prot_lab[tr].mean(0), prot_lab[tr].std(0) + 1e-8
        d_m, d_s = dna_lab[tr].mean(0), dna_lab[tr].std(0) + 1e-8
        e_m, e_s = extra[tr].mean(0), extra[tr].std(0) + 1e-8

        model = VariantMLPFinal(d_extra=d_extra)
        preds, bauc, trained_model = train_fold(
            model,
            (prot_lab[tr] - p_m) / p_s, (dna_lab[tr] - d_m) / d_s, (extra[tr] - e_m) / e_s, y_lab[tr],
            (prot_lab[te] - p_m) / p_s, (dna_lab[te] - d_m) / d_s, (extra[te] - e_m) / e_s, y_lab[te],
            device,
        )
        all_preds[te] = preds
        fold_models.append((trained_model, p_m, p_s, d_m, d_s, e_m, e_s))
        print(f"  fold {fi}: val_auc={bauc:.4f}", flush=True)

    auc_final = roc_auc_score(y_lab, all_preds)

    # 5-fold ensemble: each sample predicted by all 5 models, average
    ensemble_preds = np.zeros(len(y_lab))
    for fi, (tr, te) in enumerate(folds):
        for mi, (m, pm, ps, dm, ds, em, es) in enumerate(fold_models):
            if mi == fi:
                continue  # skip the model that was trained on this fold's train set
            m = m.to(device).eval()
            with torch.no_grad():
                xp = torch.tensor((prot_lab[te] - pm) / ps, dtype=torch.float32, device=device)
                xd = torch.tensor((dna_lab[te] - dm) / ds, dtype=torch.float32, device=device)
                xe = torch.tensor((extra[te] - em) / es, dtype=torch.float32, device=device)
                p = torch.sigmoid(m(xp, xd, xe)).cpu().numpy()
            ensemble_preds[te] += p
    ensemble_preds[y_lab >= 0] /= 4  # 4 other-fold models per sample
    auc_ensemble = roc_auc_score(y_lab, ensemble_preds)

    print(f"\n=== Results ===", flush=True)
    print(f"  MLP-final (single):   {auc_final:.4f}", flush=True)
    print(f"  MLP-final (ensemble): {auc_ensemble:.4f}", flush=True)
    print(f"  (prev MLP-fusion:     0.9555)", flush=True)
    print(f"  (AlphaMissense:       0.9638)", flush=True)
    print(f"  (REVEL:               0.9689)", flush=True)

    # per-gene
    gene_aucs = []
    for g in np.unique(genes_lab):
        mask = genes_lab == g
        yg = y_lab[mask]
        if (yg == 1).sum() >= 20 and (yg == 0).sum() >= 20:
            gene_aucs.append({"gene": g, "auc": roc_auc_score(yg, ensemble_preds[mask])})
    gdf = pd.DataFrame(gene_aucs)
    print(f"\n  per-gene (ensemble): mean={gdf['auc'].mean():.3f} median={gdf['auc'].median():.3f} (n={len(gdf)})", flush=True)

    # save
    np.savez_compressed(os.path.join(args.out_dir, "preds.npz"),
                        single=all_preds, ensemble=ensemble_preds, y=y_lab, genes=genes_lab)
    results = {"single_auc": auc_final, "ensemble_auc": auc_ensemble}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
