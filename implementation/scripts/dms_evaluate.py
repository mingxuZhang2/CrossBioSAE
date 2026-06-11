"""
Step 3: Evaluate DMS benchmark using saved ESM-2 + Evo2 embeddings.
Loads .npz files, runs v6 projection + SAE, computes Spearman benchmarks.
Includes both ridge (linear) and per-assay MLP (nonlinear) evaluation.
"""

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from train_variant_sae import ProjectionLayers, TopKSAE


def cv_spearman(X, y, n_splits=5):
    if X.shape[0] < 20 or np.std(y) < 1e-8:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    for tr, te in kf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    rho, _ = stats.spearmanr(y, preds)
    return rho


class SmallMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d_hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cv_spearman_mlp(X, y, device, n_splits=5, epochs=80, lr=1e-3, batch=2048):
    if X.shape[0] < 50 or np.std(y) < 1e-8:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    for tr, te in kf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=device)
        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=device)
        ytr = yt[tr]

        mlp = SmallMLP(X.shape[1], d_hidden=min(256, X.shape[1])).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        mlp.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=device)
            for b in range(0, len(tr), batch):
                idx = perm[b:b+batch]
                opt.zero_grad()
                loss_fn(mlp(Xtr[idx]), ytr[idx]).backward()
                opt.step()

        mlp.eval()
        with torch.no_grad():
            preds[te] = mlp(Xte).cpu().numpy()

    rho, _ = stats.spearmanr(y, preds)
    return rho


def load_v6_models(sae_dir, device):
    ckpt = torch.load(os.path.join(sae_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    proj = ProjectionLayers().to(device)
    for key in ["proj_state_dict", "proj_state"]:
        if key in ckpt:
            proj.load_state_dict(ckpt[key])
            break
    proj.eval()

    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    cfg = ckpt.get("config", {})
    n_features = sd["encoder.weight"].shape[0]
    sae = TopKSAE(cfg.get("d_input", 1536), n_features, k=cfg.get("k", 32)).to(device)
    sae.load_state_dict(sd)
    sae.eval()
    return proj, sae


def process_assay(esm2_path, args, proj, sae, device):
    """Process a single assay — returns result dict."""
    dms_id = os.path.basename(esm2_path).replace("_esm2.npz", "")
    d = np.load(esm2_path, allow_pickle=True)
    esm_deltas = d["esm2_edelta"]
    y_score = d["y_score"]
    n = len(y_score)

    if n < 30:
        return None

    res = {"assay": dms_id, "n_mutations": n}

    # ── ESM-2 baselines ──
    esm_norm = np.linalg.norm(esm_deltas, axis=1)
    res["rho_esm2_norm"] = stats.spearmanr(y_score, -esm_norm).statistic
    res["rho_esm2_ridge"] = cv_spearman(esm_deltas, y_score)
    res["rho_esm2_mlp"] = cv_spearman_mlp(esm_deltas, y_score, device)

    # ── Check for Evo2 ──
    evo2_path = os.path.join(args.emb_dir, f"{dms_id}_evo2.npz")
    has_evo2 = os.path.exists(evo2_path)

    if has_evo2:
        d2 = np.load(evo2_path, allow_pickle=True)
        evo_deltas = d2["evo2_edelta"]
        evo_valid = d2["evo2_valid"]
        mask = evo_valid.astype(bool)
        n_dual = int(mask.sum())
        res["n_dual"] = n_dual

        if n_dual >= 30:
            evo_norm = np.linalg.norm(evo_deltas[mask], axis=1)
            res["rho_evo2_norm"] = stats.spearmanr(y_score[mask], -evo_norm).statistic
            res["rho_evo2_ridge"] = cv_spearman(evo_deltas[mask], y_score[mask])
            res["rho_evo2_mlp"] = cv_spearman_mlp(evo_deltas[mask], y_score[mask], device)

            # Raw fusion
            concat = np.concatenate([esm_deltas[mask], evo_deltas[mask]], axis=1)
            res["rho_fusion_raw_ridge"] = cv_spearman(concat, y_score[mask])
            res["rho_fusion_raw_mlp"] = cv_spearman_mlp(concat, y_score[mask], device)

            # ── v6 projection ──
            prot_sub = esm_deltas[mask]
            dna_sub = evo_deltas[mask]
            prot_z = (prot_sub - prot_sub.mean(0)) / (prot_sub.std(0) + 1e-8)
            dna_z = (dna_sub - dna_sub.mean(0)) / (dna_sub.std(0) + 1e-8)
            y_dual = y_score[mask]

            with torch.no_grad():
                reps = []
                for b in range(0, n_dual, args.batch_size):
                    xp = torch.tensor(prot_z[b:b+args.batch_size], dtype=torch.float32, device=device)
                    xd = torch.tensor(dna_z[b:b+args.batch_size], dtype=torch.float32, device=device)
                    reps.append(proj(xp, xd).cpu().numpy())
                reps = np.concatenate(reps, axis=0)

            res["rho_v6_rep_ridge"] = cv_spearman(reps, y_dual)
            res["rho_v6_rep_mlp"] = cv_spearman_mlp(reps, y_dual, device)

            # ── SAE ──
            with torch.no_grad():
                sae_acts_list = []
                for b in range(0, n_dual, args.batch_size):
                    x = torch.tensor(reps[b:b+args.batch_size], dtype=torch.float32, device=device)
                    sae_acts_list.append(sae.encode(x).cpu().numpy())
                sae_acts = np.concatenate(sae_acts_list, axis=0)

            active_cols = (sae_acts > 0).any(0)
            res["sae_alive"] = int(active_cols.sum())
            if active_cols.sum() > 10:
                res["rho_sae_ridge"] = cv_spearman(sae_acts[:, active_cols], y_dual)
                res["rho_sae_mlp"] = cv_spearman_mlp(sae_acts[:, active_cols], y_dual, device)

            # Save SAE activations for mechanism analysis
            np.savez_compressed(
                os.path.join(args.out_dir, f"{dms_id}_sae.npz"),
                sae_acts=sae_acts, y_score=y_dual,
                mutants=d["mutants"][mask] if "mutants" in d else np.array([]),
                prot_reps=reps[:, :768], dna_reps=reps[:, 768:],
            )
    else:
        res["n_dual"] = 0

    return res


def worker_fn(rank, n_gpus, esm2_files, args, out_dir):
    """Worker for multi-GPU: each GPU processes its shard of assays."""
    device = torch.device(f"cuda:{rank}")
    proj, sae = load_v6_models(args.sae_dir, device)

    shard = [f for i, f in enumerate(esm2_files) if i % n_gpus == rank]
    results = []
    for esm2_path in shard:
        res = process_assay(esm2_path, args, proj, sae, device)
        if res is None:
            continue
        results.append(res)
        rhos = {k: v for k, v in res.items() if k.startswith("rho")}
        print(f"  [GPU{rank}] {res['assay']}: n={res['n_mutations']}"
              + "".join(f" {k}={v:.3f}" if not np.isnan(v) else f" {k}=N/A"
                        for k, v in rhos.items()),
              flush=True)

    # Save shard results
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"dms_results_shard{rank}.csv"), index=False)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--sae_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/dms_v6")
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    esm2_files = sorted(glob.glob(os.path.join(args.emb_dir, "*_esm2.npz")))
    n_gpus = torch.cuda.device_count()
    print(f"Found {len(esm2_files)} assays, {n_gpus} GPUs", flush=True)

    if n_gpus <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj, sae = load_v6_models(args.sae_dir, device)
        results = []
        for esm2_path in esm2_files:
            res = process_assay(esm2_path, args, proj, sae, device)
            if res is None:
                continue
            results.append(res)
            rhos = {k: v for k, v in res.items() if k.startswith("rho")}
            print(f"  {res['assay']}: n={res['n_mutations']}"
                  + "".join(f" {k}={v:.3f}" if not np.isnan(v) else f" {k}=N/A"
                            for k, v in rhos.items()),
                  flush=True)
    else:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        procs = []
        for rank in range(n_gpus):
            p = mp.Process(target=worker_fn,
                           args=(rank, n_gpus, esm2_files, args, args.out_dir))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        # Merge shard results
        results = []
        for rank in range(n_gpus):
            shard_path = os.path.join(args.out_dir, f"dms_results_shard{rank}.csv")
            if os.path.exists(shard_path):
                results.extend(pd.read_csv(shard_path).to_dict("records"))
                os.remove(shard_path)

    # ── Summary ──
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(args.out_dir, "dms_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("DMS BENCHMARK SUMMARY")
    print("=" * 80)
    for col in sorted(rdf.columns):
        if col.startswith("rho"):
            vals = rdf[col].dropna()
            if len(vals) > 0:
                print(f"  {col:30s}: mean={vals.mean():.4f}  median={vals.median():.4f}  n={len(vals)}")

    n_with_evo2 = (rdf["n_dual"] > 0).sum()
    print(f"\n  Assays with Evo2: {n_with_evo2}/{len(rdf)}")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
