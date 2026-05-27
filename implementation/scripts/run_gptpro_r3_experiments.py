#!/usr/bin/env python3
"""
GPT Pro Round 3 experiments: fair benchmark + hard-negatives + low-label + transfer.

Exp A: Fair raw-vs-SAE CCA benchmark (PCA sweep + SAE reconstruction)
Exp B: Hard-negative retrieval WITH CCA
Exp C: Low-label GO prediction (1%/5%/10%/25%/100%)
Exp D: Leakage-free cross-modal transfer with baseline correction
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.alignment import CCAAlignment
from src.model import StandardSAE, StandardSAEConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_h5(path):
    with h5py.File(path, "r") as f:
        acts = f["activations"][:].astype(np.float32)
        names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
    return acts, names


def align_genes(*name_act_pairs):
    """Align multiple (names, acts) pairs to common gene set."""
    common = set(name_act_pairs[0][0])
    for names, _ in name_act_pairs[1:]:
        common &= set(names)
    common = sorted(common)
    results = [common]
    for names, acts in name_act_pairs:
        idx = {n: i for i, n in enumerate(names)}
        results.append(acts[[idx[n] for n in common]])
    return results


def sae_encode(model, acts, device, batch_size=512):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            b = torch.tensor(acts[i:i+batch_size]).to(device)
            feats.append(model.encode(b).cpu().numpy())
    return np.concatenate(feats)


def sae_reconstruct(model, acts, device, batch_size=512):
    model.eval()
    recons = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            b = torch.tensor(acts[i:i+batch_size]).to(device)
            r, _ = model(b)
            recons.append(r.cpu().numpy())
    return np.concatenate(recons)


def retrieval_metrics(fa, fb, ks=[1, 5, 10]):
    n = len(fa)
    dists = cdist(fa, fb, metric="cosine")
    ranks = []
    for i in range(n):
        rank = np.where(np.argsort(dists[i]) == i)[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    r = {f"R@{k}": float(np.mean(ranks <= k)) for k in ks}
    r["MRR"] = float(np.mean(1.0 / ranks))
    return r


def cca_retrieval(train_a, train_b, test_a, test_b, n_components=50, pca_dim=256):
    cca = CCAAlignment(n_components=n_components, pca_dim=pca_dim)
    cca.fit(train_a, train_b)
    pa, pb = cca.transform(test_a, test_b)
    return retrieval_metrics(pa, pb)


# ============================================================
# EXP A: Fair raw-vs-SAE CCA benchmark
# ============================================================
def exp_a_fair_benchmark(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                         prot_sae_recon, dna_sae_recon, train_idx, test_idx):
    logger.info("=" * 60)
    logger.info("EXP A: Fair raw-vs-SAE CCA benchmark (PCA sweep)")
    logger.info("=" * 60)

    results = []
    for cca_dim in [32, 50, 128, 256]:
        for pca_pre in [64, 128, 256, 512]:
            for name, pa, da in [
                ("Raw", prot_raw, dna_raw),
                ("SAE features", prot_sae_feats, dna_sae_feats),
                ("SAE reconstruction", prot_sae_recon, dna_sae_recon),
            ]:
                try:
                    r = cca_retrieval(
                        pa[train_idx], da[train_idx],
                        pa[test_idx], da[test_idx],
                        n_components=cca_dim, pca_dim=pca_pre,
                    )
                    r["representation"] = name
                    r["cca_dim"] = cca_dim
                    r["pca_dim"] = pca_pre
                    results.append(r)
                except Exception as e:
                    logger.warning(f"{name} cca={cca_dim} pca={pca_pre}: {e}")

    df = pd.DataFrame(results)

    # Summary: best config per representation
    logger.info("\nBest R@1 per representation:")
    for name in ["Raw", "SAE features", "SAE reconstruction"]:
        subset = df[df["representation"] == name]
        if len(subset) > 0:
            best = subset.loc[subset["R@1"].idxmax()]
            logger.info(f"  {name:25s}: R@1={best['R@1']:.4f} (pca={int(best['pca_dim'])}, cca={int(best['cca_dim'])})")

    return df


# ============================================================
# EXP B: Hard-negative retrieval WITH CCA
# ============================================================
def exp_b_hard_negative_cca(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                            gene_pairs, gene_names, train_idx, test_idx):
    logger.info("=" * 60)
    logger.info("EXP B: Hard-negative retrieval WITH CCA")
    logger.info("=" * 60)

    gene_map = {p["gene_name"]: p for p in gene_pairs}
    lengths = np.array([gene_map.get(g, {}).get("protein_length", 300) for g in gene_names])

    n_bins = 10
    bins = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(lengths, bins)

    results = []
    for name, pa, da in [
        ("Raw + CCA", prot_raw, dna_raw),
        ("SAE + CCA", prot_sae_feats, dna_sae_feats),
    ]:
        # Fit CCA on train
        cca = CCAAlignment(n_components=50, pca_dim=256)
        cca.fit(pa[train_idx], da[train_idx])
        proj_p, proj_d = cca.transform(pa[test_idx], da[test_idx])

        # Easy retrieval (full gallery)
        r_easy = retrieval_metrics(proj_p, proj_d)
        r_easy["method"] = f"{name} (full gallery)"
        results.append(r_easy)

        # Hard retrieval (length-matched gallery)
        test_bins = bin_idx[test_idx]
        ranks_hard = []
        for qi in range(len(test_idx)):
            q_bin = test_bins[qi]
            gallery = np.where(test_bins == q_bin)[0]
            if len(gallery) < 5:
                continue
            dists = cdist(proj_p[qi:qi+1], proj_d[gallery], metric="cosine")[0]
            target = np.where(gallery == qi)[0]
            if len(target) == 0:
                continue
            rank = np.where(np.argsort(dists) == target[0])[0][0] + 1
            ranks_hard.append(rank)

        if ranks_hard:
            ranks_hard = np.array(ranks_hard)
            r_hard = {
                "R@1": float(np.mean(ranks_hard <= 1)),
                "R@5": float(np.mean(ranks_hard <= 5)),
                "MRR": float(np.mean(1.0 / ranks_hard)),
                "method": f"{name} (length-matched)",
                "n_queries": len(ranks_hard),
            }
            results.append(r_hard)

        logger.info(f"  {results[-2]['method']:35s}: R@1={results[-2]['R@1']:.4f}")
        if len(results) > 1 and "length-matched" in results[-1].get("method", ""):
            logger.info(f"  {results[-1]['method']:35s}: R@1={results[-1]['R@1']:.4f} (n={results[-1].get('n_queries', '?')})")

    return pd.DataFrame(results)


# ============================================================
# EXP C: Low-label GO prediction
# ============================================================
def exp_c_low_label(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                    gene_names, go_annotations):
    logger.info("=" * 60)
    logger.info("EXP C: Low-label GO prediction")
    logger.info("=" * 60)

    # Get annotated genes and top GO terms
    annotated = [g for g in gene_names if g in go_annotations and go_annotations[g]]
    ann_idx = [gene_names.index(g) for g in annotated]

    all_terms = []
    for g in annotated:
        all_terms.extend(go_annotations[g])
    top_terms = [t for t, _ in Counter(all_terms).most_common(30)]

    labels = np.zeros((len(ann_idx), len(top_terms)), dtype=np.int32)
    for i, g in enumerate(annotated):
        for j, t in enumerate(top_terms):
            if t in go_annotations[g]:
                labels[i, j] = 1

    feat_sets = {
        "ESM-2 raw": prot_raw[ann_idx],
        "NT raw": dna_raw[ann_idx],
        "Protein SAE": prot_sae_feats[ann_idx],
        "DNA SAE": dna_sae_feats[ann_idx],
        "Prot+DNA SAE": np.hstack([prot_sae_feats[ann_idx], dna_sae_feats[ann_idx]]),
        "ESM+NT raw": np.hstack([prot_raw[ann_idx], dna_raw[ann_idx]]),
    }

    label_fractions = [0.01, 0.05, 0.10, 0.25, 1.0]
    results = []

    for feat_name, X in feat_sets.items():
        pca_dim = min(256, X.shape[1], X.shape[0] - 1)
        if X.shape[1] > pca_dim:
            X = PCA(n_components=pca_dim).fit_transform(X)
        X = StandardScaler().fit_transform(X)

        for frac in label_fractions:
            term_aurocs = []
            for j in range(len(top_terms)):
                y = labels[:, j]
                if y.sum() < 10 or (1 - y).sum() < 10:
                    continue

                rng = np.random.RandomState(42)
                n = len(y)
                idx = rng.permutation(n)
                n_train = max(20, int(n * frac * 0.8))
                n_test = n - int(n * 0.8)
                train_i = idx[:n_train]
                test_i = idx[int(n * 0.8):]

                try:
                    clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
                    clf.fit(X[train_i], y[train_i])
                    pred = clf.predict_proba(X[test_i])[:, 1]
                    auroc = roc_auc_score(y[test_i], pred)
                    term_aurocs.append(auroc)
                except Exception:
                    continue

            if term_aurocs:
                results.append({
                    "features": feat_name,
                    "label_fraction": frac,
                    "mean_AUROC": np.mean(term_aurocs),
                    "n_terms": len(term_aurocs),
                })

    df = pd.DataFrame(results)

    # Print summary table
    logger.info("\nLow-label GO prediction (mean AUROC):")
    pivot = df.pivot_table(index="features", columns="label_fraction", values="mean_AUROC")
    logger.info(f"\n{pivot.to_string()}")

    return df


# ============================================================
# EXP D: Leakage-free cross-modal transfer
# ============================================================
def exp_d_transfer(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                   gene_names, go_annotations):
    logger.info("=" * 60)
    logger.info("EXP D: Leakage-free cross-modal transfer")
    logger.info("=" * 60)

    annotated = [g for g in gene_names if g in go_annotations and go_annotations[g]]
    ann_idx = [gene_names.index(g) for g in annotated]

    all_terms = []
    for g in annotated:
        all_terms.extend(go_annotations[g])
    top_terms = [t for t, _ in Counter(all_terms).most_common(30)]

    labels = np.zeros((len(ann_idx), len(top_terms)), dtype=np.int32)
    for i, g in enumerate(annotated):
        for j, t in enumerate(top_terms):
            if t in go_annotations[g]:
                labels[i, j] = 1

    # Strict 3-way split
    rng = np.random.RandomState(42)
    n = len(ann_idx)
    idx = rng.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_i = idx[:n_train]
    val_i = idx[n_train:n_train + n_val]
    test_i = idx[n_train + n_val:]

    logger.info(f"Split: {len(train_i)} train, {len(val_i)} val, {len(test_i)} test")

    results = []
    for rep_name, prot_feat, dna_feat in [
        ("Raw + CCA", prot_raw[ann_idx], dna_raw[ann_idx]),
        ("SAE + CCA", prot_sae_feats[ann_idx], dna_sae_feats[ann_idx]),
    ]:
        # CCA fit ONLY on train
        cca = CCAAlignment(n_components=50, pca_dim=256)
        cca.fit(prot_feat[train_i], dna_feat[train_i])

        proj_prot_train, _ = cca.transform(prot_feat[train_i], dna_feat[train_i])
        proj_prot_test, proj_dna_test = cca.transform(prot_feat[test_i], dna_feat[test_i])

        for j in range(len(top_terms)):
            y_train = labels[train_i, j]
            y_test = labels[test_i, j]
            if y_train.sum() < 5 or y_test.sum() < 3 or (1 - y_test).sum() < 3:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(proj_prot_train)

            clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
            clf.fit(X_train, y_train)

            # Same modality
            X_test_prot = scaler.transform(proj_prot_test)
            pred_same = clf.predict_proba(X_test_prot)[:, 1]

            # Cross-modal
            X_test_dna = scaler.transform(proj_dna_test)
            pred_cross = clf.predict_proba(X_test_dna)[:, 1]

            # DNA-only baseline (train on DNA, test on DNA)
            proj_dna_train, _ = cca.transform(prot_feat[train_i], dna_feat[train_i])
            # Actually use DNA projections for DNA baseline
            _, dna_proj_train = cca.transform(prot_feat[train_i], dna_feat[train_i])
            _, dna_proj_test_only = cca.transform(prot_feat[test_i], dna_feat[test_i])
            scaler_d = StandardScaler()
            X_dna_train = scaler_d.fit_transform(dna_proj_train)
            clf_d = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
            clf_d.fit(X_dna_train, y_train)
            pred_dna_only = clf_d.predict_proba(scaler_d.transform(dna_proj_test_only))[:, 1]

            try:
                auc_same = roc_auc_score(y_test, pred_same)
                auc_cross = roc_auc_score(y_test, pred_cross)
                auc_dna_only = roc_auc_score(y_test, pred_dna_only)

                # Corrected transfer efficiency
                excess_same = auc_same - 0.5
                excess_cross = auc_cross - 0.5
                corrected_ratio = excess_cross / max(excess_same, 0.01)

                results.append({
                    "representation": rep_name,
                    "go_term": top_terms[j],
                    "auc_protein_same": auc_same,
                    "auc_cross_modal": auc_cross,
                    "auc_dna_only": auc_dna_only,
                    "corrected_transfer": corrected_ratio,
                })
            except Exception:
                continue

    df = pd.DataFrame(results)

    if len(df) > 0:
        for rep in df["representation"].unique():
            sub = df[df["representation"] == rep]
            logger.info(f"\n{rep}:")
            logger.info(f"  Protein→Protein (same):   {sub['auc_protein_same'].mean():.4f}")
            logger.info(f"  Protein→DNA (cross):      {sub['auc_cross_modal'].mean():.4f}")
            logger.info(f"  DNA→DNA (baseline):       {sub['auc_dna_only'].mean():.4f}")
            logger.info(f"  Corrected transfer ratio: {sub['corrected_transfer'].mean():.4f}")
            logger.info(f"  ({len(sub)} GO terms)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protein_sae_ckpt", required=True)
    parser.add_argument("--dna_sae_ckpt", required=True)
    parser.add_argument("--go_annotations", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    paths = config["paths"]

    # Load everything
    logger.info("Loading data...")
    prot_raw, prot_names = load_h5(paths["protein_activations"])
    dna_raw, dna_names = load_h5(paths["dna_activations"])
    gene_names, prot_raw, dna_raw = align_genes((prot_names, prot_raw), (dna_names, dna_raw))
    logger.info(f"Aligned: {len(gene_names)} genes, prot={prot_raw.shape}, dna={dna_raw.shape}")

    # Load SAE models
    prot_cfg = StandardSAEConfig(
        dim_input=config["protein_model"]["hidden_dim"],
        expansion_factor=config.get("independent_sae", {}).get("expansion_factor", config["model"]["expansion_factor"]),
        topk_k=config.get("independent_sae", {}).get("topk_k", config["model"].get("topk_k", 64)),
    )
    dna_cfg = StandardSAEConfig(
        dim_input=config["dna_model"]["hidden_dim"],
        expansion_factor=config.get("independent_sae", {}).get("expansion_factor", config["model"]["expansion_factor"]),
        topk_k=config.get("independent_sae", {}).get("topk_k", config["model"].get("topk_k", 64)),
    )

    prot_sae = StandardSAE(prot_cfg).to(device)
    prot_sae.load_state_dict(torch.load(args.protein_sae_ckpt, map_location=device, weights_only=True)["model_state_dict"])
    dna_sae = StandardSAE(dna_cfg).to(device)
    dna_sae.load_state_dict(torch.load(args.dna_sae_ckpt, map_location=device, weights_only=True)["model_state_dict"])

    prot_sae_feats = sae_encode(prot_sae, prot_raw, device)
    dna_sae_feats = sae_encode(dna_sae, dna_raw, device)
    prot_sae_recon = sae_reconstruct(prot_sae, prot_raw, device)
    dna_sae_recon = sae_reconstruct(dna_sae, dna_raw, device)
    logger.info(f"SAE features: prot={prot_sae_feats.shape}, dna={dna_sae_feats.shape}")
    logger.info(f"SAE recon: prot={prot_sae_recon.shape}, dna={dna_sae_recon.shape}")

    # Load GO + gene pairs
    go_annotations = {}
    go_path = args.go_annotations or paths.get("go_annotations", "")
    if Path(go_path).exists():
        with open(go_path) as f:
            go_annotations = json.load(f)

    gene_pairs = []
    gp_path = f"{config['data']['data_dir']}/gene_pairs_{config['data']['split']}.json"
    if Path(gp_path).exists():
        with open(gp_path) as f:
            gene_pairs = json.load(f)

    # Train/test split
    rng = np.random.RandomState(42)
    n = len(gene_names)
    idx = rng.permutation(n)
    train_idx = idx[:int(n * 0.8)]
    test_idx = idx[int(n * 0.8):]

    output_dir = Path(paths.get("output_dir", "results/full")) / "gptpro_r3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all experiments
    df_a = exp_a_fair_benchmark(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                                prot_sae_recon, dna_sae_recon, train_idx, test_idx)
    df_a.to_csv(output_dir / "exp_a_fair_benchmark.csv", index=False)

    if gene_pairs:
        df_b = exp_b_hard_negative_cca(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                                       gene_pairs, gene_names, train_idx, test_idx)
        df_b.to_csv(output_dir / "exp_b_hard_negative.csv", index=False)

    if go_annotations:
        df_c = exp_c_low_label(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                               gene_names, go_annotations)
        df_c.to_csv(output_dir / "exp_c_low_label.csv", index=False)

        df_d = exp_d_transfer(prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                              gene_names, go_annotations)
        df_d.to_csv(output_dir / "exp_d_transfer.csv", index=False)

    logger.info(f"\nAll results saved to {output_dir}")
    logger.info("=" * 60)
    logger.info("GPT PRO R3 EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
