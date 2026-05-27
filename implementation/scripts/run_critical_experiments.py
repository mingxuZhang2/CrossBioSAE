#!/usr/bin/env python3
"""
Five critical experiments to determine if SAE learns real biological representations.
All use existing activations — no retraining needed.

Experiment 1: Raw activation vs SAE vs PCA baseline
Experiment 2: Family-split retrieval (Pfam cluster holdout)
Experiment 3: Hard-negative retrieval (length/composition-matched)
Experiment 4: Held-out GO/Pfam prediction benchmark
Experiment 5: Cross-modal zero-shot transfer (train on protein, test on DNA)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.alignment import CCAAlignment, ProcrustesAlignment, CrossModalRetrieval, AlignmentBenchmark
from src.model import StandardSAE, StandardSAEConfig, CrossBioSAE, CrossBioSAEConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_activations(h5_path):
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
    return acts.astype(np.float32), names


def load_sae_features(model, acts, device, batch_size=512):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = torch.tensor(acts[i:i+batch_size]).to(device)
            feats = model.encode(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def retrieval_recall(features_a, features_b, ks=[1, 5, 10]):
    """Compute retrieval recall@k and MRR. A queries B."""
    n = len(features_a)
    # PCA reduce for speed if high-dim
    if features_a.shape[1] > 512 or features_b.shape[1] > 512:
        dim = min(256, features_a.shape[1], features_b.shape[1], n - 1)
        pca_a = PCA(n_components=dim).fit_transform(features_a)
        pca_b = PCA(n_components=dim).fit_transform(features_b)
    else:
        pca_a, pca_b = features_a, features_b

    dists = cdist(pca_a, pca_b, metric="cosine")
    results = {}
    ranks = []
    for i in range(n):
        sorted_idx = np.argsort(dists[i])
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    for k in ks:
        results[f"R@{k}"] = float(np.mean(ranks <= k))
    results["MRR"] = float(np.mean(1.0 / ranks))
    results["median_rank"] = float(np.median(ranks))
    return results


def align_genes(names_a, names_b, *arrays_a_b):
    """Align two gene lists and corresponding arrays."""
    common = sorted(set(names_a) & set(names_b))
    idx_a = {n: i for i, n in enumerate(names_a)}
    idx_b = {n: i for i, n in enumerate(names_b)}
    ia = [idx_a[n] for n in common]
    ib = [idx_b[n] for n in common]
    result = [common]
    for arr, idx in zip(arrays_a_b, [ia, ib] * (len(arrays_a_b) // 2 + 1)):
        result.append(arr[idx[:len(arr)]] if len(idx) >= len(arr) else arr[idx])
    return result


# ============================================================
# EXPERIMENT 1: Raw activation vs SAE vs PCA baseline
# ============================================================
def experiment_1(prot_raw, dna_raw, prot_sae, dna_sae, test_idx):
    """Compare raw activations, PCA, and SAE features on retrieval."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Raw activation vs SAE vs PCA baseline")
    logger.info("=" * 60)

    prot_raw_test = prot_raw[test_idx]
    dna_raw_test = dna_raw[test_idx]
    prot_sae_test = prot_sae[test_idx]
    dna_sae_test = dna_sae[test_idx]

    # Random projection baseline
    rng = np.random.RandomState(42)
    dim = 256
    rand_proj_p = rng.randn(prot_raw_test.shape[1], dim).astype(np.float32) / np.sqrt(dim)
    rand_proj_d = rng.randn(dna_raw_test.shape[1], dim).astype(np.float32) / np.sqrt(dim)
    prot_rand = prot_raw_test @ rand_proj_p
    dna_rand = dna_raw_test @ rand_proj_d

    # PCA baseline
    pca_dim = min(256, len(test_idx) - 1, prot_raw_test.shape[1], dna_raw_test.shape[1])
    prot_pca = PCA(n_components=pca_dim).fit_transform(prot_raw_test)
    dna_pca = PCA(n_components=pca_dim).fit_transform(dna_raw_test)

    # CCA on raw
    cca_raw = CCAAlignment(n_components=50, pca_dim=256)
    n_train = int(len(test_idx) * 0.7)
    cca_raw.fit(prot_raw_test[:n_train], dna_raw_test[:n_train])
    proj_raw_p, proj_raw_d = cca_raw.transform(prot_raw_test[n_train:], dna_raw_test[n_train:])

    # CCA on SAE
    cca_sae = CCAAlignment(n_components=50, pca_dim=256)
    cca_sae.fit(prot_sae_test[:n_train], dna_sae_test[:n_train])
    proj_sae_p, proj_sae_d = cca_sae.transform(prot_sae_test[n_train:], dna_sae_test[n_train:])

    results = []
    for name, pa, da in [
        ("Random projection", prot_rand[n_train:], dna_rand[n_train:]),
        ("Raw activation (direct)", prot_raw_test[n_train:], dna_raw_test[n_train:]),
        ("PCA (dim=256)", prot_pca[n_train:], dna_pca[n_train:]),
        ("Raw + CCA", proj_raw_p, proj_raw_d),
        ("SAE features (direct)", prot_sae_test[n_train:], dna_sae_test[n_train:]),
        ("SAE + CCA", proj_sae_p, proj_sae_d),
    ]:
        r = retrieval_recall(pa, da)
        r["method"] = name
        results.append(r)
        logger.info(f"  {name:30s}: R@1={r['R@1']:.4f}, R@5={r['R@5']:.4f}, MRR={r['MRR']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# EXPERIMENT 2: Family-split retrieval
# ============================================================
def experiment_2(prot_sae, dna_sae, gene_names, go_annotations):
    """Retrieval with gene-family holdout split."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Family-split retrieval")
    logger.info("=" * 60)

    # Group genes by GO Molecular Function (top-level) as proxy for family
    gene_to_group = {}
    for gene in gene_names:
        if gene in go_annotations:
            terms = go_annotations[gene]
            mf_terms = [t for t in terms if t.startswith("GO:0003") or t.startswith("GO:0004") or t.startswith("GO:0005")]
            if mf_terms:
                gene_to_group[gene] = mf_terms[0][:10]  # use prefix as group
            else:
                gene_to_group[gene] = terms[0][:10] if terms else "unknown"
        else:
            gene_to_group[gene] = "unknown"

    groups = np.array([gene_to_group.get(g, "unknown") for g in gene_names])
    unique_groups = list(set(groups))

    # Hold out 20% of groups
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)
    n_holdout = max(1, len(unique_groups) // 5)
    holdout_groups = set(unique_groups[:n_holdout])
    train_mask = np.array([g not in holdout_groups for g in groups])
    test_mask = ~train_mask

    n_test = test_mask.sum()
    logger.info(f"Family split: {train_mask.sum()} train genes, {n_test} test genes, {n_holdout} held-out groups")

    if n_test < 20:
        logger.warning("Too few test genes for family split. Skipping.")
        return pd.DataFrame()

    # CCA on train, retrieval on test
    cca = CCAAlignment(n_components=50, pca_dim=256)
    cca.fit(prot_sae[train_mask], dna_sae[train_mask])
    proj_p, proj_d = cca.transform(prot_sae[test_mask], dna_sae[test_mask])

    r_family = retrieval_recall(proj_p, proj_d)
    r_family["method"] = "SAE + CCA (family-split)"

    # Compare with random split of same size
    rand_idx = rng.permutation(len(gene_names))
    rand_test = rand_idx[:n_test]
    rand_train = rand_idx[n_test:]
    cca_rand = CCAAlignment(n_components=50, pca_dim=256)
    cca_rand.fit(prot_sae[rand_train], dna_sae[rand_train])
    proj_rp, proj_rd = cca_rand.transform(prot_sae[rand_test], dna_sae[rand_test])
    r_random = retrieval_recall(proj_rp, proj_rd)
    r_random["method"] = "SAE + CCA (random-split)"

    results = [r_family, r_random]
    for r in results:
        logger.info(f"  {r['method']:40s}: R@1={r['R@1']:.4f}, R@5={r['R@5']:.4f}, MRR={r['MRR']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# EXPERIMENT 3: Hard-negative retrieval
# ============================================================
def experiment_3(prot_sae, dna_sae, prot_raw, dna_raw, gene_pairs):
    """Retrieval with length/composition-matched hard negatives."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Hard-negative retrieval (length-matched)")
    logger.info("=" * 60)

    gene_map = {p["gene_name"]: p for p in gene_pairs}

    # Compute confound features per gene
    lengths = []
    for g in [p["gene_name"] for p in gene_pairs]:
        if g in gene_map:
            lengths.append(gene_map[g].get("protein_length", 0))
        else:
            lengths.append(0)

    lengths = np.array(lengths[:len(prot_sae)], dtype=np.float32)

    # Bin genes by protein length
    n_bins = 20
    bins = np.percentile(lengths[lengths > 0], np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(lengths, bins)

    # For each gene, restrict retrieval gallery to same length bin
    n = min(len(prot_sae), len(dna_sae), len(lengths))
    pca_dim = min(256, n - 1, prot_sae.shape[1], dna_sae.shape[1])
    prot_pca = PCA(n_components=pca_dim).fit_transform(prot_sae[:n])
    dna_pca = PCA(n_components=pca_dim).fit_transform(dna_sae[:n])

    # Sample test set
    rng = np.random.RandomState(42)
    test_size = min(1000, n // 3)
    test_idx = rng.choice(n, test_size, replace=False)

    ranks_hard = []
    ranks_easy = []

    for qi in test_idx:
        q_bin = bin_idx[qi]
        # Hard gallery: same length bin
        gallery_hard = np.where(bin_idx[:n] == q_bin)[0]
        if len(gallery_hard) < 5:
            continue

        # Distances to hard gallery
        dists = cdist(prot_pca[qi:qi+1], dna_pca[gallery_hard], metric="cosine")[0]
        target_pos = np.where(gallery_hard == qi)[0]
        if len(target_pos) == 0:
            continue
        sorted_idx = np.argsort(dists)
        rank_hard = np.where(sorted_idx == target_pos[0])[0][0] + 1
        ranks_hard.append(rank_hard)

        # Easy: full gallery
        dists_full = cdist(prot_pca[qi:qi+1], dna_pca[:n], metric="cosine")[0]
        sorted_full = np.argsort(dists_full)
        rank_easy = np.where(sorted_full == qi)[0][0] + 1
        ranks_easy.append(rank_easy)

    ranks_hard = np.array(ranks_hard)
    ranks_easy = np.array(ranks_easy)

    results = []
    for name, ranks in [("Full gallery (easy)", ranks_easy), ("Length-matched gallery (hard)", ranks_hard)]:
        r = {
            "method": name,
            "R@1": float(np.mean(ranks <= 1)),
            "R@5": float(np.mean(ranks <= 5)),
            "MRR": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "n_queries": len(ranks),
        }
        results.append(r)
        logger.info(f"  {name:40s}: R@1={r['R@1']:.4f}, R@5={r['R@5']:.4f}, MRR={r['MRR']:.4f} (n={len(ranks)})")

    return pd.DataFrame(results)


# ============================================================
# EXPERIMENT 4: Held-out GO prediction
# ============================================================
def experiment_4(prot_raw, dna_raw, prot_sae, dna_sae, gene_names, go_annotations):
    """Held-out Gene Ontology function prediction benchmark."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Held-out GO function prediction")
    logger.info("=" * 60)

    # Filter genes with GO annotations
    annotated_genes = [g for g in gene_names if g in go_annotations and len(go_annotations[g]) > 0]
    annotated_idx = [gene_names.index(g) for g in annotated_genes if g in gene_names]

    if len(annotated_idx) < 100:
        logger.warning("Too few annotated genes. Skipping.")
        return pd.DataFrame()

    # Get top-50 most frequent GO terms for multi-label classification
    all_terms = []
    for g in annotated_genes:
        all_terms.extend(go_annotations[g])
    from collections import Counter
    term_counts = Counter(all_terms)
    top_terms = [t for t, c in term_counts.most_common(50)]

    # Build label matrix
    labels = np.zeros((len(annotated_idx), len(top_terms)), dtype=np.int32)
    for i, g in enumerate(annotated_genes):
        for j, t in enumerate(top_terms):
            if t in go_annotations[g]:
                labels[i, j] = 1

    # Features for annotated genes
    feat_sets = {
        "ESM-2 raw": prot_raw[annotated_idx],
        "NT raw": dna_raw[annotated_idx],
        "ESM-2+NT concat raw": np.hstack([prot_raw[annotated_idx], dna_raw[annotated_idx]]),
        "Protein SAE": prot_sae[annotated_idx],
        "DNA SAE": dna_sae[annotated_idx],
        "Protein+DNA SAE concat": np.hstack([prot_sae[annotated_idx], dna_sae[annotated_idx]]),
    }

    # PCA reduce all to same dim for fair comparison
    pca_dim = 256

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for feat_name, X in feat_sets.items():
        # PCA reduce
        actual_dim = min(pca_dim, X.shape[1], X.shape[0] - 1)
        if X.shape[1] > actual_dim:
            X_reduced = PCA(n_components=actual_dim).fit_transform(X)
        else:
            X_reduced = X

        X_scaled = StandardScaler().fit_transform(X_reduced)

        # Per-term AUROC via cross-validation
        term_aurocs = []
        for j in range(len(top_terms)):
            y = labels[:, j]
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            try:
                clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
                # 5-fold CV
                preds = np.zeros(len(y))
                for train_i, test_i in cv.split(X_scaled, y):
                    clf.fit(X_scaled[train_i], y[train_i])
                    preds[test_i] = clf.predict_proba(X_scaled[test_i])[:, 1]
                auroc = roc_auc_score(y, preds)
                term_aurocs.append(auroc)
            except Exception:
                continue

        if term_aurocs:
            mean_auroc = np.mean(term_aurocs)
            results.append({
                "method": feat_name,
                "mean_GO_AUROC": mean_auroc,
                "n_terms_evaluated": len(term_aurocs),
            })
            logger.info(f"  {feat_name:30s}: mean GO AUROC={mean_auroc:.4f} ({len(term_aurocs)} terms)")

    return pd.DataFrame(results)


# ============================================================
# EXPERIMENT 5: Cross-modal zero-shot transfer
# ============================================================
def experiment_5(prot_sae, dna_sae, gene_names, go_annotations):
    """Train classifier on protein SAE, test on CCA-aligned DNA SAE."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Cross-modal zero-shot transfer")
    logger.info("=" * 60)

    annotated_genes = [g for g in gene_names if g in go_annotations and len(go_annotations[g]) > 0]
    annotated_idx = [gene_names.index(g) for g in annotated_genes]

    if len(annotated_idx) < 100:
        logger.warning("Too few annotated genes.")
        return pd.DataFrame()

    from collections import Counter
    all_terms = []
    for g in annotated_genes:
        all_terms.extend(go_annotations[g])
    top_terms = [t for t, _ in Counter(all_terms).most_common(30)]

    labels = np.zeros((len(annotated_idx), len(top_terms)), dtype=np.int32)
    for i, g in enumerate(annotated_genes):
        for j, t in enumerate(top_terms):
            if t in go_annotations[g]:
                labels[i, j] = 1

    prot_feat = prot_sae[annotated_idx]
    dna_feat = dna_sae[annotated_idx]

    # Train/test split
    rng = np.random.RandomState(42)
    n = len(annotated_idx)
    idx = rng.permutation(n)
    n_train = int(n * 0.7)
    train_i, test_i = idx[:n_train], idx[n_train:]

    # CCA alignment fitted on train
    cca = CCAAlignment(n_components=50, pca_dim=256)
    cca.fit(prot_feat[train_i], dna_feat[train_i])
    proj_prot_train, _ = cca.transform(prot_feat[train_i], dna_feat[train_i])
    proj_prot_test, proj_dna_test = cca.transform(prot_feat[test_i], dna_feat[test_i])

    results = []
    for term_j in range(len(top_terms)):
        y_train = labels[train_i, term_j]
        y_test = labels[test_i, term_j]
        if y_train.sum() < 5 or y_test.sum() < 3:
            continue

        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")

        # Train on protein CCA features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(proj_prot_train)
        clf.fit(X_train, y_train)

        # Test on protein CCA features (same modality)
        X_test_prot = scaler.transform(proj_prot_test)
        pred_same = clf.predict_proba(X_test_prot)[:, 1]

        # Test on DNA CCA features (cross-modal transfer!)
        X_test_dna = scaler.transform(proj_dna_test)
        pred_cross = clf.predict_proba(X_test_dna)[:, 1]

        try:
            auroc_same = roc_auc_score(y_test, pred_same)
            auroc_cross = roc_auc_score(y_test, pred_cross)
            results.append({
                "go_term": top_terms[term_j],
                "auroc_same_modality": auroc_same,
                "auroc_cross_modal": auroc_cross,
                "transfer_ratio": auroc_cross / max(auroc_same, 0.5),
            })
        except Exception:
            continue

    if results:
        df = pd.DataFrame(results)
        mean_same = df["auroc_same_modality"].mean()
        mean_cross = df["auroc_cross_modal"].mean()
        mean_ratio = df["transfer_ratio"].mean()
        logger.info(f"  Same-modality mean AUROC:  {mean_same:.4f}")
        logger.info(f"  Cross-modal mean AUROC:    {mean_cross:.4f}")
        logger.info(f"  Transfer ratio:            {mean_ratio:.4f}")
        logger.info(f"  ({len(df)} GO terms evaluated)")
        return df

    return pd.DataFrame()


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

    # Load raw activations
    logger.info("Loading raw activations...")
    prot_raw, prot_names = load_activations(paths["protein_activations"])
    dna_raw, dna_names = load_activations(paths["dna_activations"])

    # Align gene order
    common = sorted(set(prot_names) & set(dna_names))
    prot_idx = {n: i for i, n in enumerate(prot_names)}
    dna_idx = {n: i for i, n in enumerate(dna_names)}
    ci_p = [prot_idx[n] for n in common]
    ci_d = [dna_idx[n] for n in common]
    prot_raw = prot_raw[ci_p]
    dna_raw = dna_raw[ci_d]
    gene_names = common
    logger.info(f"Aligned: {len(gene_names)} genes")

    # Load independent SAE models and extract features
    logger.info("Loading SAE models and extracting features...")
    prot_sae_cfg = StandardSAEConfig(
        dim_input=config["protein_model"]["hidden_dim"],
        expansion_factor=config.get("independent_sae", {}).get("expansion_factor", config["model"]["expansion_factor"]),
        topk_k=config.get("independent_sae", {}).get("topk_k", config["model"].get("topk_k", 64)),
    )
    dna_sae_cfg = StandardSAEConfig(
        dim_input=config["dna_model"]["hidden_dim"],
        expansion_factor=config.get("independent_sae", {}).get("expansion_factor", config["model"]["expansion_factor"]),
        topk_k=config.get("independent_sae", {}).get("topk_k", config["model"].get("topk_k", 64)),
    )

    prot_sae_model = StandardSAE(prot_sae_cfg).to(device)
    ckpt = torch.load(args.protein_sae_ckpt, map_location=device, weights_only=True)
    prot_sae_model.load_state_dict(ckpt["model_state_dict"])

    dna_sae_model = StandardSAE(dna_sae_cfg).to(device)
    ckpt = torch.load(args.dna_sae_ckpt, map_location=device, weights_only=True)
    dna_sae_model.load_state_dict(ckpt["model_state_dict"])

    prot_sae = load_sae_features(prot_sae_model, prot_raw, device)
    dna_sae = load_sae_features(dna_sae_model, dna_raw, device)
    logger.info(f"SAE features: protein={prot_sae.shape}, DNA={dna_sae.shape}")

    # Load GO annotations
    go_annotations = {}
    go_path = args.go_annotations or paths.get("go_annotations", "data/full/go_annotations.json")
    if Path(go_path).exists():
        with open(go_path) as f:
            go_annotations = json.load(f)
        logger.info(f"Loaded GO annotations: {len(go_annotations)} genes")

    # Load gene pairs for length info
    data_dir = config["data"]["data_dir"]
    split = config["data"]["split"]
    gene_pairs_path = f"{data_dir}/gene_pairs_{split}.json"
    gene_pairs = []
    if Path(gene_pairs_path).exists():
        with open(gene_pairs_path) as f:
            gene_pairs = json.load(f)

    # Train/test split (shared across experiments)
    rng = np.random.RandomState(42)
    n = len(gene_names)
    all_idx = rng.permutation(n)
    test_idx = all_idx[:n // 5]

    output_dir = Path(paths.get("output_dir", "results/full")) / "critical_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all experiments
    all_results = {}

    df1 = experiment_1(prot_raw, dna_raw, prot_sae, dna_sae, test_idx)
    all_results["exp1_raw_vs_sae"] = df1
    df1.to_csv(output_dir / "exp1_raw_vs_sae.csv", index=False)

    if go_annotations:
        df2 = experiment_2(prot_sae, dna_sae, gene_names, go_annotations)
        all_results["exp2_family_split"] = df2
        df2.to_csv(output_dir / "exp2_family_split.csv", index=False)

    if gene_pairs:
        df3 = experiment_3(prot_sae, dna_sae, prot_raw, dna_raw, gene_pairs)
        all_results["exp3_hard_negative"] = df3
        df3.to_csv(output_dir / "exp3_hard_negative.csv", index=False)

    if go_annotations:
        df4 = experiment_4(prot_raw, dna_raw, prot_sae, dna_sae, gene_names, go_annotations)
        all_results["exp4_go_prediction"] = df4
        df4.to_csv(output_dir / "exp4_go_prediction.csv", index=False)

    if go_annotations:
        df5 = experiment_5(prot_sae, dna_sae, gene_names, go_annotations)
        all_results["exp5_cross_modal_transfer"] = df5
        df5.to_csv(output_dir / "exp5_cross_modal_transfer.csv", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("CRITICAL EXPERIMENTS SUMMARY")
    print("=" * 70)
    for name, df in all_results.items():
        if len(df) > 0:
            print(f"\n--- {name} ---")
            print(df.to_string(index=False))
    print("=" * 70)

    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
