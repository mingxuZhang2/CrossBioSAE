"""
Cross-modal prediction tasks: test whether protein+DNA combined embeddings
outperform single-modality on tasks that inherently need both modalities.

Tasks:
1. Gene constraint (LOEUF) regression — gnomAD
2. Gene essentiality classification — DepMap
3. dN/dS prediction — Ensembl
4. Gene age classification — phylostratigraphy

For each task, compare:
- Protein-only (ESM-2 raw)
- DNA-only (NT raw)
- Concat (protein + DNA raw)
- CCA-fused (CCA protein + CCA DNA)
- CCA-shared (mean of CCA protein + CCA DNA)
- CCA consistency (per-gene cosine similarity in CCA space — scalar feature)
- Protein + CCA consistency (protein + consistency as extra feature)
"""

import argparse
import gzip
import json
import logging
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_activations(h5_path):
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
    return acts, names


def load_gnomad_loeuf(path):
    """Load gnomAD LOEUF scores. Returns dict: gene_symbol -> LOEUF."""
    labels = {}
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split("\t")
        gene_col = header.index("gene")
        loeuf_col = header.index("oe_lof_upper")
        pli_col = header.index("pLI")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(gene_col, loeuf_col):
                gene = parts[gene_col]
                loeuf = parts[loeuf_col]
                if loeuf and loeuf != "NA" and gene:
                    try:
                        labels[gene] = {
                            "loeuf": float(loeuf),
                            "pli": float(parts[pli_col]) if parts[pli_col] != "NA" else None,
                        }
                    except ValueError:
                        pass
    return labels


def load_depmap_essentiality(essentials_path, nonessentials_path):
    """Load DepMap essential/nonessential gene lists."""
    labels = {}
    with open(essentials_path) as f:
        for line in f:
            gene = line.strip().split(",")[0].split(" (")[0]
            if gene and gene != "gene":
                labels[gene] = 1
    with open(nonessentials_path) as f:
        for line in f:
            gene = line.strip().split(",")[0].split(" (")[0]
            if gene and gene != "gene":
                labels[gene] = 0
    return labels


def load_dnds(path):
    """Load dN/dS values. Returns dict: gene_symbol -> {dn, ds, dnds}."""
    labels = {}
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        gene = row.get("gene_symbol") or row.get("external_gene_name")
        dn = row.get("dn") or row.get("mmusculus_homolog_dn")
        ds = row.get("ds") or row.get("mmusculus_homolog_ds")
        if pd.notna(gene) and pd.notna(dn) and pd.notna(ds) and ds > 0:
            labels[gene] = {"dn": float(dn), "ds": float(ds), "dnds": float(dn) / float(ds)}
    return labels


def load_gene_age(path):
    """Load gene age (phylostratum). Returns dict: gene_symbol -> phylostratum."""
    labels = {}
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        gene = row.get("gene_symbol") or row.get("symbol")
        ps = row.get("phylostratum") or row.get("ps")
        if pd.notna(gene) and pd.notna(ps):
            labels[gene] = int(ps)
    return labels


def build_representations(prot_acts, dna_acts, pca_dim=256, cca_dim=64):
    """Build all representation variants from aligned protein and DNA activations."""
    n = prot_acts.shape[0]

    # PCA
    pca_target = min(pca_dim, n - 1, prot_acts.shape[1], dna_acts.shape[1])
    pca_prot = PCA(n_components=pca_target).fit_transform(prot_acts)
    pca_dna = PCA(n_components=pca_target).fit_transform(dna_acts)

    # CCA
    scaler_p = StandardScaler()
    scaler_d = StandardScaler()
    prot_scaled = scaler_p.fit_transform(pca_prot)
    dna_scaled = scaler_d.fit_transform(pca_dna)

    n_cca = min(cca_dim, pca_target, n - 1)
    cca = CCA(n_components=n_cca, max_iter=2000)
    cca.fit(prot_scaled, dna_scaled)
    cca_prot, cca_dna = cca.transform(prot_scaled, dna_scaled)

    # CCA consistency: per-gene cosine similarity
    norms_p = np.linalg.norm(cca_prot, axis=1, keepdims=True) + 1e-8
    norms_d = np.linalg.norm(cca_dna, axis=1, keepdims=True) + 1e-8
    consistency = (cca_prot / norms_p * cca_dna / norms_d).sum(axis=1, keepdims=True)

    representations = {
        "Protein-only": prot_acts,
        "DNA-only": dna_acts,
        "Concat": np.hstack([prot_acts, dna_acts]),
        "CCA-fused": np.hstack([cca_prot, cca_dna]),
        "CCA-shared": (cca_prot + cca_dna) / 2.0,
        "Consistency-only": consistency,
        "Protein+consistency": np.hstack([prot_acts, consistency]),
        "Concat+consistency": np.hstack([prot_acts, dna_acts, consistency]),
    }

    logger.info(f"CCA canonical correlations: top5={np.mean([np.corrcoef(cca_prot[:,i], cca_dna[:,i])[0,1] for i in range(min(5, n_cca))]):.3f}")
    logger.info(f"Mean consistency: {consistency.mean():.3f} ± {consistency.std():.3f}")

    return representations


def run_regression(X, y, name, n_splits=5):
    """Run regression with Ridge and GBR, report R² and MSE."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Ridge
    ridge_r2 = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=kf, scoring="r2")
    ridge_mse = -cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=kf, scoring="neg_mean_squared_error")

    # GBR (on PCA-reduced if high-dim)
    if X.shape[1] > 200:
        pca = PCA(n_components=min(100, X.shape[1], X.shape[0] - 1))
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    gbr_r2 = cross_val_score(
        GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        X_reduced, y, cv=kf, scoring="r2",
    )

    return {
        "name": name,
        "ridge_r2": ridge_r2.mean(),
        "ridge_r2_std": ridge_r2.std(),
        "ridge_mse": ridge_mse.mean(),
        "gbr_r2": gbr_r2.mean(),
        "gbr_r2_std": gbr_r2.std(),
        "dim": X.shape[1],
        "n": X.shape[0],
    }


def run_classification(X, y, name, n_splits=5):
    """Run classification with LogReg and GBC, report AUROC."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # LogReg
    lr_auc = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        X_scaled, y, cv=kf, scoring="roc_auc",
    )

    # GBC
    if X.shape[1] > 200:
        pca = PCA(n_components=min(100, X.shape[1], X.shape[0] - 1))
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    gbc_auc = cross_val_score(
        GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        X_reduced, y, cv=kf, scoring="roc_auc",
    )

    return {
        "name": name,
        "lr_auc": lr_auc.mean(),
        "lr_auc_std": lr_auc.std(),
        "gbc_auc": gbc_auc.mean(),
        "gbc_auc_std": gbc_auc.std(),
        "dim": X.shape[1],
        "n": X.shape[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_h5", required=True)
    parser.add_argument("--dna_h5", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["loeuf"],
                        choices=["loeuf", "essentiality", "dnds", "gene_age"])
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--cca_dim", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load activations
    prot_acts, prot_names = load_activations(args.protein_h5)
    dna_acts, dna_names = load_activations(args.dna_h5)

    # Align genes
    prot_map = {n: i for i, n in enumerate(prot_names)}
    dna_map = {n: i for i, n in enumerate(dna_names)}
    common = sorted(set(prot_names) & set(dna_names))
    prot_aligned = prot_acts[[prot_map[g] for g in common]]
    dna_aligned = dna_acts[[dna_map[g] for g in common]]
    logger.info(f"Aligned: {len(common)} common genes")

    for task in args.tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task}")
        logger.info(f"{'='*60}")

        # Load labels
        if task == "loeuf":
            gnomad_path = os.path.join(args.labels_dir, "gnomad_constraint.txt.bgz")
            raw_labels = load_gnomad_loeuf(gnomad_path)
            gene_labels = {g: raw_labels[g]["loeuf"] for g in common if g in raw_labels}
            task_type = "regression"
            label_name = "LOEUF"
        elif task == "essentiality":
            ess_path = os.path.join(args.labels_dir, "depmap_common_essentials.csv")
            noness_path = os.path.join(args.labels_dir, "depmap_nonessentials.csv")
            raw_labels = load_depmap_essentiality(ess_path, noness_path)
            gene_labels = {g: raw_labels[g] for g in common if g in raw_labels}
            task_type = "classification"
            label_name = "Essential"
        elif task == "dnds":
            dnds_path = os.path.join(args.labels_dir, "human_mouse_dnds.tsv")
            raw_labels = load_dnds(dnds_path)
            gene_labels = {g: raw_labels[g]["dnds"] for g in common if g in raw_labels}
            task_type = "regression"
            label_name = "dN/dS"
        elif task == "gene_age":
            age_path = os.path.join(args.labels_dir, "gene_age.tsv")
            raw_labels = load_gene_age(age_path)
            gene_labels = {g: raw_labels[g] for g in common if g in raw_labels}
            task_type = "regression"
            label_name = "Phylostratum"

        if not gene_labels:
            logger.warning(f"No labels found for {task}, skipping")
            continue

        # Get indices for labeled genes
        common_list = list(common)
        labeled_genes = sorted(gene_labels.keys())
        indices = [common_list.index(g) for g in labeled_genes]
        y = np.array([gene_labels[g] for g in labeled_genes])

        logger.info(f"Labeled genes: {len(labeled_genes)}")
        logger.info(f"Label stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

        # Build representations for labeled genes only
        prot_sub = prot_aligned[indices]
        dna_sub = dna_aligned[indices]

        logger.info("Building representations...")
        reps = build_representations(prot_sub, dna_sub, args.pca_dim, args.cca_dim)

        # Run predictions
        results = []
        for rep_name, X in reps.items():
            logger.info(f"  {rep_name} ({X.shape[1]} dims)...")
            if task_type == "regression":
                res = run_regression(X, y, rep_name)
                logger.info(f"    Ridge R²={res['ridge_r2']:.4f}±{res['ridge_r2_std']:.4f}, "
                            f"GBR R²={res['gbr_r2']:.4f}±{res['gbr_r2_std']:.4f}")
            else:
                res = run_classification(X, y, rep_name)
                logger.info(f"    LogReg AUC={res['lr_auc']:.4f}±{res['lr_auc_std']:.4f}, "
                            f"GBC AUC={res['gbc_auc']:.4f}±{res['gbc_auc_std']:.4f}")
            results.append(res)

        # Save and print summary
        df = pd.DataFrame(results)
        out_path = os.path.join(args.output_dir, f"{task}_results.csv")
        df.to_csv(out_path, index=False)

        print(f"\n{'='*70}")
        print(f"  {task.upper()} — {label_name} prediction ({len(labeled_genes)} genes)")
        print(f"{'='*70}")
        if task_type == "regression":
            df_sorted = df.sort_values("gbr_r2", ascending=False)
            print(f"{'Representation':25s} {'Ridge R²':>10s} {'GBR R²':>10s} {'Dim':>5s}")
            print("-" * 55)
            for _, row in df_sorted.iterrows():
                print(f"{row['name']:25s} {row['ridge_r2']:10.4f} {row['gbr_r2']:10.4f} {int(row['dim']):5d}")
        else:
            df_sorted = df.sort_values("gbc_auc", ascending=False)
            print(f"{'Representation':25s} {'LogReg AUC':>12s} {'GBC AUC':>10s} {'Dim':>5s}")
            print("-" * 55)
            for _, row in df_sorted.iterrows():
                print(f"{row['name']:25s} {row['lr_auc']:12.4f} {row['gbc_auc']:10.4f} {int(row['dim']):5d}")

        print(f"\nSaved to {out_path}")

    # Also run dN/dS decomposition if dnds labels available
    if "dnds" in args.tasks:
        dnds_path = os.path.join(args.labels_dir, "human_mouse_dnds.tsv")
        raw_labels = load_dnds(dnds_path)
        genes_with_dnds = [g for g in common if g in raw_labels]
        if genes_with_dnds:
            logger.info("\n=== dN/dS Decomposition: dN vs dS separately ===")
            indices = [common_list.index(g) for g in genes_with_dnds]
            prot_sub = prot_aligned[indices]
            dna_sub = dna_aligned[indices]

            dn = np.array([raw_labels[g]["dn"] for g in genes_with_dnds])
            ds = np.array([raw_labels[g]["ds"] for g in genes_with_dnds])

            for target_name, target in [("dN", dn), ("dS", ds)]:
                print(f"\n--- Predicting {target_name} ---")
                for feat_name, feat in [("Protein-only", prot_sub), ("DNA-only", dna_sub), ("Concat", np.hstack([prot_sub, dna_sub]))]:
                    res = run_regression(feat, target, feat_name)
                    print(f"  {feat_name:20s}: Ridge R²={res['ridge_r2']:.4f}, GBR R²={res['gbr_r2']:.4f}")


if __name__ == "__main__":
    main()
