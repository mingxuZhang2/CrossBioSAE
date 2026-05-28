"""
Train Shared-Private VAE and evaluate on downstream tasks.

Pipeline:
1. Load protein + DNA activations
2. Train SP-VAE (80/20 split)
3. Extract shared/private representations
4. Evaluate on LOEUF regression + essentiality classification
5. Compare: shared-only, shared+private, protein-only, DNA-only, concat
"""

import argparse
import gzip
import json
import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossmodal_vae import SharedPrivateVAE, SPVAEConfig, SPVAETrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_activations(h5_path):
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
    return acts, names


def load_loeuf(path):
    labels = {}
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split("\t")
        gene_col = header.index("gene")
        loeuf_col = header.index("oe_lof_upper")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(gene_col, loeuf_col):
                gene, loeuf = parts[gene_col], parts[loeuf_col]
                if loeuf and loeuf != "NA":
                    try:
                        labels[gene] = float(loeuf)
                    except ValueError:
                        pass
    return labels


def load_essentiality(path):
    labels = {}
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        if row["essential"] in (0, 1):
            labels[row["gene_symbol"]] = int(row["essential"])
    return labels


def evaluate_representations(reps_dict, gene_names, labels, task_type, task_name):
    """Evaluate multiple representations on a task."""
    labeled_genes = [g for g in gene_names if g in labels]
    indices = [gene_names.index(g) for g in labeled_genes]
    y = np.array([labels[g] for g in labeled_genes])

    logger.info(f"\n{task_name}: {len(labeled_genes)} genes, y_mean={y.mean():.3f}")

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for rep_name, X_full in reps_dict.items():
        X = X_full[indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task_type == "regression":
            ridge_scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=kf, scoring="r2")
            gbr_scores = cross_val_score(
                GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
                X_scaled, y, cv=kf, scoring="r2",
            )
            results.append({
                "name": rep_name, "dim": X.shape[1],
                "ridge_r2": ridge_scores.mean(), "ridge_std": ridge_scores.std(),
                "gbr_r2": gbr_scores.mean(), "gbr_std": gbr_scores.std(),
            })
            logger.info(f"  {rep_name:30s} ({X.shape[1]:4d}d): Ridge R²={ridge_scores.mean():.4f}±{ridge_scores.std():.4f}  GBR R²={gbr_scores.mean():.4f}±{gbr_scores.std():.4f}")
        else:
            lr_scores = cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
                X_scaled, y, cv=kf, scoring="roc_auc",
            )
            gbc_scores = cross_val_score(
                GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
                X_scaled, y, cv=kf, scoring="roc_auc",
            )
            results.append({
                "name": rep_name, "dim": X.shape[1],
                "lr_auc": lr_scores.mean(), "lr_std": lr_scores.std(),
                "gbc_auc": gbc_scores.mean(), "gbc_std": gbc_scores.std(),
            })
            logger.info(f"  {rep_name:30s} ({X.shape[1]:4d}d): LR AUC={lr_scores.mean():.4f}±{lr_scores.std():.4f}  GBC AUC={gbc_scores.mean():.4f}±{gbc_scores.std():.4f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_h5", required=True)
    parser.add_argument("--dna_h5", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_align", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    prot_acts, prot_names = load_activations(args.protein_h5)
    dna_acts, dna_names = load_activations(args.dna_h5)

    # Align
    prot_map = {n: i for i, n in enumerate(prot_names)}
    dna_map = {n: i for i, n in enumerate(dna_names)}
    common = sorted(set(prot_names) & set(dna_names))
    prot_aligned = prot_acts[[prot_map[g] for g in common]]
    dna_aligned = dna_acts[[dna_map[g] for g in common]]
    logger.info(f"Aligned: {len(common)} genes, prot={prot_aligned.shape}, dna={dna_aligned.shape}")

    # Standardize
    prot_scaler = StandardScaler()
    dna_scaler = StandardScaler()
    prot_norm = prot_scaler.fit_transform(prot_aligned)
    dna_norm = dna_scaler.fit_transform(dna_aligned)

    # Train/val split
    n = len(common)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    prot_train = torch.tensor(prot_norm[train_idx], dtype=torch.float32)
    dna_train = torch.tensor(dna_norm[train_idx], dtype=torch.float32)
    prot_val = torch.tensor(prot_norm[val_idx], dtype=torch.float32)
    dna_val = torch.tensor(dna_norm[val_idx], dtype=torch.float32)

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Build model
    config = SPVAEConfig(
        prot_dim=prot_aligned.shape[1],
        dna_dim=dna_aligned.shape[1],
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        alpha_align=args.alpha_align,
    )
    model = SharedPrivateVAE(config)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = SPVAETrainer(model, config, lr=args.lr, device=args.device)
    save_path = os.path.join(args.output_dir, "spvae_best.pt")
    history = trainer.fit(
        prot_train, dna_train, prot_val, dna_val,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        patience=30, save_path=save_path,
    )

    # Extract representations for ALL genes
    prot_all = torch.tensor(prot_norm, dtype=torch.float32)
    dna_all = torch.tensor(dna_norm, dtype=torch.float32)
    reps = trainer.extract_all(prot_all, dna_all)

    logger.info(f"Extracted: shared={reps['shared'].shape}, "
                f"prot_private={reps['prot_private'].shape}, "
                f"dna_private={reps['dna_private'].shape}")

    # Build representation dict for evaluation
    gene_names = list(common)
    reps_dict = {
        "Protein-only (raw)": prot_norm,
        "DNA-only (raw)": dna_norm,
        "Raw concat": np.hstack([prot_norm, dna_norm]),
        "SP-VAE shared": reps["shared"],
        "SP-VAE prot_private": reps["prot_private"],
        "SP-VAE dna_private": reps["dna_private"],
        "SP-VAE shared+prot_priv": np.hstack([reps["shared"], reps["prot_private"]]),
        "SP-VAE shared+dna_priv": np.hstack([reps["shared"], reps["dna_private"]]),
        "SP-VAE all": np.hstack([reps["shared"], reps["prot_private"], reps["dna_private"]]),
    }

    # Evaluate on LOEUF
    loeuf_path = os.path.join(args.labels_dir, "gnomad_constraint.txt.bgz")
    if os.path.exists(loeuf_path):
        loeuf_labels = load_loeuf(loeuf_path)
        df_loeuf = evaluate_representations(reps_dict, gene_names, loeuf_labels, "regression", "LOEUF")
        df_loeuf.to_csv(os.path.join(args.output_dir, "loeuf_results.csv"), index=False)
        print("\n" + "=" * 80)
        print("LOEUF Prediction")
        print("=" * 80)
        print(df_loeuf.sort_values("gbr_r2", ascending=False).to_string(index=False))

    # Evaluate on essentiality
    ess_path = os.path.join(args.labels_dir, "depmap_gene_essentiality.tsv")
    if os.path.exists(ess_path):
        ess_labels = load_essentiality(ess_path)
        df_ess = evaluate_representations(reps_dict, gene_names, ess_labels, "classification", "Essentiality")
        df_ess.to_csv(os.path.join(args.output_dir, "essentiality_results.csv"), index=False)
        print("\n" + "=" * 80)
        print("Gene Essentiality Prediction")
        print("=" * 80)
        print(df_ess.sort_values("gbc_auc", ascending=False).to_string(index=False))

    # Save representations
    np.savez_compressed(
        os.path.join(args.output_dir, "spvae_representations.npz"),
        shared=reps["shared"],
        prot_private=reps["prot_private"],
        dna_private=reps["dna_private"],
        gene_names=np.array(gene_names),
    )

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    logger.info(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
