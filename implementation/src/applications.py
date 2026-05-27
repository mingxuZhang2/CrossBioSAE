"""
Four downstream applications for the protein-DNA shared gene space.

App 1: Cross-modal functional transfer benchmark
App 2: Consensus high-confidence annotation
App 3: Gene model / annotation QC
App 4: Discordance atlas with confound controls
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .alignment import CCAAlignment

logger = logging.getLogger(__name__)


# =============================================================================
# Shared utilities
# =============================================================================

def prepare_go_labels(gene_names, go_annotations, min_positive=20, min_negative=20, max_terms=50):
    annotated_mask = np.array([g in go_annotations and len(go_annotations[g]) > 0 for g in gene_names])
    annotated_idx = np.where(annotated_mask)[0]
    annotated_genes = [gene_names[i] for i in annotated_idx]

    all_terms = []
    for g in annotated_genes:
        all_terms.extend(go_annotations[g])
    top_terms = [t for t, c in Counter(all_terms).most_common(200)
                 if c >= min_positive][:max_terms]

    labels = np.zeros((len(annotated_idx), len(top_terms)), dtype=np.int32)
    for i, g in enumerate(annotated_genes):
        for j, t in enumerate(top_terms):
            if t in go_annotations[g]:
                labels[i, j] = 1

    valid_terms = []
    valid_cols = []
    for j, t in enumerate(top_terms):
        if labels[:, j].sum() >= min_positive and (1 - labels[:, j]).sum() >= min_negative:
            valid_terms.append(t)
            valid_cols.append(j)
    labels = labels[:, valid_cols]

    return annotated_idx, valid_terms, labels


def fit_cca_shared_space(prot_train, dna_train, prot_all, dna_all,
                         n_components=50, pca_dim=256):
    cca = CCAAlignment(n_components=n_components, pca_dim=pca_dim)
    cca.fit(prot_train, dna_train)
    proj_prot, proj_dna = cca.transform(prot_all, dna_all)
    return cca, proj_prot, proj_dna


def compute_fmax(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(np.nanmax(f1))


# =============================================================================
# App 1: Cross-modal functional transfer benchmark
# =============================================================================

def run_transfer_benchmark(
    prot_raw, dna_raw,
    prot_sae_feats, dna_sae_feats,
    gene_names, go_annotations,
    output_dir,
    seed=42,
):
    logger.info("=" * 60)
    logger.info("APP 1: Cross-modal functional transfer benchmark")
    logger.info("=" * 60)

    ann_idx, go_terms, labels = prepare_go_labels(gene_names, go_annotations)
    logger.info(f"Annotated genes: {len(ann_idx)}, GO terms: {len(go_terms)}")

    rng = np.random.RandomState(seed)
    n = len(ann_idx)
    idx = rng.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_i, val_i, test_i = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    results = []

    for rep_name, prot_feat, dna_feat in [
        ("Raw+CCA", prot_raw[ann_idx], dna_raw[ann_idx]),
        ("SAE+CCA", prot_sae_feats[ann_idx], dna_sae_feats[ann_idx]),
    ]:
        cca = CCAAlignment(n_components=50, pca_dim=256)
        cca.fit(prot_feat[train_i], dna_feat[train_i])

        proj_prot_train, proj_dna_train = cca.transform(prot_feat[train_i], dna_feat[train_i])
        proj_prot_test, proj_dna_test = cca.transform(prot_feat[test_i], dna_feat[test_i])

        scaler_prot = StandardScaler().fit(proj_prot_train)
        scaler_dna = StandardScaler().fit(proj_dna_train)
        X_prot_train = scaler_prot.transform(proj_prot_train)
        X_prot_test = scaler_prot.transform(proj_prot_test)
        X_dna_test = scaler_prot.transform(proj_dna_test)
        X_dna_train_own = scaler_dna.transform(proj_dna_train)
        X_dna_test_own = scaler_dna.transform(proj_dna_test)

        for j, term in enumerate(go_terms):
            y_train = labels[train_i, j]
            y_test = labels[test_i, j]
            if y_train.sum() < 5 or y_test.sum() < 3 or (1 - y_test).sum() < 3:
                continue

            clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
            clf.fit(X_prot_train, y_train)

            # Protein -> Protein (same modality upper bound)
            pred_same = clf.predict_proba(X_prot_test)[:, 1]
            # Protein -> DNA (cross-modal transfer)
            pred_cross = clf.predict_proba(X_dna_test)[:, 1]

            # DNA -> DNA (DNA-supervised baseline)
            clf_dna = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
            clf_dna.fit(X_dna_train_own, y_train)
            pred_dna_only = clf_dna.predict_proba(X_dna_test_own)[:, 1]

            try:
                r = {
                    "representation": rep_name,
                    "go_term": term,
                    "prot2prot_auroc": roc_auc_score(y_test, pred_same),
                    "prot2dna_auroc": roc_auc_score(y_test, pred_cross),
                    "dna2dna_auroc": roc_auc_score(y_test, pred_dna_only),
                    "prot2prot_auprc": average_precision_score(y_test, pred_same),
                    "prot2dna_auprc": average_precision_score(y_test, pred_cross),
                    "dna2dna_auprc": average_precision_score(y_test, pred_dna_only),
                    "prot2prot_fmax": compute_fmax(y_test, pred_same),
                    "prot2dna_fmax": compute_fmax(y_test, pred_cross),
                    "dna2dna_fmax": compute_fmax(y_test, pred_dna_only),
                }
                excess_same = r["prot2prot_auroc"] - 0.5
                excess_cross = r["prot2dna_auroc"] - 0.5
                r["transfer_efficiency"] = excess_cross / max(excess_same, 0.01)
                results.append(r)
            except Exception:
                continue

        # DNA-supervised at various label fractions (as comparison)
        for frac in [0.01, 0.05, 0.10, 0.25]:
            n_label = max(10, int(len(train_i) * frac))
            label_subset = train_i[:n_label]

            for j, term in enumerate(go_terms):
                y_sub = labels[label_subset, j]
                y_test = labels[test_i, j]
                if y_sub.sum() < 2 or y_test.sum() < 3 or (1 - y_test).sum() < 3:
                    continue

                try:
                    X_sub = scaler_dna.transform(
                        cca.transform(prot_feat[train_i[:n_label]], dna_feat[train_i[:n_label]])[1]
                    )
                    clf_sub = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
                    clf_sub.fit(X_sub, y_sub)
                    pred_sub = clf_sub.predict_proba(X_dna_test_own)[:, 1]
                    results.append({
                        "representation": f"{rep_name}_DNA-supervised-{frac}",
                        "go_term": term,
                        "dna_lowlabel_auroc": roc_auc_score(y_test, pred_sub),
                        "label_fraction": frac,
                        "n_labels": n_label,
                    })
                except Exception:
                    continue

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "app1_transfer_benchmark.csv", index=False)

    # Summary
    logger.info("\n--- App 1: Transfer Benchmark Summary ---")
    for rep in ["Raw+CCA", "SAE+CCA"]:
        sub = df[df["representation"] == rep]
        if len(sub) == 0:
            continue
        logger.info(f"\n{rep} ({len(sub)} GO terms):")
        logger.info(f"  Prot->Prot AUROC: {sub['prot2prot_auroc'].mean():.4f}")
        logger.info(f"  Prot->DNA  AUROC: {sub['prot2dna_auroc'].mean():.4f}")
        logger.info(f"  DNA->DNA   AUROC: {sub['dna2dna_auroc'].mean():.4f}")
        logger.info(f"  Transfer efficiency: {sub['transfer_efficiency'].mean():.4f}")
        logger.info(f"  Prot->Prot AUPRC: {sub['prot2prot_auprc'].mean():.4f}")
        logger.info(f"  Prot->DNA  AUPRC: {sub['prot2dna_auprc'].mean():.4f}")
        logger.info(f"  Prot->Prot Fmax:  {sub['prot2prot_fmax'].mean():.4f}")
        logger.info(f"  Prot->DNA  Fmax:  {sub['prot2dna_fmax'].mean():.4f}")

    # DNA low-label comparison
    for frac in [0.01, 0.05, 0.10, 0.25]:
        for rep in ["Raw+CCA", "SAE+CCA"]:
            tag = f"{rep}_DNA-supervised-{frac}"
            sub = df[df["representation"] == tag]
            if len(sub) > 0:
                logger.info(f"  {tag}: DNA-lowlabel AUROC = {sub['dna_lowlabel_auroc'].mean():.4f}")

    return df


# =============================================================================
# App 2: Consensus high-confidence annotation
# =============================================================================

def run_consensus_annotation(
    prot_raw, dna_raw,
    prot_sae_feats, dna_sae_feats,
    gene_names, go_annotations,
    output_dir,
    seed=42,
):
    logger.info("=" * 60)
    logger.info("APP 2: Consensus high-confidence annotation")
    logger.info("=" * 60)

    ann_idx, go_terms, labels = prepare_go_labels(gene_names, go_annotations)

    rng = np.random.RandomState(seed)
    n = len(ann_idx)
    idx = rng.permutation(n)
    n_train = int(n * 0.6)
    train_i, test_i = idx[:n_train], idx[n_train:]

    all_results = []

    for rep_name, prot_feat, dna_feat in [
        ("Raw+CCA", prot_raw[ann_idx], dna_raw[ann_idx]),
        ("SAE+CCA", prot_sae_feats[ann_idx], dna_sae_feats[ann_idx]),
    ]:
        cca = CCAAlignment(n_components=50, pca_dim=256)
        cca.fit(prot_feat[train_i], dna_feat[train_i])

        proj_prot_train, proj_dna_train = cca.transform(prot_feat[train_i], dna_feat[train_i])
        proj_prot_test, proj_dna_test = cca.transform(prot_feat[test_i], dna_feat[test_i])

        scaler = StandardScaler().fit(proj_prot_train)
        X_prot_train = scaler.transform(proj_prot_train)
        X_prot_test = scaler.transform(proj_prot_test)
        X_dna_test = scaler.transform(proj_dna_test)

        for j, term in enumerate(go_terms):
            y_train = labels[train_i, j]
            y_test = labels[test_i, j]
            if y_train.sum() < 5 or y_test.sum() < 3 or (1 - y_test).sum() < 3:
                continue

            clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
            clf.fit(X_prot_train, y_train)

            prot_score = clf.predict_proba(X_prot_test)[:, 1]
            dna_score = clf.predict_proba(X_dna_test)[:, 1]

            # Consensus: geometric mean of protein and DNA scores
            consensus_score = np.sqrt(prot_score * dna_score)
            # Agreement: absolute difference (low = high agreement)
            agreement = 1.0 - np.abs(prot_score - dna_score)

            for i_test in range(len(test_i)):
                all_results.append({
                    "representation": rep_name,
                    "go_term": term,
                    "gene": gene_names[ann_idx[test_i[i_test]]],
                    "true_label": int(y_test[i_test]),
                    "prot_score": float(prot_score[i_test]),
                    "dna_score": float(dna_score[i_test]),
                    "consensus_score": float(consensus_score[i_test]),
                    "agreement": float(agreement[i_test]),
                })

    predictions_df = pd.DataFrame(all_results)
    predictions_df.to_csv(output_dir / "app2_predictions_raw.csv", index=False)

    # Compute precision-coverage curves
    summary_results = []
    for rep_name in predictions_df["representation"].unique():
        rep_df = predictions_df[predictions_df["representation"] == rep_name]

        for score_col, method_name in [
            ("prot_score", "Protein-only"),
            ("dna_score", "DNA-transferred"),
            ("consensus_score", "Consensus (geom. mean)"),
        ]:
            # Pool across all GO terms: rank by score, measure precision at top-k%
            for go_term in rep_df["go_term"].unique():
                term_df = rep_df[rep_df["go_term"] == go_term].copy()
                term_df = term_df.sort_values(score_col, ascending=False)
                n_total = len(term_df)

                for top_frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
                    k = max(1, int(n_total * top_frac))
                    top_k = term_df.head(k)
                    precision = top_k["true_label"].mean()
                    coverage = top_k["true_label"].sum() / max(term_df["true_label"].sum(), 1)

                    summary_results.append({
                        "representation": rep_name,
                        "method": method_name,
                        "go_term": go_term,
                        "top_fraction": top_frac,
                        "k": k,
                        "precision_at_k": float(precision),
                        "coverage_at_k": float(coverage),
                    })

        # High-agreement selective prediction
        rep_df_copy = rep_df.copy()
        for go_term in rep_df_copy["go_term"].unique():
            term_df = rep_df_copy[rep_df_copy["go_term"] == go_term].copy()
            # Only keep high-agreement predictions
            for agreement_threshold in [0.7, 0.8, 0.9]:
                high_agree = term_df[term_df["agreement"] >= agreement_threshold]
                if len(high_agree) < 5:
                    continue
                try:
                    auroc = roc_auc_score(high_agree["true_label"], high_agree["consensus_score"])
                    summary_results.append({
                        "representation": rep_name,
                        "method": f"Consensus (agreement>{agreement_threshold})",
                        "go_term": go_term,
                        "agreement_threshold": agreement_threshold,
                        "n_predictions": len(high_agree),
                        "coverage_fraction": len(high_agree) / len(term_df),
                        "auroc_selective": float(auroc),
                        "precision_selective": float(high_agree["true_label"].mean()),
                    })
                except Exception:
                    continue

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(output_dir / "app2_consensus_summary.csv", index=False)

    # Print summary
    logger.info("\n--- App 2: Consensus Annotation Summary ---")
    for rep_name in ["Raw+CCA", "SAE+CCA"]:
        logger.info(f"\n{rep_name}:")
        for method in ["Protein-only", "DNA-transferred", "Consensus (geom. mean)"]:
            sub = summary_df[
                (summary_df["representation"] == rep_name) &
                (summary_df["method"] == method) &
                (summary_df["top_fraction"] == 0.05)
            ]
            if len(sub) > 0:
                logger.info(f"  {method:30s}: precision@5% = {sub['precision_at_k'].mean():.4f}, "
                           f"coverage@5% = {sub['coverage_at_k'].mean():.4f}")

        for thresh in [0.7, 0.8, 0.9]:
            sub = summary_df[
                (summary_df["representation"] == rep_name) &
                (summary_df["method"] == f"Consensus (agreement>{thresh})")
            ]
            if len(sub) > 0:
                logger.info(f"  Agreement>{thresh}: AUROC={sub['auroc_selective'].mean():.4f}, "
                           f"n_pred={sub['n_predictions'].mean():.0f}, "
                           f"precision={sub['precision_selective'].mean():.4f}")

    return summary_df


# =============================================================================
# App 3: Gene model / annotation QC
# =============================================================================

def run_annotation_qc(
    prot_raw, dna_raw,
    gene_names, gene_pairs,
    output_dir,
    seed=42,
):
    logger.info("=" * 60)
    logger.info("APP 3: Gene model / annotation QC")
    logger.info("=" * 60)

    n = len(gene_names)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Fit CCA on train
    cca = CCAAlignment(n_components=50, pca_dim=256)
    cca.fit(prot_raw[train_idx], dna_raw[train_idx])
    proj_prot, proj_dna = cca.transform(prot_raw, dna_raw)

    # Correct-pair consistency scores
    correct_cosine = np.array([
        1 - cdist(proj_prot[i:i+1], proj_dna[i:i+1], metric="cosine")[0, 0]
        for i in range(n)
    ])

    # Build gene family map from gene names
    gene_map = {p["gene_name"]: p for p in gene_pairs} if gene_pairs else {}
    family_prefixes = ["ZNF", "OR", "KRT", "SLC", "HLA", "TAS", "RPS", "RPL",
                       "MT-", "HIST", "TMEM", "FAM", "ADAM", "MYH", "COL",
                       "KRTAP", "OLFR", "GPC", "PCDH", "CDH", "ABCB"]

    def get_family(name):
        for p in family_prefixes:
            if name.startswith(p):
                return p
        return None

    families = [get_family(g) for g in gene_names]

    # Construct negative pairs
    neg_types = {}

    # Type 1: Random shuffle
    shuf_idx = rng.permutation(n)
    while np.any(shuf_idx == np.arange(n)):
        shuf_idx = rng.permutation(n)
    neg_types["random_shuffle"] = shuf_idx

    # Type 2: Same-family swap (hard negative)
    family_members = {}
    for i, f in enumerate(families):
        if f is not None:
            family_members.setdefault(f, []).append(i)

    same_family_pairs = np.arange(n)
    for fam, members in family_members.items():
        if len(members) < 2:
            continue
        shuffled = rng.permutation(members)
        # Circular shift to avoid self-match
        shifted = np.roll(shuffled, 1)
        for orig, new in zip(shuffled, shifted):
            same_family_pairs[orig] = new
    neg_types["same_family_swap"] = same_family_pairs

    # Type 3: Same-length-bin swap
    if gene_pairs:
        lengths = np.array([gene_map.get(g, {}).get("protein_length", 300) for g in gene_names])
    else:
        lengths = np.full(n, 300)
    n_bins = 20
    bins = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(lengths, bins)

    length_swap = np.arange(n)
    for b in range(1, n_bins + 2):
        members = np.where(bin_idx == b)[0]
        if len(members) < 2:
            continue
        shuffled = rng.permutation(members)
        shifted = np.roll(shuffled, 1)
        for orig, new in zip(shuffled, shifted):
            length_swap[orig] = new
    neg_types["length_matched_swap"] = length_swap

    # Compute negative-pair cosine scores
    results = []

    for neg_type, neg_idx in neg_types.items():
        neg_cosine = np.array([
            1 - cdist(proj_prot[i:i+1], proj_dna[neg_idx[i]:neg_idx[i]+1], metric="cosine")[0, 0]
            for i in range(n)
        ])

        # Binary classification: correct (1) vs wrong (0)
        all_scores = np.concatenate([correct_cosine, neg_cosine])
        all_labels = np.concatenate([np.ones(n), np.zeros(n)])

        auroc = roc_auc_score(all_labels, all_scores)
        auprc = average_precision_score(all_labels, all_scores)

        # Precision at various thresholds
        sorted_idx = np.argsort(-all_scores)
        sorted_labels = all_labels[sorted_idx]
        for top_k in [100, 500, 1000]:
            if top_k <= len(sorted_labels):
                p_at_k = sorted_labels[:top_k].mean()
                results.append({
                    "negative_type": neg_type,
                    "metric": f"precision@{top_k}",
                    "value": float(p_at_k),
                })

        # Score statistics
        results.append({"negative_type": neg_type, "metric": "AUROC", "value": float(auroc)})
        results.append({"negative_type": neg_type, "metric": "AUPRC", "value": float(auprc)})
        results.append({"negative_type": neg_type, "metric": "correct_mean_cosine",
                        "value": float(correct_cosine.mean())})
        results.append({"negative_type": neg_type, "metric": "negative_mean_cosine",
                        "value": float(neg_cosine.mean())})
        results.append({"negative_type": neg_type, "metric": "cohen_d",
                        "value": float((correct_cosine.mean() - neg_cosine.mean()) /
                                      np.sqrt((correct_cosine.var() + neg_cosine.var()) / 2 + 1e-8))})

        logger.info(f"  {neg_type:25s}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
                    f"correct_cos={correct_cosine.mean():.4f}, neg_cos={neg_cosine.mean():.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "app3_annotation_qc.csv", index=False)

    # Per-gene consistency scores for further analysis
    gene_scores = pd.DataFrame({
        "gene_name": gene_names,
        "cca_cosine": correct_cosine,
        "family": families,
    })
    gene_scores.to_csv(output_dir / "app3_gene_consistency_scores.csv", index=False)

    logger.info("\n--- App 3: Annotation QC Summary ---")
    for neg_type in neg_types:
        sub = results_df[(results_df["negative_type"] == neg_type) & (results_df["metric"] == "AUROC")]
        if len(sub) > 0:
            logger.info(f"  {neg_type}: AUROC = {sub['value'].values[0]:.4f}")

    return results_df


# =============================================================================
# App 4: Discordance atlas with confound controls
# =============================================================================

def run_discordance_atlas(
    prot_raw, dna_raw,
    prot_sae_feats, dna_sae_feats,
    gene_names, gene_pairs, go_annotations,
    output_dir,
    seed=42,
):
    logger.info("=" * 60)
    logger.info("APP 4: Discordance atlas with confound controls")
    logger.info("=" * 60)

    n = len(gene_names)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:int(n * 0.8)], idx[int(n * 0.8):]

    # CCA shared space
    cca = CCAAlignment(n_components=50, pca_dim=256)
    cca.fit(prot_raw[train_idx], dna_raw[train_idx])
    proj_prot, proj_dna = cca.transform(prot_raw, dna_raw)

    # Per-gene consistency (cosine in CCA space)
    consistency = np.array([
        1 - cdist(proj_prot[i:i+1], proj_dna[i:i+1], metric="cosine")[0, 0]
        for i in range(n)
    ])

    # Euclidean distance in CCA space
    euc_distance = np.linalg.norm(proj_prot - proj_dna, axis=1)

    # Build gene metadata
    gene_map = {p["gene_name"]: p for p in gene_pairs} if gene_pairs else {}

    metadata = []
    for i, g in enumerate(gene_names):
        info = gene_map.get(g, {})
        prot_seq = info.get("protein_seq", "")
        cds_seq = info.get("cds_seq", "")

        prot_len = len(prot_seq) if prot_seq else info.get("protein_length", 0)
        cds_len = len(cds_seq) if cds_seq else 0

        gc = 0.0
        gc3 = 0.0
        if cds_seq and len(cds_seq) > 0:
            gc = (cds_seq.count("G") + cds_seq.count("C")) / len(cds_seq)
            # GC3: GC content at third codon position
            third_positions = cds_seq[2::3]
            if len(third_positions) > 0:
                gc3 = (third_positions.count("G") + third_positions.count("C")) / len(third_positions)

        # Amino acid composition features
        aa_disorder = 0.0
        if prot_seq:
            disorder_aa = set("ADEGKNPQRS")
            aa_disorder = sum(1 for aa in prot_seq if aa in disorder_aa) / max(len(prot_seq), 1)

        metadata.append({
            "gene_name": g,
            "protein_length": prot_len,
            "cds_length": cds_len,
            "gc_content": gc,
            "gc3": gc3,
            "disorder_fraction": aa_disorder,
        })

    meta_df = pd.DataFrame(metadata)

    # Build confound matrix
    confound_cols = ["protein_length", "cds_length", "gc_content", "gc3", "disorder_fraction"]
    has_metadata = meta_df["protein_length"] > 0
    valid_mask = has_metadata.values

    if valid_mask.sum() > 100:
        X_confound = meta_df[confound_cols].values[valid_mask]
        X_confound = StandardScaler().fit_transform(X_confound)
        y_consistency = consistency[valid_mask]

        # Regress out confounds
        reg = LinearRegression()
        reg.fit(X_confound, y_consistency)
        predicted = reg.predict(X_confound)
        residual = y_consistency - predicted
        r2 = reg.score(X_confound, y_consistency)

        logger.info(f"Confound regression R^2 = {r2:.4f}")
        logger.info(f"Confound coefficients:")
        for col, coef in zip(confound_cols, reg.coef_):
            logger.info(f"  {col:20s}: {coef:+.6f}")
    else:
        residual = consistency.copy()
        r2 = 0.0
        logger.warning("Not enough metadata for confound regression")

    # Gene family enrichment analysis
    family_prefixes = ["ZNF", "OR", "KRT", "SLC", "HLA", "TAS", "RPS", "RPL",
                       "MT-", "HIST", "TMEM", "FAM", "ADAM", "MYH", "COL",
                       "KRTAP", "PCDH", "CDH", "ABCB", "CYP", "UGT", "GSTA",
                       "IL", "IFNA", "TRBV", "TRAV", "IGLV", "IGHV", "KIR",
                       "RBM", "HNRNP", "DDX", "SRSF"]

    # Discordant = bottom 10%, concordant = top 10%
    n_extreme = max(100, int(n * 0.1))
    sorted_by_consistency = np.argsort(consistency)
    discordant_idx = set(sorted_by_consistency[:n_extreme])
    concordant_idx = set(sorted_by_consistency[-n_extreme:])

    enrichment_results = []
    for prefix in family_prefixes:
        family_genes = [i for i, g in enumerate(gene_names) if g.startswith(prefix)]
        if len(family_genes) < 3:
            continue

        n_fam = len(family_genes)
        n_disc = len(set(family_genes) & discordant_idx)
        n_conc = len(set(family_genes) & concordant_idx)
        expected = n_fam * n_extreme / n

        # Fisher exact for enrichment in discordant
        _, p_disc = stats.fisher_exact([
            [n_disc, n_extreme - n_disc],
            [n_fam - n_disc, n - n_extreme - (n_fam - n_disc)],
        ], alternative="greater")

        # Same for concordant
        _, p_conc = stats.fisher_exact([
            [n_conc, n_extreme - n_conc],
            [n_fam - n_conc, n - n_extreme - (n_fam - n_conc)],
        ], alternative="greater")

        # Mean consistency for family
        fam_consistency = consistency[family_genes]
        fam_mean = fam_consistency.mean()
        overall_mean = consistency.mean()
        overall_std = consistency.std()
        z = (fam_mean - overall_mean) / max(overall_std, 1e-8)

        # If we have confound-corrected residuals
        if valid_mask.sum() > 100:
            valid_fam = [i for i in family_genes if valid_mask[i]]
            if valid_fam:
                valid_fam_local = [np.where(np.where(valid_mask)[0] == vi)[0][0]
                                   for vi in valid_fam
                                   if vi in np.where(valid_mask)[0]]
                if valid_fam_local:
                    fam_residual_mean = residual[valid_fam_local].mean()
                    residual_z = (fam_residual_mean - residual.mean()) / max(residual.std(), 1e-8)
                else:
                    fam_residual_mean = np.nan
                    residual_z = np.nan
            else:
                fam_residual_mean = np.nan
                residual_z = np.nan
        else:
            fam_residual_mean = np.nan
            residual_z = np.nan

        enrichment_results.append({
            "family": prefix,
            "n_genes": n_fam,
            "mean_consistency": float(fam_mean),
            "z_score": float(z),
            "in_discordant": n_disc,
            "in_concordant": n_conc,
            "expected": round(expected, 1),
            "fold_enrichment_discordant": n_disc / max(expected, 0.1),
            "p_discordant": float(p_disc),
            "p_concordant": float(p_conc),
            "residual_mean": float(fam_residual_mean) if not np.isnan(fam_residual_mean) else None,
            "residual_z": float(residual_z) if not np.isnan(residual_z) else None,
        })

    enrich_df = pd.DataFrame(enrichment_results).sort_values("p_discordant")
    enrich_df.to_csv(output_dir / "app4_family_enrichment.csv", index=False)

    # SAE feature analysis for top discordant genes
    sae_analysis = []
    top_discordant_genes = [gene_names[i] for i in sorted_by_consistency[:50]]
    top_concordant_genes = [gene_names[i] for i in sorted_by_consistency[-50:]]

    if prot_sae_feats is not None and dna_sae_feats is not None:
        disc_indices = sorted_by_consistency[:50]
        conc_indices = sorted_by_consistency[-50:]

        # Feature-level analysis: which features differ most between modalities in discordant genes?
        for label, indices in [("discordant", disc_indices), ("concordant", conc_indices)]:
            prot_feats = prot_sae_feats[indices]
            dna_feats = dna_sae_feats[indices]

            # Per-feature modality difference
            prot_active = (prot_feats > 0).mean(axis=0)
            dna_active = (dna_feats > 0).mean(axis=0)
            modality_diff = prot_active - dna_active

            # Top protein-dominant features
            prot_dominant = np.argsort(-modality_diff)[:20]
            # Top DNA-dominant features
            dna_dominant = np.argsort(modality_diff)[:20]

            sae_analysis.append({
                "group": label,
                "mean_prot_active": float(prot_active.mean()),
                "mean_dna_active": float(dna_active.mean()),
                "n_prot_dominant_features": int((modality_diff > 0.1).sum()),
                "n_dna_dominant_features": int((modality_diff < -0.1).sum()),
                "top_prot_dominant_features": prot_dominant.tolist(),
                "top_dna_dominant_features": dna_dominant.tolist(),
            })

    sae_df = pd.DataFrame(sae_analysis)
    sae_df.to_csv(output_dir / "app4_sae_feature_analysis.csv", index=False)

    # Full gene-level atlas
    atlas = pd.DataFrame({
        "gene_name": gene_names,
        "cca_consistency": consistency,
        "cca_euc_distance": euc_distance,
        "consistency_rank": np.argsort(np.argsort(consistency)) + 1,
    })
    atlas = atlas.merge(meta_df, on="gene_name", how="left")
    atlas = atlas.sort_values("cca_consistency")
    atlas.to_csv(output_dir / "app4_gene_atlas.csv", index=False)

    # Print summary
    logger.info("\n--- App 4: Discordance Atlas Summary ---")
    logger.info(f"Genes: {n}, Confound R^2: {r2:.4f}")
    logger.info(f"Mean consistency: {consistency.mean():.4f} +/- {consistency.std():.4f}")
    logger.info(f"\nTop discordant gene families (enriched in bottom 10%):")
    sig = enrich_df[enrich_df["p_discordant"] < 0.05]
    for _, row in sig.iterrows():
        residual_str = f", residual_z={row['residual_z']:.2f}" if row.get("residual_z") is not None else ""
        logger.info(
            f"  {row['family']:8s}: {row['in_discordant']}/{row['n_genes']} in discordant "
            f"(fold={row['fold_enrichment_discordant']:.1f}x, p={row['p_discordant']:.2e}, "
            f"z={row['z_score']:.2f}{residual_str})"
        )

    logger.info(f"\nTop 10 most discordant genes:")
    for i in sorted_by_consistency[:10]:
        logger.info(f"  {gene_names[i]:15s}: consistency={consistency[i]:.4f}")

    logger.info(f"\nTop 10 most concordant genes:")
    for i in sorted_by_consistency[-10:][::-1]:
        logger.info(f"  {gene_names[i]:15s}: consistency={consistency[i]:.4f}")

    if sae_analysis:
        logger.info(f"\nSAE feature analysis:")
        for row in sae_analysis:
            logger.info(f"  {row['group']}: prot_active={row['mean_prot_active']:.4f}, "
                       f"dna_active={row['mean_dna_active']:.4f}, "
                       f"prot_dominant={row['n_prot_dominant_features']}, "
                       f"dna_dominant={row['n_dna_dominant_features']}")

    return enrich_df, atlas
