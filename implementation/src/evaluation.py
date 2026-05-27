"""
Evaluation and downstream tasks for CrossBioSAE.

Three downstream tasks:
1. Dual-modality pathogenic variant prediction
2. Function prediction for uncharacterized genes
3. Cross-modal anomaly detection for biological discovery
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from tqdm import tqdm

from .model import CrossBioSAE, CrossBioSAEConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Feature analysis utilities
# =============================================================================

def compute_feature_statistics(
    model: CrossBioSAE,
    dna_h5_path: str,
    protein_h5_path: str,
    device: str = "cuda",
    batch_size: int = 512,
) -> dict:
    """
    Compute feature activation statistics across the dataset.

    Returns:
        Dict with:
        - shared_feature_indices: features active in both modalities
        - dna_specific_indices: features active only in DNA
        - protein_specific_indices: features active only in protein
        - feature_activation_freq_dna: per-feature activation frequency (DNA)
        - feature_activation_freq_protein: per-feature activation frequency (protein)
        - crossmodal_correlation: per-gene cosine similarity of feature patterns
    """
    model.eval()
    n_features = model.config.n_features

    # Load activations
    with h5py.File(dna_h5_path, "r") as f:
        dna_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
        dna_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]

    with h5py.File(protein_h5_path, "r") as f:
        prot_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
        prot_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]

    # Match gene names
    dna_name_to_idx = {n: i for i, n in enumerate(dna_names)}
    prot_name_to_idx = {n: i for i, n in enumerate(prot_names)}
    common = sorted(set(dna_names) & set(prot_names))

    dna_indices = [dna_name_to_idx[n] for n in common]
    prot_indices = [prot_name_to_idx[n] for n in common]

    # Extract features
    all_features_dna = []
    all_features_protein = []

    with torch.no_grad():
        for start in range(0, len(common), batch_size):
            end = min(start + batch_size, len(common))
            d_idx = dna_indices[start:end]
            p_idx = prot_indices[start:end]

            d_batch = dna_acts[d_idx].to(device)
            p_batch = prot_acts[p_idx].to(device)

            feat_d = model.encode_dna(d_batch)
            feat_p = model.encode_protein(p_batch)

            all_features_dna.append(feat_d.cpu())
            all_features_protein.append(feat_p.cpu())

    features_dna = torch.cat(all_features_dna, dim=0)
    features_protein = torch.cat(all_features_protein, dim=0)

    # Feature activation frequencies
    freq_dna = (features_dna > 0).float().mean(dim=0).numpy()
    freq_protein = (features_protein > 0).float().mean(dim=0).numpy()

    # Classify features using TWO criteria:
    # 1. Marginal activity: feature fires in >1% of genes in each modality
    # 2. Matched-gene co-activation: per-feature correlation across matched genes
    threshold = 0.01
    dna_active = freq_dna > threshold
    prot_active = freq_protein > threshold

    # Per-feature matched-gene correlation (the stricter, reviewer-requested definition)
    feat_correlations = np.zeros(n_features)
    for j in range(n_features):
        d_col = features_dna[:, j].numpy()
        p_col = features_protein[:, j].numpy()
        if d_col.std() > 1e-8 and p_col.std() > 1e-8:
            feat_correlations[j] = np.corrcoef(d_col, p_col)[0, 1]

    # Shared: marginally active in both AND positively correlated across matched genes
    coactivation_threshold = 0.1
    shared_indices = np.where(
        dna_active & prot_active & (feat_correlations > coactivation_threshold)
    )[0]
    # Marginal-only shared (old definition, for comparison)
    marginal_shared = np.where(dna_active & prot_active)[0]
    dna_specific = np.where(dna_active & ~prot_active)[0]
    prot_specific = np.where(~dna_active & prot_active)[0]
    dead_features = np.where(~dna_active & ~prot_active)[0]

    logger.info(
        f"Shared features: {len(shared_indices)} (co-activation corr > {coactivation_threshold}), "
        f"{len(marginal_shared)} (marginal overlap only)"
    )

    # Per-gene cross-modal cosine similarity
    cos_sim = F.cosine_similarity(features_dna, features_protein, dim=-1).numpy()

    results = {
        "gene_names": common,
        "n_genes": len(common),
        "n_features": n_features,
        "shared_feature_indices": shared_indices,
        "marginal_shared_indices": marginal_shared,
        "dna_specific_indices": dna_specific,
        "protein_specific_indices": prot_specific,
        "dead_feature_indices": dead_features,
        "n_shared": len(shared_indices),
        "n_marginal_shared": len(marginal_shared),
        "n_dna_specific": len(dna_specific),
        "n_protein_specific": len(prot_specific),
        "n_dead": len(dead_features),
        "feature_correlations": feat_correlations,
        "feature_activation_freq_dna": freq_dna,
        "feature_activation_freq_protein": freq_protein,
        "crossmodal_cosine_sim": cos_sim,
        "mean_crossmodal_sim": float(cos_sim.mean()),
        "features_dna": features_dna.numpy(),
        "features_protein": features_protein.numpy(),
    }

    logger.info(
        f"Feature statistics: {len(shared_indices)} shared, "
        f"{len(dna_specific)} DNA-specific, {len(prot_specific)} protein-specific, "
        f"{len(dead_features)} dead. Mean cross-modal sim: {cos_sim.mean():.4f}"
    )

    return results


# =============================================================================
# Task 1: Dual-modality pathogenic variant prediction
# =============================================================================

class VariantPredictor:
    """
    Predict pathogenicity of coding DNA mutations using CrossBioSAE features.

    For each variant:
    1. Get wild-type DNA and protein feature patterns
    2. Get mutant DNA and protein feature patterns
    3. Compute feature disruption scores from both modalities
    4. Combine for pathogenicity prediction

    Key insight: if both DNA and protein SAE features are disrupted (shared features),
    the variant is more likely pathogenic.
    """

    def __init__(
        self,
        model: CrossBioSAE,
        protein_extractor,  # ActivationExtractor for ESM-2
        dna_extractor,      # ActivationExtractor for Evo-2
        device: str = "cuda",
    ):
        self.model = model
        self.protein_extractor = protein_extractor
        self.dna_extractor = dna_extractor
        self.device = device

    def predict_variant(
        self,
        wt_protein_seq: str,
        mt_protein_seq: str,
        wt_cds_seq: str,
        mt_cds_seq: str,
        gene_name: str = "",
    ) -> dict:
        """
        Predict pathogenicity of a single variant.

        Args:
            wt_protein_seq: Wild-type protein sequence
            mt_protein_seq: Mutant protein sequence
            wt_cds_seq: Wild-type CDS sequence
            mt_cds_seq: Mutant CDS sequence
            gene_name: Gene identifier

        Returns:
            Dict with prediction scores and disrupted features
        """
        self.model.eval()
        with torch.no_grad():
            # Extract activations (this would normally use the extractor)
            # For batch processing, pre-computed activations are used instead
            pass

        # This is a placeholder for the full pipeline
        return {}

    def evaluate_clinvar(
        self,
        clinvar_path: str,
        wt_protein_h5: str,
        mt_protein_h5: str,
        wt_dna_h5: str,
        mt_dna_h5: str,
        output_path: str,
    ) -> dict:
        """
        Evaluate on ClinVar temporal holdout dataset.

        ClinVar file should contain:
        - variant_id, gene, protein_change, cds_change, pathogenicity (0/1)
        - Only variants from 2025-2026 (post-training cutoff)

        Args:
            clinvar_path: Path to ClinVar holdout CSV
            wt_protein_h5: Pre-computed wild-type protein activations
            mt_protein_h5: Pre-computed mutant protein activations
            wt_dna_h5: Pre-computed wild-type DNA activations
            mt_dna_h5: Pre-computed mutant DNA activations
            output_path: Path to save results

        Returns:
            Dict with AUROC, AUPRC, and per-feature disruption statistics
        """
        self.model.eval()

        # Load ClinVar labels
        clinvar = pd.read_csv(clinvar_path)
        labels = clinvar["pathogenicity"].values

        # Load pre-computed activations
        with h5py.File(wt_protein_h5, "r") as f:
            wt_prot_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
        with h5py.File(mt_protein_h5, "r") as f:
            mt_prot_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
        with h5py.File(wt_dna_h5, "r") as f:
            wt_dna_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
        with h5py.File(mt_dna_h5, "r") as f:
            mt_dna_acts = torch.tensor(f["activations"][:], dtype=torch.float32)

        # Compute feature disruptions
        all_scores = []
        all_disruptions = []

        with torch.no_grad():
            batch_size = 256
            for start in tqdm(range(0, len(labels), batch_size), desc="ClinVar eval"):
                end = min(start + batch_size, len(labels))

                # Wild-type features
                wt_feat_p = self.model.encode_protein(wt_prot_acts[start:end].to(self.device))
                wt_feat_d = self.model.encode_dna(wt_dna_acts[start:end].to(self.device))

                # Mutant features
                mt_feat_p = self.model.encode_protein(mt_prot_acts[start:end].to(self.device))
                mt_feat_d = self.model.encode_dna(mt_dna_acts[start:end].to(self.device))

                # Feature disruption: |WT - MT| for each modality
                disrupt_p = (wt_feat_p - mt_feat_p).abs()
                disrupt_d = (wt_feat_d - mt_feat_d).abs()

                # Pathogenicity score: max disruption across shared features
                # Combined disruption = element-wise max of both modalities
                combined_disrupt = torch.max(disrupt_p, disrupt_d)
                score = combined_disrupt.sum(dim=-1)  # Total disruption

                all_scores.append(score.cpu().numpy())
                all_disruptions.append(combined_disrupt.cpu().numpy())

        scores = np.concatenate(all_scores)
        disruptions = np.concatenate(all_disruptions)

        # Compute metrics
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        results = {
            "auroc": auroc,
            "auprc": auprc,
            "n_variants": len(labels),
            "n_pathogenic": int(labels.sum()),
            "n_benign": int((1 - labels).sum()),
            "mean_score_pathogenic": float(scores[labels == 1].mean()),
            "mean_score_benign": float(scores[labels == 0].mean()),
        }

        logger.info(
            f"ClinVar evaluation: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
            f"N={len(labels)} ({int(labels.sum())} pathogenic)"
        )

        # Save detailed results
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "variant_scores.npy", scores)
        np.save(output_dir / "feature_disruptions.npy", disruptions)

        results_df = clinvar.copy()
        results_df["crossbiosae_score"] = scores
        results_df.to_csv(output_dir / "predictions.csv", index=False)

        return results


# =============================================================================
# Task 2: Function prediction for uncharacterized genes
# =============================================================================

class FunctionPredictor:
    """
    Predict gene functions using CrossBioSAE shared features.

    Approach:
    1. Train SAE on characterized genes with known GO terms
    2. Auto-label each SAE feature with enriched GO terms
    3. For uncharacterized genes, predict function from active features
    """

    def __init__(
        self,
        model: CrossBioSAE,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.feature_labels = None

    def label_features(
        self,
        gene_features: np.ndarray,
        gene_names: list[str],
        go_annotations: dict[str, list[str]],
        min_genes_per_feature: int = 10,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Auto-label each SAE feature with enriched GO terms.

        For each feature, find genes where it is active, then compute
        enrichment of each GO term using Fisher's exact test.

        Args:
            gene_features: (n_genes, n_features) feature activation matrix
            gene_names: Gene names matching rows of gene_features
            go_annotations: Dict mapping gene_name -> list of GO term IDs
            min_genes_per_feature: Skip features active in fewer genes

        Returns:
            Dict mapping feature_index -> list of (GO_term, p_value)
        """
        n_genes, n_features = gene_features.shape
        feature_labels = {}

        # Binary activation matrix
        active = gene_features > 0
        n_total = n_genes

        # Get all GO terms
        all_go_terms = set()
        for terms in go_annotations.values():
            all_go_terms.update(terms)

        # Gene-to-GO binary matrix
        gene_to_idx = {n: i for i, n in enumerate(gene_names)}
        go_terms_list = sorted(all_go_terms)
        go_term_to_idx = {t: i for i, t in enumerate(go_terms_list)}
        go_matrix = np.zeros((n_genes, len(go_terms_list)), dtype=bool)

        for gene, terms in go_annotations.items():
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                for term in terms:
                    if term in go_term_to_idx:
                        go_matrix[idx, go_term_to_idx[term]] = True

        # Only label active features (skip dead ones)
        feat_counts = active.sum(axis=0)
        active_feat_indices = np.where(feat_counts >= min_genes_per_feature)[0]
        logger.info(
            f"Labeling {len(active_feat_indices)} active features "
            f"(of {n_features} total) against {len(go_terms_list)} GO terms"
        )

        # Vectorized: compute overlap matrix (n_active_features x n_go_terms)
        active_subset = active[:, active_feat_indices].astype(np.float32)  # (n_genes, n_active_feat)
        go_float = go_matrix.astype(np.float32)  # (n_genes, n_go)

        # a = overlap count: feature active AND GO annotated
        overlap = active_subset.T @ go_float  # (n_active_feat, n_go)
        feat_sums = active_subset.sum(axis=0)  # (n_active_feat,)
        go_sums = go_float.sum(axis=0)  # (n_go,)

        # Hypergeometric p-value via scipy (vectorized per feature)
        from scipy.stats import hypergeom
        from statsmodels.stats.multitest import multipletests

        # Collect candidate p-values and count total tests for proper FDR
        all_pvalues = []  # (feat_idx, go_idx, p_value)
        total_tests = 0

        for i, feat_idx in enumerate(tqdm(active_feat_indices, desc="Computing enrichment p-values")):
            n_feat = int(feat_sums[i])
            for go_idx in range(len(go_terms_list)):
                a = int(overlap[i, go_idx])
                if a < 3:
                    continue
                total_tests += 1
                n_go = int(go_sums[go_idx])
                p_value = hypergeom.sf(a - 1, n_genes, n_go, n_feat)
                if p_value < 0.05:
                    all_pvalues.append((int(feat_idx), go_idx, float(p_value)))

        logger.info(f"Collected {len(all_pvalues)} candidate enrichments (p < 0.05) from {total_tests} total tests")

        # BH-FDR correction accounting for ALL tested hypotheses
        # Adjust p-values by total_tests / len(candidates) to correct for pre-filtering
        if all_pvalues:
            raw_ps = np.array([x[2] for x in all_pvalues])
            # Scale factor: we only kept p<0.05 candidates, but BH needs to account for all tests
            correction_factor = total_tests / len(raw_ps) if len(raw_ps) > 0 else 1.0
            adjusted_ps = np.minimum(raw_ps * correction_factor, 1.0)
            reject, fdr_corrected, _, _ = multipletests(adjusted_ps, alpha=0.05, method="fdr_bh")

            n_significant = reject.sum()
            logger.info(f"After BH-FDR correction (alpha=0.05): {n_significant} significant enrichments")

            for idx, (feat_idx, go_idx, raw_p) in enumerate(all_pvalues):
                if not reject[idx]:
                    continue
                if feat_idx not in feature_labels:
                    feature_labels[feat_idx] = []
                feature_labels[feat_idx].append((
                    go_terms_list[go_idx], float(fdr_corrected[idx])
                ))

            # Sort and keep top 5 per feature
            for feat_idx in feature_labels:
                feature_labels[feat_idx].sort(key=lambda x: x[1])
                feature_labels[feat_idx] = feature_labels[feat_idx][:5]

        self.feature_labels = feature_labels
        logger.info(f"Labeled {len(feature_labels)} features with FDR-corrected GO term enrichments")
        return feature_labels

    def predict_function(
        self,
        dna_h5_path: str,
        protein_h5_path: str,
        output_path: str,
    ) -> pd.DataFrame:
        """
        Predict functions for all genes based on active features.

        Returns DataFrame with columns: gene_name, predicted_go_terms, confidence
        """
        if self.feature_labels is None:
            raise RuntimeError("Call label_features() first")

        self.model.eval()

        with h5py.File(dna_h5_path, "r") as f:
            dna_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
            gene_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]

        with h5py.File(protein_h5_path, "r") as f:
            prot_acts = torch.tensor(f["activations"][:], dtype=torch.float32)

        predictions = []

        with torch.no_grad():
            for i in tqdm(range(len(gene_names)), desc="Predicting functions"):
                dna_feat = self.model.encode_dna(dna_acts[i:i+1].to(self.device))
                prot_feat = self.model.encode_protein(prot_acts[i:i+1].to(self.device))

                # Combine features from both modalities
                combined = (dna_feat + prot_feat) / 2.0
                active_features = torch.where(combined[0] > 0)[0].cpu().numpy()

                # Collect GO terms from active labeled features
                go_scores = {}
                for feat_idx in active_features:
                    feat_idx = int(feat_idx)
                    if feat_idx in self.feature_labels:
                        for go_term, p_val in self.feature_labels[feat_idx]:
                            if go_term not in go_scores:
                                go_scores[go_term] = 0.0
                            go_scores[go_term] += -np.log10(p_val + 1e-300)

                # Sort by aggregated score
                sorted_terms = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)

                predictions.append({
                    "gene_name": gene_names[i],
                    "predicted_go_terms": "|".join(t[0] for t in sorted_terms[:10]),
                    "go_scores": "|".join(f"{t[1]:.2f}" for t in sorted_terms[:10]),
                    "n_active_features": len(active_features),
                    "n_labeled_active": sum(
                        1 for f in active_features if int(f) in self.feature_labels
                    ),
                })

        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved function predictions for {len(df)} genes to {output_path}")
        return df


# =============================================================================
# Task 3: Cross-modal anomaly detection
# =============================================================================

class AnomalyDetector:
    """
    Detect genes with inconsistent DNA and protein feature patterns.

    Biological meaning of inconsistency:
    - Alternative splicing
    - Overlapping regulatory elements
    - Annotation errors
    - Novel biological mechanisms
    """

    def __init__(
        self,
        model: CrossBioSAE,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device

    def compute_anomaly_scores(
        self,
        dna_h5_path: str,
        protein_h5_path: str,
        output_path: str,
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """
        Compute cross-modal consistency scores for all genes.

        A low consistency score means the DNA and protein models "disagree"
        about the gene, which could indicate interesting biology.

        Returns:
            DataFrame with columns: gene_name, consistency_score, anomaly_rank,
            n_shared_active, n_dna_only, n_protein_only
        """
        self.model.eval()

        # Load activations
        with h5py.File(dna_h5_path, "r") as f:
            dna_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
            dna_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]

        with h5py.File(protein_h5_path, "r") as f:
            prot_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
            prot_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]

        # Match genes
        dna_idx_map = {n: i for i, n in enumerate(dna_names)}
        prot_idx_map = {n: i for i, n in enumerate(prot_names)}
        common = sorted(set(dna_names) & set(prot_names))

        results = []

        with torch.no_grad():
            for start in tqdm(range(0, len(common), batch_size), desc="Computing anomaly scores"):
                end = min(start + batch_size, len(common))
                batch_genes = common[start:end]

                d_idx = [dna_idx_map[g] for g in batch_genes]
                p_idx = [prot_idx_map[g] for g in batch_genes]

                d_batch = dna_acts[d_idx].to(self.device)
                p_batch = prot_acts[p_idx].to(self.device)

                feat_d = self.model.encode_dna(d_batch)
                feat_p = self.model.encode_protein(p_batch)

                # Cosine similarity
                cos_sim = F.cosine_similarity(feat_d, feat_p, dim=-1)

                # Feature pattern analysis
                active_d = (feat_d > 0).float()
                active_p = (feat_p > 0).float()

                shared_active = (active_d * active_p).sum(dim=-1)
                dna_only = (active_d * (1 - active_p)).sum(dim=-1)
                protein_only = ((1 - active_d) * active_p).sum(dim=-1)

                # Jensen-Shannon divergence of normalized feature patterns
                p_norm = F.softmax(feat_d, dim=-1) + 1e-8
                q_norm = F.softmax(feat_p, dim=-1) + 1e-8
                m = (p_norm + q_norm) / 2
                jsd = 0.5 * (
                    F.kl_div(m.log(), p_norm, reduction="none").sum(dim=-1) +
                    F.kl_div(m.log(), q_norm, reduction="none").sum(dim=-1)
                )

                for i, gene in enumerate(batch_genes):
                    results.append({
                        "gene_name": gene,
                        "consistency_score": cos_sim[i].item(),
                        "jsd_score": jsd[i].item(),
                        "n_shared_active": int(shared_active[i].item()),
                        "n_dna_only": int(dna_only[i].item()),
                        "n_protein_only": int(protein_only[i].item()),
                    })

        df = pd.DataFrame(results)

        # Rank by anomaly (lowest consistency = most anomalous)
        df = df.sort_values("consistency_score", ascending=True)
        df["anomaly_rank"] = range(1, len(df) + 1)

        # Also rank by JSD (highest JSD = most anomalous)
        df["jsd_rank"] = df["jsd_score"].rank(ascending=False).astype(int)

        # Combined rank (average of both rankings)
        df["combined_rank"] = ((df["anomaly_rank"] + df["jsd_rank"]) / 2).rank().astype(int)

        # Save
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "anomaly_scores.csv", index=False)

        # Summary
        top100 = df.head(100)
        logger.info(
            f"Anomaly detection complete: {len(df)} genes scored. "
            f"Top-100 mean consistency: {top100['consistency_score'].mean():.4f}, "
            f"Bottom-100 mean consistency: {df.tail(100)['consistency_score'].mean():.4f}"
        )

        # Systematic gene-family enrichment analysis
        self._analyze_gene_family_enrichment(df, output_dir)

        return df

    def _analyze_gene_family_enrichment(self, df: pd.DataFrame, output_dir: Path):
        """Check if specific gene families are over/under-represented in anomalies."""
        prefixes = ["ZNF", "OR", "KRT", "HLA", "SLC", "KRTAP", "TAS", "RPS", "RPL",
                     "MT-", "HIST", "TMEM", "FAM", "LINC", "LOC", "ADAM", "MYH"]

        n_genes = len(df)
        top200 = set(df.head(200)["gene_name"])
        bottom200 = set(df.tail(200)["gene_name"])

        enrichment_results = []
        for prefix in prefixes:
            family_genes = set(df[df["gene_name"].str.startswith(prefix)]["gene_name"])
            if len(family_genes) < 3:
                continue

            top_overlap = len(top200 & family_genes)
            bottom_overlap = len(bottom200 & family_genes)
            expected = len(family_genes) * 200 / n_genes

            if top_overlap > 0 or bottom_overlap > 0:
                from scipy.stats import fisher_exact
                _, p_top = fisher_exact([
                    [top_overlap, 200 - top_overlap],
                    [len(family_genes) - top_overlap, n_genes - 200 - (len(family_genes) - top_overlap)],
                ], alternative="greater")

                enrichment_results.append({
                    "family": prefix,
                    "n_family_genes": len(family_genes),
                    "in_top200_anomalous": top_overlap,
                    "in_bottom200_consistent": bottom_overlap,
                    "expected": round(expected, 1),
                    "fold_enrichment_top": round(top_overlap / max(expected, 0.1), 2),
                    "p_value_top": p_top,
                })

        if enrichment_results:
            enrich_df = pd.DataFrame(enrichment_results).sort_values("p_value_top")
            enrich_df.to_csv(output_dir / "gene_family_enrichment.csv", index=False)
            logger.info("Gene family enrichment in top-200 anomalous genes:")
            for _, row in enrich_df.iterrows():
                if row["fold_enrichment_top"] > 1.5 or row["p_value_top"] < 0.05:
                    logger.info(
                        f"  {row['family']:8s}: {row['in_top200_anomalous']}/{row['n_family_genes']} in top-200 "
                        f"(expected {row['expected']}, fold={row['fold_enrichment_top']}x, p={row['p_value_top']:.4f})"
                    )

    def validate_against_databases(
        self,
        anomaly_df: pd.DataFrame,
        alt_splicing_db: Optional[str] = None,
        appris_db: Optional[str] = None,
        output_path: str = "results/validation",
    ) -> dict:
        """
        Cross-reference top anomalous genes against known databases.

        Args:
            anomaly_df: Output from compute_anomaly_scores()
            alt_splicing_db: Path to Alternative Splicing Database file
            appris_db: Path to APPRIS principal isoform database
            output_path: Directory for validation results

        Returns:
            Dict with enrichment statistics
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"n_genes_tested": len(anomaly_df)}
        top100 = anomaly_df.head(100)["gene_name"].tolist()
        bottom100 = anomaly_df.tail(100)["gene_name"].tolist()

        # Alternative splicing enrichment
        if alt_splicing_db and Path(alt_splicing_db).exists():
            as_genes = set(pd.read_csv(alt_splicing_db)["gene_name"].tolist())

            top_overlap = len(set(top100) & as_genes)
            bottom_overlap = len(set(bottom100) & as_genes)
            bg_rate = len(as_genes) / len(anomaly_df)

            # Fisher's exact test
            n_total = len(anomaly_df)
            _, p_val = stats.fisher_exact([
                [top_overlap, 100 - top_overlap],
                [len(as_genes) - top_overlap, n_total - 100 - (len(as_genes) - top_overlap)],
            ], alternative="greater")

            results["alt_splicing"] = {
                "top100_overlap": top_overlap,
                "bottom100_overlap": bottom_overlap,
                "background_rate": bg_rate,
                "enrichment_p_value": p_val,
                "fold_enrichment": (top_overlap / 100) / max(bg_rate, 1e-10),
            }

            logger.info(
                f"Alt splicing enrichment: top-100 has {top_overlap}/100 "
                f"(bg rate {bg_rate:.3f}), p={p_val:.2e}"
            )

        # APPRIS annotation enrichment (genes with multiple principal isoforms)
        if appris_db and Path(appris_db).exists():
            appris = pd.read_csv(appris_db, sep="\t")
            multi_isoform = set(
                appris.groupby("gene_name").filter(
                    lambda x: len(x) > 1
                )["gene_name"].unique()
            )

            top_overlap = len(set(top100) & multi_isoform)
            bottom_overlap = len(set(bottom100) & multi_isoform)

            results["appris_multi_isoform"] = {
                "top100_overlap": top_overlap,
                "bottom100_overlap": bottom_overlap,
            }

        return results


# =============================================================================
# Pilot evaluation
# =============================================================================

def run_pilot_evaluation(
    model: CrossBioSAE,
    dna_h5_path: str,
    protein_h5_path: str,
    output_dir: str,
    device: str = "cuda",
) -> dict:
    """
    Run minimal pilot evaluation to assess feasibility.

    This is the Phase 0 check: does the cross-modal SAE learn meaningful shared features?

    Checks:
    1. How many features are shared vs modality-specific?
    2. What is the cross-modal alignment score?
    3. Is it above a random permutation baseline?
    4. Do shared features correspond to any known biology?
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running pilot evaluation...")

    # 1. Feature statistics
    feat_stats = compute_feature_statistics(
        model, dna_h5_path, protein_h5_path, device=device
    )

    # 2. Permutation baseline
    logger.info("Computing permutation baseline...")
    n_perms = 100
    perm_sims = []

    with h5py.File(dna_h5_path, "r") as f:
        dna_acts_all = torch.tensor(f["activations"][:], dtype=torch.float32)
    with h5py.File(protein_h5_path, "r") as f:
        prot_acts_all = torch.tensor(f["activations"][:], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        dna_feats = model.encode_dna(dna_acts_all.to(device)).cpu()
        prot_feats = model.encode_protein(prot_acts_all.to(device)).cpu()

        for _ in range(n_perms):
            perm_idx = torch.randperm(len(prot_feats))
            perm_sim = F.cosine_similarity(
                dna_feats, prot_feats[perm_idx], dim=-1
            ).mean().item()
            perm_sims.append(perm_sim)

    perm_mean = np.mean(perm_sims)
    perm_std = np.std(perm_sims)
    actual_sim = feat_stats["mean_crossmodal_sim"]
    z_score = (actual_sim - perm_mean) / max(perm_std, 1e-10)

    pilot_results = {
        "n_genes": feat_stats["n_genes"],
        "n_features": feat_stats["n_features"],
        "n_shared_features": feat_stats["n_shared"],
        "n_dna_specific_features": feat_stats["n_dna_specific"],
        "n_protein_specific_features": feat_stats["n_protein_specific"],
        "n_dead_features": feat_stats["n_dead"],
        "mean_crossmodal_sim": actual_sim,
        "permutation_baseline_mean": perm_mean,
        "permutation_baseline_std": perm_std,
        "z_score": z_score,
        "p_value": float(stats.norm.sf(z_score)),
        "shared_fraction": feat_stats["n_shared"] / max(feat_stats["n_features"], 1),
    }

    # 3. Feasibility assessment
    feasible = (
        z_score > 2.0  # Significantly above permutation baseline
        and feat_stats["n_shared"] > 10  # At least 10 shared features
        and feat_stats["n_dead"] < feat_stats["n_features"] * 0.9  # Not too many dead features
    )
    pilot_results["feasible"] = feasible

    if feasible:
        logger.info(
            f"PILOT PASSED: z-score={z_score:.2f}, "
            f"{feat_stats['n_shared']} shared features, "
            f"mean cross-modal sim={actual_sim:.4f} (baseline {perm_mean:.4f})"
        )
    else:
        logger.warning(
            f"PILOT MARGINAL/FAILED: z-score={z_score:.2f}, "
            f"{feat_stats['n_shared']} shared features, "
            f"mean cross-modal sim={actual_sim:.4f} (baseline {perm_mean:.4f}). "
            f"Consider adjusting cross-modal loss weight or architecture."
        )

    # Save results
    import json
    with open(output_path / "pilot_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in pilot_results.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable[k] = float(v)
            elif isinstance(v, np.bool_):
                serializable[k] = bool(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)

    return pilot_results
