"""
Post-hoc alignment methods for independently trained SAEs.

Evaluates whether features learned independently by separate modality SAEs
are alignable without any cross-modal training signal. This addresses the
reviewer concern that joint training with cross-modal loss creates circular
evidence for alignment.

Methods:
  1. CCA (Canonical Correlation Analysis): finds maximally correlated subspaces
  2. Orthogonal Procrustes: best orthogonal rotation to align feature spaces
  3. Linear Probe: learned linear map via ridge regression
  4. Cross-Modal Retrieval: given protein features, retrieve matching DNA gene

Dimension mismatch handling (protein n_features != DNA n_features):
  - CCA: naturally handles different dimensions
  - Procrustes: PCA to min(dim_a, dim_b) then align
  - LinearProbe: ridge regression handles any dim mapping
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CCAAlignment:
    """
    Canonical Correlation Analysis alignment.

    Finds linear projections of two feature matrices that maximize
    their correlation. Naturally handles different input dimensions.
    """

    def __init__(self, n_components: int = 50, max_iter: int = 1000, pca_dim: int = 256):
        """
        Args:
            n_components: Number of CCA components (canonical dimensions).
            max_iter: Maximum iterations for CCA convergence.
            pca_dim: PCA pre-reduction dimension (SAE features are very high-dim and sparse).
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.pca_dim = pca_dim
        self.cca = None
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()
        self.pca_a = None
        self.pca_b = None

    def fit(self, features_a: np.ndarray, features_b: np.ndarray):
        """
        Fit CCA on paired feature matrices.
        Uses PCA pre-reduction to avoid sklearn CCA being extremely slow on high-dim sparse data.
        """
        n_samples = features_a.shape[0]

        # PCA pre-reduction for high-dimensional features
        pca_target = min(self.pca_dim, n_samples - 1, features_a.shape[1], features_b.shape[1])
        if features_a.shape[1] > pca_target:
            logger.info(f"CCA: PCA pre-reducing A from {features_a.shape[1]} to {pca_target}")
            self.pca_a = PCA(n_components=pca_target)
            features_a = self.pca_a.fit_transform(features_a)
        if features_b.shape[1] > pca_target:
            logger.info(f"CCA: PCA pre-reducing B from {features_b.shape[1]} to {pca_target}")
            self.pca_b = PCA(n_components=pca_target)
            features_b = self.pca_b.fit_transform(features_b)

        max_components = min(n_samples, features_a.shape[1], features_b.shape[1])
        actual_components = min(self.n_components, max_components)

        if actual_components < self.n_components:
            logger.warning(
                f"CCA: reducing n_components from {self.n_components} to {actual_components} "
                f"(n_samples={n_samples}, dim_a={features_a.shape[1]}, dim_b={features_b.shape[1]})"
            )

        # Standardize before CCA
        fa = self.scaler_a.fit_transform(features_a)
        fb = self.scaler_b.fit_transform(features_b)

        self.cca = CCA(n_components=actual_components, max_iter=self.max_iter)
        self.cca.fit(fa, fb)
        return self

    def transform(
        self, features_a: np.ndarray, features_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project both feature matrices into the CCA space.

        Returns:
            (projected_a, projected_b) each of shape (n_samples, n_components)
        """
        if self.pca_a is not None:
            features_a = self.pca_a.transform(features_a)
        if self.pca_b is not None:
            features_b = self.pca_b.transform(features_b)
        fa = self.scaler_a.transform(features_a)
        fb = self.scaler_b.transform(features_b)
        proj_a, proj_b = self.cca.transform(fa, fb)
        return proj_a, proj_b

    def score(self, features_a: np.ndarray, features_b: np.ndarray) -> dict[str, float]:
        """
        Compute alignment scores in CCA space.

        Returns:
            Dict with:
            - mean_correlation: average canonical correlation across components
            - top1_correlation: first canonical correlation (highest)
            - mean_cosine_sim: average cosine similarity in CCA space
        """
        proj_a, proj_b = self.transform(features_a, features_b)

        # Per-component Pearson correlation
        correlations = []
        for i in range(proj_a.shape[1]):
            r = np.corrcoef(proj_a[:, i], proj_b[:, i])[0, 1]
            correlations.append(r)
        correlations = np.array(correlations)

        # Cosine similarity in CCA space
        norms_a = np.linalg.norm(proj_a, axis=1, keepdims=True) + 1e-8
        norms_b = np.linalg.norm(proj_b, axis=1, keepdims=True) + 1e-8
        cos_sims = (proj_a / norms_a * proj_b / norms_b).sum(axis=1)

        return {
            "cca_mean_correlation": float(np.mean(correlations)),
            "cca_top1_correlation": float(correlations[0]) if len(correlations) > 0 else 0.0,
            "cca_top5_correlation": float(np.mean(correlations[:5])) if len(correlations) >= 5 else float(np.mean(correlations)),
            "cca_mean_cosine_sim": float(np.mean(cos_sims)),
        }


class ProcrustesAlignment:
    """
    Orthogonal Procrustes alignment.

    Finds the best orthogonal rotation R that minimizes ||A - B @ R||_F.
    Since Procrustes requires equal dimensions, we first reduce both
    feature sets to a common dimension via PCA.
    """

    def __init__(self, n_components: Optional[int] = 256):
        """
        Args:
            n_components: Common PCA dimension. Default 256 to avoid very slow SVD on large dims.
        """
        self.n_components = n_components
        self.pca_a = None
        self.pca_b = None
        self.R = None  # Orthogonal rotation matrix
        self.scale = None
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()

    def fit(self, features_a: np.ndarray, features_b: np.ndarray):
        """
        Fit Procrustes rotation.

        1. PCA both to common dimension
        2. Find optimal orthogonal R: min ||PCA(A) - PCA(B) @ R||

        Args:
            features_a: (n_samples, dim_a) target features
            features_b: (n_samples, dim_b) features to rotate
        """
        n_samples = features_a.shape[0]
        dim_a, dim_b = features_a.shape[1], features_b.shape[1]

        if self.n_components is None:
            self.n_components = min(dim_a, dim_b, n_samples)
        actual_components = min(self.n_components, n_samples, dim_a, dim_b)

        # Standardize
        fa = self.scaler_a.fit_transform(features_a)
        fb = self.scaler_b.fit_transform(features_b)

        # PCA to common dimension
        self.pca_a = PCA(n_components=actual_components)
        self.pca_b = PCA(n_components=actual_components)
        pa = self.pca_a.fit_transform(fa)
        pb = self.pca_b.fit_transform(fb)

        # Orthogonal Procrustes: find R such that ||pa - pb @ R|| is minimized
        self.R, self.scale = orthogonal_procrustes(pb, pa)

        logger.info(
            f"Procrustes: PCA dim={actual_components}, "
            f"explained variance: A={self.pca_a.explained_variance_ratio_.sum():.3f}, "
            f"B={self.pca_b.explained_variance_ratio_.sum():.3f}"
        )
        return self

    def transform(self, features_b: np.ndarray) -> np.ndarray:
        """
        Apply Procrustes rotation to features_b to align with features_a's space.

        Args:
            features_b: (n_samples, dim_b) features to transform

        Returns:
            (n_samples, n_components) aligned features
        """
        fb = self.scaler_b.transform(features_b)
        pb = self.pca_b.transform(fb)
        return pb @ self.R

    def transform_a(self, features_a: np.ndarray) -> np.ndarray:
        """Project features_a into the PCA space (no rotation needed)."""
        fa = self.scaler_a.transform(features_a)
        return self.pca_a.transform(fa)

    def score(self, features_a: np.ndarray, features_b: np.ndarray) -> dict[str, float]:
        """
        Compute alignment scores after Procrustes rotation.

        Returns:
            Dict with:
            - procrustes_disparity: Frobenius norm of residual (lower is better)
            - procrustes_mean_cosine_sim: cosine similarity after alignment
            - procrustes_r2: R-squared (variance explained by alignment)
        """
        pa = self.transform_a(features_a)
        pb_aligned = self.transform(features_b)

        # Disparity: ||A - B@R||_F / ||A||_F
        residual = np.linalg.norm(pa - pb_aligned, "fro")
        norm_a = np.linalg.norm(pa, "fro") + 1e-8
        disparity = residual / norm_a

        # Cosine similarity per sample
        norms_a = np.linalg.norm(pa, axis=1, keepdims=True) + 1e-8
        norms_b = np.linalg.norm(pb_aligned, axis=1, keepdims=True) + 1e-8
        cos_sims = (pa / norms_a * pb_aligned / norms_b).sum(axis=1)

        # R-squared
        ss_res = np.sum((pa - pb_aligned) ** 2)
        ss_tot = np.sum((pa - pa.mean(axis=0)) ** 2) + 1e-8
        r2 = 1.0 - ss_res / ss_tot

        return {
            "procrustes_disparity": float(disparity),
            "procrustes_mean_cosine_sim": float(np.mean(cos_sims)),
            "procrustes_r2": float(r2),
        }


class LinearProbeAlignment:
    """
    Learned linear map from modality A features to modality B features
    via ridge regression.

    This tests whether a simple linear transform can predict one modality's
    features from the other, indicating shared representational structure.
    """

    def __init__(self, alpha: float = 1.0, pca_dim: int = 256):
        """
        Args:
            alpha: Ridge regression regularization strength.
            pca_dim: PCA pre-reduction dimension for efficiency.
        """
        self.alpha = alpha
        self.pca_dim = pca_dim
        self.model = None
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()
        self.pca_a = None
        self.pca_b = None

    def fit(self, features_a: np.ndarray, features_b: np.ndarray):
        """
        Fit ridge regression: PCA(A) -> PCA(B).
        PCA pre-reduction avoids fitting a huge weight matrix on high-dim sparse features.
        """
        n_samples = features_a.shape[0]

        # PCA pre-reduction
        pca_target_a = min(self.pca_dim, n_samples - 1, features_a.shape[1])
        pca_target_b = min(self.pca_dim, n_samples - 1, features_b.shape[1])

        if features_a.shape[1] > pca_target_a:
            self.pca_a = PCA(n_components=pca_target_a)
            features_a = self.pca_a.fit_transform(features_a)
        if features_b.shape[1] > pca_target_b:
            self.pca_b = PCA(n_components=pca_target_b)
            features_b = self.pca_b.fit_transform(features_b)

        fa = self.scaler_a.fit_transform(features_a)
        fb = self.scaler_b.fit_transform(features_b)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(fa, fb)

        logger.info(
            f"LinearProbe: {features_a.shape[1]} -> {features_b.shape[1]}, "
            f"alpha={self.alpha}"
        )
        return self

    def transform(self, features_a: np.ndarray) -> np.ndarray:
        """
        Predict B features from A features.
        """
        if self.pca_a is not None:
            features_a = self.pca_a.transform(features_a)
        fa = self.scaler_a.transform(features_a)
        return self.model.predict(fa)

    def score(self, features_a: np.ndarray, features_b: np.ndarray) -> dict[str, float]:
        """
        Compute alignment scores via linear probe.

        Returns:
            Dict with:
            - linear_probe_r2: R-squared of the ridge regression prediction
            - linear_probe_mse: mean squared error of prediction
            - linear_probe_cosine_sim: cosine similarity of predicted vs actual
        """
        predicted_b = self.transform(features_a)
        if self.pca_b is not None:
            features_b = self.pca_b.transform(features_b)
        fb = self.scaler_b.transform(features_b)

        # R-squared
        ss_res = np.sum((fb - predicted_b) ** 2)
        ss_tot = np.sum((fb - fb.mean(axis=0)) ** 2) + 1e-8
        r2 = 1.0 - ss_res / ss_tot

        # MSE
        mse = float(np.mean((fb - predicted_b) ** 2))

        # Cosine similarity per sample
        norms_pred = np.linalg.norm(predicted_b, axis=1, keepdims=True) + 1e-8
        norms_true = np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8
        cos_sims = (predicted_b / norms_pred * fb / norms_true).sum(axis=1)

        return {
            "linear_probe_r2": float(r2),
            "linear_probe_mse": mse,
            "linear_probe_cosine_sim": float(np.mean(cos_sims)),
        }


class CrossModalRetrieval:
    """
    Cross-modal retrieval evaluation.

    Given a protein feature vector for gene X, retrieve the matching DNA
    feature vector from a gallery of all genes. Reports recall@k and MRR.

    This is the strongest test of alignment: can we find the same gene
    across modalities without any cross-modal training?
    """

    def __init__(self, metric: str = "cosine"):
        """
        Args:
            metric: Distance metric for retrieval ("cosine", "euclidean").
        """
        self.metric = metric

    def evaluate(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        gene_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Evaluate cross-modal retrieval: for each sample in A, find its match in B.

        Args:
            features_a: (n_samples, dim_a) query features (e.g., protein)
            features_b: (n_samples, dim_b) gallery features (e.g., DNA)
                        Rows are assumed to be aligned (row i in A matches row i in B).
            gene_names: Optional gene name labels (for logging).

        Returns:
            Dict with recall@1, recall@5, recall@10, MRR, and median_rank.
        """
        n = features_a.shape[0]

        if features_a.shape[1] != features_b.shape[1]:
            # Dimensions differ; project both to min dimension via PCA
            min_dim = min(features_a.shape[1], features_b.shape[1], n)
            logger.info(
                f"CrossModalRetrieval: dim mismatch ({features_a.shape[1]} vs {features_b.shape[1]}), "
                f"PCA to {min_dim}"
            )
            pca_a = PCA(n_components=min_dim)
            pca_b = PCA(n_components=min_dim)
            features_a = pca_a.fit_transform(features_a)
            features_b = pca_b.fit_transform(features_b)

        # Compute pairwise distance matrix: (n, n)
        # dist[i, j] = distance between query i and gallery j
        if self.metric == "cosine":
            dist_matrix = cdist(features_a, features_b, metric="cosine")
        elif self.metric == "euclidean":
            dist_matrix = cdist(features_a, features_b, metric="euclidean")
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # For each query i, rank all gallery items
        # True match is at index i
        ranks = []
        for i in range(n):
            distances = dist_matrix[i]
            sorted_indices = np.argsort(distances)
            rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed rank
            ranks.append(rank)

        ranks = np.array(ranks)

        recall_at_1 = float(np.mean(ranks <= 1))
        recall_at_5 = float(np.mean(ranks <= 5))
        recall_at_10 = float(np.mean(ranks <= 10))
        mrr = float(np.mean(1.0 / ranks))
        median_rank = float(np.median(ranks))

        results = {
            "retrieval_recall@1": recall_at_1,
            "retrieval_recall@5": recall_at_5,
            "retrieval_recall@10": recall_at_10,
            "retrieval_mrr": mrr,
            "retrieval_median_rank": median_rank,
            "retrieval_n_samples": n,
        }

        logger.info(
            f"CrossModalRetrieval ({self.metric}): "
            f"R@1={recall_at_1:.4f}, R@5={recall_at_5:.4f}, R@10={recall_at_10:.4f}, "
            f"MRR={mrr:.4f}, median_rank={median_rank:.0f}/{n}"
        )

        return results


class AlignmentBenchmark:
    """
    Run all post-hoc alignment methods and collect comprehensive results.

    This is the main entry point for the independent SAE alignment evaluation.
    """

    def __init__(
        self,
        cca_components: int = 50,
        procrustes_components: Optional[int] = None,
        ridge_alpha: float = 1.0,
        retrieval_metric: str = "cosine",
    ):
        self.cca = CCAAlignment(n_components=cca_components)
        self.procrustes = ProcrustesAlignment(n_components=procrustes_components)
        self.linear_probe = LinearProbeAlignment(alpha=ridge_alpha)
        self.retrieval = CrossModalRetrieval(metric=retrieval_metric)

    def run(
        self,
        protein_features: np.ndarray,
        dna_features: np.ndarray,
        gene_names: Optional[list[str]] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run all alignment methods and return a comprehensive results DataFrame.

        Splits data into train/test. Fits alignment methods on train split,
        evaluates on test split.

        Args:
            protein_features: (n_genes, dim_protein_features) protein SAE features
            dna_features: (n_genes, dim_dna_features) DNA SAE features
            gene_names: Optional gene name labels
            train_ratio: Fraction of data for fitting alignment methods
            seed: Random seed for reproducibility

        Returns:
            DataFrame with columns: method, metric, value
        """
        n = protein_features.shape[0]
        assert n == dna_features.shape[0], (
            f"Mismatch: protein has {n} samples, DNA has {dna_features.shape[0]}"
        )

        logger.info(
            f"AlignmentBenchmark: {n} genes, "
            f"protein_features={protein_features.shape}, "
            f"dna_features={dna_features.shape}"
        )

        # Train/test split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * train_ratio)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        prot_train, prot_test = protein_features[train_idx], protein_features[test_idx]
        dna_train, dna_test = dna_features[train_idx], dna_features[test_idx]

        if gene_names is not None:
            gene_names_test = [gene_names[i] for i in test_idx]
        else:
            gene_names_test = None

        logger.info(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

        all_results = []

        # 1. CCA
        logger.info("Running CCA alignment...")
        try:
            self.cca.fit(prot_train, dna_train)
            cca_scores = self.cca.score(prot_test, dna_test)
            for metric, value in cca_scores.items():
                all_results.append({"method": "CCA", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"CCA failed: {e}")
            all_results.append({"method": "CCA", "metric": "error", "value": str(e)})

        # 2. Procrustes
        logger.info("Running Procrustes alignment...")
        try:
            self.procrustes.fit(prot_train, dna_train)
            procrustes_scores = self.procrustes.score(prot_test, dna_test)
            for metric, value in procrustes_scores.items():
                all_results.append({"method": "Procrustes", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Procrustes failed: {e}")
            all_results.append({"method": "Procrustes", "metric": "error", "value": str(e)})

        # 3. Linear Probe (protein -> DNA)
        logger.info("Running Linear Probe alignment (protein -> DNA)...")
        try:
            self.linear_probe.fit(prot_train, dna_train)
            probe_scores = self.linear_probe.score(prot_test, dna_test)
            for metric, value in probe_scores.items():
                all_results.append({"method": "LinearProbe_P2D", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Linear probe (P->D) failed: {e}")
            all_results.append({"method": "LinearProbe_P2D", "metric": "error", "value": str(e)})

        # 4. Linear Probe (DNA -> protein) — reverse direction
        logger.info("Running Linear Probe alignment (DNA -> protein)...")
        try:
            reverse_probe = LinearProbeAlignment(alpha=self.linear_probe.alpha)
            reverse_probe.fit(dna_train, prot_train)
            reverse_scores = reverse_probe.score(dna_test, prot_test)
            for metric, value in reverse_scores.items():
                all_results.append({"method": "LinearProbe_D2P", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Linear probe (D->P) failed: {e}")
            all_results.append({"method": "LinearProbe_D2P", "metric": "error", "value": str(e)})

        # 5. Cross-modal retrieval (raw features)
        logger.info("Running cross-modal retrieval (raw features)...")
        try:
            retrieval_scores = self.retrieval.evaluate(prot_test, dna_test, gene_names_test)
            for metric, value in retrieval_scores.items():
                all_results.append({"method": "Retrieval_Raw", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Retrieval (raw) failed: {e}")
            all_results.append({"method": "Retrieval_Raw", "metric": "error", "value": str(e)})

        # 6. Cross-modal retrieval (after CCA projection)
        logger.info("Running cross-modal retrieval (CCA-projected)...")
        try:
            proj_prot, proj_dna = self.cca.transform(prot_test, dna_test)
            cca_retrieval = CrossModalRetrieval(metric=self.retrieval.metric)
            cca_ret_scores = cca_retrieval.evaluate(proj_prot, proj_dna, gene_names_test)
            for metric, value in cca_ret_scores.items():
                all_results.append({"method": "Retrieval_CCA", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Retrieval (CCA) failed: {e}")
            all_results.append({"method": "Retrieval_CCA", "metric": "error", "value": str(e)})

        # 7. Cross-modal retrieval (after Procrustes alignment)
        logger.info("Running cross-modal retrieval (Procrustes-aligned)...")
        try:
            pa = self.procrustes.transform_a(prot_test)
            pb = self.procrustes.transform(dna_test)
            proc_retrieval = CrossModalRetrieval(metric=self.retrieval.metric)
            proc_ret_scores = proc_retrieval.evaluate(pa, pb, gene_names_test)
            for metric, value in proc_ret_scores.items():
                all_results.append({"method": "Retrieval_Procrustes", "metric": metric, "value": value})
        except Exception as e:
            logger.error(f"Retrieval (Procrustes) failed: {e}")
            all_results.append({"method": "Retrieval_Procrustes", "metric": "error", "value": str(e)})

        df = pd.DataFrame(all_results)
        logger.info(f"AlignmentBenchmark complete: {len(df)} results")
        return df


def extract_sae_features(
    model,
    h5_path: str,
    modality: str,
    device: str = "cuda",
    batch_size: int = 512,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract SAE feature activations for all genes in an activation file.

    Works with both StandardSAE (single-modality) and CrossBioSAE (joint).

    Args:
        model: StandardSAE or CrossBioSAE model (already on device, eval mode)
        h5_path: Path to HDF5 activation file
        modality: "protein" or "dna"
        device: Device string
        batch_size: Batch size for inference

    Returns:
        (features, gene_names): features is (n_genes, n_features), gene_names is list of str
    """
    import h5py
    import torch

    model.eval()

    with h5py.File(h5_path, "r") as f:
        activations = f["activations"][:]
        gene_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["gene_names"][:]
        ]

    all_features = []
    with torch.no_grad():
        for start in range(0, len(activations), batch_size):
            batch = torch.tensor(
                activations[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            # Handle both StandardSAE and CrossBioSAE
            from .model import StandardSAE, CrossBioSAE
            if isinstance(model, StandardSAE):
                features = model.encode(batch)
            elif isinstance(model, CrossBioSAE):
                if modality == "protein":
                    features = model.encode_protein(batch)
                elif modality == "dna":
                    features = model.encode_dna(batch)
                else:
                    raise ValueError(f"Unknown modality: {modality}")
            else:
                raise TypeError(f"Unsupported model type: {type(model)}")

            all_features.append(features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    logger.info(f"Extracted features: {features.shape} for {len(gene_names)} genes ({modality})")
    return features, gene_names


def align_gene_order(
    features_a: np.ndarray,
    names_a: list[str],
    features_b: np.ndarray,
    names_b: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Align two feature matrices by gene name, keeping only shared genes.

    Args:
        features_a: (n_a, dim_a) feature matrix
        names_a: gene names for features_a
        features_b: (n_b, dim_b) feature matrix
        names_b: gene names for features_b

    Returns:
        (aligned_a, aligned_b, common_names): aligned matrices and shared gene names
    """
    name_to_idx_a = {name: i for i, name in enumerate(names_a)}
    name_to_idx_b = {name: i for i, name in enumerate(names_b)}

    common_names = sorted(set(names_a) & set(names_b))
    if len(common_names) == 0:
        raise ValueError("No common gene names between the two feature sets")

    idx_a = [name_to_idx_a[n] for n in common_names]
    idx_b = [name_to_idx_b[n] for n in common_names]

    logger.info(
        f"Aligned genes: {len(common_names)} common "
        f"(A had {len(names_a)}, B had {len(names_b)})"
    )

    return features_a[idx_a], features_b[idx_b], common_names
