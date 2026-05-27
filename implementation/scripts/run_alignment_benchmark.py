#!/usr/bin/env python3
"""
Post-hoc alignment benchmark: compare independently trained SAEs vs joint CrossBioSAE.

This script:
1. Loads independently trained protein SAE and DNA SAE checkpoints
2. Extracts features from each
3. Runs all post-hoc alignment methods (CCA, Procrustes, LinearProbe, Retrieval)
4. Optionally loads joint CrossBioSAE checkpoint for comparison
5. Outputs a comparison table

Usage:
    # Compare independent SAEs only
    python scripts/run_alignment_benchmark.py \
        --config configs/full.yaml \
        --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
        --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt \
        --device cuda

    # Compare independent SAEs vs joint CrossBioSAE
    python scripts/run_alignment_benchmark.py \
        --config configs/full.yaml \
        --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
        --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt \
        --joint_ckpt checkpoints/full/checkpoint_best.pt \
        --device cuda

    # Also compare raw activations (no SAE, baseline)
    python scripts/run_alignment_benchmark.py \
        --config configs/full.yaml \
        --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
        --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt \
        --joint_ckpt checkpoints/full/checkpoint_best.pt \
        --include_raw \
        --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import StandardSAE, StandardSAEConfig, CrossBioSAE, CrossBioSAEConfig
from src.alignment import (
    AlignmentBenchmark,
    extract_sae_features,
    align_gene_order,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_standard_sae(ckpt_path: str, device: str) -> StandardSAE:
    """Load a trained StandardSAE from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    config_dict = checkpoint["config"]
    config = StandardSAEConfig(**config_dict)
    model = StandardSAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    modality = checkpoint.get("modality", "unknown")
    logger.info(f"Loaded StandardSAE [{modality}] from {ckpt_path} (dim={config.dim_input}, features={config.n_features})")
    return model


def load_joint_sae(ckpt_path: str, config: CrossBioSAEConfig, device: str) -> CrossBioSAE:
    """Load a trained CrossBioSAE from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = CrossBioSAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded CrossBioSAE from {ckpt_path}")
    return model


def run_benchmark_on_features(
    label: str,
    protein_features: np.ndarray,
    dna_features: np.ndarray,
    gene_names: list[str],
    benchmark: AlignmentBenchmark,
    seed: int = 42,
) -> pd.DataFrame:
    """Run alignment benchmark and tag results with a source label."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running alignment benchmark: {label}")
    logger.info(f"{'='*60}")

    df = benchmark.run(
        protein_features=protein_features,
        dna_features=dna_features,
        gene_names=gene_names,
        seed=seed,
    )
    df["source"] = label
    return df


def main():
    parser = argparse.ArgumentParser(description="Post-hoc alignment benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--protein_sae_ckpt", type=str, required=True,
                        help="Path to independently trained protein SAE checkpoint")
    parser.add_argument("--dna_sae_ckpt", type=str, required=True,
                        help="Path to independently trained DNA SAE checkpoint")
    parser.add_argument("--joint_ckpt", type=str, default=None,
                        help="Path to jointly trained CrossBioSAE checkpoint (optional)")
    parser.add_argument("--include_raw", action="store_true",
                        help="Also benchmark raw activations (no SAE) as a baseline")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/<split>/alignment_benchmark.csv)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cca_components", type=int, default=50)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    paths = config["paths"]
    model_cfg = config["model"]

    protein_h5 = paths["protein_activations"]
    dna_h5 = paths["dna_activations"]

    # Default output path
    if args.output is None:
        output_dir = Path(paths["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "alignment_benchmark.csv"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Alignment benchmark configuration
    benchmark = AlignmentBenchmark(
        cca_components=args.cca_components,
        ridge_alpha=args.ridge_alpha,
    )

    all_results = []

    # =====================================================================
    # 1. Independent SAE features
    # =====================================================================
    logger.info("Loading independent SAE models...")
    protein_sae = load_standard_sae(args.protein_sae_ckpt, device)
    dna_sae = load_standard_sae(args.dna_sae_ckpt, device)

    logger.info("Extracting independent SAE features...")
    prot_features_indep, prot_names = extract_sae_features(
        protein_sae, protein_h5, "protein", device
    )
    dna_features_indep, dna_names = extract_sae_features(
        dna_sae, dna_h5, "dna", device
    )

    # Align gene order
    prot_features_indep, dna_features_indep, common_names = align_gene_order(
        prot_features_indep, prot_names, dna_features_indep, dna_names
    )

    df_indep = run_benchmark_on_features(
        "Independent SAEs",
        prot_features_indep, dna_features_indep, common_names,
        benchmark, args.seed,
    )
    all_results.append(df_indep)

    # =====================================================================
    # 2. Joint CrossBioSAE features (optional)
    # =====================================================================
    if args.joint_ckpt is not None:
        logger.info("Loading joint CrossBioSAE model...")
        joint_config = CrossBioSAEConfig(
            dim_dna=model_cfg["dim_dna"],
            dim_protein=model_cfg["dim_protein"],
            dim_shared=model_cfg["dim_shared"],
            expansion_factor=model_cfg["expansion_factor"],
            sparsity_type=model_cfg.get("sparsity_type", "topk"),
            topk_k=model_cfg.get("topk_k", 64),
            normalize_inputs=model_cfg.get("normalize_inputs", True),
            tied_decoder=model_cfg.get("tied_decoder", False),
            bias=model_cfg.get("bias", True),
        )
        joint_model = load_joint_sae(args.joint_ckpt, joint_config, device)

        logger.info("Extracting joint SAE features...")
        prot_features_joint, prot_names_joint = extract_sae_features(
            joint_model, protein_h5, "protein", device
        )
        dna_features_joint, dna_names_joint = extract_sae_features(
            joint_model, dna_h5, "dna", device
        )

        prot_features_joint, dna_features_joint, common_names_joint = align_gene_order(
            prot_features_joint, prot_names_joint, dna_features_joint, dna_names_joint
        )

        df_joint = run_benchmark_on_features(
            "Joint CrossBioSAE",
            prot_features_joint, dna_features_joint, common_names_joint,
            benchmark, args.seed,
        )
        all_results.append(df_joint)

    # =====================================================================
    # 3. Raw activations baseline (optional)
    # =====================================================================
    if args.include_raw:
        import h5py

        logger.info("Loading raw activations for baseline...")
        with h5py.File(protein_h5, "r") as f:
            raw_prot = f["activations"][:]
            raw_prot_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f["gene_names"][:]
            ]
        with h5py.File(dna_h5, "r") as f:
            raw_dna = f["activations"][:]
            raw_dna_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f["gene_names"][:]
            ]

        raw_prot, raw_dna, common_raw = align_gene_order(
            raw_prot, raw_prot_names, raw_dna, raw_dna_names
        )

        df_raw = run_benchmark_on_features(
            "Raw Activations (no SAE)",
            raw_prot, raw_dna, common_raw,
            benchmark, args.seed,
        )
        all_results.append(df_raw)

    # =====================================================================
    # Combine and display results
    # =====================================================================
    results_df = pd.concat(all_results, ignore_index=True)

    # Save full results
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nFull results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ALIGNMENT BENCHMARK RESULTS")
    print("=" * 80)

    # Pivot for easy comparison
    numeric_results = results_df[results_df["metric"] != "error"].copy()
    numeric_results["value"] = pd.to_numeric(numeric_results["value"], errors="coerce")

    pivot = numeric_results.pivot_table(
        index=["method", "metric"],
        columns="source",
        values="value",
        aggfunc="first",
    )

    # Print key metrics
    key_metrics = [
        "retrieval_recall@1",
        "retrieval_recall@5",
        "retrieval_mrr",
        "cca_mean_correlation",
        "procrustes_mean_cosine_sim",
        "linear_probe_r2",
    ]

    print("\nKey Metrics Comparison:")
    print("-" * 80)
    for method in numeric_results["method"].unique():
        print(f"\n  {method}:")
        method_data = numeric_results[numeric_results["method"] == method]
        for _, row in method_data.iterrows():
            if row["metric"] in key_metrics:
                print(f"    {row['metric']:40s} [{row['source']:25s}] = {row['value']:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
