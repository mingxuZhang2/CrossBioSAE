#!/usr/bin/env python3
"""
Run all 4 downstream applications for the protein-DNA shared gene space.

App 1: Cross-modal functional transfer benchmark
App 2: Consensus high-confidence annotation
App 3: Gene model / annotation QC
App 4: Discordance atlas with confound controls

Usage:
    python run_applications.py --config configs/independent.yaml \
        --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
        --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt \
        --apps 1 2 3 4

Or run a single app:
    python run_applications.py --config configs/independent.yaml --apps 2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import StandardSAE, StandardSAEConfig
from src.applications import (
    run_transfer_benchmark,
    run_consensus_annotation,
    run_annotation_qc,
    run_discordance_atlas,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("applications_run.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_h5(path):
    with h5py.File(path, "r") as f:
        acts = f["activations"][:].astype(np.float32)
        names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
    return acts, names


def align_genes(*name_act_pairs):
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


def main():
    parser = argparse.ArgumentParser(description="Run CrossBioSAE downstream applications")
    parser.add_argument("--config", required=True, help="Config YAML (e.g. configs/independent.yaml)")
    parser.add_argument("--protein_sae_ckpt", default=None, help="Protein SAE checkpoint")
    parser.add_argument("--dna_sae_ckpt", default=None, help="DNA SAE checkpoint")
    parser.add_argument("--go_annotations", default=None, help="GO annotations JSON")
    parser.add_argument("--apps", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which applications to run (1-4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    paths = config["paths"]
    output_dir = Path(args.output_dir or paths.get("output_dir", "results/full")) / "applications"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load raw activations
    logger.info("Loading raw activations...")
    prot_raw, prot_names = load_h5(paths["protein_activations"])
    dna_raw, dna_names = load_h5(paths["dna_activations"])
    gene_names, prot_raw, dna_raw = align_genes((prot_names, prot_raw), (dna_names, dna_raw))
    logger.info(f"Aligned: {len(gene_names)} genes, prot={prot_raw.shape}, dna={dna_raw.shape}")

    # Load SAE models and encode
    prot_sae_feats = None
    dna_sae_feats = None

    prot_ckpt = args.protein_sae_ckpt or paths.get("checkpoint_dir", "checkpoints/full") + "/protein_sae/checkpoint_best.pt"
    dna_ckpt = args.dna_sae_ckpt or paths.get("checkpoint_dir", "checkpoints/full") + "/dna_sae/checkpoint_best.pt"

    if Path(prot_ckpt).exists() and Path(dna_ckpt).exists():
        logger.info("Loading SAE models...")
        prot_cfg = StandardSAEConfig(
            dim_input=config["protein_model"]["hidden_dim"],
            expansion_factor=config.get("independent_sae", {}).get("expansion_factor",
                            config["model"]["expansion_factor"]),
            topk_k=config.get("independent_sae", {}).get("topk_k",
                   config["model"].get("topk_k", 64)),
        )
        dna_cfg = StandardSAEConfig(
            dim_input=config["dna_model"]["hidden_dim"],
            expansion_factor=config.get("independent_sae", {}).get("expansion_factor",
                            config["model"]["expansion_factor"]),
            topk_k=config.get("independent_sae", {}).get("topk_k",
                   config["model"].get("topk_k", 64)),
        )

        prot_sae = StandardSAE(prot_cfg).to(device)
        prot_sae.load_state_dict(
            torch.load(prot_ckpt, map_location=device, weights_only=True)["model_state_dict"]
        )
        dna_sae = StandardSAE(dna_cfg).to(device)
        dna_sae.load_state_dict(
            torch.load(dna_ckpt, map_location=device, weights_only=True)["model_state_dict"]
        )

        logger.info("Encoding SAE features...")
        prot_sae_feats = sae_encode(prot_sae, prot_raw, device)
        dna_sae_feats = sae_encode(dna_sae, dna_raw, device)
        logger.info(f"SAE features: prot={prot_sae_feats.shape}, dna={dna_sae_feats.shape}")

        # Align SAE features to same gene order (already aligned via raw acts)
    else:
        logger.warning(f"SAE checkpoints not found ({prot_ckpt}, {dna_ckpt}). "
                       "Running without SAE features.")

    # Load GO annotations
    go_annotations = {}
    go_path = args.go_annotations or paths.get("go_annotations", "")
    if not go_path:
        go_path = f"{config['data']['data_dir']}/go_annotations.json"
    if Path(go_path).exists():
        with open(go_path) as f:
            go_annotations = json.load(f)
        logger.info(f"Loaded GO annotations for {len(go_annotations)} genes")
    else:
        logger.warning(f"GO annotations not found at {go_path}")

    # Load gene pairs (for metadata)
    gene_pairs = []
    gp_path = f"{config['data']['data_dir']}/gene_pairs_{config['data']['split']}.json"
    if Path(gp_path).exists():
        with open(gp_path) as f:
            gene_pairs = json.load(f)
        logger.info(f"Loaded {len(gene_pairs)} gene pairs")

    # Run applications
    app_results = {}

    if 1 in args.apps:
        if go_annotations:
            logger.info("\n" + "=" * 70)
            t1 = time.time()
            app_results[1] = run_transfer_benchmark(
                prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                gene_names, go_annotations, output_dir, seed=42,
            )
            logger.info(f"App 1 completed in {time.time() - t1:.1f}s")
        else:
            logger.warning("Skipping App 1: no GO annotations")

    if 2 in args.apps:
        if go_annotations:
            logger.info("\n" + "=" * 70)
            t1 = time.time()
            app_results[2] = run_consensus_annotation(
                prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
                gene_names, go_annotations, output_dir, seed=42,
            )
            logger.info(f"App 2 completed in {time.time() - t1:.1f}s")
        else:
            logger.warning("Skipping App 2: no GO annotations")

    if 3 in args.apps:
        logger.info("\n" + "=" * 70)
        t1 = time.time()
        app_results[3] = run_annotation_qc(
            prot_raw, dna_raw, gene_names, gene_pairs, output_dir, seed=42,
        )
        logger.info(f"App 3 completed in {time.time() - t1:.1f}s")

    if 4 in args.apps:
        logger.info("\n" + "=" * 70)
        t1 = time.time()
        app_results[4] = run_discordance_atlas(
            prot_raw, dna_raw, prot_sae_feats, dna_sae_feats,
            gene_names, gene_pairs, go_annotations, output_dir, seed=42,
        )
        logger.info(f"App 4 completed in {time.time() - t1:.1f}s")

    logger.info("\n" + "=" * 70)
    logger.info(f"ALL APPLICATIONS COMPLETE ({time.time() - t0:.1f}s total)")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
