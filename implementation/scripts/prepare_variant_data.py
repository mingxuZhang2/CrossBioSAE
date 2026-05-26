#!/usr/bin/env python3
"""
Prepare variant data for Task 1 (variant effect prediction).
Matches ClinVar variants to genes in our dataset, generates WT/MT sequences,
and computes feature disruption scores using a trained CrossBioSAE.

This script runs ON HPC with GPU access.

Usage:
    python scripts/prepare_variant_data.py \
        --config configs/full.yaml \
        --checkpoint checkpoints/full/checkpoint_best.pt \
        --clinvar data/full/clinvar_variants.csv \
        --device cuda
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CrossBioSAE, CrossBioSAEConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CODON_TABLE = {
    'A': 'GCT', 'R': 'CGT', 'N': 'AAT', 'D': 'GAT', 'C': 'TGT',
    'E': 'GAA', 'Q': 'CAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCT',
    'S': 'TCT', 'T': 'ACT', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTT',
    '*': 'TAA',
}


def mutate_protein(wt_seq: str, position: int, mt_aa: str) -> str:
    """Apply a missense mutation to a protein sequence."""
    if position < 0 or position >= len(wt_seq):
        return None
    return wt_seq[:position] + mt_aa + wt_seq[position + 1:]


def protein_to_cds(protein_seq: str) -> str:
    """Simple back-translation using most common codons."""
    return "".join(CODON_TABLE.get(aa, "NNN") for aa in protein_seq)


def parse_variant_name(name: str):
    """Parse ClinVar variant name to extract protein change.
    e.g., 'NM_000059.4(BRCA2):c.7397T>C (p.Val2466Ala)' -> (V, 2465, A)
    """
    import re
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(name))
    if not match:
        return None

    aa_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    }

    wt_aa = aa_map.get(match.group(1))
    pos = int(match.group(2)) - 1  # 0-indexed
    mt_aa = aa_map.get(match.group(3))

    if wt_aa and mt_aa:
        return wt_aa, pos, mt_aa
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clinvar", default="data/full/clinvar_variants.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_variants", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load gene pairs
    data_dir = config["data"]["data_dir"]
    split = config["data"]["split"]
    gene_pairs_path = f"{data_dir}/gene_pairs_{split}.json"
    with open(gene_pairs_path) as f:
        gene_pairs = json.load(f)
    gene_map = {p["gene_name"]: p for p in gene_pairs}
    logger.info(f"Loaded {len(gene_map)} genes")

    # Load ClinVar
    clinvar = pd.read_csv(args.clinvar)
    logger.info(f"Loaded {len(clinvar)} ClinVar variants")

    # Match variants to our genes
    matched = []
    for _, row in clinvar.iterrows():
        gene = row["GeneSymbol"]
        if gene not in gene_map:
            continue

        parsed = parse_variant_name(row.get("Name", ""))
        if parsed is None:
            continue

        wt_aa, pos, mt_aa = parsed
        protein_seq = gene_map[gene]["protein_seq"]

        if pos >= len(protein_seq):
            continue
        if protein_seq[pos] != wt_aa:
            continue

        mt_protein = mutate_protein(protein_seq, pos, mt_aa)
        wt_cds = gene_map[gene]["cds_seq"]
        mt_cds = protein_to_cds(mt_protein)

        matched.append({
            "gene_name": gene,
            "variant_name": row.get("Name", ""),
            "pathogenicity": row["pathogenicity"],
            "wt_protein": protein_seq,
            "mt_protein": mt_protein,
            "wt_cds": wt_cds,
            "mt_cds": mt_cds,
            "position": pos,
            "wt_aa": wt_aa,
            "mt_aa": mt_aa,
        })

        if len(matched) >= args.max_variants:
            break

    logger.info(f"Matched {len(matched)} variants to genes in dataset")

    n_path = sum(1 for v in matched if v["pathogenicity"] == 1)
    n_benign = sum(1 for v in matched if v["pathogenicity"] == 0)
    logger.info(f"Pathogenic: {n_path}, Benign: {n_benign}")

    if len(matched) < 50:
        logger.error("Too few matched variants. Cannot proceed.")
        return

    # Load model
    model_cfg = config["model"]
    train_cfg = config["training"]
    sae_config = CrossBioSAEConfig(
        dim_dna=model_cfg["dim_dna"],
        dim_protein=model_cfg["dim_protein"],
        dim_shared=model_cfg["dim_shared"],
        expansion_factor=model_cfg["expansion_factor"],
        sparsity_type=model_cfg.get("sparsity_type", "topk"),
        topk_k=model_cfg.get("topk_k", 64),
        normalize_inputs=model_cfg.get("normalize_inputs", True),
        crossmodal_weight=train_cfg.get("crossmodal_weight", 1.0),
        crossmodal_type=model_cfg.get("crossmodal_type", "cosine"),
    )

    model = CrossBioSAE(sae_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded")

    # Load ESM-2 for protein activations
    logger.info("Loading ESM-2 for variant activation extraction...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    prot_layer = config["protein_model"]["layer"]

    # Compute variant scores
    logger.info("Computing variant effect scores...")
    scores = []
    labels = []

    with torch.no_grad():
        for i, v in enumerate(tqdm(matched, desc="Scoring variants")):
            # Extract WT protein activation
            _, _, wt_tokens = batch_converter([("wt", v["wt_protein"][:1024])])
            wt_out = esm_model(wt_tokens.to(device), repr_layers=[prot_layer])
            wt_repr = wt_out["representations"][prot_layer]
            wt_act = wt_repr[0, 1:len(v["wt_protein"][:1024]) + 1].mean(dim=0)

            # Extract MT protein activation
            _, _, mt_tokens = batch_converter([("mt", v["mt_protein"][:1024])])
            mt_out = esm_model(mt_tokens.to(device), repr_layers=[prot_layer])
            mt_repr = mt_out["representations"][prot_layer]
            mt_act = mt_repr[0, 1:len(v["mt_protein"][:1024]) + 1].mean(dim=0)

            # Encode through CrossBioSAE
            wt_features = model.encode_protein(wt_act.unsqueeze(0))
            mt_features = model.encode_protein(mt_act.unsqueeze(0))

            # Feature disruption score
            disruption = (wt_features - mt_features).abs().sum().item()
            scores.append(disruption)
            labels.append(v["pathogenicity"])

    scores = np.array(scores)
    labels = np.array(labels)

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    logger.info(f"=== VARIANT PREDICTION RESULTS ===")
    logger.info(f"Variants: {len(labels)} ({labels.sum():.0f} pathogenic, {(1 - labels).sum():.0f} benign)")
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"Mean score pathogenic: {scores[labels == 1].mean():.4f}")
    logger.info(f"Mean score benign: {scores[labels == 0].mean():.4f}")

    # Save results
    output_dir = Path(config["paths"]["output_dir"]) / "task1_variant"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        "gene_name": [v["gene_name"] for v in matched],
        "variant": [v["variant_name"] for v in matched],
        "position": [v["position"] for v in matched],
        "wt_aa": [v["wt_aa"] for v in matched],
        "mt_aa": [v["mt_aa"] for v in matched],
        "pathogenicity": labels,
        "crossbiosae_score": scores,
    })
    results_df.to_csv(output_dir / "variant_predictions.csv", index=False)

    summary = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_variants": int(len(labels)),
        "n_pathogenic": int(labels.sum()),
        "n_benign": int((1 - labels).sum()),
        "mean_score_pathogenic": float(scores[labels == 1].mean()),
        "mean_score_benign": float(scores[labels == 0].mean()),
    }
    with open(output_dir / "variant_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
