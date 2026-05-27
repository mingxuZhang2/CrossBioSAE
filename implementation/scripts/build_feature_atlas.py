#!/usr/bin/env python3
"""
Build interpretable feature atlas: classify each SAE feature as
shared / protein-dominant / DNA-dominant / discordant.

For top features in each category, produce feature cards with:
- activation frequency per modality
- top activating genes
- GO/Pfam enrichment
- sequence property correlations (length, GC)
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
from scipy.stats import hypergeom, pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import StandardSAE, StandardSAEConfig, CrossBioSAE, CrossBioSAEConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["activations"][:].astype(np.float32), \
               [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]


def sae_encode(model, acts, device, batch_size=512):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            b = torch.tensor(acts[i:i+batch_size]).to(device)
            feats.append(model.encode(b).cpu().numpy())
    return np.concatenate(feats)


def classify_features(prot_feats, dna_feats, act_threshold=0.01, corr_threshold=0.1):
    """Classify each feature into shared/protein-dominant/DNA-dominant/discordant/dead."""
    n_genes, n_prot_feat = prot_feats.shape
    _, n_dna_feat = dna_feats.shape

    # For joint model: same feature index, compare per-gene co-activation
    if n_prot_feat == n_dna_feat:
        freq_p = (prot_feats > 0).mean(axis=0)
        freq_d = (dna_feats > 0).mean(axis=0)

        categories = []
        correlations = []
        for j in range(n_prot_feat):
            p_col = prot_feats[:, j]
            d_col = dna_feats[:, j]
            p_active = freq_p[j] > act_threshold
            d_active = freq_d[j] > act_threshold

            corr = 0.0
            if p_col.std() > 1e-8 and d_col.std() > 1e-8:
                corr = pearsonr(p_col, d_col)[0]
            correlations.append(corr)

            if not p_active and not d_active:
                categories.append("dead")
            elif p_active and d_active and corr > corr_threshold:
                categories.append("shared")
            elif p_active and d_active and corr < -corr_threshold:
                categories.append("discordant")
            elif p_active and not d_active:
                categories.append("protein_dominant")
            elif d_active and not p_active:
                categories.append("dna_dominant")
            else:
                categories.append("weak")

        return categories, correlations

    # For independent SAEs: features don't correspond by index
    # Report per-modality stats only
    return None, None


def build_feature_cards(feats, gene_names, go_annotations, gene_pairs, modality, top_k=20):
    """Build feature cards for top-k most active features."""
    freq = (feats > 0).mean(axis=0)
    top_indices = np.argsort(-freq)[:top_k * 5]  # candidates

    gene_map = {p["gene_name"]: p for p in gene_pairs}
    cards = []

    for feat_idx in top_indices[:top_k]:
        feat_idx = int(feat_idx)
        col = feats[:, feat_idx]
        active_mask = col > 0
        n_active = active_mask.sum()

        if n_active < 5:
            continue

        # Top activating genes
        top_gene_idx = np.argsort(-col)[:10]
        top_genes = [gene_names[i] for i in top_gene_idx]

        # GO enrichment (top 3 terms)
        active_genes = set(gene_names[i] for i in range(len(gene_names)) if active_mask[i])
        go_enrichments = []
        all_go = Counter()
        for g in active_genes:
            if g in go_annotations:
                for t in go_annotations[g]:
                    all_go[t] += 1

        n_total = len(gene_names)
        total_annotated = sum(1 for g in gene_names if g in go_annotations)
        for term, count in all_go.most_common(20):
            bg_count = sum(1 for g in gene_names if g in go_annotations and term in go_annotations[g])
            if bg_count < 5:
                continue
            p = hypergeom.sf(count - 1, n_total, bg_count, int(n_active))
            if p < 0.01:
                go_enrichments.append({"term": term, "overlap": count, "background": bg_count, "p_value": p})

        go_enrichments.sort(key=lambda x: x["p_value"])

        # Sequence property correlations
        lengths = [gene_map.get(g, {}).get("protein_length", 0) for g in gene_names]
        lengths = np.array(lengths, dtype=np.float32)
        length_corr = 0.0
        if lengths.std() > 0 and col.std() > 0:
            length_corr = pearsonr(col, lengths)[0]

        cards.append({
            "feature_index": feat_idx,
            "modality": modality,
            "activation_frequency": float(freq[feat_idx]),
            "n_active_genes": int(n_active),
            "top_genes": top_genes,
            "top_go_enrichments": go_enrichments[:3],
            "protein_length_correlation": float(length_corr),
        })

    return cards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protein_sae_ckpt", required=True)
    parser.add_argument("--dna_sae_ckpt", required=True)
    parser.add_argument("--joint_ckpt", default=None)
    parser.add_argument("--go_annotations", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    paths = config["paths"]
    prot_raw, prot_names = load_h5(paths["protein_activations"])
    dna_raw, dna_names = load_h5(paths["dna_activations"])

    # Align
    common = sorted(set(prot_names) & set(dna_names))
    pi = {n: i for i, n in enumerate(prot_names)}
    di = {n: i for i, n in enumerate(dna_names)}
    prot_raw = prot_raw[[pi[n] for n in common]]
    dna_raw = dna_raw[[di[n] for n in common]]
    gene_names = common

    # Load SAEs
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

    prot_feats = sae_encode(prot_sae, prot_raw, device)
    dna_feats = sae_encode(dna_sae, dna_raw, device)

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

    output_dir = Path(paths.get("output_dir", "results/full")) / "feature_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build feature cards per modality
    logger.info("Building protein feature cards...")
    prot_cards = build_feature_cards(prot_feats, gene_names, go_annotations, gene_pairs, "protein", top_k=20)

    logger.info("Building DNA feature cards...")
    dna_cards = build_feature_cards(dna_feats, gene_names, go_annotations, gene_pairs, "dna", top_k=20)

    all_cards = prot_cards + dna_cards

    # Joint model feature classification (if checkpoint provided)
    if args.joint_ckpt and Path(args.joint_ckpt).exists():
        logger.info("Loading joint CrossBioSAE for feature classification...")
        model_cfg = config["model"]
        train_cfg = config["training"]
        joint_cfg = CrossBioSAEConfig(
            dim_dna=model_cfg["dim_dna"], dim_protein=model_cfg["dim_protein"],
            dim_shared=model_cfg["dim_shared"], expansion_factor=model_cfg["expansion_factor"],
            topk_k=model_cfg.get("topk_k", 64),
            normalize_inputs=model_cfg.get("normalize_inputs", True),
            crossmodal_weight=train_cfg.get("crossmodal_weight", 1.0),
        )
        joint_model = CrossBioSAE(joint_cfg).to(device)
        ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=True)
        joint_model.load_state_dict(ckpt["model_state_dict"])
        joint_model.eval()

        with torch.no_grad():
            joint_prot = []
            joint_dna = []
            for i in range(0, len(prot_raw), 512):
                pb = torch.tensor(prot_raw[i:i+512]).to(device)
                db = torch.tensor(dna_raw[i:i+512]).to(device)
                joint_prot.append(joint_model.encode_protein(pb).cpu().numpy())
                joint_dna.append(joint_model.encode_dna(db).cpu().numpy())
            joint_prot = np.concatenate(joint_prot)
            joint_dna = np.concatenate(joint_dna)

        categories, correlations = classify_features(joint_prot, joint_dna)

        if categories:
            cat_counts = Counter(categories)
            logger.info(f"\nJoint CrossBioSAE feature classification:")
            for cat in ["shared", "protein_dominant", "dna_dominant", "discordant", "weak", "dead"]:
                logger.info(f"  {cat:20s}: {cat_counts.get(cat, 0)}")

            # Save classification
            class_df = pd.DataFrame({
                "feature_idx": range(len(categories)),
                "category": categories,
                "cross_modal_correlation": correlations,
            })
            class_df.to_csv(output_dir / "joint_feature_classification.csv", index=False)

    # Save feature cards
    with open(output_dir / "feature_cards.json", "w") as f:
        json.dump(all_cards, f, indent=2, default=str)

    # Summary
    logger.info(f"\nFeature atlas saved to {output_dir}")
    logger.info(f"  {len(prot_cards)} protein feature cards")
    logger.info(f"  {len(dna_cards)} DNA feature cards")

    # Print top 5 from each modality
    for modality, cards in [("Protein", prot_cards[:5]), ("DNA", dna_cards[:5])]:
        logger.info(f"\nTop {modality} features:")
        for c in cards:
            go_str = ", ".join(e["term"] for e in c.get("top_go_enrichments", [])[:2])
            logger.info(f"  Feature {c['feature_index']}: freq={c['activation_frequency']:.3f}, "
                       f"n_genes={c['n_active_genes']}, top_genes={c['top_genes'][:3]}, GO=[{go_str}]")


if __name__ == "__main__":
    main()
