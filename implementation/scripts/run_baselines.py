#!/usr/bin/env python3
"""
Run single-model baselines for comparison with CrossBioSAE.

Baselines:
1. ESM-2 log-likelihood ratio (per-residue, standard method)
2. ESM-2 mean-pooled activation L2 distance (same as CrossBioSAE but without SAE)
3. CrossBioSAE feature disruption (our method, for direct comparison)

Usage:
    python scripts/run_baselines.py \
        --config configs/full.yaml \
        --checkpoint checkpoints/full/checkpoint_best.pt \
        --clinvar data/full/clinvar_variants.csv \
        --device cuda
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import CrossBioSAE, CrossBioSAEConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


AA3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}


def parse_variant(name):
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(name))
    if not match:
        return None
    wt = AA3_TO_1.get(match.group(1))
    pos = int(match.group(2)) - 1
    mt = AA3_TO_1.get(match.group(3))
    if wt and mt:
        return wt, pos, mt
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clinvar", default="data/full/clinvar_variants.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_variants", type=int, default=2000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load gene pairs
    data_dir = config["data"]["data_dir"]
    split = config["data"]["split"]
    with open(f"{data_dir}/gene_pairs_{split}.json") as f:
        gene_pairs = json.load(f)
    gene_map = {p["gene_name"]: p for p in gene_pairs}

    # Load ClinVar and match
    clinvar = pd.read_csv(args.clinvar)
    matched = []
    for _, row in clinvar.iterrows():
        gene = row["GeneSymbol"]
        if gene not in gene_map:
            continue
        parsed = parse_variant(row.get("Name", ""))
        if not parsed:
            continue
        wt_aa, pos, mt_aa = parsed
        protein_seq = gene_map[gene]["protein_seq"]
        if pos >= len(protein_seq) or protein_seq[pos] != wt_aa:
            continue
        matched.append({
            "gene": gene, "pos": pos, "wt_aa": wt_aa, "mt_aa": mt_aa,
            "protein_seq": protein_seq, "label": row["pathogenicity"],
        })
        if len(matched) >= args.max_variants:
            break

    labels = np.array([v["label"] for v in matched])
    logger.info(f"Matched {len(matched)} variants ({labels.sum():.0f} path, {(1-labels).sum():.0f} benign)")

    # Load ESM-2
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    prot_layer = config["protein_model"]["layer"]

    # Load CrossBioSAE
    model_cfg = config["model"]
    train_cfg = config["training"]
    sae_config = CrossBioSAEConfig(
        dim_dna=model_cfg["dim_dna"], dim_protein=model_cfg["dim_protein"],
        dim_shared=model_cfg["dim_shared"], expansion_factor=model_cfg["expansion_factor"],
        sparsity_type=model_cfg.get("sparsity_type", "topk"),
        topk_k=model_cfg.get("topk_k", 64),
        normalize_inputs=model_cfg.get("normalize_inputs", True),
        crossmodal_weight=train_cfg.get("crossmodal_weight", 1.0),
        crossmodal_type=model_cfg.get("crossmodal_type", "cosine"),
    )
    sae_model = CrossBioSAE(sae_config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sae_model.load_state_dict(ckpt["model_state_dict"])
    sae_model.eval()

    # Score all variants
    scores_llr = []       # Baseline 1: ESM-2 log-likelihood ratio
    scores_l2 = []        # Baseline 2: ESM-2 activation L2 distance
    scores_sae = []       # Our method: CrossBioSAE feature disruption

    with torch.no_grad():
        for v in tqdm(matched, desc="Scoring"):
            seq = v["protein_seq"][:1024]
            pos = v["pos"]
            if pos >= len(seq):
                scores_llr.append(0); scores_l2.append(0); scores_sae.append(0)
                continue

            # WT forward pass
            _, _, wt_tokens = batch_converter([("wt", seq)])
            wt_tokens = wt_tokens.to(device)
            wt_out = esm_model(wt_tokens, repr_layers=[prot_layer])

            # Baseline 1: Log-likelihood ratio at mutation position
            wt_logits = wt_out["logits"][0]  # (seq_len, vocab_size)
            wt_aa_idx = alphabet.get_idx(v["wt_aa"])
            mt_aa_idx = alphabet.get_idx(v["mt_aa"])
            token_pos = pos + 1  # +1 for BOS token
            if token_pos < wt_logits.shape[0]:
                log_probs = torch.log_softmax(wt_logits[token_pos], dim=-1)
                llr = (log_probs[wt_aa_idx] - log_probs[mt_aa_idx]).item()
                scores_llr.append(llr)
            else:
                scores_llr.append(0)

            # WT activation (mean-pooled)
            wt_repr = wt_out["representations"][prot_layer]
            wt_act = wt_repr[0, 1:len(seq)+1].mean(dim=0)

            # MT forward pass
            mt_seq = seq[:pos] + v["mt_aa"] + seq[pos+1:]
            _, _, mt_tokens = batch_converter([("mt", mt_seq)])
            mt_tokens = mt_tokens.to(device)
            mt_out = esm_model(mt_tokens, repr_layers=[prot_layer])
            mt_repr = mt_out["representations"][prot_layer]
            mt_act = mt_repr[0, 1:len(mt_seq)+1].mean(dim=0)

            # Baseline 2: L2 distance in ESM-2 activation space
            l2_dist = (wt_act - mt_act).pow(2).sum().sqrt().item()
            scores_l2.append(l2_dist)

            # Our method: CrossBioSAE feature disruption
            wt_feat = sae_model.encode_protein(wt_act.unsqueeze(0))
            mt_feat = sae_model.encode_protein(mt_act.unsqueeze(0))
            sae_disrupt = (wt_feat - mt_feat).abs().sum().item()
            scores_sae.append(sae_disrupt)

    scores_llr = np.array(scores_llr)
    scores_l2 = np.array(scores_l2)
    scores_sae = np.array(scores_sae)

    # Compute metrics
    results = {}
    for name, scores in [
        ("ESM2_LLR", scores_llr),
        ("ESM2_L2", scores_l2),
        ("CrossBioSAE", scores_sae),
    ]:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        results[name] = {"auroc": float(auroc), "auprc": float(auprc)}
        logger.info(f"{name:20s}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON — Variant Effect Prediction")
    print("=" * 60)
    print(f"{'Method':<25} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 45)
    for name in ["ESM2_LLR", "ESM2_L2", "CrossBioSAE"]:
        r = results[name]
        print(f"{name:<25} {r['auroc']:<10.4f} {r['auprc']:<10.4f}")
    print("=" * 60)

    # Save
    output_dir = Path(config["paths"]["output_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame({
        "gene": [v["gene"] for v in matched],
        "position": [v["pos"] for v in matched],
        "wt_aa": [v["wt_aa"] for v in matched],
        "mt_aa": [v["mt_aa"] for v in matched],
        "label": labels,
        "esm2_llr": scores_llr,
        "esm2_l2": scores_l2,
        "crossbiosae": scores_sae,
    })
    df.to_csv(output_dir / "baseline_predictions.csv", index=False)
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
