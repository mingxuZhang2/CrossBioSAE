#!/usr/bin/env python3
"""
V2 baseline comparison with fixes:
  Fix 1: Local window extraction around mutation site (instead of mean-pool)
  Fix 2: LLR + SAE combined features via logistic regression
  Fix 3: Feature disruption normalized by variance

Methods compared:
  1. ESM2_LLR (per-residue log-likelihood ratio) — existing strong baseline
  2. ESM2_L2_global (mean-pooled activation L2) — old broken baseline
  3. ESM2_L2_local (local window L2) — Fix 1 applied to raw L2
  4. SAE_global (old CrossBioSAE, mean-pooled) — old broken method
  5. SAE_local (CrossBioSAE on local window) — Fix 1
  6. SAE_local_norm (Fix 1 + Fix 3: normalized by feature variance)
  7. LLR+SAE_combined (Fix 2: logistic regression on combined features)
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument("--window", type=int, default=10)
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

    # Match variants
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

    # Pre-compute feature variance for Fix 3 (from training data)
    logger.info("Computing feature variance from training data...")
    prot_h5 = config["paths"]["protein_activations"]
    with h5py.File(prot_h5, "r") as f:
        train_acts = torch.tensor(f["activations"][:], dtype=torch.float32)
    with torch.no_grad():
        train_feats = []
        for i in range(0, len(train_acts), 512):
            batch = train_acts[i:i+512].to(device)
            feats = sae_model.encode_protein(batch)
            train_feats.append(feats.cpu())
        train_feats = torch.cat(train_feats, dim=0)
        feature_std = train_feats.std(dim=0).numpy() + 1e-8
    logger.info(f"Feature std computed: mean={feature_std.mean():.4f}, nonzero={np.sum(feature_std > 1e-6)}")

    W = args.window  # local window half-size

    # Score all variants
    all_scores = {
        "llr": [], "l2_global": [], "l2_local": [], "l2_site": [],
        "sae_global": [], "sae_local": [], "sae_local_norm": [], "sae_site": [],
    }

    with torch.no_grad():
        for v in tqdm(matched, desc="Scoring"):
            seq = v["protein_seq"][:1024]
            pos = v["pos"]
            token_pos = pos + 1  # +1 for BOS
            if pos >= len(seq):
                for k in all_scores:
                    all_scores[k].append(0.0)
                continue

            # WT forward
            _, _, wt_tokens = batch_converter([("wt", seq)])
            wt_out = esm_model(wt_tokens.to(device), repr_layers=[prot_layer])
            wt_repr = wt_out["representations"][prot_layer][0]  # (seq_len+2, 1280)

            # LLR
            wt_logits = wt_out["logits"][0]
            wt_aa_idx = alphabet.get_idx(v["wt_aa"])
            mt_aa_idx = alphabet.get_idx(v["mt_aa"])
            if token_pos < wt_logits.shape[0]:
                log_probs = torch.log_softmax(wt_logits[token_pos], dim=-1)
                llr = (log_probs[wt_aa_idx] - log_probs[mt_aa_idx]).item()
            else:
                llr = 0.0
            all_scores["llr"].append(llr)

            # WT activations: global, local window, exact site
            seq_repr = wt_repr[1:len(seq)+1]  # strip BOS/EOS
            wt_global = seq_repr.mean(dim=0)
            start = max(0, pos - W)
            end = min(len(seq), pos + W + 1)
            wt_local = seq_repr[start:end].mean(dim=0)
            wt_site = seq_repr[pos]

            # MT forward
            mt_seq = seq[:pos] + v["mt_aa"] + seq[pos+1:]
            _, _, mt_tokens = batch_converter([("mt", mt_seq)])
            mt_out = esm_model(mt_tokens.to(device), repr_layers=[prot_layer])
            mt_repr = mt_out["representations"][prot_layer][0]
            mt_seq_repr = mt_repr[1:len(mt_seq)+1]
            mt_global = mt_seq_repr.mean(dim=0)
            mt_local = mt_seq_repr[start:end].mean(dim=0)
            mt_site = mt_seq_repr[pos]

            # L2 distances
            all_scores["l2_global"].append((wt_global - mt_global).pow(2).sum().sqrt().item())
            all_scores["l2_local"].append((wt_local - mt_local).pow(2).sum().sqrt().item())
            all_scores["l2_site"].append((wt_site - mt_site).pow(2).sum().sqrt().item())

            # SAE feature disruption: global
            wt_feat_g = sae_model.encode_protein(wt_global.unsqueeze(0))
            mt_feat_g = sae_model.encode_protein(mt_global.unsqueeze(0))
            all_scores["sae_global"].append((wt_feat_g - mt_feat_g).abs().sum().item())

            # SAE feature disruption: local window (Fix 1)
            wt_feat_l = sae_model.encode_protein(wt_local.unsqueeze(0))
            mt_feat_l = sae_model.encode_protein(mt_local.unsqueeze(0))
            disrupt_local = (wt_feat_l - mt_feat_l).abs()
            all_scores["sae_local"].append(disrupt_local.sum().item())

            # SAE feature disruption: local, normalized by variance (Fix 3)
            disrupt_norm = disrupt_local.cpu().numpy() / feature_std
            all_scores["sae_local_norm"].append(float(disrupt_norm.sum()))

            # SAE feature disruption: exact mutation site
            wt_feat_s = sae_model.encode_protein(wt_site.unsqueeze(0))
            mt_feat_s = sae_model.encode_protein(mt_site.unsqueeze(0))
            all_scores["sae_site"].append((wt_feat_s - mt_feat_s).abs().sum().item())

    # Convert to arrays
    for k in all_scores:
        all_scores[k] = np.array(all_scores[k])

    # Fix 2: Combined features via logistic regression with CV
    logger.info("Training combined models (5-fold CV)...")

    combined_results = {}

    # LLR + SAE_local
    X_combined = np.column_stack([
        all_scores["llr"],
        all_scores["sae_local"],
        all_scores["sae_local_norm"],
        all_scores["l2_local"],
        all_scores["sae_site"],
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    cv_probs = cross_val_predict(lr, X_scaled, labels, cv=5, method="predict_proba")[:, 1]
    combined_results["LLR+SAE_combined"] = cv_probs

    # LLR + SAE_site only
    X_llr_site = np.column_stack([all_scores["llr"], all_scores["sae_site"]])
    X_ls_scaled = StandardScaler().fit_transform(X_llr_site)
    lr2 = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    cv_probs2 = cross_val_predict(lr2, X_ls_scaled, labels, cv=5, method="predict_proba")[:, 1]
    combined_results["LLR+SAE_site"] = cv_probs2

    # Compute all metrics
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON V2 — Variant Effect Prediction")
    print(f"({len(labels)} variants: {labels.sum():.0f} pathogenic, {(1-labels).sum():.0f} benign)")
    print("=" * 70)
    print(f"{'Method':<30} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 50)

    all_methods = [
        ("ESM2_LLR", all_scores["llr"]),
        ("ESM2_L2_global", all_scores["l2_global"]),
        ("ESM2_L2_local", all_scores["l2_local"]),
        ("ESM2_L2_site", all_scores["l2_site"]),
        ("SAE_global (old)", all_scores["sae_global"]),
        ("SAE_local (Fix1)", all_scores["sae_local"]),
        ("SAE_local_norm (Fix1+3)", all_scores["sae_local_norm"]),
        ("SAE_site (Fix1)", all_scores["sae_site"]),
        ("LLR+SAE_combined (Fix2)", combined_results["LLR+SAE_combined"]),
        ("LLR+SAE_site (Fix2)", combined_results["LLR+SAE_site"]),
    ]

    results_dict = {}
    for name, scores in all_methods:
        try:
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            results_dict[name] = {"auroc": float(auroc), "auprc": float(auprc)}
            marker = " ***" if auroc > 0.85 else ""
            print(f"{name:<30} {auroc:<10.4f} {auprc:<10.4f}{marker}")
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")

    print("=" * 70)

    # Save
    output_dir = Path(config["paths"]["output_dir"]) / "baselines_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    df = pd.DataFrame({
        "gene": [v["gene"] for v in matched],
        "position": [v["pos"] for v in matched],
        "wt_aa": [v["wt_aa"] for v in matched],
        "mt_aa": [v["mt_aa"] for v in matched],
        "label": labels,
        **{k: v for k, v in all_scores.items()},
        "combined_prob": combined_results["LLR+SAE_combined"],
    })
    df.to_csv(output_dir / "all_predictions.csv", index=False)
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
