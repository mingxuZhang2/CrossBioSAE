#!/usr/bin/env python3
"""
V4: ESM-2 LLR + DNABERT-2 masked LLR ensemble.

DNABERT-2 uses BPE tokenization on nucleotides → natural masked LLR.
For each variant: mask the mutated codon, ask DNABERT-2 to predict it.
LLR = log P(WT codon tokens | context) - log P(MT codon tokens | context)
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AA3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

CODON_TABLE = {
    'A': 'GCT', 'R': 'CGT', 'N': 'AAT', 'D': 'GAT', 'C': 'TGT',
    'E': 'GAA', 'Q': 'CAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCT',
    'S': 'TCT', 'T': 'ACT', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTT',
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


def compute_dnabert2_llr(model, tokenizer, wt_cds, codon_pos, mt_aa, device, mask_token_id):
    """
    Compute DNABERT-2 masked LLR at the mutation site.

    Strategy: mask each nucleotide in the mutated codon one at a time,
    compute log P(WT nt) - log P(MT nt) given context, sum over codon.
    """
    mt_codon = CODON_TABLE.get(mt_aa, "NNN")
    codon_start = codon_pos * 3

    if codon_start + 3 > len(wt_cds):
        return 0.0, 0.0

    wt_codon = wt_cds[codon_start:codon_start + 3]

    # For each nucleotide position in the codon, mask and score
    total_llr = 0.0
    total_marginal = 0.0

    for offset in range(3):
        nt_pos = codon_start + offset
        wt_nt = wt_cds[nt_pos]
        mt_nt = mt_codon[offset]

        if wt_nt == mt_nt:
            continue  # synonymous at this position

        # Create masked sequence: replace this nucleotide with [MASK]
        masked_seq = wt_cds[:nt_pos] + "[MASK]" + wt_cds[nt_pos + 1:]

        # Tokenize
        inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"][0]

        # Find the mask token position
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            # [MASK] might have been absorbed into a BPE token
            # Fallback: tokenize WT and MT separately, compare likelihoods
            continue

        mask_pos = mask_positions[0].item()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_pos]  # (vocab_size,)
            log_probs = torch.log_softmax(logits, dim=-1)

        # Get token IDs for WT and MT nucleotides
        wt_token = tokenizer.encode(wt_nt, add_special_tokens=False)
        mt_token = tokenizer.encode(mt_nt, add_special_tokens=False)

        if len(wt_token) == 1 and len(mt_token) == 1:
            wt_log_p = log_probs[wt_token[0]].item()
            mt_log_p = log_probs[mt_token[0]].item()
            total_llr += (wt_log_p - mt_log_p)
            total_marginal += (-mt_log_p)

    return total_llr, total_marginal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clinvar", default="data/full/clinvar_variants.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_variants", type=int, default=2000)
    parser.add_argument("--dnabert2_path", default=None)
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

        wt_cds = gene_map[gene].get("cds_seq", "")
        if not wt_cds or len(wt_cds) < (pos + 1) * 3:
            continue

        matched.append({
            "gene": gene, "pos": pos, "wt_aa": wt_aa, "mt_aa": mt_aa,
            "protein_seq": protein_seq, "wt_cds": wt_cds,
            "label": row["pathogenicity"],
        })
        if len(matched) >= args.max_variants:
            break

    labels = np.array([v["label"] for v in matched])
    logger.info(f"Matched {len(matched)} variants ({labels.sum():.0f} path, {(1-labels).sum():.0f} benign)")

    # ============================================================
    # ESM-2 LLR
    # ============================================================
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    scores_esm_llr = []
    logger.info("Scoring with ESM-2...")
    with torch.no_grad():
        for v in tqdm(matched, desc="ESM-2"):
            seq = v["protein_seq"][:1024]
            pos = v["pos"]
            token_pos = pos + 1

            _, _, tokens = batch_converter([("seq", seq)])
            out = esm_model(tokens.to(device))
            logits = out["logits"][0]

            if token_pos < logits.shape[0]:
                log_probs = torch.log_softmax(logits[token_pos], dim=-1)
                llr = (log_probs[alphabet.get_idx(v["wt_aa"])] - log_probs[alphabet.get_idx(v["mt_aa"])]).item()
            else:
                llr = 0.0
            scores_esm_llr.append(llr)

    del esm_model
    torch.cuda.empty_cache()

    # ============================================================
    # DNABERT-2 masked LLR
    # ============================================================
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dnabert_path = args.dnabert2_path
    if dnabert_path is None:
        local_path = os.path.expanduser("~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/main")
        dnabert_path = local_path if os.path.isdir(local_path) else "zhihan1996/DNABERT-2-117M"

    logger.info(f"Loading DNABERT-2 from {dnabert_path}...")
    dna_tokenizer = AutoTokenizer.from_pretrained(dnabert_path, trust_remote_code=True)
    dna_model = AutoModelForMaskedLM.from_pretrained(dnabert_path, trust_remote_code=True).eval().to(device)

    mask_token_id = dna_tokenizer.mask_token_id
    logger.info(f"DNABERT-2 loaded. Mask token ID: {mask_token_id}")

    scores_dna_llr = []
    scores_dna_marginal = []

    logger.info("Scoring with DNABERT-2...")
    with torch.no_grad():
        for v in tqdm(matched, desc="DNABERT-2"):
            llr, marginal = compute_dnabert2_llr(
                dna_model, dna_tokenizer, v["wt_cds"],
                v["pos"], v["mt_aa"], device, mask_token_id,
            )
            scores_dna_llr.append(llr)
            scores_dna_marginal.append(marginal)

    del dna_model
    torch.cuda.empty_cache()

    # ============================================================
    # Evaluate
    # ============================================================
    scores_esm_llr = np.array(scores_esm_llr)
    scores_dna_llr = np.array(scores_dna_llr)
    scores_dna_marginal = np.array(scores_dna_marginal)

    methods = [
        ("ESM2_LLR", scores_esm_llr),
        ("DNABERT2_LLR", scores_dna_llr),
        ("DNABERT2_marginal", scores_dna_marginal),
        ("ESM2+DNABERT2 (sum)", scores_esm_llr + scores_dna_llr),
    ]

    # Learned combinations
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_sets = {
        "LR(ESM2+DNABERT2)": np.column_stack([scores_esm_llr, scores_dna_llr]),
        "LR(ESM2+DNABERT2_all)": np.column_stack([scores_esm_llr, scores_dna_llr, scores_dna_marginal]),
        "LR(all+interaction)": np.column_stack([
            scores_esm_llr, scores_dna_llr, scores_dna_marginal,
            scores_esm_llr * scores_dna_llr,
        ]),
    }

    for name, X in feature_sets.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        try:
            cv_probs = cross_val_predict(lr, X_scaled, labels, cv=cv, method="predict_proba")[:, 1]
            methods.append((name, cv_probs))
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    # Print
    print("\n" + "=" * 70)
    print("CROSS-MODAL ENSEMBLE V4 — ESM-2 + DNABERT-2")
    print(f"({len(labels)} variants: {labels.sum():.0f} pathogenic, {(1-labels).sum():.0f} benign)")
    print("=" * 70)
    print(f"{'Method':<35} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 55)

    results_dict = {}
    for name, scores in methods:
        try:
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            results_dict[name] = {"auroc": float(auroc), "auprc": float(auprc)}
            best = auroc >= max(r["auroc"] for r in results_dict.values()) - 0.001
            print(f"{name:<35} {auroc:<10.4f} {auprc:<10.4f}{'  *** BEST' if best else ''}")
        except Exception as e:
            print(f"{name:<35} ERROR: {e}")

    print("=" * 70)

    # Save
    output_dir = Path(config["paths"]["output_dir"]) / "baselines_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
