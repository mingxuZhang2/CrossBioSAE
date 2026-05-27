#!/usr/bin/env python3
"""
ESM-2 (Science-published protein LM) + Evo (Science-published DNA LM) ensemble.

Evo uses character-level tokenization (A/C/G/T), so per-nucleotide LLR is natural.
For each variant:
  - ESM-2 LLR: log P(wt_aa | protein context) - log P(mt_aa | protein context)
  - Evo LLR: log P(wt_codon | DNA context) - log P(mt_codon | DNA context)
Both are autoregressive log-likelihoods (Evo is causal LM).
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


def compute_evo_llr(model, tokenizer, wt_cds, codon_pos, mt_aa, device):
    """
    Compute Evo log-likelihood ratio at mutation site.

    Evo is autoregressive (causal LM) with character-level tokenization.
    For each nucleotide in the mutated codon:
      LLR = log P(wt_nt | left_context) - log P(mt_nt | left_context)

    We run the full WT CDS through Evo to get logits at every position.
    """
    mt_codon = CODON_TABLE.get(mt_aa, "NNN")
    codon_start = codon_pos * 3

    if codon_start + 3 > len(wt_cds):
        return 0.0, 0.0

    # Truncate CDS around the mutation for efficiency
    # Use a window of context
    window = 512  # characters on each side
    start = max(0, codon_start - window)
    end = min(len(wt_cds), codon_start + 3 + window)
    local_cds = wt_cds[start:end]
    local_codon_start = codon_start - start

    # Tokenize
    inputs = tokenizer(local_cds, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Evo character-level: each nucleotide is typically 1 token
    # But need to verify the token-to-character mapping
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # For autoregressive LM: logits[i] predicts token at position i+1
    # So to get P(token_at_pos | left_context), use logits[pos-1]
    total_llr = 0.0
    total_marginal = 0.0

    for offset in range(3):
        nt_pos = local_codon_start + offset
        wt_nt = local_cds[nt_pos]
        mt_nt = mt_codon[offset]

        if wt_nt == mt_nt:
            continue

        # Token IDs for individual nucleotides
        wt_token_ids = tokenizer.encode(wt_nt, add_special_tokens=False)
        mt_token_ids = tokenizer.encode(mt_nt, add_special_tokens=False)

        if not wt_token_ids or not mt_token_ids:
            continue

        # For autoregressive: prediction at position i uses logits[i-1]
        # But with special tokens, the mapping might be shifted
        # The input_ids include special tokens, so find the offset
        # Decode input_ids to verify alignment
        all_tokens = input_ids[0].tolist()

        # Find which token position corresponds to nt_pos
        # For character-level tokenizer, token_pos ≈ nt_pos + n_special_prefix
        # Let's use a robust approach: decode each token and find the position
        decoded_so_far = 0
        target_token_pos = None
        for tidx, tid in enumerate(all_tokens):
            token_str = tokenizer.decode([tid])
            if decoded_so_far <= nt_pos < decoded_so_far + len(token_str):
                target_token_pos = tidx
                break
            decoded_so_far += len(token_str)

        if target_token_pos is None or target_token_pos == 0:
            continue

        # Autoregressive: logits[target_token_pos - 1] predicts token at target_token_pos
        pred_logits = logits[target_token_pos - 1]
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        wt_log_p = log_probs[wt_token_ids[0]].item()
        mt_log_p = log_probs[mt_token_ids[0]].item()

        total_llr += (wt_log_p - mt_log_p)
        total_marginal += (-mt_log_p)

    return total_llr, total_marginal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clinvar", default="data/full/clinvar_variants.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_variants", type=int, default=2000)
    parser.add_argument("--evo_path", default=None)
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

    scores_esm = []
    logger.info("Scoring with ESM-2 (Science 2023)...")
    with torch.no_grad():
        for v in tqdm(matched, desc="ESM-2 LLR"):
            seq = v["protein_seq"][:1024]
            pos = v["pos"]
            token_pos = pos + 1
            _, _, tokens = batch_converter([("seq", seq)])
            out = esm_model(tokens.to(device))
            logits = out["logits"][0]
            if token_pos < logits.shape[0]:
                lp = torch.log_softmax(logits[token_pos], dim=-1)
                llr = (lp[alphabet.get_idx(v["wt_aa"])] - lp[alphabet.get_idx(v["mt_aa"])]).item()
            else:
                llr = 0.0
            scores_esm.append(llr)

    del esm_model
    torch.cuda.empty_cache()

    # ============================================================
    # Evo LLR
    # ============================================================
    from transformers import AutoModelForCausalLM, AutoTokenizer

    evo_path = args.evo_path
    if evo_path is None:
        candidates = [
            os.path.expanduser("~/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base/snapshots"),
            "/tmp/evo1",
        ]
        for c in candidates:
            if os.path.isdir(c):
                # Find the actual snapshot dir
                if "snapshots" in c:
                    snaps = os.listdir(c)
                    if snaps:
                        evo_path = os.path.join(c, snaps[0])
                        break
                else:
                    evo_path = c
                    break
        if evo_path is None:
            evo_path = "togethercomputer/evo-1-8k-base"

    logger.info(f"Loading Evo (Science 2024) from {evo_path}...")
    evo_tokenizer = AutoTokenizer.from_pretrained(evo_path, trust_remote_code=True)
    evo_model = AutoModelForCausalLM.from_pretrained(
        evo_path, trust_remote_code=True, torch_dtype=torch.float16,
    ).eval().to(device)
    logger.info(f"Evo loaded. Vocab size: {evo_tokenizer.vocab_size}")

    scores_evo_llr = []
    scores_evo_marginal = []

    logger.info("Scoring with Evo (Science 2024)...")
    with torch.no_grad():
        for v in tqdm(matched, desc="Evo LLR"):
            llr, marginal = compute_evo_llr(
                evo_model, evo_tokenizer, v["wt_cds"], v["pos"], v["mt_aa"], device,
            )
            scores_evo_llr.append(llr)
            scores_evo_marginal.append(marginal)

    del evo_model
    torch.cuda.empty_cache()

    # ============================================================
    # Combine and evaluate
    # ============================================================
    scores_esm = np.array(scores_esm)
    scores_evo_llr = np.array(scores_evo_llr)
    scores_evo_marginal = np.array(scores_evo_marginal)

    methods = [
        ("ESM-2 LLR (protein)", scores_esm),
        ("Evo LLR (DNA)", scores_evo_llr),
        ("Evo marginal (DNA)", scores_evo_marginal),
        ("ESM2 + Evo LLR (sum)", scores_esm + scores_evo_llr),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_sets = {
        "LR(ESM2 + Evo)": np.column_stack([scores_esm, scores_evo_llr]),
        "LR(ESM2 + Evo_all)": np.column_stack([scores_esm, scores_evo_llr, scores_evo_marginal]),
        "LR(all + interact)": np.column_stack([
            scores_esm, scores_evo_llr, scores_evo_marginal,
            scores_esm * scores_evo_llr,
            np.abs(scores_esm - scores_evo_llr),  # disagreement feature
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
    print("ESM-2 + Evo CROSS-MODAL ENSEMBLE — Variant Effect Prediction")
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
            print(f"{name:<35} {auroc:<10.4f} {auprc:<10.4f}{'  ***' if best else ''}")
        except Exception as e:
            print(f"{name:<35} ERROR: {e}")

    print("=" * 70)

    output_dir = Path(config["paths"]["output_dir"]) / "baselines_evo"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
