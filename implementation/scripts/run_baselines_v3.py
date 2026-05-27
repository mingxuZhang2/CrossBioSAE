#!/usr/bin/env python3
"""
V3: True cross-modal ensemble — directly combine ESM-2 LLR + NT LLR.
No SAE bottleneck for prediction. SAE is for interpretability only.

The key insight: ESM-2 LLR measures amino acid substitution plausibility,
NT LLR measures codon substitution plausibility. These are DIFFERENT signals.
Combining them should beat either alone.
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


def protein_to_cds(seq):
    return "".join(CODON_TABLE.get(aa, "NNN") for aa in seq)


def mutate_cds_at_position(wt_cds: str, pos: int, mt_aa: str) -> str:
    """Change only the codon at `pos` in the real CDS to encode `mt_aa`.
    Preserves all other codons unchanged."""
    codon_start = pos * 3
    if codon_start + 3 > len(wt_cds):
        return protein_to_cds(wt_cds)  # fallback
    mt_codon = CODON_TABLE.get(mt_aa, "NNN")
    return wt_cds[:codon_start] + mt_codon + wt_cds[codon_start + 3:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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

        # Build WT and MT CDS — only change the mutated codon, keep rest of real CDS
        wt_cds = gene_map[gene].get("cds_seq", protein_to_cds(protein_seq))
        mt_protein = protein_seq[:pos] + mt_aa + protein_seq[pos+1:]
        mt_cds = mutate_cds_at_position(wt_cds, pos, mt_aa)

        matched.append({
            "gene": gene, "pos": pos, "wt_aa": wt_aa, "mt_aa": mt_aa,
            "protein_seq": protein_seq, "mt_protein": mt_protein,
            "wt_cds": wt_cds, "mt_cds": mt_cds,
            "label": row["pathogenicity"],
        })
        if len(matched) >= args.max_variants:
            break

    labels = np.array([v["label"] for v in matched])
    logger.info(f"Matched {len(matched)} variants ({labels.sum():.0f} path, {(1-labels).sum():.0f} benign)")

    # ============================================================
    # Model 1: ESM-2 LLR (protein-level)
    # ============================================================
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    scores_esm_llr = []
    scores_esm_marginal = []  # P(mt_aa | context) — marginal likelihood of mutant

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
                wt_idx = alphabet.get_idx(v["wt_aa"])
                mt_idx = alphabet.get_idx(v["mt_aa"])
                llr = (log_probs[wt_idx] - log_probs[mt_idx]).item()
                marginal = log_probs[mt_idx].item()
            else:
                llr = 0.0
                marginal = 0.0

            scores_esm_llr.append(llr)
            scores_esm_marginal.append(-marginal)  # higher = less likely = more pathogenic

    del esm_model
    torch.cuda.empty_cache()

    # ============================================================
    # Model 2: NT v2 LLR (DNA-level)
    # ============================================================
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    nt_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--InstaDeepAI--nucleotide-transformer-v2-500m-multi-species/"
        "snapshots/main"
    )
    load_from = nt_path if os.path.isdir(nt_path) else "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

    logger.info(f"Loading NT v2 from {load_from}...")
    nt_tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
    nt_model = AutoModelForMaskedLM.from_pretrained(load_from, trust_remote_code=True).eval().to(device)

    scores_nt_llr = []
    scores_nt_marginal = []

    logger.info("Scoring with NT v2...")
    with torch.no_grad():
        for v in tqdm(matched, desc="NT"):
            wt_cds = v["wt_cds"][:2048]
            mt_cds = v["mt_cds"][:2048]

            # Find the codon position that changed
            codon_pos = v["pos"] * 3  # each amino acid = 3 nucleotides in CDS

            # Tokenize WT CDS and get logits
            wt_inputs = nt_tokenizer(wt_cds, return_tensors="pt", truncation=True, max_length=2048).to(device)
            wt_out = nt_model(**wt_inputs)
            wt_logits = wt_out.logits[0]  # (seq_len, vocab_size)

            # NT tokenizes DNA into 6-mers, so token position != nucleotide position
            # We compute a global CDS-level score instead: log P(WT CDS) vs log P(MT CDS)
            # by masking each differing position and comparing probabilities

            # Simpler approach: compute pseudo-perplexity difference
            # Score = sum of log P(wt_token | context) - log P(mt_token | context) over differing tokens
            mt_inputs = nt_tokenizer(mt_cds, return_tensors="pt", truncation=True, max_length=2048).to(device)
            mt_out = nt_model(**mt_inputs)
            mt_logits = mt_out.logits[0]

            # Pseudo-likelihood: sum of log P(token_i | all other tokens) for each position
            wt_token_ids = wt_inputs["input_ids"][0]
            mt_token_ids = mt_inputs["input_ids"][0]

            # Find tokens that differ
            min_len = min(len(wt_token_ids), len(mt_token_ids))
            wt_ids = wt_token_ids[:min_len]
            mt_ids = mt_token_ids[:min_len]

            diff_mask = (wt_ids != mt_ids)
            n_diff = diff_mask.sum().item()

            if n_diff > 0:
                wt_log_probs = torch.log_softmax(wt_logits[:min_len], dim=-1)
                mt_log_probs = torch.log_softmax(mt_logits[:min_len], dim=-1)

                # At positions where tokens differ, compare how well each model predicts its own tokens
                wt_score = wt_log_probs[diff_mask].gather(1, wt_ids[diff_mask].unsqueeze(1)).sum().item()
                mt_score = mt_log_probs[diff_mask].gather(1, mt_ids[diff_mask].unsqueeze(1)).sum().item()

                # Also: how well does WT context predict the MT token (and vice versa)?
                wt_context_mt_score = wt_log_probs[diff_mask].gather(1, mt_ids[diff_mask].unsqueeze(1)).sum().item()

                nt_llr = wt_score - wt_context_mt_score  # higher = WT is more natural than MT at these positions
                nt_marginal = -wt_context_mt_score  # higher = MT less likely in WT context
            else:
                nt_llr = 0.0
                nt_marginal = 0.0

            scores_nt_llr.append(nt_llr)
            scores_nt_marginal.append(nt_marginal)

    del nt_model
    torch.cuda.empty_cache()

    # ============================================================
    # Combine and evaluate
    # ============================================================
    scores_esm_llr = np.array(scores_esm_llr)
    scores_esm_marginal = np.array(scores_esm_marginal)
    scores_nt_llr = np.array(scores_nt_llr)
    scores_nt_marginal = np.array(scores_nt_marginal)

    # Individual model scores
    methods = [
        ("ESM2_LLR", scores_esm_llr),
        ("ESM2_marginal", scores_esm_marginal),
        ("NT_LLR", scores_nt_llr),
        ("NT_marginal", scores_nt_marginal),
    ]

    # Simple combinations (no training needed)
    methods.append(("ESM2_LLR + NT_LLR (sum)", scores_esm_llr + scores_nt_llr))
    methods.append(("ESM2_LLR + NT_marginal (sum)", scores_esm_llr + scores_nt_marginal))

    # Learned combinations via CV logistic regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_sets = {
        "LR(ESM+NT_LLR)": np.column_stack([scores_esm_llr, scores_nt_llr]),
        "LR(ESM+NT_all)": np.column_stack([scores_esm_llr, scores_esm_marginal, scores_nt_llr, scores_nt_marginal]),
        "LR(all+interaction)": np.column_stack([
            scores_esm_llr, scores_esm_marginal, scores_nt_llr, scores_nt_marginal,
            scores_esm_llr * scores_nt_llr,  # interaction term
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

    # Print results
    print("\n" + "=" * 70)
    print("CROSS-MODAL ENSEMBLE — Variant Effect Prediction")
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
            marker = " *** BEST" if auroc >= max(r["auroc"] for r in results_dict.values()) - 0.001 else ""
            print(f"{name:<35} {auroc:<10.4f} {auprc:<10.4f}{marker}")
        except Exception as e:
            print(f"{name:<35} ERROR: {e}")

    print("=" * 70)

    # Save
    output_dir = Path(config["paths"]["output_dir"]) / "baselines_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    df = pd.DataFrame({
        "gene": [v["gene"] for v in matched],
        "position": [v["pos"] for v in matched],
        "wt_aa": [v["wt_aa"] for v in matched],
        "mt_aa": [v["mt_aa"] for v in matched],
        "label": labels,
        "esm2_llr": scores_esm_llr,
        "esm2_marginal": scores_esm_marginal,
        "nt_llr": scores_nt_llr,
        "nt_marginal": scores_nt_marginal,
    })
    df.to_csv(output_dir / "all_predictions.csv", index=False)
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
