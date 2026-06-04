"""
Extract ESM-2 wildtype marginal log-likelihood ratios for all dual-modality variants.

For each unique protein:
  1. Run ESM-2 forward on full wildtype sequence (no masking) → logits
  2. For each variant at position i: LLR = log P(alt_aa) - log P(ref_aa)

This is the primary signal used by ESM-1v for variant effect prediction.
One forward pass per protein, so ~12k passes for our gene set.

Output: results/sae_pretrain_emb/esm2_llr.npz {dual_idx_local, llr}
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ESM_LAYER = 33


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx_path", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--output", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--batch_proteins", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load variant annotations
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx_path)
    sub = df.iloc[dual_idx].reset_index(drop=True)
    print(f"variants: {len(sub)}", flush=True)

    # load protein sequences
    with open(args.gene_pairs) as f:
        gp = json.load(f)
    gene_seqs = {g["gene_name"]: g["protein_seq"] for g in gp if "protein_seq" in g}

    # group variants by gene
    from collections import defaultdict
    gene_vars = defaultdict(list)
    for i, r in sub.iterrows():
        gene = r["gene"]
        if gene not in gene_seqs:
            continue
        seq = gene_seqs[gene]
        pos0 = int(r["prot_pos"]) - 1
        if pos0 < 0 or pos0 >= len(seq):
            continue
        if seq[pos0] != r["from_aa"]:
            continue
        gene_vars[gene].append((i, pos0, r["from_aa"], r["to_aa"]))

    n_genes = len(gene_vars)
    n_vars = sum(len(v) for v in gene_vars.values())
    print(f"genes: {n_genes}, variants mapped: {n_vars}", flush=True)

    # load ESM-2
    print("loading ESM-2 ...", flush=True)
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # amino acid token indices
    aa_tokens = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        tok = alphabet.get_idx(aa)
        aa_tokens[aa] = tok

    # extract LLR per gene
    llr = np.zeros(len(sub), dtype=np.float32)
    done = 0

    for gene, vars_list in gene_vars.items():
        seq = gene_seqs[gene]
        # truncate to ESM-2 max length
        max_len = 1022  # ESM-2 max is 1024 with BOS/EOS
        seq_trunc = seq[:max_len]

        try:
            _, _, tokens = batch_converter([("prot", seq_trunc)])
            tokens = tokens.to(device)

            with torch.no_grad():
                result = model(tokens)
                logits = result["logits"][0]  # (seq_len+2, vocab_size) — includes BOS/EOS
                log_probs = F.log_softmax(logits, dim=-1)

            for local_i, pos0, from_aa, to_aa in vars_list:
                # +1 for BOS token
                esm_pos = pos0 + 1
                if esm_pos >= log_probs.shape[0]:
                    continue
                if from_aa in aa_tokens and to_aa in aa_tokens:
                    lp_ref = log_probs[esm_pos, aa_tokens[from_aa]].item()
                    lp_alt = log_probs[esm_pos, aa_tokens[to_aa]].item()
                    llr[local_i] = lp_alt - lp_ref

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # fallback: skip this gene
            print(f"  OOM on {gene} (len={len(seq)}), skipping", flush=True)

        done += 1
        if done % 500 == 0:
            nz = np.count_nonzero(llr)
            print(f"  {done}/{n_genes} genes, {nz} LLRs computed", flush=True)

    nz = np.count_nonzero(llr)
    print(f"done: {nz}/{len(sub)} variants have LLR ({100*nz/len(sub):.1f}%)", flush=True)

    np.savez_compressed(args.output, llr=llr)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
