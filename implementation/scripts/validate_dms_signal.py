"""
Validate whether the cross-modal advantage on GFP fitness is real or artifact.

Three checks:
1. Synonymous variant signal: do synonymous variants (identical protein) have
   real brightness variation, or is it noise? Compare within-protein brightness
   variance to measurement noise (std column).
2. ESM-2 zero-shot baseline: standard variant effect prediction via masked-marginal
   log-likelihood at mutated positions (the correct way, not mean-pooling).
3. Per-position vs mean-pool: show mean-pooling destroys signal.
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(nt_file, ref_file, max_mutations=5):
    with open(ref_file) as f:
        wt_cds = "".join(l.strip() for l in f if not l.startswith(">"))
    from Bio.Seq import Seq
    wt_protein = str(Seq(wt_cds).translate()).rstrip("*")

    df = pd.read_csv(nt_file, sep="\t")
    variants = []
    for _, row in df.iterrows():
        nt_muts = row["nMutations"]
        brightness = row["medianBrightness"]
        std = row["std"]
        if pd.isna(brightness):
            continue
        cds = list(wt_cds)
        if pd.isna(nt_muts) or nt_muts == "":
            n_mut = 0
        else:
            muts = nt_muts.split(":")
            n_mut = len(muts)
            if n_mut > max_mutations:
                continue
            for m in muts:
                match = re.match(r"S([ACGT])(\d+)([ACGT])", m)
                if match:
                    pos = int(match.group(2)) - 1
                    if pos < len(cds):
                        cds[pos] = match.group(3)
        var_cds = "".join(cds)
        try:
            var_protein = str(Seq(var_cds).translate()).rstrip("*")
        except Exception:
            continue
        if "*" in var_protein:
            continue
        variants.append({
            "cds": var_cds, "protein": var_protein,
            "brightness": brightness, "std": std,
            "n_nt_mut": n_mut,
            "aa_mutations": _get_aa_mutations(wt_protein, var_protein),
        })
    return pd.DataFrame(variants), wt_cds, wt_protein


def _get_aa_mutations(wt, var):
    if len(wt) != len(var):
        return None
    muts = [f"{wt[i]}{i+1}{var[i]}" for i in range(len(wt)) if wt[i] != var[i]]
    return muts


# ============================================================
# Check 1: Synonymous variant signal vs noise
# ============================================================
def check_synonymous_signal(df, wt_protein):
    """Group variants by protein sequence. For proteins with multiple synonymous
    CDS variants, check if brightness variation exceeds measurement noise."""
    logger.info("\n=== Check 1: Synonymous variant signal ===")

    # Group by protein sequence
    groups = df.groupby("protein")
    multi_syn = []
    for protein, grp in groups:
        if len(grp) >= 3:  # at least 3 synonymous variants
            multi_syn.append({
                "protein": protein,
                "n_variants": len(grp),
                "brightness_std": grp["brightness"].std(),
                "brightness_range": grp["brightness"].max() - grp["brightness"].min(),
                "mean_measurement_std": grp["std"].mean(),
                "brightness_values": grp["brightness"].values,
            })

    df_syn = pd.DataFrame(multi_syn)
    logger.info(f"Proteins with >=3 synonymous CDS variants: {len(df_syn)}")

    if len(df_syn) > 0:
        # Compare between-variant std to within-variant measurement std
        valid = df_syn.dropna(subset=["brightness_std", "mean_measurement_std"])
        valid = valid[valid["mean_measurement_std"] > 0]
        logger.info(f"Valid groups: {len(valid)}")
        logger.info(f"Mean brightness std across synonymous variants: {valid['brightness_std'].mean():.4f}")
        logger.info(f"Mean measurement noise (std column): {valid['mean_measurement_std'].mean():.4f}")
        ratio = valid['brightness_std'].mean() / valid['mean_measurement_std'].mean()
        logger.info(f"Ratio (signal/noise): {ratio:.2f}")
        if ratio > 1.5:
            logger.info("  => Synonymous brightness variation EXCEEDS measurement noise: REAL signal")
        else:
            logger.info("  => Synonymous variation comparable to noise: likely NOISE")

        # Wild-type protein synonymous variants specifically
        wt_grp = df[df["protein"] == wt_protein]
        logger.info(f"\nWild-type protein has {len(wt_grp)} synonymous CDS variants")
        if len(wt_grp) > 1:
            logger.info(f"  WT synonymous brightness: mean={wt_grp['brightness'].mean():.3f}, "
                        f"std={wt_grp['brightness'].std():.3f}, "
                        f"range=[{wt_grp['brightness'].min():.3f}, {wt_grp['brightness'].max():.3f}]")
            logger.info(f"  WT measurement noise: {wt_grp['std'].mean():.3f}")

    return df_syn


# ============================================================
# Check 2: ESM-2 zero-shot baseline (proper variant effect scoring)
# ============================================================
@torch.no_grad()
def esm2_zero_shot(df, wt_protein, device, batch_size=16):
    """Compute ESM-2 masked-marginal scores for single+multi aa mutations.
    Score = sum over mutated positions of log P(mut_aa) - log P(wt_aa)."""
    logger.info("\n=== Check 2: ESM-2 zero-shot variant effect ===")
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    # Only variants with valid aa mutations (same length as WT)
    valid_df = df[df["aa_mutations"].notna()].copy()
    valid_df = valid_df[valid_df["aa_mutations"].apply(lambda x: x is not None and len(x) > 0)].copy()
    logger.info(f"Variants with aa mutations: {len(valid_df)}")

    # Get WT token probabilities (single forward pass on WT)
    _, _, wt_tokens = batch_converter([("wt", wt_protein)])
    wt_tokens = wt_tokens.to(device)
    wt_logits = model(wt_tokens)["logits"][0]  # (L+2, vocab)
    wt_log_probs = torch.log_softmax(wt_logits, dim=-1)

    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    scores = []
    for _, row in valid_df.iterrows():
        score = 0.0
        valid_mut = True
        for mut in row["aa_mutations"]:
            wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            if mut_aa not in aa_to_idx or wt_aa not in aa_to_idx:
                valid_mut = False
                break
            # token position = pos (1-indexed because of BOS at index 0)
            tok_pos = pos
            if tok_pos >= wt_log_probs.shape[0] - 1:
                valid_mut = False
                break
            score += (wt_log_probs[tok_pos, aa_to_idx[mut_aa]].item() -
                      wt_log_probs[tok_pos, aa_to_idx[wt_aa]].item())
        if valid_mut:
            scores.append(score)
        else:
            scores.append(np.nan)

    valid_df["esm_score"] = scores
    valid_df = valid_df.dropna(subset=["esm_score"])

    rho, _ = spearmanr(valid_df["brightness"], valid_df["esm_score"])
    logger.info(f"ESM-2 zero-shot Spearman ρ (n={len(valid_df)}): {rho:.4f}")
    logger.info(f"  (ProteinGym reports ~0.45-0.50 for ESM-2 650M on GFP)")

    del model
    torch.cuda.empty_cache()
    return rho, valid_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt_file", required=True)
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, wt_cds, wt_protein = load_data(args.nt_file, args.ref_file)
    logger.info(f"Loaded {len(df)} variants, WT protein {len(wt_protein)} aa")

    # Check 1: synonymous signal
    df_syn = check_synonymous_signal(df, wt_protein)

    # Check 2: ESM-2 zero-shot
    esm_rho, esm_df = esm2_zero_shot(df, wt_protein, args.device)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"ESM-2 zero-shot (proper variant scoring): ρ = {esm_rho:.4f}")
    print(f"  vs our mean-pooled protein-only: ρ = 0.338")
    print(f"  vs our mean-pooled DNA-only:     ρ = 0.466")
    print(f"  vs our mean-pooled concat:       ρ = 0.507")
    print()
    if esm_rho > 0.45:
        print("=> ESM-2 zero-shot is STRONG. Our mean-pooled protein baseline is")
        print("   artificially weak. The cross-modal 'advantage' may be a pooling artifact.")
    else:
        print("=> ESM-2 zero-shot is also weak on this variant set.")


if __name__ == "__main__":
    main()
