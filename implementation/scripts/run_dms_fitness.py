"""
DMS fitness prediction: cross-modal protein+DNA benchmark.

Task: predict GFP fluorescence from variant protein sequence + variant CDS.
Compare protein-only, DNA-only, and cross-modal representations.

This is a sequence-level task (one embedding per variant), not gene-level.
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gfp_data(nt_file, ref_file, max_mutations=3, min_barcodes=2):
    """Load GFP DMS data. Returns variant CDS sequences + brightness."""
    with open(ref_file) as f:
        lines = f.readlines()
        wt_cds = "".join(l.strip() for l in lines if not l.startswith(">"))

    wt_protein = str(__import__("Bio.Seq", fromlist=["Seq"]).Seq(wt_cds).translate()).rstrip("*")
    logger.info(f"WT CDS: {len(wt_cds)} nt, WT protein: {len(wt_protein)} aa")

    df = pd.read_csv(nt_file, sep="\t")
    df = df[df["uniqueBarcodes"] >= min_barcodes].copy()

    variants = []
    for _, row in df.iterrows():
        nt_muts = row["nMutations"]
        brightness = row["medianBrightness"]

        if pd.isna(brightness):
            continue

        # Parse nucleotide mutations (format: SA191C:SC263T)
        cds = list(wt_cds)
        if pd.isna(nt_muts) or nt_muts == "":
            var_cds = wt_cds
            n_mut = 0
        else:
            muts = nt_muts.split(":")
            n_mut = len(muts)
            if n_mut > max_mutations:
                continue
            for m in muts:
                match = re.match(r"S([ACGT])(\d+)([ACGT])", m)
                if match:
                    pos = int(match.group(2)) - 1  # 0-indexed
                    if pos < len(cds):
                        cds[pos] = match.group(3)
            var_cds = "".join(cds)

        # Translate to get protein sequence
        from Bio.Seq import Seq
        try:
            var_protein = str(Seq(var_cds).translate()).rstrip("*")
        except Exception:
            continue

        # Skip if premature stop codon
        if "*" in var_protein:
            continue

        variants.append({
            "cds": var_cds,
            "protein": var_protein,
            "brightness": brightness,
            "n_nt_mutations": n_mut,
            "has_aa_mutation": var_protein != wt_protein,
        })

    df_out = pd.DataFrame(variants)
    logger.info(f"Variants: {len(df_out)} (max {max_mutations} nt mutations, min {min_barcodes} barcodes)")
    logger.info(f"  Synonymous only: {(~df_out['has_aa_mutation']).sum()}")
    logger.info(f"  With AA changes: {df_out['has_aa_mutation'].sum()}")
    return df_out, wt_cds, wt_protein


@torch.no_grad()
def extract_esm2_embeddings(sequences, model, alphabet, device, batch_size=32):
    """Extract mean-pooled ESM-2 embeddings for a list of protein sequences."""
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = [(f"seq_{j}", s) for j, s in enumerate(sequences[i:i + batch_size])]
        _, _, batch_tokens = batch_converter(batch_seqs)
        batch_tokens = batch_tokens.to(device)

        results = model(batch_tokens, repr_layers=[model.num_layers])
        embeddings = results["representations"][model.num_layers]

        # Mean pool (exclude BOS/EOS tokens)
        for j, (_, seq) in enumerate(batch_seqs):
            emb = embeddings[j, 1:len(seq) + 1].mean(dim=0).cpu().numpy()
            all_embeddings.append(emb)

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM-2: {i + len(batch_seqs)}/{len(sequences)}")

    return np.array(all_embeddings)


@torch.no_grad()
def extract_nt_embeddings(sequences, model, tokenizer, device, batch_size=32):
    """Extract mean-pooled Nucleotide Transformer embeddings for CDS sequences."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # last layer

        # Mean pool over non-padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        all_embeddings.append(pooled.cpu().numpy())

        if (i // batch_size) % 50 == 0:
            logger.info(f"  NT: {i + len(batch)}/{len(sequences)}")

    return np.concatenate(all_embeddings, axis=0)


class MLPProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_fitness(X, y, name, device, n_splits=5, epochs=100):
    """Evaluate representation on fitness prediction. Returns Spearman rho."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_rhos = []
    for train_idx, test_idx in kf.split(X_scaled):
        model = MLPProbe(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        X_tr = torch.tensor(X_scaled[train_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
        X_te = torch.tensor(X_scaled[test_idx], dtype=torch.float32, device=device)

        model.train()
        for _ in range(epochs):
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_te).cpu().numpy()

        rho, _ = spearmanr(y[test_idx], pred)
        fold_rhos.append(rho)

    rhos = np.array(fold_rhos)
    logger.info(f"  {name:30s} ({X.shape[1]:4d}d): Spearman ρ = {rhos.mean():.4f} ± {rhos.std():.4f}")
    return rhos.mean(), rhos.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt_file", required=True, help="nucleotide_genotypes_to_brightness.tsv")
    parser.add_argument("--ref_file", required=True, help="avGFP reference FASTA")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_mutations", type=int, default=3)
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df, wt_cds, wt_protein = load_gfp_data(args.nt_file, args.ref_file, args.max_mutations)
    y = df["brightness"].values

    # Extract ESM-2 embeddings
    logger.info("Loading ESM-2...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(args.device)
    esm_model.eval()

    logger.info("Extracting ESM-2 embeddings...")
    prot_embeddings = extract_esm2_embeddings(df["protein"].tolist(), esm_model, alphabet, args.device)
    logger.info(f"Protein embeddings: {prot_embeddings.shape}")

    del esm_model
    torch.cuda.empty_cache()

    # Extract NT embeddings
    logger.info("Loading Nucleotide Transformer...")
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    nt_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    nt_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    nt_model = nt_model.to(args.device)
    nt_model.eval()

    logger.info("Extracting NT embeddings...")
    dna_embeddings = extract_nt_embeddings(df["cds"].tolist(), nt_model, nt_tokenizer, args.device)
    logger.info(f"DNA embeddings: {dna_embeddings.shape}")

    del nt_model
    torch.cuda.empty_cache()

    # Build representations
    concat = np.hstack([prot_embeddings, dna_embeddings])

    # Save embeddings
    np.savez_compressed(
        os.path.join(args.output_dir, "gfp_embeddings.npz"),
        protein=prot_embeddings, dna=dna_embeddings,
        brightness=y, has_aa_mutation=df["has_aa_mutation"].values,
    )

    # Evaluate
    logger.info("\n=== GFP Fluorescence Prediction ===")
    results = []
    for name, X in [
        ("Protein-only (ESM-2)", prot_embeddings),
        ("DNA-only (NT)", dna_embeddings),
        ("Concat (Prot+DNA)", concat),
    ]:
        rho_mean, rho_std = evaluate_fitness(X, y, name, args.device)
        results.append({"name": name, "dim": X.shape[1], "spearman": rho_mean, "std": rho_std})

    # Synonymous-only subset (where protein is identical to WT)
    syn_mask = ~df["has_aa_mutation"].values
    if syn_mask.sum() > 50:
        logger.info(f"\n=== Synonymous variants only ({syn_mask.sum()} variants) ===")
        for name, X in [
            ("DNA-only (NT)", dna_embeddings),
            ("Concat (Prot+DNA)", concat),
        ]:
            X_syn = X[syn_mask]
            y_syn = y[syn_mask]
            rho, _ = spearmanr(
                StandardScaler().fit_transform(X_syn).mean(axis=1), y_syn
            )
            logger.info(f"  {name}: {syn_mask.sum()} variants (protein-only cannot distinguish these)")

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(args.output_dir, "gfp_fitness_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("GFP Fluorescence Prediction (Spearman ρ)")
    print("=" * 70)
    for _, row in df_results.sort_values("spearman", ascending=False).iterrows():
        print(f"  {row['name']:30s} ({int(row['dim']):4d}d): ρ = {row['spearman']:.4f} ± {row['std']:.4f}")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
