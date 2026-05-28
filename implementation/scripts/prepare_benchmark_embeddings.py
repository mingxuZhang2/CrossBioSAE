"""
Prepare gene embeddings for the Zhong 2025 Gene Embedding Benchmark.

Creates multiple embedding variants from our protein (ESM-2) and DNA (NT) activations:
1. ESM2-RAW: protein raw activations (baseline)
2. NT-RAW: DNA raw activations (baseline)
3. RAW-CONCAT: concat(protein, dna) raw activations
4. CCA-PROT: protein projected into CCA space
5. CCA-DNA: DNA projected into CCA space
6. CCA-FUSED: concat(cca_prot, cca_dna) - our main entry
7. CCA-SHARED: mean(cca_prot, cca_dna) - shared representation

Each variant is saved as a subfolder with:
  - embedding.csv: (n_genes, dim) matrix, no header
  - genes.txt: one Entrez Gene ID per line
"""

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_activations(h5_path: str) -> tuple[np.ndarray, list[str]]:
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"][:]
        names = [
            n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]
        ]
    logger.info(f"Loaded {h5_path}: {acts.shape}, {len(names)} genes")
    return acts, names


def build_symbol_to_entrez_map(map_file: str | None = None) -> dict[str, str]:
    """Build gene symbol -> Entrez ID mapping from HGNC or mygene."""
    if map_file and os.path.exists(map_file):
        logger.info(f"Loading pre-built mapping from {map_file}")
        mapping = {}
        with open(map_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        return mapping

    logger.info("Building symbol->Entrez mapping via mygene...")
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        return mg
    except ImportError:
        logger.warning("mygene not installed, trying manual HGNC download...")
        return _download_hgnc_mapping()


def _download_hgnc_mapping() -> dict[str, str]:
    """Download HGNC gene mapping as fallback."""
    import urllib.request

    url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
    cache = "/tmp/hgnc_complete_set.txt"
    if not os.path.exists(cache):
        logger.info(f"Downloading HGNC mapping...")
        urllib.request.urlretrieve(url, cache)

    mapping = {}
    df = pd.read_csv(cache, sep="\t", usecols=["symbol", "entrez_id"], dtype=str)
    for _, row in df.iterrows():
        if pd.notna(row["symbol"]) and pd.notna(row["entrez_id"]):
            mapping[row["symbol"]] = row["entrez_id"].split(".")[0]
    logger.info(f"HGNC mapping: {len(mapping)} symbols")
    return mapping


def map_genes_to_entrez(
    gene_symbols: list[str], mapping
) -> tuple[list[int], list[str]]:
    """Map gene symbols to Entrez IDs. Returns (indices_to_keep, entrez_ids)."""
    if isinstance(mapping, dict):
        indices = []
        entrez_ids = []
        for i, sym in enumerate(gene_symbols):
            if sym in mapping:
                indices.append(i)
                entrez_ids.append(mapping[sym])
        return indices, entrez_ids
    else:
        # mygene object
        mg = mapping
        results = mg.querymany(
            gene_symbols,
            scopes="symbol",
            fields="entrezgene",
            species="human",
            returnall=True,
        )
        sym_to_entrez = {}
        for r in results["out"]:
            if "entrezgene" in r and "query" in r:
                sym_to_entrez[r["query"]] = str(r["entrezgene"])

        indices = []
        entrez_ids = []
        for i, sym in enumerate(gene_symbols):
            if sym in sym_to_entrez:
                indices.append(i)
                entrez_ids.append(sym_to_entrez[sym])
        return indices, entrez_ids


def save_embedding(out_dir: str, embedding: np.ndarray, entrez_ids: list[str], name: str):
    """Save embedding in benchmark format: subfolder/embedding.csv + subfolder/genes.txt"""
    sub = os.path.join(out_dir, name)
    os.makedirs(sub, exist_ok=True)
    csv_path = os.path.join(sub, f"{name}.csv")
    txt_path = os.path.join(sub, f"{name}.txt")

    np.savetxt(csv_path, embedding, delimiter=",", fmt="%.8f")
    with open(txt_path, "w") as f:
        for eid in entrez_ids:
            f.write(f"{eid}\n")

    logger.info(f"Saved {name}: {embedding.shape} to {sub}")


def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings for Zhong 2025 benchmark")
    parser.add_argument("--protein_h5", required=True, help="Protein activations HDF5")
    parser.add_argument("--dna_h5", required=True, help="DNA activations HDF5")
    parser.add_argument("--output_dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--gene_map", default=None, help="Pre-built symbol->entrez TSV")
    parser.add_argument("--cca_components", type=int, default=64, help="CCA components")
    parser.add_argument("--pca_dim", type=int, default=512, help="PCA pre-reduction dim")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load activations
    prot_acts, prot_names = load_activations(args.protein_h5)
    dna_acts, dna_names = load_activations(args.dna_h5)

    # 2. Align by gene name
    prot_idx_map = {n: i for i, n in enumerate(prot_names)}
    dna_idx_map = {n: i for i, n in enumerate(dna_names)}
    common = sorted(set(prot_names) & set(dna_names))
    logger.info(f"Common genes: {len(common)} (protein: {len(prot_names)}, dna: {len(dna_names)})")

    prot_idx = [prot_idx_map[g] for g in common]
    dna_idx = [dna_idx_map[g] for g in common]
    prot_aligned = prot_acts[prot_idx]
    dna_aligned = dna_acts[dna_idx]

    # 3. Map to Entrez IDs
    mapping = build_symbol_to_entrez_map(args.gene_map)
    keep_indices, entrez_ids = map_genes_to_entrez(common, mapping)
    logger.info(f"Mapped to Entrez: {len(entrez_ids)} / {len(common)} genes")

    prot_final = prot_aligned[keep_indices]
    dna_final = dna_aligned[keep_indices]

    # Save mapping for reference
    map_path = os.path.join(args.output_dir, "gene_mapping.tsv")
    with open(map_path, "w") as f:
        f.write("symbol\tentrez_id\n")
        for idx, eid in zip(keep_indices, entrez_ids):
            f.write(f"{common[idx]}\t{eid}\n")

    # 4. Create embedding variants
    n_genes = len(entrez_ids)
    logger.info(f"Creating embeddings for {n_genes} genes...")

    # 4a. Raw single-modality
    save_embedding(args.output_dir, prot_final, entrez_ids, "ESM2-RAW")
    save_embedding(args.output_dir, dna_final, entrez_ids, "NT-RAW")

    # 4b. Raw concatenation
    raw_concat = np.hstack([prot_final, dna_final])
    save_embedding(args.output_dir, raw_concat, entrez_ids, "RAW-CONCAT")

    # 4c. PCA-reduced single modality (to match CCA dim for fair comparison)
    pca_dim = min(args.pca_dim, n_genes - 1, prot_final.shape[1], dna_final.shape[1])

    # 4d. CCA alignment
    logger.info(f"Fitting CCA: PCA to {pca_dim}, then CCA to {args.cca_components}...")

    # PCA pre-reduction
    pca_prot = PCA(n_components=pca_dim)
    pca_dna = PCA(n_components=pca_dim)
    prot_pca = pca_prot.fit_transform(prot_final)
    dna_pca = pca_dna.fit_transform(dna_final)
    logger.info(
        f"PCA explained variance: protein={pca_prot.explained_variance_ratio_.sum():.3f}, "
        f"dna={pca_dna.explained_variance_ratio_.sum():.3f}"
    )

    # Standardize
    scaler_prot = StandardScaler()
    scaler_dna = StandardScaler()
    prot_scaled = scaler_prot.fit_transform(prot_pca)
    dna_scaled = scaler_dna.fit_transform(dna_pca)

    # Fit CCA
    n_cca = min(args.cca_components, pca_dim, n_genes - 1)
    cca = CCA(n_components=n_cca, max_iter=2000)
    cca.fit(prot_scaled, dna_scaled)
    cca_prot, cca_dna = cca.transform(prot_scaled, dna_scaled)
    logger.info(f"CCA done: prot={cca_prot.shape}, dna={cca_dna.shape}")

    # Per-component correlation
    correlations = []
    for i in range(n_cca):
        r = np.corrcoef(cca_prot[:, i], cca_dna[:, i])[0, 1]
        correlations.append(r)
    logger.info(f"CCA canonical correlations: top5={np.mean(correlations[:5]):.3f}, "
                f"mean={np.mean(correlations):.3f}")

    # Save CCA variants
    save_embedding(args.output_dir, cca_prot, entrez_ids, "CCA-PROT")
    save_embedding(args.output_dir, cca_dna, entrez_ids, "CCA-DNA")

    cca_fused = np.hstack([cca_prot, cca_dna])
    save_embedding(args.output_dir, cca_fused, entrez_ids, "CCA-FUSED")

    cca_shared = (cca_prot + cca_dna) / 2.0
    save_embedding(args.output_dir, cca_shared, entrez_ids, "CCA-SHARED")

    # 4e. PCA-reduced raw (same dim as CCA for fair comparison)
    prot_pca_reduced = prot_pca[:, :n_cca]
    dna_pca_reduced = dna_pca[:, :n_cca]
    save_embedding(args.output_dir, prot_pca_reduced, entrez_ids, "ESM2-PCA")
    save_embedding(args.output_dir, dna_pca_reduced, entrez_ids, "NT-PCA")
    pca_concat = np.hstack([prot_pca_reduced, dna_pca_reduced])
    save_embedding(args.output_dir, pca_concat, entrez_ids, "PCA-CONCAT")

    # Summary
    print("\n=== Embedding Summary ===")
    for name in ["ESM2-RAW", "NT-RAW", "RAW-CONCAT", "ESM2-PCA", "NT-PCA",
                  "PCA-CONCAT", "CCA-PROT", "CCA-DNA", "CCA-FUSED", "CCA-SHARED"]:
        sub = os.path.join(args.output_dir, name)
        csv_files = [f for f in os.listdir(sub) if f.endswith(".csv")]
        arr = np.loadtxt(os.path.join(sub, csv_files[0]), delimiter=",")
        print(f"  {name:15s}: {arr.shape[0]} genes × {arr.shape[1]} dims")

    print(f"\nTotal genes with Entrez mapping: {n_genes}")
    print(f"CCA components: {n_cca}")
    print(f"CCA mean canonical correlation: {np.mean(correlations):.4f}")


if __name__ == "__main__":
    main()
