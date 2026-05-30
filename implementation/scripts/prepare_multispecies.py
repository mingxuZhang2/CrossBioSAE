"""
Download protein+CDS gene pairs for multiple species and extract ESM-2
gene-level embeddings. Evo2 extraction runs separately on GPU (sharded).

Species chosen for: well-annotated proteomes, orthologous coverage with human.
Combined with human's 16,620 genes → ~60-80k gene pairs for pretraining.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import GenePairDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPECIES = [
    ("Mus musculus", "mouse", 20000),
    ("Danio rerio", "zebrafish", 20000),
    ("Rattus norvegicus", "rat", 18000),
]


def download_species(species_name, label, max_genes, data_dir):
    out_json = os.path.join(data_dir, f"gene_pairs_{label}.json")
    if os.path.exists(out_json):
        genes = json.load(open(out_json))
        logger.info(f"{label}: {len(genes)} genes already downloaded")
        return genes

    ds = GenePairDataset.__new__(GenePairDataset)
    ds.gene_pairs = []
    ds.pairs_file = os.path.join(data_dir, f"_tmp_{label}.json")
    raw = ds.download_gene_pairs(organism=species_name, max_genes=max_genes)
    genes = [g for g in raw if g.get("protein_seq") and g.get("cds_seq")
             and len(g["protein_seq"]) > 10 and len(g["cds_seq"]) > 30]
    with open(out_json, "w") as f:
        json.dump(genes, f)
    logger.info(f"{label}: downloaded {len(genes)} genes with both seqs")
    return genes


def extract_esm(genes, label, data_dir, device, batch_size):
    """Extract ESM-2 gene-level embeddings (mean-pool per-residue → 1280-d)."""
    import h5py
    import torch

    out_h5 = os.path.join(data_dir, f"protein_activations_{label}.h5")
    if os.path.exists(out_h5):
        import h5py as h
        with h.File(out_h5, "r") as f:
            logger.info(f"{label}: ESM embeddings already exist ({f['activations'].shape})")
        return

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings, names = [], []
    for i in range(0, len(genes), batch_size):
        batch_genes = genes[i:i + batch_size]
        data = [(g["gene_name"], g["protein_seq"][:1022]) for g in batch_genes]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
            reps = results["representations"][33]  # (B, L, 1280)
        for j, g in enumerate(batch_genes):
            seq_len = min(len(g["protein_seq"]), 1022)
            emb = reps[j, 1:seq_len + 1].mean(dim=0).cpu().numpy()
            embeddings.append(emb)
            names.append(g["gene_name"])
        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM {label}: {i + len(batch_genes)}/{len(genes)}")

    embeddings = np.stack(embeddings)
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("activations", data=embeddings.astype(np.float32))
        h.create_dataset("gene_names", data=np.array(names, dtype="S"))
    logger.info(f"saved {out_h5}: {embeddings.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/multispecies")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--esm_batch", type=int, default=8)
    ap.add_argument("--download_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    for species_name, label, max_genes in SPECIES:
        logger.info(f"\n{'='*60}\n{species_name} ({label})\n{'='*60}")
        genes = download_species(species_name, label, max_genes, args.data_dir)
        if not args.download_only:
            extract_esm(genes, label, args.data_dir, args.device, args.esm_batch)

    # summary
    total = 0
    for _, label, _ in SPECIES:
        f = os.path.join(args.data_dir, f"gene_pairs_{label}.json")
        if os.path.exists(f):
            n = len(json.load(open(f)))
            total += n
            logger.info(f"{label}: {n} genes")
    logger.info(f"total across new species: {total} (+ 16620 human = {total + 16620})")


if __name__ == "__main__":
    main()
