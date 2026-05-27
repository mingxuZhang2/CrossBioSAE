"""
Data preparation for CrossBioSAE.

Handles:
1. Downloading matched protein/coding-DNA pairs from UniProt + RefSeq
2. Extracting activations from ESM-2/ESM-C and Evo-2
3. Storing activations efficiently (HDF5)
4. Dataset classes for training
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Gene pair download and management
# =============================================================================

class GenePairDataset:
    """
    Manages matched (protein_sequence, coding_DNA_sequence) pairs.

    Data source: UniProt + NCBI RefSeq via REST API.
    For pilot: 1000 human protein-coding genes.
    For full: ~20k human genes, expandable to ~1M across species.
    """

    def __init__(self, data_dir: str, split: str = "pilot"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.pairs_file = self.data_dir / f"gene_pairs_{split}.json"
        self.fasta_dir = self.data_dir / "fasta"
        self.fasta_dir.mkdir(parents=True, exist_ok=True)

    def download_gene_pairs(
        self,
        organism: str = "Homo sapiens",
        max_genes: Optional[int] = None,
        max_protein_length: int = 1000,
        max_cds_length: int = 3000,
    ) -> list[dict]:
        """
        Download matched protein/CDS pairs from UniProt.

        Uses UniProt REST API to get reviewed human protein-coding genes
        with both protein sequence and cross-referenced EMBL CDS.

        Args:
            organism: Organism name
            max_genes: Maximum number of genes (None for all)
            max_protein_length: Filter out proteins longer than this
            max_cds_length: Filter out CDS longer than this

        Returns:
            List of dicts with keys: gene_name, uniprot_id, protein_seq, cds_seq, chromosome
        """
        import requests

        logger.info(f"Downloading gene pairs for {organism}...")

        # UniProt REST API query for reviewed human proteins
        # We request: gene name, protein sequence, cross-references to RefSeq
        base_url = "https://rest.uniprot.org/uniprotkb/stream"
        query = (
            f"organism_name:\"{organism}\" "
            f"AND reviewed:true "
            f"AND database:refseq "
            f"AND length:[50 TO {max_protein_length}]"
        )
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,gene_names,sequence,xref_refseq,organism_name,protein_existence",
            "size": max_genes if max_genes else 500,
        }

        logger.info(f"UniProt query: {query}")
        logger.info(f"Requesting up to {params['size']} entries...")

        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                f"UniProt API request failed: {e}. "
                "Use --synthetic flag explicitly for development data."
            )

        pairs = []
        results = data.get("results", [])
        logger.info(f"Got {len(results)} entries from UniProt")

        for entry in results:
            try:
                accession = entry.get("primaryAccession", "")
                gene_names = entry.get("genes", [{}])
                gene_name = ""
                if gene_names:
                    gene_name = gene_names[0].get("geneName", {}).get("value", accession)

                protein_seq = entry.get("sequence", {}).get("value", "")
                if not protein_seq or len(protein_seq) > max_protein_length:
                    continue

                # Get RefSeq cross-references for CDS
                xrefs = entry.get("uniProtKBCrossReferences", [])
                refseq_ids = [
                    x.get("id", "") for x in xrefs
                    if x.get("database") == "RefSeq"
                ]

                if not refseq_ids:
                    continue

                pairs.append({
                    "gene_name": gene_name,
                    "uniprot_id": accession,
                    "protein_seq": protein_seq,
                    "refseq_ids": refseq_ids,
                    "protein_length": len(protein_seq),
                    # CDS will be fetched separately or generated from protein
                    "cds_seq": self._protein_to_cds_placeholder(protein_seq),
                })
            except Exception as e:
                logger.warning(f"Skipping entry: {e}")
                continue

        if max_genes and len(pairs) > max_genes:
            pairs = pairs[:max_genes]

        logger.info(f"Prepared {len(pairs)} gene pairs")

        # Save
        with open(self.pairs_file, "w") as f:
            json.dump(pairs, f, indent=2)

        return pairs

    def _protein_to_cds_placeholder(self, protein_seq: str) -> str:
        """
        Generate a placeholder CDS from protein sequence using standard codon table.
        This is a simplification for the pilot. For the full run, we should fetch
        actual CDS from RefSeq/EMBL.

        NOTE: This uses the most common human codon for each amino acid.
        The actual CDS would have organism-specific codon usage.
        """
        # Most common human codons (from Kazusa codon usage database)
        codon_table = {
            "A": "GCC", "R": "CGG", "N": "AAC", "D": "GAC", "C": "TGC",
            "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC",
            "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
            "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
            "*": "TGA", "X": "NNN",
        }
        cds = "".join(codon_table.get(aa, "NNN") for aa in protein_seq)
        return cds

    def _generate_synthetic_pairs(
        self,
        n_genes: int,
        max_protein_length: int,
        max_cds_length: int,
    ) -> list[dict]:
        """Generate synthetic gene pairs for testing when API is unavailable."""
        import random

        logger.info(f"Generating {n_genes} synthetic gene pairs for development")
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        pairs = []

        for i in range(n_genes):
            prot_len = random.randint(50, min(500, max_protein_length))
            protein_seq = "".join(random.choice(amino_acids) for _ in range(prot_len))
            cds_seq = self._protein_to_cds_placeholder(protein_seq)

            pairs.append({
                "gene_name": f"GENE_{i:05d}",
                "uniprot_id": f"SYNTH_{i:05d}",
                "protein_seq": protein_seq,
                "refseq_ids": [f"NM_{i:06d}.1"],
                "protein_length": prot_len,
                "cds_seq": cds_seq,
            })

        with open(self.pairs_file, "w") as f:
            json.dump(pairs, f, indent=2)

        return pairs

    def load_pairs(self) -> list[dict]:
        """Load previously downloaded gene pairs."""
        if not self.pairs_file.exists():
            raise FileNotFoundError(
                f"Gene pairs file not found: {self.pairs_file}. "
                f"Run download_gene_pairs() first."
            )
        with open(self.pairs_file) as f:
            return json.load(f)

    def get_protein_sequences(self) -> list[tuple[str, str]]:
        """Return list of (gene_name, protein_sequence) tuples."""
        pairs = self.load_pairs()
        return [(p["gene_name"], p["protein_seq"]) for p in pairs]

    def get_dna_sequences(self) -> list[tuple[str, str]]:
        """Return list of (gene_name, cds_sequence) tuples."""
        pairs = self.load_pairs()
        return [(p["gene_name"], p["cds_seq"]) for p in pairs]


# =============================================================================
# Activation extraction
# =============================================================================

class ActivationExtractor:
    """
    Extract activations from pre-trained biological language models.

    Supports:
    - ESM-2 (fair-esm): protein language model
    - Evo-2 (arc-institute): DNA language model
    """

    def __init__(
        self,
        model_name: str,
        layer: Optional[int] = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 1024,
    ):
        """
        Args:
            model_name: Model identifier. Options:
                - "esm2_t33_650M_UR50D" (ESM-2 650M, pilot default)
                - "esm2_t36_3B_UR50D" (ESM-2 3B)
                - "esmc_600m" (ESM-C 600M, full run)
                - "evo2_7b" (Evo-2 7B)
                - "evo2_1b" (Evo-2 1B, pilot fallback)
            layer: Which layer to extract from. None = middle layer.
            device: Device to use.
            batch_size: Batch size for extraction.
            max_length: Maximum sequence length.
        """
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the pre-trained model."""
        if "esm2" in self.model_name:
            self._load_esm2()
        elif "esmc" in self.model_name:
            self._load_esmc()
        elif "evo" in self.model_name:
            self._load_evo()
        elif "nucleotide-transformer" in self.model_name or "nt_" in self.model_name:
            self._load_nt()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_esm2(self):
        """Load ESM-2 model."""
        import esm

        model_name = self.model_name
        # Map short names to full names
        name_map = {
            "esm2_t33_650M_UR50D": "esm2_t33_650M_UR50D",
            "esm2_650m": "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D": "esm2_t36_3B_UR50D",
            "esm2_3b": "esm2_t36_3B_UR50D",
        }
        model_name = name_map.get(model_name, model_name)

        logger.info(f"Loading ESM-2 model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Set default layer to middle
        n_layers = self.model.num_layers
        if self.layer is None:
            self.layer = n_layers // 2
        logger.info(f"ESM-2: {n_layers} layers, extracting from layer {self.layer}")
        logger.info(f"ESM-2 hidden dim: {self.model.embed_dim}")

    def _load_esmc(self):
        """Load ESM-C model (via HuggingFace transformers)."""
        from transformers import AutoModel, AutoTokenizer

        model_id = "facebook/esmc-600m"
        logger.info(f"Loading ESM-C model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        ).eval().to(self.device)

        # Set default layer
        if self.layer is None:
            self.layer = 24  # middle of 48 layers
        logger.info(f"ESM-C loaded, extracting from layer {self.layer}")

    def _load_evo(self):
        """Load Evo-2 model."""
        from transformers import AutoModel, AutoTokenizer

        model_map = {
            "evo2_7b": "arcinstitute/evo-2-7b",
            "evo2_1b": "arcinstitute/evo-2-1b",
            "evo_7b": "arcinstitute/evo-2-7b",
        }
        model_id = model_map.get(self.model_name, self.model_name)

        logger.info(f"Loading Evo-2 model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval().to(self.device)

        if self.layer is None:
            self.layer = 16
        logger.info(f"Evo-2 loaded, extracting from layer {self.layer}")

    def _load_nt(self):
        """Load Nucleotide Transformer (DNA model, smaller alternative to Evo-2)."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_map = {
            "nt_500m": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            "nt_250m": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            "nt_100m": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        }
        model_id = model_map.get(self.model_name, self.model_name)

        local_path = os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"),
            f"models--{model_id.replace('/', '--')}/snapshots/main",
        )
        load_from = local_path if os.path.isdir(local_path) else model_id

        logger.info(f"Loading Nucleotide Transformer: {load_from}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_from, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            load_from, trust_remote_code=True
        ).eval().to(self.device)

        if self.layer is None:
            n_layers = self.model.config.num_hidden_layers
            self.layer = n_layers // 2
        logger.info(
            f"NT loaded: {self.model.config.hidden_size} dim, "
            f"{self.model.config.num_hidden_layers} layers, extracting layer {self.layer}"
        )

    @torch.no_grad()
    def extract_esm2(
        self,
        sequences: list[tuple[str, str]],
        output_file: str,
    ) -> np.ndarray:
        """
        Extract ESM-2 activations for a list of (name, sequence) pairs.

        Returns:
            Array of shape (n_sequences, hidden_dim) with per-sequence mean pooled activations.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_activations = []
        gene_names = []

        for batch_start in tqdm(
            range(0, len(sequences), self.batch_size),
            desc=f"Extracting ESM-2 layer {self.layer}",
        ):
            batch = sequences[batch_start : batch_start + self.batch_size]

            # Truncate sequences
            batch_truncated = [
                (name, seq[:self.max_length]) for name, seq in batch
            ]

            labels, strs, tokens = self.batch_converter(batch_truncated)
            tokens = tokens.to(self.device)

            results = self.model(
                tokens,
                repr_layers=[self.layer],
                return_contacts=False,
            )

            # Get representations at the specified layer
            representations = results["representations"][self.layer]
            # representations shape: (batch, seq_len, hidden_dim)

            # Mean pool over sequence (exclude BOS/EOS tokens)
            for i, (name, seq) in enumerate(batch_truncated):
                seq_len = min(len(seq), self.max_length)
                # ESM-2 adds BOS token at position 0
                act = representations[i, 1 : seq_len + 1, :].mean(dim=0)
                all_activations.append(act.cpu().numpy())
                gene_names.append(name)

        activations = np.stack(all_activations, axis=0)
        logger.info(f"Extracted activations shape: {activations.shape}")

        # Save to HDF5
        with h5py.File(output_file, "w") as f:
            f.create_dataset("activations", data=activations, compression="gzip")
            f.create_dataset(
                "gene_names",
                data=np.array(gene_names, dtype=h5py.string_dtype()),
            )
            f.attrs["model"] = self.model_name
            f.attrs["layer"] = self.layer
            f.attrs["pooling"] = "mean"
            f.attrs["n_sequences"] = len(gene_names)

        logger.info(f"Saved activations to {output_file}")
        return activations

    @torch.no_grad()
    def extract_evo(
        self,
        sequences: list[tuple[str, str]],
        output_file: str,
    ) -> np.ndarray:
        """
        Extract Evo-2 activations for a list of (name, dna_sequence) pairs.

        Returns:
            Array of shape (n_sequences, hidden_dim) with per-sequence mean pooled activations.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_activations = []
        gene_names = []

        for batch_start in tqdm(
            range(0, len(sequences), self.batch_size),
            desc=f"Extracting Evo-2 layer {self.layer}",
        ):
            batch = sequences[batch_start : batch_start + self.batch_size]

            for name, seq in batch:
                # Truncate
                seq = seq[:self.max_length]

                inputs = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

                # Get hidden state at specified layer
                hidden_states = outputs.hidden_states
                layer_output = hidden_states[self.layer]
                # Mean pool over sequence length
                act = layer_output[0].mean(dim=0).float().cpu().numpy()
                all_activations.append(act)
                gene_names.append(name)

        activations = np.stack(all_activations, axis=0)
        logger.info(f"Extracted activations shape: {activations.shape}")

        # Save to HDF5
        with h5py.File(output_file, "w") as f:
            f.create_dataset("activations", data=activations, compression="gzip")
            f.create_dataset(
                "gene_names",
                data=np.array(gene_names, dtype=h5py.string_dtype()),
            )
            f.attrs["model"] = self.model_name
            f.attrs["layer"] = self.layer
            f.attrs["pooling"] = "mean"
            f.attrs["n_sequences"] = len(gene_names)

        logger.info(f"Saved activations to {output_file}")
        return activations

    def extract(
        self,
        sequences: list[tuple[str, str]],
        output_file: str,
    ) -> np.ndarray:
        """Extract activations using the appropriate method for the loaded model."""
        if self.model is None:
            self.load_model()

        if "esm2" in self.model_name:
            return self.extract_esm2(sequences, output_file)
        elif "esmc" in self.model_name:
            return self.extract_esmc(sequences, output_file)
        elif "evo" in self.model_name:
            return self.extract_evo(sequences, output_file)
        elif "nucleotide-transformer" in self.model_name or "nt_" in self.model_name:
            return self.extract_evo(sequences, output_file)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    @torch.no_grad()
    def extract_esmc(
        self,
        sequences: list[tuple[str, str]],
        output_file: str,
    ) -> np.ndarray:
        """Extract ESM-C activations."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_activations = []
        gene_names = []

        for batch_start in tqdm(
            range(0, len(sequences), self.batch_size),
            desc=f"Extracting ESM-C layer {self.layer}",
        ):
            batch = sequences[batch_start : batch_start + self.batch_size]

            for name, seq in batch:
                seq = seq[:self.max_length]
                inputs = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states
                layer_output = hidden_states[self.layer]
                act = layer_output[0].mean(dim=0).float().cpu().numpy()
                all_activations.append(act)
                gene_names.append(name)

        activations = np.stack(all_activations, axis=0)
        logger.info(f"Extracted activations shape: {activations.shape}")

        with h5py.File(output_file, "w") as f:
            f.create_dataset("activations", data=activations, compression="gzip")
            f.create_dataset(
                "gene_names",
                data=np.array(gene_names, dtype=h5py.string_dtype()),
            )
            f.attrs["model"] = self.model_name
            f.attrs["layer"] = self.layer
            f.attrs["pooling"] = "mean"
            f.attrs["n_sequences"] = len(gene_names)

        logger.info(f"Saved activations to {output_file}")
        return activations


# =============================================================================
# Training dataset
# =============================================================================

class PairedActivationDataset(Dataset):
    """
    Dataset of paired (DNA, protein) activations for CrossBioSAE training.

    Reads from HDF5 files created by ActivationExtractor.
    Ensures gene names match between DNA and protein files.
    """

    def __init__(
        self,
        dna_h5_path: str,
        protein_h5_path: str,
        normalize: bool = False,
    ):
        """
        Args:
            dna_h5_path: Path to HDF5 file with DNA activations
            protein_h5_path: Path to HDF5 file with protein activations
            normalize: Whether to z-score normalize activations
        """
        self.dna_h5_path = dna_h5_path
        self.protein_h5_path = protein_h5_path

        # Load and match gene names
        with h5py.File(dna_h5_path, "r") as f:
            dna_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
            self.dna_acts = f["activations"][:]
            self.dim_dna = self.dna_acts.shape[1]

        with h5py.File(protein_h5_path, "r") as f:
            prot_names = [n.decode() if isinstance(n, bytes) else n for n in f["gene_names"][:]]
            self.prot_acts = f["activations"][:]
            self.dim_protein = self.prot_acts.shape[1]

        # Find matching genes
        dna_name_to_idx = {name: i for i, name in enumerate(dna_names)}
        prot_name_to_idx = {name: i for i, name in enumerate(prot_names)}

        common_names = sorted(set(dna_names) & set(prot_names))
        if len(common_names) == 0:
            raise ValueError(
                f"No common gene names between DNA ({len(dna_names)}) "
                f"and protein ({len(prot_names)}) activation files"
            )

        logger.info(
            f"Found {len(common_names)} matched gene pairs "
            f"(DNA: {len(dna_names)}, Protein: {len(prot_names)})"
        )

        self.dna_indices = [dna_name_to_idx[n] for n in common_names]
        self.prot_indices = [prot_name_to_idx[n] for n in common_names]
        self.gene_names = common_names

        # Optional normalization
        if normalize:
            self.dna_mean = self.dna_acts[self.dna_indices].mean(axis=0)
            self.dna_std = self.dna_acts[self.dna_indices].std(axis=0) + 1e-8
            self.prot_mean = self.prot_acts[self.prot_indices].mean(axis=0)
            self.prot_std = self.prot_acts[self.prot_indices].std(axis=0) + 1e-8
        else:
            self.dna_mean = np.zeros(self.dim_dna)
            self.dna_std = np.ones(self.dim_dna)
            self.prot_mean = np.zeros(self.dim_protein)
            self.prot_std = np.ones(self.dim_protein)

    def __len__(self) -> int:
        return len(self.gene_names)

    def __getitem__(self, idx: int) -> dict:
        dna_act = (self.dna_acts[self.dna_indices[idx]] - self.dna_mean) / self.dna_std
        prot_act = (self.prot_acts[self.prot_indices[idx]] - self.prot_mean) / self.prot_std

        return {
            "dna_acts": torch.tensor(dna_act, dtype=torch.float32),
            "protein_acts": torch.tensor(prot_act, dtype=torch.float32),
            "gene_name": self.gene_names[idx],
        }


class SingleModalityDataset(Dataset):
    """Dataset for single-modality SAE training (DNA-only or protein-only)."""

    def __init__(self, h5_path: str, modality: str):
        with h5py.File(h5_path, "r") as f:
            self.acts = f["activations"][:]
            self.gene_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f["gene_names"][:]
            ]
        self.modality = modality
        self.dim = self.acts.shape[1]

    def __len__(self) -> int:
        return len(self.gene_names)

    def __getitem__(self, idx: int) -> dict:
        return {
            f"{self.modality}_acts": torch.tensor(self.acts[idx], dtype=torch.float32),
            "gene_name": self.gene_names[idx],
        }


def create_dataloaders(
    dna_h5_path: str,
    protein_h5_path: str,
    batch_size: int = 256,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from paired activation files."""
    dataset = PairedActivationDataset(dna_h5_path, protein_h5_path)

    # Split
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_single_modality_dataloaders(
    h5_path: str,
    modality: str,
    batch_size: int = 256,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for single-modality SAE training.

    Uses the existing SingleModalityDataset class.

    Args:
        h5_path: Path to HDF5 activation file (protein or DNA).
        modality: "protein" or "dna".
        batch_size: Training batch size.
        train_ratio: Fraction of data for training.
        num_workers: DataLoader workers.
        seed: Random seed for reproducible split.

    Returns:
        (train_loader, val_loader)
    """
    dataset = SingleModalityDataset(h5_path, modality)
    logger.info(f"Single-modality dataset [{modality}]: {len(dataset)} samples, dim={dataset.dim}")

    # Train/val split
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    return train_loader, val_loader
