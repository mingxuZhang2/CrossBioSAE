#!/usr/bin/env python3
"""
Download real matched protein/CDS pairs from UniProt + NCBI.
Run on a machine WITH internet access. Transfer output to HPC3.

Usage:
    python scripts/download_real_data.py --n_genes 1000 --output data/pilot/gene_pairs_pilot.json
    python scripts/download_real_data.py --n_genes 20000 --output data/full/gene_pairs_full.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests
from Bio import Entrez, SeqIO
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

Entrez.email = "mingxuz041@gmail.com"


def fetch_uniprot_proteins(n_genes: int, max_length: int = 1000) -> list[dict]:
    """Fetch reviewed human proteins with RefSeq cross-references from UniProt."""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = (
        'organism_id:9606 '
        'AND reviewed:true '
        'AND database:refseq '
        f'AND length:[50 TO {max_length}]'
    )
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,gene_names,sequence,xref_refseq",
        "size": min(n_genes, 500),
    }

    all_results = []
    next_url = None

    logger.info(f"Fetching up to {n_genes} proteins from UniProt...")

    while len(all_results) < n_genes:
        if next_url:
            resp = requests.get(next_url, timeout=60)
        else:
            resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        logger.info(f"  Fetched {len(all_results)} / {n_genes}")

        link_header = resp.headers.get("Link", "")
        if 'rel="next"' in link_header:
            next_url = link_header.split(";")[0].strip("<>")
        else:
            break

    logger.info(f"Total UniProt entries: {len(all_results)}")
    return all_results[:n_genes]


def parse_uniprot_entries(entries: list[dict]) -> list[dict]:
    """Parse UniProt JSON entries into (gene_name, protein_seq, refseq_nm_ids)."""
    parsed = []
    for entry in entries:
        accession = entry.get("primaryAccession", "")
        genes = entry.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", accession) if genes else accession

        protein_seq = entry.get("sequence", {}).get("value", "")
        if not protein_seq:
            continue

        xrefs = entry.get("uniProtKBCrossReferences", [])
        nm_ids = []
        for x in xrefs:
            if x.get("database") == "RefSeq":
                rid = x.get("id", "")
                if rid.startswith("NP_"):
                    props = x.get("properties", [])
                    for p in props:
                        if p.get("key") == "NucleotideSequenceId":
                            nm_id = p.get("value", "")
                            if nm_id.startswith("NM_"):
                                nm_ids.append(nm_id)

        if not nm_ids:
            continue

        parsed.append({
            "gene_name": gene_name,
            "uniprot_id": accession,
            "protein_seq": protein_seq,
            "protein_length": len(protein_seq),
            "refseq_nm_id": nm_ids[0],
        })

    logger.info(f"Parsed {len(parsed)} entries with NM_ accessions")
    return parsed


def batch_fetch_cds(nm_ids: list[str], batch_size: int = 100) -> dict[str, str]:
    """Fetch CDS sequences from NCBI Nucleotide for a list of NM_ accessions."""
    nm_to_cds = {}

    for i in range(0, len(nm_ids), batch_size):
        batch = nm_ids[i:i + batch_size]
        ids_str = ",".join(batch)

        try:
            handle = Entrez.efetch(db="nucleotide", id=ids_str, rettype="fasta_cds_na", retmode="text")
            text = handle.read()
            handle.close()

            for record in SeqIO.parse(StringIO(text), "fasta"):
                seq = str(record.seq)
                desc = record.description
                for nm_id in batch:
                    base_id = nm_id.split(".")[0]
                    if base_id in desc or nm_id in desc:
                        nm_to_cds[nm_id] = seq
                        break

            logger.info(f"  NCBI batch {i // batch_size + 1}: got {len(nm_to_cds)} CDS total")
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"  NCBI batch failed: {e}, retrying...")
            time.sleep(2)
            try:
                for nm_id in batch:
                    try:
                        handle = Entrez.efetch(db="nucleotide", id=nm_id, rettype="fasta_cds_na", retmode="text")
                        text = handle.read()
                        handle.close()
                        for record in SeqIO.parse(StringIO(text), "fasta"):
                            nm_to_cds[nm_id] = str(record.seq)
                            break
                        time.sleep(0.4)
                    except Exception:
                        continue
            except Exception:
                pass

    logger.info(f"Fetched {len(nm_to_cds)} CDS sequences from NCBI")
    return nm_to_cds


def main():
    parser = argparse.ArgumentParser(description="Download real gene pairs")
    parser.add_argument("--n_genes", type=int, default=1000)
    parser.add_argument("--max_protein_length", type=int, default=1000)
    parser.add_argument("--max_cds_length", type=int, default=3000)
    parser.add_argument("--output", type=str, default="data/pilot/gene_pairs_pilot.json")
    args = parser.parse_args()

    # Step 1: Fetch from UniProt
    entries = fetch_uniprot_proteins(args.n_genes * 2, args.max_protein_length)
    parsed = parse_uniprot_entries(entries)

    # Step 2: Fetch CDS from NCBI
    nm_ids = [p["refseq_nm_id"] for p in parsed]
    nm_to_cds = batch_fetch_cds(nm_ids)

    # Step 3: Merge, verify translation consistency, and filter
    from Bio.Seq import Seq

    pairs = []
    n_mismatch = 0
    n_length_filtered = 0
    for p in parsed:
        nm_id = p["refseq_nm_id"]
        if nm_id not in nm_to_cds:
            continue
        cds = nm_to_cds[nm_id]
        if len(cds) > args.max_cds_length or len(cds) < 30:
            n_length_filtered += 1
            continue

        # Translation consistency check: translate(CDS) should match UniProt protein
        translated = str(Seq(cds).translate()).rstrip("*")
        protein = p["protein_seq"]
        if translated == protein:
            translation_match = "exact"
        elif translated.lstrip("M") == protein.lstrip("M"):
            translation_match = "methionine_processing"
        elif len(translated) == len(protein) and sum(a != b for a, b in zip(translated, protein)) <= 2:
            translation_match = "near_match"
        else:
            n_mismatch += 1
            continue

        p["cds_seq"] = cds
        p["cds_length"] = len(cds)
        p["translation_match"] = translation_match
        pairs.append(p)

        if len(pairs) >= args.n_genes:
            break

    logger.info(f"Final dataset: {len(pairs)} matched gene pairs")
    logger.info(f"Filtered: {n_mismatch} translation mismatches, {n_length_filtered} length-filtered")

    # Stats
    prot_lens = [p["protein_length"] for p in pairs]
    cds_lens = [p["cds_length"] for p in pairs]
    logger.info(f"Protein length: min={min(prot_lens)}, max={max(prot_lens)}, mean={sum(prot_lens)//len(prot_lens)}")
    logger.info(f"CDS length: min={min(cds_lens)}, max={max(cds_lens)}, mean={sum(cds_lens)//len(cds_lens)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
