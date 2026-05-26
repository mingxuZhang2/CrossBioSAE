#!/usr/bin/env python3
"""
Download GO annotations for human genes using UniProt ID mapping API.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_go_for_single_protein(uniprot_id: str) -> list[str]:
    """Fetch GO terms for a single UniProt entry."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    params = {"format": "json", "fields": "go_id"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        go_terms = []
        for xref in data.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                go_id = xref.get("id", "")
                if go_id:
                    go_terms.append(go_id)
        return go_terms
    except Exception:
        return []


def fetch_go_batch(uniprot_ids: list[str], batch_size: int = 50) -> dict[str, list[str]]:
    """Fetch GO annotations in small batches to avoid URL length limits."""
    results = {}

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]
        query = " OR ".join(f"accession:{uid}" for uid in batch)

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,go_id",
            "size": batch_size,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Batch {i // batch_size} failed: HTTP {resp.status_code}")
                # Fallback to individual fetching
                for uid in batch:
                    go = fetch_go_for_single_protein(uid)
                    if go:
                        results[uid] = go
                    time.sleep(0.2)
                continue

            data = resp.json()
            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                go_terms = []
                for xref in entry.get("uniProtKBCrossReferences", []):
                    if xref.get("database") == "GO":
                        go_id = xref.get("id", "")
                        if go_id:
                            go_terms.append(go_id)
                if go_terms:
                    results[acc] = go_terms

        except Exception as e:
            logger.warning(f"Batch {i // batch_size} failed: {e}")

        if (i // batch_size) % 20 == 0:
            logger.info(f"Progress: {len(results)} genes with GO ({i}/{len(uniprot_ids)})")
        time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    parser.add_argument("--output", default="data/full/go_annotations.json")
    args = parser.parse_args()

    with open(args.gene_pairs) as f:
        pairs = json.load(f)

    # Build UniProt ID -> gene name mapping
    uid_to_gene = {}
    for p in pairs:
        uid = p.get("uniprot_id", "")
        gene = p.get("gene_name", uid)
        if uid:
            uid_to_gene[uid] = gene

    uids = list(uid_to_gene.keys())
    logger.info(f"Fetching GO annotations for {len(uids)} genes...")

    uid_go = fetch_go_batch(uids, batch_size=50)

    # Convert to gene_name -> go_terms
    go_annotations = {}
    for uid, terms in uid_go.items():
        gene = uid_to_gene.get(uid, uid)
        go_annotations[gene] = terms

    logger.info(f"Total genes with GO: {len(go_annotations)}")
    n_terms = sum(len(v) for v in go_annotations.values())
    logger.info(f"Total GO assignments: {n_terms}, mean per gene: {n_terms / max(len(go_annotations), 1):.1f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(go_annotations, f)
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
