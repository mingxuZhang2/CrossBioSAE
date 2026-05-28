"""
Build matched (gene, protein, CDS-nucleotide, solubility) triples for the eSOL
cross-modal solubility benchmark.

Inputs:
  - eSol_train.csv / eSol_test.csv  (gene, solubility, protein sequence) from GraphSol
  - cds.fna  (E. coli K-12 MG1655 CDS, NCBI, headers carry [gene=...] [locus_tag=...])

Matching:
  1. by gene name (CDS header [gene=X]); verify translated CDS == eSOL protein.
  2. fallback: match by exact protein sequence across all CDS translations.
Reports coverage and writes esol_matched.csv.
"""

import argparse
import re

import pandas as pd
from Bio.Seq import Seq


def parse_cds(fna):
    """Return dict gene->cds_nt and dict protein_seq->cds_nt (translated, no stop)."""
    by_gene, by_prot = {}, {}
    gene, seq = None, []
    with open(fna) as f:
        for line in f:
            if line.startswith(">"):
                if gene is not None:
                    _store(gene, "".join(seq), by_gene, by_prot)
                m = re.search(r"\[gene=([^\]]+)\]", line)
                gene = m.group(1) if m else None
                seq = []
            else:
                seq.append(line.strip())
        if gene is not None:
            _store(gene, "".join(seq), by_gene, by_prot)
    return by_gene, by_prot


def _store(gene, nt, by_gene, by_prot):
    if len(nt) % 3 != 0 or len(nt) == 0:
        return
    try:
        prot = str(Seq(nt).translate()).rstrip("*")
    except Exception:
        return
    if "*" in prot:
        return
    # keep first occurrence per gene
    if gene and gene not in by_gene:
        by_gene[gene] = (nt, prot)
    by_prot.setdefault(prot, nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/solubility")
    args = ap.parse_args()

    train = pd.read_csv(f"{args.data_dir}/eSol_train.csv")
    test = pd.read_csv(f"{args.data_dir}/eSol_test.csv")
    train["split"] = "train"
    test["split"] = "test"
    esol = pd.concat([train, test], ignore_index=True)
    print(f"eSOL proteins: {len(esol)} ({(esol['split']=='train').sum()} train / "
          f"{(esol['split']=='test').sum()} test)")

    by_gene, by_prot = parse_cds(f"{args.data_dir}/cds.fna")
    print(f"K-12 CDS parsed: {len(by_gene)} genes, {len(by_prot)} unique proteins")

    rows = []
    n_gene_match, n_prot_match, n_gene_verified, n_unmatched = 0, 0, 0, 0
    for _, r in esol.iterrows():
        gene, sol, prot = r["gene"], r["solubility"], r["sequence"]
        cds_nt = None
        how = None
        if gene in by_gene:
            nt, cds_prot = by_gene[gene]
            n_gene_match += 1
            if cds_prot == prot:
                cds_nt, how = nt, "gene+verified"
                n_gene_verified += 1
            elif prot in by_prot:
                cds_nt, how = by_prot[prot], "protein-seq"
                n_prot_match += 1
            else:
                # gene matched but seq differs (isoform/annotation drift); keep CDS anyway
                cds_nt, how = nt, "gene-only"
        elif prot in by_prot:
            cds_nt, how = by_prot[prot], "protein-seq"
            n_prot_match += 1
        if cds_nt is None:
            n_unmatched += 1
            continue
        rows.append({"gene": gene, "solubility": sol, "protein": prot,
                     "cds": cds_nt, "split": r["split"], "match": how})

    out = pd.DataFrame(rows)
    print(f"\nMatched: {len(out)}/{len(esol)} ({100*len(out)/len(esol):.1f}%)")
    print(f"  gene name hit: {n_gene_match} (of which translation-verified: {n_gene_verified})")
    print(f"  matched via protein-seq fallback: {n_prot_match}")
    print(f"  unmatched (dropped): {n_unmatched}")
    print("\nmatch-type breakdown:")
    print(out["match"].value_counts().to_string())
    print(f"\nsolubility: mean={out['solubility'].mean():.3f} std={out['solubility'].std():.3f} "
          f"range=[{out['solubility'].min():.2f},{out['solubility'].max():.2f}]")
    print(f"protein len: median={out['protein'].str.len().median():.0f} "
          f"max={out['protein'].str.len().max()}")

    out_path = f"{args.data_dir}/esol_matched.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
