"""
Fetch CDS (coding DNA sequences) for all ProteinGym proteins from UniProt/EMBL.

Strategy:
1. Parse UniProt accession from DMS_id
2. Query UniProt API for cross-references to EMBL/ENA CDS
3. Download the CDS nucleotide sequence
4. Verify: translate CDS matches the protein sequence
5. Save as JSON: {dms_id: cds_seq}
"""

import argparse, json, os, re, sys, time
import requests
import pandas as pd

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L',
    'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S',
    'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
    'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate(cds):
    protein = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i+3].upper()
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def fetch_cds_from_uniprot(accession, prot_seq, session):
    """Fetch CDS for a UniProt accession. Returns CDS string or None."""
    # Step 1: Get cross-references from UniProt
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
    except:
        return None

    # Step 2: Find EMBL/ENA CDS cross-references
    xrefs = data.get("uniProtKBCrossReferences", [])
    embl_ids = []
    for xref in xrefs:
        if xref.get("database") == "EMBL":
            for prop in xref.get("properties", []):
                if prop.get("key") == "ProteinId" and prop.get("value", "-") != "-":
                    embl_ids.append(prop["value"])

    # Also check RefSeq
    for xref in xrefs:
        if xref.get("database") == "RefSeq":
            refseq_id = xref.get("id", "")
            if refseq_id.startswith("NP_") or refseq_id.startswith("XP_"):
                for prop in xref.get("properties", []):
                    if prop.get("key") == "NucleotideSequenceId":
                        embl_ids.append(prop["value"])

    if not embl_ids:
        return None

    # Step 3: Try each CDS ID
    for cds_id in embl_ids[:5]:
        cds_id_clean = cds_id.split(".")[0]
        # Try ENA
        ena_url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{cds_id_clean}?download=true"
        try:
            r2 = session.get(ena_url, timeout=30)
            if r2.status_code == 200 and len(r2.text) > 50:
                lines = r2.text.strip().split("\n")
                seq = "".join(l.strip() for l in lines if not l.startswith(">"))
                seq = seq.upper().replace("U", "T")
                if len(seq) >= len(prot_seq) * 3:
                    translated = translate(seq)
                    if translated[:20] == prot_seq[:20]:
                        return seq
        except:
            pass

    # Step 4: Try NCBI for RefSeq
    for cds_id in embl_ids[:3]:
        if cds_id.startswith("NM_") or cds_id.startswith("XM_"):
            ncbi_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={cds_id}&rettype=fasta_cds_na&retmode=text"
            try:
                r3 = session.get(ncbi_url, timeout=30)
                if r3.status_code == 200:
                    lines = r3.text.strip().split("\n")
                    seq = "".join(l.strip() for l in lines if not l.startswith(">"))
                    seq = seq.upper()
                    if len(seq) >= len(prot_seq) * 3:
                        translated = translate(seq)
                        if translated[:20] == prot_seq[:20]:
                            return seq
            except:
                pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--existing_json", default="data/full/gene_pairs_full.json")
    ap.add_argument("--out_json", default="data/proteingym/proteingym_cds.json")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref_csv)
    print(f"Total assays: {len(ref)}, unique proteins: {ref['UniProt_ID'].nunique()}")

    # Load existing CDS from gene_pairs
    existing_cds = {}
    if os.path.exists(args.existing_json):
        with open(args.existing_json) as f:
            gp = json.load(f)
        for g in gp:
            if "cds_seq" in g and "gene_name" in g:
                existing_cds[g["gene_name"].upper()] = g["cds_seq"]
        print(f"Existing CDS from gene_pairs: {len(existing_cds)}")

    # Load existing output if resuming
    results = {}
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already fetched")

    session = requests.Session()
    session.headers["User-Agent"] = "CrossBioSAE/1.0 (mingxuz041@gmail.com)"

    done = len(results)
    failed = 0
    for _, row in ref.iterrows():
        dms_id = row["DMS_id"]
        uniprot_full = str(row["UniProt_ID"])
        accession = uniprot_full  # use full entry name (e.g. BRCA1_HUMAN)

        if dms_id in results:
            continue

        # Check existing gene_pairs CDS
        gene = dms_id.split("_")[0].upper()
        if gene in existing_cds:
            # Verify it matches
            dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
            if os.path.exists(dms_path):
                dms = pd.read_csv(dms_path, nrows=1)
                prot_seq = dms["target_seq"].iloc[0] if "target_seq" in dms.columns else ""
                cds = existing_cds[gene]
                if prot_seq and translate(cds)[:20] == prot_seq[:20]:
                    results[dms_id] = cds
                    done += 1
                    print(f"[{done}] {dms_id}: from existing gene_pairs ({len(cds)} bp) ✓")
                    continue

        # Fetch from UniProt/ENA
        dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            continue
        dms = pd.read_csv(dms_path, nrows=1)
        prot_seq = dms["target_seq"].iloc[0] if "target_seq" in dms.columns else ""
        if not prot_seq:
            continue

        cds = fetch_cds_from_uniprot(accession, prot_seq, session)
        if cds:
            results[dms_id] = cds
            done += 1
            print(f"[{done}] {dms_id}: fetched from UniProt ({len(cds)} bp) ✓")
        else:
            failed += 1
            print(f"[{done}] {dms_id}: FAILED (acc={accession})")

        # Save periodically
        if done % 10 == 0:
            with open(args.out_json, "w") as f:
                json.dump(results, f)

        time.sleep(0.5)

    # Final save
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)}/{len(ref)} assays with CDS, {failed} failed")


if __name__ == "__main__":
    main()
