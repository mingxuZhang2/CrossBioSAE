"""
Fetch CDS for ProteinGym proteins. Strategy:
1. Use existing gene_pairs CDS
2. Query UniProt API → get RefSeq NM_ ID → NCBI efetch CDS
3. Verify translation matches DMS protein sequence
"""

import argparse, json, os, time
import requests
import pandas as pd

CODON = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
         'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
         'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
         'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
         'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
         'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
         'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
         'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}

def translate(cds):
    p = []
    for i in range(0, len(cds)-2, 3):
        aa = CODON.get(cds[i:i+3].upper(), 'X')
        if aa == '*': break
        p.append(aa)
    return ''.join(p)


def fetch_cds(uniprot_entry, prot_seq, session):
    """Fetch CDS via UniProt → NCBI RefSeq pipeline."""
    # Step 1: Get UniProt entry → RefSeq cross-refs
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_entry}.json"
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None, "uniprot_404"
    except:
        return None, "uniprot_timeout"

    data = r.json()
    xrefs = data.get("uniProtKBCrossReferences", [])

    # Collect all nucleotide IDs (RefSeq NM_, EMBL mRNA)
    nuc_ids = []
    for xref in xrefs:
        if xref.get("database") == "RefSeq":
            for prop in xref.get("properties", []):
                if prop.get("key") == "NucleotideSequenceId":
                    nid = prop["value"]
                    if nid.startswith("NM_"):
                        nuc_ids.insert(0, nid)  # prefer NM_
                    else:
                        nuc_ids.append(nid)
        elif xref.get("database") == "EMBL":
            mol_type = None
            for prop in xref.get("properties", []):
                if prop.get("key") == "MoleculeType":
                    mol_type = prop["value"]
            if mol_type == "mRNA":
                nuc_ids.append(xref["id"])

    if not nuc_ids:
        return None, "no_nuc_ids"

    # Step 2: Try NCBI efetch for each nucleotide ID
    for nid in nuc_ids[:8]:
        nid_clean = nid.split(".")[0]
        try:
            ncbi_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                       f"?db=nucleotide&id={nid_clean}&rettype=fasta_cds_na&retmode=text")
            r2 = session.get(ncbi_url, timeout=30)
            if r2.status_code != 200 or len(r2.text) < 50:
                continue

            # May have multiple CDS entries — try each
            entries = r2.text.strip().split(">")
            for entry in entries:
                if not entry.strip():
                    continue
                lines = entry.strip().split("\n")
                seq = "".join(l.strip() for l in lines[1:]).upper()
                if len(seq) < len(prot_seq) * 3 - 10:
                    continue
                t = translate(seq)
                if len(t) >= len(prot_seq) - 5 and t[:20] == prot_seq[:20]:
                    return seq, "ncbi_ok"
        except:
            continue
        time.sleep(0.3)

    # Step 3: Try ENA for EMBL IDs
    for xref in xrefs:
        if xref.get("database") != "EMBL":
            continue
        embl_id = xref["id"]
        for prop in xref.get("properties", []):
            if prop.get("key") == "ProteinId" and prop["value"] != "-":
                pid = prop["value"].split(".")[0]
                try:
                    ena_url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{pid}?download=true"
                    r3 = session.get(ena_url, timeout=30)
                    if r3.status_code == 200:
                        lines = r3.text.strip().split("\n")
                        seq = "".join(l.strip() for l in lines if not l.startswith(">")).upper()
                        t = translate(seq)
                        if len(t) >= len(prot_seq) - 5 and t[:20] == prot_seq[:20]:
                            return seq, "ena_ok"
                except:
                    pass
                try:
                    ena_url2 = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{embl_id}?download=true"
                    r4 = session.get(ena_url2, timeout=30)
                    if r4.status_code == 200 and len(r4.text) > 100:
                        lines = r4.text.strip().split("\n")
                        seq = "".join(l.strip() for l in lines if not l.startswith(">")).upper()
                        if len(seq) > len(prot_seq) * 3:
                            # Full genomic — need to find CDS within
                            for start in range(3):
                                t = translate(seq[start:])
                                idx = t.find(prot_seq[:20])
                                if idx >= 0:
                                    cds_start = start + idx * 3
                                    cds = seq[cds_start:cds_start + len(prot_seq)*3 + 3]
                                    t2 = translate(cds)
                                    if t2[:20] == prot_seq[:20]:
                                        return cds, "ena_genomic_ok"
                except:
                    pass

    return None, "all_failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--existing_json", default="data/full/gene_pairs_full.json")
    ap.add_argument("--out_json", default="data/proteingym/proteingym_cds.json")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref_csv)

    # Load existing CDS
    existing_cds = {}
    if os.path.exists(args.existing_json):
        with open(args.existing_json) as f:
            gp = json.load(f)
        for g in gp:
            if "cds_seq" in g and "gene_name" in g:
                existing_cds[g["gene_name"].upper()] = g["cds_seq"]

    results = {}
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            results = json.load(f)

    session = requests.Session()
    session.headers["User-Agent"] = "CrossBioSAE/1.0 (mingxuz041@gmail.com)"

    stats = {"existing": 0, "fetched": 0, "failed": 0, "skipped": 0}
    for _, row in ref.iterrows():
        dms_id = row["DMS_id"]
        uniprot_entry = str(row["UniProt_ID"])

        if dms_id in results:
            stats["skipped"] += 1
            continue

        dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            continue
        dms = pd.read_csv(dms_path, nrows=1)
        prot_seq = dms["target_seq"].iloc[0] if "target_seq" in dms.columns else ""
        if not prot_seq:
            continue

        # Try existing gene_pairs
        gene = dms_id.split("_")[0].upper()
        if gene in existing_cds:
            cds = existing_cds[gene]
            if translate(cds)[:20] == prot_seq[:20]:
                results[dms_id] = cds
                stats["existing"] += 1
                print(f"[{len(results)}] {dms_id}: gene_pairs ({len(cds)} bp) ✓", flush=True)
                continue

        # Fetch from UniProt/NCBI
        cds, method = fetch_cds(uniprot_entry, prot_seq, session)
        if cds:
            results[dms_id] = cds
            stats["fetched"] += 1
            print(f"[{len(results)}] {dms_id}: {method} ({len(cds)} bp) ✓", flush=True)
        else:
            stats["failed"] += 1
            print(f"[{len(results)}] {dms_id}: FAILED ({method})", flush=True)

        if len(results) % 10 == 0:
            with open(args.out_json, "w") as f:
                json.dump(results, f)
        time.sleep(0.5)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone: {len(results)}/217 CDS | {stats}")


if __name__ == "__main__":
    main()
