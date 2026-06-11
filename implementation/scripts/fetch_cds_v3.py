"""
Fetch CDS for remaining ProteinGym proteins that v2 missed.
More aggressive strategies:
1. UniProt ID mapping → accession → NCBI RefSeq
2. NCBI Gene search by gene name + organism
3. Ensembl REST API
4. Reverse translation (codon-optimized) as last resort
"""

import argparse, json, os, re, time
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

# Preferred codons per organism group (most frequent codon per AA)
CODON_PREFERENCE = {
    "human": {
        'F':'TTC','L':'CTG','I':'ATC','M':'ATG','V':'GTG',
        'S':'AGC','P':'CCC','T':'ACC','A':'GCC','Y':'TAC',
        'H':'CAC','Q':'CAG','N':'AAC','K':'AAG','D':'GAC',
        'E':'GAG','C':'TGC','W':'TGG','R':'CGG','G':'GGC',
        '*':'TGA'
    },
    "ecoli": {
        'F':'TTT','L':'CTG','I':'ATT','M':'ATG','V':'GTG',
        'S':'AGC','P':'CCG','T':'ACC','A':'GCG','Y':'TAT',
        'H':'CAT','Q':'CAG','N':'AAC','K':'AAA','D':'GAT',
        'E':'GAA','C':'TGC','W':'TGG','R':'CGT','G':'GGC',
        '*':'TAA'
    },
    "yeast": {
        'F':'TTC','L':'TTG','I':'ATC','M':'ATG','V':'GTT',
        'S':'TCT','P':'CCA','T':'ACT','A':'GCT','Y':'TAC',
        'H':'CAC','Q':'CAA','N':'AAC','K':'AAG','D':'GAT',
        'E':'GAA','C':'TGT','W':'TGG','R':'AGA','G':'GGT',
        '*':'TAA'
    },
    "default": {
        'F':'TTC','L':'CTG','I':'ATC','M':'ATG','V':'GTG',
        'S':'AGC','P':'CCC','T':'ACC','A':'GCC','Y':'TAC',
        'H':'CAC','Q':'CAG','N':'AAC','K':'AAG','D':'GAC',
        'E':'GAG','C':'TGC','W':'TGG','R':'CGG','G':'GGC',
        '*':'TGA'
    }
}

def translate(cds):
    p = []
    for i in range(0, len(cds)-2, 3):
        aa = CODON.get(cds[i:i+3].upper(), 'X')
        if aa == '*': break
        p.append(aa)
    return ''.join(p)


def reverse_translate(prot_seq, organism="default"):
    """Reverse translate protein to CDS using organism-specific codon preferences."""
    ct = CODON_PREFERENCE.get(organism, CODON_PREFERENCE["default"])
    codons = []
    for aa in prot_seq:
        if aa in ct:
            codons.append(ct[aa])
        else:
            codons.append("NNN")
    codons.append(ct.get("*", "TGA"))
    return "".join(codons)


def get_organism_group(uniprot_id, taxon):
    """Map UniProt ID / taxon to codon preference group."""
    uid = uniprot_id.upper()
    if "HUMAN" in uid or taxon == "Human":
        return "human"
    if "ECOLI" in uid:
        return "ecoli"
    if "YEAST" in uid:
        return "yeast"
    if taxon == "Prokaryote":
        return "ecoli"
    if "MOUSE" in uid or "CHICK" in uid:
        return "human"
    return "default"


def try_uniprot_accession(entry_name, prot_seq, session):
    """Try to get CDS via UniProt accession lookup."""
    # First get the actual accession from entry name
    url = f"https://rest.uniprot.org/uniprotkb/search?query={entry_name}&format=json&size=1"
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None, "search_failed"
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None, "no_results"

        entry = results[0]
        accession = entry.get("primaryAccession", "")
        if not accession:
            return None, "no_accession"

        # Get cross-references
        xrefs = entry.get("uniProtKBCrossReferences", [])
        nuc_ids = []
        for xref in xrefs:
            if xref.get("database") == "RefSeq":
                for prop in xref.get("properties", []):
                    if prop.get("key") == "NucleotideSequenceId":
                        nid = prop["value"]
                        if nid.startswith("NM_"):
                            nuc_ids.insert(0, nid)
                        elif nid.startswith("XM_"):
                            nuc_ids.append(nid)
            elif xref.get("database") == "EMBL":
                mol_type = None
                for prop in xref.get("properties", []):
                    if prop.get("key") == "MoleculeType":
                        mol_type = prop["value"]
                if mol_type == "mRNA":
                    nuc_ids.append(xref["id"])
                # Also try protein ID for CDS
                for prop in xref.get("properties", []):
                    if prop.get("key") == "ProteinId" and prop["value"] != "-":
                        nuc_ids.append(prop["value"])

        if not nuc_ids:
            return None, "no_nuc_ids"

        # Try NCBI efetch for each
        for nid in nuc_ids[:10]:
            nid_clean = nid.split(".")[0]
            try:
                # Try as CDS
                ncbi_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                           f"?db=nucleotide&id={nid_clean}&rettype=fasta_cds_na&retmode=text")
                r2 = session.get(ncbi_url, timeout=30)
                if r2.status_code == 200 and len(r2.text) > 50:
                    for entry_text in r2.text.strip().split(">"):
                        if not entry_text.strip():
                            continue
                        lines = entry_text.strip().split("\n")
                        seq = "".join(l.strip() for l in lines[1:]).upper()
                        if len(seq) < 30:
                            continue
                        t = translate(seq)
                        # Check if our target sequence is contained in the translation
                        if prot_seq[:20] in t or t[:20] == prot_seq[:20]:
                            return seq, "ncbi_search_ok"
                        # Also check if target is a domain within the full protein
                        idx = t.find(prot_seq[:15])
                        if idx >= 0:
                            cds_start = idx * 3
                            cds_end = cds_start + len(prot_seq) * 3 + 3
                            domain_cds = seq[cds_start:cds_end]
                            dt = translate(domain_cds)
                            if dt[:15] == prot_seq[:15]:
                                return domain_cds, "ncbi_domain_ok"
            except:
                pass
            time.sleep(0.3)

            # Also try ENA
            try:
                ena_url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{nid_clean}?download=true"
                r3 = session.get(ena_url, timeout=30)
                if r3.status_code == 200 and len(r3.text) > 50:
                    lines = r3.text.strip().split("\n")
                    seq = "".join(l.strip() for l in lines if not l.startswith(">")).upper()
                    t = translate(seq)
                    if prot_seq[:20] in t or t[:20] == prot_seq[:20]:
                        return seq, "ena_search_ok"
                    idx = t.find(prot_seq[:15])
                    if idx >= 0:
                        cds_start = idx * 3
                        domain_cds = seq[cds_start:cds_start + len(prot_seq)*3 + 3]
                        dt = translate(domain_cds)
                        if dt[:15] == prot_seq[:15]:
                            return domain_cds, "ena_domain_ok"
            except:
                pass

        return None, "nuc_ids_exhausted"
    except Exception as e:
        return None, f"error:{str(e)[:50]}"


def try_ncbi_gene_search(gene_name, organism, prot_seq, session):
    """Search NCBI Gene by name + organism, get RefSeq CDS."""
    org_map = {
        "Human": "Homo sapiens", "HUMAN": "Homo sapiens",
        "MOUSE": "Mus musculus", "ECOLI": "Escherichia coli",
        "YEAST": "Saccharomyces cerevisiae", "CHICK": "Gallus gallus",
        "ARATH": "Arabidopsis thaliana",
    }

    # Infer organism from UniProt ID suffix
    org_name = org_map.get(organism, "")
    if not org_name:
        for key, val in org_map.items():
            if key in gene_name.upper():
                org_name = val
                break

    if not org_name:
        return None, "unknown_organism"

    query = f"{gene_name}[Gene Name] AND {org_name}[Organism]"
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
           f"?db=gene&term={requests.utils.quote(query)}&retmode=json")
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None, "gene_search_failed"
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None, "no_gene_found"

        # Get linked nucleotide records
        for gid in ids[:3]:
            link_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                       f"?dbfrom=gene&db=nucleotide&id={gid}&linkname=gene_nuccore_refseqrna&retmode=json")
            r2 = session.get(link_url, timeout=30)
            if r2.status_code != 200:
                continue
            link_data = r2.json()
            linksets = link_data.get("linksets", [])
            for ls in linksets:
                for ldb in ls.get("linksetdbs", []):
                    nuc_links = ldb.get("links", [])
                    for nuc_id in nuc_links[:5]:
                        fetch_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                                    f"?db=nucleotide&id={nuc_id}&rettype=fasta_cds_na&retmode=text")
                        try:
                            r3 = session.get(fetch_url, timeout=30)
                            if r3.status_code == 200 and len(r3.text) > 50:
                                for entry_text in r3.text.strip().split(">"):
                                    if not entry_text.strip():
                                        continue
                                    lines = entry_text.strip().split("\n")
                                    seq = "".join(l.strip() for l in lines[1:]).upper()
                                    if len(seq) < 30:
                                        continue
                                    t = translate(seq)
                                    if t[:20] == prot_seq[:20]:
                                        return seq, "ncbi_gene_ok"
                                    idx = t.find(prot_seq[:15])
                                    if idx >= 0:
                                        cds_start = idx * 3
                                        domain_cds = seq[cds_start:cds_start+len(prot_seq)*3+3]
                                        dt = translate(domain_cds)
                                        if dt[:15] == prot_seq[:15]:
                                            return domain_cds, "ncbi_gene_domain_ok"
                        except:
                            pass
                        time.sleep(0.3)
        return None, "gene_no_match"
    except Exception as e:
        return None, f"gene_error:{str(e)[:50]}"


def try_ensembl(gene_name, organism, prot_seq, session):
    """Try Ensembl REST API."""
    org_map = {
        "HUMAN": "homo_sapiens", "MOUSE": "mus_musculus",
        "ECOLI": "escherichia_coli_str_k_12_substr_mg1655",
        "YEAST": "saccharomyces_cerevisiae",
        "CHICK": "gallus_gallus", "ARATH": "arabidopsis_thaliana",
    }

    species = ""
    for key, val in org_map.items():
        if key in gene_name.upper() or key in organism.upper():
            species = val
            break
    if not species:
        return None, "ensembl_unknown_species"

    # Search by gene name
    gene_short = gene_name.split("_")[0]
    url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene_short}?content-type=application/json"
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None, "ensembl_search_failed"
        hits = r.json()
        gene_ids = [h["id"] for h in hits if h.get("type") == "gene"]
        if not gene_ids:
            return None, "ensembl_no_gene"

        for gid in gene_ids[:2]:
            # Get transcripts
            tx_url = f"https://rest.ensembl.org/lookup/id/{gid}?expand=1&content-type=application/json"
            r2 = session.get(tx_url, timeout=30)
            if r2.status_code != 200:
                continue
            gene_data = r2.json()
            transcripts = gene_data.get("Transcript", [])
            for tx in transcripts:
                if tx.get("biotype") != "protein_coding":
                    continue
                tx_id = tx["id"]
                cds_url = f"https://rest.ensembl.org/sequence/id/{tx_id}?type=cds&content-type=text/plain"
                try:
                    r3 = session.get(cds_url, timeout=30)
                    if r3.status_code == 200:
                        seq = r3.text.strip().upper()
                        t = translate(seq)
                        if t[:20] == prot_seq[:20]:
                            return seq, "ensembl_ok"
                        idx = t.find(prot_seq[:15])
                        if idx >= 0:
                            cds_start = idx * 3
                            domain_cds = seq[cds_start:cds_start+len(prot_seq)*3+3]
                            dt = translate(domain_cds)
                            if dt[:15] == prot_seq[:15]:
                                return domain_cds, "ensembl_domain_ok"
                except:
                    pass
                time.sleep(0.3)
        return None, "ensembl_no_match"
    except Exception as e:
        return None, f"ensembl_error:{str(e)[:50]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--existing_json", default="data/proteingym/proteingym_cds.json")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--out_json", default="data/proteingym/proteingym_cds.json")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref_csv)

    # Load all existing CDS
    results = {}
    if os.path.exists(args.existing_json):
        with open(args.existing_json) as f:
            results = json.load(f)

    gene_cds = {}
    if os.path.exists(args.gene_pairs):
        with open(args.gene_pairs) as f:
            for g in json.load(f):
                if "cds_seq" in g:
                    gene_cds[g.get("gene_name", "").upper()] = g["cds_seq"]

    session = requests.Session()
    session.headers["User-Agent"] = "CrossBioSAE/1.0 (mingxuz041@gmail.com)"

    stats = {"uniprot": 0, "ncbi_gene": 0, "ensembl": 0, "reverse": 0, "failed": 0, "skipped": 0}
    for _, row in ref.iterrows():
        dms_id = row["DMS_id"]
        uniprot_entry = str(row["UniProt_ID"])
        taxon = str(row.get("taxon", ""))

        if dms_id in results:
            stats["skipped"] += 1
            continue

        # Check gene_pairs
        gene = dms_id.split("_")[0].upper()
        dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            continue
        dms = pd.read_csv(dms_path, nrows=1)
        prot_seq = dms["target_seq"].iloc[0] if "target_seq" in dms.columns else ""
        if not prot_seq:
            continue

        if gene in gene_cds:
            cds = gene_cds[gene]
            t = translate(cds)
            if t[:15] == prot_seq[:15]:
                results[dms_id] = cds
                print(f"[{len(results)}] {dms_id}: gene_pairs match", flush=True)
                continue
            # Maybe domain within full CDS
            idx = t.find(prot_seq[:15])
            if idx >= 0:
                cds_start = idx * 3
                domain_cds = cds[cds_start:cds_start + len(prot_seq)*3 + 3]
                dt = translate(domain_cds)
                if dt[:15] == prot_seq[:15]:
                    results[dms_id] = domain_cds
                    print(f"[{len(results)}] {dms_id}: gene_pairs domain match", flush=True)
                    continue

        # Strategy 1: UniProt search API
        cds, method = try_uniprot_accession(uniprot_entry, prot_seq, session)
        if cds:
            results[dms_id] = cds
            stats["uniprot"] += 1
            print(f"[{len(results)}] {dms_id}: {method} ({len(cds)} bp)", flush=True)
            time.sleep(0.5)
            continue

        # Strategy 2: NCBI Gene search
        gene_short = dms_id.split("_")[0]
        org_hint = uniprot_entry.split("_")[-1] if "_" in uniprot_entry else taxon
        cds, method = try_ncbi_gene_search(gene_short, org_hint, prot_seq, session)
        if cds:
            results[dms_id] = cds
            stats["ncbi_gene"] += 1
            print(f"[{len(results)}] {dms_id}: {method} ({len(cds)} bp)", flush=True)
            time.sleep(0.5)
            continue

        # Strategy 3: Ensembl
        cds, method = try_ensembl(gene_short, org_hint, prot_seq, session)
        if cds:
            results[dms_id] = cds
            stats["ensembl"] += 1
            print(f"[{len(results)}] {dms_id}: {method} ({len(cds)} bp)", flush=True)
            time.sleep(0.5)
            continue

        # Strategy 4: Reverse translation (last resort)
        org_group = get_organism_group(uniprot_entry, taxon)
        rev_cds = reverse_translate(prot_seq, org_group)
        t = translate(rev_cds)
        if t == prot_seq:
            results[dms_id] = rev_cds
            stats["reverse"] += 1
            print(f"[{len(results)}] {dms_id}: reverse_translate ({org_group}, {len(rev_cds)} bp)", flush=True)
            continue
        else:
            stats["failed"] += 1
            print(f"[{len(results)}] {dms_id}: FAILED all strategies", flush=True)

        if len(results) % 10 == 0:
            with open(args.out_json, "w") as f:
                json.dump(results, f)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone: {len(results)}/217 CDS")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
