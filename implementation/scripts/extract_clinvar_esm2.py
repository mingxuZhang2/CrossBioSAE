"""
Extract ESM-2 edelta embeddings for ClinVar missense variants.

For each variant: edelta = ESM2(mutant_seq)[mut_pos] - ESM2(wildtype_seq)[mut_pos]
Protein sequences fetched from UniProt via gene symbol.

Sharded for parallelism: --shard 0 --nshards 4

Output: results/clinvar_esm2/shard_{K}.npz
  - esm2_edelta: (N, 1280) float32
  - gene: (N,) str
  - mutant: (N,) str  (e.g., "R175H")
  - label: (N,) int   (1=pathogenic, 0=benign)
  - idx: (N,) int     (row index in source CSV)
"""

import argparse, os, re, sys, time, logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AA3TO1 = {
    'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E', 'Phe': 'F',
    'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Lys': 'K', 'Leu': 'L',
    'Met': 'M', 'Asn': 'N', 'Pro': 'P', 'Gln': 'Q', 'Arg': 'R',
    'Ser': 'S', 'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y',
}

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


def parse_clinvar_name(name):
    """Parse p.Arg175His → (R, 175, H)."""
    m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(name))
    if not m:
        return None, None, None
    ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
    ref1 = AA3TO1.get(ref3)
    alt1 = AA3TO1.get(alt3)
    if ref1 and alt1 and ref1 in VALID_AA and alt1 in VALID_AA:
        return ref1, pos, alt1
    return None, None, None


def fetch_uniprot_sequences(genes):
    """Batch fetch protein sequences from UniProt for human genes."""
    import requests
    log.info("Fetching %d unique gene sequences from UniProt..." % len(genes))
    gene_seqs = {}
    batch_size = 50
    gene_list = list(genes)

    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i + batch_size]
        query = ' OR '.join('(gene_exact:%s AND organism_id:9606)' % g for g in batch)
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': query + ' AND reviewed:true',
            'format': 'tsv',
            'fields': 'gene_primary,sequence',
            'size': 500,
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            lines = resp.text.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene = parts[0].strip()
                        seq = parts[1].strip()
                        if gene in genes and gene not in gene_seqs and len(seq) > 10:
                            gene_seqs[gene] = seq
        except Exception as e:
            log.warning("UniProt batch failed: %s" % str(e))

        if (i // batch_size + 1) % 10 == 0:
            log.info("  fetched %d/%d genes (%d found)" % (
                min(i + batch_size, len(gene_list)), len(gene_list), len(gene_seqs)))

    log.info("Got sequences for %d / %d genes" % (len(gene_seqs), len(genes)))
    return gene_seqs


def extract_esm2_edelta_batch(seq, mutations, model, batch_converter, device, layer=33, bs=32):
    """Extract ESM-2 edelta for a list of mutations on the same sequence.

    mutations: list of (pos0, alt_aa)  (0-indexed position)
    Returns: (N, 1280) edelta array
    """
    # Get wildtype embedding
    data = [("wt", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer])
    wt_repr = out['representations'][layer][0]  # (L+2, 1280), includes BOS/EOS

    deltas = []
    for start in range(0, len(mutations), bs):
        batch_muts = mutations[start:start + bs]
        mut_seqs = []
        for pos0, alt_aa in batch_muts:
            ms = seq[:pos0] + alt_aa + seq[pos0 + 1:]
            mut_seqs.append(("mut", ms))

        _, _, mtokens = batch_converter(mut_seqs)
        mtokens = mtokens.to(device)
        with torch.no_grad():
            mout = model(mtokens, repr_layers=[layer])
        mut_reprs = mout['representations'][layer]  # (B, L+2, 1280)

        for j, (pos0, _) in enumerate(batch_muts):
            tok_idx = pos0 + 1  # +1 for BOS token
            delta = mut_reprs[j, tok_idx] - wt_repr[tok_idx]
            deltas.append(delta.cpu().numpy())

    return np.stack(deltas).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar_csv", default="data/full/clinvar_variants.csv")
    ap.add_argument("--out_dir", default="results/clinvar_esm2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_cache", default="data/full/gene_sequences.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "shard_%d.npz" % args.shard)
    if os.path.exists(out_path):
        log.info("Output exists: %s, skipping" % out_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s, shard %d/%d" % (device, args.shard, args.nshards))

    # Load ClinVar
    cv = pd.read_csv(args.clinvar_csv)
    log.info("ClinVar: %d variants" % len(cv))

    # Parse protein changes
    parsed = []
    for i, row in cv.iterrows():
        ref, pos, alt = parse_clinvar_name(row['Name'])
        if ref and alt and ref != alt:
            parsed.append({
                'idx': i, 'gene': row['GeneSymbol'],
                'ref': ref, 'pos': pos, 'alt': alt,
                'label': int(row['pathogenicity']),
                'mutant': '%s%d%s' % (ref, pos, alt),
            })
    df = pd.DataFrame(parsed)
    log.info("Parsed missense: %d variants, %d genes" % (len(df), df['gene'].nunique()))

    # Shard by gene (keep all variants for same gene together)
    genes = sorted(df['gene'].unique())
    shard_genes = genes[args.shard::args.nshards]
    df_shard = df[df['gene'].isin(shard_genes)].reset_index(drop=True)
    log.info("Shard %d: %d genes, %d variants" % (args.shard, len(shard_genes), len(df_shard)))

    # Load protein sequences from pre-downloaded cache
    import json
    if os.path.exists(args.seq_cache):
        log.info("Loading cached sequences from %s" % args.seq_cache)
        with open(args.seq_cache) as f:
            gene_seqs = json.load(f)
    else:
        log.error("Sequence cache not found: %s. Run download on login node first." % args.seq_cache)
        return

    log.info("Sequences available for %d / %d shard genes" % (
        len(set(shard_genes) & set(gene_seqs.keys())), len(shard_genes)))

    # Load ESM-2
    log.info("Loading ESM-2 650M...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Extract per gene
    all_edelta, all_gene, all_mutant, all_label, all_idx = [], [], [], [], []
    n_done = 0

    MAX_SEQ_LEN = 2048

    for gene in shard_genes:
        if gene not in gene_seqs:
            continue
        seq = gene_seqs[gene]
        if len(seq) > MAX_SEQ_LEN:
            continue
        gdf = df_shard[df_shard['gene'] == gene]

        mutations = []
        valid_rows = []
        for _, row in gdf.iterrows():
            pos0 = row['pos'] - 1
            if 0 <= pos0 < len(seq) and seq[pos0] == row['ref']:
                mutations.append((pos0, row['alt']))
                valid_rows.append(row)

        if len(mutations) == 0:
            continue

        try:
            edelta = extract_esm2_edelta_batch(
                seq, mutations, esm_model, batch_converter, device,
                bs=args.batch_size)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                log.warning("OOM %s (len=%d), skipping" % (gene, len(seq)))
            else:
                log.warning("Failed %s: %s" % (gene, e))
            continue
        except Exception as e:
            log.warning("Failed %s (len=%d, n=%d): %s" % (gene, len(seq), len(mutations), e))
            continue

        for j, row in enumerate(valid_rows):
            all_edelta.append(edelta[j])
            all_gene.append(gene)
            all_mutant.append(row['mutant'])
            all_label.append(row['label'])
            all_idx.append(row['idx'])

        n_done += 1
        if n_done % 50 == 0:
            log.info("  %d genes done, %d variants extracted" % (n_done, len(all_edelta)))

    # Save
    if all_edelta:
        np.savez_compressed(out_path,
                            esm2_edelta=np.stack(all_edelta),
                            gene=np.array(all_gene),
                            mutant=np.array(all_mutant),
                            label=np.array(all_label, dtype=np.int32),
                            idx=np.array(all_idx, dtype=np.int64))
        log.info("Saved %s: %d variants" % (out_path, len(all_edelta)))
    else:
        log.warning("No variants extracted for shard %d" % args.shard)


if __name__ == "__main__":
    main()
