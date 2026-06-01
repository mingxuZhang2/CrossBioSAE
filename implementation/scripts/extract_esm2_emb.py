"""
ESM-2 embedding deltas for variants (protein side of cross-modal SAE).

For each coding missense variant:
1. Map genomic coordinates → protein sequence + position (via variant_summary)
2. Run ESM-2 on ref and alt protein sequences
3. Extract embedding delta at mutation position

Non-coding / non-missense variants get zero vectors (no protein signal).

Sharded: --shard 0..N --nshards N
Output per shard: esm2_emb_shard{K}.npz {idx, edelta}
"""

import argparse
import gzip
import logging
import os
import re

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

ESM_LAYER = 33  # last layer of ESM-2 650M (33 layers)
ESM_DIM = 1280  # hidden dim of ESM-2 650M


def load_variant_protein_mapping(variant_summary_gz, target_keys):
    """Map genomic variants to protein changes using ClinVar variant_summary.
    Only works for variants that appear in ClinVar."""
    mapping = {}
    with gzip.open(variant_summary_gz, "rt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 34 or parts[16] != "GRCh38":
                continue
            key = (parts[18], parts[31], parts[32], parts[33])
            if key not in target_keys or key in mapping:
                continue
            name = parts[2]
            gene = parts[4]
            m = re.search(r'\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)', name)
            if m:
                from_aa = AA_MAP.get(m.group(1))
                pos = int(m.group(2))
                to_aa = AA_MAP.get(m.group(3))
                if from_aa and to_aa and from_aa != to_aa:
                    mapping[key] = {
                        "gene": gene, "from_aa": from_aa,
                        "pos": pos, "to_aa": to_aa,
                    }
    logger.info(f"  ClinVar mapping: {len(mapping)} missense variants")
    return mapping


CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def map_flanking_variants_via_cds(sub, gene_pairs_json, genome_fa_gz):
    """For flanking variants not in ClinVar, attempt codon-level mapping
    using CDS coordinates. This is approximate — only works for variants
    that fall within known CDS regions."""
    # This is complex (needs CDS exon coordinates from RefSeq).
    # For now, return empty — flanking variants get zero protein embedding.
    # The ClinVar variants (40k) already have protein mapping.
    return {}


def get_uniprot_sequences(gene_names):
    """Load protein sequences for genes. Uses pre-downloaded sequences if available."""
    seq_file = "data/pretrain/protein_sequences.parquet"
    if os.path.exists(seq_file):
        df = pd.read_parquet(seq_file)
        seqs = dict(zip(df["gene_name"], df["sequence"]))
        return {g: seqs[g] for g in gene_names if g in seqs}

    # fallback: try gene_pairs
    for f in ["data/full/gene_pairs.parquet", "data/multispecies/gene_pairs.parquet",
              "data/pilot/gene_pairs.parquet"]:
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if "protein_seq" in df.columns and "gene_name" in df.columns:
                seqs = dict(zip(df["gene_name"], df["protein_seq"]))
                return {g: seqs[g] for g in gene_names if g in seqs}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/variant/gnomad_sample/gnomad_500k.parquet")
    ap.add_argument("--variant_summary", default="data/variant/variant_summary.txt.gz")
    ap.add_argument("--output_dir", default="results/gnomad_emb")
    ap.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load variants
    df = pd.read_parquet(args.parquet)
    n = len(df)
    shard_start = args.shard * n // args.nshards
    shard_end = (args.shard + 1) * n // args.nshards
    sub = df.iloc[shard_start:shard_end].reset_index(drop=True)
    logger.info(f"shard {args.shard}/{args.nshards}: {len(sub)} variants")

    # map to protein changes
    target_keys = set(zip(sub["chrom"].astype(str), sub["pos"].astype(str),
                          sub["ref"].astype(str), sub["alt"].astype(str)))
    logger.info(f"mapping {len(target_keys)} variants to protein changes ...")
    prot_map = load_variant_protein_mapping(args.variant_summary, target_keys)
    logger.info(f"  mapped: {len(prot_map)} missense variants")

    # find which variants have protein mapping
    var_info = []
    for i, r in sub.iterrows():
        key = (str(r["chrom"]), str(r["pos"]), str(r["ref"]), str(r["alt"]))
        if key in prot_map:
            var_info.append((i, prot_map[key]))
        else:
            var_info.append((i, None))

    # collect genes that need sequences
    genes_needed = set()
    for _, info in var_info:
        if info is not None:
            genes_needed.add(info["gene"])
    logger.info(f"  need sequences for {len(genes_needed)} genes")

    # load protein sequences
    gene_seqs = get_uniprot_sequences(genes_needed)
    logger.info(f"  loaded {len(gene_seqs)} gene sequences")

    # identify variants we can process (have mapping + sequence)
    processable = []
    for i, info in var_info:
        if info is not None and info["gene"] in gene_seqs:
            seq = gene_seqs[info["gene"]]
            pos = info["pos"] - 1  # 0-indexed
            if 0 <= pos < len(seq) and seq[pos] == info["from_aa"]:
                processable.append((i, info, seq))

    logger.info(f"  processable variants: {len(processable)} / {len(sub)}")

    if len(processable) == 0:
        # save zeros
        edelta = np.zeros((len(sub), ESM_DIM), dtype=np.float32)
        out = os.path.join(args.output_dir, f"esm2_emb_shard{args.shard}.npz")
        np.savez_compressed(out, idx=np.arange(shard_start, shard_end), edelta=edelta)
        logger.info(f"no processable variants, saved zeros to {out}")
        return

    # load ESM-2
    logger.info(f"loading ESM-2 ({args.esm_model}) ...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # extract embeddings
    edelta = np.zeros((len(sub), ESM_DIM), dtype=np.float32)
    bs = args.batch

    for batch_start in range(0, len(processable), bs):
        batch_items = processable[batch_start:batch_start + bs]

        # prepare ref and alt sequences
        ref_data = []
        alt_data = []
        positions = []
        indices = []

        for idx, info, seq in batch_items:
            pos = info["pos"] - 1
            ref_data.append((f"ref_{idx}", seq))
            alt_seq = seq[:pos] + info["to_aa"] + seq[pos + 1:]
            alt_data.append((f"alt_{idx}", alt_seq))
            positions.append(pos + 1)  # +1 for ESM's BOS token
            indices.append(idx)

        try:
            # ref embeddings
            _, _, ref_tokens = batch_converter(ref_data)
            ref_tokens = ref_tokens.cuda()
            with torch.no_grad():
                ref_out = model(ref_tokens, repr_layers=[ESM_LAYER])
            ref_emb = ref_out["representations"][ESM_LAYER]

            # alt embeddings
            _, _, alt_tokens = batch_converter(alt_data)
            alt_tokens = alt_tokens.cuda()
            with torch.no_grad():
                alt_out = model(alt_tokens, repr_layers=[ESM_LAYER])
            alt_emb = alt_out["representations"][ESM_LAYER]

            # extract delta at mutation position
            for j, (idx, pos) in enumerate(zip(indices, positions)):
                if pos < ref_emb.shape[1] and pos < alt_emb.shape[1]:
                    delta = (alt_emb[j, pos] - ref_emb[j, pos]).float().cpu().numpy()
                    edelta[idx] = delta

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            logger.warning(f"OOM -> batch={bs}, retrying individually")
            for idx, info, seq in batch_items:
                try:
                    pos = info["pos"] - 1
                    _, _, rt = batch_converter([(f"ref", seq)])
                    _, _, at = batch_converter([(f"alt", seq[:pos] + info["to_aa"] + seq[pos + 1:])])
                    with torch.no_grad():
                        re = model(rt.cuda(), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                        ae = model(at.cuda(), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                    edelta[idx] = (ae[0, pos + 1] - re[0, pos + 1]).float().cpu().numpy()
                except:
                    pass

        if (batch_start // max(bs, 1)) % 20 == 0:
            logger.info(f"  emb {batch_start + len(batch_items)}/{len(processable)}")

    out = os.path.join(args.output_dir, f"esm2_emb_shard{args.shard}.npz")
    np.savez_compressed(out, idx=np.arange(shard_start, shard_end), edelta=edelta)
    logger.info(f"saved {out}: {edelta.shape}, "
                f"non-zero={np.any(edelta != 0, axis=1).sum()}")


if __name__ == "__main__":
    main()
