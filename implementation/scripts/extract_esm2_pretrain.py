"""
ESM-2 embedding deltas for 500k missense variants (protein side).
Uses pre-annotated protein changes from missense_500k_annotated.csv.
All 500k are missense → all should have protein embeddings.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ESM_LAYER = 33
ESM_DIM = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--output_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load annotated variants
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    n = len(df)
    s0 = args.shard * n // args.nshards
    s1 = (args.shard + 1) * n // args.nshards
    sub = df.iloc[s0:s1].reset_index(drop=True)
    logger.info(f"shard {args.shard}/{args.nshards}: {len(sub)} variants")

    # load protein sequences
    with open(args.gene_pairs) as f:
        gp = json.load(f)
    gene_seqs = {g["gene_name"]: g["protein_seq"] for g in gp}
    logger.info(f"loaded {len(gene_seqs)} protein sequences")

    # identify processable variants
    processable = []
    for i, r in sub.iterrows():
        gene = r["gene"]
        if gene not in gene_seqs:
            continue
        seq = gene_seqs[gene]
        pos = int(r["prot_pos"]) - 1  # 0-indexed
        if pos < 0 or pos >= len(seq):
            continue
        if seq[pos] != r["from_aa"]:
            continue
        processable.append((i, r["from_aa"], r["to_aa"], pos, seq, gene))

    logger.info(f"processable: {len(processable)} / {len(sub)} "
                f"({100*len(processable)/len(sub):.1f}%)")

    if len(processable) == 0:
        edelta = np.zeros((len(sub), ESM_DIM), dtype=np.float32)
        out = os.path.join(args.output_dir, f"esm2_emb_shard{args.shard}.npz")
        np.savez_compressed(out, idx=np.arange(s0, s1), edelta=edelta)
        logger.info(f"no processable, saved zeros")
        return

    # load ESM-2
    logger.info("loading ESM-2 ...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    edelta = np.zeros((len(sub), ESM_DIM), dtype=np.float32)
    bs = args.batch

    # group by gene to batch same-length sequences
    from collections import defaultdict
    gene_groups = defaultdict(list)
    for item in processable:
        gene_groups[item[5]].append(item)

    done = 0
    for gene, items in gene_groups.items():
        for batch_start in range(0, len(items), bs):
            batch_items = items[batch_start:batch_start + bs]

            ref_data = []
            alt_data = []
            positions = []
            indices = []

            for idx, from_aa, to_aa, pos, seq, g in batch_items:
                ref_data.append((f"r{idx}", seq))
                alt_seq = seq[:pos] + to_aa + seq[pos + 1:]
                alt_data.append((f"a{idx}", alt_seq))
                positions.append(pos + 1)  # +1 for BOS
                indices.append(idx)

            while True:
                try:
                    _, _, ref_tokens = batch_converter(ref_data)
                    ref_tokens = ref_tokens[:, :1024].cuda()  # truncate long proteins
                    with torch.no_grad():
                        ref_out = model(ref_tokens, repr_layers=[ESM_LAYER])
                    ref_emb = ref_out["representations"][ESM_LAYER]

                    _, _, alt_tokens = batch_converter(alt_data)
                    alt_tokens = alt_tokens[:, :1024].cuda()
                    with torch.no_grad():
                        alt_out = model(alt_tokens, repr_layers=[ESM_LAYER])
                    alt_emb = alt_out["representations"][ESM_LAYER]

                    for j, (idx, pos) in enumerate(zip(indices, positions)):
                        if pos < ref_emb.shape[1] and pos < alt_emb.shape[1]:
                            edelta[idx] = (alt_emb[j, pos] - ref_emb[j, pos]).float().cpu().numpy()
                    break

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs <= 1:
                        for idx, from_aa, to_aa, pos, seq, g in batch_items:
                            try:
                                _, _, rt = batch_converter([("r", seq)])
                                rt = rt[:, :1024].cuda()
                                _, _, at = batch_converter([("a", seq[:pos] + to_aa + seq[pos+1:])])
                                at = at[:, :1024].cuda()
                                with torch.no_grad():
                                    re = model(rt, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                                    ae = model(at, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                                p = pos + 1
                                if p < re.shape[1]:
                                    edelta[idx] = (ae[0, p] - re[0, p]).float().cpu().numpy()
                            except:
                                pass
                        break
                    bs = max(1, bs // 2)
                    logger.warning(f"OOM -> batch={bs}")

            done += len(batch_items)
            if done % 2000 < bs:
                logger.info(f"  emb {done}/{len(processable)}")

    out = os.path.join(args.output_dir, f"esm2_emb_shard{args.shard}.npz")
    nz = np.any(edelta != 0, axis=1).sum()
    np.savez_compressed(out, idx=np.arange(s0, s1), edelta=edelta)
    logger.info(f"saved {out}: {edelta.shape}, non-zero={nz}")


if __name__ == "__main__":
    main()
