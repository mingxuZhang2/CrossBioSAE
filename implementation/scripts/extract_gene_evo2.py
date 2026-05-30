"""
Extract Evo2-7b gene-level embeddings for all 16,620 protein-coding genes.

For each gene, feed its CDS sequence through Evo2, extract the mid-layer
embedding (blocks.28.mlp.l3, 4096-d), mean-pool across tokens → one 4096-d
vector per gene. Saves as HDF5 (same format as existing protein_activations.h5).

CDS sequences are short (median 1107bp, max 3000bp) so we can use larger
batches than the 8192bp variant windows. Auto-halves on OOM.

Shardable: --shard K --nshards N processes genes[K::N].
"""

import argparse
import json
import logging
import os

import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default="data/full/gene_pairs_full.json")
    ap.add_argument("--output", default="data/full/evo2_activations.h5")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path", default="models/evo2_7b.pt")
    ap.add_argument("--layer", default="blocks.28.mlp.l3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    genes = json.load(open(args.genes))
    genes = [g for g in genes if g.get("cds_seq") and len(g["cds_seq"]) > 10]
    logger.info(f"total genes with CDS: {len(genes)}")

    if args.nshards > 1:
        genes = genes[args.shard::args.nshards]
        out_path = args.output.replace(".h5", f"_shard{args.shard}.h5")
        logger.info(f"shard {args.shard}/{args.nshards}: {len(genes)} genes -> {out_path}")
    else:
        out_path = args.output

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    embeddings = []
    names = []
    bs = args.batch

    for i in range(0, len(genes), bs):
        batch_genes = genes[i:i + bs]
        seqs = [g["cds_seq"] for g in batch_genes]

        while True:
            try:
                ids = [torch.tensor(model.tokenizer.tokenize(s), dtype=torch.int)
                       for s in seqs]
                maxlen = max(x.shape[0] for x in ids)
                batch = torch.zeros(len(ids), maxlen, dtype=torch.int)
                for j, x in enumerate(ids):
                    batch[j, :x.shape[0]] = x
                batch = batch.to("cuda:0")

                with torch.no_grad():
                    _, emb = model(batch, return_embeddings=True,
                                   layer_names=[args.layer])
                    e = emb[args.layer]  # (B, L, 4096)

                for j in range(len(seqs)):
                    seq_len = ids[j].shape[0]
                    gene_emb = e[j, :seq_len].float().mean(dim=0).cpu().numpy()
                    embeddings.append(gene_emb)
                    names.append(batch_genes[j]["gene_name"])
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    logger.error(f"OOM even at batch=1 for gene {batch_genes[0]['gene_name']}")
                    for g in batch_genes:
                        embeddings.append(np.zeros(4096, dtype=np.float32))
                        names.append(g["gene_name"])
                    break
                bs = max(1, bs // 2)
                logger.warning(f"OOM -> retry with batch={bs}")

        if (i // max(bs, 1)) % 20 == 0:
            logger.info(f"  {i + len(batch_genes)}/{len(genes)} genes done")

    embeddings = np.stack(embeddings)
    logger.info(f"embeddings shape: {embeddings.shape}")

    with h5py.File(out_path, "w") as h:
        h.create_dataset("activations", data=embeddings.astype(np.float32))
        h.create_dataset("gene_names", data=np.array(names, dtype="S"))
    logger.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
