"""
Extract Evo2-7b gene-level embeddings for multi-species genes.
Same as extract_gene_evo2.py but reads from data/multispecies/*.json
and writes to data/multispecies/evo2_activations_{species}.h5.
Shardable across species×shard.
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

SPECIES = ["mouse", "zebrafish", "rat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/multispecies")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--local_path", default="models/evo2_7b.pt")
    ap.add_argument("--layer", default="blocks.28.mlp.l3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--species", default="all", help="all or mouse,zebrafish,rat")
    args = ap.parse_args()

    species_list = SPECIES if args.species == "all" else args.species.split(",")

    from evo2.models import Evo2
    logger.info(f"loading {args.model} from {args.local_path} ...")
    model = Evo2(args.model, local_path=args.local_path)

    for sp in species_list:
        genes_file = os.path.join(args.data_dir, f"gene_pairs_{sp}.json")
        out_h5 = os.path.join(args.data_dir, f"evo2_activations_{sp}.h5")

        if os.path.exists(out_h5):
            logger.info(f"{sp}: already exists, skip")
            continue

        genes = json.load(open(genes_file))
        genes = [g for g in genes if g.get("cds_seq") and len(g["cds_seq"]) > 10]
        logger.info(f"{sp}: {len(genes)} genes")

        embeddings, names = [], []
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
                        e = emb[args.layer]
                    for j in range(len(seqs)):
                        seq_len = ids[j].shape[0]
                        gene_emb = e[j, :seq_len].float().mean(dim=0).cpu().numpy()
                        embeddings.append(gene_emb)
                        names.append(batch_genes[j]["gene_name"])
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        for g in batch_genes:
                            embeddings.append(np.zeros(4096, dtype=np.float32))
                            names.append(g["gene_name"])
                        break
                    bs = max(1, bs // 2)
                    logger.warning(f"OOM -> batch={bs}")
            if (i // max(bs, 1)) % 20 == 0:
                logger.info(f"  {sp}: {i + len(batch_genes)}/{len(genes)}")

        embeddings = np.stack(embeddings)
        with h5py.File(out_h5, "w") as h:
            h.create_dataset("activations", data=embeddings.astype(np.float32))
            h.create_dataset("gene_names", data=np.array(names, dtype="S"))
        logger.info(f"saved {out_h5}: {embeddings.shape}")


if __name__ == "__main__":
    main()
