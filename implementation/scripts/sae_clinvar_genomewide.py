"""
Genome-wide SAE on the cross-modal shared embedding for 40k ClinVar variants.

Projects Evo2 embedding deltas through the pretrained DNA encoder into the
shared space (z_dna, 256-d), trains a TopK SAE (→2048 sparse features), and
maps each feature to biological concepts using the rich ClinVar annotations.

For ~36k coding variants that also have ESM-1b scores, we additionally check
whether SAE features correlate with the protein-side signal.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TopKSAE(nn.Module):
    def __init__(self, d_in=256, d_hidden=2048, k=32):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        self.k = k

    def forward(self, x):
        h = self.encoder(x)
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        h_sparse = h * mask
        x_hat = self.decoder(h_sparse)
        return x_hat, h_sparse

    def loss(self, x, x_hat, h_sparse):
        recon = F.mse_loss(x_hat, x)
        l1 = h_sparse.abs().mean() * 0.01
        return recon + l1


def merge_emb_shards(output_dir, n_total, n_shards=8):
    edelta = np.zeros((n_total, 4096), dtype=np.float32)
    found = 0
    for s in range(n_shards):
        f = os.path.join(output_dir, f"clinvar_evo2_emb_shard{s}.npz")
        if not os.path.exists(f):
            logger.warning(f"MISSING {f}")
            continue
        d = np.load(f)
        edelta[d["idx"]] = d["edelta"]
        found += len(d["idx"])
    logger.info(f"merged embeddings: {found}/{n_total} variants")
    return edelta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/variant/clinvar.parquet")
    ap.add_argument("--emb_dir", default="results/variant")
    ap.add_argument("--checkpoint", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--output_dir", default="results/sae_genomewide")
    ap.add_argument("--sae_hidden", type=int, default=2048)
    ap.add_argument("--sae_k", type=int, default=32)
    ap.add_argument("--sae_epochs", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    df["chrom"] = df["chrom"].astype(str)
    logger.info(f"ClinVar: {len(df)} variants")

    # merge Evo2 embedding shards
    edelta = merge_emb_shards(args.emb_dir, len(df))
    has_emb = np.any(edelta != 0, axis=1)
    logger.info(f"variants with Evo2 embedding: {has_emb.sum()}/{len(df)}")

    # project through pretrained DNA encoder → shared space
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    clip = CrossModalCLIP(**cfg)
    clip.load_state_dict(ckpt["model_state"])
    clip = clip.eval().to(dev)

    with torch.no_grad():
        xd = torch.tensor(edelta[has_emb], dtype=torch.float32, device=dev)
        z_dna = clip.enc_dna(xd).cpu().numpy()  # (N, 256)
    logger.info(f"z_dna shape: {z_dna.shape}")

    df_sub = df[has_emb].reset_index(drop=True)
    y = df_sub["label"].astype(int).values
    esm = df_sub["ESM-1b"].values.astype(float)

    # standardize
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(z_dna)
    z_s = sc.transform(z_dna).astype(np.float32)
    z_t = torch.tensor(z_s, device=dev)

    # train SAE
    logger.info(f"training SAE: {z_dna.shape[1]}→{args.sae_hidden}, k={args.sae_k}")
    sae = TopKSAE(d_in=z_dna.shape[1], d_hidden=args.sae_hidden, k=args.sae_k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for ep in range(1, args.sae_epochs + 1):
        sae.train()
        x_hat, h_sparse = sae(z_t)
        loss = sae.loss(z_t, x_hat, h_sparse)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0:
            logger.info(f"  ep {ep}: loss={loss.item():.4f}")

    sae.eval()
    with torch.no_grad():
        _, h_sparse = sae(z_t)
        acts = h_sparse.cpu().numpy()

    # feature analysis
    alive = (np.abs(acts) > 1e-6).sum(axis=0)
    n_alive = (alive > 0).sum()
    logger.info(f"alive features: {n_alive}/{args.sae_hidden}")

    # annotations
    has_esm = np.isfinite(esm)
    path_base = y.mean()

    print(f"\n{'='*70}")
    print(f"Genome-wide SAE: {len(df_sub)} variants, {n_alive}/{args.sae_hidden} alive features")
    print(f"{'='*70}")

    cards = []
    for fi in range(args.sae_hidden):
        a = acts[:, fi]
        active = np.abs(a) > 1e-6
        n_act = active.sum()
        if n_act < 10:
            continue

        card = {"feature": fi, "n_active": int(n_act)}
        card["path_rate"] = y[active].mean()
        card["path_enrich"] = card["path_rate"] / max(path_base, 1e-8)
        card["esm_avail_frac"] = has_esm[active].mean()

        # mean scores for active variants
        esm_active = esm[active & has_esm]
        card["mean_ESM1b"] = float(np.mean(esm_active)) if len(esm_active) > 0 else np.nan
        for col in ["GPN-MSA", "CADD", "phyloP-100v", "phastCons-100v"]:
            if col in df_sub.columns:
                vals = df_sub[col].values[active]
                card[f"mean_{col}"] = float(np.nanmean(vals))

        # chromosome enrichment
        chrom_counts = df_sub["chrom"].values[active]
        top_chrom = pd.Series(chrom_counts).value_counts()
        card["top_chrom"] = top_chrom.index[0]
        card["top_chrom_frac"] = top_chrom.iloc[0] / n_act
        card["n_chroms"] = len(top_chrom)

        # coding/noncoding
        card["esm_coding_frac"] = has_esm[active].mean()

        # concept assignment
        if card["path_rate"] > 0.8:
            if card["esm_coding_frac"] < 0.3:
                card["concept"] = "noncoding_pathogenic"
            elif card.get("mean_CADD", 0) > 25:
                card["concept"] = "high_impact_damaging"
            else:
                card["concept"] = "pathogenic_mixed"
        elif card["path_rate"] < 0.15:
            if card["esm_coding_frac"] < 0.3:
                card["concept"] = "noncoding_benign"
            elif card["esm_coding_frac"] > 0.8:
                card["concept"] = "coding_benign"
            else:
                card["concept"] = "benign_mixed"
        elif card["n_chroms"] == 1:
            card["concept"] = f"chr{card['top_chrom']}_cluster"
        elif card.get("mean_phyloP-100v", 0) > 5:
            card["concept"] = "highly_conserved"
        elif card.get("mean_phyloP-100v", 0) < -1:
            card["concept"] = "fast_evolving"
        else:
            card["concept"] = "mixed"

        cards.append(card)

    fc = pd.DataFrame(cards)

    # concept summary
    print(f"\nConcept distribution ({len(fc)} alive features with n≥10):")
    for concept, grp in fc.groupby("concept"):
        print(f"  {concept:25s}: {len(grp):3d} feats, path_rate={grp['path_rate'].mean():.2f}, "
              f"CADD={grp['mean_CADD'].mean():.1f}, n_chroms={grp['n_chroms'].mean():.1f}")

    # top pathogenic features
    print(f"\nTop 15 pathogenic-enriched features:")
    print(f"  {'feat':>5s} {'n':>6s} {'path%':>6s} {'enr':>5s} {'cod%':>5s} "
          f"{'CADD':>5s} {'phyP':>5s} {'GPN':>5s} {'ESM':>6s} {'chroms':>6s} concept")
    for _, r in fc.sort_values("path_enrich", ascending=False).head(15).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['path_rate']:6.2f} "
              f"{r['path_enrich']:5.2f} {r['esm_coding_frac']:5.2f} "
              f"{r.get('mean_CADD',0):5.1f} {r.get('mean_phyloP-100v',0):5.1f} "
              f"{r.get('mean_GPN-MSA',0):5.1f} {r.get('mean_ESM1b',0):6.2f} "
              f"{int(r['n_chroms']):6d} {r['concept']}")

    print(f"\nTop 15 benign-enriched features:")
    for _, r in fc.sort_values("path_enrich").head(15).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['path_rate']:6.2f} "
              f"{r['path_enrich']:5.2f} {r['esm_coding_frac']:5.2f} "
              f"{r.get('mean_CADD',0):5.1f} {r.get('mean_phyloP-100v',0):5.1f} "
              f"{r.get('mean_GPN-MSA',0):5.1f} {r.get('mean_ESM1b',0):6.2f} "
              f"{int(r['n_chroms']):6d} {r['concept']}")

    fc.to_csv(os.path.join(args.output_dir, "sae_genomewide_cards.csv"), index=False)
    np.savez_compressed(os.path.join(args.output_dir, "sae_genomewide_acts.npz"), acts=acts)
    print(f"\nsaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
