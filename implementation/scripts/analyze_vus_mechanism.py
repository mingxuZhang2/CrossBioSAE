"""
VUS mechanism profiling using SAE features.

After Evo2 embedding extraction, projects VUS through the pretrained DNA encoder,
runs them through the genome-wide SAE, and generates mechanism profiles.

Key output: which VUS activate novel structural-protein features → candidates
for reclassification as Likely Pathogenic.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

OUT = "results/vus_pilot"

# ── Load VUS data ────────────────────────────────────────────────
print("Loading VUS data ...")
vus = pd.read_csv("data/variant/vus_pilot/vus_structural_annotated.csv")
vus_pq = pd.read_parquet("data/variant/vus_pilot/vus_structural.parquet")
print(f"  {len(vus)} VUS variants")

# merge Evo2 embedding shards
n = len(vus_pq)
edelta = np.zeros((n, 4096), dtype=np.float32)
found = 0
for s in range(8):
    f = os.path.join(OUT, f"clinvar_evo2_emb_shard{s}.npz")
    if not os.path.exists(f):
        print(f"  MISSING shard {s}")
        continue
    d = np.load(f)
    edelta[d["idx"]] = d["edelta"]
    found += len(d["idx"])
print(f"  Evo2 embeddings: {found}/{n}")

has_emb = np.any(edelta != 0, axis=1)
print(f"  VUS with embedding: {has_emb.sum()}")

if has_emb.sum() == 0:
    print("ERROR: No embeddings found. Run slurm_vus_evo2_emb.sh first.")
    sys.exit(1)

# ── Project through pretrained encoder ───────────────────────────
print("\nProjecting through pretrained DNA encoder ...")
ckpt = torch.load("results/pretrain/crossmodal_clip_40k.pt", map_location="cpu",
                   weights_only=False)
clip = CrossModalCLIP(**ckpt["config"])
clip.load_state_dict(ckpt["model_state"])
clip = clip.eval()

with torch.no_grad():
    z_dna = clip.enc_dna(torch.tensor(edelta[has_emb], dtype=torch.float32)).numpy()
print(f"  z_dna: {z_dna.shape}")

# standardize using the same scaler as genome-wide SAE
from sklearn.preprocessing import StandardScaler

# load the training data stats (from genome-wide SAE)
gw_edelta = np.zeros((40976, 4096), dtype=np.float32)
for s in range(8):
    f = f"results/variant/clinvar_evo2_emb_shard{s}.npz"
    if os.path.exists(f):
        d = np.load(f)
        gw_edelta[d["idx"]] = d["edelta"]
gw_has = np.any(gw_edelta != 0, axis=1)
with torch.no_grad():
    gw_zdna = clip.enc_dna(torch.tensor(gw_edelta[gw_has], dtype=torch.float32)).numpy()
sc = StandardScaler().fit(gw_zdna)
z_s = sc.transform(z_dna).astype(np.float32)

# ── Run through SAE ─────────────────────────────────────────────
print("Running through genome-wide SAE ...")
import torch.nn as nn
import torch.nn.functional as F

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

# we need the trained SAE weights — check if saved on HPC3
sae_checkpoint = os.path.join("results/sae_genomewide", "sae_model.pt")
if not os.path.exists(sae_checkpoint):
    # retrain SAE on genome-wide data (fast on CPU)
    print("  Retraining SAE (genome-wide data not saved, retraining) ...")
    gw_zs = sc.transform(gw_zdna).astype(np.float32)
    gw_zt = torch.tensor(gw_zs)
    sae = TopKSAE(d_in=256, d_hidden=2048, k=32)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for ep in range(1, 1001):
        sae.train()
        x_hat, h_sp = sae(gw_zt)
        loss = F.mse_loss(x_hat, gw_zt) + 0.01 * h_sp.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0:
            print(f"    ep {ep}: loss={loss.item():.4f}")
    torch.save(sae.state_dict(), sae_checkpoint)
    print(f"  Saved SAE to {sae_checkpoint}")
else:
    sae = TopKSAE(d_in=256, d_hidden=2048, k=32)
    sae.load_state_dict(torch.load(sae_checkpoint, map_location="cpu"))
    print("  Loaded saved SAE")

sae.eval()
with torch.no_grad():
    _, h_sparse = sae(torch.tensor(z_s))
    vus_acts = h_sparse.numpy()

print(f"  VUS activations: {vus_acts.shape}")

# ── Load feature modality labels ─────────────────────────────────
cards = pd.read_csv("results/sae_genomewide/sae_genomewide_cards_deep.csv")
feat_modality = dict(zip(cards["feature"], cards["modality"]))

# novel pathogenic features (from our analysis)
novel_path_feats = cards[(cards["modality"] == "novel") & (cards["path_rate"] > 0.85)]
novel_feat_ids = set(novel_path_feats["feature"].values)

# ── VUS mechanism profiles ──────────────────────────────────────
print(f"\n{'='*70}")
print("VUS MECHANISM PROFILES")
print(f"{'='*70}")

vus_sub = vus[has_emb].reset_index(drop=True)

# for each VUS: count activations in novel pathogenic features
novel_act_scores = np.zeros(len(vus_sub))
for fi in novel_feat_ids:
    novel_act_scores += np.abs(vus_acts[:, fi])

vus_sub["novel_path_score"] = novel_act_scores
vus_sub["n_novel_features_active"] = sum(
    (np.abs(vus_acts[:, fi]) > 1e-6).astype(int) for fi in novel_feat_ids
)

# top feature per VUS
top_feat = np.argmax(np.abs(vus_acts), axis=1)
vus_sub["top_feature"] = top_feat
vus_sub["top_feature_modality"] = [feat_modality.get(fi, "unknown") for fi in top_feat]

# ── Results by gene ─────────────────────────────────────────────
print(f"\nVUS with novel pathogenic feature activation:")
print(f"  {'gene':10s} {'total':>6s} {'gly':>5s} {'novel_active':>12s} {'frac':>6s} {'mean_score':>10s}")

for gene in sorted(vus_sub["gene"].unique()):
    g = vus_sub[vus_sub["gene"] == gene]
    n_novel = (g["n_novel_features_active"] > 0).sum()
    n_gly = g["is_gly_sub"].sum()
    ms = g["novel_path_score"].mean()
    print(f"  {gene:10s} {len(g):6d} {n_gly:5d} {n_novel:12d} "
          f"{n_novel/len(g):6.2%} {ms:10.3f}")

# ── Gly-sub VUS with novel feature activation ───────────────────
print(f"\n{'='*70}")
print("RECLASSIFICATION CANDIDATES: Gly-sub VUS with novel feature activation")
print(f"{'='*70}")

gly_vus = vus_sub[vus_sub["is_gly_sub"]]
gly_novel = gly_vus[gly_vus["n_novel_features_active"] > 0]
print(f"\n  Gly-sub VUS total: {len(gly_vus)}")
print(f"  With novel pathogenic feature: {len(gly_novel)} ({100*len(gly_novel)/max(len(gly_vus),1):.1f}%)")

if len(gly_novel) > 0:
    print(f"\n  By gene:")
    for gene, grp in gly_novel.groupby("gene"):
        print(f"    {gene:10s}: {len(grp):4d} candidates (mean novel_score={grp['novel_path_score'].mean():.3f})")

    print(f"\n  Top 20 strongest candidates:")
    top = gly_novel.sort_values("novel_path_score", ascending=False).head(20)
    print(f"  {'gene':10s} {'chrom':>5s} {'pos':>10s} {'pchange':>15s} {'n_feat':>6s} {'score':>8s}")
    for _, r in top.iterrows():
        print(f"  {r['gene']:10s} {str(r['chrom']):>5s} {int(r['pos']):10d} "
              f"{r['pchange']:>15s} {int(r['n_novel_features_active']):6d} "
              f"{r['novel_path_score']:8.3f}")

# ── Non-Gly VUS with novel feature activation (new discoveries) ──
print(f"\n{'='*70}")
print("POTENTIAL NEW DISCOVERIES: non-Gly VUS with novel feature activation")
print(f"{'='*70}")

nongly_novel = vus_sub[~vus_sub["is_gly_sub"] & (vus_sub["n_novel_features_active"] > 2)]
print(f"\n  Non-Gly VUS with 3+ novel pathogenic features: {len(nongly_novel)}")
if len(nongly_novel) > 0:
    top_ng = nongly_novel.sort_values("novel_path_score", ascending=False).head(10)
    for _, r in top_ng.iterrows():
        print(f"  {r['gene']:10s} chr{r['chrom']}:{int(r['pos'])} "
              f"{r['pchange']:>15s} n_feat={int(r['n_novel_features_active'])} "
              f"score={r['novel_path_score']:.3f}")

# save
vus_sub.to_csv(os.path.join(OUT, "vus_mechanism_profiles.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "vus_sae_acts.npz"), acts=vus_acts)
print(f"\nSaved to {OUT}/")
