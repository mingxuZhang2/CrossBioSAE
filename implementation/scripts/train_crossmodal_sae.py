"""
Train a TRUE cross-modal TopK SAE on z_concat = [z_prot(256) | z_dna(256)] = 512-d.

Unlike retrain_sae_largek.py (DNA-only, d_in=256), this decomposes BOTH modalities
jointly. Each SAE feature's decoder column splits into a protein half (dims 0:256)
and a DNA half (dims 256:512), giving a DIRECT modality attribution per feature
(protein-driven / DNA-driven / cross-modal) from the weights themselves — no
post-hoc correlation needed.

Pipeline (matches the validated normalization choice in retrain_sae_largek.py):
  1. load pretrained CrossModalCLIP (enc_prot: 1280->256, enc_dna: 4096->256)
  2. per shard: project RAW esm2 edelta -> z_prot, RAW evo2 edelta -> z_dna
     (no input norm; encoders L2-normalize their own outputs)
  3. keep dual-modality variants (both edelta non-zero), align by global idx
  4. StandardScaler on the 512-d z_concat (fit on train split)
  5. train TopKSAE(512 -> d_hidden -> 512, k)
  6. save model + scaler + per-feature modality attribution

Run on GPU via SLURM.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP


class TopKSAE(nn.Module):
    def __init__(self, d_in=512, d_hidden=4096, k=32):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        self.k = k

    def encode(self, x):
        h = self.encoder(x)
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask

    def forward(self, x):
        h_sparse = self.encode(x)
        x_hat = self.decoder(h_sparse)
        return x_hat, h_sparse


def load_dual_modality(emb_dir, clip, device, dim_prot=1280, dim_dna=4096):
    """Load all shards, project to z_prot/z_dna, keep variants present in BOTH.
    Returns z_concat (N,512) float32 and global idx (N,)."""
    # gather per-global-idx edelta for both modalities
    prot_files = sorted(glob.glob(os.path.join(emb_dir, "esm2_emb_shard*.npz")))
    dna_files = sorted(glob.glob(os.path.join(emb_dir, "clinvar_evo2_emb_shard*.npz")))
    assert prot_files and dna_files, "missing embedding shards"

    # build idx -> evo2 edelta (only non-zero) to know the dual set
    dna_map = {}
    for f in dna_files:
        d = np.load(f)
        e = d["edelta"]; idx = d["idx"]
        nz = np.any(e != 0, axis=1)
        for gi, row in zip(idx[nz], e[nz]):
            dna_map[int(gi)] = row
    print(f"  DNA non-zero variants: {len(dna_map)}", flush=True)

    z_prot_list, z_dna_list, idx_list = [], [], []
    clip = clip.to(device)
    for f in prot_files:
        d = np.load(f)
        e = d["edelta"]; idx = d["idx"]
        nz = np.any(e != 0, axis=1)
        # dual = protein non-zero AND in dna_map
        sel_idx, sel_prot, sel_dna = [], [], []
        for gi, row in zip(idx[nz], e[nz]):
            gi = int(gi)
            if gi in dna_map:
                sel_idx.append(gi); sel_prot.append(row); sel_dna.append(dna_map[gi])
        if not sel_idx:
            continue
        xp = torch.tensor(np.asarray(sel_prot), dtype=torch.float32, device=device)
        xd = torch.tensor(np.asarray(sel_dna), dtype=torch.float32, device=device)
        with torch.no_grad():
            zp = clip.enc_prot(xp).cpu().numpy()
            zd = clip.enc_dna(xd).cpu().numpy()
        z_prot_list.append(zp); z_dna_list.append(zd); idx_list.append(np.asarray(sel_idx))
        print(f"  {os.path.basename(f)}: +{len(sel_idx)} dual", flush=True)

    z_prot = np.concatenate(z_prot_list, axis=0).astype(np.float32)
    z_dna = np.concatenate(z_dna_list, axis=0).astype(np.float32)
    idx_all = np.concatenate(idx_list, axis=0)
    z_concat = np.concatenate([z_prot, z_dna], axis=1).astype(np.float32)
    print(f"  z_concat: {z_concat.shape} (prot|dna)", flush=True)
    return z_concat, idx_all


def feature_modality_attribution(sae, scaler):
    """Per-feature: protein vs DNA energy from decoder columns (dims 0:256 / 256:512)."""
    W = sae.decoder.weight.detach().cpu().numpy()  # (512, d_hidden)
    prot_norm = np.linalg.norm(W[:256, :], axis=0)   # (d_hidden,)
    dna_norm = np.linalg.norm(W[256:, :], axis=0)
    total = prot_norm + dna_norm + 1e-9
    prot_frac = prot_norm / total
    # classify
    cls = np.full(len(prot_frac), "crossmodal", dtype=object)
    cls[prot_frac > 0.70] = "protein"
    cls[prot_frac < 0.30] = "dna"
    return {"prot_norm": prot_norm, "dna_norm": dna_norm, "prot_frac": prot_frac,
            "class": cls.astype(str)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--clip_ckpt", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--out_dir", default="results/sae_crossmodal")
    ap.add_argument("--d_hidden", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # CLIP
    print("loading CLIP ...", flush=True)
    ck = torch.load(args.clip_ckpt, map_location="cpu", weights_only=False)
    clip = CrossModalCLIP(**ck["config"])
    clip.load_state_dict(ck["model_state"])
    clip = clip.eval()
    del ck

    # data
    print("loading + projecting dual-modality embeddings ...", flush=True)
    z, idx = load_dual_modality(args.emb_dir, clip, device)
    del clip
    torch.cuda.empty_cache()

    # split
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(z))
    n_val = int(len(z) * args.val_frac)
    val_i, tr_i = perm[:n_val], perm[n_val:]

    # standardize on train
    scaler = StandardScaler().fit(z[tr_i])
    z_tr = torch.tensor(scaler.transform(z[tr_i]), dtype=torch.float32)
    z_val = torch.tensor(scaler.transform(z[val_i]), dtype=torch.float32, device=device)
    print(f"  train={len(z_tr)} val={len(z_val)}", flush=True)

    # model
    sae = TopKSAE(d_in=z.shape[1], d_hidden=args.d_hidden, k=args.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    n = len(z_tr)
    best_val = float("inf")

    for ep in range(args.epochs):
        sae.train()
        ep_perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, args.batch):
            xb = z_tr[ep_perm[b:b + args.batch]].to(device)
            x_hat, h = sae(xb)
            loss = ((x_hat - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 10 == 0 or ep == args.epochs - 1:
            sae.eval()
            with torch.no_grad():
                vh, vfeat = sae(z_val)
                vloss = ((vh - z_val) ** 2).mean().item()
                var = z_val.var().item()
                fvu = vloss / (var + 1e-9)
                # alive features on val
                alive = (vfeat > 0).any(dim=0).sum().item()
            best_val = min(best_val, vloss)
            print(f"  ep {ep:3d} train_mse={tot/n:.4f} val_mse={vloss:.4f} "
                  f"FVU={fvu:.4f} alive={alive}/{args.d_hidden}", flush=True)

    # attribution
    attr = feature_modality_attribution(sae, scaler)
    cls = attr["class"]
    uniq, cnt = np.unique(cls, return_counts=True)
    print("  feature modality classes:", dict(zip(uniq.tolist(), cnt.tolist())), flush=True)

    # save
    out = os.path.join(args.out_dir, f"sae_crossmodal_k{args.k}.pt")
    torch.save({
        "model_state": sae.state_dict(),
        "config": {"d_in": int(z.shape[1]), "d_hidden": args.d_hidden, "k": args.k},
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "train_idx": idx[tr_i], "val_idx": idx[val_i],
        "attribution": attr,
        "best_val_mse": best_val,
    }, out)
    print(f"saved {out}", flush=True)
    np.savez_compressed(os.path.join(args.out_dir, "feature_attribution.npz"), **attr)
    print("done", flush=True)


if __name__ == "__main__":
    main()
