"""
Quick test: retrain SAE with larger k values on existing 40k data.
Compare coverage at k=32 (current), k=64, k=128.
"""

import os, sys, re, gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

OUT = "results/sae_largek"
os.makedirs(OUT, exist_ok=True)

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

# ── Load data ────────────────────────────────────────────────────
print("Loading CLIP encoder and genome-wide Evo2 embeddings ...")
ckpt = torch.load("results/pretrain/crossmodal_clip_40k.pt", map_location="cpu",
                   weights_only=False)
clip = CrossModalCLIP(**ckpt["config"])
clip.load_state_dict(ckpt["model_state"])
clip = clip.eval()
del ckpt

# load and encode genome-wide embeddings
sc = StandardScaler()
chunks = []
for s in range(8):
    f = f"results/variant/clinvar_evo2_emb_shard{s}.npz"
    if not os.path.exists(f):
        continue
    d = np.load(f)
    chunk = d["edelta"]
    has = np.any(chunk != 0, axis=1)
    if has.sum() == 0:
        continue
    with torch.no_grad():
        z = clip.enc_dna(torch.tensor(chunk[has], dtype=torch.float32)).numpy()
    sc.partial_fit(z)
    chunks.append((z, d["idx"][has]))
    del chunk

# collect all z_dna
all_z = []
all_idx = []
for z, idx in chunks:
    all_z.append(sc.transform(z).astype(np.float32))
    all_idx.append(idx)
del chunks

z_all = np.concatenate(all_z, axis=0)
idx_all = np.concatenate(all_idx, axis=0)
del all_z
print(f"  Training data: {z_all.shape}")

# load ClinVar labels
clinvar = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
y = clinvar["label"].astype(int).values

# load VUS
vus_acts_raw = []
for s in range(8):
    f = f"results/vus_pilot/clinvar_evo2_emb_shard{s}.npz"
    if not os.path.exists(f):
        continue
    d = np.load(f)
    chunk = d["edelta"]
    has = np.any(chunk != 0, axis=1)
    if has.sum() == 0:
        continue
    with torch.no_grad():
        z = clip.enc_dna(torch.tensor(chunk[has], dtype=torch.float32)).numpy()
    vus_acts_raw.append(sc.transform(z).astype(np.float32))
    del chunk
vus_z = np.concatenate(vus_acts_raw, axis=0)
del vus_acts_raw
del clip  # free memory
print(f"  VUS data: {vus_z.shape}")

# gene mapping for evaluation
print("Loading gene mapping ...")
clinvar_info = {}
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34 or parts[16] != "GRCh38":
            continue
        key = (parts[18], parts[31], parts[32], parts[33])
        if key not in clinvar_info:
            pchange = ""
            m = re.search(r'\(p\.([A-Za-z0-9_*=]+)\)', parts[2])
            if m:
                pchange = m.group(1)
            clinvar_info[key] = {"gene": parts[4], "pchange": pchange}

clinvar["gene"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("gene")
    for r in clinvar.itertuples()]
clinvar["pchange"] = [clinvar_info.get(
    (str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), {}).get("pchange", "")
    for r in clinvar.itertuples()]

COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A3", "COL7A1", "COL11A1", "COL11A2",
}

def is_gly_missense(pc):
    if not pc or pd.isna(pc):
        return False
    m = re.match(r'Gly\d+([A-Z][a-z]{2})', str(pc))
    return m is not None and m.group(1) != "Gly"

clinvar["is_gly_sub"] = clinvar["pchange"].apply(is_gly_missense)
clinvar["is_collagen"] = clinvar["gene"].isin(COLLAGEN_GENES)

def get_to_aa(pc):
    if not pc or pd.isna(pc):
        return None
    m = re.match(r'Gly\d+([A-Z][a-z]{2})', str(pc))
    return m.group(1) if m and m.group(1) != "Gly" else None

clinvar["gly_to_aa"] = clinvar["pchange"].apply(get_to_aa)

col_gly_path = clinvar[(y == 1) & clinvar["is_collagen"] & clinvar["gly_to_aa"].notna()]
acidic_mask = col_gly_path["gly_to_aa"].isin(["Asp", "Glu"])
basic_mask = col_gly_path["gly_to_aa"].isin(["Arg", "Ser"])

# VUS info
vus_df = pd.read_csv("results/vus_pilot/vus_mechanism_profiles.csv")
gly_vus = vus_df[vus_df["is_gly_sub"]]

# ── Train SAE at different k values (GPU) ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
zt = torch.tensor(z_all).to(device)
vus_zt = torch.tensor(vus_z).to(device)

for K in [32, 64, 128]:
    print(f"\n{'='*70}")
    print(f"TRAINING SAE: k={K}, d_hidden=2048")
    print(f"{'='*70}")

    sae = TopKSAE(d_in=256, d_hidden=2048, k=K).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for ep in range(1, 1001):
        sae.train()
        x_hat, h_sp = sae(zt)
        loss = F.mse_loss(x_hat, zt) + 0.01 * h_sp.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 500 == 0:
            print(f"  ep {ep}: loss={loss.item():.4f}")

    # get activations
    sae.eval()
    with torch.no_grad():
        _, h = sae(zt)
        acts = h.cpu().numpy()
        _, h_vus = sae(vus_zt)
        vus_acts = h_vus.cpu().numpy()

    # ── Analyze features ──
    n_alive = (np.abs(acts).max(axis=0) > 1e-6).sum()
    print(f"\n  Alive features: {n_alive}/2048")

    # find novel pathogenic features (same method as before)
    cards_data = []
    for fi in range(2048):
        a = np.abs(acts[:, fi])
        active = a > 1e-6
        if active.sum() < 5:
            continue
        pr = y[active].mean()
        cards_data.append({"feature": fi, "n_active": int(active.sum()), "path_rate": pr})

    cards = pd.DataFrame(cards_data)
    novel_feats = cards[cards["path_rate"] > 0.85]

    # check collagen Gly-sub enrichment for each
    gly_specific = []
    base_rate = (clinvar["is_collagen"] & clinvar["is_gly_sub"] & (y == 1)).sum() / max((y == 1).sum(), 1)

    for _, r in novel_feats.iterrows():
        fi = int(r["feature"])
        a = np.abs(acts[:, fi])
        active = a > 1e-6
        n_target = (active & clinvar["is_collagen"].values & clinvar["is_gly_sub"].values & (y == 1)).sum()
        target_rate = n_target / max(active.sum(), 1)
        enrichment = target_rate / max(base_rate, 1e-8)

        if enrichment >= 3.0:
            gly_specific.append(fi)

    print(f"  Novel pathogenic features (pr>0.85): {len(novel_feats)}")
    print(f"  Gly-specific (enrichment≥3x): {len(gly_specific)}")

    if not gly_specific:
        print(f"  No Gly-specific features found, skipping...")
        continue

    # coverage on known pathogenic Gly-subs
    gly_score = np.sum(np.abs(acts[:, gly_specific]), axis=1)
    clinvar[f"gly_score_k{K}"] = gly_score

    path_col_gly = col_gly_path.index
    coverage = (gly_score[path_col_gly] > 0).mean()
    print(f"\n  Coverage on known pathogenic collagen Gly-subs: {coverage:.1%}")

    # AA-specific coverage
    for aa_name, aa_list in [("Asp/Glu", ["Asp", "Glu"]), ("Arg/Ser", ["Arg", "Ser"])]:
        mask = col_gly_path["gly_to_aa"].isin(aa_list)
        sub_idx = col_gly_path[mask].index
        cov = (gly_score[sub_idx] > 0).mean()
        print(f"    Gly→{aa_name}: {cov:.1%} coverage")

    # check mechanism separation (acid vs basic features)
    # for each gly-specific feature, check acid vs basic preference
    acid_feats = []
    basic_feats = []
    for fi in gly_specific:
        a = np.abs(acts[:, fi])
        active = a > 1e-6
        acid_act = (active[col_gly_path[acidic_mask].index]).mean() if acidic_mask.sum() > 0 else 0
        basic_act = (active[col_gly_path[basic_mask].index]).mean() if basic_mask.sum() > 0 else 0
        if acid_act > basic_act * 2:
            acid_feats.append(fi)
        elif basic_act > acid_act * 2:
            basic_feats.append(fi)

    print(f"\n  Mechanism features: {len(acid_feats)} acid-type, {len(basic_feats)} basic-type")

    # overall mechanism separation
    if acid_feats and basic_feats:
        acid_score = np.sum(np.abs(acts[:, acid_feats]), axis=1)
        basic_score = np.sum(np.abs(acts[:, basic_feats]), axis=1)

        acid_on_acid = (acid_score[col_gly_path[acidic_mask].index] > 0).mean()
        acid_on_basic = (acid_score[col_gly_path[basic_mask].index] > 0).mean()
        basic_on_basic = (basic_score[col_gly_path[basic_mask].index] > 0).mean()
        basic_on_acid = (basic_score[col_gly_path[acidic_mask].index] > 0).mean()

        print(f"    Acid features on Asp/Glu: {acid_on_acid:.1%}, on Arg/Ser: {acid_on_basic:.1%}")
        print(f"    Basic features on Arg/Ser: {basic_on_basic:.1%}, on Asp/Glu: {basic_on_acid:.1%}")

    # VUS coverage
    vus_gly_score = np.sum(np.abs(vus_acts[:, gly_specific]), axis=1)
    vus_gly_idx = gly_vus.index
    vus_coverage = (vus_gly_score[vus_gly_idx] > 0).mean()
    n_annotated = (vus_gly_score[vus_gly_idx] > 0).sum()
    print(f"\n  VUS coverage: {n_annotated}/{len(gly_vus)} ({vus_coverage:.1%})")

    # save checkpoint
    torch.save(sae.state_dict(), os.path.join(OUT, f"sae_k{K}.pt"))
    np.savez_compressed(os.path.join(OUT, f"acts_k{K}.npz"), acts=acts)

print(f"\nSaved to {OUT}/")
