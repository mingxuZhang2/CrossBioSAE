"""
Cross-modal SAE interpretability analysis.

Core paper questions:
1. Do cross-modal features discriminate pathogenic/benign BETTER than single-modality?
2. What biological concepts do the top features encode?
3. Can we assign mechanism labels (protein-damage, DNA-conservation, cross-modal) to VUS?

Loads the trained SAE + CLIP, runs all 260k dual-modality variants through,
computes per-feature activation stats, pathogenicity association, and biological
characterization of top discriminating features per modality class.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP
from sklearn.preprocessing import StandardScaler

OUT = "results/sae_crossmodal/interpretability"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Load everything ──────────────────────────────────────────
print("=== Loading models + data ===", flush=True)

# CLIP
ck = torch.load("results/pretrain/crossmodal_clip_40k.pt", map_location="cpu",
                 weights_only=False)
clip = CrossModalCLIP(**ck["config"]).eval().to(device)
clip.load_state_dict(ck["model_state"])
del ck

# SAE
sae_ck = torch.load("results/sae_crossmodal/sae_crossmodal_k32.pt",
                     map_location="cpu", weights_only=False)
class TopKSAE(torch.nn.Module):
    def __init__(self, d_in=512, d_hidden=4096, k=32):
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, d_hidden)
        self.decoder = torch.nn.Linear(d_hidden, d_in)
        self.k = k
    def encode(self, x):
        h = self.encoder(x)
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask
    def forward(self, x):
        h_sparse = self.encode(x)
        return self.decoder(h_sparse), h_sparse

sae = TopKSAE(**sae_ck["config"]).eval().to(device)
sae.load_state_dict(sae_ck["model_state"])
scaler = StandardScaler()
scaler.mean_ = sae_ck["scaler_mean"]
scaler.scale_ = sae_ck["scaler_scale"]
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)
attr = sae_ck["attribution"]
feat_class = attr["class"]  # per-feature: 'protein', 'dna', 'crossmodal'
prot_frac = attr["prot_frac"]

# variant annotations
df = pd.read_csv("data/variant/sae_pretrain/missense_500k_annotated.csv", low_memory=False)
dual_idx = np.load("data/variant/sae_pretrain/dual_idx.npy")
sub = df.iloc[dual_idx].reset_index(drop=True)

# labels
is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
          ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
         ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
is_vus = sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
y = np.full(len(sub), -1, dtype=int)
y[is_path.values] = 1
y[is_ben.values] = 0


# ── 2. Run all variants through CLIP + SAE ──────────────────────
print("=== Encoding 260k variants ===", flush=True)
import glob

# load edelta -> CLIP -> z_concat -> StandardScaler -> SAE
# protein
prot_edelta = np.zeros((len(sub), 1280), dtype=np.float32)
for f in sorted(glob.glob("results/sae_pretrain_emb/esm2_emb_shard*.npz")):
    d = np.load(f)
    # map global idx to sub idx
    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    for gi, row in zip(d["idx"], d["edelta"]):
        gi = int(gi)
        if gi in g2l:
            prot_edelta[g2l[gi]] = row

# dna
dna_edelta = np.zeros((len(sub), 4096), dtype=np.float32)
for f in sorted(glob.glob("results/sae_pretrain_emb/clinvar_evo2_emb_shard*.npz")):
    d = np.load(f)
    for gi, row in zip(d["idx"], d["edelta"]):
        gi = int(gi)
        if gi in g2l:
            dna_edelta[g2l[gi]] = row

print(f"  prot non-zero: {np.any(prot_edelta!=0,axis=1).sum()}", flush=True)
print(f"  dna non-zero: {np.any(dna_edelta!=0,axis=1).sum()}", flush=True)

# project through CLIP in batches
BS = 8192
z_all = np.zeros((len(sub), 512), dtype=np.float32)
for i in range(0, len(sub), BS):
    xp = torch.tensor(prot_edelta[i:i+BS], dtype=torch.float32, device=device)
    xd = torch.tensor(dna_edelta[i:i+BS], dtype=torch.float32, device=device)
    with torch.no_grad():
        zp = clip.enc_prot(xp).cpu().numpy()
        zd = clip.enc_dna(xd).cpu().numpy()
    z_all[i:i+BS] = np.concatenate([zp, zd], axis=1)
    if i % 50000 == 0:
        print(f"  projected {i+len(xp)}/{len(sub)}", flush=True)

del prot_edelta, dna_edelta, clip
torch.cuda.empty_cache()

# standardize + SAE encode
z_s = scaler.transform(z_all).astype(np.float32)
del z_all

acts = np.zeros((len(sub), sae_ck["config"]["d_hidden"]), dtype=np.float32)
for i in range(0, len(sub), BS):
    xb = torch.tensor(z_s[i:i+BS], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, h = sae(xb)
    acts[i:i+BS] = h.cpu().numpy()
    if i % 50000 == 0:
        print(f"  SAE encoded {i+len(xb)}/{len(sub)}", flush=True)

del z_s
print(f"  activations: {acts.shape}, mean L0={int((acts>0).sum(1).mean())}", flush=True)


# ── 3. Per-feature pathogenicity discrimination ─────────────────
print("\n=== Feature pathogenicity analysis ===", flush=True)
labeled = y >= 0
al = acts[labeled]
yl = y[labeled]

n_feat = acts.shape[1]
feat_stats = []
for fi in range(n_feat):
    a = al[:, fi]
    active = a > 0
    if active.sum() < 10:
        feat_stats.append({"feat": fi, "auc": 0.5, "mean_path": 0, "mean_ben": 0,
                          "freq": 0, "modality": feat_class[fi]})
        continue
    if len(set(yl[active.values if hasattr(active, 'values') else active])) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(yl, a)
    feat_stats.append({
        "feat": fi,
        "auc": auc,
        "mean_path": float(a[yl == 1].mean()),
        "mean_ben": float(a[yl == 0].mean()),
        "freq": float(active.mean()),
        "modality": feat_class[fi],
    })

fs = pd.DataFrame(feat_stats)
fs["auc_abs"] = (fs["auc"] - 0.5).abs()
fs.to_csv(os.path.join(OUT, "feature_pathogenicity.csv"), index=False)

# summary by modality class
print("\n  Per-modality pathogenicity AUC (top-50 features per class):", flush=True)
for mc in ["protein", "dna", "crossmodal"]:
    mf = fs[fs["modality"] == mc].nlargest(50, "auc_abs")
    print(f"  {mc:12s}: median_AUC={mf['auc'].median():.3f}  "
          f"mean_AUC={mf['auc'].mean():.3f}  "
          f"max_AUC={mf['auc'].max():.3f}  "
          f"n_feat={len(fs[fs['modality']==mc])}", flush=True)

# overall: aggregate discriminating features
top_feats = fs.nlargest(100, "auc_abs")
mc_top = top_feats["modality"].value_counts()
print(f"\n  Top-100 discriminating features by modality:", flush=True)
for mc, n in mc_top.items():
    print(f"    {mc}: {n}", flush=True)


# ── 4. Biological characterization of top features ──────────────
print("\n=== Biological characterization ===", flush=True)

def characterize_feature(fi, acts, sub, y, topn_genes=5, topn_aa=5):
    """Deep characterization of a single feature."""
    a = acts[:, fi]
    active = a > 0
    active_sub = sub[active]
    if len(active_sub) < 5:
        return None

    # gene enrichment
    genes = active_sub["gene"].value_counts()
    bg_genes = sub["gene"].value_counts()
    gene_enrich = []
    for g in genes.index[:topn_genes]:
        fg = genes[g] / len(active_sub)
        bg = bg_genes.get(g, 0) / len(sub)
        enrich = fg / max(bg, 1e-6)
        gene_enrich.append({"gene": g, "count": int(genes[g]), "freq": fg, "enrich": enrich})

    # AA substitution pattern
    aa_changes = active_sub.apply(lambda r: f"{r['from_aa']}>{r['to_aa']}", axis=1)
    aa_top = aa_changes.value_counts().head(topn_aa)

    # from_aa distribution
    from_aa = active_sub["from_aa"].value_counts()

    # pathogenicity if labeled
    active_np = active.values if hasattr(active, 'values') else active
    labeled_mask = y[active_np] >= 0
    if labeled_mask.sum() > 10:
        path_rate = y[active_np][labeled_mask].mean()
    else:
        path_rate = None

    return {
        "feat": fi,
        "n_active": int(active.sum()),
        "freq": float(active.mean()),
        "modality": feat_class[fi],
        "prot_frac": float(prot_frac[fi]),
        "path_rate": path_rate,
        "top_genes": gene_enrich,
        "top_aa": aa_top.to_dict(),
        "from_aa_dist": from_aa.head(5).to_dict(),
    }

# characterize top-20 per modality class
all_cards = []
for mc in ["protein", "dna", "crossmodal"]:
    mf = fs[fs["modality"] == mc].nlargest(20, "auc_abs")
    print(f"\n--- Top-20 {mc} features ---", flush=True)
    for _, row in mf.iterrows():
        fi = int(row["feat"])
        card = characterize_feature(fi, acts, sub, y)
        if card is None:
            continue
        all_cards.append(card)
        genes_str = ", ".join(f"{g['gene']}({g['count']},x{g['enrich']:.1f})" for g in card["top_genes"][:3])
        aa_str = ", ".join(f"{k}:{v}" for k, v in list(card["top_aa"].items())[:3])
        pr = f"{card['path_rate']:.2f}" if card['path_rate'] is not None else "N/A"
        print(f"  feat {fi:4d} | AUC={row['auc']:.3f} | n={card['n_active']:5d} | "
              f"path={pr} | pf={card['prot_frac']:.2f} | "
              f"genes=[{genes_str}] | aa=[{aa_str}]", flush=True)

# save feature cards
import json
with open(os.path.join(OUT, "feature_cards.json"), "w") as f:
    json.dump(all_cards, f, indent=2, default=str)


# ── 5. Cross-modal vs single-modality comparison ────────────────
print("\n=== Cross-modal advantage analysis ===", flush=True)

# For each modality class, compute aggregate pathogenicity score
# (sum of top-k feature activations) and compare AUC
for mc in ["protein", "dna", "crossmodal"]:
    mc_feats = fs[(fs["modality"] == mc) & (fs["auc_abs"] > 0.02)].nlargest(50, "auc_abs")["feat"].values
    if len(mc_feats) == 0:
        continue
    # direction: if AUC > 0.5, feature fires more on pathogenic
    score = np.zeros(len(al))
    for fi in mc_feats:
        fi = int(fi)
        a = al[:, fi]
        auc_i = fs.loc[fs["feat"] == fi, "auc"].values[0]
        sign = 1 if auc_i > 0.5 else -1
        score += sign * a
    auc = roc_auc_score(yl, score)
    print(f"  {mc:12s} aggregate (top-{len(mc_feats):2d} feats): AUC={auc:.4f}", flush=True)

# all features combined
all_top = fs.nlargest(100, "auc_abs")["feat"].values
score_all = np.zeros(len(al))
for fi in all_top:
    fi = int(fi)
    a = al[:, fi]
    auc_i = fs.loc[fs["feat"] == fi, "auc"].values[0]
    sign = 1 if auc_i > 0.5 else -1
    score_all += sign * a
auc_all = roc_auc_score(yl, score_all)
print(f"  {'all-combined':12s} aggregate (top-100 feats): AUC={auc_all:.4f}", flush=True)


# ── 6. VUS mechanism profiling ──────────────────────────────────
print("\n=== VUS mechanism profiling ===", flush=True)
vus_mask = is_vus.values
vus_acts = acts[vus_mask]
vus_sub = sub[vus_mask].reset_index(drop=True)

# per VUS: dominant modality signal
vus_profiles = []
for i in range(len(vus_sub)):
    a = vus_acts[i]
    active_feats = np.where(a > 0)[0]
    if len(active_feats) == 0:
        continue
    # modality breakdown of active features
    mc_counts = Counter(feat_class[active_feats])
    total_act = a[active_feats].sum()
    # weighted modality energy
    prot_energy = sum(a[fi] for fi in active_feats if feat_class[fi] == "protein")
    dna_energy = sum(a[fi] for fi in active_feats if feat_class[fi] == "dna")
    xm_energy = sum(a[fi] for fi in active_feats if feat_class[fi] == "crossmodal")

    dominant = "crossmodal"
    if prot_energy > dna_energy and prot_energy > xm_energy:
        dominant = "protein"
    elif dna_energy > prot_energy and dna_energy > xm_energy:
        dominant = "dna"

    vus_profiles.append({
        "gene": vus_sub.iloc[i]["gene"],
        "from_aa": vus_sub.iloc[i]["from_aa"],
        "to_aa": vus_sub.iloc[i]["to_aa"],
        "prot_pos": int(vus_sub.iloc[i]["prot_pos"]),
        "n_active": len(active_feats),
        "prot_energy": float(prot_energy),
        "dna_energy": float(dna_energy),
        "xm_energy": float(xm_energy),
        "dominant_mechanism": dominant,
    })

vp = pd.DataFrame(vus_profiles)
vp.to_csv(os.path.join(OUT, "vus_mechanism_profiles.csv"), index=False)

mc_dist = vp["dominant_mechanism"].value_counts()
print(f"  VUS mechanism distribution ({len(vp)} variants):", flush=True)
for mc, n in mc_dist.items():
    print(f"    {mc}: {n} ({100*n/len(vp):.1f}%)", flush=True)

# per-gene mechanism signature
print(f"\n  Top-20 genes by VUS count + mechanism:", flush=True)
gene_mechs = vp.groupby("gene")["dominant_mechanism"].value_counts().unstack(fill_value=0)
gene_counts = vp["gene"].value_counts()
for g in gene_counts.index[:20]:
    row = gene_mechs.loc[g] if g in gene_mechs.index else {}
    p = row.get("protein", 0)
    d = row.get("dna", 0)
    x = row.get("crossmodal", 0)
    t = p + d + x
    dom = max([(p, "prot"), (d, "dna"), (x, "xm")], key=lambda x: x[0])[1]
    print(f"    {str(g):10s} n={t:4d}  prot={p:3d} dna={d:3d} xm={x:3d}  dominant={dom}",
          flush=True)


# ── 7. Save activation matrix (compressed) ─────────────────────
print("\n=== Saving activations ===", flush=True)
np.savez_compressed(os.path.join(OUT, "sae_acts_260k.npz"),
                    acts=acts.astype(np.float16),
                    dual_idx=dual_idx,
                    y=y)
print(f"  saved {acts.shape} activations", flush=True)

print("\n=== DONE ===", flush=True)
