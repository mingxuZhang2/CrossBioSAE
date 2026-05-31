"""
Cross-modal SAE with BOTH protein and DNA modalities.

Pipeline:
1. Map ClinVar variants → gene symbols (via variant_summary.txt.gz)
2. Match genes with protein_activations.h5 → get ESM-2 protein embeddings
3. Project both through pretrained CLIP encoders → z_prot (256-d), z_dna (256-d)
4. Train SAE on z_concat (512-d) and decompose features by modality
5. Analyze: what's protein-driven, DNA-driven, emergent from fusion?

Key insight: z_concat SAE features can be linearly decomposed:
  h_i = W_enc[i] · z_concat = W_enc[i, :256] · z_prot + W_enc[i, 256:] · z_dna
  → protein_contrib vs dna_contrib per feature per variant
"""

import gzip, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

OUT = "results/sae_crossmodal"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Map ClinVar variants to gene symbols
# ═══════════════════════════════════════════════════════════════════
print("Step 1: Mapping ClinVar variants to genes ...")

df = pd.read_parquet("data/variant/clinvar.parquet").reset_index(drop=True)
lookup_keys = set(zip(df["chrom"].astype(str), df["pos"].astype(str),
                       df["ref"].astype(str), df["alt"].astype(str)))
print(f"  ClinVar variants to map: {len(lookup_keys)}")

gene_map = {}
var_type_map = {}
with gzip.open("data/variant/variant_summary.txt.gz", "rt") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 34:
            continue
        if parts[16] != "GRCh38":
            continue
        chrom = parts[18]
        pos = parts[31]
        ref = parts[32]
        alt = parts[33]
        key = (chrom, pos, ref, alt)
        if key in lookup_keys and key not in gene_map:
            gene_map[key] = parts[4]
            var_type_map[key] = parts[1]

df["gene"] = [gene_map.get((str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), None)
              for r in df.itertuples()]
df["var_type"] = [var_type_map.get((str(r.chrom), str(r.pos), str(r.ref), str(r.alt)), None)
                  for r in df.itertuples()]

n_mapped = df["gene"].notna().sum()
print(f"  Mapped: {n_mapped}/{len(df)} ({100*n_mapped/len(df):.1f}%)")
print(f"  Unique genes: {df['gene'].nunique()}")
print(f"  Variant types: {df['var_type'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Match with protein embeddings
# ═══════════════════════════════════════════════════════════════════
print("\nStep 2: Loading protein embeddings ...")
import h5py
h = h5py.File("data/full/protein_activations.h5", "r")
prot_names = [n.decode() if isinstance(n, bytes) else n for n in h["gene_names"][:]]
prot_acts = h["activations"][:]  # (16620, 1280)
h.close()

prot_lookup = {n: i for i, n in enumerate(prot_names)}
print(f"  Protein h5: {len(prot_names)} genes, dim={prot_acts.shape[1]}")

# match ClinVar genes with protein h5
df["prot_idx"] = df["gene"].map(lambda g: prot_lookup.get(g, -1) if pd.notna(g) else -1)
has_prot = df["prot_idx"] >= 0
print(f"  ClinVar variants with protein embedding: {has_prot.sum()}/{len(df)} ({100*has_prot.sum()/len(df):.1f}%)")
print(f"  Unique genes with protein embedding: {df.loc[has_prot, 'gene'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Load Evo2 embedding deltas and pretrained model
# ═══════════════════════════════════════════════════════════════════
print("\nStep 3: Loading Evo2 embeddings + pretrained CLIP ...")

# merge Evo2 embedding shards
edelta = np.zeros((len(df), 4096), dtype=np.float32)
for s in range(8):
    f = f"results/variant/clinvar_evo2_emb_shard{s}.npz"
    if os.path.exists(f):
        d = np.load(f)
        edelta[d["idx"]] = d["edelta"]
has_evo2 = np.any(edelta != 0, axis=1)
print(f"  Evo2 embeddings: {has_evo2.sum()}/{len(df)}")

# both modalities available
both_ok = has_prot.values & has_evo2
n_both = both_ok.sum()
print(f"  Both protein + DNA available: {n_both} variants")

# load pretrained CLIP
ckpt = torch.load("results/pretrain/crossmodal_clip_40k.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
clip = CrossModalCLIP(**cfg)
clip.load_state_dict(ckpt["model_state"])
clip = clip.eval()
print(f"  CLIP: d_prot={cfg['d_prot']}, d_dna={cfg['d_dna']}, d_shared={cfg['d_shared']}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Project through pretrained encoders
# ═══════════════════════════════════════════════════════════════════
print("\nStep 4: Projecting through encoders ...")

df_both = df[both_ok].reset_index(drop=True)
y = df_both["label"].astype(int).values
is_coding = df_both["ESM-1b"].notna().values

# protein embeddings: one per gene, same for all variants in that gene
prot_idx = df_both["prot_idx"].values
xp = prot_acts[prot_idx]  # (N, 1280)

# DNA embeddings: one per variant
xd = edelta[both_ok]  # (N, 4096)

with torch.no_grad():
    z_prot = clip.enc_prot(torch.tensor(xp, dtype=torch.float32)).numpy()  # (N, 256)
    z_dna = clip.enc_dna(torch.tensor(xd, dtype=torch.float32)).numpy()  # (N, 256)

print(f"  z_prot: {z_prot.shape}, z_dna: {z_dna.shape}")
print(f"  Labels: {y.sum()} pathogenic, {(1-y).sum()} benign")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Representation analysis
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 1: Pretrained representation properties")
print(f"{'='*70}")

# cosine similarity between z_prot and z_dna
cos_sim = np.sum(z_prot * z_dna, axis=1) / (
    np.linalg.norm(z_prot, axis=1) * np.linalg.norm(z_dna, axis=1) + 1e-8)
print(f"\n  Cross-modal cosine similarity:")
print(f"    All: {cos_sim.mean():.4f} ± {cos_sim.std():.4f}")
print(f"    Pathogenic: {cos_sim[y==1].mean():.4f} ± {cos_sim[y==1].std():.4f}")
print(f"    Benign: {cos_sim[y==0].mean():.4f} ± {cos_sim[y==0].std():.4f}")
auc_cos = roc_auc_score(y, -cos_sim)
auc_cos = max(auc_cos, 1 - auc_cos)
print(f"    AUC for pathogenicity: {auc_cos:.4f}")

# per-modality AUC using simple logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

print(f"\n  Per-modality predictive power (5-fold CV AUC):")
for name, z in [("z_prot (protein)", z_prot), ("z_dna (DNA)", z_dna),
                ("z_concat (fusion)", np.concatenate([z_prot, z_dna], axis=1)),
                ("z_mean (avg fusion)", (z_prot + z_dna) / 2)]:
    sc = StandardScaler().fit(z)
    z_s = sc.transform(z)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(lr, z_s, y, cv=5, scoring="roc_auc")
    print(f"    {name:25s}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Train SAE on z_concat
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 2: Cross-modal SAE on z_concat (512-d)")
print(f"{'='*70}")

class TopKSAE(nn.Module):
    def __init__(self, d_in, d_hidden=2048, k=32):
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

z_concat = np.concatenate([z_prot, z_dna], axis=1)  # (N, 512)
sc = StandardScaler().fit(z_concat)
z_s = sc.transform(z_concat).astype(np.float32)
z_t = torch.tensor(z_s)

sae = TopKSAE(d_in=512, d_hidden=2048, k=32)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
for ep in range(1, 1001):
    sae.train()
    x_hat, h_sparse = sae(z_t)
    recon = F.mse_loss(x_hat, z_t)
    l1 = h_sparse.abs().mean() * 0.01
    loss = recon + l1
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 200 == 0:
        print(f"  ep {ep}: loss={loss.item():.4f}")

sae.eval()
with torch.no_grad():
    _, h_sparse = sae(z_t)
    acts = h_sparse.numpy()

alive = (np.abs(acts) > 1e-6).sum(axis=0)
n_alive = (alive > 0).sum()
print(f"\n  Alive features: {n_alive}/2048")

# ═══════════════════════════════════════════════════════════════════
# STEP 7: Modality decomposition per feature
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 3: Per-feature modality decomposition")
print(f"{'='*70}")

W_enc = sae.encoder.weight.detach().numpy()  # (2048, 512)
W_prot = W_enc[:, :256]  # protein part
W_dna = W_enc[:, 256:]   # DNA part

# for each feature: compute contribution from protein vs DNA
# h_i = W_enc[i] · z_s = W_prot[i] · z_s[:256] + W_dna[i] · z_s[256:]
z_prot_s = z_s[:, :256]  # (N, 256) standardized protein
z_dna_s = z_s[:, 256:]   # (N, 256) standardized DNA

feature_cards = []
score_cols = ["ESM-1b", "GPN-MSA", "CADD", "phyloP-100v", "NT"]

for fi in range(2048):
    if alive[fi] < 10:
        continue

    a = acts[:, fi]
    active = np.abs(a) > 1e-6
    n_act = active.sum()

    # modality decomposition
    prot_act = z_prot_s[active] @ W_prot[fi]  # (n_act,)
    dna_act = z_dna_s[active] @ W_dna[fi]     # (n_act,)

    prot_abs = np.abs(prot_act).mean()
    dna_abs = np.abs(dna_act).mean()
    total = prot_abs + dna_abs + 1e-8
    prot_frac = prot_abs / total
    dna_frac = dna_abs / total

    # pathogenicity
    path_rate = y[active].mean()
    coding_frac = is_coding[active].mean()

    # variant type distribution
    vt = df_both["var_type"].values[active]
    vt_counts = pd.Series(vt).value_counts()

    # gene diversity
    genes = df_both["gene"].values[active]
    n_genes = len(set(g for g in genes if pd.notna(g)))

    # correlation with scores
    corrs = {}
    for col in score_cols:
        vals = df_both[col].values[active].astype(float)
        ok = np.isfinite(vals)
        if ok.sum() > 20:
            rho, _ = stats.spearmanr(a[active][ok], vals[ok])
            corrs[col] = rho
        else:
            corrs[col] = np.nan

    card = {
        "feature": fi, "n_active": n_act,
        "prot_frac": prot_frac, "dna_frac": dna_frac,
        "path_rate": path_rate, "coding_frac": coding_frac,
        "n_genes": n_genes,
        "top_var_type": vt_counts.index[0] if len(vt_counts) > 0 else "unknown",
        "missense_frac": (vt == "single nucleotide variant").mean() if len(vt) > 0 else 0,
    }
    for col in score_cols:
        card[f"r_{col}"] = corrs.get(col, np.nan)

    # modality label
    if prot_frac > 0.65:
        card["modality"] = "protein-dominated"
    elif dna_frac > 0.65:
        card["modality"] = "dna-dominated"
    elif abs(prot_frac - 0.5) < 0.15:
        card["modality"] = "balanced"
    else:
        card["modality"] = "mixed"

    feature_cards.append(card)

fc = pd.DataFrame(feature_cards)

# modality distribution
print(f"\n  Feature modality distribution ({len(fc)} alive with n≥10):")
for mod, grp in fc.groupby("modality"):
    print(f"    {mod:20s}: {len(grp):4d} feats, "
          f"prot_frac={grp['prot_frac'].mean():.3f}, "
          f"path_rate={grp['path_rate'].mean():.3f}, "
          f"n_genes={grp['n_genes'].mean():.0f}")

# top protein-dominated features
print(f"\n  Top 10 protein-dominated features:")
print(f"  {'feat':>5s} {'n':>6s} {'prot%':>6s} {'dna%':>5s} {'path%':>6s} {'cod%':>5s} "
      f"{'rESM':>6s} {'rGPN':>6s} {'rNT':>5s} {'genes':>5s}")
for _, r in fc.sort_values("prot_frac", ascending=False).head(10).iterrows():
    print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['prot_frac']:6.3f} {r['dna_frac']:5.3f} "
          f"{r['path_rate']:6.3f} {r['coding_frac']:5.2f} "
          f"{r.get('r_ESM-1b',0):+6.3f} {r.get('r_GPN-MSA',0):+6.3f} "
          f"{r.get('r_NT',0):+5.3f} {int(r['n_genes']):5d}")

print(f"\n  Top 10 DNA-dominated features:")
for _, r in fc.sort_values("dna_frac", ascending=False).head(10).iterrows():
    print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['prot_frac']:6.3f} {r['dna_frac']:5.3f} "
          f"{r['path_rate']:6.3f} {r['coding_frac']:5.2f} "
          f"{r.get('r_ESM-1b',0):+6.3f} {r.get('r_GPN-MSA',0):+6.3f} "
          f"{r.get('r_NT',0):+5.3f} {int(r['n_genes']):5d}")

print(f"\n  Top 10 balanced features:")
for _, r in fc[fc['modality']=='balanced'].sort_values("n_active", ascending=False).head(10).iterrows():
    print(f"  {int(r['feature']):5d} {int(r['n_active']):6d} {r['prot_frac']:6.3f} {r['dna_frac']:5.3f} "
          f"{r['path_rate']:6.3f} {r['coding_frac']:5.2f} "
          f"{r.get('r_ESM-1b',0):+6.3f} {r.get('r_GPN-MSA',0):+6.3f} "
          f"{r.get('r_NT',0):+5.3f} {int(r['n_genes']):5d}")

# ═══════════════════════════════════════════════════════════════════
# STEP 8: Pathogenicity analysis by modality
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 4: Pathogenic features — which modality drives them?")
print(f"{'='*70}")

path_feats = fc[fc["path_rate"] > 0.8]
ben_feats = fc[fc["path_rate"] < 0.15]
print(f"\n  Pathogenic features (pr > 0.8): {len(path_feats)}")
print(f"    Protein-dominated: {(path_feats['modality']=='protein-dominated').sum()}")
print(f"    DNA-dominated: {(path_feats['modality']=='dna-dominated').sum()}")
print(f"    Balanced: {(path_feats['modality']=='balanced').sum()}")
print(f"    Mixed: {(path_feats['modality']=='mixed').sum()}")
print(f"    Mean prot_frac: {path_feats['prot_frac'].mean():.3f}")

print(f"\n  Benign features (pr < 0.15): {len(ben_feats)}")
print(f"    Protein-dominated: {(ben_feats['modality']=='protein-dominated').sum()}")
print(f"    DNA-dominated: {(ben_feats['modality']=='dna-dominated').sum()}")
print(f"    Balanced: {(ben_feats['modality']=='balanced').sum()}")
print(f"    Mixed: {(ben_feats['modality']=='mixed').sum()}")
print(f"    Mean prot_frac: {ben_feats['prot_frac'].mean():.3f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 9: Gene-level analysis — which genes have most cross-modal signal?
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 5: Per-gene cross-modal alignment")
print(f"{'='*70}")

gene_stats = []
for gene, grp in df_both.groupby("gene"):
    if len(grp) < 10:
        continue
    idx = grp.index.values
    c = cos_sim[idx]
    gene_stats.append({
        "gene": gene, "n_variants": len(grp),
        "path_rate": y[idx].mean(),
        "mean_cos": c.mean(), "std_cos": c.std(),
    })

gs = pd.DataFrame(gene_stats).sort_values("mean_cos")
print(f"\n  Genes with LOWEST cross-modal alignment (z_prot ≠ z_dna):")
for _, r in gs.head(10).iterrows():
    print(f"    {r['gene']:15s} n={int(r['n_variants']):5d} "
          f"cos={r['mean_cos']:.3f}±{r['std_cos']:.3f} path={r['path_rate']:.2f}")

print(f"\n  Genes with HIGHEST cross-modal alignment (z_prot ≈ z_dna):")
for _, r in gs.tail(10).iterrows():
    print(f"    {r['gene']:15s} n={int(r['n_variants']):5d} "
          f"cos={r['mean_cos']:.3f}±{r['std_cos']:.3f} path={r['path_rate']:.2f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 10: Variant type breakdown
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS 6: Modality contribution by variant type")
print(f"{'='*70}")

# use per-variant gate-like decomposition from z_concat
# for each variant, compute how much of z_concat norm comes from protein vs DNA
prot_norms = np.linalg.norm(z_prot_s, axis=1)
dna_norms = np.linalg.norm(z_dna_s, axis=1)
prot_share = prot_norms / (prot_norms + dna_norms + 1e-8)

for vt in df_both["var_type"].dropna().unique():
    mask = df_both["var_type"].values == vt
    if mask.sum() < 20:
        continue
    print(f"  {vt:35s}: n={mask.sum():6d}, "
          f"prot_share={prot_share[mask].mean():.3f}, "
          f"cos_sim={cos_sim[mask].mean():.3f}, "
          f"path_rate={y[mask].mean():.3f}")

# save results
fc.to_csv(os.path.join(OUT, "crossmodal_sae_cards.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "crossmodal_sae_acts.npz"), acts=acts,
                    z_prot=z_prot, z_dna=z_dna, cos_sim=cos_sim)
df_both[["gene", "var_type"]].to_csv(os.path.join(OUT, "variant_gene_mapping.csv"), index=False)
print(f"\nSaved to {OUT}/")
