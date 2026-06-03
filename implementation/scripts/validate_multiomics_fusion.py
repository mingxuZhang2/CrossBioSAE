"""
Genome-wide validation: multi-omics fusion > single modality.

Compares protein-LM (ESM-2) vs DNA-LM (Evo2) vs fusion for variant
pathogenicity prediction across all ClinVar genes.

Methods:
  1. ESM2-norm:  ||edelta_prot||  (1-d)
  2. Evo2-norm:  ||edelta_dna||   (1-d)
  3. Both-norms: [||prot||, ||dna||] (2-d)
  4. CLIP-prot:  z_prot  (256-d, L2-normalized by CLIP encoder)
  5. CLIP-dna:   z_dna   (256-d)
  6. CLIP-fusion: [z_prot | z_dna]  (512-d)

Evaluation:
  - 5-fold stratified CV, genome-wide AUC
  - Per-gene AUC (genes with >=20 pathogenic + >=20 benign)
  - Paired Wilcoxon: fusion vs best single modality across genes
  - Gene-level delta-AUC characterization
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

OUT = "results/multiomics_validation"
os.makedirs(OUT, exist_ok=True)


def load_embeddings(emb_dir, dual_idx):
    """Load raw edelta + CLIP projections for dual-modality variants."""
    # protein raw
    n = len(dual_idx)
    g2l = {int(g): l for l, g in enumerate(dual_idx)}

    prot_raw = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            gi = int(gi)
            if gi in g2l:
                prot_raw[g2l[gi]] = row

    dna_raw = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            gi = int(gi)
            if gi in g2l:
                dna_raw[g2l[gi]] = row

    return prot_raw, dna_raw


def project_clip(prot_raw, dna_raw, clip, device, bs=8192):
    """Project raw edelta through CLIP encoders to 256-d."""
    n = len(prot_raw)
    z_prot = np.zeros((n, 256), dtype=np.float32)
    z_dna = np.zeros((n, 256), dtype=np.float32)
    clip = clip.to(device)
    for i in range(0, n, bs):
        xp = torch.tensor(prot_raw[i:i+bs], dtype=torch.float32, device=device)
        xd = torch.tensor(dna_raw[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            z_prot[i:i+bs] = clip.enc_prot(xp).cpu().numpy()
            z_dna[i:i+bs] = clip.enc_dna(xd).cpu().numpy()
    return z_prot, z_dna


def run_cv(X, y, n_splits=5, seed=42):
    """5-fold CV with logistic regression, return per-sample predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return preds


def per_gene_auc(preds_dict, y, genes, min_pos=20, min_neg=20):
    """Compute per-gene AUC for each method. Returns DataFrame."""
    gene_arr = np.asarray(genes)
    unique_genes = np.unique(gene_arr)
    rows = []
    for g in unique_genes:
        mask = gene_arr == g
        yg = y[mask]
        n_pos = (yg == 1).sum()
        n_neg = (yg == 0).sum()
        if n_pos < min_pos or n_neg < min_neg:
            continue
        row = {"gene": g, "n_pos": int(n_pos), "n_neg": int(n_neg)}
        for method, preds in preds_dict.items():
            try:
                row[f"auc_{method}"] = roc_auc_score(yg, preds[mask])
            except ValueError:
                row[f"auc_{method}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # load variant annotations
    df = pd.read_csv("data/variant/sae_pretrain/missense_500k_annotated.csv", low_memory=False)
    dual_idx = np.load("data/variant/sae_pretrain/dual_idx.npy")
    sub = df.iloc[dual_idx].reset_index(drop=True)

    # labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = is_path | is_ben
    y = np.where(is_path, 1, 0)[labeled.values]
    genes = sub.loc[labeled, "gene"].values
    print(f"labeled: {len(y)} (path={y.sum()}, ben={(y==0).sum()})", flush=True)
    print(f"unique genes: {len(np.unique(genes))}", flush=True)

    # load raw embeddings
    print("loading raw embeddings ...", flush=True)
    prot_raw, dna_raw = load_embeddings("results/sae_pretrain_emb", dual_idx)
    prot_raw_lab = prot_raw[labeled.values]
    dna_raw_lab = dna_raw[labeled.values]

    # CLIP projection
    print("loading CLIP + projecting ...", flush=True)
    ck = torch.load("results/pretrain/crossmodal_clip_40k.pt", map_location="cpu",
                     weights_only=False)
    clip = CrossModalCLIP(**ck["config"]).eval()
    clip.load_state_dict(ck["model_state"])
    del ck
    z_prot, z_dna = project_clip(prot_raw, dna_raw, clip, device)
    del clip, prot_raw, dna_raw
    torch.cuda.empty_cache()

    z_prot_lab = z_prot[labeled.values]
    z_dna_lab = z_dna[labeled.values]
    z_concat_lab = np.concatenate([z_prot_lab, z_dna_lab], axis=1)

    # norms
    prot_norm = np.linalg.norm(prot_raw_lab, axis=1, keepdims=True)
    dna_norm = np.linalg.norm(dna_raw_lab, axis=1, keepdims=True)
    both_norms = np.concatenate([prot_norm, dna_norm], axis=1)

    # ── Genome-wide 5-fold CV ──────────────────────────────────
    print("\n=== Genome-wide 5-fold CV ===", flush=True)
    methods = {
        "ESM2-norm": prot_norm,
        "Evo2-norm": dna_norm,
        "Both-norms": both_norms,
        "CLIP-prot": z_prot_lab,
        "CLIP-dna": z_dna_lab,
        "CLIP-fusion": z_concat_lab,
    }

    preds_dict = {}
    results = []
    for name, X in methods.items():
        print(f"  {name} ({X.shape[1]}-d) ...", end=" ", flush=True)
        preds = run_cv(X, y)
        auc = roc_auc_score(y, preds)
        preds_dict[name] = preds
        results.append({"method": name, "dim": X.shape[1], "auc": auc})
        print(f"AUC={auc:.4f}", flush=True)

    res = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(f"\n{res.to_string(index=False)}", flush=True)
    res.to_csv(os.path.join(OUT, "genome_wide_auc.csv"), index=False)

    # ── Per-gene AUC ───────────────────────────────────────────
    print("\n=== Per-gene AUC (>=20 path + >=20 ben) ===", flush=True)
    gene_df = per_gene_auc(preds_dict, y, genes)
    gene_df.to_csv(os.path.join(OUT, "per_gene_auc.csv"), index=False)
    print(f"  {len(gene_df)} genes qualified", flush=True)

    # fusion vs best single modality
    gene_df["auc_best_single"] = gene_df[["auc_CLIP-prot", "auc_CLIP-dna"]].max(axis=1)
    gene_df["delta_auc"] = gene_df["auc_CLIP-fusion"] - gene_df["auc_best_single"]

    n_better = (gene_df["delta_auc"] > 0).sum()
    n_worse = (gene_df["delta_auc"] < 0).sum()
    n_tie = (gene_df["delta_auc"] == 0).sum()
    wilcox = stats.wilcoxon(gene_df["auc_CLIP-fusion"], gene_df["auc_best_single"],
                            alternative="greater")
    print(f"  fusion > best_single: {n_better}/{len(gene_df)} genes", flush=True)
    print(f"  fusion < best_single: {n_worse}/{len(gene_df)} genes", flush=True)
    print(f"  Wilcoxon p={wilcox.pvalue:.2e} (one-sided, fusion > single)", flush=True)
    print(f"  mean delta-AUC: {gene_df['delta_auc'].mean():.4f}", flush=True)
    print(f"  median delta-AUC: {gene_df['delta_auc'].median():.4f}", flush=True)

    # also: fusion vs norm baselines
    gene_df["delta_vs_norms"] = gene_df["auc_CLIP-fusion"] - gene_df["auc_Both-norms"]
    wilcox2 = stats.wilcoxon(gene_df["auc_CLIP-fusion"], gene_df["auc_Both-norms"],
                             alternative="greater")
    print(f"\n  CLIP-fusion vs Both-norms:", flush=True)
    print(f"    mean delta-AUC: {gene_df['delta_vs_norms'].mean():.4f}", flush=True)
    print(f"    Wilcoxon p={wilcox2.pvalue:.2e}", flush=True)

    # ── Per-gene: which modality dominates? ────────────────────
    print("\n=== Per-gene modality dominance ===", flush=True)
    gene_df["dominant"] = np.where(
        gene_df["auc_CLIP-prot"] > gene_df["auc_CLIP-dna"], "protein", "dna"
    )
    gene_df["prot_advantage"] = gene_df["auc_CLIP-prot"] - gene_df["auc_CLIP-dna"]
    dom_counts = gene_df["dominant"].value_counts()
    print(f"  protein-dominant: {dom_counts.get('protein', 0)}", flush=True)
    print(f"  dna-dominant: {dom_counts.get('dna', 0)}", flush=True)

    # top genes where fusion helps most
    print(f"\n  Top-15 genes where fusion helps most (delta-AUC):", flush=True)
    for _, r in gene_df.nlargest(15, "delta_auc").iterrows():
        print(f"    {str(r['gene']):10s} n={int(r['n_pos'])+int(r['n_neg']):4d}  "
              f"prot={r['auc_CLIP-prot']:.3f}  dna={r['auc_CLIP-dna']:.3f}  "
              f"fusion={r['auc_CLIP-fusion']:.3f}  Δ={r['delta_auc']:+.3f}", flush=True)

    # top genes where protein dominates
    print(f"\n  Top-10 protein-dominant genes:", flush=True)
    for _, r in gene_df.nlargest(10, "prot_advantage").iterrows():
        print(f"    {str(r['gene']):10s}  prot={r['auc_CLIP-prot']:.3f}  "
              f"dna={r['auc_CLIP-dna']:.3f}  fusion={r['auc_CLIP-fusion']:.3f}", flush=True)

    # top genes where DNA dominates
    print(f"\n  Top-10 DNA-dominant genes:", flush=True)
    for _, r in gene_df.nsmallest(10, "prot_advantage").iterrows():
        print(f"    {str(r['gene']):10s}  prot={r['auc_CLIP-prot']:.3f}  "
              f"dna={r['auc_CLIP-dna']:.3f}  fusion={r['auc_CLIP-fusion']:.3f}", flush=True)

    # ── Summary stats ──────────────────────────────────────────
    print("\n=== Summary ===", flush=True)
    for m in ["CLIP-prot", "CLIP-dna", "CLIP-fusion", "Both-norms"]:
        col = f"auc_{m}"
        if col in gene_df:
            vals = gene_df[col].dropna()
            print(f"  {m:15s}: mean={vals.mean():.3f}  median={vals.median():.3f}  "
                  f"std={vals.std():.3f}", flush=True)

    gene_df.to_csv(os.path.join(OUT, "per_gene_auc.csv"), index=False)
    print(f"\nsaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
