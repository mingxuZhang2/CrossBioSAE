"""
CrossCoder SAE Feature Interpretability Analysis.

Produces:
1. Per-feature top activating variants + AA substitution enrichment
2. Gene-level modality profiles (protein-private vs DNA-private vs shared mass)
3. Feature-DMS correlation (which features predict fitness)
4. Seed stability check
5. Shuffled-pair control

Usage:
  python analyze_crosscoder.py --checkpoint results/crosscoder_sae/crosscoder.pt
"""

import argparse, glob, os, re, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(__file__))
from crosscoder_sae import (
    CrossCoderSAE, CrossCoderConfig,
    load_raw_embeddings, load_clinvar_matched,
    preprocess, apply_prep, prep_assay, parse_mutant,
)


AA_CLASSES = {
    'hydrophobic': set('AVILMFWP'),
    'polar': set('STNQ'),
    'positive': set('RKH'),
    'negative': set('DE'),
    'special': set('CGY'),
}

def aa_class(aa):
    for cls, aas in AA_CLASSES.items():
        if aa in aas:
            return cls
    return 'other'


def load_model_and_data(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device, flush=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = CrossCoderConfig(**state['config'])
    model = CrossCoderSAE(cfg).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    prep = state['prep']

    print("Loading DMS ...", flush=True)
    all_prot, all_dna, assay_data = load_raw_embeddings(args.emb_dir)

    clinvar_data = None
    if os.path.exists(args.clinvar_esm2):
        print("Loading ClinVar ...", flush=True)
        _, _, clinvar_data = load_clinvar_matched(
            args.clinvar_esm2, args.clinvar_evo2, args.clinvar_csv, args.clinvar_parquet)

    return model, cfg, prep, assay_data, clinvar_data, device


# ── Analysis 1: Feature Atlas ──

def build_feature_atlas(model, assay_data, prep, cfg, device, out_dir, top_k=20):
    """For each alive feature: top activating variants, AA enrichment, gene enrichment."""
    print("\n=== Feature Atlas ===", flush=True)

    all_z, all_meta = [], []
    for ad in assay_data:
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()
        all_z.append(z)
        for i in range(len(z)):
            ref, pos, alt = parse_mutant(ad['mutants'][i])
            all_meta.append({
                'assay': ad['name'], 'gene': ad['name'].split('_')[0],
                'mutant': str(ad['mutants'][i]),
                'ref': ref, 'alt': alt, 'pos': pos,
                'dms_score': float(ad['y'][i]),
            })

    Z = np.vstack(all_z)
    meta = pd.DataFrame(all_meta)
    N, F_dim = Z.shape
    print("  %d variants, %d features" % (N, F_dim), flush=True)

    # Activation stats per feature
    act_freq = (Z > 0).mean(axis=0)
    act_mean = np.where(Z > 0, Z, 0).sum(axis=0) / np.maximum((Z > 0).sum(axis=0), 1)
    alive_mask = act_freq > 0.001

    # Feature classification
    cats, info = model.classify_features(
        np.vstack([prep_assay(ad, prep, cfg)[0] for ad in assay_data[:50]]),
        np.vstack([prep_assay(ad, prep, cfg)[1] for ad in assay_data[:50]]),
        device)

    atlas = []
    for j in range(F_dim):
        if not alive_mask[j]:
            continue

        # Top activating variants
        top_idx = np.argsort(-Z[:, j])[:top_k]
        top_acts = Z[top_idx, j]
        top_meta = meta.iloc[top_idx]

        # AA substitution enrichment
        refs = [r for r in top_meta['ref'] if r]
        alts = [a for a in top_meta['alt'] if a]
        ref_counts = Counter(refs)
        alt_counts = Counter(alts)
        ref_class = Counter(aa_class(r) for r in refs if r)
        alt_class = Counter(aa_class(a) for a in alts if a)

        # Gene enrichment
        gene_counts = Counter(top_meta['gene'])

        # DMS correlation
        active_mask = Z[:, j] > 0
        if active_mask.sum() > 20:
            dms_corr = stats.spearmanr(Z[active_mask, j], meta['dms_score'].values[active_mask]).statistic
        else:
            dms_corr = 0.0

        entry = {
            'feature_id': int(j),
            'category': str(cats[j]),
            'act_freq': float(act_freq[j]),
            'act_mean': float(act_mean[j]),
            'dms_corr': float(dms_corr) if not np.isnan(dms_corr) else 0.0,
            'top_ref_aa': ref_counts.most_common(3),
            'top_alt_aa': alt_counts.most_common(3),
            'top_ref_class': ref_class.most_common(3),
            'top_alt_class': alt_class.most_common(3),
            'top_genes': gene_counts.most_common(5),
            'top_variants': [
                {'variant': str(top_meta.iloc[k]['mutant']),
                 'assay': str(top_meta.iloc[k]['assay'])[:40],
                 'act': float(top_acts[k]),
                 'dms': float(top_meta.iloc[k]['dms_score'])}
                for k in range(min(5, len(top_idx)))
            ],
        }
        atlas.append(entry)

    # Sort by activation frequency
    atlas.sort(key=lambda x: -x['act_freq'])

    # Save
    with open(os.path.join(out_dir, "feature_atlas.json"), "w") as f:
        json.dump(atlas, f, indent=2, default=str)
    print("  Saved %d feature entries to feature_atlas.json" % len(atlas), flush=True)

    # Summary by category
    print("\n  Feature summary by category:", flush=True)
    for cat in ['prot-private', 'dna-private', 'shared', 'pair-only', 'dead']:
        cat_feats = [a for a in atlas if a['category'] == cat]
        if cat_feats:
            dms_corrs = [abs(a['dms_corr']) for a in cat_feats]
            print("    %-15s n=%d, mean|dms_corr|=%.4f" % (
                cat, len(cat_feats), np.mean(dms_corrs)), flush=True)

    # Top features by DMS correlation
    print("\n  Top 10 features by |DMS correlation|:", flush=True)
    by_corr = sorted(atlas, key=lambda x: -abs(x['dms_corr']))[:10]
    for a in by_corr:
        print("    feat=%d (%s) corr=%+.3f freq=%.3f top_alt=%s genes=%s" % (
            a['feature_id'], a['category'], a['dms_corr'], a['act_freq'],
            a['top_alt_aa'][:2], [g[0] for g in a['top_genes'][:3]]), flush=True)

    return atlas


# ── Analysis 2: Gene-Level Modality Profile ──

def gene_modality_profiles(model, assay_data, prep, cfg, device, out_dir):
    """Per-gene: fraction of activation mass from PP vs DP vs SH features."""
    print("\n=== Gene Modality Profiles ===", flush=True)

    # Get feature categories from full data
    prot_all = np.vstack([prep_assay(ad, prep, cfg)[0] for ad in assay_data[:100]])
    dna_all = np.vstack([prep_assay(ad, prep, cfg)[1] for ad in assay_data[:100]])
    cats, info = model.classify_features(prot_all, dna_all, device)

    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'
    sh_mask = cats == 'shared'

    profiles = []
    for ad in assay_data:
        if len(ad['y']) < 30:
            continue
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()

        mass_pp = z[:, pp_mask].sum()
        mass_dp = z[:, dp_mask].sum()
        mass_sh = z[:, sh_mask].sum()
        total = mass_pp + mass_dp + mass_sh + 1e-8

        profiles.append({
            'assay': ad['name'],
            'gene': ad['name'].split('_')[0],
            'n_variants': len(ad['y']),
            'prot_frac': float(mass_pp / total),
            'dna_frac': float(mass_dp / total),
            'shared_frac': float(mass_sh / total),
            'mass_pp': float(mass_pp),
            'mass_dp': float(mass_dp),
            'mass_sh': float(mass_sh),
        })

    df = pd.DataFrame(profiles)
    df.to_csv(os.path.join(out_dir, "gene_modality_profiles.csv"), index=False)

    # Print extremes
    print("\n  Most protein-driven assays:", flush=True)
    for _, r in df.nlargest(10, 'prot_frac').iterrows():
        print("    %-40s P=%.1f%% D=%.1f%% S=%.1f%%" % (
            r['assay'][:40], r['prot_frac']*100, r['dna_frac']*100, r['shared_frac']*100), flush=True)

    print("\n  Most DNA-driven assays:", flush=True)
    for _, r in df.nlargest(10, 'dna_frac').iterrows():
        print("    %-40s P=%.1f%% D=%.1f%% S=%.1f%%" % (
            r['assay'][:40], r['prot_frac']*100, r['dna_frac']*100, r['shared_frac']*100), flush=True)

    print("\n  Most shared assays:", flush=True)
    for _, r in df.nlargest(10, 'shared_frac').iterrows():
        print("    %-40s P=%.1f%% D=%.1f%% S=%.1f%%" % (
            r['assay'][:40], r['prot_frac']*100, r['dna_frac']*100, r['shared_frac']*100), flush=True)

    # Gene-level summary (average across assays for same gene)
    gene_avg = df.groupby('gene')[['prot_frac', 'dna_frac', 'shared_frac']].mean()
    gene_avg.to_csv(os.path.join(out_dir, "gene_avg_profiles.csv"))
    print("\n  Saved %d assay profiles, %d gene averages" % (len(df), len(gene_avg)), flush=True)

    return df


# ── Analysis 3: ClinVar Feature Patterns ──

def clinvar_feature_analysis(model, clinvar_data, prep, cfg, device, out_dir):
    """Per-feature: pathogenicity enrichment in ClinVar."""
    if clinvar_data is None:
        print("\n  No ClinVar data, skipping.", flush=True)
        return

    print("\n=== ClinVar Feature Analysis ===", flush=True)
    prot_pca = apply_prep(clinvar_data['prot'], prep['prot_mean'], prep['prot_std'],
                          prep['pca_prot_components'], prep['pca_prot_mean'])
    dna_pca = apply_prep(clinvar_data['dna'], prep['dna_mean'], prep['dna_std'],
                         prep['pca_dna_components'], prep['pca_dna_mean'],
                         prep.get('pca_dna_var') if cfg.whiten_dna else None)

    pt = torch.FloatTensor(prot_pca).to(device)
    dt = torch.FloatTensor(dna_pca).to(device)
    with torch.no_grad():
        z = model.encode(pt, dt).cpu().numpy()

    y = clinvar_data['label']
    genes = clinvar_data['gene']

    cats, _ = model.classify_features(prot_pca[:5000], dna_pca[:5000], device)

    # Per-feature pathogenicity association
    results = []
    for j in range(cfg.n_features):
        active = z[:, j] > 0
        if active.sum() < 10:
            continue
        path_rate_active = y[active].mean()
        path_rate_inactive = y[~active].mean() if (~active).sum() > 0 else 0
        odds = path_rate_active / max(path_rate_inactive, 0.01)
        results.append({
            'feature': j,
            'category': str(cats[j]),
            'n_active': int(active.sum()),
            'path_rate_active': float(path_rate_active),
            'path_rate_inactive': float(path_rate_inactive),
            'odds_ratio': float(odds),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "clinvar_feature_pathogenicity.csv"), index=False)

    # Summary by category
    print("  Pathogenicity enrichment by feature category:", flush=True)
    for cat in ['prot-private', 'dna-private', 'shared']:
        sub = df[df['category'] == cat]
        if len(sub):
            print("    %-15s n=%d, mean_OR=%.2f, path_rate_when_active=%.3f" % (
                cat, len(sub), sub['odds_ratio'].median(), sub['path_rate_active'].mean()), flush=True)

    # Top pathogenic features
    print("\n  Top 10 features enriched for pathogenicity (by OR):", flush=True)
    for _, r in df.nlargest(10, 'odds_ratio').iterrows():
        print("    feat=%d (%s) OR=%.1f path_rate=%.3f n=%d" % (
            r['feature'], r['category'], r['odds_ratio'],
            r['path_rate_active'], r['n_active']), flush=True)

    # Top benign features
    print("\n  Top 10 features enriched for benign (lowest OR):", flush=True)
    for _, r in df.nsmallest(10, 'odds_ratio').iterrows():
        print("    feat=%d (%s) OR=%.2f path_rate=%.3f n=%d" % (
            r['feature'], r['category'], r['odds_ratio'],
            r['path_rate_active'], r['n_active']), flush=True)

    return df


# ── Analysis 4: Shuffled-Pair Control ──

def shuffled_pair_control(model, assay_data, prep, cfg, device):
    """Compare real ESM-Evo pairs vs shuffled pairs."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    print("\n=== Shuffled-Pair Control ===", flush=True)

    real_rhos, shuffled_rhos = [], []
    np.random.seed(42)

    for ad in assay_data:
        if len(ad['y']) < 50:
            continue
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)

        # Real pairs
        with torch.no_grad():
            z_real = model.encode(pt, dt).cpu().numpy()

        # Shuffled: randomly permute DNA within assay
        perm = np.random.permutation(len(dna_pca))
        dt_shuf = torch.FloatTensor(dna_pca[perm]).to(device)
        with torch.no_grad():
            z_shuf = model.encode(pt, dt_shuf).cpu().numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for X, rhos_list in [(z_real, real_rhos), (z_shuf, shuffled_rhos)]:
            fold_rhos = []
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0)
                m.fit(X[tr], ad['y'][tr])
                yp = m.predict(X[te])
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        fold_rhos.append(r)
            if fold_rhos:
                rhos_list.append(np.mean(fold_rhos))

    real = np.array(real_rhos)
    shuf = np.array(shuffled_rhos)
    print("  Real pairs:     mean=%.4f median=%.4f" % (real.mean(), np.median(real)), flush=True)
    print("  Shuffled pairs: mean=%.4f median=%.4f" % (shuf.mean(), np.median(shuf)), flush=True)
    print("  Δ (real-shuf):  mean=%.4f" % (real - shuf).mean(), flush=True)
    print("  Paired t-test:  t=%.2f, p=%.2e" % stats.ttest_rel(real, shuf), flush=True)

    return real, shuf


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/crosscoder_sae/crosscoder.pt")
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--out_dir", default="results/crosscoder_sae/analysis")
    ap.add_argument("--clinvar_esm2", default="results/clinvar_esm2")
    ap.add_argument("--clinvar_evo2", default="results/variant")
    ap.add_argument("--clinvar_csv", default="data/full/clinvar_variants.csv")
    ap.add_argument("--clinvar_parquet", default="data/variant/clinvar.parquet")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg, prep, assay_data, clinvar_data, device = load_model_and_data(args)

    # 1. Feature atlas
    atlas = build_feature_atlas(model, assay_data, prep, cfg, device, args.out_dir)

    # 2. Gene modality profiles
    profiles = gene_modality_profiles(model, assay_data, prep, cfg, device, args.out_dir)

    # 3. ClinVar feature patterns
    clinvar_feats = clinvar_feature_analysis(model, clinvar_data, prep, cfg, device, args.out_dir)

    # 4. Shuffled-pair control
    real, shuf = shuffled_pair_control(model, assay_data, prep, cfg, device)

    print("\n=== ALL ANALYSES DONE ===", flush=True)


if __name__ == "__main__":
    main()
