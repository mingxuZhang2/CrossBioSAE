"""
CrossCoder SAE Feature Interpretability Analysis.

Produces:
1. Canonical feature categories (single source of truth for all analyses)
2. Per-feature atlas: top activating variants, AA/gene enrichment, DMS correlation
3. Gene-level modality profiles (raw, per-feature-normalized, active-feature-normalized)
4. ClinVar feature pathogenicity (Fisher exact OR + BH-FDR)
5. Control A: inference-time pairing disruption
6. Control B: train-on-shuffled-pairs negative control (stub, full retrain on HPC)
7. analysis_summary.json aggregating all results

Fixes per GPT Pro review (2026-06-11):
- Feature categories computed ONCE on full pretrain data, saved to canonical file
- ClinVar OR replaced with Fisher exact + Haldane-Anscombe + BH-FDR
- Gene modality profile outputs three normalization variants
- Shuffle control split into inference-time (A) vs retrain (B)
- All results saved as structured JSON + CSV

Usage:
  python analyze_crosscoder.py --checkpoint results/crosscoder_sae/crosscoder.pt
"""

import argparse, glob, os, re, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import fisher_exact
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


def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    ranked = np.argsort(pvalues)
    qvalues = np.zeros(n)
    for i in range(n):
        qvalues[ranked[i]] = pvalues[ranked[i]] * n / (i + 1)
    # Enforce monotonicity from the bottom
    qvalues = np.minimum.accumulate(qvalues[np.argsort(np.argsort(pvalues))[::-1]])[::-1]
    qvalues = np.clip(qvalues, 0, 1)
    # Re-sort
    out = np.zeros(n)
    inv_rank = np.argsort(np.argsort(pvalues))
    for i in range(n):
        out[i] = qvalues[inv_rank[i]]
    return out


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


# ── Step 0: Canonical Feature Categories ──

def compute_canonical_categories(model, assay_data, prep, cfg, device, out_dir):
    """Compute feature categories ONCE on ALL pretrain data. Save as canonical file."""
    print("\n=== Computing Canonical Feature Categories ===", flush=True)

    cat_file = os.path.join(out_dir, "feature_categories.npz")

    all_prot, all_dna = [], []
    for ad in assay_data:
        p, d = prep_assay(ad, prep, cfg)
        all_prot.append(p)
        all_dna.append(d)
    prot_all = np.vstack(all_prot)
    dna_all = np.vstack(all_dna)
    print("  Classifying on %d variants (all DMS)" % len(prot_all), flush=True)

    cats, info = model.classify_features(prot_all, dna_all, device)

    np.savez(cat_file,
             categories=cats,
             trigger_prot_frac=info['trigger_prot_frac'],
             decoder_prot_frac=info['decoder_prot_frac'],
             count_pair=info['count_pair'],
             mass_pair=info['mass_pair'],
             mass_prot=info['mass_prot'],
             mass_dna=info['mass_dna'],
             alive=info['alive'])

    for cat in ['prot-private', 'dna-private', 'shared', 'pair-only', 'dead']:
        print("    %-15s %d" % (cat, (cats == cat).sum()), flush=True)
    print("  Saved to %s" % cat_file, flush=True)

    return cats, info


def load_canonical_categories(out_dir):
    """Load saved canonical feature categories."""
    cat_file = os.path.join(out_dir, "feature_categories.npz")
    d = np.load(cat_file, allow_pickle=True)
    return d['categories'], {k: d[k] for k in d.files}


# ── Analysis 1: Feature Atlas ──

def build_feature_atlas(model, assay_data, prep, cfg, device, out_dir, cats, top_k=20):
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

    act_freq = (Z > 0).mean(axis=0)
    act_mean = np.where(Z > 0, Z, 0).sum(axis=0) / np.maximum((Z > 0).sum(axis=0), 1)
    alive_mask = act_freq > 0.001

    atlas = []
    for j in range(F_dim):
        if not alive_mask[j]:
            continue

        top_idx = np.argsort(-Z[:, j])[:top_k]
        top_acts = Z[top_idx, j]
        top_meta = meta.iloc[top_idx]

        refs = [r for r in top_meta['ref'] if r]
        alts = [a for a in top_meta['alt'] if a]
        ref_counts = Counter(refs)
        alt_counts = Counter(alts)
        ref_class = Counter(aa_class(r) for r in refs if r)
        alt_class = Counter(aa_class(a) for a in alts if a)
        gene_counts = Counter(top_meta['gene'])

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
            'n_active': int(active_mask.sum()),
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

    atlas.sort(key=lambda x: -x['act_freq'])

    with open(os.path.join(out_dir, "feature_atlas.json"), "w") as f:
        json.dump(atlas, f, indent=2, default=str)
    print("  Saved %d feature entries to feature_atlas.json" % len(atlas), flush=True)

    # Summary by category
    summary_by_cat = {}
    print("\n  Feature summary by category:", flush=True)
    for cat in ['prot-private', 'dna-private', 'shared', 'pair-only']:
        cat_feats = [a for a in atlas if a['category'] == cat]
        if cat_feats:
            dms_corrs = [abs(a['dms_corr']) for a in cat_feats]
            act_freqs = [a['act_freq'] for a in cat_feats]
            summary_by_cat[cat] = {
                'n_features': len(cat_feats),
                'mean_abs_dms_corr': float(np.mean(dms_corrs)),
                'median_abs_dms_corr': float(np.median(dms_corrs)),
                'mean_act_freq': float(np.mean(act_freqs)),
                'median_act_freq': float(np.median(act_freqs)),
            }
            print("    %-15s n=%d, mean|dms_corr|=%.4f, mean_freq=%.4f" % (
                cat, len(cat_feats), np.mean(dms_corrs), np.mean(act_freqs)), flush=True)

    # Top features by DMS correlation
    print("\n  Top 10 features by |DMS correlation|:", flush=True)
    by_corr = sorted(atlas, key=lambda x: -abs(x['dms_corr']))[:10]
    for a in by_corr:
        print("    feat=%d (%s) corr=%+.3f freq=%.3f top_alt=%s genes=%s" % (
            a['feature_id'], a['category'], a['dms_corr'], a['act_freq'],
            a['top_alt_aa'][:2], [g[0] for g in a['top_genes'][:3]]), flush=True)

    return atlas, summary_by_cat


# ── Analysis 2: Gene-Level Modality Profile ──

def gene_modality_profiles(model, assay_data, prep, cfg, device, out_dir, cats):
    """Per-gene: fraction of activation mass from PP vs DP vs SH features.
    Three normalization modes per GPT Pro review."""
    print("\n=== Gene Modality Profiles ===", flush=True)

    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'
    sh_mask = cats == 'shared'

    n_pp = pp_mask.sum()
    n_dp = dp_mask.sum()
    n_sh = sh_mask.sum()
    print("  Using canonical categories: PP=%d DP=%d SH=%d" % (n_pp, n_dp, n_sh), flush=True)

    profiles = []
    for ad in assay_data:
        if len(ad['y']) < 30:
            continue
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()

        # Raw mass
        mass_pp = z[:, pp_mask].sum()
        mass_dp = z[:, dp_mask].sum()
        mass_sh = z[:, sh_mask].sum()
        total_raw = mass_pp + mass_dp + mass_sh + 1e-8

        # Per-feature-mean mass (divides by feature count to remove count bias)
        pfm_pp = mass_pp / max(n_pp, 1)
        pfm_dp = mass_dp / max(n_dp, 1)
        pfm_sh = mass_sh / max(n_sh, 1)
        total_pfm = pfm_pp + pfm_dp + pfm_sh + 1e-8

        # Active-feature-normalized mass
        n_active_pp = (z[:, pp_mask] > 0).any(axis=0).sum()
        n_active_dp = (z[:, dp_mask] > 0).any(axis=0).sum()
        n_active_sh = (z[:, sh_mask] > 0).any(axis=0).sum()
        afn_pp = mass_pp / max(n_active_pp, 1)
        afn_dp = mass_dp / max(n_active_dp, 1)
        afn_sh = mass_sh / max(n_active_sh, 1)
        total_afn = afn_pp + afn_dp + afn_sh + 1e-8

        profiles.append({
            'assay': ad['name'],
            'gene': ad['name'].split('_')[0],
            'n_variants': len(ad['y']),
            # Raw mass fraction
            'raw_prot_frac': float(mass_pp / total_raw),
            'raw_dna_frac': float(mass_dp / total_raw),
            'raw_shared_frac': float(mass_sh / total_raw),
            'mass_pp': float(mass_pp),
            'mass_dp': float(mass_dp),
            'mass_sh': float(mass_sh),
            # Per-feature-mean fraction (corrects for PP/DP count asymmetry)
            'pfm_prot_frac': float(pfm_pp / total_pfm),
            'pfm_dna_frac': float(pfm_dp / total_pfm),
            'pfm_shared_frac': float(pfm_sh / total_pfm),
            # Active-feature-normalized fraction
            'afn_prot_frac': float(afn_pp / total_afn),
            'afn_dna_frac': float(afn_dp / total_afn),
            'afn_shared_frac': float(afn_sh / total_afn),
            # Active feature counts
            'n_active_pp': int(n_active_pp),
            'n_active_dp': int(n_active_dp),
            'n_active_sh': int(n_active_sh),
        })

    df = pd.DataFrame(profiles)
    df.to_csv(os.path.join(out_dir, "gene_modality_profiles.csv"), index=False)

    # Print extremes for each normalization
    for norm_name, p_col, d_col, s_col in [
        ('raw_mass', 'raw_prot_frac', 'raw_dna_frac', 'raw_shared_frac'),
        ('per_feature_mean', 'pfm_prot_frac', 'pfm_dna_frac', 'pfm_shared_frac'),
        ('active_feature_norm', 'afn_prot_frac', 'afn_dna_frac', 'afn_shared_frac'),
    ]:
        print("\n  [%s] Most protein-driven:" % norm_name, flush=True)
        for _, r in df.nlargest(5, p_col).iterrows():
            print("    %-40s P=%.1f%% D=%.1f%% S=%.1f%%" % (
                r['assay'][:40], r[p_col]*100, r[d_col]*100, r[s_col]*100), flush=True)
        print("  [%s] Most DNA-driven:" % norm_name, flush=True)
        for _, r in df.nlargest(5, d_col).iterrows():
            print("    %-40s P=%.1f%% D=%.1f%% S=%.1f%%" % (
                r['assay'][:40], r[p_col]*100, r[d_col]*100, r[s_col]*100), flush=True)

    # Gene-level averages (all three norms)
    agg_cols = [c for c in df.columns if c.startswith(('raw_', 'pfm_', 'afn_'))]
    gene_avg = df.groupby('gene')[agg_cols].mean()
    gene_avg.to_csv(os.path.join(out_dir, "gene_avg_profiles.csv"))
    print("\n  Saved %d assay profiles, %d gene averages" % (len(df), len(gene_avg)), flush=True)

    # Check: do the three norms agree on direction?
    agree_prot = ((df['raw_prot_frac'] > df['raw_dna_frac']) ==
                  (df['pfm_prot_frac'] > df['pfm_dna_frac'])).mean()
    agree_afn = ((df['raw_prot_frac'] > df['raw_dna_frac']) ==
                 (df['afn_prot_frac'] > df['afn_dna_frac'])).mean()
    print("  Norm agreement (prot>dna direction): raw vs pfm=%.1f%%, raw vs afn=%.1f%%" % (
        agree_prot * 100, agree_afn * 100), flush=True)

    return df


# ── Analysis 3: ClinVar Feature Pathogenicity (Fisher exact + BH-FDR) ──

def clinvar_feature_analysis(model, clinvar_data, prep, cfg, device, out_dir, cats):
    """Per-feature: Fisher exact test OR with BH-FDR correction."""
    if clinvar_data is None:
        print("\n  No ClinVar data, skipping.", flush=True)
        return None

    print("\n=== ClinVar Feature Analysis (Fisher exact + BH-FDR) ===", flush=True)
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
    n_path_total = (y == 1).sum()
    n_ben_total = (y == 0).sum()
    print("  ClinVar: %d pathogenic, %d benign" % (n_path_total, n_ben_total), flush=True)

    # Per-feature 2x2 Fisher exact test
    results = []
    for j in range(cfg.n_features):
        active = z[:, j] > 0
        n_active = active.sum()
        if n_active < 10:
            continue

        # 2x2 contingency table
        a = (active & (y == 1)).sum()   # active pathogenic
        b = (active & (y == 0)).sum()   # active benign
        c = (~active & (y == 1)).sum()  # inactive pathogenic
        d = (~active & (y == 0)).sum()  # inactive benign

        # Haldane-Anscombe correction: add 0.5 to all cells if any cell is 0
        if a == 0 or b == 0 or c == 0 or d == 0:
            a_h, b_h, c_h, d_h = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        else:
            a_h, b_h, c_h, d_h = a, b, c, d

        odds_ratio = (a_h * d_h) / (b_h * c_h)
        log2_or = float(np.log2(odds_ratio))

        # Fisher exact test
        _, pval = fisher_exact([[a, b], [c, d]])

        results.append({
            'feature': j,
            'category': str(cats[j]),
            'n_active': int(n_active),
            'active_path': int(a),
            'active_ben': int(b),
            'inactive_path': int(c),
            'inactive_ben': int(d),
            'odds_ratio': float(odds_ratio),
            'log2_or': log2_or,
            'p_value': float(pval),
            'path_rate_active': float(a / max(n_active, 1)),
            'ben_rate_active': float(b / max(n_active, 1)),
        })

    df = pd.DataFrame(results)

    # BH-FDR correction
    df['q_value'] = bh_fdr(df['p_value'].values)

    df.to_csv(os.path.join(out_dir, "clinvar_feature_pathogenicity_fdr.csv"), index=False)

    # Summary by category
    print("\n  Pathogenicity enrichment by feature category:", flush=True)
    for cat in ['prot-private', 'dna-private', 'shared']:
        sub = df[df['category'] == cat]
        if len(sub):
            n_sig = (sub['q_value'] < 0.05).sum()
            n_path_enriched = ((sub['q_value'] < 0.05) & (sub['log2_or'] > 0)).sum()
            n_ben_enriched = ((sub['q_value'] < 0.05) & (sub['log2_or'] < 0)).sum()
            print("    %-15s n=%d, FDR<0.05: %d (path-enriched=%d, ben-enriched=%d), "
                  "median_OR=%.2f, median_log2OR=%+.2f" % (
                cat, len(sub), n_sig, n_path_enriched, n_ben_enriched,
                sub['odds_ratio'].median(), sub['log2_or'].median()), flush=True)

    # Top pathogenic features (by OR, FDR < 0.05)
    sig = df[df['q_value'] < 0.05]
    print("\n  Top 10 FDR-significant pathogenic features:", flush=True)
    for _, r in sig.nlargest(10, 'odds_ratio').iterrows():
        print("    feat=%d (%s) OR=%.1f log2OR=%+.1f q=%.2e path_rate=%.3f n=%d" % (
            r['feature'], r['category'], r['odds_ratio'], r['log2_or'],
            r['q_value'], r['path_rate_active'], r['n_active']), flush=True)

    # Top benign features
    print("\n  Top 10 FDR-significant benign-enriched features:", flush=True)
    for _, r in sig.nsmallest(10, 'odds_ratio').iterrows():
        print("    feat=%d (%s) OR=%.2f log2OR=%+.2f q=%.2e path_rate=%.3f n=%d" % (
            r['feature'], r['category'], r['odds_ratio'], r['log2_or'],
            r['q_value'], r['path_rate_active'], r['n_active']), flush=True)

    return df


# ── Control A: Inference-Time Pairing Disruption ──

def control_a_inference_shuffle(model, assay_data, prep, cfg, device, out_dir):
    """Tests whether trained model depends on correct ESM/Evo pairing at inference time."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    print("\n=== Control A: Inference-Time Pairing Disruption ===", flush=True)

    results = []
    np.random.seed(42)

    for ad in assay_data:
        if len(ad['y']) < 50:
            continue
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)

        with torch.no_grad():
            z_real = model.encode(pt, dt).cpu().numpy()

        perm = np.random.permutation(len(dna_pca))
        dt_shuf = torch.FloatTensor(dna_pca[perm]).to(device)
        with torch.no_grad():
            z_shuf = model.encode(pt, dt_shuf).cpu().numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rho_real = rho_shuf = 0.0

        for label, X in [('real', z_real), ('shuffled', z_shuf)]:
            fold_rhos = []
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0)
                m.fit(X[tr], ad['y'][tr])
                yp = m.predict(X[te])
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        fold_rhos.append(r)
            mean_rho = np.mean(fold_rhos) if fold_rhos else 0.0
            if label == 'real':
                rho_real = mean_rho
            else:
                rho_shuf = mean_rho

        results.append({
            'assay': ad['name'],
            'gene': ad['name'].split('_')[0],
            'n': len(ad['y']),
            'rho_real': float(rho_real),
            'rho_shuffled': float(rho_shuf),
            'delta': float(rho_real - rho_shuf),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "control_a_inference_shuffle.csv"), index=False)

    real = df['rho_real'].values
    shuf = df['rho_shuffled'].values
    delta = df['delta'].values
    t_stat, p_val = stats.ttest_rel(real, shuf)

    print("  %d assays evaluated" % len(df), flush=True)
    print("  Real pairs:     mean=%.4f median=%.4f" % (real.mean(), np.median(real)), flush=True)
    print("  Shuffled pairs: mean=%.4f median=%.4f" % (shuf.mean(), np.median(shuf)), flush=True)
    print("  Δ (real-shuf):  mean=%.4f median=%.4f" % (delta.mean(), np.median(delta)), flush=True)
    print("  Paired t-test:  t=%.2f, p=%.2e" % (t_stat, p_val), flush=True)
    print("  Assays with Δ>0: %d/%d (%.1f%%)" % (
        (delta > 0).sum(), len(delta), (delta > 0).mean() * 100), flush=True)

    summary = {
        'n_assays': len(df),
        'real_mean': float(real.mean()),
        'real_median': float(np.median(real)),
        'shuffled_mean': float(shuf.mean()),
        'shuffled_median': float(np.median(shuf)),
        'delta_mean': float(delta.mean()),
        'delta_median': float(np.median(delta)),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'frac_delta_positive': float((delta > 0).mean()),
    }
    return df, summary


# ── Control B: Train-on-Shuffled-Pairs (stub for HPC) ──

def control_b_retrain_stub(out_dir):
    """Placeholder for train-on-shuffled-pairs negative control.

    This requires a full CrossCoder retrain (~2 min GPU) on shuffled ESM-Evo pairings.
    Should be run as a separate SLURM job:
        python crosscoder_sae.py --phase all --shuffle_pairs --out_dir results/crosscoder_sae_shuffled
    Then compare:
        PP/DP/SH counts, DMS ablation, ClinVar AUROC, feature enrichment
    between real-pair and shuffled-pair models.

    Expected: shuffled model should have degraded shared features, less stable
    gene modality profiles, and weaker biological enrichment.
    """
    print("\n=== Control B: Train-on-Shuffled-Pairs ===", flush=True)
    print("  STUB: requires separate HPC job. See docstring for instructions.", flush=True)
    print("  Run: python crosscoder_sae.py --phase all --shuffle_pairs "
          "--out_dir results/crosscoder_sae_shuffled", flush=True)

    stub_info = {
        'status': 'not_run',
        'instructions': 'Run crosscoder_sae.py --shuffle_pairs on HPC GPU node',
        'expected_outputs': [
            'results/crosscoder_sae_shuffled/crosscoder.pt',
            'results/crosscoder_sae_shuffled/ablation.csv',
            'results/crosscoder_sae_shuffled/config.json',
        ],
        'comparison_metrics': [
            'PP/DP/SH feature counts',
            'DMS mean Spearman (full, PP-only, DP-only, SH-only)',
            'ClinVar AUROC (cc_all, cc_prot_priv, cc_dna_priv, cc_shared)',
            'Gene modality profile stability (Spearman corr of profiles vs real)',
            'Feature enrichment FDR counts',
        ],
    }
    with open(os.path.join(out_dir, "control_b_retrain_stub.json"), "w") as f:
        json.dump(stub_info, f, indent=2)
    return stub_info


# ── Summary ──

def write_analysis_summary(out_dir, cats, atlas_summary, profiles_df,
                           clinvar_df, ctrl_a_summary, ctrl_b_info):
    """Write analysis_summary.json aggregating all results."""
    print("\n=== Writing analysis_summary.json ===", flush=True)

    summary = {
        'feature_categories': {
            'prot_private': int((cats == 'prot-private').sum()),
            'dna_private': int((cats == 'dna-private').sum()),
            'shared': int((cats == 'shared').sum()),
            'pair_only': int((cats == 'pair-only').sum()),
            'dead': int((cats == 'dead').sum()),
            'total': int(len(cats)),
        },
        'feature_atlas': atlas_summary,
    }

    if profiles_df is not None and len(profiles_df):
        summary['gene_modality_profiles'] = {
            'n_assays': len(profiles_df),
            'n_genes': int(profiles_df['gene'].nunique()),
            'raw_mean_prot_frac': float(profiles_df['raw_prot_frac'].mean()),
            'raw_mean_dna_frac': float(profiles_df['raw_dna_frac'].mean()),
            'raw_mean_shared_frac': float(profiles_df['raw_shared_frac'].mean()),
            'pfm_mean_prot_frac': float(profiles_df['pfm_prot_frac'].mean()),
            'pfm_mean_dna_frac': float(profiles_df['pfm_dna_frac'].mean()),
            'pfm_mean_shared_frac': float(profiles_df['pfm_shared_frac'].mean()),
            'afn_mean_prot_frac': float(profiles_df['afn_prot_frac'].mean()),
            'afn_mean_dna_frac': float(profiles_df['afn_dna_frac'].mean()),
            'afn_mean_shared_frac': float(profiles_df['afn_shared_frac'].mean()),
        }

    if clinvar_df is not None and len(clinvar_df):
        n_sig = int((clinvar_df['q_value'] < 0.05).sum())
        summary['clinvar_pathogenicity'] = {
            'n_features_tested': len(clinvar_df),
            'n_fdr_significant': n_sig,
            'n_path_enriched_fdr05': int(
                ((clinvar_df['q_value'] < 0.05) & (clinvar_df['log2_or'] > 0)).sum()),
            'n_ben_enriched_fdr05': int(
                ((clinvar_df['q_value'] < 0.05) & (clinvar_df['log2_or'] < 0)).sum()),
            'median_or': float(clinvar_df['odds_ratio'].median()),
            'by_category': {},
        }
        for cat in ['prot-private', 'dna-private', 'shared']:
            sub = clinvar_df[clinvar_df['category'] == cat]
            if len(sub):
                summary['clinvar_pathogenicity']['by_category'][cat] = {
                    'n': len(sub),
                    'n_fdr05': int((sub['q_value'] < 0.05).sum()),
                    'median_or': float(sub['odds_ratio'].median()),
                    'median_log2or': float(sub['log2_or'].median()),
                }

    if ctrl_a_summary:
        summary['control_a_inference_shuffle'] = ctrl_a_summary

    summary['control_b_retrain'] = ctrl_b_info

    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved analysis_summary.json", flush=True)

    return summary


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

    # 0. Canonical feature categories (computed ONCE, used everywhere)
    cats, cat_info = compute_canonical_categories(
        model, assay_data, prep, cfg, device, args.out_dir)

    # 1. Feature atlas (uses canonical cats)
    atlas, atlas_summary = build_feature_atlas(
        model, assay_data, prep, cfg, device, args.out_dir, cats)

    # 2. Gene modality profiles (uses canonical cats, three normalizations)
    profiles = gene_modality_profiles(
        model, assay_data, prep, cfg, device, args.out_dir, cats)

    # 3. ClinVar feature pathogenicity (Fisher exact + BH-FDR, uses canonical cats)
    clinvar_feats = clinvar_feature_analysis(
        model, clinvar_data, prep, cfg, device, args.out_dir, cats)

    # 4. Control A: Inference-time shuffle
    ctrl_a_df, ctrl_a_summary = control_a_inference_shuffle(
        model, assay_data, prep, cfg, device, args.out_dir)

    # 5. Control B: Train-on-shuffled-pairs (stub)
    ctrl_b_info = control_b_retrain_stub(args.out_dir)

    # 6. Aggregate summary
    write_analysis_summary(
        args.out_dir, cats, atlas_summary, profiles,
        clinvar_feats, ctrl_a_summary, ctrl_b_info)

    print("\n=== ALL ANALYSES DONE ===", flush=True)


if __name__ == "__main__":
    main()
