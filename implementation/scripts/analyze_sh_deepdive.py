"""
Shared-Feature Deep Dive Analysis.

Resolves the SH tension: highest ClinVar OR but negligible DMS importance.
Produces variant-level and feature-level evidence for the "rare but potent" narrative.

Analyses:
1. Variant-level SH activation bins → pathogenic rate by SH mass quartile
2. Top-10 SH feature biological annotation (genes, domains, AA substitutions)
3. Label-stratified SH activation: ClinVar path vs benign, DMS severe vs neutral
4. ClinVar ablation: PP+SH vs PP-only vs all features
5. SH feature coverage vs effect-size decomposition
6. Within-gene shuffle control (finer than global shuffle)

Usage:
  python analyze_sh_deepdive.py --checkpoint results/crosscoder_sae/crosscoder.pt
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import fisher_exact
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR correction (standard implementation)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked_p = p[order]
    q_sorted = ranked_p * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    prep = ckpt['preprocessor']

    from model import CrossCoderConfig, FactorisedCrossCoder
    cc_cfg = CrossCoderConfig(
        dim_prot=cfg['dim_prot'], dim_dna=cfg['dim_dna'],
        n_features=cfg['n_features'], top_k=cfg['top_k'],
    )
    model = FactorisedCrossCoder(cc_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    return model, cfg, prep


def apply_prep(raw, mean, std, pca_components, pca_mean, pca_var=None):
    x = (raw - mean) / (std + 1e-8)
    x_pca = (x - pca_mean) @ pca_components.T
    if pca_var is not None:
        x_pca = x_pca / (np.sqrt(pca_var) + 1e-8)
    return x_pca.astype(np.float32)


def prep_assay(ad, prep, cfg):
    prot_pca = apply_prep(ad['prot'], prep['prot_mean'], prep['prot_std'],
                          prep['pca_prot_components'], prep['pca_prot_mean'])
    dna_pca = apply_prep(ad['dna'], prep['dna_mean'], prep['dna_std'],
                         prep['pca_dna_components'], prep['pca_dna_mean'],
                         prep.get('pca_dna_var') if cfg.get('whiten_dna', True) else None)
    return prot_pca, dna_pca


def load_dms_data(emb_dir):
    import glob
    assay_data = []
    for npz_path in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))):
        d = np.load(npz_path, allow_pickle=True)
        assay_data.append({
            'name': str(d['assay_name']),
            'prot': d['prot_edelta'],
            'dna': d['dna_edelta'],
            'y': d['y'],
            'mutants': d['mutants'],
        })
    return assay_data


def load_clinvar(clinvar_esm2, clinvar_evo2, clinvar_csv, clinvar_parquet):
    if os.path.exists(clinvar_parquet):
        cv = pd.read_parquet(clinvar_parquet)
    elif os.path.exists(clinvar_csv):
        cv = pd.read_csv(clinvar_csv)
    else:
        return None

    esm_shards = sorted([f for f in os.listdir(clinvar_esm2) if f.endswith('.npz')])
    evo_shards = sorted([f for f in os.listdir(clinvar_evo2) if f.endswith('.npz')])
    if not esm_shards or not evo_shards:
        return None

    prot_list, dna_list, label_list, gene_list = [], [], [], []
    for sf in esm_shards:
        d = np.load(os.path.join(clinvar_esm2, sf), allow_pickle=True)
        prot_list.append(d['edelta'])
        if 'labels' in d:
            label_list.append(d['labels'])
        if 'genes' in d:
            gene_list.append(d['genes'])
    for sf in evo_shards:
        d = np.load(os.path.join(clinvar_evo2, sf), allow_pickle=True)
        dna_list.append(d['edelta'] if 'edelta' in d else d['embeddings'])

    prot = np.vstack(prot_list)
    dna = np.vstack(dna_list)
    n = min(len(prot), len(dna))
    prot, dna = prot[:n], dna[:n]

    if label_list:
        labels = np.concatenate(label_list)[:n]
    else:
        labels = cv['pathogenicity'].values[:n]

    genes = np.concatenate(gene_list)[:n] if gene_list else cv['GeneSymbol'].values[:n]

    return {'prot': prot, 'dna': dna, 'label': labels, 'genes': genes}


# ── Analysis 1: Variant-level SH activation bins ──

def variant_level_sh_bins(z_clinvar, y_clinvar, genes_clinvar, cats, out_dir):
    """Bin ClinVar variants by total SH activation mass, report pathogenic rate per bin."""
    print("\n=== Analysis 1: Variant-level SH Activation Bins ===", flush=True)

    sh_mask = cats == 'shared'
    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'

    sh_mass = z_clinvar[:, sh_mask].sum(axis=1)
    pp_mass = z_clinvar[:, pp_mask].sum(axis=1)
    dp_mass = z_clinvar[:, dp_mask].sum(axis=1)

    sh_any = (z_clinvar[:, sh_mask] > 0).any(axis=1)
    sh_count = (z_clinvar[:, sh_mask] > 0).sum(axis=1)

    results = {
        'total_variants': int(len(y_clinvar)),
        'n_pathogenic': int((y_clinvar == 1).sum()),
        'n_benign': int((y_clinvar == 0).sum()),
        'sh_activation_coverage': float(sh_any.mean()),
    }

    # Bin by SH mass quartiles (among variants with any SH activation)
    active_idx = np.where(sh_any)[0]
    inactive_idx = np.where(~sh_any)[0]

    bins_data = []

    # Bin 0: no SH activation
    n0 = len(inactive_idx)
    n0_path = (y_clinvar[inactive_idx] == 1).sum()
    bins_data.append({
        'bin': 'no_SH',
        'n': int(n0),
        'n_path': int(n0_path),
        'path_rate': float(n0_path / max(n0, 1)),
        'mean_sh_mass': 0.0,
        'mean_pp_mass': float(pp_mass[inactive_idx].mean()) if n0 > 0 else 0,
        'mean_dp_mass': float(dp_mass[inactive_idx].mean()) if n0 > 0 else 0,
    })

    if len(active_idx) > 20:
        sh_mass_active = sh_mass[active_idx]
        q25, q50, q75 = np.percentile(sh_mass_active, [25, 50, 75])

        for label, lo, hi in [('low_SH', 0, q25), ('med_low_SH', q25, q50),
                               ('med_high_SH', q50, q75), ('high_SH', q75, np.inf)]:
            mask = (sh_mass_active > lo) & (sh_mass_active <= hi) if lo > 0 else \
                   (sh_mass_active > 0) & (sh_mass_active <= hi)
            if label == 'high_SH':
                mask = sh_mass_active > q75
            idx = active_idx[mask]
            n_bin = len(idx)
            n_path = (y_clinvar[idx] == 1).sum()
            bins_data.append({
                'bin': label,
                'n': int(n_bin),
                'n_path': int(n_path),
                'path_rate': float(n_path / max(n_bin, 1)),
                'mean_sh_mass': float(sh_mass[idx].mean()) if n_bin > 0 else 0,
                'mean_pp_mass': float(pp_mass[idx].mean()) if n_bin > 0 else 0,
                'mean_dp_mass': float(dp_mass[idx].mean()) if n_bin > 0 else 0,
            })

    results['bins'] = bins_data

    # Fisher: SH-active vs SH-inactive
    a = ((y_clinvar == 1) & sh_any).sum()
    b = ((y_clinvar == 0) & sh_any).sum()
    c = ((y_clinvar == 1) & ~sh_any).sum()
    d = ((y_clinvar == 0) & ~sh_any).sum()
    or_val, p_val = fisher_exact([[a, b], [c, d]])
    results['sh_active_vs_inactive'] = {
        'active_path': int(a), 'active_ben': int(b),
        'inactive_path': int(c), 'inactive_ben': int(d),
        'odds_ratio': float(or_val), 'p_value': float(p_val),
    }

    # Gene enrichment among high-SH variants
    if len(active_idx) > 20:
        high_sh_idx = active_idx[sh_mass[active_idx] > q75]
        gene_counts = Counter(genes_clinvar[high_sh_idx])
        results['high_sh_gene_enrichment'] = dict(gene_counts.most_common(20))

    print("  SH activation coverage: %.1f%% of ClinVar variants" % (sh_any.mean() * 100))
    for b in bins_data:
        print("    %-15s n=%d, path_rate=%.3f, mean_sh=%.3f" % (
            b['bin'], b['n'], b['path_rate'], b['mean_sh_mass']))

    if 'sh_active_vs_inactive' in results:
        r = results['sh_active_vs_inactive']
        print("  SH-active vs inactive: OR=%.3f, p=%.2e" % (r['odds_ratio'], r['p_value']))

    return results


# ── Analysis 2: Top-10 SH features deep dive ──

def top_sh_features_deepdive(z_clinvar, y_clinvar, genes_clinvar, z_dms, dms_meta,
                             cats, clinvar_fdr_df, out_dir):
    """Biological annotation of top SH features by ClinVar OR."""
    print("\n=== Analysis 2: Top-10 SH Feature Deep Dive ===", flush=True)

    sh_features = clinvar_fdr_df[clinvar_fdr_df['category'] == 'shared'].copy()
    sh_features = sh_features.sort_values('odds_ratio', ascending=False)
    top10 = sh_features.head(10)

    feature_cards = []
    for _, row in top10.iterrows():
        fid = int(row['feature'])
        card = {
            'feature_id': fid,
            'odds_ratio': float(row['odds_ratio']),
            'log2_or': float(row['log2_or']),
            'p_value': float(row['p_value']),
            'q_value': float(row['q_value']),
            'n_active_clinvar': int(row['n_active']),
            'active_path': int(row['active_path']),
            'active_ben': int(row['active_ben']),
            'path_rate': float(row['path_rate_active']),
        }

        # Gene distribution in ClinVar
        active_cv = z_clinvar[:, fid] > 0
        gene_counts_cv = Counter(genes_clinvar[active_cv])
        card['top_genes_clinvar'] = dict(gene_counts_cv.most_common(10))

        # Gene distribution in DMS
        if z_dms is not None:
            active_dms = z_dms[:, fid] > 0
            if active_dms.sum() > 0:
                gene_counts_dms = Counter(dms_meta[active_dms]['gene'])
                card['top_genes_dms'] = dict(gene_counts_dms.most_common(10))

                # DMS correlation for this feature
                if active_dms.sum() > 20:
                    corr = stats.spearmanr(z_dms[active_dms, fid],
                                           dms_meta[active_dms]['dms_score']).statistic
                    card['dms_spearman'] = float(corr) if not np.isnan(corr) else 0.0

        # AA substitution enrichment in ClinVar (from DMS meta if available)
        if z_dms is not None and active_dms.sum() > 0:
            refs = Counter(dms_meta[active_dms]['ref'])
            alts = Counter(dms_meta[active_dms]['alt'])
            card['ref_aa_enrichment'] = dict(refs.most_common(5))
            card['alt_aa_enrichment'] = dict(alts.most_common(5))

        feature_cards.append(card)
        print("  Feature %d: OR=%.2f, q=%.2e, n=%d, path_rate=%.3f" % (
            fid, card['odds_ratio'], card['q_value'], card['n_active_clinvar'], card['path_rate']))
        if 'top_genes_clinvar' in card:
            top3 = list(card['top_genes_clinvar'].items())[:3]
            print("    Top genes (ClinVar): %s" % ', '.join('%s(%d)' % (g, c) for g, c in top3))

    return feature_cards


# ── Analysis 3: Label-stratified SH activation ──

def label_stratified_activation(z_clinvar, y_clinvar, z_dms, dms_y, cats, out_dir):
    """Compare SH activation patterns: pathogenic vs benign, DMS severe vs neutral."""
    print("\n=== Analysis 3: Label-Stratified SH Activation ===", flush=True)

    sh_mask = cats == 'shared'
    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'

    results = {}

    # ClinVar: pathogenic vs benign
    path_idx = y_clinvar == 1
    ben_idx = y_clinvar == 0

    for label, cat_mask, cat_name in [
        ('SH', sh_mask, 'shared'), ('PP', pp_mask, 'prot-private'), ('DP', dp_mask, 'dna-private')
    ]:
        z_cat = z_clinvar[:, cat_mask]
        results['clinvar_%s' % label] = {
            'path_mean_mass': float(z_cat[path_idx].sum(axis=1).mean()),
            'ben_mean_mass': float(z_cat[ben_idx].sum(axis=1).mean()),
            'path_mean_count': float((z_cat[path_idx] > 0).sum(axis=1).mean()),
            'ben_mean_count': float((z_cat[ben_idx] > 0).sum(axis=1).mean()),
            'path_any_active': float((z_cat[path_idx] > 0).any(axis=1).mean()),
            'ben_any_active': float((z_cat[ben_idx] > 0).any(axis=1).mean()),
        }
        r = results['clinvar_%s' % label]
        t_mass, p_mass = stats.mannwhitneyu(
            z_cat[path_idx].sum(axis=1), z_cat[ben_idx].sum(axis=1), alternative='two-sided')
        r['mass_mannwhitney_p'] = float(p_mass)
        print("  ClinVar %s: path mass=%.3f, ben mass=%.3f, MW p=%.2e" % (
            label, r['path_mean_mass'], r['ben_mean_mass'], p_mass))

    # DMS: severe (bottom 10%) vs neutral (middle 40-60%)
    if z_dms is not None and len(dms_y) > 100:
        q10 = np.percentile(dms_y, 10)
        q40 = np.percentile(dms_y, 40)
        q60 = np.percentile(dms_y, 60)
        severe_idx = dms_y < q10
        neutral_idx = (dms_y >= q40) & (dms_y <= q60)

        for label, cat_mask in [('SH', sh_mask), ('PP', pp_mask), ('DP', dp_mask)]:
            z_cat = z_dms[:, cat_mask]
            key = 'dms_%s' % label
            results[key] = {
                'severe_mean_mass': float(z_cat[severe_idx].sum(axis=1).mean()),
                'neutral_mean_mass': float(z_cat[neutral_idx].sum(axis=1).mean()),
                'severe_mean_count': float((z_cat[severe_idx] > 0).sum(axis=1).mean()),
                'neutral_mean_count': float((z_cat[neutral_idx] > 0).sum(axis=1).mean()),
            }
            r = results[key]
            t_mass, p_mass = stats.mannwhitneyu(
                z_cat[severe_idx].sum(axis=1), z_cat[neutral_idx].sum(axis=1),
                alternative='two-sided')
            r['mass_mannwhitney_p'] = float(p_mass)
            print("  DMS %s: severe mass=%.3f, neutral mass=%.3f, MW p=%.2e" % (
                label, r['severe_mean_mass'], r['neutral_mean_mass'], p_mass))

    return results


# ── Analysis 4: ClinVar ablation (PP+SH vs PP-only) ──

def clinvar_feature_ablation(z_clinvar, y_clinvar, cats, out_dir):
    """Compare ClinVar prediction using different feature subsets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print("\n=== Analysis 4: ClinVar Feature Ablation ===", flush=True)

    sh_mask = cats == 'shared'
    pp_mask = cats == 'prot-private'
    dp_mask = cats == 'dna-private'
    alive_mask = cats != 'dead'

    subsets = {
        'all': alive_mask,
        'PP_only': pp_mask,
        'DP_only': dp_mask,
        'SH_only': sh_mask,
        'PP+SH': pp_mask | sh_mask,
        'PP+DP': pp_mask | dp_mask,
        'no_SH': pp_mask | dp_mask,
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, mask in subsets.items():
        if mask.sum() == 0:
            continue
        X = z_clinvar[:, mask]
        aucs = []
        for tr, te in skf.split(X, y_clinvar):
            lr = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs')
            lr.fit(X[tr], y_clinvar[tr])
            prob = lr.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y_clinvar[te], prob)
            aucs.append(auc)
        mean_auc = np.mean(aucs)
        results[name] = {
            'n_features': int(mask.sum()),
            'mean_auroc': float(mean_auc),
            'std_auroc': float(np.std(aucs)),
            'fold_aucs': [float(a) for a in aucs],
        }
        print("  %-10s n_feat=%d, AUROC=%.4f ± %.4f" % (
            name, mask.sum(), mean_auc, np.std(aucs)))

    # Key comparison: PP+SH vs PP-only
    if 'PP+SH' in results and 'PP_only' in results:
        delta = results['PP+SH']['mean_auroc'] - results['PP_only']['mean_auroc']
        results['delta_PP_SH_vs_PP'] = float(delta)
        print("\n  PP+SH vs PP-only: Δ AUROC = %+.4f" % delta)

    return results


# ── Analysis 5: Within-gene shuffle control ──

def within_gene_shuffle_control(model, assay_data, prep, cfg, device, cats, out_dir, n_perm=10):
    """Finer shuffle: permute DNA within the same gene (preserving gene identity)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    print("\n=== Analysis 5: Within-Gene Shuffle Control ===", flush=True)

    sh_mask = cats == 'shared'
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

        # Within-gene shuffle: permute DNA vectors within this assay
        shuf_rhos = []
        shuf_sh_masses = []
        for _ in range(n_perm):
            perm = np.random.permutation(len(dna_pca))
            dt_shuf = torch.FloatTensor(dna_pca[perm]).to(device)
            with torch.no_grad():
                z_shuf = model.encode(pt, dt_shuf).cpu().numpy()

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_rhos = []
            for tr, te in kf.split(z_shuf):
                m = Ridge(alpha=1.0)
                m.fit(z_shuf[tr], ad['y'][tr])
                yp = m.predict(z_shuf[te])
                if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                    r = stats.spearmanr(yp, ad['y'][te]).statistic
                    if not np.isnan(r):
                        fold_rhos.append(r)
            shuf_rhos.append(np.mean(fold_rhos) if fold_rhos else 0.0)
            shuf_sh_masses.append(float(z_shuf[:, sh_mask].sum(axis=1).mean()))

        # Real prediction
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        real_rhos = []
        for tr, te in kf.split(z_real):
            m = Ridge(alpha=1.0)
            m.fit(z_real[tr], ad['y'][tr])
            yp = m.predict(z_real[te])
            if np.std(yp) > 1e-8 and np.std(ad['y'][te]) > 1e-8:
                r = stats.spearmanr(yp, ad['y'][te]).statistic
                if not np.isnan(r):
                    real_rhos.append(r)

        real_rho = np.mean(real_rhos) if real_rhos else 0.0
        real_sh_mass = float(z_real[:, sh_mask].sum(axis=1).mean())

        results.append({
            'assay': ad['name'],
            'gene': ad['name'].split('_')[0],
            'n': len(ad['y']),
            'rho_real': float(real_rho),
            'rho_within_gene_shuf_mean': float(np.mean(shuf_rhos)),
            'rho_within_gene_shuf_std': float(np.std(shuf_rhos)),
            'delta': float(real_rho - np.mean(shuf_rhos)),
            'sh_mass_real': real_sh_mass,
            'sh_mass_shuf_mean': float(np.mean(shuf_sh_masses)),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "within_gene_shuffle.csv"), index=False)

    deltas = df['delta'].values
    t_stat, p_val = stats.ttest_rel(df['rho_real'].values,
                                     df['rho_within_gene_shuf_mean'].values)

    print("  %d assays evaluated" % len(df))
    print("  Real mean rho: %.4f" % df['rho_real'].mean())
    print("  Within-gene shuffled mean rho: %.4f" % df['rho_within_gene_shuf_mean'].mean())
    print("  Delta: %.4f" % deltas.mean())
    print("  Paired t-test: t=%.2f, p=%.2e" % (t_stat, p_val))
    print("  SH mass real: %.4f, shuffled: %.4f" % (
        df['sh_mass_real'].mean(), df['sh_mass_shuf_mean'].mean()))

    summary = {
        'n_assays': len(df),
        'real_mean_rho': float(df['rho_real'].mean()),
        'shuf_mean_rho': float(df['rho_within_gene_shuf_mean'].mean()),
        'delta_mean': float(deltas.mean()),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'sh_mass_real': float(df['sh_mass_real'].mean()),
        'sh_mass_shuf': float(df['sh_mass_shuf_mean'].mean()),
    }
    return df, summary


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/crosscoder_sae/crosscoder.pt")
    ap.add_argument("--emb_dir", default="results/dms_embeddings")
    ap.add_argument("--out_dir", default="results/crosscoder_sae/sh_deepdive")
    ap.add_argument("--clinvar_esm2", default="results/clinvar_esm2")
    ap.add_argument("--clinvar_evo2", default="results/variant")
    ap.add_argument("--clinvar_csv", default="data/full/clinvar_variants.csv")
    ap.add_argument("--clinvar_parquet", default="data/variant/clinvar.parquet")
    ap.add_argument("--clinvar_fdr_csv",
                    default="results/crosscoder_sae/analysis_bhfix/clinvar_feature_pathogenicity_fdr.csv")
    ap.add_argument("--cat_file",
                    default="results/crosscoder_sae/analysis_bhfix/feature_categories.npz")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading checkpoint: %s" % args.checkpoint, flush=True)
    model, cfg, prep = load_checkpoint(args.checkpoint, device)

    print("Loading feature categories: %s" % args.cat_file, flush=True)
    cat_data = np.load(args.cat_file, allow_pickle=True)
    cats = cat_data['categories']
    print("  PP=%d, DP=%d, SH=%d, Dead=%d" % (
        (cats == 'prot-private').sum(), (cats == 'dna-private').sum(),
        (cats == 'shared').sum(), (cats == 'dead').sum()))

    print("Loading DMS data...", flush=True)
    assay_data = load_dms_data(args.emb_dir)
    print("  %d assays loaded" % len(assay_data))

    # Encode all DMS data
    all_z_dms, all_meta = [], []
    for ad in assay_data:
        prot_pca, dna_pca = prep_assay(ad, prep, cfg)
        pt = torch.FloatTensor(prot_pca).to(device)
        dt = torch.FloatTensor(dna_pca).to(device)
        with torch.no_grad():
            z = model.encode(pt, dt).cpu().numpy()
        all_z_dms.append(z)
        for i in range(len(z)):
            mut = str(ad['mutants'][i])
            ref = mut[0] if len(mut) > 0 else ''
            alt = mut[-1] if len(mut) > 0 else ''
            all_meta.append({
                'assay': ad['name'], 'gene': ad['name'].split('_')[0],
                'mutant': mut, 'ref': ref, 'alt': alt,
                'dms_score': float(ad['y'][i]),
            })
    z_dms = np.vstack(all_z_dms)
    dms_meta = pd.DataFrame(all_meta)
    dms_y = dms_meta['dms_score'].values
    print("  DMS: %d total variants" % len(z_dms))

    # Load ClinVar
    print("Loading ClinVar data...", flush=True)
    clinvar = load_clinvar(args.clinvar_esm2, args.clinvar_evo2,
                           args.clinvar_csv, args.clinvar_parquet)
    if clinvar is None:
        print("  ERROR: ClinVar data not found!")
        return

    prot_pca_cv = apply_prep(clinvar['prot'], prep['prot_mean'], prep['prot_std'],
                             prep['pca_prot_components'], prep['pca_prot_mean'])
    dna_pca_cv = apply_prep(clinvar['dna'], prep['dna_mean'], prep['dna_std'],
                            prep['pca_dna_components'], prep['pca_dna_mean'],
                            prep.get('pca_dna_var') if cfg.get('whiten_dna', True) else None)
    pt_cv = torch.FloatTensor(prot_pca_cv).to(device)
    dt_cv = torch.FloatTensor(dna_pca_cv).to(device)
    with torch.no_grad():
        z_clinvar = model.encode(pt_cv, dt_cv).cpu().numpy()

    y_clinvar = clinvar['label']
    genes_clinvar = clinvar['genes']
    print("  ClinVar: %d variants (%d path, %d ben)" % (
        len(y_clinvar), (y_clinvar == 1).sum(), (y_clinvar == 0).sum()))

    # Load ClinVar FDR results
    clinvar_fdr_df = pd.read_csv(args.clinvar_fdr_csv)

    # Run all analyses
    all_results = {}

    # 1. Variant-level SH bins
    all_results['variant_sh_bins'] = variant_level_sh_bins(
        z_clinvar, y_clinvar, genes_clinvar, cats, args.out_dir)

    # 2. Top-10 SH features
    all_results['top_sh_features'] = top_sh_features_deepdive(
        z_clinvar, y_clinvar, genes_clinvar, z_dms, dms_meta, cats, clinvar_fdr_df, args.out_dir)

    # 3. Label-stratified activation
    all_results['label_stratified'] = label_stratified_activation(
        z_clinvar, y_clinvar, z_dms, dms_y, cats, args.out_dir)

    # 4. ClinVar ablation
    all_results['clinvar_ablation'] = clinvar_feature_ablation(
        z_clinvar, y_clinvar, cats, args.out_dir)

    # 5. Within-gene shuffle
    wg_df, wg_summary = within_gene_shuffle_control(
        model, assay_data, prep, cfg, device, cats, args.out_dir)
    all_results['within_gene_shuffle'] = wg_summary

    # Save all results
    out_path = os.path.join(args.out_dir, "sh_deepdive_summary.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n=== All results saved to %s ===" % out_path)


if __name__ == '__main__':
    main()
