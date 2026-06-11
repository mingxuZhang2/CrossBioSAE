"""
Deep-dive: modality discordance analysis on SAE features.

Key questions:
1. Per-gene modality profile: which genes are protein-driven vs DNA-driven vs balanced?
2. Discordant variants: PROT says damaging but DNA says benign (or vice versa)
3. Cys asymmetry: PROT allocates 8/20 features to Cys; what does this mean at variant level?
4. Position-level modality map within individual assays
5. Do PROT-only and DNA-only signals correlate with different structural/functional annotations?
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDROPHOBIC = set('AVILMFWP')
POLAR = set('STCNQDEYKRHG')


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_modality_deepdive")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load encoder weights for modality attribution ──
    import torch
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()  # (n_features, 1536)
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    mod_score = prot_norm / (prot_norm + dna_norm + 1e-8)  # >0.6=PROT, <0.4=DNA

    n_prot_feat = (mod_score > 0.6).sum()
    n_dna_feat = (mod_score < 0.4).sum()
    n_cross_feat = ((mod_score >= 0.4) & (mod_score <= 0.6)).sum()
    print(f"Features: {n_prot_feat} PROT, {n_dna_feat} DNA, {n_cross_feat} CROSS")

    # ── Load all SAE activations ──
    print("Loading SAE activations ...")
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))
    print(f"Found {len(sae_files)} SAE files")

    all_data = []
    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        sae_acts = d["sae_acts"]
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue
        all_data.append((assay_name, mutants, sae_acts, fitness))

    print(f"Loaded {len(all_data)} assays")

    # ── Define feature groups by modality ──
    prot_mask = mod_score > 0.6
    dna_mask = mod_score < 0.4
    cross_mask = (mod_score >= 0.4) & (mod_score <= 0.6)

    # ===================================================================
    # ANALYSIS 1: Per-gene modality profile
    # For each assay: compute mean |activation| of PROT vs DNA features
    # for the most damaging variants (bottom 20% fitness)
    # ===================================================================
    print("\n=== Analysis 1: Per-gene modality profile ===")

    gene_profiles = []
    for assay_name, mutants, sae_acts, fitness in all_data:
        n_var = len(fitness)
        if n_var < 100:
            continue

        # Bottom 20% fitness = most damaging
        threshold = np.percentile(fitness, 20)
        damaging = fitness <= threshold

        # Mean activation of PROT / DNA / CROSS features for damaging variants
        nf = min(sae_acts.shape[1], args.n_features)
        pm = prot_mask[:nf]
        dm = dna_mask[:nf]
        cm = cross_mask[:nf]

        acts_dam = sae_acts[damaging, :nf]
        acts_ben = sae_acts[~damaging, :nf]

        # Mean activation magnitude for damaging variants
        prot_dam = np.mean(np.abs(acts_dam[:, pm])) if pm.sum() > 0 else 0
        dna_dam = np.mean(np.abs(acts_dam[:, dm])) if dm.sum() > 0 else 0
        cross_dam = np.mean(np.abs(acts_dam[:, cm])) if cm.sum() > 0 else 0

        # Same for benign
        prot_ben = np.mean(np.abs(acts_ben[:, pm])) if pm.sum() > 0 else 0
        dna_ben = np.mean(np.abs(acts_ben[:, dm])) if dm.sum() > 0 else 0
        cross_ben = np.mean(np.abs(acts_ben[:, cm])) if cm.sum() > 0 else 0

        # Delta (damaging - benign) = signal from each modality
        prot_delta = prot_dam - prot_ben
        dna_delta = dna_dam - dna_ben
        cross_delta = cross_dam - cross_ben

        # Modality dominance for this gene
        total_delta = abs(prot_delta) + abs(dna_delta) + abs(cross_delta) + 1e-8
        prot_frac = abs(prot_delta) / total_delta
        dna_frac = abs(dna_delta) / total_delta

        gene_name = assay_name.split('_')[0]

        gene_profiles.append({
            'assay': assay_name,
            'gene': gene_name,
            'n_variants': n_var,
            'prot_dam': prot_dam, 'dna_dam': dna_dam, 'cross_dam': cross_dam,
            'prot_ben': prot_ben, 'dna_ben': dna_ben, 'cross_ben': cross_ben,
            'prot_delta': prot_delta, 'dna_delta': dna_delta, 'cross_delta': cross_delta,
            'prot_frac': prot_frac, 'dna_frac': dna_frac,
        })

    gene_df = pd.DataFrame(gene_profiles)
    gene_df.to_csv(os.path.join(args.out_dir, "gene_modality_profiles.csv"), index=False)

    # Classify genes
    gene_df['category'] = 'balanced'
    gene_df.loc[gene_df['prot_frac'] > 0.5, 'category'] = 'protein-driven'
    gene_df.loc[gene_df['dna_frac'] > 0.5, 'category'] = 'DNA-driven'

    print(f"Gene classification:")
    print(gene_df['category'].value_counts())

    # Top protein-driven and DNA-driven genes
    print("\nTop 15 protein-driven genes:")
    prot_genes = gene_df.nlargest(15, 'prot_frac')
    for _, row in prot_genes.iterrows():
        print(f"  {row['assay']:40s} PROT={row['prot_frac']:.1%} DNA={row['dna_frac']:.1%} "
              f"(Δprot={row['prot_delta']:.4f} Δdna={row['dna_delta']:.4f})")

    print("\nTop 15 DNA-driven genes:")
    dna_genes = gene_df.nlargest(15, 'dna_frac')
    for _, row in dna_genes.iterrows():
        print(f"  {row['assay']:40s} DNA={row['dna_frac']:.1%} PROT={row['prot_frac']:.1%} "
              f"(Δdna={row['dna_delta']:.4f} Δprot={row['prot_delta']:.4f})")

    # ===================================================================
    # ANALYSIS 2: Per-variant modality decomposition & discordance
    # For each variant: compute PROT-signal vs DNA-signal
    # Find variants where they disagree
    # ===================================================================
    print("\n=== Analysis 2: Variant-level modality discordance ===")

    # For each variant, compute:
    #   prot_signal = sum(activation * sign(fitness_correlation)) for PROT features
    #   dna_signal  = sum(activation * sign(fitness_correlation)) for DNA features
    # Then find discordant variants

    # First: compute per-feature fitness direction (positive rho = protective, negative = damaging)
    print("Computing per-feature fitness correlation direction ...")
    feat_rho = np.zeros(args.n_features)
    feat_rho_count = np.zeros(args.n_features)
    for assay_name, mutants, sae_acts, fitness in all_data:
        if len(fitness) < 50:
            continue
        nf = min(sae_acts.shape[1], args.n_features)
        for fi in range(nf):
            col = sae_acts[:, fi]
            if (col > 0).sum() < 10:
                continue
            rho = stats.spearmanr(col, fitness).statistic
            if not np.isnan(rho):
                feat_rho[fi] += rho
                feat_rho_count[fi] += 1

    feat_mean_rho = np.divide(feat_rho, feat_rho_count,
                               where=feat_rho_count > 0,
                               out=np.zeros_like(feat_rho))

    # Variant-level: weighted activation
    # Use top 100 PROT and top 100 DNA features with strongest signal
    prot_indices = np.where(prot_mask)[0]
    dna_indices = np.where(dna_mask)[0]
    cross_indices = np.where(cross_mask)[0]

    # Rank by |mean_rho|
    prot_ranked = sorted(prot_indices, key=lambda i: -abs(feat_mean_rho[i]))[:100]
    dna_ranked = sorted(dna_indices, key=lambda i: -abs(feat_mean_rho[i]))[:100]

    print(f"Using top {len(prot_ranked)} PROT features and {len(dna_ranked)} DNA features")

    # Feature weight = sign of rho (damaging = negative)
    prot_weights = np.array([feat_mean_rho[i] for i in prot_ranked])
    dna_weights = np.array([feat_mean_rho[i] for i in dna_ranked])

    discordant_variants = []
    concordant_damaging = []
    concordant_benign = []
    all_variant_signals = []

    for assay_name, mutants, sae_acts, fitness in all_data:
        if len(fitness) < 50:
            continue
        nf = min(sae_acts.shape[1], args.n_features)
        gene_name = assay_name.split('_')[0]

        for i in range(len(mutants)):
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa is None:
                continue

            # Compute PROT and DNA signals
            prot_acts = np.array([sae_acts[i, fi] for fi in prot_ranked if fi < nf])
            dna_acts = np.array([sae_acts[i, fi] for fi in dna_ranked if fi < nf])

            pw = prot_weights[:len(prot_acts)]
            dw = dna_weights[:len(dna_acts)]

            # Weighted signal: positive = protective, negative = damaging
            prot_signal = np.dot(prot_acts, pw) / (np.abs(pw).sum() + 1e-8)
            dna_signal = np.dot(dna_acts, dw) / (np.abs(dw).sum() + 1e-8)

            record = {
                'assay': assay_name, 'gene': gene_name,
                'mutant': str(mutants[i]),
                'ref_aa': ref_aa, 'pos': pos, 'alt_aa': alt_aa,
                'fitness': fitness[i],
                'prot_signal': prot_signal, 'dna_signal': dna_signal,
            }
            all_variant_signals.append(record)

    var_df = pd.DataFrame(all_variant_signals)
    print(f"Total variants analyzed: {len(var_df)}")

    # Normalize signals per-assay (z-score within assay)
    for assay in var_df['assay'].unique():
        mask = var_df['assay'] == assay
        for col in ['prot_signal', 'dna_signal']:
            vals = var_df.loc[mask, col]
            mu, sd = vals.mean(), vals.std()
            if sd > 1e-8:
                var_df.loc[mask, f'{col}_z'] = (vals - mu) / sd
            else:
                var_df.loc[mask, f'{col}_z'] = 0

    # Discordance = PROT says damaging (z < -1) but DNA says benign (z > 0), or vice versa
    prot_dam_dna_ben = var_df[(var_df['prot_signal_z'] < -1.0) & (var_df['dna_signal_z'] > 0.5)]
    dna_dam_prot_ben = var_df[(var_df['dna_signal_z'] < -1.0) & (var_df['prot_signal_z'] > 0.5)]
    concordant_dam = var_df[(var_df['prot_signal_z'] < -1.0) & (var_df['dna_signal_z'] < -1.0)]

    print(f"\nDiscordant: PROT-damaging & DNA-benign: {len(prot_dam_dna_ben)} variants")
    print(f"Discordant: DNA-damaging & PROT-benign: {len(dna_dam_prot_ben)} variants")
    print(f"Concordant damaging: {len(concordant_dam)} variants")

    # What amino acid substitutions dominate each discordant class?
    print("\n--- PROT-damaging & DNA-benign: amino acid breakdown ---")
    if len(prot_dam_dna_ben) > 0:
        ref_counts = Counter(prot_dam_dna_ben['ref_aa'])
        print(f"  Ref AA: {ref_counts.most_common(10)}")
        alt_counts = Counter(prot_dam_dna_ben['alt_aa'])
        print(f"  Alt AA: {alt_counts.most_common(10)}")
        # Gene breakdown
        gene_counts = Counter(prot_dam_dna_ben['gene'])
        print(f"  Top genes: {gene_counts.most_common(10)}")
        # Mean fitness
        print(f"  Mean fitness: {prot_dam_dna_ben['fitness'].mean():.4f} "
              f"(vs global mean {var_df['fitness'].mean():.4f})")

    print("\n--- DNA-damaging & PROT-benign: amino acid breakdown ---")
    if len(dna_dam_prot_ben) > 0:
        ref_counts = Counter(dna_dam_prot_ben['ref_aa'])
        print(f"  Ref AA: {ref_counts.most_common(10)}")
        alt_counts = Counter(dna_dam_prot_ben['alt_aa'])
        print(f"  Alt AA: {alt_counts.most_common(10)}")
        gene_counts = Counter(dna_dam_prot_ben['gene'])
        print(f"  Top genes: {gene_counts.most_common(10)}")
        print(f"  Mean fitness: {dna_dam_prot_ben['fitness'].mean():.4f}")

    # ===================================================================
    # ANALYSIS 3: Cys asymmetry deep dive
    # PROT has 8/20 features dedicated to Cys
    # Compare Cys variants where PROT fires vs DNA fires
    # ===================================================================
    print("\n=== Analysis 3: Cysteine asymmetry deep dive ===")

    cys_variants = var_df[var_df['ref_aa'] == 'C'].copy()
    print(f"Total Cys→X variants: {len(cys_variants)}")

    if len(cys_variants) > 0:
        # Classify Cys variants by modality signal
        cys_prot_only = cys_variants[(cys_variants['prot_signal_z'] < -1) &
                                      (cys_variants['dna_signal_z'] > -0.5)]
        cys_dna_only = cys_variants[(cys_variants['dna_signal_z'] < -1) &
                                     (cys_variants['prot_signal_z'] > -0.5)]
        cys_both = cys_variants[(cys_variants['prot_signal_z'] < -1) &
                                 (cys_variants['dna_signal_z'] < -1)]

        print(f"  PROT-only damaging: {len(cys_prot_only)} "
              f"(mean fitness={cys_prot_only['fitness'].mean():.4f})")
        print(f"  DNA-only damaging:  {len(cys_dna_only)} "
              f"(mean fitness={cys_dna_only['fitness'].mean():.4f})")
        print(f"  Both damaging:      {len(cys_both)} "
              f"(mean fitness={cys_both['fitness'].mean():.4f})")

        # What alt AAs in each class?
        if len(cys_prot_only) > 10:
            print(f"  PROT-only Cys→?: {Counter(cys_prot_only['alt_aa']).most_common(5)}")
            print(f"  PROT-only genes: {Counter(cys_prot_only['gene']).most_common(5)}")
        if len(cys_dna_only) > 10:
            print(f"  DNA-only Cys→?:  {Counter(cys_dna_only['alt_aa']).most_common(5)}")
            print(f"  DNA-only genes:  {Counter(cys_dna_only['gene']).most_common(5)}")

        # Is PROT-only Cys damage actually more damaging on average?
        # This would support: ESM-2 detects Cys structural role (disulfide etc.)
        # that evolutionary conservation alone doesn't capture
        if len(cys_prot_only) > 10 and len(cys_dna_only) > 10:
            stat, pval = stats.mannwhitneyu(cys_prot_only['fitness'],
                                             cys_dna_only['fitness'],
                                             alternative='two-sided')
            print(f"  PROT-only vs DNA-only fitness: U={stat:.0f}, p={pval:.2e}")

    # ===================================================================
    # ANALYSIS 4: Per-position modality map for select assays
    # For top DMS assays, compute PROT vs DNA signal at each position
    # ===================================================================
    print("\n=== Analysis 4: Position-level modality maps ===")

    # Pick assays with most variants and clear modality signal
    big_assays = gene_df.nlargest(20, 'n_variants')['assay'].tolist()

    for assay_name in big_assays[:10]:
        assay_var = var_df[var_df['assay'] == assay_name]
        if len(assay_var) < 200:
            continue

        # Per-position: mean PROT and DNA signal
        pos_data = defaultdict(lambda: {'prot': [], 'dna': [], 'fitness': []})
        for _, row in assay_var.iterrows():
            if row['pos'] is not None:
                pos_data[row['pos']]['prot'].append(row['prot_signal_z'])
                pos_data[row['pos']]['dna'].append(row['dna_signal_z'])
                pos_data[row['pos']]['fitness'].append(row['fitness'])

        positions = sorted(pos_data.keys())
        if len(positions) < 20:
            continue

        # Find positions with strongest discordance
        pos_disc = []
        for p in positions:
            pm = np.mean(pos_data[p]['prot'])
            dm = np.mean(pos_data[p]['dna'])
            fm = np.mean(pos_data[p]['fitness'])
            disc = pm - dm  # positive = PROT more damaging, negative = DNA more damaging
            pos_disc.append((p, pm, dm, disc, fm, len(pos_data[p]['prot'])))

        pos_disc.sort(key=lambda x: x[3])

        gene_name = assay_name.split('_')[0]
        print(f"\n  {assay_name} ({len(assay_var)} variants, {len(positions)} positions)")

        # Top 5 DNA-dominant positions
        print(f"    Top 5 DNA-dominant positions (DNA says damaging, PROT doesn't):")
        for p, pm, dm, disc, fm, n in pos_disc[:5]:
            print(f"      pos {p:4d}: PROT_z={pm:+.2f}, DNA_z={dm:+.2f}, "
                  f"disc={disc:+.2f}, fitness={fm:.3f}, n={n}")

        # Top 5 PROT-dominant positions
        print(f"    Top 5 PROT-dominant positions (PROT says damaging, DNA doesn't):")
        for p, pm, dm, disc, fm, n in reversed(pos_disc[-5:]):
            print(f"      pos {p:4d}: PROT_z={pm:+.2f}, DNA_z={dm:+.2f}, "
                  f"disc={disc:+.2f}, fitness={fm:.3f}, n={n}")

    # ===================================================================
    # ANALYSIS 5: Discordance vs actual fitness
    # Key question: when PROT and DNA disagree, who is right?
    # ===================================================================
    print("\n=== Analysis 5: Who is right when they disagree? ===")

    # Bin variants by concordance/discordance and compare fitness
    var_df['discord_class'] = 'neutral'
    var_df.loc[(var_df['prot_signal_z'] < -1) & (var_df['dna_signal_z'] < -1),
               'discord_class'] = 'both_damaging'
    var_df.loc[(var_df['prot_signal_z'] < -1) & (var_df['dna_signal_z'] > 0.5),
               'discord_class'] = 'prot_only_dam'
    var_df.loc[(var_df['dna_signal_z'] < -1) & (var_df['prot_signal_z'] > 0.5),
               'discord_class'] = 'dna_only_dam'
    var_df.loc[(var_df['prot_signal_z'] > 0.5) & (var_df['dna_signal_z'] > 0.5),
               'discord_class'] = 'both_benign'

    discord_stats = var_df.groupby('discord_class')['fitness'].agg(['mean', 'std', 'count'])
    print(discord_stats.sort_values('mean'))

    # Statistical tests
    for cls in ['prot_only_dam', 'dna_only_dam']:
        subset = var_df[var_df['discord_class'] == cls]['fitness']
        both_dam = var_df[var_df['discord_class'] == 'both_damaging']['fitness']
        both_ben = var_df[var_df['discord_class'] == 'both_benign']['fitness']

        if len(subset) > 20 and len(both_ben) > 20:
            stat, pval = stats.mannwhitneyu(subset, both_ben, alternative='less')
            print(f"\n  {cls} vs both_benign: U={stat:.0f}, p={pval:.2e}, "
                  f"mean_fit={subset.mean():.4f} vs {both_ben.mean():.4f}")

        if len(subset) > 20 and len(both_dam) > 20:
            stat, pval = stats.mannwhitneyu(subset, both_dam, alternative='greater')
            print(f"  {cls} vs both_damaging: U={stat:.0f}, p={pval:.2e}, "
                  f"mean_fit={subset.mean():.4f} vs {both_dam.mean():.4f}")

    # ===================================================================
    # ANALYSIS 6: Feature-level cross-modal cooperation
    # Which specific PROT features and DNA features are complementary?
    # (i.e., anti-correlated: when one fires, the other doesn't)
    # ===================================================================
    print("\n=== Analysis 6: Cross-modal feature complementarity ===")

    # Sample a few assays to compute cross-correlation
    sample_assays = [d for d in all_data if len(d[3]) > 500][:30]

    # Compute pairwise correlation between top PROT and DNA features
    # across all variants in sampled assays
    top_prot = prot_ranked[:30]
    top_dna = dna_ranked[:30]

    all_prot_acts = []
    all_dna_acts = []
    for assay_name, mutants, sae_acts, fitness in sample_assays:
        nf = min(sae_acts.shape[1], args.n_features)
        p_acts = np.array([[sae_acts[i, fi] for fi in top_prot if fi < nf]
                           for i in range(len(mutants))])
        d_acts = np.array([[sae_acts[i, fi] for fi in top_dna if fi < nf]
                           for i in range(len(mutants))])
        all_prot_acts.append(p_acts)
        all_dna_acts.append(d_acts)

    all_prot_acts = np.vstack(all_prot_acts)
    all_dna_acts = np.vstack(all_dna_acts)

    print(f"Computing cross-modal correlation on {len(all_prot_acts)} variants ...")

    # Compute correlation matrix (30 PROT x 30 DNA)
    cross_corr = np.zeros((len(top_prot), len(top_dna)))
    for pi in range(len(top_prot)):
        for di in range(len(top_dna)):
            rho = stats.spearmanr(all_prot_acts[:, pi], all_dna_acts[:, di]).statistic
            if not np.isnan(rho):
                cross_corr[pi, di] = rho

    # Find most anti-correlated pairs (complementary)
    pairs = []
    for pi in range(len(top_prot)):
        for di in range(len(top_dna)):
            pairs.append((top_prot[pi], top_dna[di], cross_corr[pi, di]))

    pairs.sort(key=lambda x: x[2])

    print("\nMost anti-correlated PROT-DNA feature pairs (complementary):")
    for pf, df, rho in pairs[:10]:
        print(f"  PROT F{pf:05d} × DNA F{df:05d}: rho={rho:.4f}"
              f" (prot_mod={mod_score[pf]:.2f}, dna_mod={mod_score[df]:.2f},"
              f" prot_rho={feat_mean_rho[pf]:.3f}, dna_rho={feat_mean_rho[df]:.3f})")

    print("\nMost positively correlated PROT-DNA feature pairs (redundant):")
    pairs.sort(key=lambda x: -x[2])
    for pf, df, rho in pairs[:10]:
        print(f"  PROT F{pf:05d} × DNA F{df:05d}: rho={rho:.4f}")

    # ===================================================================
    # ANALYSIS 7: Amino acid specificity by discordance class
    # Are certain substitution types inherently discordant?
    # ===================================================================
    print("\n=== Analysis 7: Which substitutions cause discordance? ===")

    for aa in AA_ORDER:
        ref_subset = var_df[var_df['ref_aa'] == aa]
        if len(ref_subset) < 200:
            continue
        discord_frac = (ref_subset['discord_class'].isin(
            ['prot_only_dam', 'dna_only_dam'])).mean()
        prot_only = (ref_subset['discord_class'] == 'prot_only_dam').mean()
        dna_only = (ref_subset['discord_class'] == 'dna_only_dam').mean()
        if discord_frac > 0.05:
            print(f"  Ref={aa}: {discord_frac:.1%} discordant "
                  f"(PROT-only={prot_only:.1%}, DNA-only={dna_only:.1%}, n={len(ref_subset)})")

    # Same for alt AA
    print("\nBy alt AA:")
    for aa in AA_ORDER:
        alt_subset = var_df[var_df['alt_aa'] == aa]
        if len(alt_subset) < 200:
            continue
        prot_only = (alt_subset['discord_class'] == 'prot_only_dam').mean()
        dna_only = (alt_subset['discord_class'] == 'dna_only_dam').mean()
        if prot_only > 0.03 or dna_only > 0.03:
            print(f"  Alt={aa}: PROT-only={prot_only:.1%}, DNA-only={dna_only:.1%}, "
                  f"n={len(alt_subset)}")

    # ===================================================================
    # ANALYSIS 8: Does discordance predict functional category?
    # Hypothesis: PROT-only damaging = structural damage (buried, disulfide)
    #             DNA-only damaging = codon/regulatory constraint
    # Test: are PROT-only damage variants at more hydrophobic ref positions?
    # ===================================================================
    print("\n=== Analysis 8: Structural context of discordant variants ===")

    for cls in ['prot_only_dam', 'dna_only_dam', 'both_damaging']:
        subset = var_df[var_df['discord_class'] == cls]
        if len(subset) < 50:
            continue
        hydro_ref_frac = subset['ref_aa'].isin(list(HYDROPHOBIC)).mean()
        polar_ref_frac = subset['ref_aa'].isin(list(POLAR)).mean()
        cys_ref_frac = (subset['ref_aa'] == 'C').mean()
        gly_ref_frac = (subset['ref_aa'] == 'G').mean()
        pro_alt_frac = (subset['alt_aa'] == 'P').mean()

        print(f"\n  {cls} (n={len(subset)}):")
        print(f"    Hydrophobic ref: {hydro_ref_frac:.1%}")
        print(f"    Polar ref: {polar_ref_frac:.1%}")
        print(f"    Cys ref: {cys_ref_frac:.1%}")
        print(f"    Gly ref: {gly_ref_frac:.1%}")
        print(f"    Pro alt (→P): {pro_alt_frac:.1%}")

    # ===================================================================
    # Save full variant-level data
    # ===================================================================
    var_df.to_csv(os.path.join(args.out_dir, "variant_modality_signals.csv.gz"),
                  index=False, compression='gzip')
    print(f"\nSaved variant signals to {args.out_dir}/variant_modality_signals.csv.gz")

    # ===================================================================
    # Summary report
    # ===================================================================
    report_path = os.path.join(args.out_dir, "modality_deepdive_report.txt")
    with open(report_path, 'w') as f:
        import io, sys
        # Re-run key prints to file
        f.write("MODALITY DEEP-DIVE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Gene classification:\n")
        f.write(gene_df['category'].value_counts().to_string() + "\n\n")

        f.write("Discordance stats:\n")
        f.write(discord_stats.sort_values('mean').to_string() + "\n\n")

        f.write("Top 20 protein-driven genes:\n")
        for _, row in gene_df.nlargest(20, 'prot_frac').iterrows():
            f.write(f"  {row['assay']:45s} PROT={row['prot_frac']:.1%} "
                    f"Δprot={row['prot_delta']:.5f} Δdna={row['dna_delta']:.5f}\n")

        f.write("\nTop 20 DNA-driven genes:\n")
        for _, row in gene_df.nlargest(20, 'dna_frac').iterrows():
            f.write(f"  {row['assay']:45s} DNA={row['dna_frac']:.1%} "
                    f"Δdna={row['dna_delta']:.5f} Δprot={row['prot_delta']:.5f}\n")

    print(f"\nReport saved to {report_path}")
    print("DONE")


if __name__ == "__main__":
    main()
