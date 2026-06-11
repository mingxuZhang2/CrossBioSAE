"""
Interpretable variant effect prediction v2: modality-first decomposition.

Level 1: PROT / DNA / CROSS contribution (from encoder weight modality)
Level 2: Within each modality, what specific mechanism (Cys loss, Pro intro, etc.)

For each variant:
  "PTEN G132C is pathogenic:
    Modality: PROT=60% | DNA=30% | CROSS=10%
    PROT mechanism: Gly loss (backbone flexibility)
    DNA mechanism: evolutionary conservation
    → Conclusion: structural + conservation dual constraint"

This is what ONLY cross-modal SAE can do — no single-model method can decompose like this.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDROPHOBIC = set('AVILMFWP')
CHARGE = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def classify_substitution(ref_aa, alt_aa):
    """Human-readable substitution mechanism label."""
    if ref_aa == 'C':
        return 'Cys_loss'
    if alt_aa == 'C':
        return 'Cys_gain'
    if alt_aa == 'P':
        return 'Pro_intro'
    if ref_aa == 'P':
        return 'Pro_loss'
    if ref_aa == 'G':
        return 'Gly_loss'
    if ref_aa == 'W':
        return 'Trp_loss'
    ref_c = CHARGE.get(ref_aa, 0)
    alt_c = CHARGE.get(alt_aa, 0)
    if ref_c * alt_c < 0:
        return 'charge_reversal'
    if ref_c == 0 and abs(alt_c) > 0:
        return 'charge_intro'
    if abs(ref_c) > 0 and alt_c == 0:
        return 'charge_loss'
    if ref_aa in HYDROPHOBIC and alt_aa not in HYDROPHOBIC:
        return 'hydro→polar'
    if ref_aa not in HYDROPHOBIC and alt_aa in HYDROPHOBIC:
        return 'polar→hydro'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_interpretable_v2")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load encoder weights for modality attribution ──
    import torch
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()  # (n_features, 1536)

    # Modality score per feature
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    total_norm = prot_norm + dna_norm + 1e-8
    # Continuous: fraction of weight from protein side
    prot_frac_per_feat = prot_norm / total_norm  # 0=pure DNA, 1=pure PROT
    dna_frac_per_feat = dna_norm / total_norm

    n_prot = (prot_frac_per_feat > 0.6).sum()
    n_dna = (prot_frac_per_feat < 0.4).sum()
    n_cross = args.n_features - n_prot - n_dna
    print(f"Features: {n_prot} PROT, {n_dna} DNA, {n_cross} CROSS", flush=True)

    # ── Load SAE activations ──
    print("Loading SAE activations ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    all_assay_names = []
    all_sae_acts = []
    all_mutants = []
    all_fitness = []

    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        sae_acts = d["sae_acts"]
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue
        all_assay_names.append(assay_name)
        all_sae_acts.append(sae_acts)
        all_mutants.append(mutants)
        all_fitness.append(fitness)

    print(f"Loaded {len(all_assay_names)} assays", flush=True)

    # ── Per-feature fitness correlation (for weighting importance) ──
    print("Computing per-feature fitness correlation ...", flush=True)
    feat_rho_sum = np.zeros(args.n_features)
    feat_rho_count = np.zeros(args.n_features)
    for sae_acts, fitness in zip(all_sae_acts, all_fitness):
        if len(fitness) < 50:
            continue
        nf = min(sae_acts.shape[1], args.n_features)
        for fi in range(nf):
            col = sae_acts[:, fi]
            if (col > 0).sum() < 10:
                continue
            rho = stats.spearmanr(col, fitness).statistic
            if not np.isnan(rho):
                feat_rho_sum[fi] += rho
                feat_rho_count[fi] += 1

    feat_mean_rho = np.divide(feat_rho_sum, feat_rho_count,
                               where=feat_rho_count > 0,
                               out=np.zeros_like(feat_rho_sum))
    feat_importance = np.abs(feat_mean_rho)  # |rho| = how much this feature matters

    print(f"Features with signal (count>20): {(feat_rho_count > 20).sum()}", flush=True)

    # ===================================================================
    # Per-variant modality-first decomposition
    # For each variant and each active feature:
    #   contribution = activation * |rho| (importance-weighted)
    #   split into PROT / DNA portion using encoder weight ratio
    # ===================================================================
    print("\n=== Per-variant modality decomposition ===", flush=True)

    # Precompute: for each feature, its PROT and DNA weight fractions
    feat_prot_w = prot_frac_per_feat  # continuous, 0-1
    feat_dna_w = dna_frac_per_feat

    # Only use features with meaningful signal
    useful_mask = feat_rho_count > 20  # active in >20 assays
    useful_indices = np.where(useful_mask)[0]
    print(f"Using {len(useful_indices)} features for decomposition", flush=True)

    variant_records = []

    for assay_idx in range(len(all_assay_names)):
        assay_name = all_assay_names[assay_idx]
        sae_acts = all_sae_acts[assay_idx]
        mutants = all_mutants[assay_idx]
        fitness = all_fitness[assay_idx]
        gene_name = assay_name.split('_')[0]
        nf = min(sae_acts.shape[1], args.n_features)

        for i in range(len(mutants)):
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa is None:
                continue

            acts = sae_acts[i, :nf]

            # For each active feature: contribution = act * importance
            # Split by modality using encoder weight fraction
            prot_contrib = 0.0
            dna_contrib = 0.0
            cross_contrib = 0.0
            total_contrib = 0.0

            # Top feature tracking (by modality)
            top_prot_feat = (-1, 0.0)
            top_dna_feat = (-1, 0.0)
            top_cross_feat = (-1, 0.0)

            for fi in useful_indices:
                if fi >= nf or acts[fi] <= 0:
                    continue
                contrib = acts[fi] * feat_importance[fi]
                if contrib <= 0:
                    continue

                # Split contribution by modality weight
                p_frac = feat_prot_w[fi]
                d_frac = feat_dna_w[fi]

                # Three-way split: PROT portion, DNA portion, CROSS portion
                # CROSS = min(p_frac, d_frac) * 2 (the shared part)
                # Remaining goes to dominant modality
                cross_portion = min(p_frac, d_frac) * 2
                if p_frac > d_frac:
                    prot_portion = 1.0 - cross_portion
                    dna_portion = 0.0
                else:
                    dna_portion = 1.0 - cross_portion
                    prot_portion = 0.0

                prot_contrib += contrib * prot_portion
                dna_contrib += contrib * dna_portion
                cross_contrib += contrib * cross_portion
                total_contrib += contrib

                # Track top features
                if p_frac > 0.6 and contrib > top_prot_feat[1]:
                    top_prot_feat = (fi, contrib)
                elif p_frac < 0.4 and contrib > top_dna_feat[1]:
                    top_dna_feat = (fi, contrib)
                elif 0.4 <= p_frac <= 0.6 and contrib > top_cross_feat[1]:
                    top_cross_feat = (fi, contrib)

            if total_contrib < 1e-10:
                prot_pct = dna_pct = cross_pct = 0.0
            else:
                prot_pct = prot_contrib / total_contrib
                dna_pct = dna_contrib / total_contrib
                cross_pct = cross_contrib / total_contrib

            sub_type = classify_substitution(ref_aa, alt_aa) if ref_aa and alt_aa else ''

            record = {
                'assay': assay_name,
                'gene': gene_name,
                'mutant': str(mutants[i]),
                'ref_aa': ref_aa,
                'pos': pos,
                'alt_aa': alt_aa,
                'fitness': fitness[i],
                'prot_pct': prot_pct,
                'dna_pct': dna_pct,
                'cross_pct': cross_pct,
                'total_signal': total_contrib,
                'substitution_type': sub_type,
                'top_prot_feature': f'F{top_prot_feat[0]:05d}' if top_prot_feat[0] >= 0 else '',
                'top_dna_feature': f'F{top_dna_feat[0]:05d}' if top_dna_feat[0] >= 0 else '',
                'top_cross_feature': f'F{top_cross_feat[0]:05d}' if top_cross_feat[0] >= 0 else '',
            }
            variant_records.append(record)

        if (assay_idx + 1) % 50 == 0:
            print(f"  Processed {assay_idx + 1}/{len(all_assay_names)} assays ...", flush=True)

    var_df = pd.DataFrame(variant_records)
    print(f"\nTotal variants: {len(var_df)}", flush=True)

    # Per-assay fitness percentile
    var_df['fitness_pctile'] = 0.0
    for assay in var_df['assay'].unique():
        mask = var_df['assay'] == assay
        var_df.loc[mask, 'fitness_pctile'] = var_df.loc[mask, 'fitness'].rank(pct=True)

    # Save
    var_df.to_csv(os.path.join(args.out_dir, "variant_modality_decomp.csv.gz"),
                  index=False, compression='gzip')

    # ===================================================================
    # Analysis 1: Modality profile by fitness quantile
    # ===================================================================
    print("\n=== Modality profile by fitness quantile ===", flush=True)

    bins = [(0, 0.05, 'most_damaging_5%'),
            (0.05, 0.20, 'damaging_5-20%'),
            (0.20, 0.50, 'moderate_20-50%'),
            (0.50, 0.80, 'neutral_50-80%'),
            (0.80, 1.01, 'benign_80-100%')]

    print(f"{'Quantile':22s} {'PROT%':>7s} {'DNA%':>7s} {'CROSS%':>7s} {'N':>8s}", flush=True)
    for lo, hi, label in bins:
        mask = (var_df['fitness_pctile'] >= lo) & (var_df['fitness_pctile'] < hi)
        subset = var_df[mask]
        if len(subset) > 0:
            print(f"{label:22s} {subset['prot_pct'].mean():7.1%} "
                  f"{subset['dna_pct'].mean():7.1%} "
                  f"{subset['cross_pct'].mean():7.1%} "
                  f"{len(subset):8d}", flush=True)

    # ===================================================================
    # Analysis 2: Per-gene modality profile (most damaging variants)
    # ===================================================================
    print("\n=== Per-gene dominant modality (bottom 10% fitness) ===", flush=True)

    dam = var_df[var_df['fitness_pctile'] < 0.10]
    gene_profile = dam.groupby('gene')[['prot_pct', 'dna_pct', 'cross_pct']].mean()
    gene_profile['n'] = dam.groupby('gene').size()
    gene_profile['dominant'] = gene_profile[['prot_pct', 'dna_pct', 'cross_pct']].idxmax(axis=1)

    # Classification counts
    dom_counts = gene_profile['dominant'].value_counts()
    print(f"Dominant modality distribution:", flush=True)
    for mod, cnt in dom_counts.items():
        print(f"  {mod}: {cnt} genes", flush=True)

    # Top protein-driven
    print(f"\nTop 15 PROT-driven genes:", flush=True)
    prot_genes = gene_profile.nlargest(15, 'prot_pct')
    for gene, row in prot_genes.iterrows():
        print(f"  {gene:20s}: PROT={row['prot_pct']:.1%} DNA={row['dna_pct']:.1%} "
              f"CROSS={row['cross_pct']:.1%} (n={row['n']:.0f})", flush=True)

    # Top DNA-driven
    print(f"\nTop 15 DNA-driven genes:", flush=True)
    dna_genes = gene_profile.nlargest(15, 'dna_pct')
    for gene, row in dna_genes.iterrows():
        print(f"  {gene:20s}: DNA={row['dna_pct']:.1%} PROT={row['prot_pct']:.1%} "
              f"CROSS={row['cross_pct']:.1%} (n={row['n']:.0f})", flush=True)

    # Top CROSS-driven
    print(f"\nTop 15 CROSS-driven genes:", flush=True)
    cross_genes = gene_profile.nlargest(15, 'cross_pct')
    for gene, row in cross_genes.iterrows():
        print(f"  {gene:20s}: CROSS={row['cross_pct']:.1%} PROT={row['prot_pct']:.1%} "
              f"DNA={row['dna_pct']:.1%} (n={row['n']:.0f})", flush=True)

    # ===================================================================
    # Analysis 3: Example interpretable predictions (known genes)
    # ===================================================================
    print("\n=== Example interpretable predictions ===", flush=True)

    example_genes = ['BRCA1', 'P53', 'PTEN', 'MSH2', 'SRC', 'CASP3',
                     'TP53', 'GFP', 'SPIKE', 'ENV']

    for gene in example_genes:
        gene_var = var_df[var_df['gene'] == gene]
        if len(gene_var) < 20:
            continue

        avg = gene_var[['prot_pct', 'dna_pct', 'cross_pct']].mean()
        print(f"\n  === {gene} ({len(gene_var)} variants) ===", flush=True)
        print(f"  Average: PROT={avg['prot_pct']:.1%} DNA={avg['dna_pct']:.1%} "
              f"CROSS={avg['cross_pct']:.1%}", flush=True)

        # Most damaging
        most_dam = gene_var.nsmallest(5, 'fitness')
        print(f"  Most damaging:", flush=True)
        for _, row in most_dam.iterrows():
            print(f"    {row['mutant']:10s} fit={row['fitness']:8.3f} | "
                  f"PROT={row['prot_pct']:.0%} DNA={row['dna_pct']:.0%} "
                  f"CROSS={row['cross_pct']:.0%} | "
                  f"{row['substitution_type']:15s} "
                  f"prot_F={row['top_prot_feature']} "
                  f"dna_F={row['top_dna_feature']}", flush=True)

        # Most benign
        most_ben = gene_var.nlargest(3, 'fitness')
        print(f"  Most benign:", flush=True)
        for _, row in most_ben.iterrows():
            print(f"    {row['mutant']:10s} fit={row['fitness']:8.3f} | "
                  f"PROT={row['prot_pct']:.0%} DNA={row['dna_pct']:.0%} "
                  f"CROSS={row['cross_pct']:.0%} | "
                  f"{row['substitution_type']:15s}", flush=True)

    # ===================================================================
    # Analysis 4: Does modality profile change with fitness?
    # Within same gene, do damaging vs benign variants differ in modality?
    # ===================================================================
    print("\n=== Modality shift: damaging vs benign within genes ===", flush=True)

    shifts = []
    for gene in gene_profile.index:
        gene_var = var_df[var_df['gene'] == gene]
        if len(gene_var) < 100:
            continue
        dam_v = gene_var[gene_var['fitness_pctile'] < 0.2]
        ben_v = gene_var[gene_var['fitness_pctile'] > 0.8]
        if len(dam_v) < 20 or len(ben_v) < 20:
            continue

        shift_prot = dam_v['prot_pct'].mean() - ben_v['prot_pct'].mean()
        shift_dna = dam_v['dna_pct'].mean() - ben_v['dna_pct'].mean()
        shift_cross = dam_v['cross_pct'].mean() - ben_v['cross_pct'].mean()

        shifts.append({
            'gene': gene,
            'shift_prot': shift_prot,
            'shift_dna': shift_dna,
            'shift_cross': shift_cross,
            'dam_prot': dam_v['prot_pct'].mean(),
            'ben_prot': ben_v['prot_pct'].mean(),
            'dam_dna': dam_v['dna_pct'].mean(),
            'ben_dna': ben_v['dna_pct'].mean(),
        })

    shift_df = pd.DataFrame(shifts)
    if len(shift_df) > 0:
        # Genes where damaging variants have MORE protein signal
        print(f"Genes where damaging = more PROT signal:", flush=True)
        prot_shift = shift_df.nlargest(10, 'shift_prot')
        for _, row in prot_shift.iterrows():
            print(f"  {row['gene']:20s}: dam_PROT={row['dam_prot']:.1%} → "
                  f"ben_PROT={row['ben_prot']:.1%} (shift={row['shift_prot']:+.1%})", flush=True)

        print(f"\nGenes where damaging = more DNA signal:", flush=True)
        dna_shift = shift_df.nlargest(10, 'shift_dna')
        for _, row in dna_shift.iterrows():
            print(f"  {row['gene']:20s}: dam_DNA={row['dam_dna']:.1%} → "
                  f"ben_DNA={row['ben_dna']:.1%} (shift={row['shift_dna']:+.1%})", flush=True)

    # ===================================================================
    # Analysis 5: Substitution type × modality
    # Do certain substitution types consistently activate one modality?
    # ===================================================================
    print("\n=== Substitution type × modality ===", flush=True)

    sub_modality = var_df.groupby('substitution_type')[['prot_pct', 'dna_pct', 'cross_pct']].mean()
    sub_modality['n'] = var_df.groupby('substitution_type').size()
    sub_modality = sub_modality[sub_modality['n'] > 500].sort_values('prot_pct', ascending=False)

    print(f"{'Substitution':20s} {'PROT%':>7s} {'DNA%':>7s} {'CROSS%':>7s} {'N':>8s}", flush=True)
    for st, row in sub_modality.iterrows():
        print(f"{st:20s} {row['prot_pct']:7.1%} {row['dna_pct']:7.1%} "
              f"{row['cross_pct']:7.1%} {row['n']:8.0f}", flush=True)

    # ===================================================================
    # Analysis 6: Can modality decomposition improve prediction?
    # Test: for variants predicted as VUS (middle fitness), does modality
    # profile help distinguish borderline pathogenic from borderline benign?
    # ===================================================================
    print("\n=== Modality signal for borderline variants ===", flush=True)

    borderline = var_df[(var_df['fitness_pctile'] >= 0.3) & (var_df['fitness_pctile'] <= 0.7)]
    if len(borderline) > 100:
        # Split borderline into lower half (more damaging) and upper half (more benign)
        mid = borderline['fitness_pctile'].median()
        lower = borderline[borderline['fitness_pctile'] < mid]
        upper = borderline[borderline['fitness_pctile'] >= mid]

        print(f"  Borderline lower (more damaging, n={len(lower)}):", flush=True)
        print(f"    PROT={lower['prot_pct'].mean():.1%} DNA={lower['dna_pct'].mean():.1%} "
              f"CROSS={lower['cross_pct'].mean():.1%}", flush=True)
        print(f"  Borderline upper (more benign, n={len(upper)}):", flush=True)
        print(f"    PROT={upper['prot_pct'].mean():.1%} DNA={upper['dna_pct'].mean():.1%} "
              f"CROSS={upper['cross_pct'].mean():.1%}", flush=True)

        # Statistical test
        for col in ['prot_pct', 'dna_pct', 'cross_pct']:
            stat, pval = stats.mannwhitneyu(lower[col], upper[col], alternative='two-sided')
            diff = lower[col].mean() - upper[col].mean()
            print(f"    {col}: diff={diff:+.2%}, p={pval:.2e}", flush=True)

    # ===================================================================
    # Save gene profiles
    # ===================================================================
    gene_profile.to_csv(os.path.join(args.out_dir, "gene_modality_profile.csv"))
    if len(shift_df) > 0:
        shift_df.to_csv(os.path.join(args.out_dir, "gene_modality_shift.csv"), index=False)

    # ===================================================================
    # Summary report
    # ===================================================================
    report_path = os.path.join(args.out_dir, "interpretable_v2_report.txt")
    with open(report_path, 'w') as f:
        f.write("INTERPRETABLE VARIANT EFFECT PREDICTION v2\n")
        f.write("Modality-first decomposition: PROT / DNA / CROSS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total variants: {len(var_df)}\n")
        f.write(f"Features used: {len(useful_indices)} (with signal in >20 assays)\n")
        f.write(f"Feature modality: {n_prot} PROT, {n_dna} DNA, {n_cross} CROSS\n\n")

        f.write("Modality profile by fitness quantile:\n")
        for lo, hi, label in bins:
            mask = (var_df['fitness_pctile'] >= lo) & (var_df['fitness_pctile'] < hi)
            subset = var_df[mask]
            if len(subset) > 0:
                f.write(f"  {label:22s} PROT={subset['prot_pct'].mean():.1%} "
                        f"DNA={subset['dna_pct'].mean():.1%} "
                        f"CROSS={subset['cross_pct'].mean():.1%} n={len(subset)}\n")

        f.write("\nPer-gene dominant modality (bottom 10%):\n")
        for gene, row in gene_profile.iterrows():
            f.write(f"  {gene:20s}: PROT={row['prot_pct']:.1%} DNA={row['dna_pct']:.1%} "
                    f"CROSS={row['cross_pct']:.1%} dominant={row['dominant']}\n")

    print(f"\nReport saved to {report_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
