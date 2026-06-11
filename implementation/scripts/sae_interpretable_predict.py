"""
Interpretable variant effect prediction via cross-modal SAE decomposition.

For each variant, produces:
1. Pathogenicity score (from v6 model)
2. Mechanism decomposition: % contribution from PROT / DNA / CROSS features
3. Top activated SAE features with human-readable mechanism labels
4. Specific biochemical mechanism annotation (e.g., "Cys loss", "Pro intro", "charge reversal")

This is the key paper contribution: not just "pathogenic or not", but WHY.
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
CHARGE = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}

# ── Mechanism taxonomy ──
# Each SAE feature is assigned to a mechanism category based on its activation pattern
MECHANISM_CATEGORIES = {
    'structural_cys': 'Cysteine loss (disulfide/metal/redox)',
    'structural_pro_intro': 'Proline introduction (helix/sheet disruption)',
    'structural_pro_loss': 'Proline loss (turn/kink disruption)',
    'structural_gly_loss': 'Glycine loss (backbone flexibility)',
    'structural_trp_loss': 'Tryptophan loss (aromatic core)',
    'structural_buried_hydro': 'Buried hydrophobic disruption',
    'biochem_charge_reversal': 'Charge reversal (+↔-)',
    'biochem_charge_intro': 'Charge introduction (neutral→charged)',
    'biochem_charge_loss': 'Charge loss (charged→neutral)',
    'biochem_hydro_to_polar': 'Hydrophobic→polar transition',
    'biochem_polar_to_hydro': 'Polar→hydrophobic transition',
    'biochem_size_change': 'Size/volume change',
    'conservation_dna': 'DNA-level evolutionary conservation',
    'conservation_cross': 'Cross-modal conservation signal',
    'general_damage': 'General perturbation (non-specific)',
}


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def classify_feature_mechanism(feat_idx, sae_acts_list, mutants_list, fitness_list,
                                mod_score, feat_mean_rho):
    """
    Classify a single SAE feature into a mechanism category.
    Uses activation patterns across all DMS variants.
    """
    # Collect activations with mutation context
    ref_aa_acts = defaultdict(list)
    alt_aa_acts = defaultdict(list)
    sub_type_acts = defaultdict(list)

    for sae_acts, mutants, fitness in zip(sae_acts_list, mutants_list, fitness_list):
        if sae_acts.shape[1] <= feat_idx:
            continue
        acts = sae_acts[:, feat_idx]
        for i in range(len(mutants)):
            if acts[i] <= 0:
                continue
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa is None or ref_aa not in AA_TO_IDX or alt_aa not in AA_TO_IDX:
                continue
            ref_aa_acts[ref_aa].append(acts[i])
            alt_aa_acts[alt_aa].append(acts[i])

            # Classify substitution
            if ref_aa == 'C':
                sub_type_acts['cys_loss'].append(acts[i])
            elif alt_aa == 'P':
                sub_type_acts['pro_intro'].append(acts[i])
            elif ref_aa == 'P':
                sub_type_acts['pro_loss'].append(acts[i])
            elif ref_aa == 'G':
                sub_type_acts['gly_loss'].append(acts[i])
            elif ref_aa == 'W':
                sub_type_acts['trp_loss'].append(acts[i])
            elif ref_aa in HYDROPHOBIC and alt_aa not in HYDROPHOBIC:
                sub_type_acts['hydro_to_polar'].append(acts[i])
            elif ref_aa not in HYDROPHOBIC and alt_aa in HYDROPHOBIC:
                sub_type_acts['polar_to_hydro'].append(acts[i])

            ref_c = CHARGE.get(ref_aa, 0)
            alt_c = CHARGE.get(alt_aa, 0)
            if ref_c * alt_c < 0:
                sub_type_acts['charge_reversal'].append(acts[i])
            elif ref_c == 0 and abs(alt_c) > 0:
                sub_type_acts['charge_intro'].append(acts[i])
            elif abs(ref_c) > 0 and alt_c == 0:
                sub_type_acts['charge_loss'].append(acts[i])

    total_active = sum(len(v) for v in ref_aa_acts.values())
    if total_active < 50:
        return 'general_damage', 0.0, 'Too few activations'

    # Determine mechanism by dominant pattern
    # Check ref AA concentration
    ref_total = sum(len(v) for v in ref_aa_acts.values())
    ref_fracs = {aa: len(acts) / ref_total for aa, acts in ref_aa_acts.items()}
    top_ref = max(ref_fracs, key=ref_fracs.get)
    top_ref_frac = ref_fracs[top_ref]

    alt_total = sum(len(v) for v in alt_aa_acts.values())
    alt_fracs = {aa: len(acts) / alt_total for aa, acts in alt_aa_acts.items()}
    top_alt = max(alt_fracs, key=alt_fracs.get)
    top_alt_frac = alt_fracs[top_alt]

    # Sub-type fractions
    sub_fracs = {st: len(acts) / total_active for st, acts in sub_type_acts.items()}

    # Decision tree for mechanism classification
    modality = 'PROT' if mod_score > 0.6 else ('DNA' if mod_score < 0.4 else 'CROSS')

    # Specific patterns first (high specificity)
    if top_ref == 'C' and top_ref_frac > 0.15:
        return 'structural_cys', top_ref_frac, f'ref=C {top_ref_frac:.0%}'
    if sub_fracs.get('pro_intro', 0) > 0.20:
        return 'structural_pro_intro', sub_fracs['pro_intro'], f'→P {sub_fracs["pro_intro"]:.0%}'
    if top_ref == 'P' and top_ref_frac > 0.20:
        return 'structural_pro_loss', top_ref_frac, f'ref=P {top_ref_frac:.0%}'
    if top_ref == 'G' and top_ref_frac > 0.15:
        return 'structural_gly_loss', top_ref_frac, f'ref=G {top_ref_frac:.0%}'
    if top_ref == 'W' and top_ref_frac > 0.15:
        return 'structural_trp_loss', top_ref_frac, f'ref=W {top_ref_frac:.0%}'

    if sub_fracs.get('charge_reversal', 0) > 0.15:
        return 'biochem_charge_reversal', sub_fracs['charge_reversal'], 'charge flip'
    if sub_fracs.get('charge_intro', 0) > 0.15:
        return 'biochem_charge_intro', sub_fracs['charge_intro'], 'charge gain'
    if sub_fracs.get('charge_loss', 0) > 0.15:
        return 'biochem_charge_loss', sub_fracs['charge_loss'], 'charge loss'
    if sub_fracs.get('hydro_to_polar', 0) > 0.25:
        return 'biochem_hydro_to_polar', sub_fracs['hydro_to_polar'], 'hydro→polar'
    if sub_fracs.get('polar_to_hydro', 0) > 0.20:
        return 'biochem_polar_to_hydro', sub_fracs['polar_to_hydro'], 'polar→hydro'

    # Modality-based fallback
    if modality == 'DNA':
        return 'conservation_dna', 1.0, 'DNA-driven'
    if modality == 'CROSS':
        return 'conservation_cross', 1.0, 'cross-modal'

    return 'general_damage', 0.0, 'non-specific'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_interpretable")
    ap.add_argument("--n_features", type=int, default=12288)
    ap.add_argument("--top_k_features", type=int, default=500,
                    help="Number of features to classify mechanisms for")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load encoder weights ──
    import torch
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()  # (n_features, 1536)
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    mod_score = prot_norm / (prot_norm + dna_norm + 1e-8)

    # Also get decoder weights for contribution analysis
    dec_w = sd["decoder.weight"].numpy()  # (1536, n_features)

    print(f"Features: {(mod_score>0.6).sum()} PROT, {(mod_score<0.4).sum()} DNA, "
          f"{((mod_score>=0.4)&(mod_score<=0.6)).sum()} CROSS", flush=True)

    # ── Load all SAE activations ──
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

    # ── Compute per-feature fitness correlation ──
    print("Computing per-feature fitness correlation ...", flush=True)
    feat_rho = np.zeros(args.n_features)
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
                feat_rho[fi] += rho
                feat_rho_count[fi] += 1

    feat_mean_rho = np.divide(feat_rho, feat_rho_count,
                               where=feat_rho_count > 0,
                               out=np.zeros_like(feat_rho))

    # ── Select top features by |rho| and classify mechanisms ──
    feat_abs_rho = np.abs(feat_mean_rho)
    top_feat_indices = np.argsort(-feat_abs_rho)[:args.top_k_features]

    print(f"Classifying mechanisms for top {args.top_k_features} features ...", flush=True)
    feature_mechanisms = {}
    for fi in top_feat_indices:
        if feat_rho_count[fi] < 20:
            continue
        mech, confidence, detail = classify_feature_mechanism(
            fi, all_sae_acts, all_mutants, all_fitness,
            mod_score[fi], feat_mean_rho[fi])
        modality = 'PROT' if mod_score[fi] > 0.6 else ('DNA' if mod_score[fi] < 0.4 else 'CROSS')
        feature_mechanisms[fi] = {
            'mechanism': mech,
            'mechanism_label': MECHANISM_CATEGORIES[mech],
            'confidence': confidence,
            'detail': detail,
            'modality': modality,
            'mod_score': mod_score[fi],
            'fitness_rho': feat_mean_rho[fi],
        }

    # Save feature mechanism catalog
    mech_rows = []
    for fi, info in sorted(feature_mechanisms.items()):
        mech_rows.append({
            'feature': f'F{fi:05d}',
            'feature_idx': fi,
            **info
        })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(os.path.join(args.out_dir, "feature_mechanism_catalog.csv"), index=False)

    # Summary of mechanism distribution
    print("\nFeature mechanism distribution:", flush=True)
    mech_counts = mech_df['mechanism'].value_counts()
    for mech, count in mech_counts.items():
        label = MECHANISM_CATEGORIES[mech]
        print(f"  {mech:30s} ({label:45s}): {count}", flush=True)

    # ===================================================================
    # Per-variant interpretable prediction
    # For each variant: decompose SAE activation into mechanism contributions
    # ===================================================================
    print("\n=== Per-variant mechanism decomposition ===", flush=True)

    classified_features = set(feature_mechanisms.keys())

    # Group features by mechanism category
    mech_to_features = defaultdict(list)
    for fi, info in feature_mechanisms.items():
        mech_to_features[info['mechanism']].append(fi)

    # Group mechanisms into high-level categories
    HIGH_LEVEL = {
        'structural': ['structural_cys', 'structural_pro_intro', 'structural_pro_loss',
                        'structural_gly_loss', 'structural_trp_loss', 'structural_buried_hydro'],
        'biochemical': ['biochem_charge_reversal', 'biochem_charge_intro', 'biochem_charge_loss',
                        'biochem_hydro_to_polar', 'biochem_polar_to_hydro', 'biochem_size_change'],
        'conservation': ['conservation_dna', 'conservation_cross'],
        'general': ['general_damage'],
    }

    high_to_features = defaultdict(list)
    for high, mechs in HIGH_LEVEL.items():
        for m in mechs:
            high_to_features[high].extend(mech_to_features.get(m, []))

    variant_records = []
    for assay_idx, (assay_name, sae_acts, mutants, fitness) in enumerate(
            zip(all_assay_names, all_sae_acts, all_mutants, all_fitness)):
        if len(fitness) < 50:
            continue
        nf = min(sae_acts.shape[1], args.n_features)
        gene_name = assay_name.split('_')[0]

        for i in range(len(mutants)):
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa is None:
                continue

            acts = sae_acts[i, :nf]

            # High-level mechanism contribution
            # Weight = |activation * fitness_rho| (how much this feature contributes to prediction)
            high_scores = {}
            for high, feat_list in high_to_features.items():
                score = 0
                for fi in feat_list:
                    if fi < nf and acts[fi] > 0:
                        score += abs(acts[fi] * feat_mean_rho[fi])
                high_scores[high] = score

            total_score = sum(high_scores.values()) + 1e-10
            high_fracs = {k: v / total_score for k, v in high_scores.items()}

            # Top 3 specific mechanisms
            mech_scores = {}
            for mech, feat_list in mech_to_features.items():
                score = 0
                for fi in feat_list:
                    if fi < nf and acts[fi] > 0:
                        score += abs(acts[fi] * feat_mean_rho[fi])
                if score > 0:
                    mech_scores[mech] = score

            top_mechs = sorted(mech_scores.items(), key=lambda x: -x[1])[:3]

            # Top 3 activated features
            active_classified = [(fi, acts[fi]) for fi in classified_features
                                  if fi < nf and acts[fi] > 0]
            active_classified.sort(key=lambda x: -x[1])
            top_features = [(fi, acts[fi], feature_mechanisms[fi]['mechanism_label'])
                           for fi, _ in active_classified[:3]]

            record = {
                'assay': assay_name,
                'gene': gene_name,
                'mutant': str(mutants[i]),
                'ref_aa': ref_aa,
                'pos': pos,
                'alt_aa': alt_aa,
                'fitness': fitness[i],
                'structural_frac': high_fracs.get('structural', 0),
                'biochemical_frac': high_fracs.get('biochemical', 0),
                'conservation_frac': high_fracs.get('conservation', 0),
                'general_frac': high_fracs.get('general', 0),
                'top_mechanism_1': top_mechs[0][0] if len(top_mechs) > 0 else '',
                'top_mechanism_1_label': MECHANISM_CATEGORIES.get(
                    top_mechs[0][0], '') if len(top_mechs) > 0 else '',
                'top_mechanism_1_frac': top_mechs[0][1] / total_score if len(top_mechs) > 0 else 0,
                'top_mechanism_2': top_mechs[1][0] if len(top_mechs) > 1 else '',
                'top_mechanism_2_frac': top_mechs[1][1] / total_score if len(top_mechs) > 1 else 0,
                'top_feature_1': f'F{top_features[0][0]:05d}' if len(top_features) > 0 else '',
                'top_feature_1_label': top_features[0][2] if len(top_features) > 0 else '',
                'top_feature_1_act': top_features[0][1] if len(top_features) > 0 else 0,
                'n_active_features': sum(1 for fi in classified_features
                                         if fi < nf and acts[fi] > 0),
            }
            variant_records.append(record)

    var_df = pd.DataFrame(variant_records)
    print(f"Total variants with mechanism decomposition: {len(var_df)}", flush=True)

    # Save full variant-level data
    var_df.to_csv(os.path.join(args.out_dir, "variant_mechanisms.csv.gz"),
                  index=False, compression='gzip')

    # ===================================================================
    # Analysis: mechanism profiles by fitness quantile
    # ===================================================================
    print("\n=== Mechanism profiles by fitness quantile ===", flush=True)

    # Per-assay fitness percentile
    var_df['fitness_pctile'] = 0.0
    for assay in var_df['assay'].unique():
        mask = var_df['assay'] == assay
        var_df.loc[mask, 'fitness_pctile'] = var_df.loc[mask, 'fitness'].rank(pct=True)

    bins = [(0, 0.1, 'most_damaging'),
            (0.1, 0.3, 'damaging'),
            (0.3, 0.7, 'neutral'),
            (0.7, 1.0, 'benign')]

    print(f"{'Category':20s} {'Structural':>10s} {'Biochemical':>11s} "
          f"{'Conservation':>12s} {'General':>8s} {'N':>8s}", flush=True)

    for lo, hi, label in bins:
        mask = (var_df['fitness_pctile'] >= lo) & (var_df['fitness_pctile'] < hi)
        subset = var_df[mask]
        if len(subset) == 0:
            continue
        print(f"{label:20s} {subset['structural_frac'].mean():10.1%} "
              f"{subset['biochemical_frac'].mean():11.1%} "
              f"{subset['conservation_frac'].mean():12.1%} "
              f"{subset['general_frac'].mean():8.1%} "
              f"{len(subset):8d}", flush=True)

    # ===================================================================
    # Top mechanism for most damaging variants per gene
    # ===================================================================
    print("\n=== Most damaging variants: dominant mechanism per gene ===", flush=True)

    most_dam = var_df[var_df['fitness_pctile'] < 0.1]
    gene_mechs = most_dam.groupby('gene')[['structural_frac', 'biochemical_frac',
                                            'conservation_frac']].mean()
    gene_mechs['dominant'] = gene_mechs.idxmax(axis=1).str.replace('_frac', '')
    gene_mechs['n_dam'] = most_dam.groupby('gene').size()

    # Show genes with clear dominant mechanism
    for mech_type in ['structural', 'biochemical', 'conservation']:
        subset = gene_mechs[gene_mechs['dominant'] == mech_type].nlargest(
            10, f'{mech_type}_frac')
        print(f"\n  Top {mech_type}-dominant genes:", flush=True)
        for gene, row in subset.iterrows():
            print(f"    {gene:15s}: struct={row['structural_frac']:.1%} "
                  f"biochem={row['biochemical_frac']:.1%} "
                  f"conserv={row['conservation_frac']:.1%} "
                  f"(n={row['n_dam']:.0f})", flush=True)

    # ===================================================================
    # Example interpretable predictions
    # ===================================================================
    print("\n=== Example interpretable predictions ===", flush=True)

    # Pick well-known genes and show top/bottom variants
    example_genes = ['BRCA1', 'P53', 'PTEN', 'MSH2', 'SRC', 'CASP3', 'GFP']
    for gene in example_genes:
        gene_var = var_df[var_df['gene'] == gene]
        if len(gene_var) < 20:
            continue

        print(f"\n  === {gene} ({len(gene_var)} variants) ===", flush=True)

        # Most damaging
        most_dam_gene = gene_var.nsmallest(5, 'fitness')
        print(f"  Most damaging:", flush=True)
        for _, row in most_dam_gene.iterrows():
            print(f"    {row['mutant']:10s} fitness={row['fitness']:8.3f} | "
                  f"struct={row['structural_frac']:.0%} "
                  f"biochem={row['biochemical_frac']:.0%} "
                  f"conserv={row['conservation_frac']:.0%} | "
                  f"top: {row['top_mechanism_1_label']}", flush=True)

        # Most benign
        most_ben_gene = gene_var.nlargest(5, 'fitness')
        print(f"  Most benign:", flush=True)
        for _, row in most_ben_gene.iterrows():
            print(f"    {row['mutant']:10s} fitness={row['fitness']:8.3f} | "
                  f"struct={row['structural_frac']:.0%} "
                  f"biochem={row['biochemical_frac']:.0%} "
                  f"conserv={row['conservation_frac']:.0%} | "
                  f"top: {row['top_mechanism_1_label']}", flush=True)

    # ===================================================================
    # Mechanism-stratified prediction performance
    # Does knowing the mechanism improve prediction?
    # ===================================================================
    print("\n=== Mechanism-stratified prediction accuracy ===", flush=True)

    # For each dominant mechanism type, compute correlation of
    # structural/biochem/conservation score with fitness
    for mech_type in ['structural', 'biochemical', 'conservation']:
        col = f'{mech_type}_frac'
        # Variants where this mechanism dominates (>50%)
        mask = var_df[col] > 0.5
        subset = var_df[mask]
        if len(subset) < 100:
            continue

        # Correlation of mechanism score with fitness
        rho = stats.spearmanr(subset[col], subset['fitness']).statistic
        print(f"  {mech_type:15s}-dominant (n={len(subset):6d}): "
              f"rho(score, fitness)={rho:.4f}", flush=True)

    # ===================================================================
    # Save summary report
    # ===================================================================
    report_path = os.path.join(args.out_dir, "interpretable_prediction_report.txt")
    with open(report_path, 'w') as f:
        f.write("INTERPRETABLE VARIANT EFFECT PREDICTION\n")
        f.write("Cross-modal SAE mechanism decomposition\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Features classified: {len(feature_mechanisms)}\n")
        f.write(f"Variants decomposed: {len(var_df)}\n\n")

        f.write("Feature mechanism distribution:\n")
        for mech, count in mech_counts.items():
            label = MECHANISM_CATEGORIES[mech]
            f.write(f"  {mech:30s} ({label:45s}): {count}\n")

        f.write("\nMechanism profiles by fitness quantile:\n")
        for lo, hi, label in bins:
            mask = (var_df['fitness_pctile'] >= lo) & (var_df['fitness_pctile'] < hi)
            subset = var_df[mask]
            if len(subset) > 0:
                f.write(f"  {label:20s} struct={subset['structural_frac'].mean():.1%} "
                        f"biochem={subset['biochemical_frac'].mean():.1%} "
                        f"conserv={subset['conservation_frac'].mean():.1%} "
                        f"general={subset['general_frac'].mean():.1%} n={len(subset)}\n")

        f.write("\nDominant mechanism per gene (most damaging variants):\n")
        for gene, row in gene_mechs.iterrows():
            f.write(f"  {gene:15s}: {row['dominant']:15s} "
                    f"(struct={row['structural_frac']:.1%} "
                    f"biochem={row['biochemical_frac']:.1%} "
                    f"conserv={row['conservation_frac']:.1%})\n")

    print(f"\nReport saved to {report_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
