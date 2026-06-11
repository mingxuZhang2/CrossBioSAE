"""
Self-Explanatory Cross-Modal Variant Effect Predictor.

Architecture:
  variant → ESM-2 edelta → proj(768) ─┐
                                        ├─→ SAE encoder → TopK(k=32) sparse concepts
  variant → Evo2 edelta  → proj(768) ─┘         │
                                           concept activations c_i
                                                 │
                                        score = Σ(c_i × w_i) + b
                                                 │
                                        "Pathogenic because:
                                         concept_A (Cys loss, PROT) = 40%
                                         concept_B (conservation, DNA) = 30%
                                         concept_C (unknown#7, CROSS) = 20%"

The prediction is inherently interpretable:
- Each concept has a modality (PROT/DNA/CROSS from encoder weights)
- Each concept has a biological label (from activation pattern analysis)
- The contribution of each concept = activation × weight
- Unknown high-weight concepts = candidate scientific discoveries

Training: freeze SAE, train only linear concept weights on DMS fitness data.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


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
    if ref_aa == 'C': return 'Cys_loss'
    if alt_aa == 'C': return 'Cys_gain'
    if alt_aa == 'P': return 'Pro_intro'
    if ref_aa == 'P': return 'Pro_loss'
    if ref_aa == 'G': return 'Gly_loss'
    if ref_aa == 'W': return 'Trp_loss'
    rc = CHARGE.get(ref_aa, 0)
    ac = CHARGE.get(alt_aa, 0)
    if rc * ac < 0: return 'charge_reversal'
    if rc == 0 and abs(ac) > 0: return 'charge_intro'
    if abs(rc) > 0 and ac == 0: return 'charge_loss'
    if ref_aa in HYDROPHOBIC and alt_aa not in HYDROPHOBIC: return 'hydro→polar'
    if ref_aa not in HYDROPHOBIC and alt_aa in HYDROPHOBIC: return 'polar→hydro'
    return 'other'


def characterize_concept(feat_idx, all_sae_acts, all_mutants, mod_score_val):
    """Characterize what biological concept a SAE feature encodes."""
    ref_counts = Counter()
    alt_counts = Counter()
    sub_counts = Counter()
    total_active = 0

    for sae_acts, mutants in zip(all_sae_acts, all_mutants):
        if sae_acts.shape[1] <= feat_idx:
            continue
        acts = sae_acts[:, feat_idx]
        for i in range(len(mutants)):
            if acts[i] <= 0:
                continue
            ref_aa, pos, alt_aa = parse_mutant(mutants[i])
            if ref_aa is None or ref_aa not in AA_TO_IDX or alt_aa not in AA_TO_IDX:
                continue
            ref_counts[ref_aa] += 1
            alt_counts[alt_aa] += 1
            sub_counts[classify_substitution(ref_aa, alt_aa)] += 1
            total_active += 1

    if total_active < 50:
        return {'label': 'rare', 'detail': f'n={total_active}', 'known': False}

    # Dominant substitution type
    top_sub, top_sub_n = sub_counts.most_common(1)[0]
    top_sub_frac = top_sub_n / total_active

    # Dominant ref/alt AA
    top_ref, top_ref_n = ref_counts.most_common(1)[0]
    top_ref_frac = top_ref_n / total_active
    top_alt, top_alt_n = alt_counts.most_common(1)[0]
    top_alt_frac = top_alt_n / total_active

    # Modality
    modality = 'PROT' if mod_score_val > 0.6 else ('DNA' if mod_score_val < 0.4 else 'CROSS')

    # Classification
    known = True
    if top_ref == 'C' and top_ref_frac > 0.15:
        label = f'Cys_loss_detector (ref=C {top_ref_frac:.0%})'
    elif top_alt == 'P' and top_alt_frac > 0.15:
        label = f'Pro_intro_detector (alt=P {top_alt_frac:.0%})'
    elif top_ref == 'G' and top_ref_frac > 0.15:
        label = f'Gly_loss_detector (ref=G {top_ref_frac:.0%})'
    elif top_ref == 'W' and top_ref_frac > 0.15:
        label = f'Trp_loss_detector (ref=W {top_ref_frac:.0%})'
    elif top_ref == 'P' and top_ref_frac > 0.15:
        label = f'Pro_loss_detector (ref=P {top_ref_frac:.0%})'
    elif top_sub_frac > 0.25:
        label = f'{top_sub}_detector ({top_sub_frac:.0%})'
    elif top_ref_frac > 0.20:
        label = f'ref_{top_ref}_detector ({top_ref_frac:.0%})'
    elif top_alt_frac > 0.20:
        label = f'alt_{top_alt}_detector ({top_alt_frac:.0%})'
    else:
        # No clear single concept — potentially novel
        label = f'mixed (top_sub={top_sub} {top_sub_frac:.0%}, ref={top_ref} {top_ref_frac:.0%})'
        known = False

    return {
        'label': label,
        'modality': modality,
        'top_sub': top_sub,
        'top_sub_frac': top_sub_frac,
        'top_ref': top_ref,
        'top_ref_frac': top_ref_frac,
        'top_alt': top_alt,
        'top_alt_frac': top_alt_frac,
        'n_active': total_active,
        'known': known,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_selfexplain")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load SAE encoder weights for modality attribution ──
    import torch
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    mod_score = prot_norm / (prot_norm + dna_norm + 1e-8)

    print(f"Features: {(mod_score>0.6).sum()} PROT, {(mod_score<0.4).sum()} DNA, "
          f"{((mod_score>=0.4)&(mod_score<=0.6)).sum()} CROSS", flush=True)

    # ── Load all DMS SAE activations ──
    print("Loading SAE activations ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    assay_data = []
    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        sae_acts = d["sae_acts"]
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue
        assay_data.append({
            'name': assay_name,
            'gene': assay_name.split('_')[0],
            'sae_acts': sae_acts,
            'fitness': fitness,
            'mutants': mutants,
        })

    print(f"Loaded {len(assay_data)} assays", flush=True)

    # ===================================================================
    # PART 1: Per-assay self-interpretable prediction (DMS benchmark)
    # For each assay: score = SAE_acts @ w + b (ridge regression)
    # Compare with raw embedding baseline
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 1: Per-assay self-interpretable prediction", flush=True)
    print("=" * 60, flush=True)

    per_assay_results = []
    for ad in assay_data:
        X = ad['sae_acts']
        y = ad['fitness']
        if len(y) < 50 or X.shape[0] != len(y):
            continue

        nf = min(X.shape[1], args.n_features)
        X = X[:, :nf]

        # 5-fold CV ridge regression
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if np.std(y_pred) > 1e-8 and np.std(y_test) > 1e-8:
                rho = stats.spearmanr(y_pred, y_test).statistic
                if not np.isnan(rho):
                    rhos.append(rho)

        if rhos:
            mean_rho = np.mean(rhos)
        else:
            mean_rho = 0.0

        per_assay_results.append({
            'assay': ad['name'],
            'gene': ad['gene'],
            'n_variants': len(y),
            'spearman_sae_linear': mean_rho,
        })

    results_df = pd.DataFrame(per_assay_results)
    results_df.to_csv(os.path.join(args.out_dir, "per_assay_results.csv"), index=False)

    mean_rho = results_df['spearman_sae_linear'].mean()
    median_rho = results_df['spearman_sae_linear'].median()
    print(f"\nPer-assay Spearman (SAE linear): mean={mean_rho:.4f}, median={median_rho:.4f}",
          flush=True)
    print(f"Assays evaluated: {len(results_df)}", flush=True)

    # ===================================================================
    # PART 2: Global self-interpretable model
    # Pool all DMS data, train single linear model
    # The weights ARE the concept importance scores
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 2: Global concept importance model", flush=True)
    print("=" * 60, flush=True)

    # Pool all data with per-assay z-score normalization of fitness
    all_X = []
    all_y = []
    all_meta = []  # (assay, mutant, gene)

    for ad in assay_data:
        X = ad['sae_acts'][:, :min(ad['sae_acts'].shape[1], args.n_features)]
        y = ad['fitness']
        if len(y) < 50:
            continue
        # Z-score fitness within assay
        y_z = (y - np.mean(y)) / (np.std(y) + 1e-8)
        all_X.append(X)
        all_y.append(y_z)
        for i in range(len(y)):
            all_meta.append((ad['name'], str(ad['mutants'][i]), ad['gene']))

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    print(f"Pooled data: {all_X.shape[0]} variants, {all_X.shape[1]} features", flush=True)

    # ── Filter rare features: only keep features active in >= 500 variants ──
    feat_active_count = (all_X > 0).sum(axis=0)
    MIN_ACTIVE = 500
    active_mask = feat_active_count >= MIN_ACTIVE
    n_active_feats = active_mask.sum()
    active_indices = np.where(active_mask)[0]
    print(f"Features active in >= {MIN_ACTIVE} variants: {n_active_feats} / {all_X.shape[1]}",
          flush=True)
    print(f"  PROT: {(active_mask & (mod_score[:all_X.shape[1]] > 0.6)).sum()}, "
          f"DNA: {(active_mask & (mod_score[:all_X.shape[1]] < 0.4)).sum()}, "
          f"CROSS: {(active_mask & (mod_score[:all_X.shape[1]] >= 0.4) & (mod_score[:all_X.shape[1]] <= 0.6)).sum()}",
          flush=True)

    X_filtered = all_X[:, active_mask]
    print(f"Filtered data: {X_filtered.shape[0]} variants × {X_filtered.shape[1]} features",
          flush=True)

    # Train global ridge model on filtered features
    print("Training global ridge model ...", flush=True)
    global_model = Ridge(alpha=10.0)
    global_model.fit(X_filtered, all_y)

    # Map weights back to full feature space
    concept_weights = np.zeros(all_X.shape[1])
    concept_weights[active_mask] = global_model.coef_
    concept_bias = global_model.intercept_

    # Global Spearman
    y_pred_global = global_model.predict(X_filtered)
    global_rho = stats.spearmanr(y_pred_global, all_y).statistic
    print(f"Global Spearman (pooled, filtered): {global_rho:.4f}", flush=True)

    # ===================================================================
    # PART 3: Concept catalog — characterize each important concept
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 3: Concept catalog", flush=True)
    print("=" * 60, flush=True)

    # Sort by |weight| — most important concepts first (only active features)
    abs_weights = np.abs(concept_weights)
    # Only rank features that passed the activation filter
    ranked_active = sorted(active_indices, key=lambda i: -abs_weights[i])

    # Characterize top 200 active concepts
    n_characterize = min(200, len(ranked_active))
    top_concepts = ranked_active[:n_characterize]
    print(f"Characterizing top {n_characterize} concepts (from {len(ranked_active)} active) ...",
          flush=True)

    all_sae_acts_list = [ad['sae_acts'] for ad in assay_data]
    all_mutants_list = [ad['mutants'] for ad in assay_data]

    concept_catalog = []
    for rank, fi in enumerate(top_concepts[:n_characterize]):
        w = concept_weights[fi]
        direction = 'damaging' if w < 0 else 'protective'

        char = characterize_concept(fi, all_sae_acts_list, all_mutants_list, mod_score[fi])
        modality = 'PROT' if mod_score[fi] > 0.6 else ('DNA' if mod_score[fi] < 0.4 else 'CROSS')

        concept_catalog.append({
            'rank': rank + 1,
            'feature': f'F{fi:05d}',
            'feature_idx': fi,
            'weight': w,
            'abs_weight': abs_weights[fi],
            'direction': direction,
            'modality': modality,
            'mod_score': mod_score[fi],
            'label': char['label'],
            'known_concept': char['known'],
            'n_active': char.get('n_active', 0),
            'top_sub': char.get('top_sub', ''),
            'top_sub_frac': char.get('top_sub_frac', 0),
            'top_ref': char.get('top_ref', ''),
            'top_ref_frac': char.get('top_ref_frac', 0),
        })

    catalog_df = pd.DataFrame(concept_catalog)
    catalog_df.to_csv(os.path.join(args.out_dir, "concept_catalog.csv"), index=False)

    # Print top concepts
    print(f"\nTop 30 most important concepts (by |weight|):", flush=True)
    print(f"{'Rank':>4s} {'Feature':>8s} {'Weight':>8s} {'Dir':>10s} {'Modality':>8s} "
          f"{'Known':>5s} {'Label'}", flush=True)
    for _, row in catalog_df.head(30).iterrows():
        print(f"{row['rank']:4d} {row['feature']:>8s} {row['weight']:+8.5f} "
              f"{row['direction']:>10s} {row['modality']:>8s} "
              f"{'✓' if row['known_concept'] else '✗':>5s} {row['label']}", flush=True)

    # Summary statistics
    n_known = catalog_df['known_concept'].sum()
    n_unknown = len(catalog_df) - n_known
    print(f"\nOf top {n_characterize} concepts: {n_known} known, {n_unknown} unknown", flush=True)

    # Modality distribution of important concepts
    mod_dist = catalog_df['modality'].value_counts()
    print(f"\nModality distribution of top concepts:", flush=True)
    for mod, cnt in mod_dist.items():
        print(f"  {mod}: {cnt} ({cnt/len(catalog_df):.0%})", flush=True)

    # Direction distribution
    dir_dist = catalog_df['direction'].value_counts()
    print(f"\nDirection: {dir_dist.to_dict()}", flush=True)

    # Per-modality top concepts
    for mod_name in ['PROT', 'DNA', 'CROSS']:
        mod_df = catalog_df[catalog_df['modality'] == mod_name].head(10)
        if len(mod_df) == 0:
            continue
        print(f"\n  Top 10 {mod_name} concepts:", flush=True)
        for _, row in mod_df.iterrows():
            print(f"    Rank {row['rank']:3d} {row['feature']} w={row['weight']:+.5f} "
                  f"{row['direction']:10s} n={row['n_active']:5d} "
                  f"{'✓' if row['known_concept'] else '✗'} {row['label']}", flush=True)

    # ===================================================================
    # PART 4: Unknown concepts — candidate scientific discoveries
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 4: Unknown concepts (candidate discoveries)", flush=True)
    print("=" * 60, flush=True)

    unknown_df = catalog_df[~catalog_df['known_concept']].copy()
    print(f"\n{len(unknown_df)} unknown concepts in top {n_characterize}:", flush=True)

    for _, row in unknown_df.head(20).iterrows():
        print(f"\n  Rank {row['rank']:3d}: {row['feature']} "
              f"(weight={row['weight']:+.5f}, {row['direction']}, {row['modality']})", flush=True)
        print(f"    Label: {row['label']}", flush=True)
        print(f"    Active in {row['n_active']} variants", flush=True)
        print(f"    Top sub: {row['top_sub']} ({row['top_sub_frac']:.0%}), "
              f"top ref: {row['top_ref']} ({row['top_ref_frac']:.0%})", flush=True)

    # ===================================================================
    # PART 5: Example decompositions for well-known genes
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 5: Example self-explanatory predictions", flush=True)
    print("=" * 60, flush=True)

    example_genes = ['BRCA1', 'P53', 'PTEN', 'MSH2', 'SRC', 'CASP3', 'SPIKE', 'GFP']

    for gene in example_genes:
        gene_assays = [ad for ad in assay_data if ad['gene'] == gene]
        if not gene_assays:
            continue

        ad = gene_assays[0]  # Use first assay for this gene
        X = ad['sae_acts'][:, :min(ad['sae_acts'].shape[1], args.n_features)]
        y = ad['fitness']

        # Pick most damaging and most benign variants
        dam_idx = np.argsort(y)[:3]
        ben_idx = np.argsort(y)[-3:]

        print(f"\n  === {gene} ({ad['name']}, {len(y)} variants) ===", flush=True)

        for label, indices in [("Most damaging", dam_idx), ("Most benign", ben_idx)]:
            print(f"  {label}:", flush=True)
            for idx in indices:
                mutant = str(ad['mutants'][idx])
                fitness = y[idx]
                acts = X[idx]

                # Compute per-concept contributions
                contributions = acts * concept_weights[:len(acts)]
                score = contributions.sum() + concept_bias

                # Top contributing concepts (by |contribution|)
                abs_contrib = np.abs(contributions)
                top_k = np.argsort(-abs_contrib)[:5]

                total_abs = abs_contrib.sum() + 1e-10

                # Modality-level aggregation
                prot_contrib = sum(contributions[fi] for fi in range(len(acts))
                                   if mod_score[fi] > 0.6 and abs(contributions[fi]) > 0)
                dna_contrib = sum(contributions[fi] for fi in range(len(acts))
                                  if mod_score[fi] < 0.4 and abs(contributions[fi]) > 0)
                cross_contrib = sum(contributions[fi] for fi in range(len(acts))
                                    if 0.4 <= mod_score[fi] <= 0.6 and abs(contributions[fi]) > 0)

                total_modal = abs(prot_contrib) + abs(dna_contrib) + abs(cross_contrib) + 1e-10

                ref_aa, pos, alt_aa = parse_mutant(mutant)
                sub_type = classify_substitution(ref_aa, alt_aa) if ref_aa else ''

                print(f"    {mutant:10s} fitness={fitness:8.3f} → score={score:+.4f}", flush=True)
                print(f"      Modality: PROT={abs(prot_contrib)/total_modal:.0%} "
                      f"DNA={abs(dna_contrib)/total_modal:.0%} "
                      f"CROSS={abs(cross_contrib)/total_modal:.0%} "
                      f"({sub_type})", flush=True)
                print(f"      Top concepts:", flush=True)
                for fi in top_k:
                    if abs(contributions[fi]) < 1e-8:
                        continue
                    # Find this concept in catalog
                    cat_match = catalog_df[catalog_df['feature_idx'] == fi]
                    if len(cat_match) > 0:
                        clabel = cat_match.iloc[0]['label']
                        modality = cat_match.iloc[0]['modality']
                        known = '✓' if cat_match.iloc[0]['known_concept'] else '✗'
                    else:
                        modality = 'PROT' if mod_score[fi] > 0.6 else (
                            'DNA' if mod_score[fi] < 0.4 else 'CROSS')
                        clabel = '(not in top-200)'
                        known = '?'
                    pct = abs(contributions[fi]) / total_abs
                    sign = '+' if contributions[fi] > 0 else '-'
                    print(f"        F{fi:05d} [{modality:5s}] {sign}{pct:.0%} "
                          f"known={known} {clabel}", flush=True)

    # ===================================================================
    # PART 6: Modality-stratified concept analysis
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART 6: Concept importance by modality", flush=True)
    print("=" * 60, flush=True)

    # Total weight mass per modality
    prot_weight_mass = np.sum(abs_weights[mod_score > 0.6])
    dna_weight_mass = np.sum(abs_weights[mod_score < 0.4])
    cross_weight_mass = np.sum(abs_weights[(mod_score >= 0.4) & (mod_score <= 0.6)])
    total_weight = prot_weight_mass + dna_weight_mass + cross_weight_mass

    print(f"Total weight mass by modality:", flush=True)
    print(f"  PROT:  {prot_weight_mass:.4f} ({prot_weight_mass/total_weight:.1%})", flush=True)
    print(f"  DNA:   {dna_weight_mass:.4f} ({dna_weight_mass/total_weight:.1%})", flush=True)
    print(f"  CROSS: {cross_weight_mass:.4f} ({cross_weight_mass/total_weight:.1%})", flush=True)

    # Damaging vs protective by modality
    for modality, mask in [('PROT', mod_score > 0.6),
                           ('DNA', mod_score < 0.4),
                           ('CROSS', (mod_score >= 0.4) & (mod_score <= 0.6))]:
        dam_w = concept_weights[mask & (concept_weights < 0)]
        prot_w = concept_weights[mask & (concept_weights > 0)]
        print(f"\n  {modality}: {mask.sum()} features", flush=True)
        print(f"    Damaging concepts: {len(dam_w)}, total weight: {np.sum(np.abs(dam_w)):.4f}",
              flush=True)
        print(f"    Protective concepts: {len(prot_w)}, total weight: {np.sum(np.abs(prot_w)):.4f}",
              flush=True)

    # ===================================================================
    # Save everything
    # ===================================================================
    np.savez(os.path.join(args.out_dir, "concept_weights.npz"),
             concept_weights=concept_weights,
             concept_bias=np.array([concept_bias]),
             mod_score=mod_score)

    # Summary report
    report_path = os.path.join(args.out_dir, "selfexplain_report.txt")
    with open(report_path, 'w') as f:
        f.write("SELF-EXPLANATORY CROSS-MODAL VARIANT EFFECT PREDICTOR\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Architecture: variant → SAE(12288 concepts, TopK=32) → linear → score\n")
        f.write(f"Prediction: score = Σ(concept_activation_i × concept_weight_i) + bias\n\n")
        f.write(f"Data: {len(assay_data)} DMS assays, {all_X.shape[0]} variants\n")
        f.write(f"Features: {(mod_score>0.6).sum()} PROT, {(mod_score<0.4).sum()} DNA, "
                f"{((mod_score>=0.4)&(mod_score<=0.6)).sum()} CROSS\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Per-assay Spearman (5-fold CV): mean={mean_rho:.4f}, median={median_rho:.4f}\n")
        f.write(f"  Global Spearman (pooled): {global_rho:.4f}\n\n")
        f.write(f"Concept catalog:\n")
        f.write(f"  Top {n_characterize} concepts: {n_known} known, {n_unknown} unknown\n")
        f.write(f"  Modality: {mod_dist.to_dict()}\n\n")
        f.write(f"Weight mass by modality:\n")
        f.write(f"  PROT: {prot_weight_mass/total_weight:.1%}\n")
        f.write(f"  DNA: {dna_weight_mass/total_weight:.1%}\n")
        f.write(f"  CROSS: {cross_weight_mass/total_weight:.1%}\n")

    print(f"\nReport saved to {report_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
