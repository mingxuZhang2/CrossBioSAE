"""
Hybrid Self-Explanatory Variant Effect Predictor.

Combines:
  1. SAE-discovered concepts (data-driven, cross-modal)
  2. Human-defined concepts (BLOSUM62, Grantham, AA properties)

score = Σ(sae_concept_i × w_i) + Σ(human_concept_j × w_j) + b

Fully interpretable: every term has a biological meaning.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDROPHOBIC = set('AVILMFWP')
CHARGE_MAP = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}

# ── BLOSUM62 substitution scores ──
BLOSUM62 = {}
_blosum_str = """
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_lines = [l.strip() for l in _blosum_str.strip().split('\n') if l.strip()]
for i, line in enumerate(_lines):
    parts = line.split()
    aa1 = parts[0]
    scores = [int(x) for x in parts[1:]]
    for j, s in enumerate(scores):
        BLOSUM62[(aa1, AA_ORDER[j])] = s

# ── Grantham distance ──
GRANTHAM = {}
_grantham_data = {
    'S': (1.42, 9.2, 1.67), 'R': (0.65, 10.5, 124.0), 'L': (0.00, 4.9, 111.0),
    'P': (0.39, 8.0, 32.5), 'T': (0.71, 8.6, 32.5), 'A': (0.00, 8.1, 31.0),
    'V': (0.00, 5.9, 84.0), 'G': (0.74, 9.0, 3.0), 'I': (0.00, 5.2, 111.0),
    'F': (0.00, 5.2, 132.0), 'Y': (0.20, 6.2, 136.0), 'C': (2.75, 5.5, 55.0),
    'H': (0.58, 10.4, 96.0), 'Q': (0.89, 10.5, 85.0), 'N': (1.33, 11.6, 56.0),
    'K': (0.33, 11.3, 119.0), 'D': (1.38, 13.0, 54.0), 'E': (0.92, 12.3, 83.0),
    'M': (0.00, 5.7, 105.0), 'W': (0.13, 5.4, 170.0),
}
for aa1 in AA_ORDER:
    for aa2 in AA_ORDER:
        if aa1 == aa2:
            GRANTHAM[(aa1, aa2)] = 0
        elif aa1 in _grantham_data and aa2 in _grantham_data:
            c1, p1, v1 = _grantham_data[aa1]
            c2, p2, v2 = _grantham_data[aa2]
            d = ((1.833 * (c1-c2))**2 + (0.1018 * (p1-p2))**2 + (0.000399 * (v1-v2))**2) ** 0.5
            GRANTHAM[(aa1, aa2)] = d
        else:
            GRANTHAM[(aa1, aa2)] = 100

# ── AA property features ──
AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def compute_human_features(ref_aa, alt_aa):
    """Compute human-interpretable substitution features."""
    feats = {}

    # BLOSUM62 score
    feats['blosum62'] = BLOSUM62.get((ref_aa, alt_aa), 0)

    # Grantham distance
    feats['grantham'] = GRANTHAM.get((ref_aa, alt_aa), 100)

    # Charge change
    ref_c = CHARGE_MAP.get(ref_aa, 0)
    alt_c = CHARGE_MAP.get(alt_aa, 0)
    feats['charge_change'] = alt_c - ref_c

    # Volume change
    feats['volume_change'] = AA_VOLUME.get(alt_aa, 140) - AA_VOLUME.get(ref_aa, 140)

    # Hydrophobicity change
    feats['hydro_change'] = AA_HYDRO.get(alt_aa, 0) - AA_HYDRO.get(ref_aa, 0)

    # Binary features
    feats['is_cys_loss'] = float(ref_aa == 'C')
    feats['is_cys_gain'] = float(alt_aa == 'C')
    feats['is_pro_intro'] = float(alt_aa == 'P')
    feats['is_pro_loss'] = float(ref_aa == 'P')
    feats['is_gly_loss'] = float(ref_aa == 'G')
    feats['is_trp_loss'] = float(ref_aa == 'W')
    feats['is_charge_reversal'] = float(ref_c * alt_c < 0)
    feats['is_hydro_to_polar'] = float(ref_aa in HYDROPHOBIC and alt_aa not in HYDROPHOBIC)
    feats['is_polar_to_hydro'] = float(ref_aa not in HYDROPHOBIC and alt_aa in HYDROPHOBIC)

    # Absolute changes
    feats['abs_charge_change'] = abs(alt_c - ref_c)
    feats['abs_volume_change'] = abs(feats['volume_change'])
    feats['abs_hydro_change'] = abs(feats['hydro_change'])

    return feats

HUMAN_FEAT_NAMES = list(compute_human_features('A', 'V').keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_selfexplain_hybrid")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load SAE encoder weights ──
    import torch
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    mod_score = prot_norm / (prot_norm + dna_norm + 1e-8)

    # ── Load all data ──
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

    print("Loaded %d assays" % len(assay_data), flush=True)

    # ── Compute human features for all variants ──
    print("Computing human-interpretable features ...", flush=True)
    for ad in assay_data:
        n = len(ad['mutants'])
        hf = np.zeros((n, len(HUMAN_FEAT_NAMES)))
        for i in range(n):
            ref_aa, pos, alt_aa = parse_mutant(ad['mutants'][i])
            if ref_aa and alt_aa and ref_aa in AA_TO_IDX and alt_aa in AA_TO_IDX:
                feats = compute_human_features(ref_aa, alt_aa)
                hf[i] = [feats[k] for k in HUMAN_FEAT_NAMES]
        ad['human_feats'] = hf

    # ===================================================================
    # Benchmark: 5-fold CV per assay for multiple feature sets
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("PER-ASSAY BENCHMARK (5-fold CV Ridge)", flush=True)
    print("=" * 70, flush=True)

    per_assay_results = []
    for ad in assay_data:
        y = ad['fitness']
        if len(y) < 50:
            continue
        nf = min(ad['sae_acts'].shape[1], args.n_features)

        feature_sets = {
            'human_only': ad['human_feats'],
            'sae_only': ad['sae_acts'][:, :nf],
            'hybrid': np.hstack([ad['sae_acts'][:, :nf], ad['human_feats']]),
        }

        result = {'assay': ad['name'], 'gene': ad['gene'], 'n': len(y)}

        for fs_name, X in feature_sets.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rhos = []
            for train_idx, test_idx in kf.split(X):
                model = Ridge(alpha=1.0)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                if np.std(y_pred) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                    rho = stats.spearmanr(y_pred, y[test_idx]).statistic
                    if not np.isnan(rho):
                        rhos.append(rho)
            result['rho_' + fs_name] = np.mean(rhos) if rhos else 0.0

        per_assay_results.append(result)

        if (len(per_assay_results)) % 50 == 0:
            print("  Processed %d assays ..." % len(per_assay_results), flush=True)

    res_df = pd.DataFrame(per_assay_results)
    res_df.to_csv(os.path.join(args.out_dir, "per_assay_benchmark.csv"), index=False)

    # Merge with existing results
    existing = os.path.join(args.sae_dir, "..", "dms_v6_mlp", "dms_results.csv")
    if not os.path.exists(existing):
        existing = os.path.join(args.sae_dir, "dms_results.csv")

    if os.path.exists(existing):
        ext_df = pd.read_csv(existing)
        merged = ext_df.merge(res_df[['assay', 'rho_human_only', 'rho_sae_only', 'rho_hybrid']],
                              on='assay', how='inner')
    else:
        merged = res_df
        print("Warning: existing results not found at %s" % existing, flush=True)

    # Summary table
    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 70, flush=True)

    methods = [
        ("ESM2 zero-shot",    "rho_esm2_norm",       "zero-shot",     False),
        ("Evo2 zero-shot",    "rho_evo2_norm",        "zero-shot",     False),
        ("Human features",    "rho_human_only",       "interpretable", True),
        ("Evo2 ridge",        "rho_evo2_ridge",       "linear",        False),
        ("SAE only (linear)", "rho_sae_only",         "interpretable", True),
        ("v6 proj ridge",     "rho_v6_rep_ridge",     "linear",        False),
        ("Fusion ridge",      "rho_fusion_raw_ridge",  "linear",       False),
        ("ESM2 ridge",        "rho_esm2_ridge",       "linear",        False),
        ("HYBRID (ours)",     "rho_hybrid",           "INTERPRETABLE", True),
        ("Evo2 MLP",          "rho_evo2_mlp",         "black-box",     False),
        ("Fusion MLP",        "rho_fusion_raw_mlp",    "black-box",    False),
        ("v6 proj MLP",       "rho_v6_rep_mlp",        "black-box",   False),
        ("ESM2 MLP",          "rho_esm2_mlp",          "black-box",   False),
    ]

    print("%-25s %8s %8s %15s" % ("Method", "Mean", "Median", "Type"), flush=True)
    print("-" * 60, flush=True)
    for name, col, mtype, _ in methods:
        if col in merged.columns:
            vals = merged[col].dropna()
            print("%-25s %8.4f %8.4f %15s" % (name, vals.mean(), vals.median(), mtype),
                  flush=True)

    # Win rates for hybrid
    print("\nWin rates (HYBRID vs baselines):", flush=True)
    if 'rho_hybrid' in merged.columns:
        for name, col, mtype, _ in methods:
            if col in merged.columns and col != 'rho_hybrid':
                both = merged[[col, 'rho_hybrid']].dropna()
                wins = (both['rho_hybrid'] > both[col]).sum()
                total = len(both)
                print("  vs %-25s: %d/%d (%.0f%%)" % (name, wins, total, 100*wins/total),
                      flush=True)

    # ===================================================================
    # Improvement analysis: where does hybrid help most?
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("WHERE DOES HYBRID HELP?", flush=True)
    print("=" * 70, flush=True)

    if 'rho_hybrid' in merged.columns and 'rho_sae_only' in merged.columns:
        merged['hybrid_gain'] = merged['rho_hybrid'] - merged['rho_sae_only']

        print("\nTop 10 assays where hybrid > SAE-only:", flush=True)
        top_gain = merged.nlargest(10, 'hybrid_gain')
        for _, row in top_gain.iterrows():
            print("  %-40s SAE=%.4f Hybrid=%.4f gain=+%.4f" % (
                row['assay'], row['rho_sae_only'], row['rho_hybrid'], row['hybrid_gain']),
                flush=True)

        print("\nMean gain: %.4f" % merged['hybrid_gain'].mean(), flush=True)
        print("Median gain: %.4f" % merged['hybrid_gain'].median(), flush=True)

    # ===================================================================
    # Global model: concept weights for hybrid
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("GLOBAL HYBRID CONCEPT WEIGHTS", flush=True)
    print("=" * 70, flush=True)

    all_X_sae = []
    all_X_human = []
    all_y = []
    for ad in assay_data:
        if len(ad['fitness']) < 50:
            continue
        nf = min(ad['sae_acts'].shape[1], args.n_features)
        y_z = (ad['fitness'] - np.mean(ad['fitness'])) / (np.std(ad['fitness']) + 1e-8)
        all_X_sae.append(ad['sae_acts'][:, :nf])
        all_X_human.append(ad['human_feats'])
        all_y.append(y_z)

    all_X_sae = np.vstack(all_X_sae)
    all_X_human = np.vstack(all_X_human)
    all_y = np.concatenate(all_y)

    # Filter rare SAE features
    MIN_ACTIVE = 500
    active_mask = (all_X_sae > 0).sum(axis=0) >= MIN_ACTIVE
    X_sae_filtered = all_X_sae[:, active_mask]
    active_indices = np.where(active_mask)[0]
    print("Active SAE features (>=%d variants): %d" % (MIN_ACTIVE, active_mask.sum()), flush=True)

    X_hybrid_global = np.hstack([X_sae_filtered, all_X_human])
    print("Hybrid feature dim: %d (SAE) + %d (human) = %d" % (
        X_sae_filtered.shape[1], all_X_human.shape[1], X_hybrid_global.shape[1]), flush=True)

    global_model = Ridge(alpha=10.0)
    global_model.fit(X_hybrid_global, all_y)
    global_rho = stats.spearmanr(global_model.predict(X_hybrid_global), all_y).statistic
    print("Global Spearman (hybrid): %.4f" % global_rho, flush=True)

    # Human concept weights
    n_sae = X_sae_filtered.shape[1]
    human_weights = global_model.coef_[n_sae:]
    sae_weights = global_model.coef_[:n_sae]

    print("\nHuman concept weights:", flush=True)
    for i, name in enumerate(HUMAN_FEAT_NAMES):
        w = human_weights[i]
        direction = "damaging" if w < 0 else "protective"
        print("  %-25s %+.5f  %s" % (name, w, direction), flush=True)

    # Total weight mass: SAE vs human
    sae_mass = np.abs(sae_weights).sum()
    human_mass = np.abs(human_weights).sum()
    print("\nWeight mass: SAE=%.4f (%.0f%%), Human=%.4f (%.0f%%)" % (
        sae_mass, 100*sae_mass/(sae_mass+human_mass),
        human_mass, 100*human_mass/(sae_mass+human_mass)), flush=True)

    # Save
    np.savez(os.path.join(args.out_dir, "hybrid_weights.npz"),
             sae_weights=sae_weights,
             human_weights=human_weights,
             human_feat_names=np.array(HUMAN_FEAT_NAMES),
             active_sae_indices=active_indices,
             mod_score=mod_score,
             bias=np.array([global_model.intercept_]))

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
