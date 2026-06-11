"""
Deep mechanistic analysis of SAE features on DMS data.
For each DNA-driven / universal feature:
1. Build substitution matrix (ref_aa x alt_aa → mean activation)
2. Identify biochemical property it encodes
3. Find cross-gene patterns
4. Compare DNA-driven vs protein-driven feature specificity
5. Look for codon-level constraints (DNA features detecting nucleotide, not AA properties)
"""

import argparse, glob, os, sys, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats

# ── Amino acid properties ──
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

CHARGE = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
HYDROPHOBIC = set('AVILMFWP')
POLAR = set('STCNQDEYKRHG')
SMALL = set('GASTCVP')
AROMATIC = set('FWY')
ALIPHATIC = set('AVIL')

AA_GROUPS = {
    'positive': set('RKH'),
    'negative': set('DE'),
    'hydrophobic': set('AVILMFW'),
    'polar_uncharged': set('STNQ'),
    'aromatic': set('FWY'),
    'tiny': set('GAS'),
    'proline': set('P'),
    'cysteine': set('C'),
}

GROUP_NAMES = {
    'positive': 'Pos(R/K/H)',
    'negative': 'Neg(D/E)',
    'hydrophobic': 'Hydro(AVILMFW)',
    'polar_uncharged': 'Polar(STNQ)',
    'aromatic': 'Arom(FWY)',
    'tiny': 'Tiny(GAS)',
    'proline': 'Pro',
    'cysteine': 'Cys',
}

# Codons per amino acid (genetic code degeneracy)
CODONS_PER_AA = {
    'F': 2, 'L': 6, 'I': 3, 'M': 1, 'V': 4,
    'S': 6, 'P': 4, 'T': 4, 'A': 4,
    'Y': 2, 'H': 2, 'Q': 2, 'N': 2, 'K': 2, 'D': 2, 'E': 2,
    'C': 2, 'W': 1, 'R': 6, 'G': 4, '*': 3,
}

# Min nucleotide changes needed for AA substitution (approximate, codon-averaged)
# This is simplified — actual value depends on specific codons
def min_nt_changes(ref_aa, alt_aa):
    """Rough estimate of minimum nucleotide changes for an AA substitution."""
    if ref_aa == alt_aa:
        return 0
    # Use standard genetic code to compute
    CODON_TABLE_FULL = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    AA_TO_CODONS = defaultdict(list)
    for codon, aa in CODON_TABLE_FULL.items():
        AA_TO_CODONS[aa].append(codon)

    ref_codons = AA_TO_CODONS.get(ref_aa, [])
    alt_codons = AA_TO_CODONS.get(alt_aa, [])
    if not ref_codons or not alt_codons:
        return 3
    min_changes = 3
    for rc in ref_codons:
        for ac in alt_codons:
            changes = sum(1 for a, b in zip(rc, ac) if a != b)
            min_changes = min(min_changes, changes)
    return min_changes


def parse_mutant(mutant_str):
    """Parse 'A123V' or 'p.A123V' into (ref_aa, pos, alt_aa). Also handles multi-mutants."""
    mutant_str = str(mutant_str).strip()
    # Handle multi-mutants (take first)
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    # Remove p. prefix
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def get_aa_group(aa):
    """Return which group an amino acid belongs to."""
    for group, members in AA_GROUPS.items():
        if aa in members:
            return group
    return 'other'


def substitution_type(ref_aa, alt_aa):
    """Classify the substitution type."""
    ref_g = get_aa_group(ref_aa)
    alt_g = get_aa_group(alt_aa)

    # Specific interesting patterns
    if ref_aa == 'C':
        return f'Cys_loss(C→{alt_aa})'
    if alt_aa == 'C':
        return f'Cys_gain(→C)'
    if alt_aa == 'P':
        return f'Pro_intro(→P)'
    if ref_aa == 'P':
        return f'Pro_loss(P→{alt_aa})'
    if ref_aa == 'G':
        return f'Gly_loss(G→{alt_aa})'

    if ref_aa in HYDROPHOBIC and alt_aa not in HYDROPHOBIC:
        charge_str = ""
        if alt_aa in CHARGE and CHARGE[alt_aa] != 0:
            charge_str = "_charged"
        return f'hydro→polar{charge_str}'
    if ref_aa not in HYDROPHOBIC and alt_aa in HYDROPHOBIC:
        return 'polar→hydro'

    ref_c = CHARGE.get(ref_aa, 0)
    alt_c = CHARGE.get(alt_aa, 0)
    if ref_c * alt_c < 0:
        return 'charge_reversal'
    if ref_c == 0 and abs(alt_c) > 0:
        return 'neutral→charged'
    if abs(ref_c) > 0 and alt_c == 0:
        return 'charged→neutral'

    return f'{ref_g}→{alt_g}'


def analyze_feature_mechanism(feat_idx, all_data, n_features):
    """
    Analyze what biochemical constraint a single feature encodes.
    all_data: list of (assay_name, mutations, activations, fitness, prot_reps, dna_reps)
    """
    # Collect all variants where this feature activates
    active_variants = []  # (assay, ref_aa, pos, alt_aa, activation, fitness)
    inactive_variants = []

    for assay_name, mutations, sae_acts, fitness, prot_reps, dna_reps in all_data:
        if sae_acts.shape[1] <= feat_idx:
            continue
        acts = sae_acts[:, feat_idx]
        for i in range(len(mutations)):
            ref_aa, pos, alt_aa = parse_mutant(mutations[i])
            if ref_aa is None or ref_aa not in AA_TO_IDX or alt_aa not in AA_TO_IDX:
                continue
            if acts[i] > 0:
                active_variants.append((assay_name, ref_aa, pos, alt_aa,
                                       acts[i], fitness[i]))
            else:
                inactive_variants.append((assay_name, ref_aa, pos, alt_aa,
                                         0.0, fitness[i]))

    if len(active_variants) < 50:
        return None

    # ── 1. Substitution matrix ──
    sub_matrix = np.zeros((20, 20))
    sub_counts = np.zeros((20, 20))
    for _, ref_aa, _, alt_aa, act, _ in active_variants:
        ri, ai = AA_TO_IDX[ref_aa], AA_TO_IDX[alt_aa]
        sub_matrix[ri, ai] += act
        sub_counts[ri, ai] += 1

    # Also build background count matrix (all variants, active or not)
    bg_counts = np.zeros((20, 20))
    for _, ref_aa, _, alt_aa, _, _ in active_variants + inactive_variants:
        ri, ai = AA_TO_IDX.get(ref_aa), AA_TO_IDX.get(alt_aa)
        if ri is not None and ai is not None:
            bg_counts[ri, ai] += 1

    # Mean activation per substitution
    mean_act = np.divide(sub_matrix, sub_counts, where=sub_counts > 0,
                         out=np.zeros_like(sub_matrix))

    # Enrichment: P(active | sub_type) / P(active | overall)
    overall_rate = len(active_variants) / max(len(active_variants) + len(inactive_variants), 1)
    enrichment = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if bg_counts[i, j] > 10:
                rate = sub_counts[i, j] / bg_counts[i, j]
                enrichment[i, j] = rate / max(overall_rate, 1e-8)

    # ── 2. Top substitution types ──
    sub_type_acts = defaultdict(list)
    sub_type_counts = Counter()
    for _, ref_aa, _, alt_aa, act, fit in active_variants:
        st = substitution_type(ref_aa, alt_aa)
        sub_type_acts[st].append(act)
        sub_type_counts[st] += 1

    sub_type_summary = []
    for st, acts in sub_type_acts.items():
        sub_type_summary.append({
            'type': st,
            'count': len(acts),
            'mean_act': np.mean(acts),
            'max_act': np.max(acts),
            'frac': len(acts) / len(active_variants),
        })
    sub_type_summary.sort(key=lambda x: -x['mean_act'])

    # ── 3. Ref AA specificity ──
    ref_aa_acts = defaultdict(list)
    for _, ref_aa, _, alt_aa, act, _ in active_variants:
        ref_aa_acts[ref_aa].append(act)
    ref_specificity = {aa: (np.mean(acts), len(acts))
                       for aa, acts in ref_aa_acts.items()}

    # Alt AA specificity
    alt_aa_acts = defaultdict(list)
    for _, ref_aa, _, alt_aa, act, _ in active_variants:
        alt_aa_acts[alt_aa].append(act)
    alt_specificity = {aa: (np.mean(acts), len(acts))
                       for aa, acts in alt_aa_acts.items()}

    # ── 4. Cross-gene patterns ──
    gene_acts = defaultdict(list)
    for assay, _, _, _, act, _ in active_variants:
        gene = assay.split('_')[0]  # rough gene name
        gene_acts[gene].append(act)
    gene_summary = {g: (np.mean(a), len(a)) for g, a in gene_acts.items()}

    # ── 5. Fitness correlation for active variants ──
    if len(active_variants) > 50:
        acts_arr = np.array([v[4] for v in active_variants])
        fit_arr = np.array([v[5] for v in active_variants])
        rho_active = stats.spearmanr(acts_arr, fit_arr).statistic
    else:
        rho_active = np.nan

    # ── 6. Nucleotide change analysis ──
    nt_change_acts = defaultdict(list)
    for _, ref_aa, _, alt_aa, act, _ in active_variants:
        nc = min_nt_changes(ref_aa, alt_aa)
        nt_change_acts[nc].append(act)

    # ── 7. Feature specificity score ──
    # High specificity = feature activates on few substitution types
    # Use entropy of substitution type distribution
    type_probs = np.array([s['count'] for s in sub_type_summary], dtype=float)
    type_probs = type_probs / type_probs.sum()
    entropy = -np.sum(type_probs * np.log2(type_probs + 1e-10))
    max_entropy = np.log2(len(sub_type_summary) + 1e-10)
    specificity = 1 - entropy / max(max_entropy, 1)

    # ── 8. Group-level transition matrix ──
    group_names = list(AA_GROUPS.keys())
    group_matrix = np.zeros((len(group_names), len(group_names)))
    group_matrix_bg = np.zeros((len(group_names), len(group_names)))

    for _, ref_aa, _, alt_aa, act, _ in active_variants:
        rg = get_aa_group(ref_aa)
        ag = get_aa_group(alt_aa)
        if rg in group_names and ag in group_names:
            ri = group_names.index(rg)
            ai = group_names.index(ag)
            group_matrix[ri, ai] += act

    for _, ref_aa, _, alt_aa, _, _ in active_variants + inactive_variants:
        rg = get_aa_group(ref_aa)
        ag = get_aa_group(alt_aa)
        if rg in group_names and ag in group_names:
            ri = group_names.index(rg)
            ai = group_names.index(ag)
            group_matrix_bg[ri, ai] += 1

    return {
        'feat_idx': feat_idx,
        'n_active': len(active_variants),
        'n_total': len(active_variants) + len(inactive_variants),
        'activation_rate': len(active_variants) / max(len(active_variants) + len(inactive_variants), 1),
        'mean_activation': np.mean([v[4] for v in active_variants]),
        'specificity': specificity,
        'rho_fitness': rho_active,
        'top_sub_types': sub_type_summary[:10],
        'ref_specificity': ref_specificity,
        'alt_specificity': alt_specificity,
        'gene_summary': gene_summary,
        'nt_changes': {k: (np.mean(v), len(v)) for k, v in nt_change_acts.items()},
        'n_genes': len(gene_summary),
        'sub_matrix_mean': mean_act,
        'enrichment_matrix': enrichment,
        'group_matrix': group_matrix,
        'group_names': group_names,
    }


def summarize_mechanism(result):
    """Generate human-readable mechanism summary."""
    if result is None:
        return "Insufficient data"

    lines = []
    fi = result['feat_idx']
    lines.append(f"F{fi:05d}: {result['n_active']}/{result['n_total']} active "
                 f"({result['activation_rate']:.1%}), "
                 f"specificity={result['specificity']:.3f}, "
                 f"fitness_rho={result['rho_fitness']:.3f}")

    # Top substitution types
    lines.append("  Top substitution types:")
    for s in result['top_sub_types'][:5]:
        lines.append(f"    {s['type']:30s} n={s['count']:5d}  "
                     f"mean_act={s['mean_act']:.3f}  frac={s['frac']:.1%}")

    # Ref AA with highest mean activation
    ref_sorted = sorted(result['ref_specificity'].items(),
                       key=lambda x: -x[1][0])
    lines.append("  Top ref AAs (by mean activation):")
    for aa, (mean_a, cnt) in ref_sorted[:5]:
        if cnt >= 20:
            lines.append(f"    {aa}: mean_act={mean_a:.3f}  n={cnt}")

    # Alt AA with highest mean activation
    alt_sorted = sorted(result['alt_specificity'].items(),
                       key=lambda x: -x[1][0])
    lines.append("  Top alt AAs (by mean activation):")
    for aa, (mean_a, cnt) in alt_sorted[:5]:
        if cnt >= 20:
            lines.append(f"    {aa}: mean_act={mean_a:.3f}  n={cnt}")

    # Nucleotide change pattern
    lines.append("  Nucleotide changes needed:")
    for nc in sorted(result['nt_changes'].keys()):
        mean_a, cnt = result['nt_changes'][nc]
        lines.append(f"    {nc} nt changes: mean_act={mean_a:.3f}  n={cnt}")

    # Cross-gene breadth
    top_genes = sorted(result['gene_summary'].items(),
                      key=lambda x: -x[1][0])[:5]
    lines.append(f"  Active in {result['n_genes']} genes. Top:")
    for g, (mean_a, cnt) in top_genes:
        lines.append(f"    {g}: mean_act={mean_a:.3f}  n={cnt}")

    # Mechanism hypothesis
    lines.append("  --- MECHANISM HYPOTHESIS ---")

    # Check for specific patterns
    top_type = result['top_sub_types'][0]['type'] if result['top_sub_types'] else ''

    # Check if ref AA is highly specific
    ref_entropy = 0
    ref_total = sum(v[1] for v in result['ref_specificity'].values())
    for aa, (_, cnt) in result['ref_specificity'].items():
        p = cnt / max(ref_total, 1)
        if p > 0:
            ref_entropy -= p * np.log2(p)
    max_ref_entropy = np.log2(len(result['ref_specificity']))

    # Check if alt AA is highly specific
    alt_top = alt_sorted[0] if alt_sorted else ('?', (0, 0))
    alt_frac = alt_top[1][1] / max(sum(v[1] for v in result['alt_specificity'].values()), 1)

    # Generate hypothesis
    if alt_top[0] == 'P' and alt_frac > 0.15:
        lines.append("  ** X→Pro detector: proline introduction disrupts backbone **")
    elif ref_sorted[0][0] == 'C' and ref_sorted[0][1][1] / max(ref_total, 1) > 0.10:
        lines.append("  ** Cysteine loss detector: Cys→X disruption (disulfide/metal) **")
    elif ref_sorted[0][0] == 'G' and ref_sorted[0][1][1] / max(ref_total, 1) > 0.12:
        lines.append("  ** Glycine loss detector: Gly→X at constrained positions **")
    elif 'hydro→polar_charged' in top_type or 'hydro→polar' in top_type:
        lines.append("  ** Hydrophobic core disruption: buried hydrophobic→polar/charged **")
    elif 'charge_reversal' in top_type:
        lines.append("  ** Charge reversal detector: +→- or -→+ at charge-critical sites **")
    elif result['specificity'] > 0.7:
        lines.append(f"  ** Highly specific: dominant pattern = {top_type} **")
    elif result['specificity'] < 0.3:
        lines.append("  ** Non-specific: broad damage detector (general perturbation) **")
    else:
        lines.append(f"  ** Moderate specificity: primary = {top_type} **")

    # Check for nucleotide-level signal
    nt_acts = result['nt_changes']
    if 1 in nt_acts and 2 in nt_acts:
        nt1_mean, nt1_n = nt_acts[1]
        nt2_mean, nt2_n = nt_acts[2]
        if nt1_n > 50 and nt2_n > 50:
            ratio = nt1_mean / max(nt2_mean, 1e-6)
            if ratio > 1.3:
                lines.append(f"  ** CODON-LEVEL SIGNAL: 1-nt changes activate {ratio:.2f}x "
                             f"stronger than 2-nt changes **")
            elif ratio < 0.7:
                lines.append(f"  ** ANTI-CODON: 2-nt changes activate {1/ratio:.2f}x stronger "
                             f"(non-local DNA effect?) **")

    return '\n'.join(lines)


def find_novel_mechanisms(all_results, all_data):
    """
    Cross-feature analysis to find genuinely novel patterns.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("NOVEL MECHANISM SEARCH")
    lines.append("=" * 80)

    # ── 1. Features that differ between DNA-driven and protein-driven ──
    lines.append("\n--- 1. DNA vs Protein feature mechanism comparison ---")

    dna_features = [(r['feat_idx'], r) for r in all_results
                    if r is not None and r.get('modality') == 'DNA']
    prot_features = [(r['feat_idx'], r) for r in all_results
                     if r is not None and r.get('modality') == 'PROT']

    # Compare substitution specificity
    if dna_features and prot_features:
        dna_spec = np.mean([r['specificity'] for _, r in dna_features])
        prot_spec = np.mean([r['specificity'] for _, r in prot_features])
        lines.append(f"  DNA feature avg specificity: {dna_spec:.3f}")
        lines.append(f"  PROT feature avg specificity: {prot_spec:.3f}")

    # ── 2. Features with strong nucleotide-level signal ──
    lines.append("\n--- 2. Features with codon-level signal ---")
    codon_features = []
    for r in all_results:
        if r is None:
            continue
        nt = r.get('nt_changes', {})
        if 1 in nt and 2 in nt:
            nt1_mean, nt1_n = nt[1]
            nt2_mean, nt2_n = nt[2]
            if nt1_n > 100 and nt2_n > 100:
                ratio = nt1_mean / max(nt2_mean, 1e-6)
                codon_features.append((r['feat_idx'], ratio, nt1_mean, nt2_mean,
                                       nt1_n, nt2_n, r.get('modality', '?')))

    codon_features.sort(key=lambda x: -abs(x[1] - 1))
    lines.append(f"  Features with codon-level bias (1-nt vs 2-nt change activation ratio):")
    for fi, ratio, m1, m2, n1, n2, mod in codon_features[:15]:
        direction = "1-nt stronger" if ratio > 1 else "2-nt stronger"
        lines.append(f"    F{fi:05d} [{mod:5s}]: ratio={ratio:.3f} "
                     f"(1nt: {m1:.3f} n={n1}, 2nt: {m2:.3f} n={n2}) → {direction}")

    # ── 3. Cross-gene conserved features ──
    lines.append("\n--- 3. Cross-gene mechanism conservation ---")
    # Features active in many genes with consistent substitution pattern
    for r in all_results:
        if r is None or r['n_genes'] < 20:
            continue
        if r['specificity'] > 0.5:
            top = r['top_sub_types'][0] if r['top_sub_types'] else {}
            lines.append(f"  F{r['feat_idx']:05d}: {r['n_genes']} genes, "
                         f"spec={r['specificity']:.3f}, "
                         f"top_pattern={top.get('type', '?')} "
                         f"({top.get('frac', 0):.0%}), "
                         f"mod={r.get('modality', '?')}")

    # ── 4. Position-dependent features ──
    lines.append("\n--- 4. Position-dependent activation patterns ---")
    # For each feature, check if activation correlates with position in protein
    for r in all_results:
        if r is None:
            continue
        # Check if specific ref AAs dominate
        ref_spec = r['ref_specificity']
        total = sum(v[1] for v in ref_spec.values())
        for aa, (mean_a, cnt) in ref_spec.items():
            frac = cnt / max(total, 1)
            if frac > 0.25 and cnt > 100:
                lines.append(f"  F{r['feat_idx']:05d}: {frac:.0%} of activations are "
                             f"from ref={aa} (n={cnt}, mean_act={mean_a:.3f}), "
                             f"mod={r.get('modality', '?')}")

    # ── 5. DNA features detecting protein-level invisible damage ──
    lines.append("\n--- 5. DNA features with unique biological signal ---")
    for r in all_results:
        if r is None or r.get('modality') != 'DNA':
            continue
        if abs(r['rho_fitness']) > 0.05 and r['n_active'] > 500:
            lines.append(f"  F{r['feat_idx']:05d}: rho_fitness={r['rho_fitness']:.3f}, "
                         f"n_active={r['n_active']}, n_genes={r['n_genes']}, "
                         f"spec={r['specificity']:.3f}")
            for s in r['top_sub_types'][:3]:
                lines.append(f"    {s['type']:30s} n={s['count']} mean_act={s['mean_act']:.3f}")

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_mechanism")
    ap.add_argument("--n_features", type=int, default=12288)
    ap.add_argument("--top_k", type=int, default=60,
                    help="Number of features to analyze in detail")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load encoder weights for modality attribution ──
    import torch
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()  # (n_features, d_input=1536)
    # First 768 = protein, last 768 = DNA
    prot_norm = np.linalg.norm(enc_w[:, :768], axis=1)
    dna_norm = np.linalg.norm(enc_w[:, 768:], axis=1)
    modality_score = prot_norm / (prot_norm + dna_norm + 1e-8)
    # modality_score > 0.6 = PROT, < 0.4 = DNA, else CROSS

    # ── Load all SAE activations ──
    print("Loading SAE activations ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))
    print(f"Found {len(sae_files)} SAE files", flush=True)

    all_data = []
    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        sae_acts = d["sae_acts"]
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        prot_reps = d["prot_reps"] if "prot_reps" in d else None
        dna_reps = d["dna_reps"] if "dna_reps" in d else None

        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue
        all_data.append((assay_name, mutants, sae_acts, fitness, prot_reps, dna_reps))

    print(f"Loaded {len(all_data)} assays with mutant labels", flush=True)

    # ── Select features to analyze ──
    # Top DNA features + top universal + top protein
    dna_feats = np.where(modality_score < 0.4)[0]
    prot_feats = np.where(modality_score > 0.6)[0]
    cross_feats = np.where((modality_score >= 0.4) & (modality_score <= 0.6))[0]

    # Compute per-feature DMS signal strength (|mean rho| across assays)
    print("Computing per-feature DMS signal ...", flush=True)
    feat_signal = np.zeros(args.n_features)
    feat_n_assays = np.zeros(args.n_features)
    for assay_name, mutants, sae_acts, fitness, _, _ in all_data:
        if len(fitness) < 50:
            continue
        for fi in range(min(sae_acts.shape[1], args.n_features)):
            col = sae_acts[:, fi]
            if (col > 0).sum() < 10:
                continue
            rho = stats.spearmanr(col, fitness).statistic
            if not np.isnan(rho):
                feat_signal[fi] += abs(rho)
                feat_n_assays[fi] += 1

    feat_mean_signal = np.divide(feat_signal, feat_n_assays,
                                  where=feat_n_assays > 0,
                                  out=np.zeros_like(feat_signal))

    # Select top features from each modality
    features_to_analyze = set()

    # Top 20 DNA-driven by signal
    dna_signal = [(fi, feat_mean_signal[fi]) for fi in dna_feats
                  if feat_n_assays[fi] > 50]
    dna_signal.sort(key=lambda x: -x[1])
    for fi, _ in dna_signal[:20]:
        features_to_analyze.add(fi)

    # Top 20 protein-driven
    prot_signal = [(fi, feat_mean_signal[fi]) for fi in prot_feats
                   if feat_n_assays[fi] > 50]
    prot_signal.sort(key=lambda x: -x[1])
    for fi, _ in prot_signal[:20]:
        features_to_analyze.add(fi)

    # Top 20 cross-modal
    cross_signal = [(fi, feat_mean_signal[fi]) for fi in cross_feats
                    if feat_n_assays[fi] > 50]
    cross_signal.sort(key=lambda x: -x[1])
    for fi, _ in cross_signal[:20]:
        features_to_analyze.add(fi)

    features_to_analyze = sorted(features_to_analyze)
    print(f"\nAnalyzing {len(features_to_analyze)} features "
          f"(DNA: {sum(1 for f in features_to_analyze if modality_score[f] < 0.4)}, "
          f"PROT: {sum(1 for f in features_to_analyze if modality_score[f] > 0.6)}, "
          f"CROSS: {sum(1 for f in features_to_analyze if 0.4 <= modality_score[f] <= 0.6)})",
          flush=True)

    # ── Analyze each feature ──
    all_results = []
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SAE FEATURE MECHANISM ANALYSIS")
    report_lines.append("=" * 80)

    for feat_idx in features_to_analyze:
        mod = modality_score[feat_idx]
        mod_label = "DNA" if mod < 0.4 else ("PROT" if mod > 0.6 else "CROSS")
        print(f"  Analyzing F{feat_idx:05d} (mod={mod:.3f}, {mod_label}) ...", flush=True)

        result = analyze_feature_mechanism(feat_idx, all_data, args.n_features)
        if result is not None:
            result['modality'] = mod_label
            result['modality_score'] = mod
        all_results.append(result)

        report_lines.append(f"\n{'─' * 60}")
        report_lines.append(f"[{mod_label}] mod={mod:.3f}  signal={feat_mean_signal[feat_idx]:.4f}")
        report_lines.append(summarize_mechanism(result))

    # ── Novel mechanism search ──
    novel_report = find_novel_mechanisms(all_results, all_data)
    report_lines.append("\n\n" + novel_report)

    # ── Save report ──
    report_text = '\n'.join(report_lines)
    with open(os.path.join(args.out_dir, "mechanism_report.txt"), 'w') as f:
        f.write(report_text)
    print(report_text)

    # ── Save structured data ──
    summary_rows = []
    for r in all_results:
        if r is None:
            continue
        top_type = r['top_sub_types'][0] if r['top_sub_types'] else {}
        top_ref = max(r['ref_specificity'].items(), key=lambda x: x[1][0])
        top_alt = max(r['alt_specificity'].items(), key=lambda x: x[1][0])

        # Codon ratio
        nt = r.get('nt_changes', {})
        codon_ratio = np.nan
        if 1 in nt and 2 in nt and nt[1][1] > 50 and nt[2][1] > 50:
            codon_ratio = nt[1][0] / max(nt[2][0], 1e-6)

        summary_rows.append({
            'feature': f"F{r['feat_idx']:05d}",
            'modality': r.get('modality', '?'),
            'mod_score': r.get('modality_score', np.nan),
            'n_active': r['n_active'],
            'n_total': r['n_total'],
            'activation_rate': r['activation_rate'],
            'specificity': r['specificity'],
            'rho_fitness': r['rho_fitness'],
            'n_genes': r['n_genes'],
            'top_sub_type': top_type.get('type', ''),
            'top_sub_frac': top_type.get('frac', 0),
            'top_ref_aa': top_ref[0],
            'top_alt_aa': top_alt[0],
            'codon_ratio_1nt_vs_2nt': codon_ratio,
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "mechanism_summary.csv"), index=False)

    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
