#!/bin/bash
#SBATCH --job-name=seed_bhfix
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/crosscoder_seed_stability_bhfix/slurm_%j.log

set -e
cd /data/user/mzhang630/data/bioinfo/implementation
source activate sake

mkdir -p results/crosscoder_seed_stability_bhfix

echo "=== Re-running BH-fixed analysis on seed 0 ==="
python scripts/analyze_crosscoder.py \
  --checkpoint results/crosscoder_seed_stability/seed_0/crosscoder.pt \
  --out_dir results/crosscoder_seed_stability_bhfix/seed_0

echo "=== Re-running BH-fixed analysis on seed 1 ==="
python scripts/analyze_crosscoder.py \
  --checkpoint results/crosscoder_seed_stability/seed_1/crosscoder.pt \
  --out_dir results/crosscoder_seed_stability_bhfix/seed_1

echo "=== Re-running BH-fixed analysis on seed 2 ==="
python scripts/analyze_crosscoder.py \
  --checkpoint results/crosscoder_seed_stability/seed_2/crosscoder.pt \
  --out_dir results/crosscoder_seed_stability_bhfix/seed_2

echo "=== Generating Control B comparison summary ==="
python3 -c "
import json, os

main = json.load(open('results/crosscoder_sae/analysis_bhfix/analysis_summary.json'))
shuf = json.load(open('results/crosscoder_sae_shuffled/analysis_bhfix/analysis_summary.json'))

def extract(d):
    fc = d['feature_categories']
    cv = d['clinvar_pathogenicity']['by_category']
    return {
        'PP': fc['prot_private'],
        'DP': fc['dna_private'],
        'SH': fc['shared'],
        'dead': fc['dead'],
        'SH_median_OR': cv.get('shared', {}).get('median_or', None),
        'PP_median_OR': cv.get('prot-private', {}).get('median_or', None),
        'DP_median_OR': cv.get('dna-private', {}).get('median_or', None),
        'SH_FDR05': cv.get('shared', {}).get('n_fdr05', None),
        'PP_FDR05': cv.get('prot-private', {}).get('n_fdr05', None),
        'DP_FDR05': cv.get('dna-private', {}).get('n_fdr05', None),
        'ctrl_a_t': d.get('control_a_inference_shuffle', {}).get('t_statistic', None),
        'ctrl_a_p': d.get('control_a_inference_shuffle', {}).get('p_value', None),
    }

summary = {
    'real_pairs': extract(main),
    'shuffled_pairs': extract(shuf),
    'delta_SH_count': extract(main)['SH'] - extract(shuf)['SH'],
    'note': 'BH-FDR fixed (standard implementation). Control B = train-time shuffle of ESM-Evo pairings.'
}

os.makedirs('results/crosscoder_control_b_bhfix', exist_ok=True)
with open('results/crosscoder_control_b_bhfix/comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Control B comparison summary written.')
"

echo "=== Generating seed stability BH-fixed summary ==="
python3 -c "
import json, os, csv

seeds = [42, 0, 1, 2]
seed_files = {
    42: 'results/crosscoder_sae/analysis_bhfix/analysis_summary.json',
    0: 'results/crosscoder_seed_stability_bhfix/seed_0/analysis_summary.json',
    1: 'results/crosscoder_seed_stability_bhfix/seed_1/analysis_summary.json',
    2: 'results/crosscoder_seed_stability_bhfix/seed_2/analysis_summary.json',
}

rows = []
for s in seeds:
    d = json.load(open(seed_files[s]))
    fc = d['feature_categories']
    cv = d['clinvar_pathogenicity']['by_category']
    ca = d.get('control_a_inference_shuffle', {})
    rows.append({
        'seed': s,
        'PP': fc['prot_private'],
        'DP': fc['dna_private'],
        'SH': fc['shared'],
        'dead': fc['dead'],
        'SH_median_OR': cv.get('shared', {}).get('median_or'),
        'PP_median_OR': cv.get('prot-private', {}).get('median_or'),
        'DP_median_OR': cv.get('dna-private', {}).get('median_or'),
        'SH_FDR05': cv.get('shared', {}).get('n_fdr05'),
        'PP_FDR05': cv.get('prot-private', {}).get('n_fdr05'),
        'DP_FDR05': cv.get('dna-private', {}).get('n_fdr05'),
        'ctrl_a_t': ca.get('t_statistic'),
        'ctrl_a_delta': ca.get('delta_mean'),
    })

outpath = 'results/crosscoder_seed_stability_bhfix/seed_stability_bhfix_summary.csv'
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'Seed stability BH-fixed summary: {outpath}')
print()
for r in rows:
    print(f'Seed {r[\"seed\"]}: PP={r[\"PP\"]}, DP={r[\"DP\"]}, SH={r[\"SH\"]}, dead={r[\"dead\"]}')
    print(f'  OR: SH={r[\"SH_median_OR\"]}, PP={r[\"PP_median_OR\"]}, DP={r[\"DP_median_OR\"]}')
    print(f'  FDR05: SH={r[\"SH_FDR05\"]}, PP={r[\"PP_FDR05\"]}, DP={r[\"DP_FDR05\"]}')
    print(f'  CtrlA t={r[\"ctrl_a_t\"]}, delta={r[\"ctrl_a_delta\"]}')
"

echo "=== ALL DONE ==="
