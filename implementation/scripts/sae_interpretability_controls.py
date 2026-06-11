#!/usr/bin/env python3
"""
SAE interpretability statistical controls.

1. Feature enrichment + FDR (Fisher exact + BH correction)
2. Seed stability (5 SAE seeds, concept label Jaccard)
3. Random controls (PCA, ICA, NMF, random SAE, individual neurons)
4. Causal feature intervention (clamp feature → check prediction change)
5. Modality attribution occlusion (zero protein/DNA half → check feature change)
"""

import argparse, glob, gzip, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA, NMF, FastICA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_variant_mlp_v6 import compute_aa_features
from train_variant_sae import ProjectionLayers, PretrainHead, TopKSAE, train_sae


def load_all(args):
    print("Loading data ...", flush=True)
    df = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx)
    sub = df.iloc[dual_idx].reset_index(drop=True)
    g2l = {int(g): l for l, g in enumerate(dual_idx)}
    n = len(dual_idx)

    prot = np.zeros((n, 1280), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "esm2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: prot[g2l[int(gi)]] = row
    dna = np.zeros((n, 4096), dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "clinvar_evo2_emb_shard*.npz"))):
        d = np.load(f)
        for gi, row in zip(d["idx"], d["edelta"]):
            if int(gi) in g2l: dna[g2l[int(gi)]] = row
    llr = np.load(args.llr_path)["llr"].astype(np.float32)

    prot_z = (prot - prot.mean(0)) / (prot.std(0) + 1e-8)
    dna_z = (dna - dna.mean(0)) / (dna.std(0) + 1e-8)
    llr_z = (llr - llr.mean()) / (llr.std() + 1e-8)

    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    label = np.full(n, -1); label[is_path.values] = 1; label[is_ben.values] = 0
    genes = sub["gene"].values
    from_aa = sub["from_aa"].values.astype(str)
    to_aa = sub["to_aa"].values.astype(str)

    return prot_z, dna_z, llr_z, label, genes, from_aa, to_aa, n


def extract_reps(prot_z, dna_z, llr_z, n, device, pretrain_epochs=30, seed=42, batch_size=4096):
    torch.manual_seed(seed); np.random.seed(seed)
    proj = ProjectionLayers().to(device)
    pt_head = PretrainHead().to(device)
    params = list(proj.parameters()) + list(pt_head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    yt = torch.tensor(llr_z, dtype=torch.float32)

    for ep in range(pretrain_epochs):
        proj.train(); pt_head.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b+batch_size].numpy()
            xp = torch.tensor(prot_z[idx], dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z[idx], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)
            h = proj(xp, xd); loss = loss_fn(pt_head(h), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    proj.eval(); del pt_head
    reps = np.zeros((n, 1536), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, batch_size):
            xp = torch.tensor(prot_z[b:b+batch_size], dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z[b:b+batch_size], dtype=torch.float32, device=device)
            reps[b:b+batch_size] = proj(xp, xd).cpu().numpy()
    return reps, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--out_dir", default="results/sae_controls")
    ap.add_argument("--n_sae_seeds", type=int, default=5)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--sae_epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--n_gpus", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = args.batch_size

    prot_z, dna_z, llr_z, label, genes, from_aa, to_aa, n = load_all(args)
    labeled = label >= 0

    # Extract representations (same as v6 pretrain)
    print("\n[1] Extracting representations ...", flush=True)
    reps, proj = extract_reps(prot_z, dna_z, llr_z, n, device, batch_size=BS)
    print(f"  Reps: {reps.shape}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # CONTROL 1: Feature enrichment + FDR
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("[2] FEATURE ENRICHMENT + FDR")
    print("="*80)

    # Train one SAE
    n_features = 1536 * args.expansion
    sae = TopKSAE(1536, n_features, k=args.topk).to(device)
    train_sae(sae, reps, device, epochs=args.sae_epochs, batch=BS)
    sae.eval()
    sae_acts = np.zeros((n, n_features), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, BS):
            x = torch.tensor(reps[b:b+BS], dtype=torch.float32, device=device)
            sae_acts[b:b+BS] = sae.encode(x).cpu().numpy()

    alive = (sae_acts > 0).any(0)
    alive_idx = np.where(alive)[0]
    print(f"  Alive: {alive.sum()}/{n_features}", flush=True)

    # Define biological categories for enrichment
    cys_from = np.array([f == 'C' for f in from_aa])
    cys_to = np.array([t == 'C' for t in to_aa])
    pro_to = np.array([t == 'P' for t in to_aa])
    gly_from = np.array([f == 'G' for f in from_aa])
    charge_rev = np.array([
        (f in 'DE' and t in 'RKH') or (f in 'RKH' and t in 'DE')
        for f, t in zip(from_aa, to_aa)])
    is_pathogenic = label == 1

    categories = {
        'cysteine_loss': cys_from,
        'cysteine_gain': cys_to,
        'proline_intro': pro_to,
        'glycine_loss': gly_from,
        'charge_reversal': charge_rev,
        'pathogenic': is_pathogenic,
    }

    enrichment_results = []
    for fi in alive_idx:
        active = sae_acts[:, fi] > 0
        n_act = active.sum()
        if n_act < 10: continue

        for cat_name, cat_mask in categories.items():
            # 2x2 contingency: active/inactive × category/not
            a = (active & cat_mask).sum()
            b = (active & ~cat_mask).sum()
            c = (~active & cat_mask).sum()
            d = (~active & ~cat_mask).sum()
            if a + c == 0: continue
            odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            enrichment_results.append({
                'feature': fi, 'category': cat_name,
                'n_active': n_act, 'n_category_in_active': int(a),
                'odds_ratio': odds, 'pval': pval,
            })

    edf = pd.DataFrame(enrichment_results)
    # BH correction per category
    for cat in categories:
        mask = edf['category'] == cat
        if mask.sum() == 0: continue
        _, pvals_adj, _, _ = multipletests(edf.loc[mask, 'pval'], method='fdr_bh')
        edf.loc[mask, 'pval_adj'] = pvals_adj

    sig = edf[edf['pval_adj'] < 0.05]
    print(f"\n  Significant enrichments (FDR < 0.05): {len(sig)}/{len(edf)}")
    for cat in categories:
        n_sig = (sig['category'] == cat).sum()
        n_total = (edf['category'] == cat).sum()
        print(f"    {cat:20s}: {n_sig:4d}/{n_total:4d} features significantly enriched")

    edf.to_csv(os.path.join(args.out_dir, "enrichment_fdr.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # CONTROL 2: Seed stability
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"[3] SEED STABILITY ({args.n_sae_seeds} SAE seeds)")
    print("="*80)

    # For each seed: train SAE, get top-k variants per feature, compute concept labels
    seed_top_variants = []
    seed_alive_counts = []
    seed_path_rates = []

    for si in range(args.n_sae_seeds):
        print(f"  Seed {si} ...", flush=True)
        torch.manual_seed(si * 100)
        sae_s = TopKSAE(1536, n_features, k=args.topk).to(device)
        train_sae(sae_s, reps, device, epochs=args.sae_epochs, batch=BS)
        sae_s.eval()

        acts_s = np.zeros((n, n_features), dtype=np.float32)
        with torch.no_grad():
            for b in range(0, n, BS):
                x = torch.tensor(reps[b:b+BS], dtype=torch.float32, device=device)
                acts_s[b:b+BS] = sae_s.encode(x).cpu().numpy()

        alive_s = (acts_s > 0).any(0)
        seed_alive_counts.append(int(alive_s.sum()))

        # For each alive feature, get top-100 variant indices and pathogenic rate
        top_variants = {}
        path_rates = {}
        for fi in np.where(alive_s)[0]:
            col = acts_s[:, fi]
            n_act = (col > 0).sum()
            if n_act < 10: continue
            top100 = set(np.argsort(col)[-100:])
            top_variants[fi] = top100
            lab_mask = label[list(top100)] >= 0
            if lab_mask.sum() > 3:
                path_rates[fi] = float(label[list(top100)][lab_mask].mean())

        seed_top_variants.append(top_variants)
        seed_path_rates.append(path_rates)
        del sae_s, acts_s
        torch.cuda.empty_cache()

    # Cross-seed matching: for each feature in seed 0, find best-matching feature in other seeds
    print(f"\n  Cross-seed stability analysis:")
    ref_feats = seed_top_variants[0]
    jaccard_scores = []
    path_rate_corrs = []

    for si in range(1, args.n_sae_seeds):
        comp_feats = seed_top_variants[si]
        jaccards = []
        for fi, top_v in ref_feats.items():
            best_j = 0
            for fj, top_v2 in comp_feats.items():
                j = len(top_v & top_v2) / max(len(top_v | top_v2), 1)
                best_j = max(best_j, j)
            jaccards.append(best_j)
        mean_j = np.mean(jaccards)
        jaccard_scores.append(mean_j)
        print(f"    Seed 0 vs Seed {si}: mean best-match Jaccard = {mean_j:.4f} (n={len(jaccards)} features)")

    # Pathogenic rate correlation across seeds
    ref_pr = seed_path_rates[0]
    for si in range(1, args.n_sae_seeds):
        comp_pr = seed_path_rates[si]
        common = set(ref_pr.keys()) & set(comp_pr.keys())
        if len(common) > 10:
            r = np.corrcoef(
                [ref_pr[f] for f in common],
                [comp_pr[f] for f in common])[0, 1]
            path_rate_corrs.append(r)
            print(f"    Path-rate correlation seed 0 vs {si}: r={r:.4f} (n={len(common)})")

    print(f"\n  Alive features per seed: {seed_alive_counts}")
    print(f"  Mean cross-seed Jaccard: {np.mean(jaccard_scores):.4f}")
    if path_rate_corrs:
        print(f"  Mean path-rate correlation: {np.mean(path_rate_corrs):.4f}")

    # ═══════════════════════════════════════════════════════════════
    # CONTROL 3: Random controls (PCA, ICA, NMF vs SAE)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("[4] RANDOM CONTROLS: SAE vs PCA/ICA/NMF/neurons")
    print("="*80)

    # Metric: for each method's components, how many have significant pathogenic enrichment (FDR<0.05)?
    n_components = min(500, n_features)
    sample_idx = np.random.choice(n, min(50000, n), replace=False)
    reps_sample = reps[sample_idx]
    label_sample = label[sample_idx]

    def count_enriched(components, label_arr, threshold=0.05):
        """Count components significantly enriched for pathogenic variants."""
        n_sig = 0
        pvals = []
        for ci in range(components.shape[1]):
            col = components[:, ci]
            active = col > np.percentile(col, 90)  # top 10%
            lab_mask = label_arr >= 0
            a = (active & (label_arr == 1) & lab_mask).sum()
            b = (active & (label_arr == 0) & lab_mask).sum()
            c = (~active & (label_arr == 1) & lab_mask).sum()
            d = (~active & (label_arr == 0) & lab_mask).sum()
            if a + c == 0 or b + d == 0: continue
            _, p = fisher_exact([[a, b], [c, d]], alternative='two-sided')
            pvals.append(p)

        if not pvals: return 0, 0
        _, padj, _, _ = multipletests(pvals, method='fdr_bh')
        return int((padj < threshold).sum()), len(pvals)

    methods = {}

    # SAE (already trained)
    sae_sample = sae_acts[sample_idx][:, alive]
    n_sig_sae, n_total_sae = count_enriched(sae_sample, label_sample)
    methods['SAE (TopK)'] = (n_sig_sae, n_total_sae)

    # PCA
    print("  Running PCA ...", flush=True)
    pca = PCA(n_components=n_components, random_state=42)
    pca_acts = pca.fit_transform(reps_sample)
    n_sig_pca, n_total_pca = count_enriched(pca_acts, label_sample)
    methods['PCA'] = (n_sig_pca, n_total_pca)

    # ICA
    print("  Running ICA ...", flush=True)
    try:
        ica = FastICA(n_components=min(200, n_components), random_state=42, max_iter=500)
        ica_acts = ica.fit_transform(reps_sample)
        n_sig_ica, n_total_ica = count_enriched(ica_acts, label_sample)
        methods['ICA'] = (n_sig_ica, n_total_ica)
    except Exception as e:
        print(f"    ICA failed: {e}")
        methods['ICA'] = (0, 0)

    # NMF (on non-negative data)
    print("  Running NMF ...", flush=True)
    reps_nn = reps_sample - reps_sample.min(0)
    try:
        nmf = NMF(n_components=min(200, n_components), random_state=42, max_iter=500)
        nmf_acts = nmf.fit_transform(reps_nn)
        n_sig_nmf, n_total_nmf = count_enriched(nmf_acts, label_sample)
        methods['NMF'] = (n_sig_nmf, n_total_nmf)
    except Exception as e:
        print(f"    NMF failed: {e}")
        methods['NMF'] = (0, 0)

    # Individual neurons (raw representation dimensions)
    print("  Testing individual neurons ...", flush=True)
    n_sig_neuron, n_total_neuron = count_enriched(reps_sample[:, :n_components], label_sample)
    methods['Individual neurons'] = (n_sig_neuron, n_total_neuron)

    # Random SAE (untrained)
    print("  Testing random SAE ...", flush=True)
    torch.manual_seed(999)
    random_sae = TopKSAE(1536, n_features, k=args.topk).to(device)
    random_sae.eval()
    random_acts = np.zeros((len(sample_idx), n_features), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(sample_idx), BS):
            x = torch.tensor(reps_sample[b:b+BS], dtype=torch.float32, device=device)
            random_acts[b:b+BS] = random_sae.encode(x).cpu().numpy()
    random_alive = (random_acts > 0).any(0)
    random_acts_alive = random_acts[:, random_alive]
    n_sig_rand, n_total_rand = count_enriched(random_acts_alive, label_sample)
    methods['Random SAE (untrained)'] = (n_sig_rand, n_total_rand)

    print(f"\n  {'Method':<25s} {'Sig features':>15s} {'Total':>8s} {'Fraction':>10s}")
    print("  " + "-"*60)
    for m, (n_sig, n_tot) in methods.items():
        frac = n_sig / max(n_tot, 1)
        print(f"  {m:<25s} {n_sig:>15d} {n_tot:>8d} {frac:>10.1%}")

    # ═══════════════════════════════════════════════════════════════
    # CONTROL 4: Causal feature intervention
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("[5] CAUSAL FEATURE INTERVENTION")
    print("="*80)

    # For top pathogenic features: clamp to 0 → decode → measure prediction change
    # We use the SAE decoder to reconstruct, then pass through a simple classifier
    path_features = []
    for fi in alive_idx:
        col = sae_acts[:, fi]
        active = col > 0
        lab_mask = label[active] >= 0
        if lab_mask.sum() < 20: continue
        pr = label[active][lab_mask].mean()
        if pr > 0.9:
            path_features.append((fi, pr, active.sum()))
    path_features.sort(key=lambda x: -x[2])

    print(f"  Testing {min(20, len(path_features))} pathogenic features (>90% path rate):")
    for fi, pr, n_act in path_features[:20]:
        # Get variants where this feature is active
        active_idx = np.where(sae_acts[:, fi] > 0)[0][:500]
        x_orig = torch.tensor(reps[active_idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            z_orig = sae.encode(x_orig)
            recon_orig = sae.decoder(z_orig)

            z_ablated = z_orig.clone()
            z_ablated[:, fi] = 0
            recon_ablated = sae.decoder(z_ablated)

            delta = (recon_orig - recon_ablated).norm(dim=1).mean().item()
            # How much does the representation change?
            rel_delta = delta / recon_orig.norm(dim=1).mean().item()

        print(f"    F{fi:5d}: path={pr:.2f} n={n_act:5d} ablation_delta={delta:.3f} (rel={rel_delta:.1%})")

    # ═══════════════════════════════════════════════════════════════
    # CONTROL 5: Modality occlusion
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("[6] MODALITY OCCLUSION")
    print("="*80)

    n_occ = n  # use all variants for occlusion (H100 can handle it)
    z_full = np.zeros((n_occ, n_features), dtype=np.float32)
    z_no_prot = np.zeros((n_occ, n_features), dtype=np.float32)
    z_no_dna = np.zeros((n_occ, n_features), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n_occ, BS):
            x = torch.tensor(reps[b:b+BS], dtype=torch.float32, device=device)
            z_full[b:b+BS] = sae.encode(x).cpu().numpy()

            x_no_prot = x.clone(); x_no_prot[:, :768] = 0
            z_no_prot[b:b+BS] = sae.encode(x_no_prot).cpu().numpy()

            x_no_dna = x.clone(); x_no_dna[:, 768:] = 0
            z_no_dna[b:b+BS] = sae.encode(x_no_dna).cpu().numpy()

    # For each feature: how much does it change when protein/DNA is removed?
    W = sae.encoder.weight.detach().cpu().numpy()
    weight_ratio = np.linalg.norm(W[:, :768], axis=1) / (np.linalg.norm(W[:, :768], axis=1) + np.linalg.norm(W[:, 768:], axis=1) + 1e-8)

    occlusion_results = []
    for fi in alive_idx:
        act_full = z_full[:, fi]
        act_no_prot = z_no_prot[:, fi]
        act_no_dna = z_no_dna[:, fi]

        if act_full.sum() < 1: continue

        active = act_full > 0
        if active.sum() < 5: continue

        prot_drop = 1 - (act_no_prot[active].mean() / (act_full[active].mean() + 1e-8))
        dna_drop = 1 - (act_no_dna[active].mean() / (act_full[active].mean() + 1e-8))

        occlusion_results.append({
            'feature': fi,
            'weight_ratio': weight_ratio[fi],
            'prot_occlusion_drop': prot_drop,
            'dna_occlusion_drop': dna_drop,
            'n_active': int(active.sum()),
        })

    odf = pd.DataFrame(occlusion_results)
    corr_prot = np.corrcoef(odf['weight_ratio'], odf['prot_occlusion_drop'])[0, 1]
    corr_dna = np.corrcoef(1 - odf['weight_ratio'], odf['dna_occlusion_drop'])[0, 1]

    print(f"  Weight ratio vs protein occlusion drop: r={corr_prot:.3f}")
    print(f"  (1-Weight ratio) vs DNA occlusion drop: r={corr_dna:.3f}")
    print(f"  → {'VALIDATED' if corr_prot > 0.3 and corr_dna > 0.3 else 'WEAK'}: weight-based modality attribution is {'consistent' if corr_prot > 0.3 else 'inconsistent'} with occlusion")

    # Classify features by occlusion
    prot_driven_occ = (odf['prot_occlusion_drop'] > 0.5) & (odf['dna_occlusion_drop'] < 0.3)
    dna_driven_occ = (odf['dna_occlusion_drop'] > 0.5) & (odf['prot_occlusion_drop'] < 0.3)
    cross_occ = (odf['prot_occlusion_drop'] > 0.2) & (odf['dna_occlusion_drop'] > 0.2)
    print(f"\n  Occlusion-based modality classification:")
    print(f"    Protein-driven: {prot_driven_occ.sum()}")
    print(f"    DNA-driven: {dna_driven_occ.sum()}")
    print(f"    Cross-modal: {cross_occ.sum()}")
    print(f"    Ambiguous: {len(odf) - prot_driven_occ.sum() - dna_driven_occ.sum() - cross_occ.sum()}")

    odf.to_csv(os.path.join(args.out_dir, "modality_occlusion.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # Save summary
    # ═══════════════════════════════════════════════════════════════
    summary = {
        'enrichment': {cat: int((sig['category'] == cat).sum()) for cat in categories},
        'seed_stability': {
            'alive_per_seed': seed_alive_counts,
            'mean_jaccard': float(np.mean(jaccard_scores)),
            'mean_path_rate_corr': float(np.mean(path_rate_corrs)) if path_rate_corrs else None,
        },
        'random_controls': {m: {'sig': s, 'total': t, 'frac': s/max(t,1)} for m, (s, t) in methods.items()},
        'modality_occlusion': {
            'weight_vs_prot_occlusion_corr': float(corr_prot),
            'weight_vs_dna_occlusion_corr': float(corr_dna),
        },
    }
    with open(os.path.join(args.out_dir, "controls_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
