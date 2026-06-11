"""
TopK sweep: re-encode with different sparsity levels and benchmark.
Current SAE was trained with TopK=32. Test k=16,32,64,128,256,512,all.
More features = more information = better prediction, but less sparse/interpretable.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def parse_mutant(mutant_str):
    mutant_str = str(mutant_str).strip()
    if ':' in mutant_str:
        mutant_str = mutant_str.split(':')[0]
    mutant_str = mutant_str.replace('p.', '')
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutant_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


HYDROPHOBIC = set('AVILMFWP')
CHARGE_MAP = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

BLOSUM62_FLAT = {}
_b = "A4R-1N-2D-2C0Q-1E-1G0H-2I-1K-1L-1M-1F-2P-1S1T0W-3Y-2V0"


def compute_human_features(ref_aa, alt_aa):
    feats = np.zeros(9)
    rc = CHARGE_MAP.get(ref_aa, 0)
    ac = CHARGE_MAP.get(alt_aa, 0)
    feats[0] = ac - rc  # charge_change
    feats[1] = AA_VOLUME.get(alt_aa, 140) - AA_VOLUME.get(ref_aa, 140)  # volume_change
    feats[2] = AA_HYDRO.get(alt_aa, 0) - AA_HYDRO.get(ref_aa, 0)  # hydro_change
    feats[3] = float(ref_aa == 'C')  # is_cys_loss
    feats[4] = float(alt_aa == 'P')  # is_pro_intro
    feats[5] = float(ref_aa == 'P')  # is_pro_loss
    feats[6] = float(ref_aa == 'G')  # is_gly_loss
    feats[7] = float(ref_aa == 'W')  # is_trp_loss
    feats[8] = float(rc * ac < 0)    # is_charge_reversal
    return feats


def topk_activation(z, k):
    """Apply TopK sparsity: keep top-k values, zero the rest."""
    if k >= z.shape[1]:
        return z.copy()
    result = np.zeros_like(z)
    for i in range(z.shape[0]):
        idx = np.argpartition(z[i], -k)[-k:]
        result[i, idx] = z[i, idx]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="results/dms_v6_mlp")
    ap.add_argument("--sae_model_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/sae_topk_sweep")
    ap.add_argument("--n_features", type=int, default=12288)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    # ── Load SAE model ──
    print("Loading SAE model ...", flush=True)
    ckpt = torch.load(os.path.join(args.sae_model_dir, "sae_model.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("sae_state", {}))
    enc_w = sd["encoder.weight"].numpy()  # (n_features, 1536)
    enc_b = sd["encoder.bias"].numpy()    # (n_features,)
    print("Encoder: %s, bias: %s" % (enc_w.shape, enc_b.shape), flush=True)

    # ── Load data: use prot_reps + dna_reps to re-encode ──
    print("Loading representations ...", flush=True)
    sae_files = sorted(glob.glob(os.path.join(args.sae_dir, "*_sae.npz")))

    assay_data = []
    for sf in sae_files:
        assay_name = os.path.basename(sf).replace("_sae.npz", "")
        d = np.load(sf, allow_pickle=True)
        fitness = d["y_score"]
        mutants = d["mutants"] if "mutants" in d else np.array([])
        prot_reps = d["prot_reps"] if "prot_reps" in d else None
        dna_reps = d["dna_reps"] if "dna_reps" in d else None
        if prot_reps is None or dna_reps is None:
            continue
        if len(mutants) == 0 or len(mutants) != len(fitness):
            continue
        assay_data.append({
            'name': assay_name,
            'prot_reps': prot_reps,
            'dna_reps': dna_reps,
            'fitness': fitness,
            'mutants': mutants,
        })

    print("Loaded %d assays with representations" % len(assay_data), flush=True)

    # ── Re-encode with SAE and apply different TopK ──
    print("Re-encoding with SAE encoder ...", flush=True)
    for ad in assay_data:
        x = np.concatenate([ad['prot_reps'], ad['dna_reps']], axis=1)  # (n, 1536)
        z = np.maximum(0, x @ enc_w.T + enc_b)  # ReLU(Wx + b), shape (n, 12288)
        ad['z_full'] = z

        # Compute human features
        n = len(ad['mutants'])
        hf = np.zeros((n, 9))
        for i in range(n):
            ref_aa, pos, alt_aa = parse_mutant(ad['mutants'][i])
            if ref_aa and alt_aa and ref_aa in AA_TO_IDX and alt_aa in AA_TO_IDX:
                hf[i] = compute_human_features(ref_aa, alt_aa)
        ad['human_feats'] = hf

    # ── TopK sweep benchmark ──
    k_values = [16, 32, 64, 128, 256, 512, 12288]
    k_labels = ['k=16', 'k=32', 'k=64', 'k=128', 'k=256', 'k=512', 'k=all']

    print("\n" + "=" * 70, flush=True)
    print("TOPK SWEEP: Per-assay 5-fold CV Ridge", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    for ad_idx, ad in enumerate(assay_data):
        y = ad['fitness']
        if len(y) < 50:
            continue

        result = {'assay': ad['name'], 'n': len(y)}
        z_full = ad['z_full']
        hf = ad['human_feats']

        for k, k_label in zip(k_values, k_labels):
            z_k = topk_activation(z_full, k)

            # SAE only
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rhos_sae = []
            rhos_hybrid = []
            for train_idx, test_idx in kf.split(z_k):
                # SAE only
                model = Ridge(alpha=1.0)
                model.fit(z_k[train_idx], y[train_idx])
                yp = model.predict(z_k[test_idx])
                if np.std(yp) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                    r = stats.spearmanr(yp, y[test_idx]).statistic
                    if not np.isnan(r):
                        rhos_sae.append(r)

                # Hybrid (SAE + human)
                X_hyb = np.hstack([z_k[train_idx], hf[train_idx]])
                model2 = Ridge(alpha=1.0)
                model2.fit(X_hyb, y[train_idx])
                yp2 = model2.predict(np.hstack([z_k[test_idx], hf[test_idx]]))
                if np.std(yp2) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                    r2 = stats.spearmanr(yp2, y[test_idx]).statistic
                    if not np.isnan(r2):
                        rhos_hybrid.append(r2)

            result['sae_' + k_label] = np.mean(rhos_sae) if rhos_sae else 0
            result['hybrid_' + k_label] = np.mean(rhos_hybrid) if rhos_hybrid else 0

        # Also: raw concat baseline (prot_reps + dna_reps, 1536-dim, no SAE)
        x_raw = np.concatenate([ad['prot_reps'], ad['dna_reps']], axis=1)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos_raw = []
        rhos_raw_hybrid = []
        for train_idx, test_idx in kf.split(x_raw):
            model = Ridge(alpha=1.0)
            model.fit(x_raw[train_idx], y[train_idx])
            yp = model.predict(x_raw[test_idx])
            if np.std(yp) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                r = stats.spearmanr(yp, y[test_idx]).statistic
                if not np.isnan(r):
                    rhos_raw.append(r)

            X_hyb = np.hstack([x_raw[train_idx], hf[train_idx]])
            model2 = Ridge(alpha=1.0)
            model2.fit(X_hyb, y[train_idx])
            yp2 = model2.predict(np.hstack([x_raw[test_idx], hf[test_idx]]))
            if np.std(yp2) > 1e-8 and np.std(y[test_idx]) > 1e-8:
                r2 = stats.spearmanr(yp2, y[test_idx]).statistic
                if not np.isnan(r2):
                    rhos_raw_hybrid.append(r2)

        result['raw_concat'] = np.mean(rhos_raw) if rhos_raw else 0
        result['raw_hybrid'] = np.mean(rhos_raw_hybrid) if rhos_raw_hybrid else 0

        all_results.append(result)
        if (len(all_results)) % 50 == 0:
            print("  Processed %d assays ..." % len(all_results), flush=True)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(args.out_dir, "topk_sweep_results.csv"), index=False)

    # ── Summary ──
    print("\n" + "=" * 70, flush=True)
    print("TOPK SWEEP SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print("%-25s %8s %8s   %8s %8s" % ("Config", "SAE Mean", "SAE Med", "Hyb Mean", "Hyb Med"),
          flush=True)
    print("-" * 70, flush=True)

    for k, k_label in zip(k_values, k_labels):
        sae_col = 'sae_' + k_label
        hyb_col = 'hybrid_' + k_label
        print("%-25s %8.4f %8.4f   %8.4f %8.4f" % (
            k_label,
            res_df[sae_col].mean(), res_df[sae_col].median(),
            res_df[hyb_col].mean(), res_df[hyb_col].median()), flush=True)

    print("%-25s %8.4f %8.4f   %8.4f %8.4f" % (
        "raw_concat (no SAE)",
        res_df['raw_concat'].mean(), res_df['raw_concat'].median(),
        res_df['raw_hybrid'].mean(), res_df['raw_hybrid'].median()), flush=True)

    # Best interpretable config
    best_k = None
    best_rho = 0
    for k, k_label in zip(k_values, k_labels):
        r = res_df['hybrid_' + k_label].mean()
        if r > best_rho:
            best_rho = r
            best_k = k_label
    print("\nBest interpretable config: %s (hybrid mean=%.4f)" % (best_k, best_rho), flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
