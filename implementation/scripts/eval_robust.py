#!/usr/bin/env python3
"""
Robust evaluation: temporal holdout + gene-heldout + constraint ablation.

Addresses key reviewer concerns from GPT Pro review:
1. Temporal holdout: train on ClinVar ≤2023, test on 2024+
2. Gene-heldout: entire genes excluded from training
3. Constraint ablation: with vs without gnomAD gene constraint
"""

import argparse, glob, gzip, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Import from v6 ───────────────────────────────────────────────────
# Reuse AA features, model, training functions from v6
from train_variant_mlp_v6 import (
    VariantMLPv6, PretrainHead, compute_aa_features,
    pretrain, finetune_fold, cosine_lr,
)


def load_data(args):
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

    llr_all = np.load(args.llr_path)["llr"].astype(np.float32)

    prot_mean, prot_std = prot.mean(0), prot.std(0) + 1e-8
    dna_mean, dna_std = dna.mean(0), dna.std(0) + 1e-8
    prot_z = (prot - prot_mean) / prot_std
    dna_z = (dna - dna_mean) / dna_std
    llr_mean, llr_std = llr_all.mean(), llr_all.std() + 1e-8
    llr_z = (llr_all - llr_mean) / llr_std

    # Labels
    is_path = sub["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = sub["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~sub["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    labeled = (is_path | is_ben).values
    y = np.where(is_path.values, 1.0, 0.0)
    genes = sub["gene"].values

    # AA features
    aa_feats = compute_aa_features(sub["from_aa"].values, sub["to_aa"].values)

    # Gene constraint
    gc_map = {}
    if os.path.exists(args.constraint):
        with gzip.open(args.constraint, "rt") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                vals = line.strip().split("\t")
                rec = dict(zip(header, vals))
                try:
                    gc_map[rec["gene"]] = {
                        "pLI": float(rec["pLI"]) if rec["pLI"] not in ("NA","") else np.nan,
                        "oe_mis": float(rec["oe_mis"]) if rec["oe_mis"] not in ("NA","") else np.nan,
                        "mis_z": float(rec["mis_z"]) if rec["mis_z"] not in ("NA","") else np.nan,
                    }
                except: pass
    pli = np.array([gc_map.get(str(g), {}).get("pLI", np.nan) for g in genes], dtype=np.float32)
    oe_mis = np.array([gc_map.get(str(g), {}).get("oe_mis", np.nan) for g in genes], dtype=np.float32)
    mis_z = np.array([gc_map.get(str(g), {}).get("mis_z", np.nan) for g in genes], dtype=np.float32)
    for arr in [pli, oe_mis, mis_z]:
        arr[np.isnan(arr)] = np.nanmedian(arr)
    gene_feats = np.column_stack([pli, oe_mis, mis_z])

    # Extra features
    prot_norm = np.linalg.norm(prot, axis=1, keepdims=True)
    dna_norm = np.linalg.norm(dna, axis=1, keepdims=True)
    extra_base = np.concatenate([
        prot_norm, dna_norm, llr_all[:, None],
        np.abs(prot).max(axis=1, keepdims=True),
        np.abs(dna).max(axis=1, keepdims=True),
        prot.std(axis=1, keepdims=True),
        dna.std(axis=1, keepdims=True),
        prot_norm / (dna_norm + 1e-8),
        aa_feats,
    ], axis=1).astype(np.float32)

    extra_full = np.concatenate([extra_base, gene_feats], axis=1).astype(np.float32)

    # ClinVar dates
    dates = load_clinvar_dates(args.variant_summary, sub)

    return {
        'prot_z': prot_z, 'dna_z': dna_z, 'llr_z': llr_z,
        'extra_base': extra_base, 'extra_full': extra_full,
        'y': y, 'labeled': labeled, 'genes': genes, 'dates': dates,
        'n': n,
    }


def load_clinvar_dates(path, sub):
    dates = pd.Series([pd.NaT] * len(sub))
    if not os.path.exists(path):
        return dates

    date_map = {}
    with gzip.open(path, 'rt') as f:
        header = f.readline().strip().split('\t')
        date_idx = header.index('LastEvaluated')
        chrom_idx = header.index('Chromosome')
        pos_idx = header.index('PositionVCF')
        ref_idx = header.index('ReferenceAlleleVCF')
        alt_idx = header.index('AlternateAlleleVCF')
        assembly_idx = header.index('Assembly')
        type_idx = header.index('Type')

        for line in f:
            vals = line.strip().split('\t')
            if len(vals) <= max(date_idx, assembly_idx, type_idx):
                continue
            if vals[assembly_idx] != 'GRCh38' or vals[type_idx] != 'single nucleotide variant':
                continue
            key = f"{vals[chrom_idx]}:{vals[pos_idx]}:{vals[ref_idx]}:{vals[alt_idx]}"
            try:
                d = pd.to_datetime(vals[date_idx], format='%b %d, %Y')
                if key not in date_map or d > date_map[key]:
                    date_map[key] = d
            except:
                pass

    sub_keys = sub['chrom'].astype(str) + ':' + sub['pos'].astype(str) + ':' + sub['ref'] + ':' + sub['alt']
    dates = sub_keys.map(date_map)
    return dates


def train_and_eval(prot_z, dna_z, llr_z, extra, y, tr_idx, te_idx,
                   device, n_seeds=5, pretrain_epochs=30, d_extra=None):
    if d_extra is None:
        d_extra = extra.shape[1]

    prot_tr, prot_te = prot_z[tr_idx], prot_z[te_idx]
    dna_tr, dna_te = dna_z[tr_idx], dna_z[te_idx]
    extra_tr, extra_te = extra[tr_idx], extra[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Normalize extra features from train stats
    em, es = extra_tr.mean(0), extra_tr.std(0) + 1e-8
    extra_tr_n = (extra_tr - em) / es
    extra_te_n = (extra_te - em) / es

    all_preds = np.zeros(len(te_idx), dtype=np.float64)

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Pretrain
        model = VariantMLPv6(d_extra=d_extra).to(device)
        pt_head = PretrainHead().to(device)
        pretrain(model.proj_prot, model.proj_dna, pt_head,
                 prot_z, dna_z, llr_z, device, epochs=pretrain_epochs)
        del pt_head
        pt_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "proj_" in k}

        # Fine-tune on train, eval on test
        model_ft = VariantMLPv6(d_extra=d_extra).to(device)
        sd = model_ft.state_dict()
        for k, v in pt_state.items():
            if k in sd: sd[k] = v.clone()
        model_ft.load_state_dict(sd)

        preds, auc = finetune_fold(
            model_ft, prot_tr, dna_tr, extra_tr_n, y_tr,
            prot_te, dna_te, extra_te_n, y_te, device)
        all_preds += preds
        print(f"    seed {seed}: AUC={auc:.4f}", flush=True)

    all_preds /= n_seeds
    auc = roc_auc_score(y_te, all_preds)
    auprc = average_precision_score(y_te, all_preds)
    return auc, auprc, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/sae_pretrain_emb")
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--llr_path", default="results/sae_pretrain_emb/esm2_llr.npz")
    ap.add_argument("--constraint", default="data/full/labels/gnomad_constraint.txt.bgz")
    ap.add_argument("--variant_summary", default="data/variant/variant_summary.txt.gz")
    ap.add_argument("--out_dir", default="results/eval_robust")
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args)
    prot_z, dna_z, llr_z = data['prot_z'], data['dna_z'], data['llr_z']
    y, labeled, genes, dates = data['y'], data['labeled'], data['genes'], data['dates']
    extra_full, extra_base = data['extra_full'], data['extra_base']

    results = {}

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("EVAL 1: TEMPORAL HOLDOUT (train ≤2023, test 2024+)")
    print("="*80)

    cutoff = pd.Timestamp('2024-01-01')
    has_date = dates.notna() & labeled
    tr_temporal = np.where(has_date & (dates < cutoff))[0]
    te_temporal = np.where(has_date & (dates >= cutoff))[0]
    print(f"  Train: {len(tr_temporal)} (P:{(y[tr_temporal]==1).sum()}, B:{(y[tr_temporal]==0).sum()})")
    print(f"  Test:  {len(te_temporal)} (P:{(y[te_temporal]==1).sum()}, B:{(y[te_temporal]==0).sum()})")

    if len(tr_temporal) > 100 and len(te_temporal) > 100:
        print("\n  With gene constraint:")
        auc_t, auprc_t, preds_t = train_and_eval(
            prot_z, dna_z, llr_z, extra_full, y,
            tr_temporal, te_temporal, device, args.n_seeds)
        print(f"  >>> Temporal AUC={auc_t:.4f}, AUPRC={auprc_t:.4f}")
        results['temporal_full'] = {'auc': auc_t, 'auprc': auprc_t,
                                    'n_train': len(tr_temporal), 'n_test': len(te_temporal)}

        print("\n  Without gene constraint (ablation):")
        auc_t2, auprc_t2, _ = train_and_eval(
            prot_z, dna_z, llr_z, extra_base, y,
            tr_temporal, te_temporal, device, args.n_seeds,
            d_extra=extra_base.shape[1])
        print(f"  >>> Temporal (no constraint) AUC={auc_t2:.4f}, AUPRC={auprc_t2:.4f}")
        results['temporal_noconstraint'] = {'auc': auc_t2, 'auprc': auprc_t2}

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("EVAL 2: GENE-HELDOUT (5-fold, entire genes excluded)")
    print("="*80)

    lab_idx = np.where(labeled)[0]
    lab_genes = genes[lab_idx]
    lab_y = y[lab_idx]

    unique_genes = np.unique(lab_genes)
    np.random.seed(42)
    np.random.shuffle(unique_genes)
    n_folds = 5
    gene_folds = np.array_split(unique_genes, n_folds)

    oof_gene = np.zeros(len(lab_idx), dtype=np.float64)
    fold_aucs = []

    for fi in range(n_folds):
        test_genes = set(gene_folds[fi])
        te_mask = np.array([g in test_genes for g in lab_genes])
        tr_mask = ~te_mask
        tr_i = lab_idx[tr_mask]
        te_i = lab_idx[te_mask]
        print(f"\n  Fold {fi}: train={len(tr_i)} ({len(unique_genes)-len(test_genes)} genes), test={len(te_i)} ({len(test_genes)} genes)")

        auc_g, _, preds_g = train_and_eval(
            prot_z, dna_z, llr_z, extra_full, y,
            tr_i, te_i, device, args.n_seeds)
        oof_gene[te_mask] = preds_g
        fold_aucs.append(auc_g)
        print(f"  >>> Gene-heldout fold {fi}: AUC={auc_g:.4f}")

    overall_gene_auc = roc_auc_score(lab_y, oof_gene)
    print(f"\n  >>> Gene-heldout overall: AUC={overall_gene_auc:.4f} (per-fold: {[f'{a:.4f}' for a in fold_aucs]})")
    results['gene_heldout'] = {'overall_auc': overall_gene_auc, 'fold_aucs': fold_aucs}

    # Gene-heldout without constraint
    print("\n  Gene-heldout without gene constraint:")
    oof_gene_nc = np.zeros(len(lab_idx), dtype=np.float64)
    fold_aucs_nc = []
    for fi in range(n_folds):
        test_genes = set(gene_folds[fi])
        te_mask = np.array([g in test_genes for g in lab_genes])
        tr_mask = ~te_mask
        tr_i = lab_idx[tr_mask]
        te_i = lab_idx[te_mask]

        auc_g, _, preds_g = train_and_eval(
            prot_z, dna_z, llr_z, extra_base, y,
            tr_i, te_i, device, args.n_seeds,
            d_extra=extra_base.shape[1])
        oof_gene_nc[te_mask] = preds_g
        fold_aucs_nc.append(auc_g)
        print(f"    Fold {fi} (no constraint): AUC={auc_g:.4f}")

    overall_nc = roc_auc_score(lab_y, oof_gene_nc)
    print(f"  >>> Gene-heldout (no constraint) overall: AUC={overall_nc:.4f}")
    results['gene_heldout_noconstraint'] = {'overall_auc': overall_nc, 'fold_aucs': fold_aucs_nc}

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Setting':<45s} {'AUC':>8s} {'AUPRC':>8s}")
    print("-"*65)
    for k, v in results.items():
        auc = v.get('overall_auc', v.get('auc', 0))
        auprc = v.get('auprc', '-')
        auprc_str = f"{auprc:.4f}" if isinstance(auprc, float) else auprc
        print(f"  {k:<43s} {auc:>8.4f} {auprc_str:>8s}")
    print(f"  {'AlphaMissense (reference)':<43s} {'0.9638':>8s} {'—':>8s}")
    print(f"  {'Random 5-fold CV (v6, for reference)':<43s} {'0.9629':>8s} {'—':>8s}")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
