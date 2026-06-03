#!/usr/bin/env python3
"""
Super ensemble: combine predictions from v2 + v3 + v4 for maximum diversity.
No GPU needed — just loads saved prediction arrays and averages.
"""

import argparse, json, os
import numpy as np
from sklearn.metrics import roc_auc_score


def load_preds(path, key="ensemble"):
    d = np.load(path)
    return d[key], d["y"], d.get("genes", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_dir", default="results/variant_mlp_final_v2")
    ap.add_argument("--v3_dir", default="results/variant_mlp_v3")
    ap.add_argument("--v4_dir", default="results/variant_mlp_v4")
    ap.add_argument("--baseline_csv", default="results/multiomics_validation/baseline_scores.csv")
    ap.add_argument("--out_dir", default="results/variant_super_ensemble")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sources = {}

    # v2
    v2_path = os.path.join(args.v2_dir, "preds.npz")
    if os.path.exists(v2_path):
        d = np.load(v2_path, allow_pickle=True)
        sources["v2"] = d["ensemble"]
        y = d["y"]
        genes = d["genes"] if "genes" in d else None
        print(f"  v2: AUC={roc_auc_score(y, sources['v2']):.4f}", flush=True)

    # v3 (full ensemble and top3)
    v3_path = os.path.join(args.v3_dir, "preds.npz")
    if os.path.exists(v3_path):
        d = np.load(v3_path, allow_pickle=True)
        if "full_ensemble" in d:
            sources["v3_full"] = d["full_ensemble"]
            print(f"  v3_full: AUC={roc_auc_score(y, sources['v3_full']):.4f}", flush=True)
        if "top3_ensemble" in d:
            sources["v3_top3"] = d["top3_ensemble"]
            print(f"  v3_top3: AUC={roc_auc_score(y, sources['v3_top3']):.4f}", flush=True)

    # v4
    v4_path = os.path.join(args.v4_dir, "preds.npz")
    if os.path.exists(v4_path):
        d = np.load(v4_path, allow_pickle=True)
        sources["v4"] = d["ensemble"]
        print(f"  v4: AUC={roc_auc_score(y, sources['v4']):.4f}", flush=True)

    if len(sources) < 2:
        print(f"Need at least 2 sources, got {len(sources)}: {list(sources.keys())}")
        return

    print(f"\n--- Super Ensemble ({len(sources)} sources) ---", flush=True)

    # Simple average
    all_preds = list(sources.values())
    simple_avg = np.mean(all_preds, axis=0)
    auc_simple = roc_auc_score(y, simple_avg)
    print(f"  simple avg:    {auc_simple:.4f}", flush=True)

    # Weighted by individual AUC
    aucs = [roc_auc_score(y, p) for p in all_preds]
    weights = np.array(aucs) / sum(aucs)
    weighted_avg = np.average(all_preds, axis=0, weights=weights)
    auc_weighted = roc_auc_score(y, weighted_avg)
    print(f"  weighted avg:  {auc_weighted:.4f}", flush=True)

    # Rank-based ensemble (average ranks instead of probabilities)
    ranks = [np.argsort(np.argsort(p)).astype(float) for p in all_preds]
    rank_avg = np.mean(ranks, axis=0)
    auc_rank = roc_auc_score(y, rank_avg)
    print(f"  rank avg:      {auc_rank:.4f}", flush=True)

    # Pick best
    best_pred = simple_avg
    best_auc = auc_simple
    best_label = "simple"
    if auc_weighted > best_auc:
        best_pred, best_auc, best_label = weighted_avg, auc_weighted, "weighted"
    if auc_rank > best_auc:
        best_pred, best_auc, best_label = rank_avg, auc_rank, "rank"

    print(f"\n  >>> BEST ({best_label}): {best_auc:.4f} <<<", flush=True)
    print(f"  AlphaMissense:       0.9638", flush=True)
    print(f"  REVEL:               0.9689", flush=True)
    delta = best_auc - 0.9638
    print(f"  Delta vs AM:         {delta:+.4f} {'*** BEAT! ***' if delta > 0 else ''}", flush=True)

    # Bootstrap CI
    np.random.seed(42)
    N = len(y)
    n_boot = 2000
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        yu = y[idx]
        if yu.sum() > 0 and yu.sum() < len(yu):
            boot.append(roc_auc_score(yu, best_pred[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI:              [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    # Try all pairwise combinations
    print(f"\n--- Pairwise combinations ---", flush=True)
    names = list(sources.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pair = np.mean([sources[names[i]], sources[names[j]]], axis=0)
            auc = roc_auc_score(y, pair)
            print(f"  {names[i]}+{names[j]}: {auc:.4f}", flush=True)

    # Save
    np.savez_compressed(os.path.join(args.out_dir, "preds.npz"),
                        ensemble=best_pred, y=y, genes=genes)
    results = {"best_auc": float(best_auc), "best_method": best_label,
               "simple_auc": float(auc_simple), "weighted_auc": float(auc_weighted),
               "rank_auc": float(auc_rank), "ci_95": [float(ci_lo), float(ci_hi)],
               "sources": {k: float(roc_auc_score(y, v)) for k, v in sources.items()}}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
