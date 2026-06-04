#!/usr/bin/env python3
"""
Stacking ensemble: train a meta-learner on OOF predictions from multiple models.

Uses logistic regression / ridge on per-model OOF predictions to find optimal
combination weights. Evaluates via nested CV (outer 5-fold, inner fit).
"""

import argparse, json, os
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_preds(dirs):
    preds = {}
    y_ref = None
    for d in dirs:
        name = os.path.basename(d)
        f = os.path.join(d, "preds.npz")
        if not os.path.exists(f):
            print(f"  SKIP {name}: no preds.npz")
            continue
        data = np.load(f, allow_pickle=True)
        p = data["ensemble"].astype(np.float64)
        y = data["y"].astype(np.float64)
        if y_ref is None:
            y_ref = y
        else:
            assert np.array_equal(y, y_ref), f"{name} has different labels!"
        auc = roc_auc_score(y, p)
        preds[name] = p
        print(f"  {name}: AUC={auc:.6f} (n={len(p)})")
    return preds, y_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dirs", nargs="+", default=[
        "results/variant_mlp_final_v2",
        "results/variant_mlp_v4",
        "results/variant_mlp_v5",
        "results/variant_mlp_v5b",
        "results/variant_mlp_v6",
    ])
    ap.add_argument("--out_dir", default="results/stack_ensemble")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading predictions ...", flush=True)
    preds, y = load_preds(args.result_dirs)
    if not preds:
        print("No predictions found!"); return

    names = sorted(preds.keys())
    X = np.column_stack([preds[n] for n in names])
    n, d = X.shape
    print(f"\nStacking {d} models on {n} samples\n", flush=True)

    # ── Simple combos ──
    print("=== Simple combinations ===")
    avg = X.mean(axis=1)
    print(f"  Mean avg: {roc_auc_score(y, avg):.6f}")

    ranks = np.column_stack([rankdata(X[:, i]) for i in range(d)])
    rank_avg = ranks.mean(axis=1)
    print(f"  Rank avg: {roc_auc_score(y, rank_avg):.6f}")

    # ── Stacking with nested CV ──
    print("\n=== Stacking (nested 5-fold CV) ===")

    for method_name, make_model in [
        ("LogReg C=1", lambda: LogisticRegression(C=1.0, max_iter=1000)),
        ("LogReg C=10", lambda: LogisticRegression(C=10.0, max_iter=1000)),
        ("LogReg C=0.1", lambda: LogisticRegression(C=0.1, max_iter=1000)),
    ]:
        oof = np.zeros(n)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            m = make_model()
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  {method_name}: {auc:.6f}")

    # ── Stacking with rank features ──
    print("\n=== Stacking on ranks ===")
    for method_name, make_model in [
        ("LogReg C=1 (ranks)", lambda: LogisticRegression(C=1.0, max_iter=1000)),
        ("LogReg C=10 (ranks)", lambda: LogisticRegression(C=10.0, max_iter=1000)),
    ]:
        oof = np.zeros(n)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(ranks, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(ranks[tr])
            Xte = scaler.transform(ranks[te])
            m = make_model()
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  {method_name}: {auc:.6f}")

    # ── Stacking with both raw + rank ──
    print("\n=== Stacking on raw + ranks ===")
    X_both = np.column_stack([X, ranks])
    oof = np.zeros(n)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X_both, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_both[tr])
        Xte = scaler.transform(X_both[te])
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  LogReg C=1 (raw+ranks): {auc:.6f}")

    # ── Best result with bootstrap CI ──
    print("\n=== Bootstrap CI for best ===")
    best_name = "rank_avg"
    best_preds = rank_avg
    best_auc = roc_auc_score(y, best_preds)

    cis = []
    for _ in range(2000):
        idx = np.random.choice(n, n, replace=True)
        cis.append(roc_auc_score(y[idx], best_preds[idx]))
    ci = np.percentile(cis, [2.5, 97.5])
    print(f"  {best_name}: {best_auc:.6f} 95%CI [{ci[0]:.6f}, {ci[1]:.6f}]")

    print(f"\n  >>> AlphaMissense: 0.963800")
    print(f"  >>> REVEL:         0.968900")

    # Save
    results = {
        "models": names,
        "mean_avg_auc": float(roc_auc_score(y, avg)),
        "rank_avg_auc": float(roc_auc_score(y, rank_avg)),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
