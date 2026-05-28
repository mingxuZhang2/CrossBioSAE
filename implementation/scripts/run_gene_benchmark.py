"""
Run Zhong 2025 Gene Embedding Benchmark on our embeddings.

Runs gene-level (GO, OMIM) and gene-pair (NG, SL, TF) benchmarks
on all embedding variants created by prepare_benchmark_embeddings.py.
"""

import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

C_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]


def precision_at_k(y_true, y_scores, k=10):
    if len(y_scores) > k:
        idx = np.argsort(y_scores)[-k:][::-1]
    else:
        idx = np.argsort(y_scores)[::-1]
    return np.mean(np.array(y_true)[idx])


def load_embedding(subfolder: str):
    csvs = glob.glob(os.path.join(subfolder, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {subfolder}")
    emb = pd.read_csv(csvs[0], header=None).values

    txts = glob.glob(os.path.join(subfolder, "*.txt"))
    if not txts:
        raise FileNotFoundError(f"No TXT in {subfolder}")
    with open(txts[0]) as f:
        genes = [l.strip() for l in f if l.strip()]

    return emb, genes


def get_xy(df, emb_df):
    genes = df["gene"].values
    present = emb_df.index.intersection(genes)
    filtered = df[df["gene"].isin(present)]
    X = emb_df.loc[filtered["gene"].values]
    y = filtered["result"].values
    return X, y


def run_gene_level(subfolder, split_dir, task_name, out_dir):
    """Run gene-level benchmark (GO or OMIM)."""
    emb, genes = load_embedding(subfolder)
    name = os.path.basename(subfolder)
    embedding_df = pd.DataFrame(emb, index=genes)

    fold_files = sorted(glob.glob(os.path.join(split_dir, "*_cv_fold*_dict_all.pkl")))
    holdout_file = glob.glob(os.path.join(split_dir, "*_holdout_dict_all.pkl"))[0]

    with open(holdout_file, "rb") as f:
        holdout_dict = pickle.load(f)

    fold_dicts = []
    for ff in fold_files:
        with open(ff, "rb") as f:
            fold_dicts.append(pickle.load(f))

    results = []
    for task_term in holdout_dict.keys():
        fold_dfs = [fd[task_term] for fd in fold_dicts]
        holdout_df = holdout_dict[task_term]

        # Find best C via 3-fold CV
        best_auc = -1
        best_C = 1.0
        for C_val in C_VALUES:
            fold_aucs = []
            for i, test_df in enumerate(fold_dfs):
                train_df = pd.concat([fd for j, fd in enumerate(fold_dfs) if j != i], ignore_index=True)
                X_train, y_train = get_xy(train_df, embedding_df)
                X_test, y_test = get_xy(test_df, embedding_df)
                if len(X_train) == 0 or len(X_test) == 0 or len(np.unique(y_train)) < 2:
                    continue
                clf = SVC(C=C_val, class_weight="balanced", probability=False)
                clf.fit(X_train, y_train)
                y_prob = clf.decision_function(X_test)
                if len(np.unique(y_test)) > 1:
                    fold_aucs.append(roc_auc_score(y_test, y_prob))
            if fold_aucs and np.mean(fold_aucs) > best_auc:
                best_auc = np.mean(fold_aucs)
                best_C = C_val

        # Holdout evaluation
        X_train_all, y_train_all = get_xy(
            pd.concat(fold_dfs, ignore_index=True), embedding_df
        )
        X_holdout, y_holdout = get_xy(holdout_df, embedding_df)

        if len(X_train_all) == 0 or len(X_holdout) == 0 or len(np.unique(y_train_all)) < 2:
            continue

        clf = SVC(C=best_C, class_weight="balanced", probability=False)
        clf.fit(X_train_all, y_train_all)
        y_prob = clf.decision_function(X_holdout)

        holdout_auc = roc_auc_score(y_holdout, y_prob) if len(np.unique(y_holdout)) > 1 else np.nan
        holdout_auprc = average_precision_score(y_holdout, y_prob)
        holdout_pr10 = precision_at_k(y_holdout, y_prob, k=10)

        results.append({
            "embedding": name,
            "task": task_term,
            "holdout_AUC": holdout_auc,
            "holdout_AUPRC": holdout_auprc,
            "holdout_PR@10": holdout_pr10,
            "best_C": best_C,
        })
        print(f"  {task_term}: AUC={holdout_auc:.4f}, C={best_C}", flush=True)

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f"{name}_{task_name}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"  Mean AUC: {df['holdout_AUC'].mean():.4f}")
    return df


def run_gene_pair(subfolder, split_pkl, task_name, operation, out_dir):
    """Run gene-pair benchmark."""
    emb, genes = load_embedding(subfolder)
    name = os.path.basename(subfolder)
    g2i = {g: i for i, g in enumerate(genes)}

    with open(split_pkl, "rb") as f:
        cv_data = pickle.load(f)
    pairs, labels, cv_splits = cv_data["pairs"], cv_data["labels"], cv_data["cv_splits"]

    ref_genes = set(genes)
    fp, fl, idx_master = [], [], []
    for i, ((g1, g2), lbl) in enumerate(zip(pairs, labels)):
        if str(g1) in ref_genes and str(g2) in ref_genes:
            fp.append((str(g1), str(g2)))
            fl.append(lbl)
            idx_master.append(i)

    if not fp:
        print(f"  No pairs survive filtering for {name}")
        return None

    if operation == "concat":
        X = np.hstack([emb[[g2i[g1] for g1, _ in fp]], emb[[g2i[g2] for _, g2 in fp]]])
    elif operation == "sum":
        X = np.array([emb[g2i[g1]] + emb[g2i[g2]] for g1, g2 in fp])
    else:
        X = np.array([emb[g2i[g1]] * emb[g2i[g2]] for g1, g2 in fp])
    y = np.array(fl)

    print(f"  {name} {task_name} ({operation}): {len(fp)} pairs, {sum(fl)} positive", flush=True)

    results = []
    pr10_scorer = make_scorer(precision_at_k, needs_threshold=True, k=10)

    for fold, splits in cv_splits.items():
        train_m, outer_m = splits["train_idx"], splits["test_idx"]
        train_idx = [i for i, m in enumerate(idx_master) if m in train_m]
        outer_idx = [i for i, m in enumerate(idx_master) if m in outer_m]

        X_train, y_train = X[train_idx], y[train_idx]
        X_outer, y_outer = X[outer_idx], y[outer_idx]

        train_master = [m for m in idx_master if m in train_m]
        full2train = {m: i for i, m in enumerate(train_master)}
        inner_splits_mapped = []
        for tr_m, val_m in splits["inner_splits"]:
            tr_idx = [full2train[m] for m in tr_m if m in full2train]
            val_idx = [full2train[m] for m in val_m if m in full2train]
            if tr_idx and val_idx:
                inner_splits_mapped.append((tr_idx, val_idx))

        grid = GridSearchCV(
            SVC(class_weight="balanced", probability=False),
            param_grid={"C": C_VALUES},
            scoring="roc_auc",
            refit=True,
            cv=inner_splits_mapped,
            n_jobs=3,
        )
        grid.fit(X_train, y_train)

        outer_scores = grid.decision_function(X_outer)
        outer_auc = roc_auc_score(y_outer, outer_scores)
        outer_auprc = average_precision_score(y_outer, outer_scores)

        results.append({
            "fold": fold,
            "best_C": grid.best_params_["C"],
            "outer_AUC": outer_auc,
            "outer_AUPRC": outer_auprc,
        })

    df = pd.DataFrame(results)
    avg_auc = df["outer_AUC"].mean()
    avg_auprc = df["outer_AUPRC"].mean()
    print(f"  Mean outer AUC: {avg_auc:.4f}, AUPRC: {avg_auprc:.4f}")

    out_path = os.path.join(out_dir, f"{name}_{task_name}_{operation}_results.csv")
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True, help="Dir with embedding subfolders")
    parser.add_argument("--benchmark_dir", required=True, help="gene-embedding-benchmarks repo root")
    parser.add_argument("--output_dir", required=True, help="Results output dir")
    parser.add_argument("--tasks", nargs="+", default=["go", "omim", "ng", "sl", "tf"],
                        help="Which tasks to run")
    parser.add_argument("--embeddings", nargs="+", default=None,
                        help="Which embeddings to test (default: all)")
    parser.add_argument("--pair_op", default="concat", choices=["sum", "product", "concat"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.benchmark_dir, "data", "data_splits")

    # Discover embeddings
    if args.embeddings:
        embed_names = args.embeddings
    else:
        embed_names = sorted([
            d for d in os.listdir(args.embedding_dir)
            if os.path.isdir(os.path.join(args.embedding_dir, d))
        ])

    print(f"Embeddings to test: {embed_names}")
    print(f"Tasks: {args.tasks}")

    all_results = {}

    for emb_name in embed_names:
        subfolder = os.path.join(args.embedding_dir, emb_name)
        print(f"\n{'='*60}")
        print(f"Embedding: {emb_name}")
        print(f"{'='*60}")

        for task in args.tasks:
            t0 = time.time()
            if task == "go":
                split_dir = os.path.join(data_dir, "gene_level_benchmark", "go_folds_splits")
                df = run_gene_level(subfolder, split_dir, "go", args.output_dir)
            elif task == "omim":
                split_dir = os.path.join(data_dir, "gene_level_benchmark", "omim_folds_splits")
                df = run_gene_level(subfolder, split_dir, "omim", args.output_dir)
            elif task in ["ng", "sl", "tf"]:
                split_pkl = os.path.join(data_dir, "gene_pair_benchmark", f"{task}_nested_cv_splits.pkl")
                df = run_gene_pair(subfolder, split_pkl, task, args.pair_op, args.output_dir)
            else:
                print(f"Unknown task: {task}")
                continue

            elapsed = time.time() - t0
            print(f"  {task} done in {elapsed:.1f}s")

            if df is not None:
                key = f"{emb_name}_{task}"
                all_results[key] = df

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_rows = []
    for emb_name in embed_names:
        row = {"embedding": emb_name}
        for task in args.tasks:
            key = f"{emb_name}_{task}"
            if key in all_results:
                df = all_results[key]
                if "holdout_AUC" in df.columns:
                    row[f"{task}_AUC"] = df["holdout_AUC"].mean()
                elif "outer_AUC" in df.columns:
                    row[f"{task}_AUC"] = df["outer_AUC"].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
