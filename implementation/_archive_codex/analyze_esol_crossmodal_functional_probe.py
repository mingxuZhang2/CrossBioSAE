#!/usr/bin/env python
"""Fast eSOL functional-phenotype cross-modal probe with shuffled-DNA control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("results/interpretability_applications")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
        "r2": float(r2_score(y, pred)),
    }


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_random_cv(
    reps: dict[str, np.ndarray],
    y: np.ndarray,
    prot: np.ndarray,
    dna: np.ndarray,
    n_splits: int,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    pred_rows = []
    rng = np.random.default_rng(seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(y)):
        fold_reps = dict(reps)
        train_perm = rng.permutation(train_idx)
        test_perm = rng.permutation(test_idx)
        fold_reps["Concat shuffled-DNA"] = np.vstack(
            [
                np.hstack([prot[train_idx], dna[train_perm]]),
                np.hstack([prot[test_idx], dna[test_perm]]),
            ]
        )
        fold_indices = np.concatenate([train_idx, test_idx])
        fold_train = np.arange(len(train_idx))
        fold_test = np.arange(len(train_idx), len(fold_indices))
        for name, X in fold_reps.items():
            if name == "Concat shuffled-DNA":
                X_fold = X
            else:
                X_fold = X[fold_indices]
            pred = fit_predict(X_fold[fold_train], y[train_idx], X_fold[fold_test], alpha)
            row = {
                "eval": "cv5",
                "fold": fold,
                "rep": name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            row.update(metrics(y[test_idx], pred))
            fold_rows.append(row)
            pred_rows.extend(
                {
                    "eval": "cv5",
                    "fold": fold,
                    "rep": name,
                    "row_index": int(idx),
                    "y_true": float(obs),
                    "y_pred": float(pr),
                }
                for idx, obs, pr in zip(test_idx, y[test_idx], pred)
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def evaluate_official_split(
    reps: dict[str, np.ndarray],
    y: np.ndarray,
    prot: np.ndarray,
    dna: np.ndarray,
    is_train: np.ndarray,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]
    fold_reps = dict(reps)
    fold_reps["Concat shuffled-DNA"] = {
        "train": np.hstack([prot[train_idx], dna[rng.permutation(train_idx)]]),
        "test": np.hstack([prot[test_idx], dna[rng.permutation(test_idx)]]),
    }
    rows = []
    for name, X in fold_reps.items():
        if name == "Concat shuffled-DNA":
            pred = fit_predict(X["train"], y[train_idx], X["test"], alpha)
        else:
            pred = fit_predict(X[train_idx], y[train_idx], X[test_idx], alpha)
        row = {
            "eval": "official_split",
            "fold": 0,
            "rep": name,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(metrics(y[test_idx], pred))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(fold_results: pd.DataFrame, split_results: pd.DataFrame) -> pd.DataFrame:
    cv_summary = (
        fold_results.groupby(["eval", "rep"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
        )
    )
    split_summary = split_results.rename(
        columns={
            "spearman": "spearman_mean",
            "pearson": "pearson_mean",
            "r2": "r2_mean",
        }
    )
    split_summary["n_folds"] = 1
    split_summary["spearman_sd"] = np.nan
    split_summary["pearson_sd"] = np.nan
    split_summary["r2_sd"] = np.nan
    split_summary = split_summary[
        ["eval", "rep", "n_folds", "spearman_mean", "spearman_sd", "pearson_mean", "pearson_sd", "r2_mean", "r2_sd"]
    ]
    summary = pd.concat([cv_summary, split_summary], ignore_index=True)
    rows = []
    for eval_name in summary["eval"].unique():
        sub = summary.loc[summary["eval"].eq(eval_name)].set_index("rep")
        for metric in ["spearman_mean", "pearson_mean", "r2_mean"]:
            if "Concat" in sub.index and "Protein-only (ESM2)" in sub.index:
                rows.append(
                    {
                        "eval": eval_name,
                        "comparison": f"Concat minus Protein-only ({metric})",
                        "delta": sub.loc["Concat", metric] - sub.loc["Protein-only (ESM2)", metric],
                    }
                )
            if "Concat" in sub.index and "Concat shuffled-DNA" in sub.index:
                rows.append(
                    {
                        "eval": eval_name,
                        "comparison": f"Concat minus shuffled-DNA ({metric})",
                        "delta": sub.loc["Concat", metric] - sub.loc["Concat shuffled-DNA", metric],
                    }
                )
    return summary, pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows.\n"
    return df.to_markdown(index=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--matched-csv", type=Path, default=Path("data/solubility/esol_matched.csv"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("results/solubility/esol_emb.npz"))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(repo / args.matched_csv)
    df = df[df["match"].isin(["gene+verified", "protein-seq"])].reset_index(drop=True)
    y = df["solubility"].to_numpy(dtype=float)
    is_train = df["split"].eq("train").to_numpy()
    emb = np.load(repo / args.embedding_cache)
    prot = emb["prot"]
    dna = emb["dna"]
    if prot.shape[0] != len(df) or dna.shape[0] != len(df):
        raise ValueError(f"embedding rows {prot.shape[0]}/{dna.shape[0]} do not match dataframe {len(df)}")

    reps = {
        "Protein-only (ESM2)": prot,
        "DNA-only (NT)": dna,
        "Concat": np.hstack([prot, dna]),
    }
    folds, preds = evaluate_random_cv(reps, y, prot, dna, args.splits, args.alpha, args.seed)
    split = evaluate_official_split(reps, y, prot, dna, is_train, args.alpha, args.seed)
    summary, deltas = summarize(folds, split)

    paths = {
        "summary": out / "esol_crossmodal_functional_probe_summary.csv",
        "folds": out / "esol_crossmodal_functional_probe_folds.csv",
        "split": out / "esol_crossmodal_functional_probe_official_split.csv",
        "predictions": out / "esol_crossmodal_functional_probe_predictions.csv",
        "deltas": out / "esol_crossmodal_functional_probe_deltas.csv",
        "report": out / "esol_crossmodal_functional_probe.md",
    }
    summary.to_csv(paths["summary"], index=False)
    folds.to_csv(paths["folds"], index=False)
    split.to_csv(paths["split"], index=False)
    preds.to_csv(paths["predictions"], index=False)
    deltas.to_csv(paths["deltas"], index=False)

    cv = summary.loc[summary["eval"].eq("cv5")].sort_values("r2_mean", ascending=False)
    official = summary.loc[summary["eval"].eq("official_split")].sort_values("r2_mean", ascending=False)
    report = [
        "# eSOL Cross-Modal Functional Probe",
        "",
        "## Purpose",
        "",
        "This is a fast functional-phenotype audit for the interpretability/application track.",
        "It asks whether a DNA-side language-model embedding adds solubility signal beyond",
        "protein sequence embeddings on the eSOL benchmark. It is not an SAE causal",
        "intervention; it is a functional-assay context check for cross-modal utility.",
        "",
        "## Dataset",
        "",
        f"- Verified protein-CDS rows: {len(df)}.",
        f"- Official split train/test: {int(is_train.sum())}/{int((~is_train).sum())}.",
        f"- Representations: ESM2 protein embedding, NT DNA embedding, concat, and concat with shuffled DNA as a negative control.",
        f"- Probe: StandardScaler + Ridge(alpha={args.alpha}).",
        "",
        "## 5-Fold CV Summary",
        "",
        md_table(cv),
        "",
        "## Official Split Summary",
        "",
        md_table(official),
        "",
        "## Cross-Modal Deltas",
        "",
        md_table(deltas),
        "",
        "## Claim Boundary",
        "",
        "- If concat is not better than protein-only and shuffled-DNA, eSOL should be treated as a negative/control result.",
        "- If concat is better, it supports cross-modal functional phenotype utility, but still does not prove SAE feature interpretability.",
        "- A manuscript-grade eSOL claim would require a locked nonlinear probe, seed repeats, and a mechanistic feature-level analysis.",
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")

    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
