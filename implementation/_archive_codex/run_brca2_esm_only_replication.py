#!/usr/bin/env python3
"""BRCA2 SGE protein-side replication using ESM deltas only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument(
        "--variants",
        type=Path,
        default=root / "data" / "variant" / "brca2" / "brca2_variants.csv",
    )
    p.add_argument(
        "--esm",
        type=Path,
        default=root / "results" / "variant" / "brca2_esm_delta.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    p.add_argument("--output-prefix", default="brca2_esm_only_replication")
    p.add_argument("--group-col", default="pos")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--permutation-sets",
        type=int,
        default=20,
        help="Number of label-permutation null ESM-only CV repeats for all and missense-only scopes.",
    )
    p.add_argument("--seed", type=int, default=630)
    return p.parse_args()


def safe_auc(y: np.ndarray, s: np.ndarray, larger_is_damaging: bool = True) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_damaging else -s[ok]
    return float(roc_auc_score(y[ok], score))


def safe_auprc(y: np.ndarray, s: np.ndarray, larger_is_damaging: bool = True) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_damaging else -s[ok]
    return float(average_precision_score(y[ok], score))


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def score_metric_rows(df: pd.DataFrame, y: np.ndarray, function_score: np.ndarray | None) -> pd.DataFrame:
    metrics = [
        ("ESM delta L2", "esm_delta_l2", True),
        ("ESM delta L1 mean", "esm_delta_l1_mean", True),
        ("ESM delta max abs", "esm_delta_max_abs", True),
        ("ESM protein mask", "esm_protein_mask", True),
        ("ESM-only CV logistic", "esm_only_cv_pred", True),
        ("ESM-only missense-only CV logistic", "esm_only_missense_cv_pred", True),
        ("annotation CV baseline", "annotation_cv_pred", True),
        ("ESM+annotation CV logistic", "esm_annotation_cv_pred", True),
    ]
    rows = []
    for name, col, larger in metrics:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        rows.append(
            {
                "metric": name,
                "n": int(np.isfinite(vals).sum()),
                "auroc_for_sge_lof": safe_auc(y, vals, larger),
                "auprc_for_sge_lof": safe_auprc(y, vals, larger),
                "spearman_vs_function_score": safe_spearman(vals, function_score)
                if function_score is not None
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def annotation_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_cols = [c for c in ["is_missense", "is_synonymous", "is_nonsense", "affect_splicing"] if c in df.columns]
    cat_cols = [c for c in ["vtype", "consequence", "brca2_domain"] if c in df.columns]
    out = df[numeric_cols + cat_cols].copy()
    for c in numeric_cols:
        out[c] = out[c].astype(float)
    for c in cat_cols:
        out[c] = out[c].fillna("NA").astype(str)
    return out, numeric_cols, cat_cols


def make_annotation_model(numeric_cols: list[str], cat_cols: list[str]):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return make_pipeline(
        pre,
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        ),
    )


def make_esm_model(random_state: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
    )


def cv_esm_only_predictions(
    pdelta: np.ndarray,
    y_fit: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed_offset: int = 0,
    y_eval: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    y_fit = np.asarray(y_fit, dtype=int)
    y_eval = y_fit if y_eval is None else np.asarray(y_eval, dtype=int)
    actual_splits = min(n_splits, len(np.unique(groups)))
    if actual_splits < 2:
        raise ValueError("Need at least two unique groups for GroupKFold")

    gkf = GroupKFold(n_splits=actual_splits)
    pred = np.zeros(len(y_fit), dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(pdelta, y_fit, groups), start=1):
        model = make_esm_model(seed_offset + fold)
        if len(np.unique(y_fit[train_idx])) < 2:
            pred[test_idx] = float(np.mean(y_fit[train_idx]))
        else:
            model.fit(pdelta[train_idx], y_fit[train_idx])
            pred[test_idx] = model.predict_proba(pdelta[test_idx])[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_idx)),
                "fit_lof_rate_test": float(np.mean(y_fit[test_idx])),
                "eval_lof_rate_test": float(np.mean(y_eval[test_idx])),
                "esm_only_auroc": safe_auc(y_eval[test_idx], pred[test_idx]),
                "esm_only_auprc": safe_auprc(y_eval[test_idx], pred[test_idx]),
            }
        )
    return pd.DataFrame(rows), pred


def cv_predictions(
    pdelta: np.ndarray,
    anno: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)
    pred_esm = np.zeros(len(y), dtype=float)
    pred_anno = np.zeros(len(y), dtype=float)
    pred_both = np.zeros(len(y), dtype=float)
    rows = []

    x_both = np.hstack([pdelta, anno[numeric_cols].to_numpy(dtype=float)]) if numeric_cols else pdelta.copy()
    for fold, (train_idx, test_idx) in enumerate(gkf.split(pdelta, y, groups), start=1):
        esm_model = make_esm_model(fold)
        anno_model = make_annotation_model(numeric_cols, cat_cols)
        both_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=fold + 100,
            ),
        )

        esm_model.fit(pdelta[train_idx], y[train_idx])
        anno_model.fit(anno.iloc[train_idx], y[train_idx])
        both_model.fit(x_both[train_idx], y[train_idx])

        pred_esm[test_idx] = esm_model.predict_proba(pdelta[test_idx])[:, 1]
        pred_anno[test_idx] = anno_model.predict_proba(anno.iloc[test_idx])[:, 1]
        pred_both[test_idx] = both_model.predict_proba(x_both[test_idx])[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_idx)),
                "lof_rate_test": float(np.mean(y[test_idx])),
                "esm_only_auroc": safe_auc(y[test_idx], pred_esm[test_idx]),
                "esm_only_auprc": safe_auprc(y[test_idx], pred_esm[test_idx]),
                "annotation_auroc": safe_auc(y[test_idx], pred_anno[test_idx]),
                "annotation_auprc": safe_auprc(y[test_idx], pred_anno[test_idx]),
                "esm_annotation_auroc": safe_auc(y[test_idx], pred_both[test_idx]),
                "esm_annotation_auprc": safe_auprc(y[test_idx], pred_both[test_idx]),
            }
        )
    return pd.DataFrame(rows), pred_esm, pred_anno, pred_both


def permutation_controls(
    pdelta: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    repeats: int,
    seed: int,
    scope: str,
    observed_scores: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    observed_auroc = safe_auc(y_true, observed_scores)
    observed_auprc = safe_auprc(y_true, observed_scores)
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        y_perm = rng.permutation(y_true)
        _, pred = cv_esm_only_predictions(
            pdelta,
            y_perm,
            groups,
            n_splits,
            seed_offset=seed + 1000 * (repeat + 1),
            y_eval=y_true,
        )
        rows.append(
            {
                "scope": scope,
                "repeat": repeat,
                "auroc_vs_true_labels": safe_auc(y_true, pred),
                "auprc_vs_true_labels": safe_auprc(y_true, pred),
            }
        )
    detail = pd.DataFrame(rows)
    auroc_null = detail["auroc_vs_true_labels"].to_numpy(dtype=float)
    auprc_null = detail["auprc_vs_true_labels"].to_numpy(dtype=float)
    summary = {
        "scope": scope,
        "n": int(len(y_true)),
        "n_lof": int(np.sum(y_true)),
        "observed_auroc": observed_auroc,
        "permuted_mean_auroc": float(np.nanmean(auroc_null)),
        "permuted_sd_auroc": float(np.nanstd(auroc_null, ddof=1)) if repeats > 1 else float("nan"),
        "empirical_p_permuted_auroc_ge_observed": float((1 + np.sum(auroc_null >= observed_auroc)) / (repeats + 1)),
        "observed_auprc": observed_auprc,
        "permuted_mean_auprc": float(np.nanmean(auprc_null)),
        "permuted_sd_auprc": float(np.nanstd(auprc_null, ddof=1)) if repeats > 1 else float("nan"),
        "empirical_p_permuted_auprc_ge_observed": float((1 + np.sum(auprc_null >= observed_auprc)) / (repeats + 1)),
        "permutation_sets": int(repeats),
    }
    return detail, summary


def group_summary(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(df), dtype=bool))]
    for col in ["vtype", "consequence", "brca2_domain"]:
        if col not in df.columns:
            continue
        for val, sub in df.groupby(col, dropna=False):
            mask = np.zeros(len(df), dtype=bool)
            mask[sub.index.to_numpy()] = True
            if mask.sum() >= 40:
                groups.append((f"{col}:{val}", mask))
    if "is_missense" in df.columns:
        groups.append(("is_missense:true", df["is_missense"].astype(bool).to_numpy()))
        groups.append(("is_missense:false", ~df["is_missense"].astype(bool).to_numpy()))

    rows = []
    for name, mask in groups:
        yy = y[mask]
        if len(yy) < 20 or len(np.unique(yy)) < 2:
            continue
        row = {
            "group": name,
            "n": int(mask.sum()),
            "n_lof": int(yy.sum()),
            "lof_rate": float(yy.mean()),
        }
        for col in [
            "esm_delta_l2",
            "esm_delta_l1_mean",
            "esm_delta_max_abs",
            "esm_only_cv_pred",
            "annotation_cv_pred",
            "esm_annotation_cv_pred",
        ]:
            row[f"auroc_{col}"] = safe_auc(yy, df.loc[mask, col].to_numpy(dtype=float))
            row[f"auprc_{col}"] = safe_auprc(yy, df.loc[mask, col].to_numpy(dtype=float))
        rows.append(row)
    return pd.DataFrame(rows)


def domain_effect_tests(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for group_col in ["consequence", "brca2_domain", "vtype"]:
        if group_col not in df.columns:
            continue
        for group, sub in df.groupby(group_col, dropna=False):
            if len(sub) < 40 or sub["label"].nunique() < 2:
                continue
            lof = sub[sub["label"].eq(1)]
            func = sub[sub["label"].eq(0)]
            row = {
                "group_kind": group_col,
                "group": str(group),
                "n": len(sub),
                "n_lof": int(lof.shape[0]),
                "lof_rate": float(sub["label"].mean()),
            }
            for col in ["esm_delta_l2", "esm_only_cv_pred", "annotation_cv_pred", "esm_annotation_cv_pred"]:
                a = lof[col].to_numpy(dtype=float)
                b = func[col].to_numpy(dtype=float)
                row[f"mean_{col}_lof"] = float(np.mean(a))
                row[f"mean_{col}_func"] = float(np.mean(b))
                row[f"delta_{col}_lof_minus_func"] = float(np.mean(a) - np.mean(b))
                row[f"mwu_{col}_lof_gt_func_p"] = float(mannwhitneyu(a, b, alternative="greater").pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    missense_folds: pd.DataFrame,
    groups: pd.DataFrame,
    tests: pd.DataFrame,
    null_summary: pd.DataFrame,
    null_controls: pd.DataFrame,
    n: int,
    n_lof: int,
    n_pmask: int,
) -> None:
    lines = [
        "# BRCA2 ESM-Only SGE Replication",
        "",
        "## Purpose",
        "",
        "Use the independent BRCA2 saturation genome editing table as an external functional dataset and ask how much signal is already present on the protein side before Evo2 DNA and cross-modal fusion finish.",
        "",
        "This is not the full CrossBioSAE mechanism replication. It is a protein-side baseline and readiness check for the BRCA2 replication branch.",
        "",
        "## Dataset",
        "",
        f"- Variants: {n}",
        f"- SGE LOF/pathogenic-like variants: {n_lof}",
        f"- Variants with protein ESM mask: {n_pmask}",
        "",
        "## Metric Summary",
        "",
        table(metrics),
        "",
        "## Fold Metrics",
        "",
        table(folds),
        "",
        "## Missense-Only ESM Fold Metrics",
        "",
        table(missense_folds),
        "",
        "## Label-Permutation Nulls",
        "",
        table(null_summary),
        "",
        "The permutation controls retrain the ESM-only model on shuffled labels under the same grouped cross-validation splits and score the resulting predictions against the true SGE labels. They test whether the ESM representation carries real functional signal rather than producing the observed AUROC from fold structure alone.",
        "",
        "### Null Repeats",
        "",
        table(null_controls, max_rows=80),
        "",
        "## Stratum Metrics",
        "",
        table(groups, max_rows=80),
        "",
        "## Group Effect Tests",
        "",
        table(tests, max_rows=80),
        "",
        "## Interpretation",
        "",
        "The dedicated missense-only ESM CV result is the strongest protein-side readiness evidence because it is trained and evaluated only where ESM deltas are biologically meaningful. The label-permutation nulls make the result harder to dismiss as grouped-fold structure or class imbalance.",
        "",
        "This is still not the full BRCA2 CrossBioSAE mechanism replication. The annotation baseline is intentionally reported beside ESM-only prediction; where annotation is stronger, the top-journal claim should emphasize cross-modal mechanism, causal intervention, and downstream utility rather than generic pathogenicity prediction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    z = np.load(args.esm)
    pdelta = z["pdelta"].astype(np.float32)
    pmask = z["pmask"].astype(bool)
    if len(df) != pdelta.shape[0]:
        raise RuntimeError(f"variant rows {len(df)} != ESM rows {pdelta.shape[0]}")
    if args.group_col not in df.columns:
        raise KeyError(f"group column {args.group_col!r} not found")

    y = df["label"].astype(int).to_numpy()
    groups = df[args.group_col].to_numpy()
    df = df.copy()
    df["esm_delta_l2"] = np.linalg.norm(pdelta, axis=1)
    df["esm_delta_l1_mean"] = np.mean(np.abs(pdelta), axis=1)
    df["esm_delta_max_abs"] = np.max(np.abs(pdelta), axis=1)
    df["esm_protein_mask"] = pmask.astype(float)

    anno, numeric_cols, cat_cols = annotation_frame(df)
    folds, pred_esm, pred_anno, pred_both = cv_predictions(
        pdelta,
        anno,
        numeric_cols,
        cat_cols,
        y,
        groups,
        args.n_splits,
    )
    df["esm_only_cv_pred"] = pred_esm
    df["annotation_cv_pred"] = pred_anno
    df["esm_annotation_cv_pred"] = pred_both

    missense_mask = df["is_missense"].astype(bool).to_numpy() if "is_missense" in df.columns else pmask.copy()
    missense_folds, missense_pred = cv_esm_only_predictions(
        pdelta[missense_mask],
        y[missense_mask],
        groups[missense_mask],
        args.n_splits,
        seed_offset=5000,
    )
    df["esm_only_missense_cv_pred"] = np.nan
    df.loc[missense_mask, "esm_only_missense_cv_pred"] = missense_pred

    null_tables = []
    null_summaries = []
    if args.permutation_sets > 0:
        detail, summary = permutation_controls(
            pdelta,
            y,
            groups,
            args.n_splits,
            args.permutation_sets,
            args.seed,
            "all",
            df["esm_only_cv_pred"].to_numpy(dtype=float),
        )
        null_tables.append(detail)
        null_summaries.append(summary)
        detail, summary = permutation_controls(
            pdelta[missense_mask],
            y[missense_mask],
            groups[missense_mask],
            args.n_splits,
            args.permutation_sets,
            args.seed + 17,
            "missense_only",
            missense_pred,
        )
        null_tables.append(detail)
        null_summaries.append(summary)
    null_controls = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()
    null_summary = pd.DataFrame(null_summaries)

    function_score = df["function_score"].to_numpy(dtype=float) if "function_score" in df.columns else None
    metrics = score_metric_rows(df, y, function_score)
    strata = group_summary(df, y)
    tests = domain_effect_tests(df, y)

    prefix = args.output_prefix
    metrics.to_csv(out / f"{prefix}_metric_summary.csv", index=False)
    folds.to_csv(out / f"{prefix}_folds.csv", index=False)
    missense_folds.to_csv(out / f"{prefix}_missense_folds.csv", index=False)
    strata.to_csv(out / f"{prefix}_strata.csv", index=False)
    tests.to_csv(out / f"{prefix}_group_tests.csv", index=False)
    null_summary.to_csv(out / f"{prefix}_null_summary.csv", index=False)
    null_controls.to_csv(out / f"{prefix}_null_controls.csv", index=False)
    df.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(
        out / f"{prefix}.md",
        metrics,
        folds,
        missense_folds,
        strata,
        tests,
        null_summary,
        null_controls,
        len(df),
        int(y.sum()),
        int(pmask.sum()),
    )

    print(f"Wrote BRCA2 ESM-only replication outputs to {out}")
    print(metrics.to_string(index=False))
    print()
    print(folds.to_string(index=False))
    print()
    print(missense_folds.to_string(index=False))
    if not null_summary.empty:
        print()
        print(null_summary.to_string(index=False))


if __name__ == "__main__":
    main()
