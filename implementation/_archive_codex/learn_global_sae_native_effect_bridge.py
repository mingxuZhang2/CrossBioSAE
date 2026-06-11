#!/usr/bin/env python3
"""Learn a held-out bridge from global SAE features to BRCA1 native-SAE necessity."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument(
        "--variant-scores",
        type=Path,
        default=root
        / "results"
        / "interpretability_applications"
        / "brca1_native_finetuned_sae_sufficiency_variant_scores.csv",
    )
    p.add_argument(
        "--acts-npz",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca1_sge_sae_acts.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    p.add_argument("--output-prefix", default="brca1_global_sae_native_effect_bridge")
    p.add_argument("--top-k-features", type=int, default=64)
    p.add_argument("--min-active", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-sets", type=int, default=100)
    p.add_argument("--permutation-sets", type=int, default=50)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def eligible_features(x_train: np.ndarray, min_active: int) -> np.ndarray:
    active = (x_train > 0).sum(axis=0)
    n = x_train.shape[0]
    return np.where((active >= min_active) & (active <= n - min_active))[0]


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_active: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = eligible_features(x_train, min_active)
    if len(eligible) == 0:
        raise RuntimeError("No eligible SAE features passed the activity filter")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(x_train[:, eligible], y_train)
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    k = min(top_k, len(eligible))
    order = np.argsort(scores)[::-1][:k]
    return eligible[order], scores[order]


def fit_predict(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_bin_train: np.ndarray,
    y_cont_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        ),
    )
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    clf.fit(x_train, y_bin_train)
    reg.fit(x_train, y_cont_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = reg.predict(x_test)
    log_coef = clf.named_steps["logisticregression"].coef_[0]
    ridge_coef = reg.named_steps["ridge"].coef_
    return prob, pred, log_coef, ridge_coef


def metric_row(
    source: str,
    fold: int,
    repeat: int,
    y_bin: np.ndarray,
    y_cont: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    n_features: int,
) -> dict[str, float | int | str]:
    return {
        "source": source,
        "fold": fold,
        "repeat": repeat,
        "n_test": int(len(y_bin)),
        "n_features": int(n_features),
        "high_effect_rate_test": float(np.mean(y_bin)),
        "high_effect_auroc": safe_auroc(y_bin, prob),
        "high_effect_auprc": safe_auprc(y_bin, prob),
        "native_effect_r2": float(r2_score(y_cont, pred)),
        "native_effect_spearman": safe_spearman(y_cont, pred),
    }


def summarize_distribution(df: pd.DataFrame, metric: str, observed: float) -> tuple[float, float, float]:
    if df.empty or metric not in df.columns:
        return float("nan"), float("nan"), float("nan")
    by_repeat = df.groupby("repeat")[metric].mean()
    vals = by_repeat.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    p = (1.0 + float((vals >= observed).sum())) / (len(vals) + 1.0)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, p


def load_cards(repo_root: Path) -> pd.DataFrame:
    cards_path = repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv"
    if not cards_path.exists():
        cards_path = repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards.csv"
    cards = read_csv(cards_path)

    sge_path = repo_root / "results" / "interpretability_applications" / "brca1_sge_sae_feature_summary.csv"
    if sge_path.exists():
        sge = pd.read_csv(sge_path)
        keep = [
            "feature",
            "n_active_brca1",
            "active_frac_brca1",
            "lof_rate_active",
            "lof_rate_inactive",
            "auroc_for_sge_lof",
            "auprc_for_sge_lof",
            "spearman_vs_function_score",
            "concept_v2",
            "modality",
            "path_rate",
            "path_enrich",
        ]
        keep = [c for c in keep if c in sge.columns]
        cards = cards.merge(sge[keep], on="feature", how="left", suffixes=("", "_brca1_sge"))
        for col in ["concept_v2", "modality", "path_rate", "path_enrich"]:
            alt = f"{col}_brca1_sge"
            if col in cards.columns and alt in cards.columns:
                cards[col] = cards[col].combine_first(cards[alt])
    return cards


def baseline_matrix(vs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_cols = [c for c in ["CADD", "phyloP", "is_missense"] if c in vs.columns]
    cat_cols = [c for c in ["vtype", "brca1_region", "consequence"] if c in vs.columns]
    parts = []
    names: list[str] = []
    if numeric_cols:
        num = vs[numeric_cols].copy()
        for c in numeric_cols:
            num[c] = pd.to_numeric(num[c], errors="coerce")
            num[c] = num[c].fillna(num[c].median())
        parts.append(num.to_numpy(dtype=float))
        names.extend(numeric_cols)
    for c in cat_cols:
        dummies = pd.get_dummies(vs[c].fillna("NA").astype(str), prefix=c, dtype=float)
        parts.append(dummies.to_numpy(dtype=float))
        names.extend(list(dummies.columns))
    if not parts:
        return np.ones((len(vs), 1), dtype=float), ["intercept_only"]
    return np.hstack(parts), names


def evaluate_baseline(
    x: np.ndarray,
    y_bin: np.ndarray,
    y_cont: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        prob, pred, _, _ = fit_predict(
            x[train_idx],
            x[test_idx],
            y_bin[train_idx],
            y_cont[train_idx],
        )
        rows.append(
            metric_row(
                "annotation_baseline",
                fold,
                0,
                y_bin[test_idx],
                y_cont[test_idx],
                prob,
                pred,
                x.shape[1],
            )
        )
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 100 or (abs(x) < 0.001 and x != 0):
        return f"{x:.3e}"
    return f"{x:.4f}"


def markdown_table(df: pd.DataFrame, n: int, cols: list[str]) -> str:
    if df.empty:
        return "No rows."
    show = df.head(n).copy()
    cols = [c for c in cols if c in show.columns]
    show = show[cols]
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(fmt)
    return show.to_markdown(index=False)


def write_report(
    out_path: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    features: pd.DataFrame,
    threshold: float,
    n_high: int,
    n_total: int,
) -> None:
    summary_show = summary.copy()
    for col in summary_show.columns:
        if pd.api.types.is_float_dtype(summary_show[col]):
            summary_show[col] = summary_show[col].map(fmt)

    global_rows = folds[folds["source"].eq("global_sae_selected")]
    base_rows = folds[folds["source"].eq("annotation_baseline")]

    lines = [
        "# BRCA1 Global SAE to Native-SAE Necessity Learned Bridge",
        "",
        "## Question",
        "",
        "Can named genome-wide SAE activations predict, on held-out BRCA1 SGE variants, which variants lose the most prediction support when the fold-native SAE directions are ablated?",
        "",
        "## Setup",
        "",
        f"- Variants: {n_total}",
        f"- High native-effect definition: top decile of `sae_recon_pred - top_feature_ablate_pred`, threshold {fmt(threshold)}.",
        f"- High native-effect variants: {n_high}",
        "- Feature selection is performed inside each training fold only.",
        "- Nulls: same-size random SAE feature sets and label-permuted feature selection/training.",
        "",
        "## Summary",
        "",
        summary_show.to_markdown(index=False),
        "",
        "## Fold Metrics",
        "",
        markdown_table(
            pd.concat([global_rows, base_rows], ignore_index=True),
            20,
            [
                "source",
                "fold",
                "n_features",
                "high_effect_auroc",
                "high_effect_auprc",
                "native_effect_r2",
                "native_effect_spearman",
            ],
        ),
        "",
        "## Recurrently Selected Global SAE Features",
        "",
        markdown_table(
            features,
            25,
            [
                "feature",
                "selection_count",
                "mean_f_score",
                "mean_abs_logistic_coef",
                "mean_abs_ridge_coef",
                "n_active_brca1",
                "auroc_for_sge_lof",
                "concept_v2",
                "modality",
                "path_rate",
                "path_enrich",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "A positive held-out bridge means the global, named SAE space carries information about which BRCA1 variants depend on native fine-tuned sparse directions. This supports a variant-level biological bridge between the genome-wide feature cards and the stronger native-SAE causal ablation result.",
        "",
        "This is still not a one-to-one mapping from individual native SAE features to individual global feature cards. It supports predictability and biological alignment, not exact feature identity.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vs = read_csv(args.variant_scores.resolve()).reset_index(drop=True)
    acts_npz = np.load(args.acts_npz.resolve())
    if "acts" not in acts_npz.files:
        raise KeyError(f"{args.acts_npz} does not contain an 'acts' array")
    acts = np.asarray(acts_npz["acts"], dtype=np.float32)
    if len(vs) != acts.shape[0]:
        raise RuntimeError(f"variant scores have {len(vs)} rows but SAE acts have {acts.shape[0]}")
    for col in ["sae_recon_pred", "top_feature_ablate_pred", "label"]:
        if col not in vs.columns:
            raise KeyError(f"Missing required column in variant scores: {col}")

    native_effect = (
        pd.to_numeric(vs["sae_recon_pred"], errors="coerce")
        - pd.to_numeric(vs["top_feature_ablate_pred"], errors="coerce")
    ).to_numpy(dtype=float)
    ok = np.isfinite(native_effect) & np.isfinite(pd.to_numeric(vs["label"], errors="coerce").to_numpy(dtype=float))
    if not ok.all():
        vs = vs.loc[ok].reset_index(drop=True)
        acts = acts[ok]
        native_effect = native_effect[ok]

    y_bin_label = vs["label"].astype(int).to_numpy()
    threshold = float(np.quantile(native_effect, 0.9))
    high_effect = native_effect >= threshold
    y_high = high_effect.astype(int)
    x = np.log1p(np.abs(acts).astype(np.float64))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(x, y_high))
    rng = np.random.default_rng(args.seed)

    fold_rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, float | int]] = []
    random_rows: list[dict[str, float | int | str]] = []
    perm_rows: list[dict[str, float | int | str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            feat, scores = select_features(
                x[train_idx],
                y_high[train_idx],
                args.min_active,
                args.top_k_features,
            )
            prob, pred, log_coef, ridge_coef = fit_predict(
                x[train_idx][:, feat],
                x[test_idx][:, feat],
                y_high[train_idx],
                native_effect[train_idx],
            )
            fold_rows.append(
                metric_row(
                    "global_sae_selected",
                    fold,
                    0,
                    y_high[test_idx],
                    native_effect[test_idx],
                    prob,
                    pred,
                    len(feat),
                )
            )
            for rank, (feature, f_score, lc, rc) in enumerate(zip(feat, scores, log_coef, ridge_coef), start=1):
                selected_rows.append(
                    {
                        "fold": fold,
                        "rank": rank,
                        "feature": int(feature),
                        "f_score": float(f_score),
                        "logistic_coef": float(lc),
                        "ridge_coef": float(rc),
                    }
                )

            eligible = eligible_features(x[train_idx], args.min_active)
            k = min(args.top_k_features, len(eligible))
            for repeat in range(1, args.random_sets + 1):
                feat_rand = rng.choice(eligible, size=k, replace=False)
                prob_rand, pred_rand, _, _ = fit_predict(
                    x[train_idx][:, feat_rand],
                    x[test_idx][:, feat_rand],
                    y_high[train_idx],
                    native_effect[train_idx],
                )
                random_rows.append(
                    metric_row(
                        "random_sae_features",
                        fold,
                        repeat,
                        y_high[test_idx],
                        native_effect[test_idx],
                        prob_rand,
                        pred_rand,
                        k,
                    )
                )

            for repeat in range(1, args.permutation_sets + 1):
                perm = rng.permutation(len(y_high))
                y_high_perm = y_high[perm]
                native_perm = native_effect[perm]
                feat_perm, _ = select_features(
                    x[train_idx],
                    y_high_perm[train_idx],
                    args.min_active,
                    args.top_k_features,
                )
                prob_perm, pred_perm, _, _ = fit_predict(
                    x[train_idx][:, feat_perm],
                    x[test_idx][:, feat_perm],
                    y_high_perm[train_idx],
                    native_perm[train_idx],
                )
                perm_rows.append(
                    metric_row(
                        "label_permuted",
                        fold,
                        repeat,
                        y_high[test_idx],
                        native_effect[test_idx],
                        prob_perm,
                        pred_perm,
                        len(feat_perm),
                    )
                )

    folds = pd.DataFrame(fold_rows)
    random_controls = pd.DataFrame(random_rows)
    permutation_controls = pd.DataFrame(perm_rows)
    selected = pd.DataFrame(selected_rows)

    baseline_x, baseline_names = baseline_matrix(vs)
    baseline_folds = evaluate_baseline(baseline_x, y_high, native_effect, splits)
    folds_all = pd.concat([folds, baseline_folds], ignore_index=True)

    cards = load_cards(repo_root)
    feature_summary = (
        selected.groupby("feature")
        .agg(
            selection_count=("fold", "nunique"),
            mean_rank=("rank", "mean"),
            mean_f_score=("f_score", "mean"),
            mean_logistic_coef=("logistic_coef", "mean"),
            mean_abs_logistic_coef=("logistic_coef", lambda s: float(np.mean(np.abs(s)))),
            mean_ridge_coef=("ridge_coef", "mean"),
            mean_abs_ridge_coef=("ridge_coef", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
        .sort_values(["selection_count", "mean_f_score"], ascending=[False, False])
    )
    feature_summary = feature_summary.merge(cards, on="feature", how="left")

    obs = folds["high_effect_auroc"].mean()
    obs_auprc = folds["high_effect_auprc"].mean()
    obs_r2 = folds["native_effect_r2"].mean()
    obs_spear = folds["native_effect_spearman"].mean()
    base = baseline_folds.mean(numeric_only=True)

    rows = []
    for metric, observed in [
        ("high_effect_auroc", obs),
        ("high_effect_auprc", obs_auprc),
        ("native_effect_r2", obs_r2),
        ("native_effect_spearman", obs_spear),
    ]:
        rand_mean, rand_sd, rand_p = summarize_distribution(random_controls, metric, observed)
        perm_mean, perm_sd, perm_p = summarize_distribution(permutation_controls, metric, observed)
        rows.append(
            {
                "metric": metric,
                "global_sae_selected_mean": float(observed),
                "annotation_baseline_mean": float(base.get(metric, np.nan)),
                "random_feature_mean": rand_mean,
                "random_feature_sd": rand_sd,
                "empirical_p_random_ge_observed": rand_p,
                "label_permuted_mean": perm_mean,
                "label_permuted_sd": perm_sd,
                "empirical_p_permuted_ge_observed": perm_p,
            }
        )
    summary = pd.DataFrame(rows)
    summary["n_variants"] = len(vs)
    summary["n_high_effect"] = int(y_high.sum())
    summary["high_effect_threshold"] = threshold
    summary["top_k_features"] = args.top_k_features
    summary["baseline_feature_count"] = len(baseline_names)
    summary["sge_lof_rate_high_effect"] = float(y_bin_label[y_high.astype(bool)].mean())
    summary["sge_lof_rate_rest"] = float(y_bin_label[~y_high.astype(bool)].mean())

    prefix = args.output_prefix
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    folds_all.to_csv(out_dir / f"{prefix}_folds.csv", index=False)
    feature_summary.to_csv(out_dir / f"{prefix}_selected_features.csv", index=False)
    random_controls.to_csv(out_dir / f"{prefix}_random_controls.csv", index=False)
    permutation_controls.to_csv(out_dir / f"{prefix}_permutation_controls.csv", index=False)
    vs.assign(native_effect=native_effect, high_native_effect=y_high).to_csv(
        out_dir / f"{prefix}_variant_scores.csv",
        index=False,
    )
    write_report(
        out_dir / f"{prefix}.md",
        summary,
        folds_all,
        feature_summary,
        threshold,
        int(y_high.sum()),
        len(vs),
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
