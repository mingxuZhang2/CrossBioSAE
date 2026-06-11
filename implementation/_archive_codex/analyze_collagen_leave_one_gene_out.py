#!/usr/bin/env python3
"""Leave-one-gene-out validation for collagen Gly-X-Y sparse features."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("results/interpretability_applications")
COLLAGEN_GENES = {
    "COL1A1",
    "COL1A2",
    "COL2A1",
    "COL3A1",
    "COL4A3",
    "COL4A4",
    "COL4A5",
    "COL5A1",
    "COL5A2",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "COL7A1",
    "COL9A1",
    "COL11A1",
    "COL11A2",
    "COL17A1",
}
STRUCTURAL_GENES = COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}


@dataclass
class SelectionResult:
    holdout_gene: str
    selected_features: np.ndarray
    feature_table: pd.DataFrame


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=repo_root / OUT_DIR)
    parser.add_argument("--top-k-features", type=int, default=16)
    parser.add_argument("--min-train-cases", type=int, default=50)
    parser.add_argument("--min-test-cases", type=int, default=5)
    parser.add_argument("--min-control-cases", type=int, default=10)
    parser.add_argument("--min-train-active-frac", type=float, default=0.05)
    parser.add_argument("--n-random-sets", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=630)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def strict_gly_substitution(value: object) -> bool:
    text = "" if pd.isna(value) else str(value)
    return bool(re.match(r"^Gly\d+[A-Z][a-z]{2}$", text))


def strict_missense_pchange(value: object) -> bool:
    text = "" if pd.isna(value) else str(value)
    return bool(re.match(r"^[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", text))


def read_inputs(repo: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    known = pd.read_csv(require_file(repo / OUT_DIR / "known_variant_novel_scores.csv"))
    acts = np.load(require_file(repo / "results/sae_genomewide/sae_genomewide_acts.npz"))["acts"]
    cards = pd.read_csv(require_file(repo / "results/sae_genomewide/sae_genomewide_cards_deep.csv"))
    if len(known) != acts.shape[0]:
        raise ValueError(f"Known rows and SAE activations differ: {len(known)} vs {acts.shape[0]}")

    out = known.copy()
    out["gene"] = out["gene"].fillna("")
    out["label"] = out["label"].astype(int)
    out["is_collagen_gene"] = out["gene"].isin(COLLAGEN_GENES)
    out["is_structural_gene"] = out["gene"].isin(STRUCTURAL_GENES)
    out["is_gly_sub_strict"] = out["pchange"].apply(strict_gly_substitution)
    out["is_missense_like"] = out["pchange"].apply(strict_missense_pchange)
    return out, acts.astype(np.float32, copy=False), cards


def safe_auc(case_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if len(case_scores) < 2 or len(control_scores) < 2:
        return float("nan")
    y = np.r_[np.ones(len(case_scores), dtype=int), np.zeros(len(control_scores), dtype=int)]
    s = np.r_[case_scores, control_scores]
    if not np.isfinite(s).all() or len(np.unique(s)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_mannwhitney(case_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if len(case_scores) < 2 or len(control_scores) < 2:
        return float("nan")
    return float(mannwhitneyu(case_scores, control_scores, alternative="greater").pvalue)


def score_features(acts: np.ndarray, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(acts.shape[0], dtype=float)
    return np.abs(acts[:, features]).sum(axis=1)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


def select_features(
    df: pd.DataFrame,
    acts: np.ndarray,
    holdout_gene: str,
    args: argparse.Namespace,
) -> SelectionResult | None:
    train_case_mask = (
        df["is_collagen_gene"]
        & df["is_gly_sub_strict"]
        & df["is_missense_like"]
        & df["label"].eq(1)
        & df["gene"].ne(holdout_gene)
    )
    train_control_mask = (
        df["label"].eq(1)
        & df["is_missense_like"]
        & ~(
            df["is_collagen_gene"]
            & df["is_gly_sub_strict"]
        )
        & df["gene"].ne(holdout_gene)
    )
    if int(train_case_mask.sum()) < args.min_train_cases or int(train_control_mask.sum()) < args.min_control_cases:
        return None

    case_abs = np.abs(acts[train_case_mask.to_numpy()])
    control_abs = np.abs(acts[train_control_mask.to_numpy()])
    case_mean = case_abs.mean(axis=0)
    control_mean = control_abs.mean(axis=0)
    case_active_frac = (case_abs > 1e-6).mean(axis=0)
    delta = case_mean - control_mean

    pvalues = np.full(acts.shape[1], np.nan, dtype=float)
    candidate_idx = np.where((case_active_frac >= args.min_train_active_frac) & (delta > 0))[0]
    for feature in candidate_idx:
        pvalues[feature] = safe_mannwhitney(case_abs[:, feature], control_abs[:, feature])
    qvalues = bh_qvalues(pvalues)

    table = pd.DataFrame(
        {
            "feature": np.arange(acts.shape[1], dtype=int),
            "holdout_gene": holdout_gene,
            "train_n_case": int(train_case_mask.sum()),
            "train_n_control": int(train_control_mask.sum()),
            "train_case_mean_abs_activation": case_mean,
            "train_control_mean_abs_activation": control_mean,
            "train_delta_mean_abs_activation": delta,
            "train_case_active_frac": case_active_frac,
            "train_mannwhitney_p": pvalues,
            "train_mannwhitney_q": qvalues,
        }
    )
    selected = (
        table.loc[candidate_idx]
        .sort_values(["train_mannwhitney_p", "train_delta_mean_abs_activation"], ascending=[True, False])
        .head(args.top_k_features)
        .copy()
    )
    if selected.empty:
        return None
    return SelectionResult(
        holdout_gene=holdout_gene,
        selected_features=selected["feature"].astype(int).to_numpy(),
        feature_table=selected,
    )


def matched_random_sets(
    selected: np.ndarray,
    cards: pd.DataFrame,
    n_features: int,
    n_sets: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    meta = pd.DataFrame({"feature": np.arange(n_features, dtype=int)})
    cards_small = cards[["feature", "n_active", "path_rate"]].copy()
    meta = meta.merge(cards_small, on="feature", how="left")
    meta["n_active"] = meta["n_active"].fillna(meta["n_active"].median())
    meta["path_rate"] = meta["path_rate"].fillna(meta["path_rate"].median())
    meta["active_bin"] = pd.qcut(meta["n_active"].rank(method="first"), q=10, labels=False, duplicates="drop")
    meta["path_bin"] = pd.qcut(meta["path_rate"].rank(method="first"), q=10, labels=False, duplicates="drop")
    selected_set = set(int(x) for x in selected)
    rows_by_feature = meta.set_index("feature")
    all_pool = meta.loc[~meta["feature"].isin(selected_set), "feature"].astype(int).to_numpy()
    random_sets: list[np.ndarray] = []
    for _ in range(n_sets):
        chosen: list[int] = []
        used = set(selected_set)
        for feature in selected:
            row = rows_by_feature.loc[int(feature)]
            pool = meta[
                (meta["active_bin"].eq(row["active_bin"]))
                & (meta["path_bin"].eq(row["path_bin"]))
                & ~meta["feature"].isin(used)
            ]["feature"].astype(int).to_numpy()
            if len(pool) == 0:
                pool = np.array([x for x in all_pool if x not in used], dtype=int)
            if len(pool) == 0:
                pool = all_pool
            pick = int(rng.choice(pool))
            chosen.append(pick)
            used.add(pick)
        random_sets.append(np.array(chosen, dtype=int))
    return random_sets


def control_masks(df: pd.DataFrame, holdout_gene: str) -> dict[str, pd.Series]:
    return {
        "pathogenic_nonstructural_missense": (
            df["label"].eq(1)
            & df["is_missense_like"]
            & ~df["is_structural_gene"]
        ),
        "benign_nonstructural_missense": (
            df["label"].eq(0)
            & df["is_missense_like"]
            & ~df["is_structural_gene"]
        ),
        "holdout_pathogenic_non_gly_missense": (
            df["gene"].eq(holdout_gene)
            & df["label"].eq(1)
            & df["is_missense_like"]
            & ~df["is_gly_sub_strict"]
        ),
        "holdout_benign_missense": (
            df["gene"].eq(holdout_gene)
            & df["label"].eq(0)
            & df["is_missense_like"]
        ),
    }


def evaluate_holdout(
    df: pd.DataFrame,
    acts: np.ndarray,
    selection: SelectionResult,
    random_sets: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_gene = selection.holdout_gene
    case_mask = (
        df["gene"].eq(holdout_gene)
        & df["is_gly_sub_strict"]
        & df["is_missense_like"]
        & df["label"].eq(1)
    )
    n_case = int(case_mask.sum())
    if n_case < args.min_test_cases:
        return pd.DataFrame(), pd.DataFrame()

    observed_scores = score_features(acts, selection.selected_features)
    random_score_matrix = [score_features(acts, random_features) for random_features in random_sets]
    rows = []
    null_rows = []
    for control_name, control_mask in control_masks(df, holdout_gene).items():
        n_control = int(control_mask.sum())
        if n_control < args.min_control_cases:
            continue
        case_scores = observed_scores[case_mask.to_numpy()]
        control_scores = observed_scores[control_mask.to_numpy()]
        observed_auc = safe_auc(case_scores, control_scores)
        observed_p = safe_mannwhitney(case_scores, control_scores)
        random_aucs = []
        random_deltas = []
        for i, random_scores in enumerate(random_score_matrix):
            r_case = random_scores[case_mask.to_numpy()]
            r_control = random_scores[control_mask.to_numpy()]
            random_auc = safe_auc(r_case, r_control)
            random_delta = float(np.mean(r_case) - np.mean(r_control))
            random_aucs.append(random_auc)
            random_deltas.append(random_delta)
            null_rows.append(
                {
                    "holdout_gene": holdout_gene,
                    "control": control_name,
                    "random_set": i,
                    "random_auroc": random_auc,
                    "random_delta_mean": random_delta,
                }
            )
        random_auc_arr = np.asarray(random_aucs, dtype=float)
        finite = np.isfinite(random_auc_arr)
        empirical_p = (
            (1.0 + float(np.sum(random_auc_arr[finite] >= observed_auc))) / (1.0 + float(np.sum(finite)))
            if np.isfinite(observed_auc) and finite.any()
            else float("nan")
        )
        rows.append(
            {
                "holdout_gene": holdout_gene,
                "control": control_name,
                "n_case": n_case,
                "n_control": n_control,
                "n_selected_features": len(selection.selected_features),
                "observed_case_mean": float(np.mean(case_scores)),
                "observed_control_mean": float(np.mean(control_scores)),
                "observed_delta_mean": float(np.mean(case_scores) - np.mean(control_scores)),
                "observed_auroc": observed_auc,
                "observed_mannwhitney_p": observed_p,
                "random_auroc_mean": float(np.nanmean(random_auc_arr)) if finite.any() else float("nan"),
                "random_auroc_sd": float(np.nanstd(random_auc_arr)) if finite.any() else float("nan"),
                "observed_minus_random_auroc_mean": (
                    float(observed_auc - np.nanmean(random_auc_arr)) if np.isfinite(observed_auc) and finite.any() else float("nan")
                ),
                "empirical_p_random_auroc_ge_observed": empirical_p,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary = summary[summary["control"].eq("pathogenic_nonstructural_missense")].copy()
    passed = primary[
        (primary["observed_auroc"] >= 0.60)
        & (primary["empirical_p_random_auroc_ge_observed"] <= 0.05)
    ]
    lines = [
        "# Collagen Leave-One-Gene-Out Feature Validation",
        "",
        "## Purpose",
        "",
        "This analysis tests whether collagen Gly-X-Y sparse-feature evidence generalizes across collagen genes. For each held-out collagen gene, feature selection uses only pathogenic Gly-X-Y variants from the other collagen genes and pathogenic non-collagen/non-Gly missense controls. The held-out gene is used only for evaluation.",
        "",
        "## Configuration",
        "",
        f"- top-k selected features per fold: {args.top_k_features}",
        f"- random matched feature sets: {args.n_random_sets}",
        f"- minimum held-out pathogenic Gly-X-Y cases: {args.min_test_cases}",
        "",
        "## Primary Result",
        "",
        f"- held-out genes tested against pathogenic nonstructural missense controls: {len(primary)}",
        f"- genes passing AUROC >= 0.60 and empirical random-feature p <= 0.05: {len(passed)}",
        f"- median observed AUROC: {fmt(primary['observed_auroc'].median() if not primary.empty else float('nan'))}",
        f"- median observed-minus-random AUROC: {fmt(primary['observed_minus_random_auroc_mean'].median() if not primary.empty else float('nan'))}",
        "",
        "## Per-Gene Primary Table",
        "",
    ]
    if primary.empty:
        lines.append("_No primary rows._")
    else:
        cols = [
            "holdout_gene",
            "n_case",
            "n_control",
            "observed_auroc",
            "random_auroc_mean",
            "observed_minus_random_auroc_mean",
            "empirical_p_random_auroc_ge_observed",
        ]
        lines.append(primary[cols].sort_values("observed_auroc", ascending=False).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive rows support cross-gene transfer of collagen Gly-X-Y sparse features, which is stronger than a pooled enrichment test.",
            "- This remains a ClinVar-label and feature-score validation, not a causal intervention on the predictor and not a VUS reclassification endpoint.",
            "- Genes that fail or lack controls define where collagen evidence should stay as manual-review context.",
            "",
            "## Outputs",
            "",
            "- `collagen_leave_one_gene_out_primary_summary.csv`",
            "- `collagen_leave_one_gene_out_control_tests.csv`",
            "- `collagen_leave_one_gene_out_selected_features.csv`",
            "- `collagen_leave_one_gene_out_random_null.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    df, acts, cards = read_inputs(repo)
    tested_genes = sorted(
        gene
        for gene, sub in df[df["is_collagen_gene"]].groupby("gene")
        if int((sub["is_gly_sub_strict"] & sub["is_missense_like"] & sub["label"].eq(1)).sum()) >= args.min_test_cases
    )
    summary_rows = []
    null_rows = []
    selected_rows = []
    for gene in tested_genes:
        selection = select_features(df, acts, gene, args)
        if selection is None:
            continue
        random_sets = matched_random_sets(
            selection.selected_features,
            cards,
            acts.shape[1],
            args.n_random_sets,
            rng,
        )
        tests, null = evaluate_holdout(df, acts, selection, random_sets, args)
        if tests.empty:
            continue
        summary_rows.append(tests)
        null_rows.append(null)
        selected_rows.append(selection.feature_table)

    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    primary = (
        summary[summary["control"].eq("pathogenic_nonstructural_missense")].copy()
        if not summary.empty
        else pd.DataFrame()
    )

    primary.to_csv(out / "collagen_leave_one_gene_out_primary_summary.csv", index=False)
    summary.to_csv(out / "collagen_leave_one_gene_out_control_tests.csv", index=False)
    selected.to_csv(out / "collagen_leave_one_gene_out_selected_features.csv", index=False)
    null.to_csv(out / "collagen_leave_one_gene_out_random_null.csv", index=False)
    write_report(out / "collagen_leave_one_gene_out.md", summary, selected, args)

    print(f"wrote {out / 'collagen_leave_one_gene_out_primary_summary.csv'}")
    print(f"wrote {out / 'collagen_leave_one_gene_out_control_tests.csv'}")
    print(f"wrote {out / 'collagen_leave_one_gene_out_selected_features.csv'}")
    print(f"wrote {out / 'collagen_leave_one_gene_out_random_null.csv'}")
    print(f"wrote {out / 'collagen_leave_one_gene_out.md'}")


if __name__ == "__main__":
    main()
