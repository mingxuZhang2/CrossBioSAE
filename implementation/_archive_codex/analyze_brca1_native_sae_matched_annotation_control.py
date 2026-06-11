#!/usr/bin/env python
"""Annotation-matched control for BRCA1 native-SAE biological localization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OUT_DIR = Path("results/interpretability_applications")


def empirical_p_greater(null: np.ndarray, observed: float) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return float("nan")
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def empirical_p_less(null: np.ndarray, observed: float) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return float("nan")
    return float((np.sum(null <= observed) + 1) / (len(null) + 1))


def build_pools(df: pd.DataFrame, high_mask: np.ndarray) -> dict[tuple[str, str, str], np.ndarray]:
    pools: dict[tuple[str, str, str], np.ndarray] = {}
    keys = {
        ("consequence_region", "consequence", "brca1_region"),
        ("consequence", "consequence", ""),
        ("region", "brca1_region", ""),
        ("all", "", ""),
    }
    low_df = df.loc[~high_mask].copy()
    for kind, col_a, col_b in keys:
        if kind == "all":
            pools[(kind, "all", "all")] = low_df.index.to_numpy()
            continue
        if col_b:
            group_cols = [col_a, col_b]
        else:
            group_cols = [col_a]
        for vals, group in low_df.groupby(group_cols, dropna=False):
            if not isinstance(vals, tuple):
                vals = (vals, "")
            elif len(vals) == 1:
                vals = (vals[0], "")
            pools[(kind, str(vals[0]), str(vals[1]))] = group.index.to_numpy()
    return pools


def candidate_pools_for_high_rows(
    high_rows: pd.DataFrame,
    pools: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[np.ndarray], list[str]]:
    candidate_pools: list[np.ndarray] = []
    levels: list[str] = []
    all_pool = pools[("all", "all", "all")]
    for _, row in high_rows.iterrows():
        candidates = pools.get(("consequence_region", str(row["consequence"]), str(row["brca1_region"])))
        level = "consequence_region"
        if candidates is None or len(candidates) == 0:
            candidates = pools.get(("consequence", str(row["consequence"]), ""))
            level = "consequence"
        if candidates is None or len(candidates) == 0:
            candidates = pools.get(("region", str(row["brca1_region"]), ""))
            level = "region"
        if candidates is None or len(candidates) == 0:
            candidates = all_pool
            level = "all"
        candidate_pools.append(candidates)
        levels.append(level)
    return candidate_pools, levels


def sample_matched_indices(
    candidate_pools: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    for candidates in candidate_pools:
        selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=int)


def metric_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "label": df["label"].to_numpy(dtype=float),
        "function_score": df["function_score"].to_numpy(dtype=float),
        "CADD": df["CADD"].to_numpy(dtype=float),
        "phyloP": df["phyloP"].to_numpy(dtype=float),
        "is_missense": df["is_missense"].astype(bool).to_numpy(dtype=float),
        "is_noncoding": df["vtype"].eq("noncoding").to_numpy(dtype=float),
    }


def summarize_idx(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    return {
        "lof_rate": float(np.nanmean(arrays["label"][idx])),
        "mean_function_score": float(np.nanmean(arrays["function_score"][idx])),
        "mean_cadd": float(np.nanmean(arrays["CADD"][idx])),
        "mean_phylop": float(np.nanmean(arrays["phyloP"][idx])),
        "missense_rate": float(np.nanmean(arrays["is_missense"][idx])),
        "noncoding_rate": float(np.nanmean(arrays["is_noncoding"][idx])),
    }


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if n is not None:
        df = df.head(n)
    if df.empty:
        return "No rows.\n"
    return df[cols].to_markdown(index=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    scores = pd.read_csv(out / "brca1_native_finetuned_sae_intervention_variant_scores.csv")
    required = [
        "consequence",
        "brca1_region",
        "label",
        "function_score",
        "CADD",
        "phyloP",
        "is_missense",
        "vtype",
        "sae_recon_pred",
        "top_feature_ablate_pred",
    ]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    scores = scores.reset_index(drop=True)
    scores["native_effect"] = scores["sae_recon_pred"] - scores["top_feature_ablate_pred"]
    threshold = float(scores["native_effect"].quantile(0.9))
    high_mask = scores["native_effect"].to_numpy() >= threshold
    high_rows = scores.loc[high_mask].copy()
    high_idx = high_rows.index.to_numpy()
    arrays = metric_arrays(scores)

    observed = summarize_idx(arrays, high_idx)
    rest_idx = scores.index.to_numpy()[~high_mask]
    rest = summarize_idx(arrays, rest_idx)
    odds, fisher_p = fisher_exact(
        [
            [int(scores.loc[high_idx, "label"].sum()), int(len(high_idx) - scores.loc[high_idx, "label"].sum())],
            [int(scores.loc[rest_idx, "label"].sum()), int(len(rest_idx) - scores.loc[rest_idx, "label"].sum())],
        ],
        alternative="greater",
    )

    rng = np.random.default_rng(args.seed)
    pools = build_pools(scores, high_mask)
    candidate_pools, levels = candidate_pools_for_high_rows(high_rows, pools)
    level_counts: dict[str, int] = {}
    for level in levels:
        level_counts[level] = level_counts.get(level, 0) + 1
    null_rows: list[dict[str, float | int]] = []
    for i in range(args.permutations):
        idx = sample_matched_indices(candidate_pools, rng)
        vals = summarize_idx(arrays, idx)
        vals["permutation"] = i
        null_rows.append(vals)
    null = pd.DataFrame(null_rows)

    summary_rows = [
        {
            "comparison": "top_decile_native_effect_vs_rest_unmatched",
            "n_top_decile": len(high_idx),
            "observed_lof_rate": observed["lof_rate"],
            "rest_or_null_mean_lof_rate": rest["lof_rate"],
            "rest_or_null_sd_lof_rate": np.nan,
            "observed_minus_rest_or_null": observed["lof_rate"] - rest["lof_rate"],
            "odds_ratio": float(odds),
            "p_value": float(fisher_p),
            "interpretation": "Original biological-bridge enrichment; includes annotation/composition effects.",
        },
        {
            "comparison": "top_decile_native_effect_vs_consequence_region_matched_null",
            "n_top_decile": len(high_idx),
            "observed_lof_rate": observed["lof_rate"],
            "rest_or_null_mean_lof_rate": float(null["lof_rate"].mean()),
            "rest_or_null_sd_lof_rate": float(null["lof_rate"].std(ddof=1)),
            "observed_minus_rest_or_null": float(observed["lof_rate"] - null["lof_rate"].mean()),
            "odds_ratio": np.nan,
            "p_value": empirical_p_greater(null["lof_rate"].to_numpy(), observed["lof_rate"]),
            "interpretation": "Tests whether native-effect LOF enrichment exceeds matched consequence/region composition.",
        },
    ]
    summary = pd.DataFrame(summary_rows)

    balance_rows = []
    for metric in ["mean_function_score", "mean_cadd", "mean_phylop", "missense_rate", "noncoding_rate"]:
        balance_rows.append(
            {
                "metric": metric,
                "observed_top_decile": observed[metric],
                "rest": rest[metric],
                "matched_null_mean": float(null[metric].mean()),
                "matched_null_sd": float(null[metric].std(ddof=1)),
                "empirical_p_greater": empirical_p_greater(null[metric].to_numpy(), observed[metric]),
                "empirical_p_less": empirical_p_less(null[metric].to_numpy(), observed[metric]),
            }
        )
    balance = pd.DataFrame(balance_rows)

    strata = (
        high_rows.groupby(["consequence", "brca1_region"], dropna=False)
        .agg(n_high_effect=("label", "size"), lof_rate=("label", "mean"), mean_native_effect=("native_effect", "mean"))
        .reset_index()
        .sort_values(["n_high_effect", "lof_rate"], ascending=[False, False])
    )

    level_total = sum(level_counts.values())
    level_df = pd.DataFrame(
        [
            {
                "match_level": level,
                "n_draws": count,
                "frac_draws": count / max(level_total, 1),
            }
            for level, count in sorted(level_counts.items())
        ]
    )

    paths = {
        "summary": out / "brca1_native_sae_matched_annotation_control_summary.csv",
        "null": out / "brca1_native_sae_matched_annotation_control_null.csv",
        "balance": out / "brca1_native_sae_matched_annotation_control_balance.csv",
        "strata": out / "brca1_native_sae_matched_annotation_control_strata.csv",
        "match_levels": out / "brca1_native_sae_matched_annotation_control_match_levels.csv",
        "report": out / "brca1_native_sae_matched_annotation_control.md",
    }
    summary.to_csv(paths["summary"], index=False)
    null.to_csv(paths["null"], index=False)
    balance.to_csv(paths["balance"], index=False)
    strata.to_csv(paths["strata"], index=False)
    level_df.to_csv(paths["match_levels"], index=False)

    report_lines = [
        "# BRCA1 Native-SAE Annotation-Matched Control",
        "",
        "## Purpose",
        "",
        "This audit asks whether the BRCA1 native-SAE biological bridge is stronger than a",
        "null set matched to the top native-effect variants by consequence and BRCA1 region.",
        "It is a claim-boundary check: passing would show signal beyond annotation",
        "composition; failing would mean the current bridge should be described as",
        "biological localization rather than annotation-independent mechanism discovery.",
        "",
        "## Primary Result",
        "",
        md_table(
            summary,
            [
                "comparison",
                "n_top_decile",
                "observed_lof_rate",
                "rest_or_null_mean_lof_rate",
                "rest_or_null_sd_lof_rate",
                "observed_minus_rest_or_null",
                "odds_ratio",
                "p_value",
            ],
        ),
        "",
        "## Balance Checks",
        "",
        md_table(
            balance,
            [
                "metric",
                "observed_top_decile",
                "rest",
                "matched_null_mean",
                "matched_null_sd",
                "empirical_p_greater",
                "empirical_p_less",
            ],
        ),
        "",
        "## Matching Levels",
        "",
        md_table(level_df, ["match_level", "n_draws", "frac_draws"]),
        "",
        "## Top Native-Effect Annotation Strata",
        "",
        md_table(strata, ["consequence", "brca1_region", "n_high_effect", "lof_rate", "mean_native_effect"], n=20),
        "",
        "## Interpretation",
        "",
        "- The unmatched enrichment is the original Figure 3 biological-localization signal.",
        "- The matched-null row is the stricter reviewer-facing test for whether this signal exceeds consequence/domain composition.",
        "- If the matched-null p-value is not significant, keep the claim as localization/bridge evidence and avoid saying the BRCA1 feature cards discover annotation-independent biology.",
        "",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")

    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
