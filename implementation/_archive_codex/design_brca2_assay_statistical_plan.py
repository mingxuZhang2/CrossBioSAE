#!/usr/bin/env python3
"""Create a statistical analysis plan for the BRCA2 prospective assay panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, fisher_exact


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_prospective_followup_panel.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_assay_statistical_plan")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--strict-alpha", type=float, default=0.01)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def one_sided_fisher(k_pos: int, n_pos: int, k_neg: int, n_neg: int) -> float:
    table_2x2 = [[k_pos, n_pos - k_pos], [k_neg, n_neg - k_neg]]
    _, pval = fisher_exact(table_2x2, alternative="greater")
    return float(pval)


def fisher_thresholds(n_pos: int, n_neg: int, alpha: float, strict_alpha: float) -> pd.DataFrame:
    rows = []
    for k_neg in range(n_neg + 1):
        min_alpha = None
        min_strict = None
        for k_pos in range(n_pos + 1):
            pval = one_sided_fisher(k_pos, n_pos, k_neg, n_neg)
            if min_alpha is None and pval < alpha:
                min_alpha = k_pos
            if min_strict is None and pval < strict_alpha:
                min_strict = k_pos
        rows.append(
            {
                "benign_control_lof_count": k_neg,
                "benign_control_lof_rate": k_neg / n_neg,
                f"min_pathogenic_lof_for_p_lt_{alpha:g}": min_alpha,
                f"min_pathogenic_lof_rate_for_p_lt_{alpha:g}": None if min_alpha is None else min_alpha / n_pos,
                f"min_pathogenic_lof_for_p_lt_{strict_alpha:g}": min_strict,
                f"min_pathogenic_lof_rate_for_p_lt_{strict_alpha:g}": None if min_strict is None else min_strict / n_pos,
            }
        )
    return pd.DataFrame(rows)


def exact_power(n_pos: int, n_neg: int, p_pos: float, p_neg: float, alpha: float) -> float:
    power = 0.0
    for k_pos in range(n_pos + 1):
        prob_pos = binom.pmf(k_pos, n_pos, p_pos)
        if prob_pos == 0:
            continue
        for k_neg in range(n_neg + 1):
            pval = one_sided_fisher(k_pos, n_pos, k_neg, n_neg)
            if pval < alpha:
                power += prob_pos * binom.pmf(k_neg, n_neg, p_neg)
    return float(power)


def power_grid(n_pos: int, n_neg: int, alpha: float) -> pd.DataFrame:
    rows = []
    for p_pos in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for p_neg in [0.0, 0.05, 0.10, 0.20, 0.30]:
            rows.append(
                {
                    "pathogenic_target_true_lof_rate": p_pos,
                    "benign_control_true_lof_rate": p_neg,
                    "alpha": alpha,
                    "power": exact_power(n_pos, n_neg, p_pos, p_neg, alpha),
                }
            )
    return pd.DataFrame(rows)


def split_mechanism_thresholds(n_split: int, background_rates: list[tuple[str, float]], alpha: float) -> pd.DataFrame:
    rows = []
    for label, p0 in background_rates:
        min_k = None
        min_p = None
        for k in range(n_split + 1):
            pval = binomtest(k, n_split, p0, alternative="greater").pvalue
            if pval < alpha:
                min_k = k
                min_p = pval
                break
        rows.append(
            {
                "null_rate_name": label,
                "null_lof_rate": p0,
                "n_split_variants": n_split,
                f"min_split_lof_for_binom_p_lt_{alpha:g}": min_k,
                f"min_split_lof_rate_for_binom_p_lt_{alpha:g}": None if min_k is None else min_k / n_split,
                "p_value_at_threshold": min_p,
            }
        )
    return pd.DataFrame(rows)


def endpoint_plan(panel: pd.DataFrame) -> pd.DataFrame:
    counts = panel.groupby("panel_arm").size().to_dict()
    return pd.DataFrame(
        [
            {
                "endpoint": "primary_binary_assay",
                "arms": "prospective_pathogenic_review vs prospective_benign_controls",
                "n_pathogenic_arm": counts.get("prospective_pathogenic_review", 0),
                "n_benign_arm": counts.get("prospective_benign_controls", 0),
                "test": "one-sided Fisher exact test",
                "success_rule": "External LOF fraction is significantly higher in pathogenic-review arm than benign-control arm.",
                "claim_enabled": "Explained BRCA2 old-VUS triage separates functional assay outcomes.",
            },
            {
                "endpoint": "secondary_continuous_assay",
                "arms": "prospective_pathogenic_review vs prospective_benign_controls",
                "n_pathogenic_arm": counts.get("prospective_pathogenic_review", 0),
                "n_benign_arm": counts.get("prospective_benign_controls", 0),
                "test": "Wilcoxon rank-sum or permutation test on quantitative HDR/function score",
                "success_rule": "Pathogenic-review arm has lower quantitative function than benign-control arm.",
                "claim_enabled": "Mechanism-guided prioritization tracks continuous BRCA2 function.",
            },
            {
                "endpoint": "split_mechanism_retention",
                "arms": "prospective_split_mechanism_tests",
                "n_pathogenic_arm": counts.get("prospective_split_mechanism_tests", 0),
                "n_benign_arm": 0,
                "test": "binomial enrichment against pre-specified background plus native-SAE qualitative split after embeddings finish",
                "success_rule": "Split-mechanism rows remain LOF-enriched and separate into DNA-side/protein-side explanation groups.",
                "claim_enabled": "Explanation distinguishes mechanism channels rather than only ranking variants.",
            },
            {
                "endpoint": "model_conflict_error_analysis",
                "arms": "prospective_model_conflict_controls",
                "n_pathogenic_arm": counts.get("prospective_model_conflict_controls", 0),
                "n_benign_arm": 0,
                "test": "descriptive error analysis",
                "success_rule": "Rows remain mostly functional, identifying contexts where checkpoint evidence overcalls risk.",
                "claim_enabled": "The method includes calibrated failure controls and does not overclaim all high-scoring variants.",
            },
        ]
    )


def write_report(
    path: Path,
    alpha: float,
    strict_alpha: float,
    n_pos: int,
    n_neg: int,
    thresholds: pd.DataFrame,
    power: pd.DataFrame,
    split_thresholds: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> None:
    common_thresholds = thresholds[thresholds["benign_control_lof_count"].isin([0, 1, 2, 3, 4])].copy()
    likely_power = power[
        power["pathogenic_target_true_lof_rate"].isin([0.7, 0.8, 0.9])
        & power["benign_control_true_lof_rate"].isin([0.0, 0.05, 0.10, 0.20])
    ].copy()
    lines = [
        "# BRCA2 Assay Statistical Analysis Plan",
        "",
        "## Purpose",
        "",
        "Pre-specify how the BRCA2 prospective follow-up panel would be evaluated if external manual review or functional assay outcomes are generated. This plan turns the interpretability application into testable arm-level claims.",
        "",
        "## Primary Comparison",
        "",
        f"- Pathogenic-review arm size: {n_pos}.",
        f"- Benign-control arm size: {n_neg}.",
        "- Primary endpoint: binary external functional classification, where LOF is counted as assay/pathogenic functional impairment.",
        f"- Primary test: one-sided Fisher exact test at alpha={alpha:g}.",
        f"- Strict reporting threshold: alpha={strict_alpha:g}.",
        "",
        "## Endpoint Plan",
        "",
        table(endpoints),
        "",
        "## Fisher Thresholds for Common Benign-Control Outcomes",
        "",
        table(common_thresholds),
        "",
        "## Power Grid",
        "",
        table(likely_power),
        "",
        "## Split-Mechanism Binomial Thresholds",
        "",
        table(split_thresholds),
        "",
        "## Interpretation",
        "",
        "- If benign controls have 0 LOF calls, the pathogenic-review arm needs 6/24 LOF calls for p<0.05 and 8/24 for p<0.01 by one-sided Fisher exact test.",
        "- If benign controls have 1 LOF call, the pathogenic-review arm needs 8/24 for p<0.05 and 11/24 for p<0.01.",
        "- Therefore the panel is powered for a strong but realistic effect, not for subtle differences.",
        "- The split-mechanism arm is a mechanistic stress test and should not be pooled into the primary Fisher comparison.",
        "- These tests support review/assay validation of the triage protocol; they still do not constitute clinical reclassification without expert curation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(require_file(args.panel))

    counts = panel.groupby("panel_arm").size().to_dict()
    n_pos = int(counts.get("prospective_pathogenic_review", 0))
    n_neg = int(counts.get("prospective_benign_controls", 0))
    n_split = int(counts.get("prospective_split_mechanism_tests", 0))
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("Primary pathogenic and benign arms must be non-empty.")

    thresholds = fisher_thresholds(n_pos, n_neg, args.alpha, args.strict_alpha)
    power = power_grid(n_pos, n_neg, args.alpha)
    split_thresholds = split_mechanism_thresholds(
        n_split,
        [
            ("brca2_missense_checkpoint_background", 763 / 4274),
            ("brca2_all_checkpoint_background", 1328 / 6198),
            ("conservative_20_percent_background", 0.20),
        ],
        args.alpha,
    )
    endpoints = endpoint_plan(panel)

    prefix = out / args.output_prefix
    thresholds.to_csv(prefix.with_name(prefix.name + "_fisher_thresholds.csv"), index=False)
    power.to_csv(prefix.with_name(prefix.name + "_power_grid.csv"), index=False)
    split_thresholds.to_csv(prefix.with_name(prefix.name + "_split_thresholds.csv"), index=False)
    endpoints.to_csv(prefix.with_name(prefix.name + "_endpoints.csv"), index=False)
    write_report(
        prefix.with_suffix(".md"),
        args.alpha,
        args.strict_alpha,
        n_pos,
        n_neg,
        thresholds,
        power,
        split_thresholds,
        endpoints,
    )
    print(f"Wrote {prefix.with_suffix('.md')}")
    print(thresholds.head(8).to_string(index=False))
    print(power.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
