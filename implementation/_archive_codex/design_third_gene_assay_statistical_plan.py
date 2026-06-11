#!/usr/bin/env python
"""Create a statistical analysis plan for the BAP1/RAD51C review/assay panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binom, fisher_exact


OUT_DIR = Path("results/interpretability_applications")


def one_sided_fisher(k_pos: int, n_pos: int, k_neg: int, n_neg: int) -> float:
    _, pval = fisher_exact([[k_pos, n_pos - k_pos], [k_neg, n_neg - k_neg]], alternative="greater")
    return float(pval)


def threshold_rows(comparison: str, n_pos: int, n_neg: int, alpha: float, strict_alpha: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                "comparison": comparison,
                "n_candidate_or_positive": n_pos,
                "n_negative_control": n_neg,
                "negative_control_positive_count": k_neg,
                "negative_control_positive_rate": k_neg / n_neg if n_neg else 0.0,
                f"min_candidate_positive_for_p_lt_{alpha:g}": min_alpha,
                f"min_candidate_positive_rate_for_p_lt_{alpha:g}": None if min_alpha is None else min_alpha / n_pos,
                f"min_candidate_positive_for_p_lt_{strict_alpha:g}": min_strict,
                f"min_candidate_positive_rate_for_p_lt_{strict_alpha:g}": None if min_strict is None else min_strict / n_pos,
            }
        )
    return rows


def exact_power(n_pos: int, n_neg: int, p_pos: float, p_neg: float, alpha: float) -> float:
    power = 0.0
    for k_pos in range(n_pos + 1):
        prob_pos = binom.pmf(k_pos, n_pos, p_pos)
        if prob_pos == 0:
            continue
        for k_neg in range(n_neg + 1):
            if one_sided_fisher(k_pos, n_pos, k_neg, n_neg) < alpha:
                power += prob_pos * binom.pmf(k_neg, n_neg, p_neg)
    return float(power)


def power_grid(comparison: str, n_pos: int, n_neg: int, alpha: float) -> pd.DataFrame:
    rows = []
    for p_pos in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for p_neg in [0.0, 0.05, 0.10, 0.20, 0.30]:
            rows.append(
                {
                    "comparison": comparison,
                    "candidate_true_positive_rate": p_pos,
                    "negative_control_true_positive_rate": p_neg,
                    "alpha": alpha,
                    "power": exact_power(n_pos, n_neg, p_pos, p_neg, alpha),
                }
            )
    return pd.DataFrame(rows)


def build_comparisons(panel: pd.DataFrame) -> list[dict[str, object]]:
    def n(arm: str) -> int:
        return int(panel["panel_arm"].eq(arm).sum())

    return [
        {
            "comparison": "BAP1_candidate_vs_benign_control",
            "candidate_arm": "BAP1_depleted_vus_review",
            "negative_arm": "BAP1_known_benign_controls",
            "n_candidate": n("BAP1_depleted_vus_review"),
            "n_negative": n("BAP1_known_benign_controls"),
            "endpoint": "manual pathogenic-review call or binary assay LOF call",
            "primary": True,
        },
        {
            "comparison": "RAD51C_candidate_vs_benign_control",
            "candidate_arm": "RAD51C_depleted_review",
            "negative_arm": "RAD51C_known_benign_controls",
            "n_candidate": n("RAD51C_depleted_review"),
            "n_negative": n("RAD51C_known_benign_controls"),
            "endpoint": "manual pathogenic-review call or binary assay LOF call",
            "primary": True,
        },
        {
            "comparison": "combined_candidate_vs_benign_control",
            "candidate_arm": "BAP1_depleted_vus_review + RAD51C_depleted_review",
            "negative_arm": "BAP1_known_benign_controls + RAD51C_known_benign_controls",
            "n_candidate": n("BAP1_depleted_vus_review") + n("RAD51C_depleted_review"),
            "n_negative": n("BAP1_known_benign_controls") + n("RAD51C_known_benign_controls"),
            "endpoint": "pooled manual pathogenic-review call or binary assay LOF call",
            "primary": True,
        },
        {
            "comparison": "BAP1_positive_control_sanity",
            "candidate_arm": "BAP1_known_pathogenic_controls",
            "negative_arm": "BAP1_known_benign_controls",
            "n_candidate": n("BAP1_known_pathogenic_controls"),
            "n_negative": n("BAP1_known_benign_controls"),
            "endpoint": "positive-control assay/review sanity check",
            "primary": False,
        },
        {
            "comparison": "RAD51C_positive_control_sanity",
            "candidate_arm": "RAD51C_known_pathogenic_controls",
            "negative_arm": "RAD51C_known_benign_controls",
            "n_candidate": n("RAD51C_known_pathogenic_controls"),
            "n_negative": n("RAD51C_known_benign_controls"),
            "endpoint": "positive-control assay/review sanity check",
            "primary": False,
        },
    ]


def design_sanity(panel: pd.DataFrame, comparisons: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    labels = pd.to_numeric(panel["label"], errors="coerce")
    for item in comparisons:
        candidate_arms = [x.strip() for x in str(item["candidate_arm"]).split("+")]
        negative_arms = [x.strip() for x in str(item["negative_arm"]).split("+")]
        pos = panel[panel["panel_arm"].isin(candidate_arms)]
        neg = panel[panel["panel_arm"].isin(negative_arms)]
        k_pos = int(pd.to_numeric(pos["label"], errors="coerce").fillna(0).sum())
        k_neg = int(pd.to_numeric(neg["label"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "comparison": item["comparison"],
                "design_candidate_sge_positive": k_pos,
                "design_candidate_n": int(len(pos)),
                "design_negative_sge_positive": k_neg,
                "design_negative_n": int(len(neg)),
                "design_one_sided_fisher_p": one_sided_fisher(k_pos, len(pos), k_neg, len(neg)) if len(pos) and len(neg) else None,
                "note": "uses internal SGE labels for design sanity only; not an external endpoint",
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def write_report(
    out_path: Path,
    alpha: float,
    strict_alpha: float,
    endpoints: pd.DataFrame,
    thresholds: pd.DataFrame,
    power: pd.DataFrame,
    sanity: pd.DataFrame,
) -> None:
    common_thresholds = thresholds[thresholds["negative_control_positive_count"].isin([0, 1, 2, 3])].copy()
    likely_power = power[
        power["candidate_true_positive_rate"].isin([0.7, 0.8, 0.9])
        & power["negative_control_true_positive_rate"].isin([0.0, 0.05, 0.10, 0.20])
    ].copy()
    lines = [
        "# Third-Gene Review/Assay Statistical Plan",
        "",
        "## Purpose",
        "",
        "Pre-specify how the BAP1/RAD51C 64-row panel should be evaluated once blinded expert review calls or assay readouts are returned. This protects the third-gene application from post-hoc threshold selection.",
        "",
        "## Endpoint Plan",
        "",
        md_table(endpoints),
        "",
        "## Binary Success Thresholds",
        "",
        f"Primary one-sided Fisher exact alpha={alpha:g}; strict reporting threshold={strict_alpha:g}. Counts below are minimum candidate/review-positive or assay-LOF calls needed for each possible negative-control positive count.",
        "",
        md_table(common_thresholds),
        "",
        "## Power Grid",
        "",
        md_table(likely_power, max_rows=40),
        "",
        "## Design Sanity Only",
        "",
        "The table below uses internal SGE labels already used for panel design. It is not an external validation result and must not be reported as a locked endpoint.",
        "",
        md_table(sanity),
        "",
        "## Claim Boundary",
        "",
        "Passing this plan would support review/assay prioritization for BAP1/RAD51C. It would still not by itself prove third-gene CrossBioSAE generalization; that requires Evo2/ESM checkpoint metrics and preferably native-SAE intervention.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--strict-alpha", type=float, default=0.01)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    panel_path = out / "third_gene_review_assay_panel_internal.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing; run design_third_gene_review_assay_panel.py first")
    panel = pd.read_csv(panel_path)
    comparisons = build_comparisons(panel)
    endpoints = pd.DataFrame(comparisons)

    threshold_parts = []
    power_parts = []
    for item in comparisons:
        threshold_parts.extend(
            threshold_rows(
                str(item["comparison"]),
                int(item["n_candidate"]),
                int(item["n_negative"]),
                args.alpha,
                args.strict_alpha,
            )
        )
        power_parts.append(
            power_grid(
                str(item["comparison"]),
                int(item["n_candidate"]),
                int(item["n_negative"]),
                args.alpha,
            )
        )

    thresholds = pd.DataFrame(threshold_parts)
    power = pd.concat(power_parts, ignore_index=True)
    sanity = design_sanity(panel, comparisons)

    endpoints.to_csv(out / "third_gene_assay_endpoint_plan.csv", index=False)
    thresholds.to_csv(out / "third_gene_assay_statistical_thresholds.csv", index=False)
    power.to_csv(out / "third_gene_assay_power_grid.csv", index=False)
    sanity.to_csv(out / "third_gene_assay_design_sanity.csv", index=False)
    report = out / "third_gene_assay_statistical_plan.md"
    write_report(report, args.alpha, args.strict_alpha, endpoints, thresholds, power, sanity)
    print(f"wrote {out / 'third_gene_assay_endpoint_plan.csv'}")
    print(f"wrote {out / 'third_gene_assay_statistical_thresholds.csv'}")
    print(f"wrote {out / 'third_gene_assay_power_grid.csv'}")
    print(f"wrote {out / 'third_gene_assay_design_sanity.csv'}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
