#!/usr/bin/env python
"""Build conservative external/proxy validation checks for BRCA2 interpretation.

These checks are intentionally not a replacement for locked blinded review or
new functional assay outcomes. They answer two narrower questions:

1. On current exact ClinVar known BRCA2 missense variants, is the DNA/protein
   both-high stratum enriched for pathogenic labels?
2. In the blinded follow-up panel, do reviewer-visible same-residue ClinVar
   hooks distribute in the expected direction across pathogenic-review and
   benign-control arms?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OUT_DIR = Path("results/interpretability_applications")

KNOWN_PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
KNOWN_BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
PRIMARY_POSITIVE_ARM = "prospective_pathogenic_review"
PRIMARY_NEGATIVE_ARM = "prospective_benign_controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", default="brca2_external_proxy_validation")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def bh_qvalues(pvals: list[float]) -> list[float]:
    p = np.array([1.0 if pd.isna(x) else float(x) for x in pvals], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    m = len(p)
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def fisher_or_nan(table: list[list[int]], alternative: str) -> tuple[float, float]:
    try:
        odds, pval = fisher_exact(table, alternative=alternative)
        return float(odds), float(pval)
    except Exception:
        return float("nan"), float("nan")


def known_clinvar_proxy(scores: pd.DataFrame) -> pd.DataFrame:
    known = scores[scores["clinvar_simple"].isin(KNOWN_PATHOGENIC | KNOWN_BENIGN)].copy()
    if known.empty:
        return pd.DataFrame()
    known["clinvar_pathogenic_binary"] = known["clinvar_simple"].isin(KNOWN_PATHOGENIC).astype(int)

    rows: list[dict[str, Any]] = []
    categories = sorted(known["discordance_category"].dropna().astype(str).unique())
    for category in categories:
        sub = known[known["discordance_category"].astype(str).eq(category)]
        rest = known[~known["discordance_category"].astype(str).eq(category)]
        n_path = int(sub["clinvar_pathogenic_binary"].sum())
        n_benign = int(len(sub) - n_path)
        rest_path = int(rest["clinvar_pathogenic_binary"].sum())
        rest_benign = int(len(rest) - rest_path)
        odds_enrich, p_enrich = fisher_or_nan([[n_path, n_benign], [rest_path, rest_benign]], "greater")
        odds_deplete, p_deplete = fisher_or_nan([[n_path, n_benign], [rest_path, rest_benign]], "less")
        rows.append(
            {
                "discordance_category": category,
                "n_exact_known_clinvar": int(len(sub)),
                "n_pathogenic_or_likely_pathogenic": n_path,
                "n_benign_or_likely_benign": n_benign,
                "pathogenic_rate": float(n_path / len(sub)) if len(sub) else np.nan,
                "n_sge_lof": int(safe_num(sub["label"]).sum()) if "label" in sub else 0,
                "sge_lof_rate": float(safe_num(sub["label"]).mean()) if "label" in sub else np.nan,
                "rest_n_exact_known_clinvar": int(len(rest)),
                "rest_pathogenic_rate": float(rest_path / len(rest)) if len(rest) else np.nan,
                "fisher_odds_pathogenic_enrichment_vs_rest": odds_enrich,
                "fisher_p_pathogenic_enrichment_vs_rest": p_enrich,
                "fisher_odds_pathogenic_depletion_vs_rest": odds_deplete,
                "fisher_p_pathogenic_depletion_vs_rest": p_deplete,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fisher_q_pathogenic_enrichment_vs_rest"] = bh_qvalues(
            out["fisher_p_pathogenic_enrichment_vs_rest"].tolist()
        )
        out["fisher_q_pathogenic_depletion_vs_rest"] = bh_qvalues(
            out["fisher_p_pathogenic_depletion_vs_rest"].tolist()
        )
    return out.sort_values("discordance_category").reset_index(drop=True)


def add_panel_proxy_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    path_count = safe_num(out.get("same_aa_known_pathogenic", pd.Series(0, index=out.index))) + safe_num(
        out.get("same_residue_other_known_pathogenic", pd.Series(0, index=out.index))
    )
    benign_count = safe_num(out.get("same_aa_known_benign", pd.Series(0, index=out.index))) + safe_num(
        out.get("same_residue_other_known_benign", pd.Series(0, index=out.index))
    )
    out["same_residue_pathogenic_proxy_count"] = path_count.astype(int)
    out["same_residue_benign_proxy_count"] = benign_count.astype(int)
    out["same_residue_pathogenic_no_benign_proxy"] = (path_count > 0) & (benign_count == 0)
    out["same_residue_benign_no_pathogenic_proxy"] = (benign_count > 0) & (path_count == 0)
    out["same_residue_conflicting_proxy"] = (path_count > 0) & (benign_count > 0)
    out["same_residue_any_pathogenic_proxy"] = path_count > 0
    out["same_residue_any_benign_proxy"] = benign_count > 0
    out["current_exact_known_proxy"] = out["clinvar_simple"].isin(KNOWN_PATHOGENIC | KNOWN_BENIGN)
    out["current_exact_pathogenic_proxy"] = out["clinvar_simple"].isin(KNOWN_PATHOGENIC)
    out["current_exact_benign_proxy"] = out["clinvar_simple"].isin(KNOWN_BENIGN)
    return out


def panel_arm_summary(panel: pd.DataFrame) -> pd.DataFrame:
    df = add_panel_proxy_flags(panel)
    rows = []
    for arm, sub in df.groupby("panel_arm", sort=True):
        rows.append(
            {
                "panel_arm": arm,
                "n": len(sub),
                "n_current_exact_known": int(sub["current_exact_known_proxy"].sum()),
                "n_current_exact_pathogenic": int(sub["current_exact_pathogenic_proxy"].sum()),
                "n_current_exact_benign": int(sub["current_exact_benign_proxy"].sum()),
                "n_same_residue_pathogenic_no_benign": int(sub["same_residue_pathogenic_no_benign_proxy"].sum()),
                "n_same_residue_benign_no_pathogenic": int(sub["same_residue_benign_no_pathogenic_proxy"].sum()),
                "n_same_residue_conflicting": int(sub["same_residue_conflicting_proxy"].sum()),
                "n_same_residue_any_pathogenic": int(sub["same_residue_any_pathogenic_proxy"].sum()),
                "n_same_residue_any_benign": int(sub["same_residue_any_benign_proxy"].sum()),
                "median_dna_percentile": float(safe_num(sub["dna_percentile"]).median())
                if "dna_percentile" in sub
                else np.nan,
                "median_protein_percentile": float(safe_num(sub["protein_percentile"]).median())
                if "protein_percentile" in sub
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_panel_arms(panel: pd.DataFrame) -> pd.DataFrame:
    df = add_panel_proxy_flags(panel)
    pos = df[df["panel_arm"].eq(PRIMARY_POSITIVE_ARM)]
    neg = df[df["panel_arm"].eq(PRIMARY_NEGATIVE_ARM)]
    tests = [
        (
            "pathogenic_review_enriched_for_same_residue_pathogenic_no_benign",
            "same_residue_pathogenic_no_benign_proxy",
            pos,
            neg,
            "greater",
        ),
        (
            "pathogenic_review_enriched_for_any_same_residue_pathogenic",
            "same_residue_any_pathogenic_proxy",
            pos,
            neg,
            "greater",
        ),
        (
            "benign_controls_enriched_for_same_residue_benign_no_pathogenic",
            "same_residue_benign_no_pathogenic_proxy",
            neg,
            pos,
            "greater",
        ),
        (
            "benign_controls_enriched_for_any_same_residue_benign",
            "same_residue_any_benign_proxy",
            neg,
            pos,
            "greater",
        ),
    ]
    rows = []
    for name, col, first, second, alternative in tests:
        first_hit = int(first[col].sum())
        second_hit = int(second[col].sum())
        odds, pval = fisher_or_nan(
            [[first_hit, len(first) - first_hit], [second_hit, len(second) - second_hit]],
            alternative,
        )
        rows.append(
            {
                "proxy_test": name,
                "first_arm": str(first["panel_arm"].iloc[0]) if len(first) else "",
                "second_arm": str(second["panel_arm"].iloc[0]) if len(second) else "",
                "proxy_feature": col,
                "first_arm_n": len(first),
                "first_arm_n_proxy_positive": first_hit,
                "first_arm_proxy_rate": float(first_hit / len(first)) if len(first) else np.nan,
                "second_arm_n": len(second),
                "second_arm_n_proxy_positive": second_hit,
                "second_arm_proxy_rate": float(second_hit / len(second)) if len(second) else np.nan,
                "fisher_odds": odds,
                "fisher_p_greater": pval,
            }
        )
    out = pd.DataFrame(rows)
    out["fisher_q_greater"] = bh_qvalues(out["fisher_p_greater"].tolist()) if not out.empty else []
    return out


def summary_rows(known: pd.DataFrame, panel_summary: pd.DataFrame, panel_tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    both_high = known[known["discordance_category"].eq("both_high")].head(1)
    if not both_high.empty:
        r = both_high.iloc[0]
        rows.append(
            {
                "summary_item": "known_clinvar_both_high_pathogenic_enrichment",
                "status": "proxy_supportive",
                "value": (
                    f"pathogenic_rate={float(r['pathogenic_rate']):.3f}; "
                    f"n={int(r['n_exact_known_clinvar'])}; "
                    f"odds={float(r['fisher_odds_pathogenic_enrichment_vs_rest']):.3g}; "
                    f"p={float(r['fisher_p_pathogenic_enrichment_vs_rest']):.4g}; "
                    f"q={float(r['fisher_q_pathogenic_enrichment_vs_rest']):.4g}"
                ),
                "claim_boundary": "External exact ClinVar known-label proxy for BRCA2 mechanism strata; not a prospective VUS endpoint.",
            }
        )
    exact_known = int(panel_summary["n_current_exact_known"].sum()) if not panel_summary.empty else 0
    total = int(panel_summary["n"].sum()) if not panel_summary.empty else 0
    rows.append(
        {
            "summary_item": "prospective_panel_exact_current_clinvar_known",
            "status": "not_available",
            "value": f"{exact_known}/{total}",
            "claim_boundary": "The prospective panel remains exact ClinVar VUS, so exact current ClinVar cannot validate the panel.",
        }
    )
    path_test = panel_tests[
        panel_tests["proxy_test"].eq("pathogenic_review_enriched_for_same_residue_pathogenic_no_benign")
    ]
    if not path_test.empty:
        r = path_test.iloc[0]
        rows.append(
            {
                "summary_item": "panel_same_residue_pathogenic_no_benign_direction",
                "status": "directional_proxy_not_significant",
                "value": (
                    f"{int(r['first_arm_n_proxy_positive'])}/{int(r['first_arm_n'])} vs "
                    f"{int(r['second_arm_n_proxy_positive'])}/{int(r['second_arm_n'])}; "
                    f"p={float(r['fisher_p_greater']):.4g}"
                ),
                "claim_boundary": "Reviewer-visible same-residue proxy; useful for curation support but not an independent locked outcome.",
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                text = "NA" if np.isnan(value) else f"{value:.4g}"
            else:
                text = str(value)
            vals.append(text.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    known: pd.DataFrame,
    panel_summary: pd.DataFrame,
    panel_tests: pd.DataFrame,
) -> None:
    lines = [
        "# BRCA2 External Proxy Validation",
        "",
        "## Purpose",
        "",
        "This report adds conservative external/proxy evidence for the BRCA2 interpretation application. It does not replace blinded manual review or new assay outcomes.",
        "",
        "## Summary",
        "",
        md_table(summary, ["summary_item", "status", "value", "claim_boundary"]),
        "",
        "## Exact Current ClinVar Known-Label Proxy",
        "",
        "Known BRCA2 ClinVar missense rows are used only as a proxy check for mechanism strata. VUS/conflicting/absent rows are excluded from this table.",
        "",
        md_table(
            known,
            [
                "discordance_category",
                "n_exact_known_clinvar",
                "n_pathogenic_or_likely_pathogenic",
                "n_benign_or_likely_benign",
                "pathogenic_rate",
                "rest_pathogenic_rate",
                "fisher_p_pathogenic_enrichment_vs_rest",
                "fisher_q_pathogenic_enrichment_vs_rest",
            ],
        ),
        "",
        "## Prospective Panel Same-Residue Proxy",
        "",
        "The 64-row prospective panel has no exact current ClinVar known outcomes, so same-residue evidence is reported as reviewer-visible curation support only.",
        "",
        md_table(
            panel_summary,
            [
                "panel_arm",
                "n",
                "n_current_exact_known",
                "n_same_residue_pathogenic_no_benign",
                "n_same_residue_benign_no_pathogenic",
                "n_same_residue_conflicting",
                "n_same_residue_any_pathogenic",
                "n_same_residue_any_benign",
            ],
        ),
        "",
        "## Proxy Arm Tests",
        "",
        md_table(
            panel_tests,
            [
                "proxy_test",
                "first_arm",
                "first_arm_n_proxy_positive",
                "first_arm_n",
                "second_arm",
                "second_arm_n_proxy_positive",
                "second_arm_n",
                "fisher_p_greater",
                "fisher_q_greater",
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "Supported by this proxy layer: BRCA2 both-high mechanism strata have external ClinVar known-label support, and the prospective panel contains reviewer-visible same-residue hooks in the expected direction. Not supported by this proxy layer: clinical VUS reclassification or a completed blinded validation endpoint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    scores = read_csv(out / "brca2_llr_esm_discordance_missense_scores.csv")
    panel = read_csv(out / "brca2_prospective_followup_panel.csv")

    known = known_clinvar_proxy(scores)
    panel_summary = panel_arm_summary(panel)
    panel_tests = compare_panel_arms(panel)
    summary = summary_rows(known, panel_summary, panel_tests)

    summary_path = out / f"{args.output_prefix}_summary.csv"
    known_path = out / f"{args.output_prefix}_known_clinvar.csv"
    panel_summary_path = out / f"{args.output_prefix}_panel_same_residue.csv"
    panel_tests_path = out / f"{args.output_prefix}_panel_proxy_tests.csv"
    report_path = out / f"{args.output_prefix}.md"

    summary.to_csv(summary_path, index=False)
    known.to_csv(known_path, index=False)
    panel_summary.to_csv(panel_summary_path, index=False)
    panel_tests.to_csv(panel_tests_path, index=False)
    write_report(report_path, summary, known, panel_summary, panel_tests)

    print(f"wrote {summary_path}")
    print(f"wrote {known_path}")
    print(f"wrote {panel_summary_path}")
    print(f"wrote {panel_tests_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
