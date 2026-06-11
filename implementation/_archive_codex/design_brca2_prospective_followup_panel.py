#!/usr/bin/env python3
"""Design a BRCA2 follow-up panel restricted to old VUS/conflicting candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OLD_UNRESOLVED = {"old_clean_vus", "old_conflicting", "old_vus_mixed", "old_absent"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root
        / "results"
        / "interpretability_applications"
        / "brca2_clinvar_temporal_context_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_prospective_followup_panel")
    parser.add_argument("--pathogenic-quota", type=int, default=24)
    parser.add_argument("--benign-quota", type=int, default=16)
    parser.add_argument("--split-quota", type=int, default=12)
    parser.add_argument("--conflict-quota", type=int, default=12)
    return parser.parse_args()


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def count_positive(sub: pd.DataFrame, col: str) -> int:
    if col not in sub.columns:
        return 0
    return int(sub[col].fillna(0).gt(0).sum())


def select_panel(candidates: pd.DataFrame, tier: str, quota: int, arm: str) -> pd.DataFrame:
    sub = candidates[
        candidates["review_tier"].eq(tier)
        & candidates["old_status_class"].isin(OLD_UNRESOLVED)
    ].copy()
    if sub.empty:
        return sub

    sub["_old_clean_sort"] = sub["old_status_class"].eq("old_clean_vus").astype(int)
    sub["_old_conflict_sort"] = sub["old_status_class"].eq("old_conflicting").astype(int)
    sub["_hotspot_sort"] = sub["hotspot_n_variants"].fillna(0)
    sub["_same_aa_path_sort"] = sub.get("same_aa_known_pathogenic", 0)
    sub["_same_aa_benign_sort"] = sub.get("same_aa_known_benign", 0)
    sub["_same_path_sort"] = sub["same_residue_known_pathogenic"].fillna(0)
    sub["_same_benign_sort"] = sub["same_residue_known_benign"].fillna(0)
    sub["_dna_sort"] = sub["dna_percentile"].fillna(0)
    sub["_protein_sort"] = sub["protein_percentile"].fillna(0)
    if "lof" in tier:
        sub["_function_sort"] = sub["function_score"].astype(float)
    else:
        sub["_function_sort"] = -sub["function_score"].astype(float)

    if tier == "tier1_functional_benign_concordant_low":
        sort_cols = [
            "_old_clean_sort",
            "_same_aa_benign_sort",
            "_same_benign_sort",
            "_hotspot_sort",
            "_function_sort",
            "_dna_sort",
            "_protein_sort",
        ]
        ascending = [False, False, False, False, True, True, True]
    elif tier == "tier2_functional_benign_model_high_conflict":
        sort_cols = [
            "_old_clean_sort",
            "_same_path_sort",
            "_same_benign_sort",
            "_hotspot_sort",
            "_function_sort",
            "_dna_sort",
            "_protein_sort",
        ]
        ascending = [False, False, False, False, True, False, False]
    else:
        sort_cols = [
            "_old_clean_sort",
            "_same_aa_path_sort",
            "_same_path_sort",
            "_hotspot_sort",
            "_function_sort",
            "_dna_sort",
            "_protein_sort",
        ]
        ascending = [False, False, False, False, True, False, False]

    out = sub.sort_values(sort_cols, ascending=ascending).head(quota).copy()
    out["panel_arm"] = arm
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, sub in panel.groupby("panel_arm", sort=False):
        rows.append(
            {
                "panel_arm": arm,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "n_old_clean_vus": int(sub["old_status_class"].eq("old_clean_vus").sum()),
                "n_old_conflicting": int(sub["old_status_class"].eq("old_conflicting").sum()),
                "n_domains": int(sub["brca2_domain"].nunique()),
                "n_recurrent_hotspot": int(sub["hotspot_n_variants"].fillna(0).ge(2).sum()),
                "n_same_aa_pathogenic": count_positive(sub, "same_aa_known_pathogenic"),
                "n_same_residue_other_pathogenic": count_positive(sub, "same_residue_other_known_pathogenic"),
                "n_same_residue_other_benign": count_positive(sub, "same_residue_other_known_benign"),
                "median_function_score": float(sub["function_score"].median()),
                "median_dna_percentile": float(sub["dna_percentile"].median()),
                "median_protein_percentile": float(sub["protein_percentile"].median()),
            }
        )
    return pd.DataFrame(rows)


def domain_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, domain), sub in panel.groupby(["panel_arm", "brca2_domain"], sort=False):
        rows.append(
            {
                "panel_arm": arm,
                "brca2_domain": domain,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "old_clean_vus": int(sub["old_status_class"].eq("old_clean_vus").sum()),
                "old_conflicting": int(sub["old_status_class"].eq("old_conflicting").sum()),
                "median_function_score": float(sub["function_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)

    arms = [
        ("tier1_functional_lof_concordant_high", args.pathogenic_quota, "prospective_pathogenic_review"),
        ("tier1_functional_benign_concordant_low", args.benign_quota, "prospective_benign_controls"),
        ("tier2_functional_lof_split_mechanism", args.split_quota, "prospective_split_mechanism_tests"),
        ("tier2_functional_benign_model_high_conflict", args.conflict_quota, "prospective_model_conflict_controls"),
    ]
    panel = pd.concat([select_panel(candidates, tier, quota, arm) for tier, quota, arm in arms], ignore_index=True)
    panel["panel_rank"] = np.arange(1, len(panel) + 1)
    summary = build_summary(panel)
    domains = domain_summary(panel)

    sanity = candidates[
        candidates["review_tier"].isin(
            ["tier1_functional_lof_concordant_high", "tier2_functional_benign_model_high_conflict"]
        )
        & candidates["old_status_class"].eq("old_pathogenic_or_likely_pathogenic")
    ].copy()

    cols = [
        "panel_rank",
        "panel_arm",
        "review_tier",
        "review_direction",
        "id",
        "AA.change",
        "brca2_domain",
        "clinvar_simple",
        "label",
        "func_class",
        "function_score",
        "discordance_category",
        "dna_percentile",
        "protein_percentile",
        "old_status_class",
        "old_clinsig",
        "old_review_status",
        "old_last_evaluated",
        "old_number_submitters_max",
        "hotspot_n_variants",
        "hotspot_n_lof",
        "hotspot_lof_rate",
        "same_aa_known_pathogenic",
        "same_aa_known_benign",
        "same_aa_known_pathogenic_changes",
        "same_residue_other_known_pathogenic",
        "same_residue_other_known_benign",
        "same_residue_other_known_pathogenic_changes",
        "same_residue_other_known_benign_changes",
        "c.nom",
        "g.nom",
        "dbSNP.ID",
    ]
    sanity_cols = [c for c in cols if c in sanity.columns and c != "panel_rank"]
    panel[cols].to_csv(out / f"{args.output_prefix}.csv", index=False)
    summary.to_csv(out / f"{args.output_prefix}_summary.csv", index=False)
    domains.to_csv(out / f"{args.output_prefix}_domain_summary.csv", index=False)
    sanity[sanity_cols].to_csv(out / f"{args.output_prefix}_old_pathogenic_sanity_checks.csv", index=False)

    report = [
        "# BRCA2 Prospective Follow-up Panel",
        "",
        "## Purpose",
        "",
        "A stricter BRCA2 follow-up panel restricted to candidates that were unresolved in archived ClinVar 2025-01 (`old_clean_vus`, `old_conflicting`, `old_vus_mixed`, or absent). This separates prospective-style review targets from old P/LP sanity checks.",
        "",
        "## Panel Arms",
        "",
        table(summary),
        "",
        "## Domain Coverage",
        "",
        table(domains),
        "",
        "## Panel Variants",
        "",
        table(panel[cols]),
        "",
        "## Old P/LP-like Sanity Checks Excluded from Prospective Panel",
        "",
        table(sanity[sanity_cols], max_rows=20),
        "",
        "## Interpretation",
        "",
        "- This is the cleanest BRCA2 application panel so far because it removes variants that were already P/LP-like in the old ClinVar archive from the primary follow-up set.",
        "- The pathogenic and split-mechanism arms are the main assay targets; the benign and model-conflict arms are negative/error-analysis controls.",
        "- The panel is still not clinical reclassification. It is a review/assay design supported by SGE function, model-mechanism concordance or discordance, and archived ClinVar status.",
        "",
    ]
    (out / f"{args.output_prefix}.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
