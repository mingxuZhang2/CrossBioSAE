#!/usr/bin/env python3
"""Design a compact BRCA2 interpretability review/assay panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root
        / "results"
        / "interpretability_applications"
        / "brca2_clinvar_interpretability_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_interpretability_review_panel")
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


def select_panel(candidates: pd.DataFrame, tier: str, quota: int, arm: str) -> pd.DataFrame:
    sub = candidates[candidates["review_tier"].eq(tier)].copy()
    if sub.empty:
        return sub
    if "lof" in tier:
        sub["_function_sort"] = sub["function_score"].astype(float)
    else:
        sub["_function_sort"] = -sub["function_score"].astype(float)
    sub["_hotspot_sort"] = sub["hotspot_n_variants"].fillna(0)
    sub["_same_aa_path_sort"] = sub.get("same_aa_known_pathogenic", 0)
    sub["_same_aa_benign_sort"] = sub.get("same_aa_known_benign", 0)
    sub["_same_path_sort"] = sub["same_residue_known_pathogenic"].fillna(0)
    sub["_same_benign_sort"] = sub["same_residue_known_benign"].fillna(0)
    sub["_dna_sort"] = sub["dna_percentile"].fillna(0)
    sub["_protein_sort"] = sub["protein_percentile"].fillna(0)
    if tier == "tier1_functional_benign_concordant_low":
        sort_cols = ["_same_aa_benign_sort", "_same_benign_sort", "_hotspot_sort", "_function_sort", "_dna_sort", "_protein_sort"]
        ascending = [False, False, False, True, True, True]
    elif tier == "tier2_functional_benign_model_high_conflict":
        sort_cols = ["_same_aa_path_sort", "_same_path_sort", "_same_aa_benign_sort", "_same_benign_sort", "_hotspot_sort", "_function_sort", "_dna_sort", "_protein_sort"]
        ascending = [False, False, False, False, False, True, False, False]
    else:
        sort_cols = ["_same_aa_path_sort", "_same_path_sort", "_hotspot_sort", "_function_sort", "_dna_sort", "_protein_sort"]
        ascending = [False, False, False, True, False, False]
    out = sub.sort_values(sort_cols, ascending=ascending).head(quota).copy()
    out["panel_arm"] = arm
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, sub in panel.groupby("panel_arm", sort=False):
        def count_positive(col: str) -> int:
            if col not in sub.columns:
                return 0
            return int(sub[col].fillna(0).gt(0).sum())

        rows.append(
            {
                "panel_arm": arm,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "n_uncertain": int(sub["clinvar_simple"].eq("Uncertain significance").sum()),
                "n_conflicting": int(sub["clinvar_simple"].eq("Conflicting interpretations of pathogenicity").sum()),
                "n_domains": int(sub["brca2_domain"].nunique()),
                "n_recurrent_hotspot": int(sub["hotspot_n_variants"].fillna(0).ge(2).sum()),
                "n_same_aa_pathogenic": count_positive("same_aa_known_pathogenic"),
                "n_same_aa_benign": count_positive("same_aa_known_benign"),
                "n_same_residue_other_pathogenic": count_positive("same_residue_other_known_pathogenic"),
                "n_same_residue_other_benign": count_positive("same_residue_other_known_benign"),
                "n_same_residue_pathogenic": int(sub["same_residue_known_pathogenic"].fillna(0).gt(0).sum()),
                "n_same_residue_benign": int(sub["same_residue_known_benign"].fillna(0).gt(0).sum()),
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
        (
            "tier1_functional_lof_concordant_high",
            args.pathogenic_quota,
            "pathogenic_review_high_confidence",
        ),
        (
            "tier1_functional_benign_concordant_low",
            args.benign_quota,
            "benign_review_controls",
        ),
        (
            "tier2_functional_lof_split_mechanism",
            args.split_quota,
            "split_mechanism_tests",
        ),
        (
            "tier2_functional_benign_model_high_conflict",
            args.conflict_quota,
            "model_high_sge_benign_conflicts",
        ),
    ]
    panel = pd.concat([select_panel(candidates, tier, quota, arm) for tier, quota, arm in arms], ignore_index=True)
    panel["panel_rank"] = np.arange(1, len(panel) + 1)
    summary = build_summary(panel)

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
        "hotspot_n_variants",
        "hotspot_n_lof",
        "hotspot_lof_rate",
        "same_aa_known_pathogenic",
        "same_aa_known_benign",
        "same_aa_known_pathogenic_changes",
        "same_aa_known_benign_changes",
        "same_residue_other_known_pathogenic",
        "same_residue_other_known_benign",
        "same_residue_other_known_pathogenic_changes",
        "same_residue_other_known_benign_changes",
        "same_residue_known_pathogenic",
        "same_residue_known_benign",
        "same_residue_known_pathogenic_changes",
        "same_residue_known_benign_changes",
        "c.nom",
        "g.nom",
        "dbSNP.ID",
    ]
    prefix = args.output_prefix
    panel[cols].to_csv(out / f"{prefix}.csv", index=False)
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)

    report = [
        "# BRCA2 Interpretability Review Panel",
        "",
        "## Purpose",
        "",
        "A compact panel for manual review or follow-up assay design. It is selected from BRCA2 ClinVar VUS/conflicting missense candidates using SGE function, ESM/Evo2 LLR mechanism concordance, recurrent residue hotspots, exact same-AA-change ClinVar evidence, and same-residue-other ClinVar evidence.",
        "",
        "## Panel Arms",
        "",
        table(summary),
        "",
        "## Panel Variants",
        "",
        table(panel[cols]),
        "",
        "## Interpretation",
        "",
        "- Pathogenic-review high-confidence rows test whether concordant protein+DNA mechanism evidence marks VUS/conflicting variants that already have SGE LOF support.",
        "- Benign-review controls test whether concordant low mechanism evidence marks functional SGE variants and provides negative controls.",
        "- Split-mechanism rows are the stress test for the eventual full cross-modal/native-SAE analysis: the explanation should say why protein and DNA channels disagree.",
        "- Model-high/SGE-benign conflict rows are error-analysis controls and should be used to bound overclaiming.",
        "",
    ]
    (out / f"{prefix}.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
