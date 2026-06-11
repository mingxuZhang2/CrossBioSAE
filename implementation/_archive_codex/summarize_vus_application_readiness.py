#!/usr/bin/env python3
"""
Summarize the VUS downstream application readiness.

This consolidates the collagen/structural VUS triage artifacts into a stricter
evidence table that separates clinical-review hooks from hypothesis-only model
signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument("--output-dir", type=Path, default=root / "results" / "interpretability_applications")
    p.add_argument("--top-n", type=int, default=40)
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def fmt_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def evidence_level(row: pd.Series) -> str:
    nonsyn = not bool(row.get("is_synonymous_pchange", False))
    same_path = int(row.get("same_residue_pathogenic_count", 0)) > 0
    same_benign = int(row.get("same_residue_benign_count", 0)) > 0
    feature_path = float(row.get("feature_known_path_rate", 0.0))
    feature_cg = int(row.get("feature_known_collagen_gly_pathogenic_count", 0))
    tier = str(row.get("candidate_tier", ""))
    gene_context = bool(row.get("has_gene_disease_id", False)) or bool(row.get("has_lsdb_link", False))

    if nonsyn and same_path and not same_benign:
        return "A_same_residue_pathogenic_no_benign"
    if (
        nonsyn
        and tier.startswith("T1_")
        and feature_path >= 0.85
        and feature_cg >= 10
        and gene_context
    ):
        return "B_T1_feature_known_pathogenic_gene_context"
    if nonsyn and tier.startswith("T1_"):
        return "C_T1_model_mechanism_only"
    if nonsyn and feature_path >= 0.85 and feature_cg >= 10:
        return "D_feature_only_review"
    return "E_low_support_or_synonymous"


def support_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["application_evidence_level"] = out.apply(evidence_level, axis=1)
    out["is_actionable_review_hook"] = out["application_evidence_level"].isin(
        [
            "A_same_residue_pathogenic_no_benign",
            "B_T1_feature_known_pathogenic_gene_context",
        ]
    )
    out["has_common_negative_warning"] = False
    return out


def summary_counts(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_candidates": np.ones(len(df), dtype=bool),
        "tier1_collagen_gly": df["candidate_tier"].astype(str).str.startswith("T1_").to_numpy(),
        "strict_nonsyn_collagen_gly": (
            df["is_gly_sub_strict"].astype(bool) & ~df["is_synonymous_pchange"].astype(bool)
        ).to_numpy(),
    }
    rows = []
    for group, mask in groups.items():
        sub = df[mask]
        rows.append(
            {
                "group": group,
                "n": len(sub),
                "n_actionable_review_hooks": int(sub["is_actionable_review_hook"].sum()),
                "n_A_same_residue_pathogenic": int(
                    sub["application_evidence_level"].eq("A_same_residue_pathogenic_no_benign").sum()
                ),
                "n_B_feature_gene_context": int(
                    sub["application_evidence_level"].eq("B_T1_feature_known_pathogenic_gene_context").sum()
                ),
                "n_C_T1_model_only": int(sub["application_evidence_level"].eq("C_T1_model_mechanism_only").sum()),
                "n_same_residue_benign": int((sub["same_residue_benign_count"] > 0).sum()),
                "median_collagen_gly_score": float(sub["collagen_gly_feature_score"].median()) if len(sub) else float("nan"),
                "p90_collagen_gly_score": float(sub["collagen_gly_feature_score"].quantile(0.9)) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def control_summary(out_dir: Path) -> pd.DataFrame:
    pair = read_csv(out_dir / "collagen_gly_pairwise_controls.csv")
    rand_known = read_csv(out_dir / "collagen_gly_random_feature_null_summary.csv")
    rand_vus = read_csv(out_dir / "collagen_gly_vus_random_feature_summary.csv")
    matched = read_csv(out_dir / "collagen_gly_matched_control_summary.csv")

    rows = []
    for _, r in pair.iterrows():
        if r["control"] in {"benign_all_mapped", "pathogenic_nonstructural_missense", "benign_structural_missense"}:
            rows.append(
                {
                    "control_type": f"known_case_vs_{r['control']}",
                    "n_case": int(r["n_case"]),
                    "n_control": int(r["n_control"]),
                    "effect": float(r["delta_mean"]),
                    "auroc_or_p": float(r["auroc_case_vs_control"]),
                    "p_value": float(r["mannwhitney_greater_p"]),
                    "interpretation": "known pathogenic collagen Gly feature specificity",
                }
            )
    if not matched.empty:
        r = matched.iloc[0]
        rows.append(
            {
                "control_type": "known_case_vs_baseline_matched_pathogenic",
                "n_case": int(r["n_pairs"]),
                "n_control": int(r["unique_control_variants"]),
                "effect": float(r["mean_delta"]),
                "auroc_or_p": float(r["frac_case_greater"]),
                "p_value": float(r["wilcoxon_greater_p"]),
                "interpretation": "stronger than pathogenic variants matched on baseline scores",
            }
        )
    if not rand_known.empty:
        r = rand_known.iloc[0]
        rows.append(
            {
                "control_type": "known_random_feature_null",
                "n_case": np.nan,
                "n_control": np.nan,
                "effect": float(r["observed_delta_mean_score"]),
                "auroc_or_p": float(r["empirical_p_delta_mean_score"]),
                "p_value": float(r["empirical_p_delta_mean_score"]),
                "interpretation": "known collagen Gly effect exceeds matched random feature sets",
            }
        )
    if not rand_vus.empty:
        r = rand_vus.iloc[0]
        rows.append(
            {
                "control_type": "vus_random_feature_null",
                "n_case": int(r["strict_collagen_gly_vus"]),
                "n_control": int(r["background_structural_vus"]),
                "effect": float(r["observed_strict_vs_background_score_enrichment"]),
                "auroc_or_p": float(r["empirical_p_score_enrichment"]),
                "p_value": float(r["empirical_p_score_enrichment"]),
                "interpretation": "VUS set-level enrichment is not significant; keep VUS claims hypothesis-generating",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(out_dir / "vus_candidate_clinvar_residue_evidence.csv")
    candidates = support_flags(candidates)
    controls = control_summary(out_dir)
    counts = summary_counts(candidates)

    review_cols = [
        "application_evidence_level",
        "rank",
        "candidate_tier",
        "variant_id",
        "gene",
        "pchange",
        "mechanism_hint",
        "collagen_gly_feature_score",
        "same_residue_pathogenic_count",
        "same_residue_benign_count",
        "same_residue_pathogenic_pchanges",
        "feature_known_path_rate",
        "feature_known_collagen_gly_pathogenic_count",
        "gnomad_pLI",
        "gnomad_oe_lof_upper",
        "gnomad_mis_z",
        "hgnc_omim_id",
        "hgnc_orphanet_id",
        "has_lsdb_link",
    ]
    review = (
        candidates[candidates["is_actionable_review_hook"]]
        .sort_values(
            [
                "application_evidence_level",
                "same_residue_pathogenic_count",
                "collagen_gly_feature_score",
            ],
            ascending=[True, False, False],
        )[review_cols]
        .reset_index(drop=True)
    )
    review.insert(0, "review_rank", np.arange(1, len(review) + 1))

    candidates.to_csv(out_dir / "vus_application_readiness_candidates.csv", index=False)
    counts.to_csv(out_dir / "vus_application_readiness_counts.csv", index=False)
    controls.to_csv(out_dir / "vus_application_readiness_controls.csv", index=False)
    review.to_csv(out_dir / "vus_application_review_hooks.csv", index=False)

    n_a = int(counts.loc[counts["group"].eq("tier1_collagen_gly"), "n_A_same_residue_pathogenic"].iloc[0])
    n_b = int(counts.loc[counts["group"].eq("tier1_collagen_gly"), "n_B_feature_gene_context"].iloc[0])
    vus_null = controls.loc[controls["control_type"].eq("vus_random_feature_null")]
    vus_null_sentence = ""
    if not vus_null.empty:
        r = vus_null.iloc[0]
        vus_null_sentence = (
            f"The VUS set-level random feature null is not significant "
            f"(score-enrichment empirical p={float(r['p_value']):.3g}); this prevents a broad clinical claim."
        )

    lines = [
        "# VUS Application Readiness",
        "",
        "## Bottom Line",
        "",
        "The structural/collagen VUS application has useful manual-review hooks, but it is not yet a clinical reclassification result. The strongest support is candidate-level same-residue pathogenic ClinVar evidence plus gene/disease context; broad VUS-set enrichment remains weak.",
        "",
        f"- Tier 1 collagen-gly candidates with same-residue pathogenic/no-benign evidence: {n_a}.",
        f"- Tier 1 collagen-gly candidates with feature-level known pathogenic support plus gene context: {n_b}.",
        f"- {vus_null_sentence}",
        "",
        "## Candidate Evidence Counts",
        "",
        fmt_table(counts),
        "",
        "## Negative and Specificity Controls",
        "",
        fmt_table(controls),
        "",
        "## Highest-Readiness Review Hooks",
        "",
        fmt_table(review, max_rows=args.top_n),
        "",
        "## Interpretation",
        "",
        "This application is publishable only as a prioritized review/functional-follow-up workflow unless temporal ClinVar reclassification, external LSDB/ClinGen evidence, gnomAD variant-level common-benign depletion, or wet-lab assay validation is added. The known-variant controls support the collagen-Gly feature as a mechanism signal; the VUS cohort itself needs stronger external validation before it can support claims of clinical pathogenicity.",
        "",
    ]
    (out_dir / "vus_application_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'vus_application_readiness.md'}")
    print(counts.to_string(index=False))
    print(controls.to_string(index=False))


if __name__ == "__main__":
    main()
