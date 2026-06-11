#!/usr/bin/env python3
"""Create BRCA2 ClinVar VUS/conflicting interpretability-review candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


VUS_LABELS = {
    "Uncertain significance",
    "Conflicting interpretations of pathogenicity",
}
KNOWN_PATHOGENIC_LABELS = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
KNOWN_BENIGN_LABELS = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missense-scores",
        type=Path,
        default=root
        / "results"
        / "interpretability_applications"
        / "brca2_llr_esm_discordance_missense_scores.csv",
    )
    parser.add_argument(
        "--hotspots",
        type=Path,
        default=root
        / "results"
        / "interpretability_applications"
        / "brca2_llr_esm_discordance_residue_hotspots.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_clinvar_interpretability_candidates")
    parser.add_argument("--top-n", type=int, default=60)
    return parser.parse_args()


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def candidate_tier(row: pd.Series) -> str:
    category = str(row["discordance_category"])
    label = int(row["label"])
    if label == 1 and category == "both_high":
        return "tier1_functional_lof_concordant_high"
    if label == 0 and category == "both_low":
        return "tier1_functional_benign_concordant_low"
    if label == 1 and category in {"protein_high_dna_low", "dna_high_protein_low"}:
        return "tier2_functional_lof_split_mechanism"
    if label == 0 and category == "both_high":
        return "tier2_functional_benign_model_high_conflict"
    if label == 1 and category == "both_low":
        return "tier2_functional_lof_model_low_conflict"
    return "tier3_other_vus_conflicting"


def tier_direction(tier: str) -> str:
    if "functional_lof" in tier:
        return "pathogenic_review"
    if "functional_benign" in tier:
        return "benign_review"
    return "uncertain_review"


def add_hotspot_context(candidates: pd.DataFrame, hotspots: pd.DataFrame) -> pd.DataFrame:
    if hotspots.empty:
        candidates["hotspot_n_variants"] = np.nan
        candidates["hotspot_n_lof"] = np.nan
        candidates["hotspot_lof_rate"] = np.nan
        candidates["hotspot_aa_changes"] = ""
        return candidates
    key_cols = ["discordance_category", "aa_pos", "brca2_domain"]
    keep = hotspots.rename(columns={"category": "discordance_category"}).copy()
    keep = keep[key_cols + ["n_variants", "n_lof", "lof_rate", "aa_changes"]]
    keep = keep.rename(
        columns={
            "n_variants": "hotspot_n_variants",
            "n_lof": "hotspot_n_lof",
            "lof_rate": "hotspot_lof_rate",
            "aa_changes": "hotspot_aa_changes",
        }
    )
    return candidates.merge(keep, on=key_cols, how="left")


def add_same_residue_context(candidates: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    known = scores[scores["clinvar_simple"].isin(KNOWN_PATHOGENIC_LABELS | KNOWN_BENIGN_LABELS)].copy()
    support_cols = [
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
    ]
    if known.empty:
        for col in support_cols:
            candidates[col] = "" if col.endswith("_changes") else 0
        return candidates

    def pack(sub: pd.DataFrame, labels: set[str]) -> tuple[int, str]:
        hit = sub[sub["clinvar_simple"].isin(labels)]
        return int(len(hit)), ";".join(sorted(hit["AA.change"].dropna().astype(str).unique())[:12])

    known = known.copy()
    known["_aa_change_str"] = known["AA.change"].fillna("").astype(str)
    rows = []
    for _, row in candidates.iterrows():
        aa_change = str(row.get("AA.change", "") or "")
        same_residue = known[
            known["aa_pos"].eq(row["aa_pos"])
            & known["brca2_domain"].eq(row["brca2_domain"])
        ]
        same_aa = same_residue[same_residue["_aa_change_str"].eq(aa_change)]
        other_aa = same_residue[~same_residue["_aa_change_str"].eq(aa_change)]
        same_aa_path, same_aa_path_changes = pack(same_aa, KNOWN_PATHOGENIC_LABELS)
        same_aa_benign, same_aa_benign_changes = pack(same_aa, KNOWN_BENIGN_LABELS)
        other_path, other_path_changes = pack(other_aa, KNOWN_PATHOGENIC_LABELS)
        other_benign, other_benign_changes = pack(other_aa, KNOWN_BENIGN_LABELS)
        all_path, all_path_changes = pack(same_residue, KNOWN_PATHOGENIC_LABELS)
        all_benign, all_benign_changes = pack(same_residue, KNOWN_BENIGN_LABELS)
        rows.append(
            {
                "same_aa_known_pathogenic": same_aa_path,
                "same_aa_known_benign": same_aa_benign,
                "same_aa_known_pathogenic_changes": same_aa_path_changes,
                "same_aa_known_benign_changes": same_aa_benign_changes,
                "same_residue_other_known_pathogenic": other_path,
                "same_residue_other_known_benign": other_benign,
                "same_residue_other_known_pathogenic_changes": other_path_changes,
                "same_residue_other_known_benign_changes": other_benign_changes,
                "same_residue_known_pathogenic": all_path,
                "same_residue_known_benign": all_benign,
                "same_residue_known_pathogenic_changes": all_path_changes,
                "same_residue_known_benign_changes": all_benign_changes,
            }
        )
    support = pd.DataFrame(rows, index=candidates.index)
    return pd.concat([candidates, support], axis=1)


def summary_by_tier(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, sub in candidates.groupby("review_tier", sort=True):
        rows.append(
            {
                "review_tier": tier,
                "review_direction": tier_direction(tier),
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "n_uncertain": int(sub["clinvar_simple"].eq("Uncertain significance").sum()),
                "n_conflicting": int(sub["clinvar_simple"].eq("Conflicting interpretations of pathogenicity").sum()),
                "median_function_score": float(sub["function_score"].median()),
                "median_dna_percentile": float(sub["dna_percentile"].median()),
                "median_protein_percentile": float(sub["protein_percentile"].median()),
                "n_recurrent_hotspot": int(sub["hotspot_n_variants"].fillna(0).ge(2).sum()),
                "n_same_aa_known_pathogenic": int(sub["same_aa_known_pathogenic"].gt(0).sum()),
                "n_same_aa_known_pathogenic_no_benign": int(
                    (sub["same_aa_known_pathogenic"].gt(0) & sub["same_aa_known_benign"].eq(0)).sum()
                ),
                "n_same_aa_known_benign": int(sub["same_aa_known_benign"].gt(0).sum()),
                "n_same_residue_other_known_pathogenic": int(
                    sub["same_residue_other_known_pathogenic"].gt(0).sum()
                ),
                "n_same_residue_other_known_benign": int(
                    sub["same_residue_other_known_benign"].gt(0).sum()
                ),
                "n_same_residue_known_pathogenic": int(sub["same_residue_known_pathogenic"].gt(0).sum()),
                "n_same_residue_known_pathogenic_no_benign": int(
                    (sub["same_residue_known_pathogenic"].gt(0) & sub["same_residue_known_benign"].eq(0)).sum()
                ),
                "n_same_residue_known_benign": int(sub["same_residue_known_benign"].gt(0).sum()),
            }
        )
    order = {
        "tier1_functional_lof_concordant_high": 0,
        "tier1_functional_benign_concordant_low": 1,
        "tier2_functional_lof_split_mechanism": 2,
        "tier2_functional_benign_model_high_conflict": 3,
        "tier2_functional_lof_model_low_conflict": 4,
        "tier3_other_vus_conflicting": 5,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["review_tier"].map(order).fillna(99)
    return out.sort_values(["_order", "review_tier"]).drop(columns="_order").reset_index(drop=True)


def summary_by_clinvar_category(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (clinvar, category), sub in candidates.groupby(["clinvar_simple", "discordance_category"], sort=True):
        rows.append(
            {
                "clinvar_simple": clinvar,
                "discordance_category": category,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "mean_function_score": float(sub["function_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["clinvar_simple", "discordance_category"]).reset_index(drop=True)


def validation_table(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["clinvar_simple"].isin(KNOWN_PATHOGENIC_LABELS | KNOWN_BENIGN_LABELS)].copy()
    if known.empty:
        return pd.DataFrame()
    known["clinvar_binary"] = known["clinvar_simple"].isin(KNOWN_PATHOGENIC_LABELS).astype(int)
    rows = []
    for category, sub in known.groupby("discordance_category", sort=True):
        rest = known[~known["discordance_category"].eq(category)]
        odds, p = (float("nan"), float("nan"))
        if not rest.empty and sub["clinvar_binary"].nunique() > 0 and rest["clinvar_binary"].nunique() > 0:
            odds, p = fisher_exact(
                [
                    [int(sub["clinvar_binary"].sum()), int((1 - sub["clinvar_binary"]).sum())],
                    [int(rest["clinvar_binary"].sum()), int((1 - rest["clinvar_binary"]).sum())],
                ]
            )
        rows.append(
            {
                "discordance_category": category,
                "n_known_clinvar": int(len(sub)),
                "n_pathogenic_clinvar": int(sub["clinvar_binary"].sum()),
                "pathogenic_clinvar_rate": float(sub["clinvar_binary"].mean()),
                "n_sge_lof": int(sub["label"].sum()),
                "sge_lof_rate": float(sub["label"].mean()),
                "fisher_odds_pathogenic_clinvar_vs_rest": odds,
                "fisher_p_pathogenic_clinvar_vs_rest": p,
            }
        )
    return pd.DataFrame(rows).sort_values(["discordance_category"]).reset_index(drop=True)


def ranked_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    tier_order = {
        "tier1_functional_lof_concordant_high": 0,
        "tier1_functional_benign_concordant_low": 1,
        "tier2_functional_lof_split_mechanism": 2,
        "tier2_functional_benign_model_high_conflict": 3,
        "tier2_functional_lof_model_low_conflict": 4,
        "tier3_other_vus_conflicting": 5,
    }
    out["_tier_order"] = out["review_tier"].map(tier_order).fillna(99)
    out["_pathogenic_sort"] = np.where(out["label"].eq(1), -out["function_score"], out["function_score"])
    return out.sort_values(
        [
            "_tier_order",
            "same_aa_known_pathogenic",
            "same_aa_known_benign",
            "hotspot_n_variants",
            "same_residue_known_pathogenic",
            "same_residue_known_benign",
            "label",
            "_pathogenic_sort",
            "dna_percentile",
            "protein_percentile",
        ],
        ascending=[True, False, False, False, False, False, False, False, False, False],
    ).drop(columns=["_tier_order", "_pathogenic_sort"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.missense_scores)
    hotspots = pd.read_csv(args.hotspots) if args.hotspots.exists() else pd.DataFrame()
    candidates = scores[scores["clinvar_simple"].isin(VUS_LABELS)].copy()
    if candidates.empty:
        raise RuntimeError("No BRCA2 ClinVar VUS/conflicting missense candidates found")
    candidates["review_tier"] = candidates.apply(candidate_tier, axis=1)
    candidates["review_direction"] = candidates["review_tier"].map(tier_direction)
    candidates = add_hotspot_context(candidates, hotspots)
    candidates = add_same_residue_context(candidates, scores)
    candidates = ranked_candidates(candidates)

    tier_summary = summary_by_tier(candidates)
    clinvar_category_summary = summary_by_clinvar_category(candidates)
    known_validation = validation_table(scores)

    cols = [
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
        "hotspot_aa_changes",
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
    candidates.to_csv(out / f"{prefix}.csv", index=False)
    tier_summary.to_csv(out / f"{prefix}_tier_summary.csv", index=False)
    clinvar_category_summary.to_csv(out / f"{prefix}_clinvar_category_summary.csv", index=False)
    known_validation.to_csv(out / f"{prefix}_known_validation.csv", index=False)
    candidates[cols].head(args.top_n).to_csv(out / f"{prefix}_top_candidates.csv", index=False)

    report = [
        "# BRCA2 ClinVar Interpretability Candidates",
        "",
        "## Purpose",
        "",
        "Prioritize BRCA2 ClinVar VUS or conflicting missense variants for manual review using independent SGE functional labels plus the protein-side ESM and DNA-side Evo2 LLR checkpoint mechanism categories. This is an evidence triage table, not clinical reclassification.",
        "",
        "## Scope",
        "",
        f"- Candidate ClinVar categories: {', '.join(sorted(VUS_LABELS))}.",
        f"- Candidate missense rows: {len(candidates):,}.",
        f"- Source score table: `{args.missense_scores}`.",
        "",
        "## Review-Tier Summary",
        "",
        table(tier_summary),
        "",
        "## ClinVar Category by Mechanism Category",
        "",
        table(clinvar_category_summary),
        "",
        "## Known ClinVar Sanity Check",
        "",
        table(known_validation),
        "",
        "## Top Review Candidates",
        "",
        table(candidates[cols], max_rows=args.top_n),
        "",
        "## Interpretation",
        "",
        "- Tier 1 pathogenic-review candidates have VUS/conflicting ClinVar labels, SGE LOF labels, and concordant high protein+DNA model evidence. These are the cleanest manual-review candidates.",
        "- Tier 1 benign-review candidates have VUS/conflicting ClinVar labels, functional SGE labels, and concordant low protein+DNA model evidence. These are useful negative-control and possible benign-review candidates.",
        "- Same-AA-change and same-residue ClinVar pathogenic or benign rows provide additional review hooks when present; they are support for manual review, not automatic reclassification.",
        "- Tier 2 split-mechanism candidates are SGE LOF but protein-high/DNA-low or DNA-high/protein-low. They should be inspected after the full cross-modal/native-SAE BRCA2 run, because they test whether the model can explain modality-specific mechanisms.",
        "- Model-high but SGE-benign and model-low but SGE-LOF rows are explicit conflict cases; they are valuable for error analysis and should not be promoted as confident interpretations without additional evidence.",
        "",
    ]
    (out / f"{prefix}.md").write_text("\n".join(report), encoding="utf-8")
    print(tier_summary.to_string(index=False))


if __name__ == "__main__":
    main()
