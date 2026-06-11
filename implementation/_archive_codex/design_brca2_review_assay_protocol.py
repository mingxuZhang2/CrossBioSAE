#!/usr/bin/env python3
"""Create a manuscript-style BRCA2 review/assay protocol from the prospective panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ARM_METADATA = {
    "prospective_pathogenic_review": {
        "assay_role": "primary_positive_targets",
        "primary_question": "Do old VUS with concordant DNA+protein mechanism evidence behave as BRCA2 loss-of-function variants?",
        "expected_function_class": "LOF",
        "expected_review_action": "pathogenic_review_or_functional_followup",
        "mechanism_readout": "concordant_high_DNA_and_protein",
        "success_criterion": "High LOF fraction and clear separation from benign controls; native-SAE features should localize the mechanism after full embeddings finish.",
        "failure_interpretation": "If many rows validate as functional, concordant checkpoint evidence is over-prioritizing hotspots and should be treated as a triage signal only.",
    },
    "prospective_benign_controls": {
        "assay_role": "negative_controls",
        "primary_question": "Do old VUS with concordant low DNA+protein mechanism evidence behave as functional/benign-like variants?",
        "expected_function_class": "functional",
        "expected_review_action": "benign_review_control_or_negative_assay_control",
        "mechanism_readout": "concordant_low_DNA_and_protein",
        "success_criterion": "Low LOF fraction and low disruption scores; these controls should remain separated from positive targets.",
        "failure_interpretation": "If many rows validate as LOF, the concordant-low rule is missing BRCA2 mechanisms and must be revised.",
    },
    "prospective_split_mechanism_tests": {
        "assay_role": "mechanism_stress_tests",
        "primary_question": "Can the final cross-modal/native-SAE analysis explain old unresolved SGE-LOF variants where DNA and protein checkpoint signals disagree?",
        "expected_function_class": "LOF",
        "expected_review_action": "mechanism_assay_or_native_SAE_followup",
        "mechanism_readout": "DNA_high_protein_low_or_protein_high_DNA_low",
        "success_criterion": "LOF is retained, but explanations split into DNA-side versus protein-side mechanisms with domain-specific support.",
        "failure_interpretation": "If native-SAE explanations do not distinguish these rows, the mechanism claim should stay at checkpoint-level stratification.",
    },
    "prospective_model_conflict_controls": {
        "assay_role": "error_analysis_controls",
        "primary_question": "Which old VUS are model-high but SGE functional, and what failure mode do they represent?",
        "expected_function_class": "functional",
        "expected_review_action": "model_error_analysis_not_pathogenic_claim",
        "mechanism_readout": "model_high_but_SGE_benign",
        "success_criterion": "Rows remain functional/benign-like and reveal hotspot/domain contexts where model evidence overcalls risk.",
        "failure_interpretation": "If many rows validate as LOF in external assays, SGE labels or assay context may be incomplete and should be investigated.",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_prospective_followup_panel.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_prospective_followup_panel_summary.csv",
    )
    parser.add_argument(
        "--domain-summary",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_prospective_followup_panel_domain_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_review_assay_protocol")
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


def evidence_summary(row: pd.Series) -> str:
    bits = [
        f"old={row['old_status_class']}",
        f"SGE={row['func_class']}",
        f"mechanism={row['discordance_category']}",
        f"domain={row['brca2_domain']}",
        f"dna_pct={float(row['dna_percentile']):.3f}",
        f"protein_pct={float(row['protein_percentile']):.3f}",
    ]
    if pd.notna(row.get("hotspot_n_variants")):
        bits.append(f"hotspot_n={int(row['hotspot_n_variants'])}")
    if row.get("same_residue_other_known_pathogenic", 0) > 0:
        bits.append("same_residue_other_pathogenic")
    if row.get("same_residue_other_known_benign", 0) > 0:
        bits.append("same_residue_other_benign")
    return "; ".join(bits)


def build_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in panel.iterrows():
        meta = ARM_METADATA[row["panel_arm"]]
        rows.append(
            {
                "panel_rank": int(row["panel_rank"]),
                "panel_arm": row["panel_arm"],
                "assay_role": meta["assay_role"],
                "id": row["id"],
                "aa_change": row["AA.change"],
                "brca2_domain": row["brca2_domain"],
                "c_nom": row["c.nom"],
                "g_nom": row["g.nom"],
                "dbsnp_id": row.get("dbSNP.ID", ""),
                "old_status_class": row["old_status_class"],
                "old_clinsig": row["old_clinsig"],
                "old_review_status": row["old_review_status"],
                "old_number_submitters_max": int(row["old_number_submitters_max"]),
                "sge_label": int(row["label"]),
                "sge_func_class": row["func_class"],
                "function_score": float(row["function_score"]),
                "discordance_category": row["discordance_category"],
                "dna_percentile": float(row["dna_percentile"]),
                "protein_percentile": float(row["protein_percentile"]),
                "expected_function_class": meta["expected_function_class"],
                "primary_question": meta["primary_question"],
                "recommended_followup": meta["expected_review_action"],
                "mechanism_readout": meta["mechanism_readout"],
                "decision_rule": meta["success_criterion"],
                "failure_interpretation": meta["failure_interpretation"],
                "same_aa_known_pathogenic": int(row.get("same_aa_known_pathogenic", 0)),
                "same_residue_other_known_pathogenic": int(row.get("same_residue_other_known_pathogenic", 0)),
                "same_residue_other_known_benign": int(row.get("same_residue_other_known_benign", 0)),
                "evidence_summary": evidence_summary(row),
            }
        )
    return pd.DataFrame(rows)


def build_hypotheses(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        meta = ARM_METADATA[row["panel_arm"]]
        rows.append(
            {
                "panel_arm": row["panel_arm"],
                "n": int(row["n"]),
                "assay_role": meta["assay_role"],
                "primary_hypothesis": meta["primary_question"],
                "expected_function_class": meta["expected_function_class"],
                "mechanism_readout": meta["mechanism_readout"],
                "success_criterion": meta["success_criterion"],
                "failure_interpretation": meta["failure_interpretation"],
                "n_old_clean_vus": int(row["n_old_clean_vus"]),
                "n_old_conflicting": int(row["n_old_conflicting"]),
                "n_domains": int(row["n_domains"]),
                "n_recurrent_hotspot": int(row["n_recurrent_hotspot"]),
                "median_function_score": float(row["median_function_score"]),
                "median_dna_percentile": float(row["median_dna_percentile"]),
                "median_protein_percentile": float(row["median_protein_percentile"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    hypotheses: pd.DataFrame,
    manifest: pd.DataFrame,
    domain_summary: pd.DataFrame,
) -> None:
    key_cols = [
        "panel_rank",
        "panel_arm",
        "id",
        "aa_change",
        "brca2_domain",
        "old_status_class",
        "expected_function_class",
        "mechanism_readout",
        "recommended_followup",
    ]
    lines = [
        "# BRCA2 Review/Assay Protocol",
        "",
        "## Purpose",
        "",
        "Turn the BRCA2 prospective follow-up panel into an executable review and assay design. The protocol is intentionally conservative: it is for manual review and functional follow-up, not clinical reclassification.",
        "",
        "## Evidence Inputs",
        "",
        "- Functional labels: Sahu et al. BRCA2 SGE function scores already parsed in `brca2_variants.csv`.",
        "- Mechanism layer: BRCA2 ESM-vs-Evo2 LLR checkpoint categories and percentiles.",
        "- Temporal context: archived ClinVar 2025-01 status.",
        "- Candidate set: old unresolved ClinVar rows only for primary prospective arms; old P/LP-like rows are excluded as sanity checks.",
        "",
        "## Arm Hypotheses",
        "",
        table(hypotheses),
        "",
        "## Domain Balance",
        "",
        table(domain_summary),
        "",
        "## Variant Manifest",
        "",
        table(manifest[key_cols], max_rows=80),
        "",
        "## Recommended Validation Plan",
        "",
        "1. Blind manual review: give reviewers variant identifiers, old ClinVar status, domain context, and same-residue evidence, but hold out SGE labels and model arm during first-pass ACMG-style evidence review.",
        "2. Functional follow-up: use a BRCA2 homology-directed repair or saturation-style mini-panel assay for the 24 pathogenic targets, 16 benign controls, and 12 split-mechanism tests.",
        "3. Mechanism readout: after full BRCA2 embeddings finish, overlay native-SAE feature ablation and feature activation signatures on the same panel.",
        "4. Primary statistical readout: compare LOF fraction and function-score distribution between pathogenic targets and benign controls; report split-mechanism rows separately.",
        "5. Error analysis: use model-conflict controls to identify domains or residues where high checkpoint scores overcall SGE-functional variants.",
        "",
        "## Claim Boundary",
        "",
        "This protocol can support a top-journal-style application claim if external review or assay outcomes validate the arm-level hypotheses. It does not by itself reclassify any BRCA2 variant clinically.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(require_file(args.panel))
    summary = pd.read_csv(require_file(args.summary))
    domains = pd.read_csv(require_file(args.domain_summary))

    manifest = build_manifest(panel)
    hypotheses = build_hypotheses(summary)

    prefix = out / args.output_prefix
    manifest.to_csv(prefix.with_name(prefix.name + "_manifest.csv"), index=False)
    hypotheses.to_csv(prefix.with_name(prefix.name + "_arm_hypotheses.csv"), index=False)
    write_report(prefix.with_suffix(".md"), hypotheses, manifest, domains)
    print(f"Wrote {prefix.with_suffix('.md')}")
    print(hypotheses.to_string(index=False))


if __name__ == "__main__":
    main()
