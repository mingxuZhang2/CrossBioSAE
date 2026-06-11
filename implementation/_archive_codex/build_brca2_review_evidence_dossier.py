#!/usr/bin/env python3
"""Build BRCA2 review/assay evidence dossiers.

This converts the BRCA2 panel, temporal ClinVar context, and blinded handoff
files into two practical artifacts:

1. a blinded reviewer packet index that exposes only curation-safe evidence, and
2. an internal unblinded dossier that records model/SGE evidence, expected
   outcomes, assay endpoints, and claim boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


ARM_LABELS = {
    "prospective_pathogenic_review": "Pathogenic-review targets",
    "prospective_benign_controls": "Benign-like controls",
    "prospective_split_mechanism_tests": "Split-mechanism stress tests",
    "prospective_model_conflict_controls": "Model-conflict error controls",
    "pathogenic_review_high_confidence": "Pathogenic-review targets",
    "benign_review_controls": "Benign-like controls",
    "split_mechanism_tests": "Split-mechanism stress tests",
    "model_high_sge_benign_conflicts": "Model-conflict error controls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def old_unresolved(row: pd.Series) -> bool:
    return str(row.get("old_status_class", "")).lower() in {"old_clean_vus", "old_conflicting", "old_absent"}


def safe_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def first_notna(*values: object) -> object:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return ""


def same_residue_summary(row: pd.Series) -> str:
    bits = []
    if safe_int(row.get("same_aa_known_pathogenic", 0)) > 0:
        bits.append("same-AA known pathogenic")
    if safe_int(row.get("same_aa_known_benign", 0)) > 0:
        bits.append("same-AA known benign")
    if safe_int(row.get("same_residue_other_known_pathogenic", 0)) > 0:
        bits.append("same-residue other pathogenic")
    if safe_int(row.get("same_residue_other_known_benign", 0)) > 0:
        bits.append("same-residue other benign")
    return "; ".join(bits) if bits else "no same-residue hook"


def evidence_flags(row: pd.Series) -> dict[str, object]:
    label = int(row.get("sge_label", row.get("label", 0)) or 0)
    dna = float(row.get("dna_percentile", np.nan))
    protein = float(row.get("protein_percentile", np.nan))
    both_high = str(row.get("discordance_category", "")) == "both_high"
    both_low = str(row.get("discordance_category", "")) == "both_low"
    split = str(row.get("discordance_category", "")) in {"dna_high_protein_low", "protein_high_dna_low"}
    same_path = safe_int(row.get("same_aa_known_pathogenic", 0)) + safe_int(
        row.get("same_residue_other_known_pathogenic", 0)
    )
    same_benign = safe_int(row.get("same_aa_known_benign", 0)) + safe_int(
        row.get("same_residue_other_known_benign", 0)
    )
    hotspot_n = int(row.get("hotspot_n_variants", 0) or 0) if pd.notna(row.get("hotspot_n_variants", np.nan)) else 0

    pathogenic_score = 0
    pathogenic_score += 2 if old_unresolved(row) else 0
    pathogenic_score += 3 if label == 1 else 0
    pathogenic_score += 2 if both_high else 0
    pathogenic_score += 1 if split and label == 1 else 0
    pathogenic_score += 1 if dna >= 0.8 else 0
    pathogenic_score += 1 if protein >= 0.8 else 0
    pathogenic_score += 1 if same_path > 0 and same_benign == 0 else 0
    pathogenic_score += 1 if hotspot_n >= 3 else 0

    benign_score = 0
    benign_score += 2 if old_unresolved(row) else 0
    benign_score += 3 if label == 0 else 0
    benign_score += 2 if both_low else 0
    benign_score += 1 if dna <= 0.2 else 0
    benign_score += 1 if protein <= 0.2 else 0
    benign_score += 1 if same_benign > 0 and same_path == 0 else 0

    conflict_score = 0
    conflict_score += 2 if label == 0 and both_high else 0
    conflict_score += 1 if same_path > 0 and label == 0 else 0
    conflict_score += 1 if same_benign > 0 and label == 1 else 0
    conflict_score += 1 if str(row.get("old_status_class", "")) in {"old_pathogenic_or_likely_pathogenic", "old_benign_or_likely_benign"} else 0

    return {
        "old_unresolved_or_absent": old_unresolved(row),
        "model_both_high": both_high,
        "model_both_low": both_low,
        "split_mechanism": split,
        "same_residue_pathogenic_count": same_path,
        "same_residue_benign_count": same_benign,
        "recurrent_hotspot_n": hotspot_n,
        "pathogenic_evidence_score": pathogenic_score,
        "benign_evidence_score": benign_score,
        "conflict_evidence_score": conflict_score,
    }


def proposed_analysis_role(row: pd.Series) -> str:
    arm = str(row.get("panel_arm", ""))
    if "pathogenic" in arm:
        return "primary_positive_target"
    if "benign" in arm:
        return "primary_negative_control"
    if "split" in arm:
        return "mechanism_stress_test"
    if "conflict" in arm or "model_high" in arm:
        return "error_analysis_control"
    return "exploratory_review"


def allowed_claim(row: pd.Series) -> str:
    role = proposed_analysis_role(row)
    if role == "primary_positive_target":
        return "Assay/review target for old unresolved BRCA2 VUS with concordant high model and SGE LOF evidence; not clinical reclassification."
    if role == "primary_negative_control":
        return "Benign-like control for specificity of the triage rule; not clinical reclassification."
    if role == "mechanism_stress_test":
        return "Stress test for DNA-vs-protein mechanism attribution; not a primary pathogenicity claim."
    if role == "error_analysis_control":
        return "Model-failure/error-analysis control; should not support pathogenic claims."
    return "Exploratory evidence only."


def build_dossiers(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = repo / OUT_DIR
    key = require_csv(out / "brca2_blinded_unblinding_key.csv")
    review = require_csv(out / "brca2_blinded_review_sheet.csv")
    panel_temporal = require_csv(out / "brca2_clinvar_temporal_context_panel.csv")
    manifest = require_csv(out / "brca2_review_assay_protocol_manifest.csv")

    manifest_join = manifest.rename(columns={"aa_change": "AA.change", "c_nom": "c.nom", "g_nom": "g.nom"})
    merged = key.merge(
        panel_temporal,
        left_on="id",
        right_on="id",
        how="left",
        suffixes=("", "_panel"),
    )
    merged = merged.merge(
        manifest_join[
            [
                "id",
                "c.nom",
                "g.nom",
                "dbsnp_id",
                "assay_role",
                "primary_question",
                "decision_rule",
                "failure_interpretation",
                "same_aa_known_pathogenic",
                "same_residue_other_known_pathogenic",
                "same_residue_other_known_benign",
            ]
        ],
        on="id",
        how="left",
        suffixes=("", "_manifest"),
    )

    rows = []
    for _, row in merged.iterrows():
        flags = evidence_flags(row)
        out_row = {
            "blinded_id": row["blinded_id"],
            "plate_id": row["plate_id"],
            "well": row["well"],
            "panel_rank": int(row["panel_rank"]),
            "panel_arm": row["panel_arm"],
            "panel_arm_label": ARM_LABELS.get(str(row["panel_arm"]), str(row["panel_arm"])),
            "analysis_role": proposed_analysis_role(row),
            "id": row["id"],
            "aa_change": row["aa_change"],
            "c_nom": first_notna(row.get("c.nom"), row.get("c.nom_manifest"), row.get("c_nom")),
            "g_nom": first_notna(row.get("g.nom"), row.get("g.nom_manifest"), row.get("g_nom")),
            "dbsnp_id": first_notna(row.get("dbSNP.ID"), row.get("dbsnp_id"), row.get("dbsnp_id_manifest")),
            "brca2_domain": row["brca2_domain"],
            "old_status_class": row["old_status_class"],
            "old_clinsig": row.get("old_clinsig", ""),
            "old_review_status": row.get("old_review_status", ""),
            "old_number_submitters_max": row.get("old_number_submitters_max", np.nan),
            "sge_label": int(row["sge_label"]),
            "sge_func_class": row["sge_func_class"],
            "function_score": float(row["function_score"]),
            "discordance_category": row["discordance_category"],
            "dna_percentile": float(row["dna_percentile"]),
            "protein_percentile": float(row["protein_percentile"]),
            "expected_function_class": row["expected_function_class"],
            "mechanism_readout": row["mechanism_readout"],
            "recommended_followup": row["recommended_followup"],
            "primary_question": row.get("primary_question", ""),
            "decision_rule": row["decision_rule"],
            "failure_interpretation": row["failure_interpretation"],
            "same_residue_summary": same_residue_summary(row),
            "allowed_claim": allowed_claim(row),
            **flags,
        }
        rows.append(out_row)
    internal = pd.DataFrame(rows)

    blinded_cols = [
        "blinded_id",
        "aa_change",
        "c_nom",
        "g_nom",
        "dbsnp_id",
        "brca2_domain",
        "old_status_class",
        "old_clinsig",
        "old_review_status",
        "old_number_submitters_max",
        "same_residue_summary",
        "reviewer_task",
        "manual_classification",
        "manual_evidence_codes",
        "manual_confidence_1_to_5",
        "review_notes",
    ]
    blinded = review.copy()
    same_map = internal.set_index("blinded_id")["same_residue_summary"].to_dict()
    blinded["same_residue_summary"] = blinded["blinded_id"].map(same_map)
    blinded["reviewer_task"] = (
        "Classify using external/manual evidence only; do not infer from hidden SGE/model arm. "
        "Allowed labels: pathogenic_support, benign_support, uncertain, insufficient_evidence."
    )
    for col in ["manual_classification", "manual_evidence_codes", "manual_confidence_1_to_5", "review_notes"]:
        if col not in blinded:
            blinded[col] = ""
    blinded = blinded[[c for c in blinded_cols if c in blinded.columns]]

    arm_summary = (
        internal.groupby(["panel_arm", "panel_arm_label", "analysis_role"], dropna=False)
        .agg(
            n=("id", "size"),
            n_old_unresolved_or_absent=("old_unresolved_or_absent", "sum"),
            n_sge_lof=("sge_label", "sum"),
            median_function_score=("function_score", "median"),
            median_dna_percentile=("dna_percentile", "median"),
            median_protein_percentile=("protein_percentile", "median"),
            n_same_residue_pathogenic=("same_residue_pathogenic_count", lambda s: int((s > 0).sum())),
            n_same_residue_benign=("same_residue_benign_count", lambda s: int((s > 0).sum())),
            median_pathogenic_evidence_score=("pathogenic_evidence_score", "median"),
            median_benign_evidence_score=("benign_evidence_score", "median"),
            median_conflict_evidence_score=("conflict_evidence_score", "median"),
        )
        .reset_index()
    )
    return internal, blinded, arm_summary


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    widths = [max(len(str(c)), *(len(str(v)) for v in show[c])) for c in show.columns]
    lines = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(show.columns, widths)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def write_report(repo: Path, internal: pd.DataFrame, blinded: pd.DataFrame, arm_summary: pd.DataFrame) -> Path:
    out = repo / OUT_DIR / "brca2_review_evidence_dossier.md"
    primary = internal[internal["analysis_role"].eq("primary_positive_target")].copy()
    benign = internal[internal["analysis_role"].eq("primary_negative_control")].copy()
    split = internal[internal["analysis_role"].eq("mechanism_stress_test")].copy()
    conflict = internal[internal["analysis_role"].eq("error_analysis_control")].copy()
    old_unresolved_primary = int(primary["old_unresolved_or_absent"].sum()) if not primary.empty else 0
    old_unresolved_benign = int(benign["old_unresolved_or_absent"].sum()) if not benign.empty else 0

    lines = [
        "# BRCA2 Review Evidence Dossier",
        "",
        "## Purpose",
        "",
        "This dossier turns the BRCA2 interpretability output into a review/assay application package. It separates blinded reviewer-visible evidence from internal unblinded model/SGE evidence and keeps the claim boundary explicit: these rows are review or assay priorities, not clinical reclassifications.",
        "",
        "## Files",
        "",
        "- `brca2_review_evidence_dossier_internal.csv`: unblinded analyst evidence table.",
        "- `brca2_review_evidence_dossier_blinded_packet.csv`: reviewer-facing packet index that keeps SGE/model evidence hidden.",
        "- `brca2_review_evidence_dossier_arm_summary.csv`: arm-level evidence balance.",
        "",
        "## Arm Summary",
        "",
        md_table(
            arm_summary,
            [
                "panel_arm",
                "n",
                "n_old_unresolved_or_absent",
                "n_sge_lof",
                "median_function_score",
                "median_dna_percentile",
                "median_protein_percentile",
                "n_same_residue_pathogenic",
                "n_same_residue_benign",
            ],
            max_rows=12,
        ),
        "",
        "## Main Application Claim",
        "",
        f"- Primary pathogenic-review targets: n={len(primary)}, old-unresolved/absent n={old_unresolved_primary}, SGE LOF n={int(primary['sge_label'].sum()) if not primary.empty else 0}.",
        f"- Benign-like controls: n={len(benign)}, old-unresolved/absent n={old_unresolved_benign}, SGE LOF n={int(benign['sge_label'].sum()) if not benign.empty else 0}.",
        f"- Split-mechanism stress tests: n={len(split)}.",
        f"- Model-conflict error controls: n={len(conflict)}.",
        "",
        "Interpretation: this is a concrete downstream use of the interpretability layer. It creates a blinded review/assay panel that can test whether DNA/protein mechanism strata prioritize old unresolved BRCA2 variants, while also including negative and model-conflict controls.",
        "",
        "## Highest-Priority Internal Evidence Rows",
        "",
        md_table(
            internal.sort_values(
                ["analysis_role", "pathogenic_evidence_score", "conflict_evidence_score"],
                ascending=[True, False, False],
            ),
            [
                "blinded_id",
                "panel_arm",
                "id",
                "aa_change",
                "old_status_class",
                "sge_func_class",
                "discordance_category",
                "dna_percentile",
                "protein_percentile",
                "same_residue_summary",
                "pathogenic_evidence_score",
                "benign_evidence_score",
                "conflict_evidence_score",
            ],
            max_rows=24,
        ),
        "",
        "## Blinding Boundary",
        "",
        "The blinded packet contains variant identity, old ClinVar context, domain context, and same-residue hooks. It does not contain panel arm, SGE label/function score, DNA/protein percentiles, model category, or expected class. The unblinding key and internal dossier should be opened only after manual classifications or assay readouts are locked.",
        "",
        "## Success Criteria",
        "",
        "- Primary success: pathogenic-review targets show a higher external LOF/pathogenic-support rate than benign controls under the pre-specified Fisher test.",
        "- Secondary success: split-mechanism rows retain external functional evidence while separating into DNA-high/protein-low or protein-high/DNA-low explanations.",
        "- Error-analysis success: model-conflict controls remain functional/benign-like and identify overcalled regions or same-residue contradictions.",
        "",
        "## Claim Boundary",
        "",
        "Supported now: a reviewable and assayable BRCA2 VUS triage application. Not supported now: clinical VUS reclassification, because no locked external review or new assay readout exists yet.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    internal, blinded, arm_summary = build_dossiers(repo)
    internal.to_csv(out / "brca2_review_evidence_dossier_internal.csv", index=False)
    blinded.to_csv(out / "brca2_review_evidence_dossier_blinded_packet.csv", index=False)
    arm_summary.to_csv(out / "brca2_review_evidence_dossier_arm_summary.csv", index=False)
    report = write_report(repo, internal, blinded, arm_summary)
    print(f"wrote {out / 'brca2_review_evidence_dossier_internal.csv'}")
    print(f"wrote {out / 'brca2_review_evidence_dossier_blinded_packet.csv'}")
    print(f"wrote {out / 'brca2_review_evidence_dossier_arm_summary.csv'}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
