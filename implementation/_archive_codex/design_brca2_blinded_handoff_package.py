#!/usr/bin/env python
"""Create blinded BRCA2 review and assay handoff files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


REVIEW_COLUMNS = [
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
    "same_aa_known_pathogenic",
    "same_residue_other_known_pathogenic",
    "same_residue_other_known_benign",
    "reviewer_id",
    "review_date",
    "manual_classification",
    "manual_evidence_codes",
    "manual_confidence_1_to_5",
    "review_notes",
]


def well_ids(n: int) -> list[str]:
    wells = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
    if n > len(wells):
        raise ValueError(f"Need {n} wells but only one 96-well plate is implemented.")
    return wells[:n]


def build_handoff(repo: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = repo / OUT_DIR / "brca2_review_assay_protocol_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    shuffled = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True).copy()
    shuffled["blinded_id"] = [f"BRCA2-BLIND-{i:03d}" for i in range(1, len(shuffled) + 1)]
    shuffled["plate_id"] = "BRCA2_MINIPANEL_P1"
    shuffled["well"] = well_ids(len(shuffled))

    for col in ["dbsnp_id", "old_number_submitters_max"]:
        if col not in shuffled.columns:
            shuffled[col] = pd.NA

    review = shuffled.copy()
    for col in [
        "reviewer_id",
        "review_date",
        "manual_classification",
        "manual_evidence_codes",
        "manual_confidence_1_to_5",
        "review_notes",
    ]:
        review[col] = ""
    review = review[REVIEW_COLUMNS]

    assay = shuffled[
        [
            "blinded_id",
            "plate_id",
            "well",
            "aa_change",
            "c_nom",
            "g_nom",
            "dbsnp_id",
            "brca2_domain",
        ]
    ].copy()
    for col in [
        "replicate_1_function_score",
        "replicate_2_function_score",
        "replicate_3_function_score",
        "mean_function_score",
        "binary_external_call",
        "assay_qc_status",
        "assay_notes",
    ]:
        assay[col] = ""

    key_cols = [
        "blinded_id",
        "plate_id",
        "well",
        "panel_rank",
        "panel_arm",
        "assay_role",
        "id",
        "aa_change",
        "brca2_domain",
        "old_status_class",
        "sge_label",
        "sge_func_class",
        "function_score",
        "discordance_category",
        "dna_percentile",
        "protein_percentile",
        "expected_function_class",
        "mechanism_readout",
        "recommended_followup",
        "decision_rule",
        "failure_interpretation",
        "evidence_summary",
    ]
    for col in key_cols:
        if col not in shuffled.columns:
            shuffled[col] = pd.NA
    key = shuffled[key_cols].copy()

    balance = (
        key.groupby(["panel_arm", "brca2_domain"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["panel_arm", "brca2_domain"])
    )
    return review, assay, key, balance


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_readme(repo: Path, seed: int, review: pd.DataFrame, assay: pd.DataFrame, key: pd.DataFrame, balance: pd.DataFrame) -> Path:
    out = repo / OUT_DIR / "brca2_blinded_handoff_readme.md"
    arm_counts = key["panel_arm"].value_counts().rename_axis("panel_arm").reset_index(name="n")
    lines = [
        "# BRCA2 Blinded Review And Assay Handoff Package",
        "",
        "## Purpose",
        "",
        "This package turns the BRCA2 prospective panel into files that can be handed to manual reviewers or assay collaborators. It is conservative by design: blinded sheets omit SGE labels, model arm, expected function class, DNA/protein percentiles, and decision rules.",
        "",
        "## Files",
        "",
        "- `brca2_blinded_review_sheet.csv`: reviewer-facing manual curation sheet.",
        "- `brca2_blinded_assay_readout_template.csv`: assay-facing plate/readout template.",
        "- `brca2_blinded_unblinding_key.csv`: internal key with hidden arm, SGE, model, and expected-class fields.",
        "- `brca2_blinded_arm_balance.csv`: internal arm/domain balance check.",
        "- `brca2_blinded_validation_*.csv/md`: generated after running the locked-outcome analyzer.",
        "",
        "## Blinding",
        "",
        f"- Random seed: {seed}.",
        "- Reviewers see variant identifiers, old ClinVar context, domain context, and same-residue evidence.",
        "- Reviewers do not see panel arm, SGE label, SGE function score, model category, DNA/protein percentiles, or expected function class.",
        "- The assay sheet includes plate/well locations and variant identifiers, but hides arm and expected class.",
        "",
        "## Arm Counts In Hidden Key",
        "",
        md_table(arm_counts, ["panel_arm", "n"]),
        "",
        "## Analysis After Unblinding",
        "",
        "1. Lock reviewer classifications and assay readouts before opening the unblinding key.",
        "2. Run `python scripts/analyze_brca2_blinded_review_assay_results.py --repo-root .`.",
        "3. The analyzer refuses to unblind incomplete sheets and reports `not_ready_no_locked_outcomes` until locked outcomes exist.",
        "4. Primary analysis: compare LOF fraction in `prospective_pathogenic_review` vs `prospective_benign_controls` using the pre-specified one-sided Fisher exact test.",
        "5. Secondary analysis: compare continuous assay/function scores between those arms.",
        "6. Analyze `prospective_split_mechanism_tests` separately as a DNA/protein mechanism stress test.",
        "7. Use `prospective_model_conflict_controls` only for error analysis, not for pathogenic claims.",
        "",
        "## Claim Boundary",
        "",
        "These files support blinded review and functional follow-up. They do not clinically reclassify any variant without expert adjudication and external evidence.",
        "",
        "## Internal Balance Check",
        "",
        md_table(balance, ["panel_arm", "brca2_domain", "n"]),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    review, assay, key, balance = build_handoff(repo, args.seed)
    outputs = {
        "brca2_blinded_review_sheet.csv": review,
        "brca2_blinded_assay_readout_template.csv": assay,
        "brca2_blinded_unblinding_key.csv": key,
        "brca2_blinded_arm_balance.csv": balance,
    }
    for name, frame in outputs.items():
        path = out / name
        frame.to_csv(path, index=False)
        print(f"wrote {path}")
    readme = write_readme(repo, args.seed, review, assay, key, balance)
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
