#!/usr/bin/env python
"""Design a blinded BAP1/RAD51C third-gene review/assay panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def add_common_columns(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    out = df.copy()
    out["gene"] = gene
    if "variant_key" not in out.columns:
        if {"chrom", "pos_hg38", "ref", "alt"}.issubset(out.columns):
            out["variant_key"] = (
                out["chrom"].astype(str)
                + ":"
                + out["pos_hg38"].astype(str)
                + ":"
                + out["ref"].astype(str)
                + ":"
                + out["alt"].astype(str)
            )
        elif "chrom_pos_ref_alt" in out.columns:
            out["variant_key"] = out["chrom_pos_ref_alt"].astype(str).str.replace("_", ":", regex=False)
        else:
            out["variant_key"] = out["id"].astype(str)
    if "external_lof_score" not in out.columns:
        out["external_lof_score"] = -pd.to_numeric(out.get("function_score", np.nan), errors="coerce")
    if "clinvar_simple" not in out.columns:
        out["clinvar_simple"] = out.get("clinvar_clinical_significance_slim", "NA")
    if "clinvar_clnsig" not in out.columns:
        out["clinvar_clnsig"] = out.get("clinvar_clinical_significance", out["clinvar_simple"])
    if "clinvar_review_status" not in out.columns:
        out["clinvar_review_status"] = ""
    if "clinvar_disease" not in out.columns:
        out["clinvar_disease"] = ""
    if "gnomad_af" not in out.columns:
        out["gnomad_af"] = out.get("gnomAD_AF_v3", out.get("gnomad_af_v3_numeric", ""))
    if "is_missense" not in out.columns:
        out["is_missense"] = out.get("consequence", "").astype(str).str.lower().eq("missense")
    return out


def format_panel_rows(df: pd.DataFrame, arm: str, role: str, quota: int) -> pd.DataFrame:
    out = df.head(quota).copy()
    out["panel_arm"] = arm
    out["panel_role"] = role
    return out


def select_bap1_review(bap1: pd.DataFrame, candidates: pd.DataFrame, quota: int) -> pd.DataFrame:
    sub = add_common_columns(bap1[bap1["id"].isin(candidates["id"])], "BAP1")
    sub["_not_gnomad"] = sub.get("is_in_gnomad", "").astype(str).str.upper().ne("Y").astype(int)
    sub["_domain"] = sub.get("domains", "-").astype(str).ne("-").astype(int)
    sub = sub.sort_values(
        ["_not_gnomad", "_domain", "external_lof_score"],
        ascending=[False, False, False],
    )
    return format_panel_rows(sub, "BAP1_depleted_vus_review", "candidate_review", quota)


def select_bap1_pathogenic_controls(bap1: pd.DataFrame, quota: int) -> pd.DataFrame:
    slim = bap1["clinvar_clinical_significance_slim"].fillna("")
    label = pd.to_numeric(bap1["label"], errors="coerce")
    sub = add_common_columns(
        bap1[slim.eq("Pathogenic/Likely pathogenic") & label.eq(1.0)].copy(),
        "BAP1",
    )
    sub = sub.sort_values("external_lof_score", ascending=False)
    return format_panel_rows(sub, "BAP1_known_pathogenic_controls", "positive_control", quota)


def select_bap1_benign_controls(bap1: pd.DataFrame, quota: int) -> pd.DataFrame:
    slim = bap1["clinvar_clinical_significance_slim"].fillna("")
    label = pd.to_numeric(bap1["label"], errors="coerce")
    sub = add_common_columns(
        bap1[
            slim.eq("Benign/Likely benign")
            & label.eq(0.0)
            & bap1["functional_classification"].astype(str).eq("unchanged")
        ].copy(),
        "BAP1",
    )
    sub["_abs_function_score"] = pd.to_numeric(sub["function_score"], errors="coerce").abs()
    sub = sub.sort_values(["_abs_function_score", "external_lof_score"], ascending=[True, True])
    return format_panel_rows(sub, "BAP1_known_benign_controls", "negative_control", quota)


def select_rad51c_review(rad_candidates: pd.DataFrame, quota: int) -> pd.DataFrame:
    sub = add_common_columns(rad_candidates, "RAD51C")
    priority = {"Conflicting": 3, "Uncertain": 2, "Unobserved": 1}
    sub["_clinvar_priority"] = sub["clinvar_simple"].map(priority).fillna(0)
    sub["_domain"] = sub.get("domains", "-").astype(str).ne("-").astype(int)
    sub = sub.sort_values(
        ["_clinvar_priority", "_domain", "external_lof_score"],
        ascending=[False, False, False],
    )
    return format_panel_rows(sub, "RAD51C_depleted_review", "candidate_review", quota)


def select_rad51c_pathogenic_controls(rad: pd.DataFrame, quota: int) -> pd.DataFrame:
    sub = add_common_columns(
        rad[rad["clinvar_simple"].eq("Pathogenic/Likely_pathogenic") & pd.to_numeric(rad["label"], errors="coerce").eq(1.0)].copy(),
        "RAD51C",
    )
    sub = sub.sort_values("external_lof_score", ascending=False)
    return format_panel_rows(sub, "RAD51C_known_pathogenic_controls", "positive_control", quota)


def select_rad51c_benign_controls(rad: pd.DataFrame, quota: int) -> pd.DataFrame:
    sub = add_common_columns(
        rad[
            rad["clinvar_simple"].eq("Benign/Likely_benign")
            & pd.to_numeric(rad["label"], errors="coerce").eq(0.0)
            & rad["functional_classification"].astype(str).eq("unchanged")
        ].copy(),
        "RAD51C",
    )
    sub = sub.sort_values("external_lof_score", ascending=True)
    return format_panel_rows(sub, "RAD51C_known_benign_controls", "negative_control", quota)


def build_panel(repo: Path, review_quota: int, control_quota: int) -> pd.DataFrame:
    out = repo / OUT_DIR
    bap1 = read_csv(out / "bap1_sge_variants.csv")
    bap1_candidates = read_csv(out / "bap1_sge_external_anchors_vus_candidates.csv")
    rad = read_csv(out / "rad51c_sge_external_anchors_annotated.csv")
    rad_candidates = read_csv(out / "rad51c_sge_external_anchors_candidate_review.csv")

    parts = [
        select_bap1_review(bap1, bap1_candidates, review_quota),
        select_bap1_pathogenic_controls(bap1, control_quota),
        select_bap1_benign_controls(bap1, control_quota),
        select_rad51c_review(rad_candidates, review_quota),
        select_rad51c_pathogenic_controls(rad, control_quota),
        select_rad51c_benign_controls(rad, control_quota),
    ]
    panel = pd.concat(parts, ignore_index=True)
    panel.insert(0, "panel_id", [f"TG{idx:03d}" for idx in range(1, len(panel) + 1)])
    panel.insert(1, "panel_rank", np.arange(1, len(panel) + 1))
    panel["checkpoint_status_at_design"] = "pending_evo2_llr_checkpoint"
    return panel


def summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gene, arm, role), sub in panel.groupby(["gene", "panel_arm", "panel_role"], sort=False):
        rows.append(
            {
                "gene": gene,
                "panel_arm": arm,
                "panel_role": role,
                "n": int(len(sub)),
                "n_depleted_label": int(pd.to_numeric(sub.get("label", np.nan), errors="coerce").fillna(0).sum()),
                "depleted_rate": float(pd.to_numeric(sub.get("label", np.nan), errors="coerce").mean()),
                "median_external_lof_score": float(pd.to_numeric(sub["external_lof_score"], errors="coerce").median()),
                "n_clinvar_pathogenic": int(sub["clinvar_simple"].astype(str).str.contains("Pathogenic", case=False, na=False).sum()),
                "n_clinvar_benign": int(sub["clinvar_simple"].astype(str).str.contains("Benign", case=False, na=False).sum()),
                "n_clinvar_uncertain_or_conflicting": int(
                    sub["clinvar_simple"].astype(str).str.contains("Uncertain|Conflicting", case=False, na=False).sum()
                ),
                "n_unobserved": int(sub["clinvar_simple"].astype(str).str.contains("Unobserved", case=False, na=False).sum()),
                "n_missense": int(as_bool(sub.get("is_missense", pd.Series(False, index=sub.index))).sum()),
            }
        )
    return pd.DataFrame(rows)


def compact_columns(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "panel_id",
        "panel_rank",
        "gene",
        "panel_arm",
        "panel_role",
        "id",
        "variant_key",
        "HGVSc",
        "HGVSp",
        "consequence",
        "functional_classification",
        "label",
        "function_score",
        "external_lof_score",
        "clinvar_simple",
        "clinvar_clnsig",
        "clinvar_review_status",
        "clinvar_disease",
        "variation_id",
        "clinvar_rs",
        "gnomad_af",
        "domains",
        "checkpoint_status_at_design",
    ]
    for col in cols:
        if col not in panel.columns:
            panel[col] = ""
    return panel[cols].copy()


def blinded(panel: pd.DataFrame) -> pd.DataFrame:
    out = compact_columns(panel)
    drop_cols = [
        "panel_arm",
        "panel_role",
        "functional_classification",
        "label",
        "function_score",
        "external_lof_score",
        "checkpoint_status_at_design",
    ]
    out = out.drop(columns=drop_cols)
    out["review_call"] = ""
    out["review_confidence"] = ""
    out["review_notes"] = ""
    return out


def assay_template(panel: pd.DataFrame) -> pd.DataFrame:
    out = blinded(panel)[["panel_id", "gene", "variant_key", "HGVSc", "HGVSp", "consequence"]].copy()
    out["assay_batch"] = ""
    out["replicate_count"] = ""
    out["raw_function_score"] = ""
    out["normalized_function_score"] = ""
    out["binary_call"] = ""
    out["assay_qc_status"] = ""
    out["assay_notes"] = ""
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def write_report(out_dir: Path, panel: pd.DataFrame, arm_summary: pd.DataFrame) -> Path:
    out = out_dir / "third_gene_review_assay_panel.md"
    rows_by_role = panel["panel_role"].value_counts().to_dict()
    lines = [
        "# Third-Gene Review/Assay Panel",
        "",
        "## Purpose",
        "",
        "This 64-row BAP1/RAD51C panel converts the third-gene benchmark anchors into an executable review or assay application. It is intentionally designed before Evo2/ESM checkpoint completion, so it tests SGE/clinical-anchor utility now and can be re-ranked later with CrossBioSAE mechanism strata.",
        "",
        "## Composition",
        "",
        f"- total rows: {len(panel)}",
        f"- candidate-review rows: {rows_by_role.get('candidate_review', 0)}",
        f"- positive-control rows: {rows_by_role.get('positive_control', 0)}",
        f"- negative-control rows: {rows_by_role.get('negative_control', 0)}",
        "- genes: BAP1 and RAD51C",
        "- checkpoint status at design: pending Evo2 LLR/checkpoint metrics",
        "",
        "## Arm Summary",
        "",
        md_table(arm_summary),
        "",
        "## Primary Endpoint",
        "",
        "Candidate-review arms should show higher expert pathogenic-review or assay-LOF rates than negative controls. Positive controls should recover high LOF/pathogenic-review rates, and negative controls should remain mostly functional/benign.",
        "",
        "## Claim Boundary",
        "",
        "This panel supports review/assay prioritization. It does not support clinical reclassification or third-gene CrossBioSAE validation until locked review/assay outcomes and checkpoint-derived mechanism scores exist.",
        "",
        "## Outputs",
        "",
        "- third_gene_review_assay_panel_internal.csv",
        "- third_gene_review_assay_panel_blinded.csv",
        "- third_gene_review_assay_panel_assay_template.csv",
        "- third_gene_review_assay_panel_summary.csv",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review-quota", type=int, default=16)
    parser.add_argument("--control-quota", type=int, default=8)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = repo / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(repo, args.review_quota, args.control_quota)
    internal = compact_columns(panel)
    arm_summary = summary(panel)
    internal.to_csv(out_dir / "third_gene_review_assay_panel_internal.csv", index=False)
    blinded(panel).to_csv(out_dir / "third_gene_review_assay_panel_blinded.csv", index=False)
    assay_template(panel).to_csv(out_dir / "third_gene_review_assay_panel_assay_template.csv", index=False)
    arm_summary.to_csv(out_dir / "third_gene_review_assay_panel_summary.csv", index=False)
    report = write_report(out_dir, panel, arm_summary)
    print(f"wrote {out_dir / 'third_gene_review_assay_panel_internal.csv'}")
    print(f"wrote {out_dir / 'third_gene_review_assay_panel_blinded.csv'}")
    print(f"wrote {out_dir / 'third_gene_review_assay_panel_assay_template.csv'}")
    print(f"wrote {out_dir / 'third_gene_review_assay_panel_summary.csv'}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
