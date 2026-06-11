#!/usr/bin/env python
"""Analyze locked BRCA2 blinded review/assay outcomes.

This script is deliberately conservative. It first checks whether the blinded
review sheet or assay readout is complete enough to unblind. If not, it writes a
status report and does not merge hidden arm/model/SGE evidence into row-level
outputs. Once locked outcomes are present, it computes the pre-specified primary
endpoint and arm-level summaries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


OUT_DIR = Path("results/interpretability_applications")

PRIMARY_POSITIVE_ARM = "prospective_pathogenic_review"
PRIMARY_NEGATIVE_ARM = "prospective_benign_controls"

PATHOGENIC_REVIEW_LABELS = {
    "pathogenic",
    "likely_pathogenic",
    "pathogenic_support",
    "pathogenic_supporting",
    "supports_pathogenic",
    "pathogenic_evidence",
    "p",
    "lp",
}
BENIGN_REVIEW_LABELS = {
    "benign",
    "likely_benign",
    "benign_support",
    "benign_supporting",
    "supports_benign",
    "benign_evidence",
    "b",
    "lb",
}
UNCERTAIN_LABELS = {
    "uncertain",
    "vus",
    "insufficient",
    "insufficient_evidence",
    "conflicting",
    "not_classified",
    "na",
    "none",
}

ASSAY_LOF_LABELS = {
    "lof",
    "loss_of_function",
    "loss_function",
    "nonfunctional",
    "non_functional",
    "impaired",
    "damaging",
    "deleterious",
    "pathogenic",
    "low_function",
    "functionally_abnormal",
}
ASSAY_FUNCTIONAL_LABELS = {
    "functional",
    "functionally_normal",
    "normal",
    "neutral",
    "benign",
    "tolerated",
    "wild_type_like",
    "wt_like",
    "wt",
}
QC_PASS_LABELS = {"pass", "passed", "ok", "valid", "yes", "y", "1", "true"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review-sheet", type=Path, default=None)
    parser.add_argument("--assay-sheet", type=Path, default=None)
    parser.add_argument("--unblinding-key", type=Path, default=None)
    parser.add_argument("--min-review-confidence", type=float, default=1.0)
    parser.add_argument("--output-prefix", default="brca2_blinded_validation")
    return parser.parse_args()


def resolve_path(repo: Path, path: Path | None, default_name: str) -> Path:
    if path is None:
        return repo / OUT_DIR / default_name
    if path.is_absolute():
        return path
    return repo / path


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def is_filled(value: object) -> bool:
    return bool(normalize_label(value))


def map_review_binary(value: object) -> float:
    label = normalize_label(value)
    if not label or label in UNCERTAIN_LABELS:
        return np.nan
    if label in PATHOGENIC_REVIEW_LABELS:
        return 1.0
    if label in BENIGN_REVIEW_LABELS:
        return 0.0
    return np.nan


def map_assay_binary(value: object) -> float:
    label = normalize_label(value)
    if not label or label in UNCERTAIN_LABELS:
        return np.nan
    if label in ASSAY_LOF_LABELS:
        return 1.0
    if label in ASSAY_FUNCTIONAL_LABELS:
        return 0.0
    return np.nan


def map_qc_pass(value: object) -> bool:
    return normalize_label(value) in QC_PASS_LABELS


def expected_binary(value: object) -> float:
    label = normalize_label(value)
    if label in {"lof", "pathogenic", "pathogenic_strong", "pathogenic_very_strong"}:
        return 1.0
    if label in {"functional", "benign", "benign_strong", "benign_very_strong"}:
        return 0.0
    return np.nan


def count_nonempty(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].map(is_filled).sum())


def prepare_review(review: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    df = review.copy()
    if "manual_classification" not in df.columns:
        df["manual_classification"] = ""
    if "manual_confidence_1_to_5" not in df.columns:
        df["manual_confidence_1_to_5"] = np.nan
    df["review_binary_pathogenic_support"] = df["manual_classification"].map(map_review_binary)
    df["review_confidence_numeric"] = pd.to_numeric(df["manual_confidence_1_to_5"], errors="coerce")
    df["review_row_locked"] = df["manual_classification"].map(is_filled) & (
        df["review_confidence_numeric"] >= min_confidence
    )
    df["review_row_binary_usable"] = df["review_row_locked"] & df["review_binary_pathogenic_support"].notna()
    keep = [
        "blinded_id",
        "manual_classification",
        "manual_evidence_codes",
        "manual_confidence_1_to_5",
        "review_notes",
        "review_binary_pathogenic_support",
        "review_confidence_numeric",
        "review_row_locked",
        "review_row_binary_usable",
    ]
    return df[[col for col in keep if col in df.columns]]


def prepare_assay(assay: pd.DataFrame) -> pd.DataFrame:
    df = assay.copy()
    if "binary_external_call" not in df.columns:
        df["binary_external_call"] = ""
    if "assay_qc_status" not in df.columns:
        df["assay_qc_status"] = ""
    if "mean_function_score" not in df.columns:
        df["mean_function_score"] = np.nan
    df["assay_binary_lof"] = df["binary_external_call"].map(map_assay_binary)
    df["assay_mean_function_score"] = pd.to_numeric(df["mean_function_score"], errors="coerce")
    df["assay_qc_pass"] = df["assay_qc_status"].map(map_qc_pass)
    df["assay_row_locked"] = df["assay_qc_pass"] & (
        df["assay_binary_lof"].notna() | df["assay_mean_function_score"].notna()
    )
    df["assay_row_binary_usable"] = df["assay_qc_pass"] & df["assay_binary_lof"].notna()
    df["assay_row_continuous_usable"] = df["assay_qc_pass"] & df["assay_mean_function_score"].notna()
    keep = [
        "blinded_id",
        "replicate_1_function_score",
        "replicate_2_function_score",
        "replicate_3_function_score",
        "mean_function_score",
        "binary_external_call",
        "assay_qc_status",
        "assay_notes",
        "assay_binary_lof",
        "assay_mean_function_score",
        "assay_qc_pass",
        "assay_row_locked",
        "assay_row_binary_usable",
        "assay_row_continuous_usable",
    ]
    return df[[col for col in keep if col in df.columns]]


def build_status(
    review: pd.DataFrame,
    assay: pd.DataFrame,
    review_ready_binary: bool,
    assay_ready_binary: bool,
    assay_ready_continuous: bool,
    primary_source: str,
) -> pd.DataFrame:
    review_any = count_nonempty(review, "manual_classification")
    review_conf = int(pd.to_numeric(review.get("manual_confidence_1_to_5", pd.Series(dtype=float)), errors="coerce").notna().sum())
    review_binary = int(review.get("manual_classification", pd.Series(dtype=object)).map(map_review_binary).notna().sum())
    assay_binary = int(assay.get("binary_external_call", pd.Series(dtype=object)).map(map_assay_binary).notna().sum())
    assay_mean = int(pd.to_numeric(assay.get("mean_function_score", pd.Series(dtype=float)), errors="coerce").notna().sum())
    assay_qc = int(assay.get("assay_qc_status", pd.Series(dtype=object)).map(map_qc_pass).sum())

    if primary_source:
        status = "ready_for_unblinded_endpoint_analysis"
    elif assay_ready_continuous:
        status = "ready_for_secondary_continuous_only_no_binary_primary"
    elif review_any == 0 and assay_binary == 0 and assay_mean == 0:
        status = "not_ready_no_locked_outcomes"
    else:
        status = "not_ready_partial_or_nonbinary_outcomes"

    return pd.DataFrame(
        [
            {
                "validation_status": status,
                "primary_outcome_source": primary_source if primary_source else "none",
                "review_rows_total": len(review),
                "review_rows_with_manual_classification": review_any,
                "review_rows_with_binary_classification": review_binary,
                "review_rows_with_confidence": review_conf,
                "review_ready_binary_all_rows": review_ready_binary,
                "assay_rows_total": len(assay),
                "assay_rows_qc_pass": assay_qc,
                "assay_rows_with_binary_call": assay_binary,
                "assay_rows_with_mean_function_score": assay_mean,
                "assay_ready_binary_all_rows": assay_ready_binary,
                "assay_ready_continuous_all_rows": assay_ready_continuous,
            }
        ]
    )


def choose_primary_source(review_prepared: pd.DataFrame, assay_prepared: pd.DataFrame) -> tuple[str, bool, bool, bool]:
    review_ready_binary = bool(review_prepared["review_row_binary_usable"].all()) if len(review_prepared) else False
    assay_ready_binary = bool(assay_prepared["assay_row_binary_usable"].all()) if len(assay_prepared) else False
    assay_ready_continuous = (
        bool(assay_prepared["assay_row_continuous_usable"].all()) if len(assay_prepared) else False
    )
    primary_source = ""
    if assay_ready_binary:
        primary_source = "assay_binary_external_call"
    elif review_ready_binary:
        primary_source = "manual_review_binary_classification"
    return primary_source, review_ready_binary, assay_ready_binary, assay_ready_continuous


def summarize_primary(df: pd.DataFrame, source_col: str, source_name: str) -> pd.DataFrame:
    primary = df[df["panel_arm"].isin([PRIMARY_POSITIVE_ARM, PRIMARY_NEGATIVE_ARM])].copy()
    pos = primary[primary["panel_arm"].eq(PRIMARY_POSITIVE_ARM)]
    neg = primary[primary["panel_arm"].eq(PRIMARY_NEGATIVE_ARM)]
    pos_values = pos[source_col].dropna().astype(int)
    neg_values = neg[source_col].dropna().astype(int)
    if len(pos_values) != len(pos) or len(neg_values) != len(neg) or len(pos) == 0 or len(neg) == 0:
        return pd.DataFrame(
            [
                {
                    "endpoint_status": "not_evaluable_missing_primary_binary_outcomes",
                    "primary_outcome_source": source_name,
                    "positive_arm": PRIMARY_POSITIVE_ARM,
                    "negative_arm": PRIMARY_NEGATIVE_ARM,
                    "positive_arm_n": len(pos),
                    "positive_arm_n_binary": len(pos_values),
                    "negative_arm_n": len(neg),
                    "negative_arm_n_binary": len(neg_values),
                    "fisher_p_greater": np.nan,
                }
            ]
        )

    pos_positive = int(pos_values.sum())
    neg_positive = int(neg_values.sum())
    pos_negative = int(len(pos_values) - pos_positive)
    neg_negative = int(len(neg_values) - neg_positive)
    odds, pval = fisher_exact([[pos_positive, pos_negative], [neg_positive, neg_negative]], alternative="greater")
    return pd.DataFrame(
        [
            {
                "endpoint_status": "evaluated",
                "primary_outcome_source": source_name,
                "positive_arm": PRIMARY_POSITIVE_ARM,
                "negative_arm": PRIMARY_NEGATIVE_ARM,
                "positive_arm_n": len(pos_values),
                "positive_arm_external_positive": pos_positive,
                "positive_arm_external_negative": pos_negative,
                "positive_arm_external_positive_rate": pos_positive / len(pos_values),
                "negative_arm_n": len(neg_values),
                "negative_arm_external_positive": neg_positive,
                "negative_arm_external_negative": neg_negative,
                "negative_arm_external_positive_rate": neg_positive / len(neg_values),
                "rate_difference": pos_positive / len(pos_values) - neg_positive / len(neg_values),
                "fisher_oddsratio": float(odds),
                "fisher_p_greater": float(pval),
                "passes_p_lt_0.05": bool(pval < 0.05),
                "passes_p_lt_0.01": bool(pval < 0.01),
            }
        ]
    )


def summarize_continuous_primary(df: pd.DataFrame) -> dict[str, Any]:
    if "assay_mean_function_score" not in df:
        return {"continuous_endpoint_status": "not_available"}
    primary = df[df["panel_arm"].isin([PRIMARY_POSITIVE_ARM, PRIMARY_NEGATIVE_ARM])].copy()
    pos = primary.loc[primary["panel_arm"].eq(PRIMARY_POSITIVE_ARM), "assay_mean_function_score"].dropna()
    neg = primary.loc[primary["panel_arm"].eq(PRIMARY_NEGATIVE_ARM), "assay_mean_function_score"].dropna()
    if len(pos) == 0 or len(neg) == 0:
        return {
            "continuous_endpoint_status": "not_evaluable_missing_assay_scores",
            "positive_arm_n_continuous": len(pos),
            "negative_arm_n_continuous": len(neg),
        }
    stat = mannwhitneyu(pos, neg, alternative="less")
    return {
        "continuous_endpoint_status": "evaluated",
        "positive_arm_n_continuous": len(pos),
        "positive_arm_median_mean_function_score": float(pos.median()),
        "negative_arm_n_continuous": len(neg),
        "negative_arm_median_mean_function_score": float(neg.median()),
        "mannwhitney_pathogenic_less_than_benign_p": float(stat.pvalue),
    }


def summarize_arms(df: pd.DataFrame, source_col: str | None) -> pd.DataFrame:
    rows = []
    for arm, group in df.groupby("panel_arm", dropna=False):
        row: dict[str, Any] = {
            "panel_arm": arm,
            "n": len(group),
            "n_expected_lof": int(group["expected_binary"].eq(1).sum()),
            "n_expected_functional": int(group["expected_binary"].eq(0).sum()),
            "n_sge_lof": int(pd.to_numeric(group.get("sge_label", pd.Series(dtype=float)), errors="coerce").eq(1).sum()),
            "median_internal_function_score": float(pd.to_numeric(group.get("function_score", pd.Series(dtype=float)), errors="coerce").median()),
            "median_dna_percentile": float(pd.to_numeric(group.get("dna_percentile", pd.Series(dtype=float)), errors="coerce").median()),
            "median_protein_percentile": float(pd.to_numeric(group.get("protein_percentile", pd.Series(dtype=float)), errors="coerce").median()),
        }
        if source_col:
            external = group[source_col].dropna().astype(int)
            row.update(
                {
                    "n_external_binary": len(external),
                    "n_external_positive": int(external.sum()),
                    "external_positive_rate": float(external.mean()) if len(external) else np.nan,
                }
            )
            expected = group.loc[group[source_col].notna(), "expected_binary"]
            if len(external) and expected.notna().all():
                ext = group.loc[group[source_col].notna(), source_col].astype(int)
                exp = expected.astype(int)
                row.update(
                    {
                        "external_vs_expected_tp": int(((ext == 1) & (exp == 1)).sum()),
                        "external_vs_expected_tn": int(((ext == 0) & (exp == 0)).sum()),
                        "external_vs_expected_fp": int(((ext == 1) & (exp == 0)).sum()),
                        "external_vs_expected_fn": int(((ext == 0) & (exp == 1)).sum()),
                        "external_vs_expected_accuracy": float((ext == exp).mean()),
                    }
                )
        if "assay_mean_function_score" in group:
            scores = group["assay_mean_function_score"].dropna()
            row["n_assay_continuous"] = len(scores)
            row["median_external_assay_mean_function_score"] = float(scores.median()) if len(scores) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("panel_arm").reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
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
                if np.isnan(value):
                    text = "NA"
                elif abs(value) < 0.001 and value != 0:
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4g}"
            else:
                text = str(value)
            vals.append(text.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_not_ready_outputs(repo: Path, prefix: str, status: pd.DataFrame) -> None:
    out = repo / OUT_DIR
    arm_summary = pd.DataFrame(
        [
            {
                "panel_arm": "not_unblinded",
                "summary_status": status.iloc[0]["validation_status"],
                "note": "Blinded review/assay outcomes are not complete; arm-level analysis intentionally not run.",
            }
        ]
    )
    primary = pd.DataFrame(
        [
            {
                "endpoint_status": status.iloc[0]["validation_status"],
                "primary_outcome_source": "none",
                "note": "No locked binary external outcome is available for the pre-specified Fisher endpoint.",
            }
        ]
    )
    status.to_csv(out / f"{prefix}_status.csv", index=False)
    arm_summary.to_csv(out / f"{prefix}_arm_summary.csv", index=False)
    primary.to_csv(out / f"{prefix}_primary_endpoint.csv", index=False)

    row = status.iloc[0]
    lines = [
        "# BRCA2 Blinded Review/Assay Validation Report",
        "",
        "## Status",
        "",
        f"- validation_status: {row['validation_status']}",
        "- unblinding_performed: false",
        "- primary_endpoint_evaluated: false",
        "",
        "## Locked Outcome Check",
        "",
        f"- Review rows with manual classification: {row['review_rows_with_manual_classification']} / {row['review_rows_total']}",
        f"- Review rows with binary classification: {row['review_rows_with_binary_classification']} / {row['review_rows_total']}",
        f"- Review rows with confidence: {row['review_rows_with_confidence']} / {row['review_rows_total']}",
        f"- Assay rows with QC pass: {row['assay_rows_qc_pass']} / {row['assay_rows_total']}",
        f"- Assay rows with binary calls: {row['assay_rows_with_binary_call']} / {row['assay_rows_total']}",
        f"- Assay rows with mean function scores: {row['assay_rows_with_mean_function_score']} / {row['assay_rows_total']}",
        "",
        "## Pre-Specified Endpoint",
        "",
        "Primary endpoint is a one-sided Fisher exact test comparing external LOF/pathogenic-support calls in `prospective_pathogenic_review` versus `prospective_benign_controls`. It will run only after all rows have a locked binary assay call, or all rows have a locked binary manual review call.",
        "",
        "## Claim Boundary",
        "",
        "Current supported claim remains BRCA2 review/assay triage design. Clinical utility or VUS reclassification is not supported until locked external outcomes are available.",
        "",
    ]
    (out / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_ready_outputs(
    repo: Path,
    prefix: str,
    status: pd.DataFrame,
    merged: pd.DataFrame,
    primary: pd.DataFrame,
    arm_summary: pd.DataFrame,
    primary_source: str,
) -> None:
    out = repo / OUT_DIR
    status.to_csv(out / f"{prefix}_status.csv", index=False)
    arm_summary.to_csv(out / f"{prefix}_arm_summary.csv", index=False)
    primary.to_csv(out / f"{prefix}_primary_endpoint.csv", index=False)

    primary_row = primary.iloc[0]
    arm_cols = [
        "panel_arm",
        "n",
        "n_expected_lof",
        "n_external_binary",
        "n_external_positive",
        "external_positive_rate",
        "external_vs_expected_accuracy",
        "n_assay_continuous",
        "median_external_assay_mean_function_score",
    ]
    arm_cols = [col for col in arm_cols if col in arm_summary.columns]
    lines = [
        "# BRCA2 Blinded Review/Assay Validation Report",
        "",
        "## Status",
        "",
        f"- validation_status: {status.iloc[0]['validation_status']}",
        "- unblinding_performed: true",
        f"- primary_outcome_source: {primary_source}",
        f"- primary_endpoint_status: {primary_row.get('endpoint_status', 'NA')}",
        "",
        "## Primary Endpoint",
        "",
        markdown_table(primary, list(primary.columns)),
        "",
        "## Arm Summary",
        "",
        markdown_table(arm_summary, arm_cols),
        "",
        "## Split-Mechanism And Conflict Controls",
        "",
        "The split-mechanism arm tests whether external functional evidence remains positive while DNA-high/protein-low and protein-high/DNA-low explanations separate. The model-conflict arm is an error-analysis control and should remain functional/benign-like if the triage rule is specific.",
        "",
        "## Claim Boundary",
        "",
        "A positive primary endpoint would support review/assay utility of the BRCA2 triage panel. It still would not constitute automatic clinical reclassification without expert adjudication.",
        "",
    ]
    continuous = summarize_continuous_primary(merged)
    if continuous.get("continuous_endpoint_status") != "not_available":
        lines[lines.index("## Arm Summary")] = "## Secondary Continuous Endpoint\n\n" + markdown_table(
            pd.DataFrame([continuous]), list(continuous.keys())
        ) + "\n\n## Arm Summary"
    (out / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(repo: Path, review_path: Path, assay_path: Path, key_path: Path, min_confidence: float, prefix: str) -> None:
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    review = require_csv(review_path)
    assay = require_csv(assay_path)
    key = require_csv(key_path)
    review_prepared = prepare_review(review, min_confidence)
    assay_prepared = prepare_assay(assay)
    primary_source, review_ready_binary, assay_ready_binary, assay_ready_continuous = choose_primary_source(
        review_prepared, assay_prepared
    )
    status = build_status(
        review,
        assay,
        review_ready_binary=review_ready_binary,
        assay_ready_binary=assay_ready_binary,
        assay_ready_continuous=assay_ready_continuous,
        primary_source=primary_source,
    )

    if not primary_source and not assay_ready_continuous:
        write_not_ready_outputs(repo, prefix, status)
        return

    key_cols = [
        "blinded_id",
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
    ]
    merged = key[[col for col in key_cols if col in key.columns]].merge(review_prepared, on="blinded_id", how="left")
    merged = merged.merge(assay_prepared, on="blinded_id", how="left")
    merged["expected_binary"] = merged["expected_function_class"].map(expected_binary)

    source_col = ""
    source_name = "none"
    if primary_source == "assay_binary_external_call":
        source_col = "assay_binary_lof"
        source_name = primary_source
    elif primary_source == "manual_review_binary_classification":
        source_col = "review_binary_pathogenic_support"
        source_name = primary_source

    if source_col:
        primary = summarize_primary(merged, source_col=source_col, source_name=source_name)
    else:
        primary = pd.DataFrame(
            [
                {
                    "endpoint_status": "not_evaluable_no_binary_primary_source",
                    "primary_outcome_source": "none",
                    "note": "Continuous assay scores are complete, but no binary external call is available.",
                }
            ]
        )
    arm_summary = summarize_arms(merged, source_col=source_col or None)
    write_ready_outputs(repo, prefix, status, merged, primary, arm_summary, source_name)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    review_path = resolve_path(repo, args.review_sheet, "brca2_blinded_review_sheet.csv")
    assay_path = resolve_path(repo, args.assay_sheet, "brca2_blinded_assay_readout_template.csv")
    key_path = resolve_path(repo, args.unblinding_key, "brca2_blinded_unblinding_key.csv")
    run_analysis(
        repo=repo,
        review_path=review_path,
        assay_path=assay_path,
        key_path=key_path,
        min_confidence=args.min_review_confidence,
        prefix=args.output_prefix,
    )
    print(f"wrote {repo / OUT_DIR / (args.output_prefix + '_status.csv')}")
    print(f"wrote {repo / OUT_DIR / (args.output_prefix + '_arm_summary.csv')}")
    print(f"wrote {repo / OUT_DIR / (args.output_prefix + '_primary_endpoint.csv')}")
    print(f"wrote {repo / OUT_DIR / (args.output_prefix + '_report.md')}")


if __name__ == "__main__":
    main()
