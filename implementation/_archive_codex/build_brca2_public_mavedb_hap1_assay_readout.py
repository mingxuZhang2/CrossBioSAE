#!/usr/bin/env python3
"""Build a public HAP1 MaveDB assay readout for the BRCA2 blinded panel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("results/interpretability_applications")
PRIMARY_POSITIVE_ARM = "prospective_pathogenic_review"
PRIMARY_NEGATIVE_ARM = "prospective_benign_controls"
DEFAULT_HAP1_URN = "urn:mavedb:00001225-a-1"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--mavedb-urn", default=DEFAULT_HAP1_URN)
    parser.add_argument("--output-prefix", default="brca2_public_mavedb_hap1")
    return parser.parse_args()


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def safe_auc(pos_scores: pd.Series, neg_scores: pd.Series) -> float:
    pos = pd.to_numeric(pos_scores, errors="coerce").dropna().astype(float)
    neg = pd.to_numeric(neg_scores, errors="coerce").dropna().astype(float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    scores = np.r_[pos.to_numpy(), neg.to_numpy()]
    labels = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    if len(np.unique(scores)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def best_youden_threshold(local: pd.DataFrame) -> dict[str, Any]:
    sub = local.copy()
    sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
    sub["external_lof_score"] = pd.to_numeric(sub["external_lof_score"], errors="coerce")
    sub = sub[sub["label"].notna() & sub["external_lof_score"].notna()].copy()
    if sub.empty or sub["label"].nunique() < 2:
        raise ValueError("HAP1 local calibration needs both positive and negative BRCA2 labels.")

    labels = sub["label"].astype(int).to_numpy()
    scores = sub["external_lof_score"].astype(float).to_numpy()
    candidates: list[dict[str, Any]] = []
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = int(((pred) & (labels == 1)).sum())
        fn = int(((~pred) & (labels == 1)).sum())
        tn = int(((~pred) & (labels == 0)).sum())
        fp = int(((pred) & (labels == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        youden = sensitivity + specificity - 1
        candidates.append(
            {
                "threshold_external_lof_score": float(threshold),
                "youden_j": float(youden),
                "balanced_accuracy": float((sensitivity + specificity) / 2),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["youden_j"],
            row["balanced_accuracy"],
            row["specificity"],
            row["sensitivity"],
        ),
    )
    best.update(
        {
            "mavedb_urn": str(sub["mavedb_urn"].iloc[0]),
            "local_matches_n": int(len(sub)),
            "local_pathogenic_n": int(labels.sum()),
            "local_functional_n": int(len(labels) - labels.sum()),
            "external_lof_auroc_vs_local_sge_label": float(roc_auc_score(labels, scores)),
            "binary_rule": "external_lof_score >= threshold_external_lof_score",
        }
    )
    return best


def load_inputs(repo: Path, urn: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = repo / OUT_DIR
    template = require_csv(out / "brca2_blinded_assay_readout_template.csv")
    key = require_csv(out / "brca2_blinded_unblinding_key.csv")
    local = require_csv(out / "brca2_mavedb_external_assay_local_matches.csv")
    panel = require_csv(out / "brca2_mavedb_external_assay_panel_matches.csv")
    local = local[local["mavedb_urn"].eq(urn) & local["match_type"].eq("cdna")].copy()
    panel = panel[panel["mavedb_urn"].eq(urn) & panel["match_type"].eq("cdna")].copy()
    panel = panel.drop_duplicates(["id", "mavedb_urn"], keep="first")
    return template, key, local, panel


def build_readout(template: pd.DataFrame, key: pd.DataFrame, panel: pd.DataFrame, threshold: float, urn: str) -> pd.DataFrame:
    key_map = key[["blinded_id", "id", "panel_arm"]].copy()
    matched = key_map.merge(
        panel[["id", "score", "external_lof_score", "orientation", "mavedb_title"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    matched["score"] = pd.to_numeric(matched["score"], errors="coerce")
    matched["external_lof_score"] = pd.to_numeric(matched["external_lof_score"], errors="coerce")
    matched["binary_external_call"] = np.where(
        matched["external_lof_score"].notna() & (matched["external_lof_score"] >= threshold),
        "lof",
        np.where(matched["external_lof_score"].notna(), "functional", ""),
    )

    readout = template.copy()
    for col in [
        "replicate_1_function_score",
        "replicate_2_function_score",
        "replicate_3_function_score",
        "mean_function_score",
        "binary_external_call",
        "assay_qc_status",
        "assay_notes",
    ]:
        if col not in readout.columns:
            readout[col] = ""
    for col in ["binary_external_call", "assay_qc_status", "assay_notes"]:
        readout[col] = readout[col].astype("object").where(readout[col].notna(), "")

    fill_cols = matched.set_index("blinded_id")
    for idx, row in readout.iterrows():
        blinded_id = row["blinded_id"]
        if blinded_id not in fill_cols.index:
            continue
        match = fill_cols.loc[blinded_id]
        if pd.notna(match["score"]):
            # Raw HAP1 score is used as function score because lower raw score means more LOF.
            readout.loc[idx, "mean_function_score"] = float(match["score"])
            readout.loc[idx, "binary_external_call"] = match["binary_external_call"]
            readout.loc[idx, "assay_qc_status"] = "pass"
            readout.loc[idx, "assay_notes"] = (
                f"public_mavedb_hap1_sge; urn={urn}; lower_raw_score_more_lof; "
                f"external_lof_score={float(match['external_lof_score']):.6g}; "
                f"binary_threshold_external_lof_score={threshold:.6g}"
            )
        else:
            readout.loc[idx, "assay_qc_status"] = "missing_public_hap1_match"
            readout.loc[idx, "assay_notes"] = f"no_cdna_match_in_public_hap1_mavedb_score_set; urn={urn}"
    return readout


def matched_panel(key: pd.DataFrame, panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    key_cols = [
        "blinded_id",
        "panel_arm",
        "assay_role",
        "id",
        "aa_change",
        "c_nom",
        "g_nom",
        "brca2_domain",
        "expected_function_class",
        "sge_label",
        "function_score",
        "discordance_category",
        "dna_percentile",
        "protein_percentile",
    ]
    merged = key[[col for col in key_cols if col in key.columns]].merge(
        panel[
            [
                "id",
                "score",
                "external_lof_score",
                "mavedb_urn",
                "mavedb_title",
                "orientation",
                "orientation_multiplier",
            ]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["external_lof_score"] = pd.to_numeric(merged["external_lof_score"], errors="coerce")
    merged["hap1_matched"] = merged["external_lof_score"].notna()
    merged["binary_external_lof"] = np.where(
        merged["hap1_matched"], (merged["external_lof_score"] >= threshold).astype(int), np.nan
    )
    merged["binary_external_call"] = np.where(
        merged["binary_external_lof"].eq(1),
        "lof",
        np.where(merged["binary_external_lof"].eq(0), "functional", ""),
    )
    return merged


def endpoint_rows(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("pathogenic_review_vs_benign_controls", PRIMARY_NEGATIVE_ARM),
        ("pathogenic_review_vs_model_conflict_controls", "prospective_model_conflict_controls"),
        ("pathogenic_review_vs_split_mechanism_tests", "prospective_split_mechanism_tests"),
    ]
    for comparison, negative_arm in comparisons:
        pos = matched[matched["panel_arm"].eq(PRIMARY_POSITIVE_ARM) & matched["hap1_matched"]].copy()
        neg = matched[matched["panel_arm"].eq(negative_arm) & matched["hap1_matched"]].copy()
        pos_bin = pos["binary_external_lof"].dropna().astype(int)
        neg_bin = neg["binary_external_lof"].dropna().astype(int)
        if len(pos_bin) and len(neg_bin):
            odds, fisher_p = fisher_exact(
                [[int(pos_bin.sum()), int(len(pos_bin) - pos_bin.sum())], [int(neg_bin.sum()), int(len(neg_bin) - neg_bin.sum())]],
                alternative="greater",
            )
        else:
            odds, fisher_p = float("nan"), float("nan")
        pos_score = pos["external_lof_score"].dropna()
        neg_score = neg["external_lof_score"].dropna()
        if len(pos_score) >= 2 and len(neg_score) >= 2:
            mw_p = float(mannwhitneyu(pos_score.astype(float), neg_score.astype(float), alternative="greater").pvalue)
        else:
            mw_p = float("nan")
        rows.append(
            {
                "comparison": comparison,
                "positive_arm": PRIMARY_POSITIVE_ARM,
                "negative_arm": negative_arm,
                "n_positive_group": int(len(pos)),
                "n_negative_group": int(len(neg)),
                "positive_binary_lof": int(pos_bin.sum()) if len(pos_bin) else 0,
                "positive_binary_functional": int(len(pos_bin) - pos_bin.sum()) if len(pos_bin) else 0,
                "positive_binary_lof_rate": float(pos_bin.mean()) if len(pos_bin) else float("nan"),
                "negative_binary_lof": int(neg_bin.sum()) if len(neg_bin) else 0,
                "negative_binary_functional": int(len(neg_bin) - neg_bin.sum()) if len(neg_bin) else 0,
                "negative_binary_lof_rate": float(neg_bin.mean()) if len(neg_bin) else float("nan"),
                "binary_rate_difference": float(pos_bin.mean() - neg_bin.mean()) if len(pos_bin) and len(neg_bin) else float("nan"),
                "fisher_oddsratio": float(odds),
                "fisher_p_greater": float(fisher_p),
                "external_lof_auroc": safe_auc(pos_score, neg_score),
                "positive_mean_external_lof_score": float(pos_score.mean()) if len(pos_score) else float("nan"),
                "negative_mean_external_lof_score": float(neg_score.mean()) if len(neg_score) else float("nan"),
                "mannwhitney_p_greater": mw_p,
            }
        )
    return pd.DataFrame(rows)


def arm_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in matched.groupby("panel_arm", dropna=False):
        sub = group[group["hap1_matched"]].copy()
        binary = sub["binary_external_lof"].dropna().astype(int)
        rows.append(
            {
                "panel_arm": arm,
                "panel_rows_total": int(len(group)),
                "hap1_matched_rows": int(len(sub)),
                "hap1_match_rate": float(len(sub) / len(group)) if len(group) else float("nan"),
                "binary_lof": int(binary.sum()) if len(binary) else 0,
                "binary_functional": int(len(binary) - binary.sum()) if len(binary) else 0,
                "binary_lof_rate": float(binary.mean()) if len(binary) else float("nan"),
                "mean_external_lof_score": float(sub["external_lof_score"].mean()) if len(sub) else float("nan"),
                "median_external_lof_score": float(sub["external_lof_score"].median()) if len(sub) else float("nan"),
                "mean_raw_hap1_score": float(sub["score"].mean()) if len(sub) else float("nan"),
                "median_raw_hap1_score": float(sub["score"].median()) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("panel_arm").reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                text = fmt(value, 4, sci=abs(float(value)) < 0.001 and value != 0)
            else:
                text = str(value)
            vals.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    prefix: str,
    calibration: dict[str, Any],
    matched: pd.DataFrame,
    endpoints: pd.DataFrame,
    arms: pd.DataFrame,
) -> None:
    primary = endpoints[endpoints["comparison"].eq("pathogenic_review_vs_benign_controls")]
    primary_row = primary.iloc[0].to_dict() if not primary.empty else {}
    total_rows = len(matched)
    matched_rows = int(matched["hap1_matched"].sum())
    lines = [
        "# BRCA2 Public HAP1 MaveDB Assay Readout",
        "",
        "## Purpose",
        "",
        "This report maps the public HAP1 BRCA2 MaveDB score set onto the 64-row BRCA2 blinded follow-up panel and evaluates the matched subset as an external functional-assay proxy.",
        "",
        "## Calibration",
        "",
        f"- score_set: {calibration['mavedb_urn']}",
        f"- local_matches_n: {calibration['local_matches_n']}",
        f"- local_pathogenic_n: {calibration['local_pathogenic_n']}",
        f"- external_lof_auc_vs_local_sge_label: {fmt(calibration['external_lof_auroc_vs_local_sge_label'], 4)}",
        f"- binary_rule: external_lof_score >= {fmt(calibration['threshold_external_lof_score'], 6)}",
        f"- calibration_sensitivity: {fmt(calibration['sensitivity'], 4)}",
        f"- calibration_specificity: {fmt(calibration['specificity'], 4)}",
        "",
        "## Coverage",
        "",
        f"- panel_rows_total: {total_rows}",
        f"- hap1_matched_rows: {matched_rows}",
        f"- unmatched_rows: {total_rows - matched_rows}",
        "",
        "## Primary Matched-Subset Endpoint",
        "",
        f"- binary endpoint: {int(primary_row.get('positive_binary_lof', 0))}/{int(primary_row.get('n_positive_group', 0))} pathogenic-review rows are HAP1 LOF versus {int(primary_row.get('negative_binary_lof', 0))}/{int(primary_row.get('n_negative_group', 0))} benign controls; Fisher one-sided p={fmt(primary_row.get('fisher_p_greater'), 4)}",
        f"- continuous endpoint: external-lof AUROC={fmt(primary_row.get('external_lof_auroc'), 4)}; Mann-Whitney one-sided p={fmt(primary_row.get('mannwhitney_p_greater'), sci=True)}",
        "",
        "## Endpoint Table",
        "",
        markdown_table(endpoints),
        "",
        "## Arm Summary",
        "",
        markdown_table(arms),
        "",
        "## Claim Boundary",
        "",
        "This is a public external functional-assay proxy. It is not a locked blinded wet-lab or expert-review outcome. The pre-specified blinded analyzer should remain not-ready unless all 64 rows receive locked binary review or assay outcomes.",
        "",
        "## Outputs",
        "",
        f"- {prefix}_assay_readout.csv",
        f"- {prefix}_matched_panel.csv",
        f"- {prefix}_calibration.csv",
        f"- {prefix}_matched_subset_endpoints.csv",
        f"- {prefix}_arm_summary.csv",
    ]
    (out / f"{prefix}_assay_readout.md").write_text("\n".join(lines), encoding="utf-8")


def run(repo: Path, urn: str, prefix: str) -> None:
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    template, key, local, panel = load_inputs(repo, urn)
    calibration = best_youden_threshold(local)
    threshold = float(calibration["threshold_external_lof_score"])
    readout = build_readout(template, key, panel, threshold, urn)
    matched = matched_panel(key, panel, threshold)
    endpoints = endpoint_rows(matched)
    arms = arm_summary(matched)

    readout.to_csv(out / f"{prefix}_assay_readout.csv", index=False)
    matched.to_csv(out / f"{prefix}_matched_panel.csv", index=False)
    pd.DataFrame([calibration]).to_csv(out / f"{prefix}_calibration.csv", index=False)
    endpoints.to_csv(out / f"{prefix}_matched_subset_endpoints.csv", index=False)
    arms.to_csv(out / f"{prefix}_arm_summary.csv", index=False)
    write_report(out, prefix, calibration, matched, endpoints, arms)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    run(repo, args.mavedb_urn, args.output_prefix)
    out = repo / OUT_DIR
    print(f"wrote {out / (args.output_prefix + '_assay_readout.csv')}")
    print(f"wrote {out / (args.output_prefix + '_matched_subset_endpoints.csv')}")
    print(f"wrote {out / (args.output_prefix + '_assay_readout.md')}")


if __name__ == "__main__":
    main()
