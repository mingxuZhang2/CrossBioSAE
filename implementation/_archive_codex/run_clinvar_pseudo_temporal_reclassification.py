#!/usr/bin/env python3
"""
Pseudo-temporal ClinVar reclassification validation.

This script asks a limited but useful question:

Among variants that are currently in the local P/B scored benchmark, which ones
were Uncertain significance in an older ClinVar variant_summary snapshot, and do
the current mechanism scores separate those that later became P/LP from those
that later became B/LB?

This is not a leakage-free prospective experiment because the SAE feature
selection and current labels are not frozen at the old date. It is still a
stronger validation than same-release ClinVar because the label transition is
anchored to an archived ClinVar snapshot.
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--old-snapshot",
        type=Path,
        default=repo_root
        / "data"
        / "variant"
        / "clinvar_archive"
        / "variant_summary_2025-01.txt.gz",
    )
    parser.add_argument("--old-label", default="2025-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--top-frac", type=float, default=0.10)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def variant_key_df(df: pd.DataFrame) -> pd.Series:
    return (
        df["chrom"].astype(str)
        + ":"
        + df["pos"].astype(str)
        + ":"
        + df["ref"].astype(str)
        + ">"
        + df["alt"].astype(str)
    )


def is_clean_vus(clinsig: str) -> bool:
    text = clinsig.lower()
    if "uncertain significance" not in text:
        return False
    if "pathogenic" in text or "benign" in text:
        return False
    return True


def is_pathogenic_text(clinsig: str) -> bool:
    text = clinsig.lower()
    return "pathogenic" in text and "uncertain significance" not in text


def is_benign_text(clinsig: str) -> bool:
    text = clinsig.lower()
    return "benign" in text and "uncertain significance" not in text


def safe_auc(y: np.ndarray, s: np.ndarray, larger_is_pathogenic: bool = True) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_pathogenic else -s[ok]
    return float(roc_auc_score(y[ok], score))


def safe_auprc(y: np.ndarray, s: np.ndarray, larger_is_pathogenic: bool = True) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_pathogenic else -s[ok]
    return float(average_precision_score(y[ok], score))


def load_scored_current(out_dir: Path) -> pd.DataFrame:
    known = pd.read_csv(require_file(out_dir / "known_variant_novel_scores.csv"))
    collagen = pd.read_csv(require_file(out_dir / "collagen_gly_known_variant_scores.csv"))
    key_cols = ["chrom", "pos", "ref", "alt", "gene", "pchange", "label"]
    for df in (known, collagen):
        df["chrom"] = df["chrom"].astype(str)
        df["pos"] = df["pos"].astype(int)
        df["ref"] = df["ref"].astype(str)
        df["alt"] = df["alt"].astype(str)
        df["gene"] = df["gene"].fillna("").astype(str)
        df["pchange"] = df["pchange"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)
    current = known.merge(
        collagen[
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "gene",
                "pchange",
                "label",
                "collagen_gly_feature_score",
                "collagen_gly_features_active",
                "structural_feature_score",
                "structural_features_active",
            ]
        ],
        on=key_cols,
        how="left",
    )
    current["variant_key"] = variant_key_df(current)
    return current


def parse_old_snapshot(snapshot: Path, target_keys: set[str]) -> pd.DataFrame:
    records: dict[str, dict[str, object]] = {}
    with gzip.open(snapshot, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        needed = [
            "Assembly",
            "Chromosome",
            "PositionVCF",
            "ReferenceAlleleVCF",
            "AlternateAlleleVCF",
            "ClinicalSignificance",
            "ReviewStatus",
            "NumberSubmitters",
            "LastEvaluated",
            "VariationID",
            "Name",
            "GeneSymbol",
        ]
        missing = [name for name in needed if name not in idx]
        if missing:
            raise ValueError(f"Old ClinVar snapshot is missing columns: {missing}")

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx.values()) or parts[idx["Assembly"]] != "GRCh38":
                continue
            key = (
                parts[idx["Chromosome"]]
                + ":"
                + parts[idx["PositionVCF"]]
                + ":"
                + parts[idx["ReferenceAlleleVCF"]]
                + ">"
                + parts[idx["AlternateAlleleVCF"]]
            )
            if key not in target_keys:
                continue
            rec = records.setdefault(
                key,
                {
                    "variant_key": key,
                    "old_clinsig_values": set(),
                    "old_review_status_values": set(),
                    "old_last_evaluated_values": set(),
                    "old_variation_ids": set(),
                    "old_names": set(),
                    "old_gene_symbols": set(),
                    "old_number_submitters_max": 0,
                },
            )
            clinsig = parts[idx["ClinicalSignificance"]]
            rec["old_clinsig_values"].add(clinsig)
            rec["old_review_status_values"].add(parts[idx["ReviewStatus"]])
            rec["old_last_evaluated_values"].add(parts[idx["LastEvaluated"]])
            rec["old_variation_ids"].add(parts[idx["VariationID"]])
            rec["old_names"].add(parts[idx["Name"]])
            rec["old_gene_symbols"].add(parts[idx["GeneSymbol"]])
            try:
                rec["old_number_submitters_max"] = max(
                    int(rec["old_number_submitters_max"]),
                    int(parts[idx["NumberSubmitters"]] or 0),
                )
            except ValueError:
                pass

    rows = []
    for rec in records.values():
        clinsig = ";".join(sorted(x for x in rec["old_clinsig_values"] if x))
        rows.append(
            {
                "variant_key": rec["variant_key"],
                "old_clinsig": clinsig,
                "old_clean_vus": is_clean_vus(clinsig),
                "old_had_pathogenic_text": is_pathogenic_text(clinsig),
                "old_had_benign_text": is_benign_text(clinsig),
                "old_review_status": ";".join(sorted(x for x in rec["old_review_status_values"] if x)),
                "old_last_evaluated": ";".join(sorted(x for x in rec["old_last_evaluated_values"] if x)),
                "old_variation_ids": ";".join(sorted(x for x in rec["old_variation_ids"] if x)),
                "old_names": ";".join(sorted(x for x in rec["old_names"] if x)[:5]),
                "old_gene_symbols": ";".join(sorted(x for x in rec["old_gene_symbols"] if x)),
                "old_number_submitters_max": int(rec["old_number_submitters_max"]),
            }
        )
    return pd.DataFrame(rows)


def metric_tables(df: pd.DataFrame, top_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {
        "all_old_vus_to_current_known": np.ones(len(df), dtype=bool),
        "mapped_gene": df["gene"].ne("").to_numpy(),
        "structural_gene": df["is_structural_gene"].astype(bool).to_numpy(),
        "collagen_gene": df["is_collagen_gene"].astype(bool).to_numpy(),
        "collagen_gly": (df["is_collagen_gene"].astype(bool) & df["is_gly_sub"].astype(bool)).to_numpy(),
    }
    metric_specs = [
        ("novel_path_score", True),
        ("collagen_gly_feature_score", True),
        ("structural_feature_score", True),
        # In the local benchmark files these baseline scores are stored in a
        # direction where lower values are generally more damaging.
        ("CADD", False),
        ("ESM-1b", False),
        ("GPN-MSA", False),
    ]

    rows = []
    enrich_rows = []
    for group, mask in groups.items():
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        y = sub["label"].to_numpy(dtype=int)
        row = {
            "group": group,
            "n": len(sub),
            "n_current_pathogenic": int(y.sum()),
            "current_path_rate": float(y.mean()) if len(y) else float("nan"),
        }
        for metric, larger in metric_specs:
            if metric not in sub:
                continue
            vals = sub[metric].to_numpy(dtype=float)
            row[f"auroc_{metric}"] = safe_auc(y, vals, larger)
            row[f"auprc_{metric}"] = safe_auprc(y, vals, larger)
        rows.append(row)

        if len(sub) >= 20 and sub["label"].nunique() == 2:
            for metric, larger in metric_specs:
                if metric not in sub:
                    continue
                vals = sub[metric].to_numpy(dtype=float)
                ok = np.isfinite(vals)
                if ok.sum() < 20:
                    continue
                score = vals if larger else -vals
                cutoff = np.quantile(score[ok], 1.0 - top_frac)
                high = score >= cutoff
                table = [
                    [int((high & sub["label"].eq(1)).sum()), int((high & sub["label"].eq(0)).sum())],
                    [int((~high & sub["label"].eq(1)).sum()), int((~high & sub["label"].eq(0)).sum())],
                ]
                odds, pval = fisher_exact(table)
                enrich_rows.append(
                    {
                        "group": group,
                        "metric": metric,
                        "top_frac": top_frac,
                        "cutoff": float(cutoff),
                        "top_n": int(high.sum()),
                        "top_pathogenic": table[0][0],
                        "top_benign": table[0][1],
                        "background_pathogenic": table[1][0],
                        "background_benign": table[1][1],
                        "odds_ratio": float(odds),
                        "fisher_p": float(pval),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(enrich_rows)


def write_report(
    path: Path,
    old_label: str,
    snapshot: Path,
    matched: pd.DataFrame,
    metrics: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> None:
    top = matched.sort_values(
        ["novel_path_score", "collagen_gly_feature_score", "CADD"],
        ascending=[False, False, False],
    ).head(30)
    top_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "pchange",
        "label",
        "novel_path_score",
        "collagen_gly_feature_score",
        "structural_feature_score",
        "CADD",
        "ESM-1b",
        "GPN-MSA",
        "old_clinsig",
        "old_review_status",
    ]
    top = top[top_cols].copy()
    for col in ["novel_path_score", "collagen_gly_feature_score", "structural_feature_score", "CADD", "ESM-1b", "GPN-MSA"]:
        top[col] = top[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    lines = [
        "# ClinVar Pseudo-Temporal Reclassification Validation",
        "",
        "## Purpose",
        "",
        f"Use archived ClinVar `{old_label}` `variant_summary` rows as an old snapshot and ask whether variants that were clean VUS then, but are P/B in the current local scored benchmark now, are separated by mechanism scores.",
        "",
        "## Inputs",
        "",
        f"- Old snapshot: `{snapshot}`",
        "- Current scored benchmark: `known_variant_novel_scores.csv` plus collagen-gly feature scores.",
        "- Matching key: GRCh38 chrom:PositionVCF:ReferenceAlleleVCF>AlternateAlleleVCF.",
        "",
        "## Caveat",
        "",
        "This is pseudo-temporal, not fully prospective: current labels and feature selection were not frozen at the old date. A top-journal version should rerun feature selection using only pre-cutoff labels.",
        "",
        "## Headline Counts",
        "",
        f"- Current scored P/B variants matched in old snapshot: {int(matched['old_clinsig'].notna().sum()):,}.",
        f"- Old clean VUS that are current P/B in this benchmark: {len(matched):,}.",
        f"- Current pathogenic among old clean VUS: {int(matched['label'].sum()):,}.",
        f"- Current benign among old clean VUS: {int((matched['label'] == 0).sum()):,}.",
        "",
        "## Metric Summary",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Top-Decile Enrichment",
        "",
        enrichment.sort_values(["group", "fisher_p"]).to_markdown(index=False) if not enrichment.empty else "_No enrichment rows._",
        "",
        "## Highest Mechanism-Score Old-VUS Variants",
        "",
        top.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "If mechanism scores separate old VUS that later became pathogenic from those that became benign, this becomes a strong temporal validation direction. If the effect is weak, use the result to narrow the VUS application to collagen-gly or feature-specific cohorts and proceed to variant-level disease databases.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    old_snapshot = args.old_snapshot.resolve()
    require_file(old_snapshot)

    current = load_scored_current(out_dir)
    old = parse_old_snapshot(old_snapshot, set(current["variant_key"]))
    joined = current.merge(old, on="variant_key", how="left")
    matched = joined[joined["old_clean_vus"].fillna(False)].copy()

    metrics, enrichment = metric_tables(matched, args.top_frac)

    out_prefix = f"clinvar_pseudo_temporal_{args.old_label}"
    matched.to_csv(out_dir / f"{out_prefix}_scores.csv", index=False)
    metrics.to_csv(out_dir / f"{out_prefix}_metric_summary.csv", index=False)
    enrichment.to_csv(out_dir / f"{out_prefix}_top_decile_enrichment.csv", index=False)
    old.to_csv(out_dir / f"{out_prefix}_old_snapshot_matches.csv", index=False)
    write_report(
        out_dir / f"{out_prefix}_reclassification.md",
        args.old_label,
        old_snapshot,
        matched,
        metrics,
        enrichment,
    )

    print(f"Wrote ClinVar pseudo-temporal outputs to {out_dir}")
    print(metrics.to_string(index=False))
    if not enrichment.empty:
        print(enrichment.sort_values(["group", "fisher_p"]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
