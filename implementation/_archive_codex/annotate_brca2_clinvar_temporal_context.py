#!/usr/bin/env python3
"""Annotate BRCA2 interpretability candidates with an archived ClinVar status."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--old-snapshot",
        type=Path,
        default=repo_root / "data" / "variant" / "clinvar_archive" / "variant_summary_2025-01.txt.gz",
    )
    parser.add_argument("--old-label", default="2025-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
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
    pos = df["pos_hg38"] if "pos_hg38" in df.columns else df["pos"]
    return (
        df["chrom"].astype(str).str.replace("chr", "", regex=False)
        + ":"
        + pos.astype(int).astype(str)
        + ":"
        + df["ref"].astype(str)
        + ">"
        + df["alt"].astype(str)
    )


def is_clean_vus(clinsig: str) -> bool:
    text = clinsig.lower()
    return "uncertain significance" in text and "pathogenic" not in text and "benign" not in text


def classify_old_status(clinsig: str, matched: bool) -> str:
    if not matched:
        return "old_absent"
    text = clinsig.lower()
    if "conflicting" in text:
        return "old_conflicting"
    if is_clean_vus(clinsig):
        return "old_clean_vus"
    if "pathogenic" in text and "uncertain significance" not in text:
        return "old_pathogenic_or_likely_pathogenic"
    if "benign" in text and "uncertain significance" not in text:
        return "old_benign_or_likely_benign"
    if "uncertain significance" in text:
        return "old_vus_mixed"
    return "old_other"


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
                parts[idx["Chromosome"]].replace("chr", "")
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
            rec["old_clinsig_values"].add(parts[idx["ClinicalSignificance"]])
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
                "old_matched": True,
                "old_clinsig": clinsig,
                "old_status_class": classify_old_status(clinsig, True),
                "old_review_status": ";".join(sorted(x for x in rec["old_review_status_values"] if x)),
                "old_last_evaluated": ";".join(sorted(x for x in rec["old_last_evaluated_values"] if x)),
                "old_variation_ids": ";".join(sorted(x for x in rec["old_variation_ids"] if x)),
                "old_names": ";".join(sorted(x for x in rec["old_names"] if x)[:5]),
                "old_gene_symbols": ";".join(sorted(x for x in rec["old_gene_symbols"] if x)),
                "old_number_submitters_max": int(rec["old_number_submitters_max"]),
            }
        )
    return pd.DataFrame(rows)


def annotate_candidates(candidates: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["variant_key"] = variant_key_df(out)
    out = out.merge(old, on="variant_key", how="left")
    out["old_matched"] = out["old_matched"].map(lambda value: bool(value) if pd.notna(value) else False)
    out["old_clinsig"] = out["old_clinsig"].fillna("")
    out["old_status_class"] = [
        classify_old_status(clinsig, matched)
        for clinsig, matched in zip(out["old_clinsig"], out["old_matched"], strict=False)
    ]
    for col in [
        "old_review_status",
        "old_last_evaluated",
        "old_variation_ids",
        "old_names",
        "old_gene_symbols",
    ]:
        out[col] = out[col].fillna("")
    out["old_number_submitters_max"] = out["old_number_submitters_max"].fillna(0).astype(int)
    return out


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        row.update(
            {
                "n": len(sub),
                "n_lof": int(sub["label"].sum()) if "label" in sub else 0,
                "lof_rate": float(sub["label"].mean()) if "label" in sub and len(sub) else float("nan"),
                "n_old_matched": int(sub["old_matched"].sum()),
                "n_old_absent": int((sub["old_status_class"] == "old_absent").sum()),
                "n_old_clean_vus": int((sub["old_status_class"] == "old_clean_vus").sum()),
                "n_old_conflicting": int((sub["old_status_class"] == "old_conflicting").sum()),
                "n_old_vus_or_conflicting": int(sub["old_status_class"].isin(["old_clean_vus", "old_conflicting", "old_vus_mixed"]).sum()),
                "n_old_pathogenic": int((sub["old_status_class"] == "old_pathogenic_or_likely_pathogenic").sum()),
                "n_old_benign": int((sub["old_status_class"] == "old_benign_or_likely_benign").sum()),
                "median_function_score": float(sub["function_score"].median()) if "function_score" in sub else float("nan"),
                "median_dna_percentile": float(sub["dna_percentile"].median()) if "dna_percentile" in sub else float("nan"),
                "median_protein_percentile": float(sub["protein_percentile"].median()) if "protein_percentile" in sub else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_columns(df: pd.DataFrame, include_panel: bool = False) -> pd.DataFrame:
    cols = []
    if include_panel:
        cols.extend(["panel_rank", "panel_arm"])
    cols.extend(
        [
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
            "old_status_class",
            "old_clinsig",
            "old_review_status",
            "old_last_evaluated",
            "old_number_submitters_max",
            "same_aa_known_pathogenic",
            "same_residue_other_known_pathogenic",
            "same_residue_other_known_benign",
            "c.nom",
            "g.nom",
            "dbSNP.ID",
        ]
    )
    return df[[c for c in cols if c in df.columns]].copy()


def write_report(
    path: Path,
    old_label: str,
    snapshot: Path,
    candidate_summary: pd.DataFrame,
    old_status_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    panel_status_summary: pd.DataFrame,
    top_panel: pd.DataFrame,
) -> None:
    top_panel = select_columns(top_panel, include_panel=True).head(64)
    lines = [
        "# BRCA2 ClinVar Temporal Context",
        "",
        "## Purpose",
        "",
        f"Audit BRCA2 interpretability-review candidates against archived ClinVar `{old_label}` status. This is not a true prospective validation, but it separates candidates that were already known from candidates that were VUS/conflicting or absent in the older release.",
        "",
        "## Inputs",
        "",
        f"- Old snapshot: `{snapshot}`",
        "- Candidate table: `brca2_clinvar_interpretability_candidates.csv`",
        "- Review panel: `brca2_interpretability_review_panel.csv`",
        "",
        "## Candidate Tier Summary",
        "",
        candidate_summary.to_markdown(index=False),
        "",
        "## Candidate Old-Status Summary",
        "",
        old_status_summary.to_markdown(index=False),
        "",
        "## Panel Arm Summary",
        "",
        panel_summary.to_markdown(index=False),
        "",
        "## Panel Old-Status Summary",
        "",
        panel_status_summary.to_markdown(index=False),
        "",
        "## Panel Variants with Old Status",
        "",
        top_panel.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Old VUS/conflicting or absent rows are stronger follow-up candidates than rows already classified P/B in the old snapshot.",
        "- Exact same-AA and same-residue-other ClinVar evidence remain manual-review hooks, not automatic clinical reclassification.",
        "- A top-journal prospective version should freeze the model and feature selection before an old ClinVar cutoff, then test future reclassification or assay outcomes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = require_file(args.old_snapshot.resolve())

    candidates_path = require_file(out_dir / "brca2_clinvar_interpretability_candidates.csv")
    panel_path = require_file(out_dir / "brca2_interpretability_review_panel.csv")
    candidates = pd.read_csv(candidates_path)
    panel = pd.read_csv(panel_path)

    candidates["variant_key"] = variant_key_df(candidates)
    old = parse_old_snapshot(snapshot, set(candidates["variant_key"]))
    annotated = annotate_candidates(candidates.drop(columns=["variant_key"]), old)

    panel = panel.merge(
        annotated[
            [
                "id",
                "variant_key",
                "chrom",
                "pos_hg38",
                "pos",
                "ref",
                "alt",
                "old_matched",
                "old_clinsig",
                "old_status_class",
                "old_review_status",
                "old_last_evaluated",
                "old_variation_ids",
                "old_names",
                "old_gene_symbols",
                "old_number_submitters_max",
            ]
        ],
        on="id",
        how="left",
    )

    candidate_summary = summarize(annotated, ["review_tier", "review_direction"]).sort_values(
        ["review_tier", "review_direction"]
    )
    old_status_summary = summarize(annotated, ["review_tier", "old_status_class"]).sort_values(
        ["review_tier", "old_status_class"]
    )
    panel_summary = summarize(panel, ["panel_arm"]).sort_values("panel_arm")
    panel_status_summary = summarize(panel, ["panel_arm", "old_status_class"]).sort_values(
        ["panel_arm", "old_status_class"]
    )

    prefix = out_dir / "brca2_clinvar_temporal_context"
    annotated.to_csv(prefix.with_name(prefix.name + "_candidates.csv"), index=False)
    panel.to_csv(prefix.with_name(prefix.name + "_panel.csv"), index=False)
    candidate_summary.to_csv(prefix.with_name(prefix.name + "_candidate_tier_summary.csv"), index=False)
    old_status_summary.to_csv(prefix.with_name(prefix.name + "_candidate_old_status_summary.csv"), index=False)
    panel_summary.to_csv(prefix.with_name(prefix.name + "_panel_summary.csv"), index=False)
    panel_status_summary.to_csv(prefix.with_name(prefix.name + "_panel_old_status_summary.csv"), index=False)
    old.to_csv(prefix.with_name(prefix.name + "_old_snapshot_matches.csv"), index=False)
    write_report(
        prefix.with_suffix(".md"),
        args.old_label,
        snapshot,
        candidate_summary,
        old_status_summary,
        panel_summary,
        panel_status_summary,
        panel,
    )

    print(f"Wrote BRCA2 ClinVar temporal context outputs to {out_dir}")
    print(candidate_summary.to_string(index=False))
    print(panel_summary.to_string(index=False))


if __name__ == "__main__":
    main()
