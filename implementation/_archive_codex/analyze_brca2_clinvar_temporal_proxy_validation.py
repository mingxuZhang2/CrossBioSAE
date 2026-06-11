#!/usr/bin/env python3
"""BRCA2 archive-to-current ClinVar pseudo-temporal proxy validation.

This is a conservative local proxy analysis. It starts from BRCA2 VUS/review
candidates annotated with old ClinVar status, then looks up the same variants in
the current ClinVar GRCh38 VCF. It does not replace locked blinded review or
new assay outcomes.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OUT_DIR = Path("results/interpretability_applications")
OLD_UNRESOLVED = {"old_clean_vus", "old_conflicting", "old_vus_mixed", "old_absent"}
POSITIVE_TIER = "tier1_functional_lof_concordant_high"
NEGATIVE_TIER = "tier1_functional_benign_concordant_low"
POSITIVE_ARM = "pathogenic_review_high_confidence"
NEGATIVE_ARM = "benign_review_controls"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--current-vcf",
        type=Path,
        default=repo / "data" / "variant" / "clinvar_GRCh38.vcf.gz",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=repo / OUT_DIR / "brca2_clinvar_temporal_context_candidates.csv",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=repo / OUT_DIR / "brca2_clinvar_temporal_context_panel.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=repo / OUT_DIR)
    parser.add_argument("--output-prefix", default="brca2_clinvar_temporal_proxy_validation")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in info.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = unquote(value)
    return out


def normalize_sig(value: str) -> str:
    return value.replace("_", " ").replace("|", ";").strip()


def classify_current(clnsig: str, matched: bool) -> str:
    if not matched:
        return "current_absent"
    values = {normalize_sig(x).lower() for x in clnsig.split(";") if x}
    joined = ";".join(sorted(values))
    exact_pathogenic = {
        "pathogenic",
        "likely pathogenic",
        "pathogenic/likely pathogenic",
    }
    exact_benign = {
        "benign",
        "likely benign",
        "benign/likely benign",
    }
    if values & exact_pathogenic and not ("conflicting classifications of pathogenicity" in joined):
        return "current_pathogenic_or_likely_pathogenic"
    if values & exact_benign and not ("conflicting classifications of pathogenicity" in joined):
        return "current_benign_or_likely_benign"
    if "conflicting classifications of pathogenicity" in joined:
        return "current_conflicting"
    if "uncertain significance" in joined:
        return "current_vus"
    return "current_other"


def parse_current_vcf(vcf: Path, target_keys: set[str]) -> tuple[pd.DataFrame, str]:
    records: dict[str, dict[str, object]] = {}
    file_date = ""
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("##fileDate="):
                file_date = line.strip().split("=", 1)[1]
                continue
            if line.startswith("#"):
                continue
            chrom, pos, var_id, ref, alt, _qual, _filter, info_text = line.rstrip("\n").split("\t", 7)
            key = f"{chrom.replace('chr', '')}:{pos}:{ref}>{alt}"
            if key not in target_keys:
                continue
            info = parse_info(info_text)
            rec = records.setdefault(
                key,
                {
                    "variant_key": key,
                    "current_matched": True,
                    "current_vcf_ids": set(),
                    "current_clnsig_values": set(),
                    "current_clnrevstat_values": set(),
                    "current_clnhgvs_values": set(),
                    "current_geneinfo_values": set(),
                    "current_rs_values": set(),
                    "current_allele_ids": set(),
                },
            )
            rec["current_vcf_ids"].add(var_id)
            rec["current_clnsig_values"].add(info.get("CLNSIG", ""))
            rec["current_clnrevstat_values"].add(info.get("CLNREVSTAT", ""))
            rec["current_clnhgvs_values"].add(info.get("CLNHGVS", ""))
            rec["current_geneinfo_values"].add(info.get("GENEINFO", ""))
            rec["current_rs_values"].add(info.get("RS", ""))
            rec["current_allele_ids"].add(info.get("ALLELEID", ""))

    rows: list[dict[str, object]] = []
    for rec in records.values():
        clnsig = ";".join(sorted(x for x in rec["current_clnsig_values"] if x))
        rows.append(
            {
                "variant_key": rec["variant_key"],
                "current_matched": True,
                "current_clinsig": normalize_sig(clnsig),
                "current_status_class": classify_current(clnsig, True),
                "current_review_status": normalize_sig(";".join(sorted(x for x in rec["current_clnrevstat_values"] if x))),
                "current_clnhgvs": ";".join(sorted(x for x in rec["current_clnhgvs_values"] if x)[:5]),
                "current_geneinfo": ";".join(sorted(x for x in rec["current_geneinfo_values"] if x)),
                "current_rs": ";".join(sorted(x for x in rec["current_rs_values"] if x)),
                "current_allele_ids": ";".join(sorted(x for x in rec["current_allele_ids"] if x)),
                "current_snapshot_date": file_date,
            }
        )
    return pd.DataFrame(rows), file_date


def attach_current(df: pd.DataFrame, current: pd.DataFrame, file_date: str) -> pd.DataFrame:
    out = df.copy()
    if "variant_key" not in out.columns:
        out["variant_key"] = variant_key_df(out)
    out = out.merge(current, on="variant_key", how="left")
    out["current_matched"] = out["current_matched"].map(lambda value: bool(value) if pd.notna(value) else False)
    out["current_clinsig"] = out["current_clinsig"].fillna("")
    out["current_status_class"] = [
        classify_current(clinsig, matched)
        for clinsig, matched in zip(out["current_clinsig"], out["current_matched"], strict=False)
    ]
    for col in [
        "current_review_status",
        "current_clnhgvs",
        "current_geneinfo",
        "current_rs",
        "current_allele_ids",
        "current_snapshot_date",
    ]:
        out[col] = out[col].fillna(file_date if col == "current_snapshot_date" else "")
    out["old_unresolved_or_absent"] = out["old_status_class"].isin(OLD_UNRESOLVED)
    out["current_exact_pathogenic"] = out["current_status_class"].eq("current_pathogenic_or_likely_pathogenic")
    out["current_exact_benign"] = out["current_status_class"].eq("current_benign_or_likely_benign")
    out["current_exact_known"] = out["current_exact_pathogenic"] | out["current_exact_benign"]
    return out


def fisher_or_nan(table: list[list[int]], alternative: str = "greater") -> tuple[float, float]:
    try:
        odds, p = fisher_exact(table, alternative=alternative)
        return float(odds), float(p)
    except Exception:
        return float("nan"), float("nan")


def compare_groups(df: pd.DataFrame, group_col: str, positive_value: str, negative_value: str, scope: str) -> dict[str, object]:
    sub = df[df["old_unresolved_or_absent"]].copy()
    pos = sub[sub[group_col].eq(positive_value)]
    neg = sub[sub[group_col].eq(negative_value)]
    known_pos = pos[pos["current_exact_known"]]
    known_neg = neg[neg["current_exact_known"]]
    pos_path = int(known_pos["current_exact_pathogenic"].sum())
    pos_benign = int(known_pos["current_exact_benign"].sum())
    neg_path = int(known_neg["current_exact_pathogenic"].sum())
    neg_benign = int(known_neg["current_exact_benign"].sum())
    odds_known, p_known = fisher_or_nan([[pos_path, pos_benign], [neg_path, neg_benign]], "greater")
    pos_path_all = int(pos["current_exact_pathogenic"].sum())
    neg_path_all = int(neg["current_exact_pathogenic"].sum())
    odds_missing, p_missing = fisher_or_nan(
        [[pos_path_all, len(pos) - pos_path_all], [neg_path_all, len(neg) - neg_path_all]],
        "greater",
    )
    return {
        "scope": scope,
        "group_col": group_col,
        "positive_group": positive_value,
        "negative_group": negative_value,
        "n_positive_old_unresolved": int(len(pos)),
        "n_negative_old_unresolved": int(len(neg)),
        "n_positive_current_exact_known": int(len(known_pos)),
        "n_negative_current_exact_known": int(len(known_neg)),
        "positive_current_pathogenic_known_only": pos_path,
        "positive_current_benign_known_only": pos_benign,
        "negative_current_pathogenic_known_only": neg_path,
        "negative_current_benign_known_only": neg_benign,
        "positive_pathogenic_rate_known_only": float(pos_path / len(known_pos)) if len(known_pos) else np.nan,
        "negative_pathogenic_rate_known_only": float(neg_path / len(known_neg)) if len(known_neg) else np.nan,
        "known_only_fisher_odds": odds_known,
        "known_only_fisher_p_greater": p_known,
        "positive_current_pathogenic_missing_as_not_pathogenic": pos_path_all,
        "negative_current_pathogenic_missing_as_not_pathogenic": neg_path_all,
        "missing_as_not_pathogenic_fisher_odds": odds_missing,
        "missing_as_not_pathogenic_fisher_p_greater": p_missing,
    }


def summarize_status(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        row.update(
            {
                "n": int(len(sub)),
                "n_old_unresolved_or_absent": int(sub["old_unresolved_or_absent"].sum()),
                "n_current_matched": int(sub["current_matched"].sum()),
                "n_current_exact_known": int(sub["current_exact_known"].sum()),
                "n_current_pathogenic": int(sub["current_exact_pathogenic"].sum()),
                "n_current_benign": int(sub["current_exact_benign"].sum()),
                "n_current_vus": int(sub["current_status_class"].eq("current_vus").sum()),
                "n_current_conflicting": int(sub["current_status_class"].eq("current_conflicting").sum()),
                "n_current_absent": int(sub["current_status_class"].eq("current_absent").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.loc[:, columns].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    widths = [max(len(str(c)), *(len(str(v)) for v in sub[c])) for c in sub.columns]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(sub.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = [
        "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths))
        + " |"
        for row in sub.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + rows)


def write_report(
    out: Path,
    prefix: str,
    file_date: str,
    candidate_tests: pd.DataFrame,
    panel_tests: pd.DataFrame,
    candidate_status: pd.DataFrame,
    panel_status: pd.DataFrame,
) -> Path:
    lines = [
        "# BRCA2 ClinVar Temporal Proxy Validation",
        "",
        "## Purpose",
        "",
        "This report compares old-unresolved BRCA2 interpretation candidates against the current ClinVar VCF. It is a pseudo-temporal proxy only; it does not replace a locked blinded review or assay endpoint.",
        "",
        "## Current Snapshot",
        "",
        f"- current ClinVar VCF fileDate={file_date}",
        "",
        "## Candidate Tier Proxy Test",
        "",
        markdown_table(
            candidate_tests,
            [
                "scope",
                "n_positive_old_unresolved",
                "n_negative_old_unresolved",
                "n_positive_current_exact_known",
                "n_negative_current_exact_known",
                "positive_pathogenic_rate_known_only",
                "negative_pathogenic_rate_known_only",
                "known_only_fisher_p_greater",
                "missing_as_not_pathogenic_fisher_p_greater",
            ],
        ),
        "",
        "## Panel Proxy Test",
        "",
        markdown_table(
            panel_tests,
            [
                "scope",
                "n_positive_old_unresolved",
                "n_negative_old_unresolved",
                "n_positive_current_exact_known",
                "n_negative_current_exact_known",
                "positive_pathogenic_rate_known_only",
                "negative_pathogenic_rate_known_only",
                "known_only_fisher_p_greater",
                "missing_as_not_pathogenic_fisher_p_greater",
            ],
        ),
        "",
        "## Candidate Current Status By Tier",
        "",
        markdown_table(
            candidate_status,
            [
                "review_tier",
                "n",
                "n_old_unresolved_or_absent",
                "n_current_exact_known",
                "n_current_pathogenic",
                "n_current_benign",
                "n_current_vus",
                "n_current_conflicting",
            ],
        ),
        "",
        "## Panel Current Status By Arm",
        "",
        markdown_table(
            panel_status,
            [
                "panel_arm",
                "n",
                "n_old_unresolved_or_absent",
                "n_current_exact_known",
                "n_current_pathogenic",
                "n_current_benign",
                "n_current_vus",
                "n_current_conflicting",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- All-candidate old-unresolved tier1 comparison provides independent temporal-proxy support for BRCA2 VUS triage.",
        "- The 64-row blinded panel still lacks enough current exact outcomes for a primary clinical-utility claim.",
        "- Any clinical reclassification claim still requires locked manual-review outcomes or assay readouts.",
        "",
        "## Output Files",
        "",
        f"- `{prefix}_candidates.csv`",
        f"- `{prefix}_panel.csv`",
        f"- `{prefix}_candidate_tests.csv`",
        f"- `{prefix}_panel_tests.csv`",
        f"- `{prefix}_candidate_status_by_tier.csv`",
        f"- `{prefix}_panel_status_by_arm.csv`",
        "",
    ]
    path = out / f"{prefix}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(require_file(args.candidates))
    panel = pd.read_csv(require_file(args.panel))
    if "variant_key" not in candidates.columns:
        candidates["variant_key"] = variant_key_df(candidates)
    if "variant_key" not in panel.columns:
        panel["variant_key"] = variant_key_df(panel)
    target_keys = set(candidates["variant_key"]) | set(panel["variant_key"])
    current, file_date = parse_current_vcf(require_file(args.current_vcf), target_keys)
    annotated_candidates = attach_current(candidates, current, file_date)
    annotated_panel = attach_current(panel, current, file_date)

    candidate_tests = pd.DataFrame(
        [
            compare_groups(
                annotated_candidates,
                "review_tier",
                POSITIVE_TIER,
                NEGATIVE_TIER,
                "all_candidates_old_unresolved_tier1",
            )
        ]
    )
    panel_tests = pd.DataFrame(
        [
            compare_groups(
                annotated_panel,
                "panel_arm",
                POSITIVE_ARM,
                NEGATIVE_ARM,
                "prospective_panel_old_unresolved_arms",
            )
        ]
    )
    candidate_status = summarize_status(annotated_candidates, ["review_tier"]).sort_values("review_tier")
    panel_status = summarize_status(annotated_panel, ["panel_arm"]).sort_values("panel_arm")

    prefix = args.output_prefix
    annotated_candidates.to_csv(out / f"{prefix}_candidates.csv", index=False)
    annotated_panel.to_csv(out / f"{prefix}_panel.csv", index=False)
    candidate_tests.to_csv(out / f"{prefix}_candidate_tests.csv", index=False)
    panel_tests.to_csv(out / f"{prefix}_panel_tests.csv", index=False)
    candidate_status.to_csv(out / f"{prefix}_candidate_status_by_tier.csv", index=False)
    panel_status.to_csv(out / f"{prefix}_panel_status_by_arm.csv", index=False)
    current.to_csv(out / f"{prefix}_current_vcf_matches.csv", index=False)
    report = write_report(out, prefix, file_date, candidate_tests, panel_tests, candidate_status, panel_status)
    print(f"wrote {out / f'{prefix}_candidate_tests.csv'}")
    print(f"wrote {report}")
    print(candidate_tests.to_string(index=False))
    print(panel_tests.to_string(index=False))


if __name__ == "__main__":
    main()
