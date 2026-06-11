#!/usr/bin/env python3
"""Analyze external anchors for the RAD51C SGE fallback benchmark.

This validates the functional-map benchmark itself. It is not a CrossBioSAE
checkpoint result; Evo2/ESM scores are evaluated by the downstream checkpoint
once the GPU jobs finish.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--variants",
        type=Path,
        default=repo_root / "data" / "variant" / "rad51c" / "rad51c_grch38_variants.csv",
    )
    parser.add_argument(
        "--clinvar-vcf",
        type=Path,
        default=repo_root / "data" / "variant" / "clinvar_GRCh38.vcf.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / OUT_DIR)
    parser.add_argument("--output-prefix", default="rad51c_sge_external_anchors")
    return parser.parse_args()


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in info.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = unquote(value)
    return out


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return unquote(str(value)).replace("_", " ").replace("|", ";").strip()


def clinvar_sig_tokens(clnsig: object) -> set[str]:
    text = normalize_text(clnsig).replace("/", ";").replace(",", ";")
    return {part.strip().lower() for part in text.split(";") if part.strip()}


def classify_clinvar(clnsig: object, matched: bool) -> str:
    if not matched:
        return "Unobserved"
    tokens = clinvar_sig_tokens(clnsig)
    joined = ";".join(sorted(tokens))
    has_conflict = "conflicting classifications of pathogenicity" in joined or "conflicting interpretations of pathogenicity" in joined
    has_uncertain = "uncertain significance" in tokens
    has_pathogenic = bool(tokens & {"pathogenic", "likely pathogenic"})
    has_benign = bool(tokens & {"benign", "likely benign"})
    if has_conflict or (has_pathogenic and has_benign):
        return "Conflicting"
    if has_pathogenic:
        return "Pathogenic/Likely_pathogenic"
    if has_benign:
        return "Benign/Likely_benign"
    if has_uncertain:
        return "Uncertain"
    return "Other"


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


def read_clinvar_subset(vcf: Path, target_keys: set[str]) -> tuple[pd.DataFrame, str]:
    target_pos: dict[str, set[int]] = {}
    for key in target_keys:
        chrom, pos, _allele = key.split(":", 2)
        target_pos.setdefault(chrom, set()).add(int(pos))

    file_date = ""
    records: dict[str, dict[str, Any]] = {}
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("##fileDate="):
                file_date = line.strip().split("=", 1)[1]
                continue
            if line.startswith("#"):
                continue
            chrom, pos_text, var_id, ref, alts, _qual, _flt, info_text = line.rstrip("\n").split("\t", 7)
            chrom = chrom.replace("chr", "")
            pos = int(pos_text)
            if pos not in target_pos.get(chrom, set()):
                continue
            info = parse_info(info_text)
            for alt in alts.split(","):
                key = f"{chrom}:{pos}:{ref}>{alt}"
                if key not in target_keys:
                    continue
                rec = records.setdefault(
                    key,
                    {
                        "variant_key": key,
                        "clinvar_matched": True,
                        "clinvar_vcf_ids": set(),
                        "clinvar_clnsig_values": set(),
                        "clinvar_review_values": set(),
                        "clinvar_geneinfo_values": set(),
                        "clinvar_rs_values": set(),
                        "clinvar_allele_ids": set(),
                        "clinvar_disease_values": set(),
                        "clinvar_molecular_consequence_values": set(),
                    },
                )
                rec["clinvar_vcf_ids"].add(var_id)
                rec["clinvar_clnsig_values"].add(info.get("CLNSIG", ""))
                rec["clinvar_review_values"].add(info.get("CLNREVSTAT", ""))
                rec["clinvar_geneinfo_values"].add(info.get("GENEINFO", ""))
                rec["clinvar_rs_values"].add(info.get("RS", ""))
                rec["clinvar_allele_ids"].add(info.get("ALLELEID", ""))
                rec["clinvar_disease_values"].add(info.get("CLNDN", ""))
                rec["clinvar_molecular_consequence_values"].add(info.get("MC", ""))

    rows: list[dict[str, Any]] = []
    for rec in records.values():
        clnsig = ";".join(sorted(x for x in rec["clinvar_clnsig_values"] if x))
        rows.append(
            {
                "variant_key": rec["variant_key"],
                "clinvar_matched": True,
                "clinvar_simple": classify_clinvar(clnsig, True),
                "clinvar_clnsig": normalize_text(clnsig),
                "clinvar_review_status": normalize_text(";".join(sorted(x for x in rec["clinvar_review_values"] if x))),
                "clinvar_geneinfo": ";".join(sorted(x for x in rec["clinvar_geneinfo_values"] if x)),
                "clinvar_rs": ";".join(sorted(x for x in rec["clinvar_rs_values"] if x)),
                "clinvar_allele_ids": ";".join(sorted(x for x in rec["clinvar_allele_ids"] if x)),
                "clinvar_disease": normalize_text(";".join(sorted(x for x in rec["clinvar_disease_values"] if x)[:5])),
                "clinvar_molecular_consequence": normalize_text(
                    ";".join(sorted(x for x in rec["clinvar_molecular_consequence_values"] if x)[:5])
                ),
                "clinvar_snapshot_date": file_date,
            }
        )
    return pd.DataFrame(rows), file_date


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


def fisher_binary(first: pd.Series, second: pd.Series, alternative: str = "greater") -> tuple[float, float]:
    first = pd.to_numeric(first, errors="coerce").dropna().astype(int)
    second = pd.to_numeric(second, errors="coerce").dropna().astype(int)
    if len(first) == 0 or len(second) == 0:
        return float("nan"), float("nan")
    table = [[int(first.sum()), int(len(first) - first.sum())], [int(second.sum()), int(len(second) - second.sum())]]
    odds, pval = fisher_exact(table, alternative=alternative)
    return float(odds), float(pval)


def mannwhitney_greater(first: pd.Series, second: pd.Series) -> float:
    first = pd.to_numeric(first, errors="coerce").dropna().astype(float)
    second = pd.to_numeric(second, errors="coerce").dropna().astype(float)
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    return float(mannwhitneyu(first, second, alternative="greater").pvalue)


def group_row(name: str, group: pd.DataFrame) -> dict[str, Any]:
    labels = pd.to_numeric(group["label"], errors="coerce")
    scores = pd.to_numeric(group["external_lof_score"], errors="coerce")
    return {
        "group": name,
        "n_total": int(len(group)),
        "n_binary": int(labels.notna().sum()),
        "n_depleted": int(labels.eq(1).sum()),
        "n_functional": int(labels.eq(0).sum()),
        "depleted_rate_binary": float(labels.dropna().mean()) if labels.notna().any() else float("nan"),
        "mean_external_lof_score": float(scores.mean()) if scores.notna().any() else float("nan"),
        "median_external_lof_score": float(scores.median()) if scores.notna().any() else float("nan"),
        "n_clinvar_matched": int(group["clinvar_matched"].sum()) if "clinvar_matched" in group else 0,
    }


def test_row(name: str, first_name: str, first: pd.DataFrame, second_name: str, second: pd.DataFrame) -> dict[str, Any]:
    odds, pval = fisher_binary(first["label"], second["label"], "greater")
    return {
        "test": name,
        "first_group": first_name,
        "second_group": second_name,
        "first_n_binary": int(pd.to_numeric(first["label"], errors="coerce").notna().sum()),
        "first_n_depleted": int(pd.to_numeric(first["label"], errors="coerce").eq(1).sum()),
        "second_n_binary": int(pd.to_numeric(second["label"], errors="coerce").notna().sum()),
        "second_n_depleted": int(pd.to_numeric(second["label"], errors="coerce").eq(1).sum()),
        "fisher_oddsratio": odds,
        "fisher_p_greater": pval,
        "external_lof_auroc": safe_auc(first["external_lof_score"], second["external_lof_score"]),
        "mannwhitney_p_greater": mannwhitney_greater(first["external_lof_score"], second["external_lof_score"]),
    }


def build_outputs(variants: pd.DataFrame, current: pd.DataFrame, file_date: str) -> tuple[pd.DataFrame, ...]:
    df = variants.copy()
    df["variant_key"] = variant_key_df(df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["external_lof_score"] = -pd.to_numeric(df["function_score"], errors="coerce")
    df = df.merge(current, on="variant_key", how="left")
    df["clinvar_matched"] = df["clinvar_matched"].map(lambda value: bool(value) if pd.notna(value) else False)
    df["clinvar_simple"] = [
        classify_clinvar(clnsig, matched)
        for clnsig, matched in zip(df["clinvar_clnsig"].fillna(""), df["clinvar_matched"], strict=False)
    ]
    fill_empty = [
        "clinvar_clnsig",
        "clinvar_review_status",
        "clinvar_geneinfo",
        "clinvar_rs",
        "clinvar_allele_ids",
        "clinvar_disease",
        "clinvar_molecular_consequence",
    ]
    for col in fill_empty:
        df[col] = df[col].fillna("")
    df["clinvar_snapshot_date"] = df["clinvar_snapshot_date"].fillna(file_date)

    groups = {
        "clinvar_pathogenic_likely_pathogenic": df[df["clinvar_simple"].eq("Pathogenic/Likely_pathogenic")],
        "clinvar_benign_likely_benign": df[df["clinvar_simple"].eq("Benign/Likely_benign")],
        "clinvar_uncertain": df[df["clinvar_simple"].eq("Uncertain")],
        "clinvar_conflicting": df[df["clinvar_simple"].eq("Conflicting")],
        "clinvar_other": df[df["clinvar_simple"].eq("Other")],
        "clinvar_unobserved": df[df["clinvar_simple"].eq("Unobserved")],
        "consequence_nonsense": df[df["consequence"].eq("Nonsense")],
        "consequence_missense": df[df["consequence"].eq("Missense")],
        "consequence_synonymous": df[df["consequence"].eq("Synonymous")],
        "consequence_utr": df[df["consequence"].eq("UTR")],
        "consequence_stop_codon": df[df["consequence"].eq("Stop codon")],
    }
    group_summary = pd.DataFrame([group_row(name, group) for name, group in groups.items()])

    tests = pd.DataFrame(
        [
            test_row(
                "clinvar_pathogenic_enriched_for_depleted_vs_benign",
                "clinvar_pathogenic_likely_pathogenic",
                groups["clinvar_pathogenic_likely_pathogenic"],
                "clinvar_benign_likely_benign",
                groups["clinvar_benign_likely_benign"],
            ),
            test_row(
                "nonsense_enriched_for_depleted_vs_synonymous",
                "consequence_nonsense",
                groups["consequence_nonsense"],
                "consequence_synonymous",
                groups["consequence_synonymous"],
            ),
            test_row(
                "missense_enriched_for_depleted_vs_synonymous",
                "consequence_missense",
                groups["consequence_missense"],
                "consequence_synonymous",
                groups["consequence_synonymous"],
            ),
            test_row(
                "missense_enriched_for_depleted_vs_utr",
                "consequence_missense",
                groups["consequence_missense"],
                "consequence_utr",
                groups["consequence_utr"],
            ),
        ]
    )

    consequence = (
        df.groupby(["consequence", "functional_classification"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["fast depleted", "slow depleted", "unchanged"]:
        if col not in consequence.columns:
            consequence[col] = 0
    consequence["depleted"] = consequence["fast depleted"] + consequence["slow depleted"]
    consequence["binary_n"] = consequence["depleted"] + consequence["unchanged"]
    consequence["depleted_rate_binary"] = consequence["depleted"] / consequence["binary_n"].replace(0, np.nan)
    consequence = consequence.sort_values(["depleted_rate_binary", "binary_n"], ascending=[False, False])

    candidates = df[
        df["clinvar_simple"].isin(["Uncertain", "Conflicting", "Unobserved"])
        & df["label"].eq(1)
    ].copy()
    candidates = candidates.sort_values(["clinvar_simple", "external_lof_score"], ascending=[True, False])
    candidate_cols = [
        "id",
        "variant_key",
        "HGVSc",
        "HGVSp",
        "consequence",
        "functional_classification",
        "function_score",
        "external_lof_score",
        "label",
        "clinvar_simple",
        "clinvar_clnsig",
        "clinvar_review_status",
        "clinvar_rs",
        "clinvar_disease",
        "domains",
    ]
    candidates = candidates[[col for col in candidate_cols if col in candidates.columns]].head(200)
    return df, group_summary, tests, consequence, candidates


def write_report(
    output_dir: Path,
    prefix: str,
    annotated: pd.DataFrame,
    group_summary: pd.DataFrame,
    tests: pd.DataFrame,
    consequence: pd.DataFrame,
    candidates: pd.DataFrame,
    clinvar_date: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_dir / f"{prefix}_annotated.csv", index=False)
    group_summary.to_csv(output_dir / f"{prefix}_group_summary.csv", index=False)
    tests.to_csv(output_dir / f"{prefix}_tests.csv", index=False)
    consequence.to_csv(output_dir / f"{prefix}_consequence_summary.csv", index=False)
    candidates.to_csv(output_dir / f"{prefix}_candidate_review.csv", index=False)

    clinvar_test = tests[tests["test"].eq("clinvar_pathogenic_enriched_for_depleted_vs_benign")].iloc[0]
    nonsense_test = tests[tests["test"].eq("nonsense_enriched_for_depleted_vs_synonymous")].iloc[0]
    missense_test = tests[tests["test"].eq("missense_enriched_for_depleted_vs_synonymous")].iloc[0]
    matched = int(annotated["clinvar_matched"].sum())
    exact_known = int(annotated["clinvar_simple"].isin(["Pathogenic/Likely_pathogenic", "Benign/Likely_benign"]).sum())

    report = [
        "# RAD51C SGE External Anchors",
        "",
        "## Purpose",
        "",
        "Validate the RAD51C SGE fallback benchmark before using it as a third-gene CrossBioSAE checkpoint.",
        "This report uses exact current ClinVar GRCh38 matches plus consequence-class controls.",
        "",
        "## Inputs",
        "",
        f"- variants: {len(annotated)} mapped binary direct-SNV rows",
        f"- ClinVar GRCh38 snapshot date: {clinvar_date or 'unknown'}",
        f"- exact ClinVar matches: {matched}",
        f"- exact ClinVar P/LP or B/LB rows: {exact_known}",
        "",
        "## Key Anchor Tests",
        "",
        (
            "- ClinVar P/LP vs B/LB: "
            f"{int(clinvar_test['first_n_depleted'])}/{int(clinvar_test['first_n_binary'])} depleted vs "
            f"{int(clinvar_test['second_n_depleted'])}/{int(clinvar_test['second_n_binary'])}; "
            f"Fisher p={fmt(clinvar_test['fisher_p_greater'], sci=True)}, "
            f"external-lof AUROC={fmt(clinvar_test['external_lof_auroc'])}"
        ),
        (
            "- Nonsense vs synonymous: "
            f"{int(nonsense_test['first_n_depleted'])}/{int(nonsense_test['first_n_binary'])} depleted vs "
            f"{int(nonsense_test['second_n_depleted'])}/{int(nonsense_test['second_n_binary'])}; "
            f"Fisher p={fmt(nonsense_test['fisher_p_greater'], sci=True)}, "
            f"external-lof AUROC={fmt(nonsense_test['external_lof_auroc'])}"
        ),
        (
            "- Missense vs synonymous: "
            f"{int(missense_test['first_n_depleted'])}/{int(missense_test['first_n_binary'])} depleted vs "
            f"{int(missense_test['second_n_depleted'])}/{int(missense_test['second_n_binary'])}; "
            f"Fisher p={fmt(missense_test['fisher_p_greater'], sci=True)}, "
            f"external-lof AUROC={fmt(missense_test['external_lof_auroc'])}"
        ),
        "",
        "## Group Summary",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Consequence Summary",
        "",
        consequence.to_markdown(index=False),
        "",
        "## Candidate Review Table",
        "",
        f"- unresolved/conflicting/unobserved depleted rows exported: {len(candidates)}",
        "",
        "## Claim Boundary",
        "",
        "These anchors support RAD51C as a credible third-gene functional benchmark.",
        "They do not prove CrossBioSAE generalization until Evo2/ESM/checkpoint metrics and, ideally, feature-level intervention are complete.",
        "",
        "## Outputs",
        "",
        f"- {prefix}_annotated.csv",
        f"- {prefix}_group_summary.csv",
        f"- {prefix}_tests.csv",
        f"- {prefix}_consequence_summary.csv",
        f"- {prefix}_candidate_review.csv",
    ]
    (output_dir / f"{prefix}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    variants_path = args.variants if args.variants.is_absolute() else args.repo_root / args.variants
    vcf_path = args.clinvar_vcf if args.clinvar_vcf.is_absolute() else args.repo_root / args.clinvar_vcf
    output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    variants = pd.read_csv(variants_path)
    variants["variant_key"] = variant_key_df(variants)
    current, clinvar_date = read_clinvar_subset(vcf_path, set(variants["variant_key"]))
    annotated, group_summary, tests, consequence, candidates = build_outputs(variants, current, clinvar_date)
    write_report(output_dir, args.output_prefix, annotated, group_summary, tests, consequence, candidates, clinvar_date)


if __name__ == "__main__":
    main()
