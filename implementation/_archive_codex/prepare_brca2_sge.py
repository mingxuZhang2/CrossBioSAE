#!/usr/bin/env python3
"""
Prepare BRCA2 SGE function-score table for CrossBioSAE replication.

Input is Supplementary Table 5 from:
Sahu et al., Saturation genome editing-based clinical classification of BRCA2
variants, Nature 2025, https://doi.org/10.1038/s41586-024-08349-1

The table covers BRCA2 exons 15-26 in GRCh38 / NC_000013.11 coordinates and
contains 6,551 SNVs. This script writes:

1. a full normalized table with all rows, including uncertain SGE classes; and
2. a binary LOF-vs-functional table excluding "Uncertain" rows, analogous to
   data/variant/brca1/brca1_variants.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


G_NOM_RE = re.compile(r"NC_000013\.11:g\.(\d+)([ACGT])>([ACGT])$")
AA_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=root / "data" / "variant" / "brca2" / "sahu_brca2_supp_table5_function_scores.xlsx",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "data" / "variant" / "brca2" / "brca2_variants.csv",
    )
    parser.add_argument(
        "--full-out",
        type=Path,
        default=root / "data" / "variant" / "brca2" / "brca2_sge_all_variants.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_sge_data_preparation.md",
    )
    return parser.parse_args()


def parse_g_nom(val: str) -> tuple[int | None, str | None, str | None]:
    m = G_NOM_RE.match(str(val))
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), m.group(3)


def parse_aa_change(val: str) -> tuple[float, str | None, str | None]:
    m = AA_RE.match(str(val))
    if not m:
        return float("nan"), None, None
    return float(m.group(2)), m.group(1), m.group(3)


def consequence(row: pd.Series) -> str:
    aa = str(row["AA.change"])
    if aa == "Intronic":
        return "Intronic"
    if row["aa_alt"] == "*":
        return "Nonsense"
    if pd.notna(row["aa_pos"]) and row["aa_ref"] == row["aa_alt"]:
        return "Synonymous"
    if pd.notna(row["aa_pos"]):
        return "Missense"
    return "Other"


def brca2_domain(aa_pos: float) -> str:
    if pd.isna(aa_pos):
        return "intronic_or_unmapped"
    p = int(aa_pos)
    if 2479 <= p <= 2668:
        return "CTDB_helical_2479_2668"
    if 2682 <= p <= 2794:
        return "CTDB_OB1_2682_2794"
    if 2804 <= p <= 3054:
        return "CTDB_OB2_2804_3054"
    if 3073 <= p <= 3167:
        return "CTDB_OB3_3073_3167"
    if 2479 <= p <= 3216:
        return "CTDB_other_2479_3216"
    return "outside_curated_CTDB"


def classification_to_label(cls: str) -> float:
    cls = str(cls)
    if cls.startswith("Pathogenic"):
        return 1.0
    if cls.startswith("Benign"):
        return 0.0
    return float("nan")


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.full_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.xlsx)
    parsed = raw["g.nom"].map(parse_g_nom)
    aa_parsed = raw["AA.change"].map(parse_aa_change)

    d = raw.copy()
    d["chrom"] = "13"
    d["pos_hg38"] = [x[0] for x in parsed]
    d["pos"] = d["pos_hg38"]
    d["ref"] = [x[1] for x in parsed]
    d["alt"] = [x[2] for x in parsed]
    d["aa_pos"] = [x[0] for x in aa_parsed]
    d["aa_ref"] = [x[1] for x in aa_parsed]
    d["aa_alt"] = [x[2] for x in aa_parsed]
    d["consequence"] = d.apply(consequence, axis=1)
    d["is_missense"] = d["consequence"].eq("Missense")
    d["is_synonymous"] = d["consequence"].eq("Synonymous")
    d["is_nonsense"] = d["consequence"].eq("Nonsense")
    d["affect_splicing"] = d["Affect.splicing"].eq("Yes")
    d["vtype"] = d["consequence"].map(lambda c: "noncoding" if c == "Intronic" else "coding")
    d["brca2_domain"] = d["aa_pos"].map(brca2_domain)
    d["function_score"] = pd.to_numeric(d["Function.score"], errors="coerce")
    d["probability"] = pd.to_numeric(d["Probability"], errors="coerce")
    d["func_class"] = d["Classification"].astype(str)
    d["acmg_class_avengers"] = d["ACMG.class.AVENGERS"].astype(str)
    d["clinvar_simple"] = d["ClinVar.annotation"].fillna("absent").astype(str)
    d["label"] = d["func_class"].map(classification_to_label)
    d["id"] = [f"brca2_{i}" for i in range(len(d))]
    d["source"] = "Sahu_2025_Nature_BRCA2_SGE_Table5"

    keep = [
        "id",
        "chrom",
        "pos_hg38",
        "pos",
        "ref",
        "alt",
        "consequence",
        "aa_pos",
        "aa_ref",
        "aa_alt",
        "AA.change",
        "c.nom",
        "g.nom",
        "dbSNP.ID",
        "function_score",
        "probability",
        "func_class",
        "acmg_class_avengers",
        "clinvar_simple",
        "affect_splicing",
        "is_missense",
        "is_synonymous",
        "is_nonsense",
        "label",
        "vtype",
        "brca2_domain",
        "source",
    ]
    full = d[keep].copy()
    binary = full[full["label"].notna()].copy()
    binary["label"] = binary["label"].astype(int)

    full.to_csv(args.full_out, index=False)
    binary.to_csv(args.out, index=False)

    class_summary = (
        full.groupby("func_class", dropna=False)
        .agg(n=("id", "size"), mean_function_score=("function_score", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    domain_summary = (
        binary.groupby("brca2_domain", dropna=False)
        .agg(n=("id", "size"), lof=("label", "sum"), lof_rate=("label", "mean"), mean_function_score=("function_score", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    consequence_summary = (
        binary.groupby(["vtype", "consequence"], dropna=False)
        .agg(n=("id", "size"), lof=("label", "sum"), lof_rate=("label", "mean"))
        .reset_index()
        .sort_values(["vtype", "n"], ascending=[True, False])
    )

    lines = [
        "# BRCA2 SGE Data Preparation",
        "",
        "## Source",
        "",
        "- Sahu et al., Nature 2025, `Saturation genome editing-based clinical classification of BRCA2 variants`.",
        "- DOI: https://doi.org/10.1038/s41586-024-08349-1",
        "- Supplementary Table 5: function scores for 6,551 SNVs across BRCA2 exons 15-26 / CTDB.",
        "",
        "## Outputs",
        "",
        f"- Full normalized table: `{args.full_out}`",
        f"- Binary LOF-vs-functional table: `{args.out}`",
        "",
        "## Counts",
        "",
        f"- Full rows: {len(full)}",
        f"- Binary rows after excluding uncertain SGE classes: {len(binary)}",
        f"- LOF/pathogenic-like rows: {int(binary['label'].sum())}",
        f"- Functional/benign-like rows: {int((binary['label'] == 0).sum())}",
        f"- Missense rows in binary table: {int(binary['is_missense'].sum())}",
        f"- Intronic/noncoding rows in binary table: {int((binary['vtype'] == 'noncoding').sum())}",
        "",
        "## Classification Summary",
        "",
        class_summary.to_markdown(index=False),
        "",
        "## Domain Summary",
        "",
        domain_summary.to_markdown(index=False),
        "",
        "## Consequence Summary",
        "",
        consequence_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This table is the right next replication dataset for the BRCA1 interpretability story: it is an independent SGE assay in another DNA-repair cancer gene, covers BRCA2 CTDB domains, contains both coding and nearby intronic SNVs, and includes functional classifications that can be used as orthogonal labels.",
        "",
        "The immediate blocker for full CrossBioSAE replication is not labels; it is representation extraction. The next computational step is to compute BRCA2 ESM protein deltas and Evo2 DNA deltas/LLRs for these 6,198 binary-classified variants, then run the BRCA1 mechanism validation and SAE feature calibration pipeline on BRCA2.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote full BRCA2 SGE table: {args.full_out} ({len(full)} rows)")
    print(f"Wrote binary BRCA2 SGE table: {args.out} ({len(binary)} rows)")
    print(class_summary.to_string(index=False))
    print(domain_summary.to_string(index=False))


if __name__ == "__main__":
    main()
