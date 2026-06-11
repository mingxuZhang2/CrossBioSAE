#!/usr/bin/env python3
"""
Annotate structural-protein VUS candidates with gene-level external context.

This does not validate any VUS clinically. It joins the current interpretability
candidate table to local HGNC, OMIM/Orphanet identifiers, LSDB links, UniProt
IDs, and gnomAD gene-constraint metrics so that the downstream application has
publishable gene/disease context rather than only model scores.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


COLLAGEN_GENES = {
    "COL1A1",
    "COL1A2",
    "COL2A1",
    "COL3A1",
    "COL4A3",
    "COL4A4",
    "COL4A5",
    "COL5A1",
    "COL5A2",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "COL7A1",
    "COL9A1",
    "COL11A1",
    "COL11A2",
    "COL17A1",
}
FIBRILLIN_GENES = {"FBN1", "FBN2"}
FILAMIN_GENES = {"FLNA", "FLNB"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def strict_gly_substitution(value: object) -> bool:
    text = clean_text(value)
    return bool(re.match(r"^Gly\d+[A-Z][a-z]{2}$", text))


def pchange_position(value: object) -> float:
    text = clean_text(value)
    match = re.search(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|Ter|\*)", text)
    return float(match.group(2)) if match else float("nan")


def mechanism_hint(row: pd.Series) -> str:
    gene = clean_text(row.get("gene"))
    pchange = clean_text(row.get("pchange"))
    if gene in COLLAGEN_GENES and strict_gly_substitution(pchange):
        return "collagen_gly_triplet_disruption"
    if gene in COLLAGEN_GENES:
        return "collagen_structural_variant"
    if gene in FIBRILLIN_GENES:
        return "fibrillin_cbEGF_TB_domain_candidate"
    if gene in FILAMIN_GENES:
        return "filamin_cytoskeletal_domain_candidate"
    if gene == "COMP":
        return "cartilage_matrix_structural_candidate"
    return "structural_protein_candidate"


def read_hgnc(repo_root: Path) -> pd.DataFrame:
    hgnc = pd.read_csv(
        require_file(repo_root / "data" / "full" / "labels" / "hgnc_complete_set.txt"),
        sep="\t",
        dtype=str,
    )
    cols = [
        "symbol",
        "name",
        "location",
        "uniprot_ids",
        "omim_id",
        "orphanet",
        "lsdb",
        "mane_select",
    ]
    out = hgnc[cols].copy()
    out = out.rename(
        columns={
            "symbol": "gene",
            "name": "hgnc_gene_name",
            "location": "cytoband",
            "uniprot_ids": "uniprot_id",
            "omim_id": "hgnc_omim_id",
            "orphanet": "hgnc_orphanet_id",
            "lsdb": "hgnc_lsdb",
        }
    )
    return out


def read_gnomad_constraint(repo_root: Path) -> pd.DataFrame:
    usecols = [
        "gene",
        "transcript",
        "pLI",
        "oe_lof",
        "oe_lof_upper",
        "mis_z",
        "oe_mis",
        "oe_mis_upper",
        "max_af",
        "constraint_flag",
        "transcript_level",
        "gene_id",
    ]
    gnomad = pd.read_csv(
        require_file(repo_root / "data" / "full" / "labels" / "gnomad_constraint.txt.bgz"),
        sep="\t",
        compression="gzip",
        usecols=usecols,
    )
    # Prefer MANE/high-confidence transcript rows when multiple rows exist.
    gnomad["transcript_level_num"] = pd.to_numeric(gnomad["transcript_level"], errors="coerce")
    gnomad = gnomad.sort_values(
        ["gene", "transcript_level_num", "oe_lof_upper"],
        ascending=[True, True, True],
        na_position="last",
    )
    gnomad = gnomad.drop_duplicates("gene", keep="first")
    gnomad = gnomad.rename(
        columns={
            "transcript": "gnomad_transcript",
            "gene_id": "ensembl_gene_id",
            "pLI": "gnomad_pLI",
            "oe_lof": "gnomad_oe_lof",
            "oe_lof_upper": "gnomad_oe_lof_upper",
            "mis_z": "gnomad_mis_z",
            "oe_mis": "gnomad_oe_mis",
            "oe_mis_upper": "gnomad_oe_mis_upper",
            "max_af": "gnomad_gene_max_af",
        }
    )
    keep = [
        "gene",
        "ensembl_gene_id",
        "gnomad_transcript",
        "gnomad_pLI",
        "gnomad_oe_lof",
        "gnomad_oe_lof_upper",
        "gnomad_mis_z",
        "gnomad_oe_mis",
        "gnomad_oe_mis_upper",
        "gnomad_gene_max_af",
        "constraint_flag",
    ]
    return gnomad[keep]


def annotate_candidates(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = repo_root / "results" / "interpretability_applications"
    candidates = pd.read_csv(require_file(out_dir / "vus_collagen_case_candidates.csv"))
    hgnc = read_hgnc(repo_root)
    gnomad = read_gnomad_constraint(repo_root)

    annotated = candidates.merge(hgnc, on="gene", how="left").merge(gnomad, on="gene", how="left")
    annotated["pchange_pos"] = annotated["pchange"].apply(pchange_position)
    annotated["mechanism_hint"] = annotated.apply(mechanism_hint, axis=1)
    annotated["has_lsdb_link"] = annotated["hgnc_lsdb"].fillna("").astype(str).str.contains("http", regex=False)
    annotated["has_gene_disease_id"] = annotated["hgnc_omim_id"].notna() | annotated["hgnc_orphanet_id"].notna()
    annotated["candidate_context_strength"] = (
        annotated["has_gene_disease_id"].astype(int)
        + annotated["has_lsdb_link"].astype(int)
        + annotated["uniprot_id"].notna().astype(int)
        + annotated["gnomad_pLI"].notna().astype(int)
    )

    gene_summary = (
        annotated.groupby("gene", dropna=False)
        .agg(
            n_candidates=("gene", "size"),
            n_tier1=("candidate_tier", lambda s: int(s.eq("T1_collagen_gly_with_collagen_gly_feature").sum())),
            max_collagen_gly_score=("collagen_gly_feature_score", "max"),
            max_structural_score=("structural_feature_score", "max"),
            top_variant=("variant_id", "first"),
            top_pchange=("pchange", "first"),
            hgnc_gene_name=("hgnc_gene_name", "first"),
            hgnc_omim_id=("hgnc_omim_id", "first"),
            hgnc_orphanet_id=("hgnc_orphanet_id", "first"),
            uniprot_id=("uniprot_id", "first"),
            has_lsdb_link=("has_lsdb_link", "max"),
            gnomad_pLI=("gnomad_pLI", "first"),
            gnomad_oe_lof_upper=("gnomad_oe_lof_upper", "first"),
            gnomad_mis_z=("gnomad_mis_z", "first"),
        )
        .reset_index()
        .sort_values(["n_tier1", "n_candidates", "max_structural_score"], ascending=[False, False, False])
    )
    return annotated, gene_summary


def write_report(path: Path, annotated: pd.DataFrame, gene_summary: pd.DataFrame, top_n: int) -> None:
    top = annotated.head(top_n).copy()
    for col in [
        "novel_path_score",
        "structural_feature_score",
        "collagen_gly_feature_score",
        "gnomad_pLI",
        "gnomad_oe_lof_upper",
        "gnomad_mis_z",
    ]:
        if col in top:
            top[col] = top[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    top_table = top[
        [
            "rank",
            "candidate_tier",
            "variant_id",
            "gene",
            "pchange",
            "mechanism_hint",
            "collagen_gly_feature_score",
            "top_structural_feature",
            "hgnc_omim_id",
            "hgnc_orphanet_id",
            "uniprot_id",
            "gnomad_pLI",
            "gnomad_oe_lof_upper",
            "gnomad_mis_z",
            "has_lsdb_link",
        ]
    ]

    gene_top = gene_summary.head(25).copy()
    for col in ["max_collagen_gly_score", "max_structural_score", "gnomad_pLI", "gnomad_oe_lof_upper", "gnomad_mis_z"]:
        gene_top[col] = gene_top[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")
    gene_table = gene_top[
        [
            "gene",
            "n_candidates",
            "n_tier1",
            "top_variant",
            "top_pchange",
            "hgnc_gene_name",
            "hgnc_omim_id",
            "hgnc_orphanet_id",
            "uniprot_id",
            "has_lsdb_link",
            "gnomad_pLI",
            "gnomad_oe_lof_upper",
            "gnomad_mis_z",
        ]
    ]

    lines = [
        "# VUS Candidate Gene Context",
        "",
        "## Purpose",
        "",
        "Join interpretability-prioritized VUS candidates to local gene-level resources: HGNC gene names, OMIM/Orphanet identifiers, LSDB links, UniProt IDs, and gnomAD constraint. This is external context, not variant-level validation.",
        "",
        "## Headline Counts",
        "",
        f"- Candidate rows annotated: {len(annotated):,}.",
        f"- Genes represented: {annotated['gene'].nunique():,}.",
        f"- Candidates with OMIM/Orphanet gene-disease identifiers: {int(annotated['has_gene_disease_id'].sum()):,}.",
        f"- Candidates with HGNC LSDB links: {int(annotated['has_lsdb_link'].sum()):,}.",
        f"- Candidates with UniProt IDs: {int(annotated['uniprot_id'].notna().sum()):,}.",
        f"- Candidates with gnomAD constraint rows: {int(annotated['gnomad_pLI'].notna().sum()):,}.",
        "",
        "## Top Candidate Context",
        "",
        top_table.to_markdown(index=False),
        "",
        "## Gene-Level Context",
        "",
        gene_table.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This table is useful for prioritizing manual review: top candidates are in known disease genes with curated identifiers and often LSDB links. The next required step is variant-level validation: exact variant or same-residue matches in ClinVar/LOVD/ClinGen, gnomAD allele frequency checks, or temporal ClinVar reclassification.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated, gene_summary = annotate_candidates(repo_root)
    annotated.to_csv(output_dir / "vus_candidate_gene_context.csv", index=False)
    gene_summary.to_csv(output_dir / "vus_candidate_gene_context_summary.csv", index=False)
    write_report(output_dir / "vus_candidate_gene_context.md", annotated, gene_summary, args.top_n)

    print(f"Wrote VUS candidate gene context outputs to {output_dir}")
    print(gene_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
