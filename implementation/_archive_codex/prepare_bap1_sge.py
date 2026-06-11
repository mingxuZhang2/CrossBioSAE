#!/usr/bin/env python3
"""Prepare the public BAP1 SGE/MaveDB dataset as a third-gene benchmark."""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")
DEFAULT_MAVEDB_URN = "urn:mavedb:00000662-0-1"
DEFAULT_EXTENDED_DATA_URL = (
    "https://raw.githubusercontent.com/team113sanger/Waters_BAP1_SGE/develop/"
    "extended_data_1_(sge_bap1_dataset).xlsx"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--mavedb-urn", default=DEFAULT_MAVEDB_URN)
    parser.add_argument("--extended-data-url", default=DEFAULT_EXTENDED_DATA_URL)
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args()


def fetch_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/csv"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def api_url(urn: str, suffix: str = "") -> str:
    return f"https://api.mavedb.org/api/v1/score-sets/{urllib.parse.quote(urn, safe='')}{suffix}"


def load_gene_pair(repo: Path) -> dict:
    path = repo / "data/full/gene_pairs_full.json"
    with path.open() as handle:
        rows = json.load(handle)
    for row in rows:
        if row.get("gene_name") == "BAP1":
            return row
    raise RuntimeError(f"BAP1 not found in {path}")


def write_fasta(path: Path, header: str, sequence: str, width: int = 80) -> None:
    lines = [f">{header}"]
    lines.extend(sequence[i : i + width] for i in range(0, len(sequence), width))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_label(value: object) -> float:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text == "depleted":
        return 1.0
    if text == "unchanged":
        return 0.0
    return float("nan")


def as_int_or_na(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if not text or text == "-":
        return float("nan")
    try:
        return float(int(float(text)))
    except ValueError:
        return float("nan")


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def prepare_variants(repo: Path, extended_data_url: str, urn: str, timeout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = fetch_json(api_url(urn), timeout)
    scores = pd.read_csv(io.StringIO(fetch_text(api_url(urn, "/scores"), timeout)))
    scores = scores[["accession", "hgvs_nt", "score"]].copy()
    scores = scores.rename(columns={"hgvs_nt": "HGVSc", "score": "mavedb_score"})

    cols = [
        "HGVSc",
        "HGVSp",
        "chrom_pos_ref_alt",
        "ref_chr",
        "pos",
        "ref",
        "alt",
        "functional_classification",
        "functional_score",
        "vep_consequence",
        "vep_consequence_slim",
        "ref_aa",
        "alt_aa",
        "cDNA_position",
        "CDS_position",
        "protein_position",
        "domains",
        "EVE_class",
        "EVE_score",
        "SIFT",
        "PolyPhen",
        "concordance",
        "is_in_gnomad",
        "gnomAD_AF_v3",
        "is_in_clinvar",
        "variation_id",
        "clinvar_clinical_significance",
        "clinvar_clinical_significance_slim",
        "dbSNP.ID",
        "processed_LFC_continuous",
        "processed_Z_continuous",
        "processed_BH_FDR_continuous",
    ]
    ext = pd.read_excel(
        extended_data_url,
        sheet_name="extended_data_1",
        header=2,
        usecols=lambda col: col in cols,
        engine="openpyxl",
    )
    merged = ext.merge(scores, on="HGVSc", how="left", validate="one_to_one")
    merged = merged.reset_index(drop=True)
    merged["id"] = [f"bap1_{i}" for i in range(len(merged))]
    merged["chrom"] = merged["ref_chr"].astype(str).str.replace("chr", "", regex=False)
    merged["pos_hg38"] = pd.to_numeric(merged["pos"], errors="coerce").astype("Int64")
    merged["label"] = merged["functional_classification"].map(normalize_label)
    merged["is_binary_sge_label"] = merged["label"].notna()
    merged["is_missense"] = merged["vep_consequence_slim"].astype(str).eq("missense")
    merged["aa_pos"] = merged["protein_position"].map(as_int_or_na)
    merged["aa_ref"] = merged["ref_aa"].replace({"-": pd.NA})
    merged["aa_alt"] = merged["alt_aa"].replace({"-": pd.NA})
    merged["c_nom"] = merged["HGVSc"]
    merged["g_nom"] = (
        "NC_000003.12:g."
        + merged["pos_hg38"].astype(str)
        + merged["ref"].astype(str)
        + ">"
        + merged["alt"].astype(str)
    )
    merged["source"] = "Waters_2024_NatGenet_BAP1_SGE"
    merged["mavedb_urn"] = urn
    merged["mavedb_title"] = str(meta.get("title") or "BAP1 SGE")
    merged["mavedb_published_date"] = str(meta.get("publishedDate") or "")
    merged["function_score"] = pd.to_numeric(merged["functional_score"], errors="coerce")
    merged["score"] = pd.to_numeric(merged["mavedb_score"], errors="coerce")
    merged["consequence"] = merged["vep_consequence_slim"]
    merged["brca2_domain"] = merged.get("domains", pd.Series([""] * len(merged)))

    keep = [
        "id",
        "chrom",
        "pos_hg38",
        "pos",
        "ref",
        "alt",
        "label",
        "is_binary_sge_label",
        "functional_classification",
        "function_score",
        "score",
        "consequence",
        "vep_consequence",
        "is_missense",
        "aa_pos",
        "aa_ref",
        "aa_alt",
        "c_nom",
        "g_nom",
        "HGVSc",
        "HGVSp",
        "chrom_pos_ref_alt",
        "domains",
        "EVE_class",
        "EVE_score",
        "SIFT",
        "PolyPhen",
        "concordance",
        "is_in_gnomad",
        "gnomAD_AF_v3",
        "is_in_clinvar",
        "variation_id",
        "clinvar_clinical_significance",
        "clinvar_clinical_significance_slim",
        "dbSNP.ID",
        "processed_LFC_continuous",
        "processed_Z_continuous",
        "processed_BH_FDR_continuous",
        "source",
        "mavedb_urn",
        "mavedb_title",
        "mavedb_published_date",
    ]
    variants = merged[[col for col in keep if col in merged.columns]].copy()
    class_summary = (
        variants.groupby(["functional_classification", "consequence"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["functional_classification", "n"], ascending=[True, False])
    )
    return variants, class_summary


def write_report(repo: Path, variants: pd.DataFrame, class_summary: pd.DataFrame, protein_seq: str, cds_seq: str) -> None:
    out = repo / OUT_DIR
    n = len(variants)
    binary = variants[variants["is_binary_sge_label"]].copy()
    missense = variants[variants["is_missense"]].copy()
    missense_binary = binary[binary["is_missense"]].copy()
    clinvar = variants[variants["is_in_clinvar"].astype(str).str.upper().eq("Y")].copy()
    lines = [
        "# BAP1 SGE Third-Gene Benchmark Preparation",
        "",
        "## Purpose",
        "",
        "This prepares the public BAP1 saturation genome editing dataset as the next independent cancer-gene functional-map benchmark for the CrossBioSAE interpretability track.",
        "",
        "## Dataset",
        "",
        f"- variants_total: {n}",
        f"- binary_LOF_unchanged_rows: {len(binary)}",
        f"- depleted_LOF_rows: {int(binary['label'].eq(1).sum())}",
        f"- unchanged_functional_rows: {int(binary['label'].eq(0).sum())}",
        f"- enriched_rows_excluded_from_binary_LOF: {int(variants['functional_classification'].eq('enriched').sum())}",
        f"- missense_rows: {len(missense)}",
        f"- missense_binary_rows: {len(missense_binary)}",
        f"- clinvar_annotated_rows_in_public_table: {len(clinvar)}",
        f"- protein_length: {len(protein_seq)}",
        f"- cds_length: {len(cds_seq)}",
        "",
        "## Class Summary",
        "",
        class_summary.head(30).to_markdown(index=False),
        "",
        "## Immediate Use",
        "",
        "- `data/variant/bap1/bap1_variants.csv` can feed Evo2 LLR extraction after the BRCA2 extraction script's `--output-prefix bap1` path.",
        "- `data/variant/bap1/bap1_Q96TC6.fasta` can feed the ESM delta extractor.",
        "- Binary BAP1 evaluation should compare `depleted` versus `unchanged`; `enriched` is biologically interesting but should be analyzed separately from LOF.",
        "",
        "## Claim Boundary",
        "",
        "This is a data-readiness artifact, not yet a CrossBioSAE replication result. It becomes a third-gene result only after Evo2/ESM/native-SAE or other CrossBioSAE scores are generated and evaluated against these BAP1 SGE labels.",
        "",
        "## Sources",
        "",
        "- Nature Genetics BAP1 SGE: https://www.nature.com/articles/s41588-024-01799-3",
        "- MaveDB score set: https://www.mavedb.org/score-sets/urn:mavedb:00000662-0-1",
        "- Source repository: https://github.com/team113sanger/Waters_BAP1_SGE",
    ]
    (out / "bap1_sge_data_preparation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    data_dir = repo / "data/variant/bap1"
    data_dir.mkdir(parents=True, exist_ok=True)

    variants, class_summary = prepare_variants(repo, args.extended_data_url, args.mavedb_urn, args.timeout)
    gene = load_gene_pair(repo)
    write_fasta(data_dir / "bap1_Q96TC6.fasta", "BAP1|Q96TC6", gene["protein_seq"])

    variants.to_csv(data_dir / "bap1_variants.csv", index=False)
    variants.to_csv(out / "bap1_sge_variants.csv", index=False)
    class_summary.to_csv(out / "bap1_sge_functional_class_summary.csv", index=False)
    write_report(repo, variants, class_summary, gene["protein_seq"], gene["cds_seq"])

    print(f"wrote {data_dir / 'bap1_variants.csv'} rows={len(variants)}")
    print(f"wrote {data_dir / 'bap1_Q96TC6.fasta'}")
    print(f"wrote {out / 'bap1_sge_data_preparation.md'}")


if __name__ == "__main__":
    main()
