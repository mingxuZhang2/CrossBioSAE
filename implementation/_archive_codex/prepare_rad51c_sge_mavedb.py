#!/usr/bin/env python3
"""Prepare the public RAD51C SGE MaveDB score set as a fallback benchmark."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests


URN = "urn:mavedb:00000673-0-1"
API = "https://api.mavedb.org/api/v1"
OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument("--urn", default=URN)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def fetch_json(url: str, timeout: int) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_scores(url: str, timeout: int) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content = r.content
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return pd.read_csv(io.BytesIO(content))


def normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    cls = df["functional_classification"].fillna("").astype(str).str.lower()
    df["label"] = pd.NA
    df.loc[cls.str.contains("depleted", na=False), "label"] = 1
    df.loc[cls.str.contains("unchanged", na=False), "label"] = 0
    df["is_binary_sge_label"] = df["label"].notna()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["id"] = [f"rad51c_{i}" for i in range(len(df))]
    df["source"] = "Stone_2024_Cell_RAD51C_SGE_MaveDB"
    df["mavedb_urn"] = URN
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    nt = df["hgvs_nt"].fillna("").astype(str)
    parsed = nt.str.extract(r":c\.([\-*]?\d+)([ACGT])>([ACGT])")
    df["cdna_pos"] = pd.to_numeric(parsed[0], errors="coerce")
    df["ref"] = parsed[1]
    df["alt"] = parsed[2]
    df["is_cdna_snv"] = parsed[1].notna() & parsed[2].notna()
    df["is_coding_cds_snv"] = df["is_cdna_snv"] & df["cdna_pos"].ge(1)
    df["is_missense_like"] = df["hgvs_pro"].fillna("").astype(str).str.contains("=", regex=False).eq(False) & df[
        "hgvs_pro"
    ].notna()
    keep = [
        "id",
        "accession",
        "hgvs_nt",
        "hgvs_splice",
        "hgvs_pro",
        "label",
        "is_binary_sge_label",
        "functional_classification",
        "score",
        "z_score_continuous",
        "BH_FDR_continuous",
        "processed_adj_lfc_continuous",
        "cdna_pos",
        "ref",
        "alt",
        "is_cdna_snv",
        "is_coding_cds_snv",
        "is_missense_like",
        "source",
        "mavedb_urn",
    ]
    return df[[col for col in keep if col in df.columns]].copy()


def table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    data_dir = repo / "data" / "variant" / "rad51c"
    out = repo / OUT_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    meta = fetch_json(f"{API}/score-sets/{args.urn}", args.timeout)
    scores = fetch_scores(f"{API}/score-sets/{args.urn}/scores", args.timeout)
    norm = normalize_scores(scores)

    data_path = data_dir / "rad51c_mavedb_scores.csv"
    meta_path = data_dir / "rad51c_mavedb_metadata.json"
    summary_path = out / "rad51c_sge_data_preparation_summary.csv"
    report_path = out / "rad51c_sge_data_preparation.md"
    norm.to_csv(data_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "mavedb_urn": args.urn,
                "title": meta.get("title", ""),
                "published_date": meta.get("publishedDate", ""),
                "target_gene": ",".join(g.get("name", "") for g in meta.get("targetGenes", [])),
                "target_accession": ",".join(
                    str(g.get("targetAccession", {}).get("accession", "")) for g in meta.get("targetGenes", [])
                ),
                "rows": len(norm),
                "binary_rows": int(norm["is_binary_sge_label"].sum()),
                "depleted_rows": int((norm["label"] == 1).sum()),
                "unchanged_rows": int((norm["label"] == 0).sum()),
                "cdna_snv_rows": int(norm["is_cdna_snv"].sum()),
                "coding_cds_snv_rows": int(norm["is_coding_cds_snv"].sum()),
                "missense_like_rows": int(norm["is_missense_like"].sum()),
            }
        ]
    )
    class_summary = (
        norm.groupby("functional_classification", dropna=False)
        .agg(n=("id", "size"), binary_n=("is_binary_sge_label", "sum"), mean_score=("score", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    summary.to_csv(summary_path, index=False)
    class_summary.to_csv(out / "rad51c_sge_functional_class_summary.csv", index=False)

    report = [
        "# RAD51C SGE MaveDB Data Preparation",
        "",
        "## Purpose",
        "",
        "Prepare RAD51C as a fallback homologous-recombination third-gene benchmark if BAP1 checkpoint evidence is weak or too indel-heavy.",
        "",
        "## Metadata",
        "",
        f"- MaveDB URN: `{args.urn}`",
        f"- Title: {meta.get('title', '')}",
        f"- Published date in MaveDB: {meta.get('publishedDate', '')}",
        f"- Target genes: {', '.join(g.get('name', '') for g in meta.get('targetGenes', []))}",
        f"- Target accession: {', '.join(str(g.get('targetAccession', {}).get('accession', '')) for g in meta.get('targetGenes', []))}",
        f"- License: {meta.get('license', {}).get('shortName', '')}",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Functional Classes",
        "",
        table(class_summary),
        "",
        "## Claim Boundary",
        "",
        "- This prepares RAD51C as a data-ready fallback benchmark only.",
        "- It does not yet provide genomic coordinates for Evo2 LLR extraction; mapped variants or transcript-to-genome conversion are the next required step.",
        "- RAD51C is attractive because it is an HR-pathway breast/ovarian cancer predisposition gene and is more SNV-centric than BAP1.",
        "",
        "## Outputs",
        "",
        "- data/variant/rad51c/rad51c_mavedb_scores.csv",
        "- data/variant/rad51c/rad51c_mavedb_metadata.json",
        "- rad51c_sge_data_preparation_summary.csv",
        "- rad51c_sge_functional_class_summary.csv",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {data_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
