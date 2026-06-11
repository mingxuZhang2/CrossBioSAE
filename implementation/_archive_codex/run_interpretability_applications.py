#!/usr/bin/env python3
"""
Interpretability applications for the current CrossBioSAE variant mechanism line.

This script is CPU-only and read-only with respect to source data. It turns the
existing genome-wide SAE feature atlas and VUS profiles into a concrete
downstream application:

  Structural-gene VUS triage using novel pathogenic SAE features.

It validates the same score on known ClinVar pathogenic/benign variants in
structural protein genes, then ranks VUS in those genes for follow-up.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


COLLAGEN_GENES = {
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A3", "COL4A4",
    "COL4A5", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
    "COL7A1", "COL9A1", "COL11A1", "COL11A2", "COL17A1",
}
FIBRILLIN_GENES = {"FBN1", "FBN2"}
STRUCTURAL_GENES = COLLAGEN_GENES | FIBRILLIN_GENES | {"COMP", "FLNA", "FLNB"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="implementation/ directory")
    p.add_argument("--out", default="results/interpretability_applications")
    p.add_argument("--novel-path-rate", type=float, default=0.85)
    p.add_argument("--top-frac", type=float, default=0.10)
    return p.parse_args()


def is_gly_sub(pchange: object) -> bool:
    if not isinstance(pchange, str) or not pchange:
        return False
    return pchange.startswith("Gly") and not pchange.startswith("Gly=")


def load_variant_info(variant_summary: Path, clinvar: pd.DataFrame) -> pd.DataFrame:
    lookup = set(
        zip(
            clinvar["chrom"].astype(str),
            clinvar["pos"].astype(str),
            clinvar["ref"].astype(str),
            clinvar["alt"].astype(str),
        )
    )
    info: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with gzip.open(variant_summary, "rt") as f:
        header = f.readline()
        if not header:
            raise RuntimeError(f"Empty variant summary: {variant_summary}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 34 or parts[16] != "GRCh38":
                continue
            key = (parts[18], parts[31], parts[32], parts[33])
            if key not in lookup or key in info:
                continue
            name = parts[2]
            m = re.search(r"\(p\.([A-Za-z0-9_*=]+)\)", name)
            info[key] = {
                "gene": parts[4],
                "variant_name": name,
                "pchange": m.group(1) if m else "",
                "variant_type": parts[1],
                "clinical_significance": parts[6],
            }

    rows = []
    for r in clinvar.itertuples(index=False):
        key = (str(r.chrom), str(r.pos), str(r.ref), str(r.alt))
        rows.append(info.get(key, {}))
    return pd.DataFrame(rows)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    auc = roc_auc_score(y[ok], score[ok])
    return float(max(auc, 1.0 - auc))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], score[ok]))


def score_known_variants(root: Path, novel_feature_ids: np.ndarray) -> pd.DataFrame:
    clinvar = pd.read_parquet(root / "data/variant/clinvar.parquet").reset_index(drop=True)
    info = load_variant_info(root / "data/variant/variant_summary.txt.gz", clinvar)
    acts = np.load(root / "results/sae_genomewide/sae_genomewide_acts.npz")["acts"]

    score = np.abs(acts[:, novel_feature_ids]).sum(axis=1)
    scored = pd.concat([clinvar, info], axis=1)
    scored["label"] = scored["label"].astype(int)
    scored["novel_path_score"] = score
    scored["n_novel_features_active"] = (np.abs(acts[:, novel_feature_ids]) > 1e-6).sum(axis=1)
    scored["is_structural_gene"] = scored["gene"].isin(STRUCTURAL_GENES)
    scored["is_collagen_gene"] = scored["gene"].isin(COLLAGEN_GENES)
    scored["is_gly_sub"] = scored["pchange"].apply(is_gly_sub)
    scored["is_mapped_gene"] = scored["gene"].notna() & (scored["gene"] != "")
    return scored


def validation_tables(scored: pd.DataFrame, top_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    groups = {
        "all_mapped": scored["is_mapped_gene"],
        "structural_genes": scored["is_structural_gene"],
        "collagen_genes": scored["is_collagen_gene"],
        "collagen_gly_sub": scored["is_collagen_gene"] & scored["is_gly_sub"],
        "non_structural_mapped": scored["is_mapped_gene"] & ~scored["is_structural_gene"],
    }
    for name, mask in groups.items():
        sub = scored[mask].copy()
        if len(sub) == 0:
            continue
        y = sub["label"].to_numpy()
        s = sub["novel_path_score"].to_numpy(dtype=float)
        rows.append({
            "group": name,
            "n": len(sub),
            "n_pathogenic": int(y.sum()),
            "path_rate": float(y.mean()) if len(y) else float("nan"),
            "mean_score_pathogenic": float(sub.loc[sub["label"] == 1, "novel_path_score"].mean()),
            "mean_score_benign": float(sub.loc[sub["label"] == 0, "novel_path_score"].mean()),
            "auc_novel_path_score": safe_auc(y, s),
            "auprc_novel_path_score": safe_auprc(y, s),
            "auc_ESM1b": safe_auc(y, sub["ESM-1b"].to_numpy(dtype=float)),
            "auc_GPN_MSA": safe_auc(y, sub["GPN-MSA"].to_numpy(dtype=float)),
            "auc_CADD": safe_auc(y, sub["CADD"].to_numpy(dtype=float)),
        })

    gene_rows = []
    for gene, g in scored[scored["is_structural_gene"]].groupby("gene"):
        if len(g) < 10:
            continue
        y = g["label"].to_numpy()
        s = g["novel_path_score"].to_numpy(dtype=float)
        gene_rows.append({
            "gene": gene,
            "n": len(g),
            "n_pathogenic": int(y.sum()),
            "path_rate": float(y.mean()),
            "n_gly_sub": int(g["is_gly_sub"].sum()),
            "auc_novel_path_score": safe_auc(y, s),
            "mean_score": float(np.mean(s)),
            "mean_score_pathogenic": float(g.loc[g["label"] == 1, "novel_path_score"].mean()),
            "mean_score_benign": float(g.loc[g["label"] == 0, "novel_path_score"].mean()),
        })

    validation = pd.DataFrame(rows)
    gene_validation = pd.DataFrame(gene_rows).sort_values(
        ["auc_novel_path_score", "n"], ascending=[False, False]
    )

    # Add top-score enrichment for structural genes.
    structural = scored[scored["is_structural_gene"]].copy()
    if len(structural) >= 20 and structural["label"].nunique() == 2:
        cutoff = structural["novel_path_score"].quantile(1.0 - top_frac)
        high = structural["novel_path_score"] >= cutoff
        table = [
            [int((high & (structural["label"] == 1)).sum()), int((high & (structural["label"] == 0)).sum())],
            [int((~high & (structural["label"] == 1)).sum()), int((~high & (structural["label"] == 0)).sum())],
        ]
        odds, pval = fisher_exact(table)
        validation.loc[validation["group"] == "structural_genes", "top_score_cutoff"] = cutoff
        validation.loc[validation["group"] == "structural_genes", "top_score_odds_ratio"] = odds
        validation.loc[validation["group"] == "structural_genes", "top_score_fisher_p"] = pval

    return validation, gene_validation


def vus_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vus = pd.read_csv(root / "results/vus_pilot/vus_mechanism_profiles.csv")
    vus["has_novel_path_feature"] = vus["n_novel_features_active"] > 0
    gene_summary = (
        vus.groupby("gene")
        .agg(
            total_vus=("gene", "size"),
            gly_sub_vus=("is_gly_sub", "sum"),
            novel_active_vus=("has_novel_path_feature", "sum"),
            mean_novel_path_score=("novel_path_score", "mean"),
            max_novel_path_score=("novel_path_score", "max"),
        )
        .reset_index()
    )
    gene_summary["novel_active_frac"] = gene_summary["novel_active_vus"] / gene_summary["total_vus"]
    gene_summary = gene_summary.sort_values(
        ["novel_active_vus", "mean_novel_path_score"], ascending=[False, False]
    )

    top_candidates = vus.sort_values(
        ["novel_path_score", "n_novel_features_active"], ascending=[False, False]
    ).head(100)
    gly_candidates = vus[vus["is_gly_sub"] & vus["has_novel_path_feature"]].sort_values(
        ["novel_path_score", "n_novel_features_active"], ascending=[False, False]
    )
    return vus, gene_summary, pd.concat(
        [
            top_candidates.assign(candidate_set="top_novel_score"),
            gly_candidates.head(100).assign(candidate_set="gly_sub_novel_active"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["chrom", "pos", "ref", "alt", "gene", "candidate_set"])


def feature_tables(root: Path, novel_path_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cards = pd.read_csv(root / "results/sae_genomewide/sae_genomewide_cards_deep.csv")
    novel = cards[(cards["modality"] == "novel") & (cards["path_rate"] > novel_path_rate)].copy()
    structural = pd.read_csv(root / "results/novel_features/novel_structural_enrichment.csv")
    structural = structural.sort_values(["struct_enrichment", "n_gly_collagen"], ascending=[False, False])
    return novel, structural


def write_markdown(
    out: Path,
    validation: pd.DataFrame,
    gene_validation: pd.DataFrame,
    novel_features: pd.DataFrame,
    structural_features: pd.DataFrame,
    vus: pd.DataFrame,
    vus_gene_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    structural_row = validation[validation["group"] == "structural_genes"].iloc[0].to_dict()
    collagen_row = validation[validation["group"] == "collagen_genes"].iloc[0].to_dict()
    n_gly_candidates = int((vus["is_gly_sub"] & (vus["n_novel_features_active"] > 0)).sum())
    n_novel_vus = int((vus["n_novel_features_active"] > 0).sum())

    lines = [
        "# Interpretability Application: Structural-Gene VUS Triage",
        "",
        "## Claim",
        "",
        "Novel pathogenic SAE features form a mechanism-aware triage score for structural-protein VUS. "
        "The same score is first sanity-checked on known ClinVar pathogenic/benign variants, then applied to VUS.",
        "",
        "## Known-Variant Validation",
        "",
        f"- Structural-gene ClinVar variants: n={int(structural_row['n'])}, "
        f"pathogenic={int(structural_row['n_pathogenic'])}, "
        f"novel-score AUROC={structural_row['auc_novel_path_score']:.3f}.",
        f"- Collagen-gene ClinVar variants: n={int(collagen_row['n'])}, "
        f"pathogenic={int(collagen_row['n_pathogenic'])}, "
        f"novel-score AUROC={collagen_row['auc_novel_path_score']:.3f}.",
        f"- Top-score structural variants are enriched for pathogenic labels: "
        f"odds ratio={structural_row.get('top_score_odds_ratio', float('nan')):.2f}, "
        f"Fisher p={structural_row.get('top_score_fisher_p', float('nan')):.2e}.",
        "",
        "## Feature Evidence",
        "",
        f"- Novel pathogenic feature count (path_rate threshold): {len(novel_features)}.",
        "- Top structural-enriched novel features:",
        "",
        structural_features.head(8).to_markdown(index=False),
        "",
        "## VUS Application",
        "",
        f"- Structural-gene VUS profiled: n={len(vus)} across {vus['gene'].nunique()} genes.",
        f"- VUS activating at least one novel pathogenic feature: {n_novel_vus}.",
        f"- Gly-substitution VUS activating novel pathogenic features: {n_gly_candidates}.",
        "",
        "Top genes by number of VUS with novel pathogenic feature activation:",
        "",
        vus_gene_summary.head(12).to_markdown(index=False),
        "",
        "Top candidate VUS:",
        "",
        candidates.head(20).to_markdown(index=False),
        "",
        "## Publication Use",
        "",
        "This is an application figure candidate: known-variant validation, feature-level biological mechanism, "
        "and VUS prioritization table. It still needs external disease-database validation before any clinical claim.",
        "",
    ]
    (out / "structural_vus_triage_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    novel_features, structural_features = feature_tables(root, args.novel_path_rate)
    novel_feature_ids = novel_features["feature"].astype(int).to_numpy()
    if len(novel_feature_ids) == 0:
        raise RuntimeError("No novel pathogenic features found; lower --novel-path-rate.")

    scored = score_known_variants(root, novel_feature_ids)
    validation, gene_validation = validation_tables(scored, args.top_frac)
    vus, vus_gene_summary, candidates = vus_tables(root)

    scored_cols = [
        "chrom", "pos", "ref", "alt", "gene", "pchange", "label",
        "is_structural_gene", "is_collagen_gene", "is_gly_sub",
        "novel_path_score", "n_novel_features_active",
        "ESM-1b", "GPN-MSA", "CADD",
    ]
    scored[scored_cols].to_csv(out / "known_variant_novel_scores.csv", index=False)
    validation.to_csv(out / "known_variant_validation_summary.csv", index=False)
    gene_validation.to_csv(out / "known_variant_gene_validation.csv", index=False)
    novel_features.to_csv(out / "novel_pathogenic_features_used.csv", index=False)
    structural_features.to_csv(out / "novel_feature_structural_evidence.csv", index=False)
    vus_gene_summary.to_csv(out / "vus_gene_priority_summary.csv", index=False)
    candidates.to_csv(out / "vus_priority_candidates.csv", index=False)
    write_markdown(out, validation, gene_validation, novel_features, structural_features, vus, vus_gene_summary, candidates)

    print(f"Wrote interpretability application outputs to {out}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
