#!/usr/bin/env python3
"""
CPU-only interpretability downstream summary for VUS structural-protein cases.

This script is intentionally read-only with respect to all upstream artifacts. It
combines existing VUS SAE activations, novel structural-feature enrichments, and
feature cards to create publication-ready CSV/Markdown summaries under
results/interpretability_applications/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

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
STRUCTURAL_GENES = COLLAGEN_GENES | FIBRILLIN_GENES | {"COMP", "FLNA", "FLNB"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the immediate interpretability application supported by "
            "existing VUS and SAE artifacts: collagen/structural-protein VUS "
            "triage using novel structural SAE features."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--activation-threshold", type=float, default=1e-6)
    parser.add_argument("--min-path-rate", type=float, default=0.85)
    parser.add_argument("--min-struct-enrichment", type=float, default=1.5)
    parser.add_argument("--min-structural-support", type=int, default=3)
    parser.add_argument("--min-collagen-support", type=int, default=2)
    parser.add_argument("--min-gly-collagen-support", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def read_inputs(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    vus = pd.read_csv(
        require_file(repo_root / "results" / "vus_pilot" / "vus_mechanism_profiles.csv")
    )
    acts = np.load(
        require_file(repo_root / "results" / "vus_pilot" / "vus_sae_acts.npz")
    )["acts"]
    novel = pd.read_csv(
        require_file(
            repo_root / "results" / "novel_features" / "novel_structural_enrichment.csv"
        )
    )
    cards = pd.read_csv(
        require_file(
            repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv"
        )
    )

    if len(vus) != acts.shape[0]:
        raise ValueError(
            "VUS profile row count does not match VUS SAE activation rows: "
            f"{len(vus)} vs {acts.shape[0]}"
        )
    return vus, acts, novel, cards


def strict_gly_substitution(pchange: object) -> bool:
    text = "" if pd.isna(pchange) else str(pchange)
    return bool(re.match(r"^Gly\d+", text)) and not text.endswith("=")


def variant_id(row: pd.Series) -> str:
    return f"chr{row['chrom']}:{int(row['pos'])}:{row['ref']}>{row['alt']}"


def safe_percent(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def format_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "_No rows._"
    rows = [[str(v) for v in row] for row in df[cols].itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows))
        for i, col in enumerate(cols)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def feature_sets(
    novel: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = novel[
        (novel["path_rate"] >= args.min_path_rate)
        & (novel["struct_enrichment"] >= args.min_struct_enrichment)
        & (novel["n_structural"] >= args.min_structural_support)
    ].copy()
    collagen_gly = structural[
        (structural["n_collagen"] >= args.min_collagen_support)
        & (structural["n_gly_collagen"] >= args.min_gly_collagen_support)
    ].copy()
    return structural, collagen_gly


def summarize_variant_scores(
    vus: pd.DataFrame,
    acts: np.ndarray,
    cards: pd.DataFrame,
    structural_features: pd.DataFrame,
    collagen_gly_features: pd.DataFrame,
    activation_threshold: float,
) -> pd.DataFrame:
    out = vus.copy()
    out["variant_id"] = out.apply(variant_id, axis=1)
    out["is_collagen_gene"] = out["gene"].isin(COLLAGEN_GENES)
    out["is_structural_gene"] = out["gene"].isin(STRUCTURAL_GENES)
    out["source_is_gly_sub"] = out["is_gly_sub"].astype(bool)
    out["is_gly_sub_strict"] = out["pchange"].apply(strict_gly_substitution)
    out["is_synonymous_pchange"] = out["pchange"].fillna("").astype(str).str.endswith("=")

    cards_small = cards[
        [
            "feature",
            "n_active",
            "path_rate",
            "path_enrich",
            "modality",
            "concept_v2",
            "corr_ESM1b",
            "corr_GPN",
            "corr_CADD",
        ]
    ].copy()
    feature_meta = structural_features.merge(
        cards_small,
        on="feature",
        how="left",
        suffixes=("_structural", "_card"),
    )
    feature_meta = feature_meta.rename(
        columns={
            "path_rate_structural": "path_rate",
            "n_active_structural": "n_active",
        }
    )

    structural_ids = feature_meta["feature"].astype(int).to_numpy()
    collagen_gly_ids = collagen_gly_features["feature"].astype(int).to_numpy()

    max_feature = acts.shape[1] - 1
    invalid = [int(fi) for fi in structural_ids if fi < 0 or fi > max_feature]
    if invalid:
        raise ValueError(f"Structural feature ids exceed activation width: {invalid}")

    structural_abs = np.abs(acts[:, structural_ids]) if len(structural_ids) else np.zeros((len(out), 0))
    out["structural_feature_score"] = structural_abs.sum(axis=1)
    out["n_structural_features_active"] = (structural_abs > activation_threshold).sum(axis=1)

    if len(collagen_gly_ids):
        collagen_gly_abs = np.abs(acts[:, collagen_gly_ids])
        out["collagen_gly_feature_score"] = collagen_gly_abs.sum(axis=1)
        out["n_collagen_gly_features_active"] = (
            collagen_gly_abs > activation_threshold
        ).sum(axis=1)
    else:
        collagen_gly_abs = np.zeros((len(out), 0))
        out["collagen_gly_feature_score"] = 0.0
        out["n_collagen_gly_features_active"] = 0

    if len(structural_ids):
        top_pos = structural_abs.argmax(axis=1)
        top_vals = structural_abs[np.arange(len(out)), top_pos]
        top_feature = structural_ids[top_pos]
        top_feature[top_vals <= activation_threshold] = -1
        out["top_structural_feature"] = top_feature
        out["top_structural_feature_activation"] = top_vals
    else:
        out["top_structural_feature"] = -1
        out["top_structural_feature_activation"] = 0.0

    top_meta = feature_meta.add_prefix("top_structural_feature_")
    top_meta = top_meta.rename(columns={"top_structural_feature_feature": "top_structural_feature"})
    out = out.merge(top_meta, on="top_structural_feature", how="left")

    out["candidate_tier"] = "background"
    out.loc[
        out["is_collagen_gene"]
        & out["is_gly_sub_strict"]
        & (out["n_collagen_gly_features_active"] > 0),
        "candidate_tier",
    ] = "T1_collagen_gly_with_collagen_gly_feature"
    out.loc[
        (out["candidate_tier"] == "background")
        & out["is_collagen_gene"]
        & (out["n_collagen_gly_features_active"] > 0),
        "candidate_tier",
    ] = "T2_collagen_gene_with_collagen_gly_feature"
    out.loc[
        (out["candidate_tier"] == "background")
        & (out["n_structural_features_active"] > 0),
        "candidate_tier",
    ] = "T3_structural_feature_active"

    tier_order = {
        "T1_collagen_gly_with_collagen_gly_feature": 1,
        "T2_collagen_gene_with_collagen_gly_feature": 2,
        "T3_structural_feature_active": 3,
        "background": 4,
    }
    out["candidate_tier_rank"] = out["candidate_tier"].map(tier_order)
    out = out.sort_values(
        [
            "candidate_tier_rank",
            "collagen_gly_feature_score",
            "structural_feature_score",
            "novel_path_score",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def summarize_genes(variant_scores: pd.DataFrame) -> pd.DataFrame:
    active = variant_scores["n_structural_features_active"] > 0
    collagen_active = variant_scores["n_collagen_gly_features_active"] > 0

    work = variant_scores.copy()
    work["has_structural_feature"] = active
    work["has_collagen_gly_feature"] = collagen_active
    work["is_tier1"] = work["candidate_tier"].eq(
        "T1_collagen_gly_with_collagen_gly_feature"
    )

    rows = []
    for gene, grp in work.groupby("gene", dropna=False):
        top = grp.sort_values(
            [
                "candidate_tier_rank",
                "collagen_gly_feature_score",
                "structural_feature_score",
            ],
            ascending=[True, False, False],
        ).iloc[0]
        rows.append(
            {
                "gene": gene,
                "n_vus": len(grp),
                "n_collagen_gly_strict": int(
                    (grp["is_collagen_gene"] & grp["is_gly_sub_strict"]).sum()
                ),
                "n_structural_feature_active": int(grp["has_structural_feature"].sum()),
                "n_collagen_gly_feature_active": int(
                    grp["has_collagen_gly_feature"].sum()
                ),
                "n_tier1_candidates": int(grp["is_tier1"].sum()),
                "pct_structural_feature_active": safe_percent(
                    grp["has_structural_feature"].sum(), len(grp)
                ),
                "mean_structural_feature_score": grp["structural_feature_score"].mean(),
                "max_structural_feature_score": grp["structural_feature_score"].max(),
                "top_variant_id": top["variant_id"],
                "top_pchange": top["pchange"],
                "top_candidate_tier": top["candidate_tier"],
                "top_structural_feature_score": top["structural_feature_score"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["n_tier1_candidates", "n_collagen_gly_feature_active", "max_structural_feature_score"],
        ascending=[False, False, False],
    )


def summarize_features(
    variant_scores: pd.DataFrame,
    acts: np.ndarray,
    cards: pd.DataFrame,
    structural_features: pd.DataFrame,
    activation_threshold: float,
) -> pd.DataFrame:
    cards_small = cards[
        ["feature", "modality", "concept_v2", "corr_ESM1b", "corr_GPN", "corr_CADD"]
    ]
    rows = []
    for _, feat in structural_features.iterrows():
        fi = int(feat["feature"])
        active = np.abs(acts[:, fi]) > activation_threshold
        active_df = variant_scores.loc[active]
        collagen_gly_active = (
            active_df["is_collagen_gene"] & active_df["is_gly_sub_strict"]
        ).sum()
        rows.append(
            {
                "feature": fi,
                "n_train_active": int(feat["n_active"]),
                "train_path_rate": feat["path_rate"],
                "train_struct_enrichment": feat["struct_enrichment"],
                "train_n_collagen": int(feat["n_collagen"]),
                "train_n_gly_collagen": int(feat["n_gly_collagen"]),
                "n_vus_active": int(active.sum()),
                "n_vus_collagen_active": int(active_df["is_collagen_gene"].sum()),
                "n_vus_collagen_gly_strict_active": int(collagen_gly_active),
                "mean_vus_activation_abs": float(np.abs(acts[active, fi]).mean())
                if active.any()
                else 0.0,
            }
        )
    out = pd.DataFrame(rows).merge(cards_small, on="feature", how="left")
    return out.sort_values(
        [
            "n_vus_collagen_gly_strict_active",
            "n_vus_collagen_active",
            "train_struct_enrichment",
        ],
        ascending=[False, False, False],
    )


def write_outputs(
    output_dir: Path,
    variant_scores: pd.DataFrame,
    gene_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    structural_features: pd.DataFrame,
    collagen_gly_features: pd.DataFrame,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_cols = [
        "rank",
        "candidate_tier",
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "pchange",
        "clin_sig",
        "is_collagen_gene",
        "is_gly_sub_strict",
        "source_is_gly_sub",
        "is_synonymous_pchange",
        "novel_path_score",
        "n_novel_features_active",
        "structural_feature_score",
        "n_structural_features_active",
        "collagen_gly_feature_score",
        "n_collagen_gly_features_active",
        "top_structural_feature",
        "top_structural_feature_activation",
        "top_structural_feature_path_rate",
        "top_structural_feature_struct_enrichment",
        "top_structural_feature_n_collagen",
        "top_structural_feature_n_gly_collagen",
        "top_structural_feature_modality",
        "top_structural_feature_concept_v2",
    ]
    candidates = variant_scores[variant_scores["candidate_tier"] != "background"][
        candidate_cols
    ].copy()

    all_scores_cols = candidate_cols + ["candidate_tier_rank"]
    all_scores = variant_scores[all_scores_cols].copy()

    candidate_path = output_dir / "vus_collagen_case_candidates.csv"
    all_scores_path = output_dir / "vus_interpretability_scores.csv"
    gene_path = output_dir / "vus_collagen_gene_summary.csv"
    feature_path = output_dir / "vus_collagen_feature_summary.csv"
    report_path = output_dir / "vus_collagen_case_study.md"

    candidates.to_csv(candidate_path, index=False)
    all_scores.to_csv(all_scores_path, index=False)
    gene_summary.to_csv(gene_path, index=False)
    feature_summary.to_csv(feature_path, index=False)

    write_markdown_report(
        report_path,
        variant_scores,
        candidates,
        gene_summary,
        feature_summary,
        structural_features,
        collagen_gly_features,
        args,
    )

    return [candidate_path, all_scores_path, gene_path, feature_path, report_path]


def write_markdown_report(
    path: Path,
    variant_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    gene_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    structural_features: pd.DataFrame,
    collagen_gly_features: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    total_vus = len(variant_scores)
    collagen_vus = int(variant_scores["is_collagen_gene"].sum())
    strict_gly = int(
        (variant_scores["is_collagen_gene"] & variant_scores["is_gly_sub_strict"]).sum()
    )
    structural_active = int((variant_scores["n_structural_features_active"] > 0).sum())
    collagen_feature_active = int(
        (variant_scores["n_collagen_gly_features_active"] > 0).sum()
    )
    tier_counts = candidates["candidate_tier"].value_counts()
    tier1_count = int(
        tier_counts.get("T1_collagen_gly_with_collagen_gly_feature", 0)
    )

    top_candidates = candidates.head(args.top_n).copy()
    for col in [
        "novel_path_score",
        "structural_feature_score",
        "collagen_gly_feature_score",
        "top_structural_feature_activation",
        "top_structural_feature_path_rate",
        "top_structural_feature_struct_enrichment",
    ]:
        top_candidates[col] = top_candidates[col].apply(format_float)

    top_candidates = top_candidates.rename(
        columns={
            "candidate_tier": "tier",
            "variant_id": "variant",
            "pchange": "protein_change",
            "structural_feature_score": "struct_score",
            "collagen_gly_feature_score": "collagen_gly_score",
            "top_structural_feature": "top_feature",
            "top_structural_feature_struct_enrichment": "feature_enrich",
        }
    )

    gene_top = gene_summary.head(12).copy()
    for col in [
        "pct_structural_feature_active",
        "mean_structural_feature_score",
        "max_structural_feature_score",
        "top_structural_feature_score",
    ]:
        gene_top[col] = gene_top[col].apply(format_float)

    feature_top = feature_summary.head(12).copy()
    for col in [
        "train_path_rate",
        "train_struct_enrichment",
        "mean_vus_activation_abs",
        "corr_ESM1b",
        "corr_GPN",
        "corr_CADD",
    ]:
        feature_top[col] = feature_top[col].apply(format_float)

    lines = [
        "# VUS Collagen Interpretability Case Study",
        "",
        "## Immediate downstream application",
        "",
        (
            "The existing artifacts support a CPU-only VUS triage case study: "
            "rank structural-protein VUS by activation of SAE features that were "
            "previously identified as novel, pathogenic, and enriched for "
            "structural/collagen-glycine variants."
        ),
        "",
        "## Data sources",
        "",
        "- `results/vus_pilot/vus_mechanism_profiles.csv`",
        "- `results/vus_pilot/vus_sae_acts.npz`",
        "- `results/novel_features/novel_structural_enrichment.csv`",
        "- `results/sae_genomewide/sae_genomewide_cards_deep.csv`",
        "",
        "## Feature filter",
        "",
        (
            f"- Novel structural feature set: path_rate >= {args.min_path_rate}, "
            f"structural enrichment >= {args.min_struct_enrichment}, "
            f"n_structural >= {args.min_structural_support}."
        ),
        (
            f"- Collagen-gly feature subset: n_collagen >= {args.min_collagen_support}, "
            f"n_gly_collagen >= {args.min_gly_collagen_support}."
        ),
        f"- Activation threshold: abs(activation) > {args.activation_threshold}.",
        "",
        "## Headline counts",
        "",
        f"- VUS profiled: {total_vus:,}.",
        f"- Collagen-gene VUS: {collagen_vus:,}.",
        f"- Strict collagen Gly substitution VUS: {strict_gly:,}.",
        f"- Novel structural SAE features retained: {len(structural_features):,}.",
        f"- Collagen-gly SAE features retained: {len(collagen_gly_features):,}.",
        (
            f"- VUS activating at least one novel structural feature: "
            f"{structural_active:,} ({safe_percent(structural_active, total_vus):.1f}%)."
        ),
        (
            f"- VUS activating at least one collagen-gly feature: "
            f"{collagen_feature_active:,} ({safe_percent(collagen_feature_active, total_vus):.1f}%)."
        ),
        (
            f"- Tier 1 candidates, strict collagen Gly substitutions with a "
            f"collagen-gly feature active: {tier1_count:,}."
        ),
        "",
        "## Candidate tiers",
        "",
        "- Tier 1: collagen gene, strict Gly substitution, and collagen-gly feature active.",
        "- Tier 2: collagen gene and collagen-gly feature active, but not a strict Gly substitution.",
        "- Tier 3: any structural-protein VUS with a novel structural feature active.",
        "",
        "## Top ranked candidates",
        "",
        markdown_table(
            top_candidates,
            [
                "rank",
                "tier",
                "variant",
                "gene",
                "protein_change",
                "struct_score",
                "collagen_gly_score",
                "top_feature",
                "feature_enrich",
            ],
        ),
        "",
        "## Gene-level summary",
        "",
        markdown_table(
            gene_top.rename(
                columns={
                    "pct_structural_feature_active": "pct_struct_active",
                    "max_structural_feature_score": "max_struct_score",
                    "top_candidate_tier": "top_tier",
                    "top_variant_id": "top_variant",
                }
            ),
            [
                "gene",
                "n_vus",
                "n_collagen_gly_strict",
                "n_structural_feature_active",
                "n_collagen_gly_feature_active",
                "n_tier1_candidates",
                "pct_struct_active",
                "max_struct_score",
                "top_variant",
                "top_pchange",
                "top_tier",
            ],
        ),
        "",
        "## Feature-level evidence",
        "",
        markdown_table(
            feature_top.rename(
                columns={
                    "train_path_rate": "path_rate",
                    "train_struct_enrichment": "struct_enrich",
                    "n_vus_collagen_gly_strict_active": "n_vus_gly_col",
                    "mean_vus_activation_abs": "mean_abs_act",
                }
            ),
            [
                "feature",
                "path_rate",
                "struct_enrich",
                "train_n_collagen",
                "train_n_gly_collagen",
                "n_vus_active",
                "n_vus_collagen_active",
                "n_vus_gly_col",
                "mean_abs_act",
                "modality",
                "concept_v2",
            ],
        ),
        "",
        "## Interpretation note",
        "",
        (
            "These outputs are an interpretability-driven prioritization, not a "
            "clinical reclassification. The strongest immediate claim is that "
            "precomputed SAE features provide a concrete collagen/structural "
            "protein VUS triage table suitable for manual review and downstream "
            "validation."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()

    vus, acts, novel, cards = read_inputs(repo_root)
    structural_features, collagen_gly_features = feature_sets(novel, args)
    if structural_features.empty:
        raise ValueError("No novel structural features passed the configured filters.")

    variant_scores = summarize_variant_scores(
        vus=vus,
        acts=acts,
        cards=cards,
        structural_features=structural_features,
        collagen_gly_features=collagen_gly_features,
        activation_threshold=args.activation_threshold,
    )
    gene_summary = summarize_genes(variant_scores)
    feature_summary = summarize_features(
        variant_scores=variant_scores,
        acts=acts,
        cards=cards,
        structural_features=structural_features,
        activation_threshold=args.activation_threshold,
    )
    written = write_outputs(
        output_dir=output_dir,
        variant_scores=variant_scores,
        gene_summary=gene_summary,
        feature_summary=feature_summary,
        structural_features=structural_features,
        collagen_gly_features=collagen_gly_features,
        args=args,
    )

    print("Interpretability application: VUS collagen/structural-protein triage")
    print(f"VUS profiled: {len(variant_scores):,}")
    print(f"Novel structural features retained: {len(structural_features):,}")
    print(f"Collagen-gly features retained: {len(collagen_gly_features):,}")
    print(
        "Tier 1 candidates: "
        f"{(variant_scores['candidate_tier'] == 'T1_collagen_gly_with_collagen_gly_feature').sum():,}"
    )
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
