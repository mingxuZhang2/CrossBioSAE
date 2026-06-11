#!/usr/bin/env python3
"""
Annotate interpretability-prioritized VUS candidates with current ClinVar evidence.

This script adds variant-level context that is still available locally:

1. exact coordinate overlap with known P/B benchmark rows,
2. same-gene/same-residue pathogenic and benign ClinVar evidence,
3. same amino-acid substitution evidence,
4. known ClinVar behavior of the candidate's top SAE structural feature.

The output is not a temporal reclassification experiment, but it moves the VUS
application beyond gene-level context and provides a manual-review evidence
table suitable for the next validation pass.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--activation-threshold", type=float, default=1e-6)
    parser.add_argument("--top-n", type=int, default=40)
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


def parse_pchange(value: object) -> tuple[str, float, str]:
    text = clean(value)
    match = re.match(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|Ter|\*)$", text)
    if not match:
        return "", float("nan"), ""
    return match.group(1), float(match.group(2)), match.group(3)


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


def add_pchange_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["pchange"].apply(parse_pchange)
    out["aa_ref"] = [x[0] for x in parsed]
    out["aa_pos"] = [x[1] for x in parsed]
    out["aa_alt"] = [x[2] for x in parsed]
    out["residue_key"] = [
        f"{gene}:{int(pos)}" if gene and np.isfinite(pos) else ""
        for gene, pos in zip(out["gene"].fillna("").astype(str), out["aa_pos"])
    ]
    out["aa_change_key"] = [
        f"{gene}:{ref}{int(pos)}{alt}" if gene and ref and alt and np.isfinite(pos) else ""
        for gene, ref, pos, alt in zip(
            out["gene"].fillna("").astype(str),
            out["aa_ref"],
            out["aa_pos"],
            out["aa_alt"],
        )
    ]
    return out


def join_unique(values: pd.Series, max_items: int = 6) -> str:
    uniq = [clean(v) for v in values.dropna().astype(str).unique() if clean(v)]
    uniq = sorted(uniq)
    if len(uniq) > max_items:
        return ";".join(uniq[:max_items]) + f";...(+{len(uniq) - max_items})"
    return ";".join(uniq)


def build_residue_summaries(known: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residue_rows = []
    for key, grp in known[known["residue_key"].ne("")].groupby("residue_key"):
        path = grp[grp["label"].eq(1)]
        benign = grp[grp["label"].eq(0)]
        residue_rows.append(
            {
                "residue_key": key,
                "same_residue_known_count": len(grp),
                "same_residue_pathogenic_count": len(path),
                "same_residue_benign_count": len(benign),
                "same_residue_pathogenic_pchanges": join_unique(path["pchange"]),
                "same_residue_benign_pchanges": join_unique(benign["pchange"]),
                "same_residue_pathogenic_coords": join_unique(path["variant_key"]),
                "same_residue_benign_coords": join_unique(benign["variant_key"]),
            }
        )

    change_rows = []
    for key, grp in known[known["aa_change_key"].ne("")].groupby("aa_change_key"):
        path = grp[grp["label"].eq(1)]
        benign = grp[grp["label"].eq(0)]
        change_rows.append(
            {
                "aa_change_key": key,
                "same_aa_change_known_count": len(grp),
                "same_aa_change_pathogenic_count": len(path),
                "same_aa_change_benign_count": len(benign),
                "same_aa_change_pathogenic_coords": join_unique(path["variant_key"]),
                "same_aa_change_benign_coords": join_unique(benign["variant_key"]),
            }
        )

    return pd.DataFrame(residue_rows), pd.DataFrame(change_rows)


def feature_known_evidence(
    repo_root: Path,
    known: pd.DataFrame,
    candidate_features: np.ndarray,
    activation_threshold: float,
) -> pd.DataFrame:
    acts = np.load(require_file(repo_root / "results" / "sae_genomewide" / "sae_genomewide_acts.npz"))["acts"]
    rows = []
    for feature in sorted(set(int(x) for x in candidate_features if int(x) >= 0)):
        if feature >= acts.shape[1]:
            continue
        active = np.abs(acts[:, feature]) > activation_threshold
        sub = known.loc[active]
        if len(sub) == 0:
            rows.append(
                {
                    "top_structural_feature": feature,
                    "feature_known_active_count": 0,
                    "feature_known_pathogenic_count": 0,
                    "feature_known_benign_count": 0,
                    "feature_known_path_rate": float("nan"),
                    "feature_known_collagen_gly_pathogenic_count": 0,
                    "feature_known_structural_pathogenic_count": 0,
                }
            )
            continue
        path = sub[sub["label"].eq(1)]
        rows.append(
            {
                "top_structural_feature": feature,
                "feature_known_active_count": len(sub),
                "feature_known_pathogenic_count": int(sub["label"].sum()),
                "feature_known_benign_count": int((sub["label"] == 0).sum()),
                "feature_known_path_rate": float(sub["label"].mean()),
                "feature_known_collagen_gly_pathogenic_count": int(
                    (path["is_collagen_gene"].astype(bool) & path["is_gly_sub"].astype(bool)).sum()
                ),
                "feature_known_structural_pathogenic_count": int(
                    (path["is_structural_gene"].astype(bool)).sum()
                ),
                "feature_known_top_pathogenic_genes": join_unique(path["gene"].value_counts().head(8).index.to_series()),
                "feature_known_top_pathogenic_pchanges": join_unique(path["pchange"], max_items=8),
            }
        )
    return pd.DataFrame(rows)


def exact_known_overlap(vus: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    exact = (
        known.groupby("variant_key")
        .agg(
            exact_known_count=("variant_key", "size"),
            exact_known_pathogenic_count=("label", "sum"),
            exact_known_benign_count=("label", lambda s: int((s == 0).sum())),
            exact_known_pchanges=("pchange", join_unique),
        )
        .reset_index()
    )
    return vus.merge(exact, on="variant_key", how="left")


def evidence_level(row: pd.Series) -> str:
    if row.get("same_aa_change_pathogenic_count", 0) > 0:
        return "A_same_amino_acid_change_pathogenic"
    if row.get("same_residue_pathogenic_count", 0) > 0 and row.get("same_residue_benign_count", 0) == 0:
        return "B_same_residue_pathogenic_no_benign"
    if row.get("same_residue_pathogenic_count", 0) > 0:
        return "C_same_residue_mixed"
    if row.get("feature_known_collagen_gly_pathogenic_count", 0) >= 10:
        return "D_feature_collagen_gly_known_pathogenic"
    if row.get("feature_known_path_rate", 0) >= 0.85:
        return "E_feature_high_path_rate"
    return "F_gene_context_only"


def write_report(path: Path, annotated: pd.DataFrame, feature_evidence: pd.DataFrame, top_n: int) -> None:
    tier1 = annotated[annotated["candidate_tier"].eq("T1_collagen_gly_with_collagen_gly_feature")]
    same_res = annotated[annotated["same_residue_pathogenic_count"].fillna(0).gt(0)]
    same_change = annotated[annotated["same_aa_change_pathogenic_count"].fillna(0).gt(0)]

    level_counts = (
        annotated["clinvar_evidence_level"]
        .value_counts()
        .rename_axis("clinvar_evidence_level")
        .reset_index(name="n_candidates")
    )
    tier1_level_counts = (
        tier1["clinvar_evidence_level"]
        .value_counts()
        .rename_axis("clinvar_evidence_level")
        .reset_index(name="n_tier1_candidates")
    )
    level_counts = level_counts.merge(tier1_level_counts, on="clinvar_evidence_level", how="left").fillna(0)

    top_cols = [
        "rank",
        "candidate_tier",
        "variant_id",
        "gene",
        "pchange",
        "collagen_gly_feature_score",
        "top_structural_feature",
        "same_residue_pathogenic_count",
        "same_residue_benign_count",
        "same_residue_pathogenic_pchanges",
        "same_aa_change_pathogenic_count",
        "feature_known_path_rate",
        "feature_known_collagen_gly_pathogenic_count",
        "clinvar_evidence_level",
    ]
    top = annotated.sort_values(
        [
            "same_aa_change_pathogenic_count",
            "same_residue_pathogenic_count",
            "collagen_gly_feature_score",
        ],
        ascending=[False, False, False],
    ).head(top_n)[top_cols].copy()
    for col in ["collagen_gly_feature_score", "feature_known_path_rate"]:
        top[col] = top[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    feat = feature_evidence.sort_values(
        ["feature_known_collagen_gly_pathogenic_count", "feature_known_path_rate"],
        ascending=[False, False],
    ).copy()
    for col in ["feature_known_path_rate"]:
        feat[col] = feat[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    lines = [
        "# VUS Candidate ClinVar Residue and Feature Evidence",
        "",
        "## Purpose",
        "",
        "Annotate interpretability-prioritized VUS candidates with current local ClinVar P/B evidence at the exact variant, amino-acid change, residue, and SAE-feature levels. This is not temporal validation.",
        "",
        "## Headline Counts",
        "",
        f"- Candidate rows annotated: {len(annotated):,}.",
        f"- Tier 1 collagen-gly VUS candidates: {len(tier1):,}.",
        f"- Candidates with same-residue known pathogenic ClinVar evidence: {len(same_res):,}.",
        f"- Tier 1 candidates with same-residue known pathogenic evidence: {int(tier1['same_residue_pathogenic_count'].fillna(0).gt(0).sum()):,}.",
        f"- Candidates with exact same amino-acid-change known pathogenic evidence: {len(same_change):,}.",
        f"- Candidates with same-residue benign evidence: {int(annotated['same_residue_benign_count'].fillna(0).gt(0).sum()):,}.",
        f"- Exact coordinate overlap with known P/B rows: {int(annotated['exact_known_count'].fillna(0).gt(0).sum()):,}.",
        "",
        "## Evidence Level Counts",
        "",
        level_counts.to_markdown(index=False),
        "",
        "## Strongest Candidate-Level ClinVar Evidence",
        "",
        top.to_markdown(index=False),
        "",
        "## Feature-Level Known ClinVar Evidence",
        "",
        feat.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Same-residue pathogenic evidence gives a concrete manual-review hook for a subset of VUS candidates. Most candidates still lack direct residue-level support, so the next top-journal validation remains temporal ClinVar reclassification or external variant databases such as LOVD/ClinGen/gnomAD AF.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vus = pd.read_csv(require_file(out_dir / "vus_candidate_gene_context.csv"))
    known = pd.read_csv(require_file(out_dir / "known_variant_novel_scores.csv"))

    vus = add_pchange_parts(vus)
    known = add_pchange_parts(known)
    vus["variant_key"] = variant_key_df(vus)
    known["variant_key"] = variant_key_df(known)
    known["label"] = known["label"].astype(int)

    residue_summary, change_summary = build_residue_summaries(known)
    feature_evidence = feature_known_evidence(
        repo_root,
        known,
        vus["top_structural_feature"].fillna(-1).astype(int).to_numpy(),
        args.activation_threshold,
    )

    annotated = exact_known_overlap(vus, known)
    annotated = annotated.merge(residue_summary, on="residue_key", how="left")
    annotated = annotated.merge(change_summary, on="aa_change_key", how="left")
    annotated = annotated.merge(feature_evidence, on="top_structural_feature", how="left")

    count_cols = [
        "exact_known_count",
        "exact_known_pathogenic_count",
        "exact_known_benign_count",
        "same_residue_known_count",
        "same_residue_pathogenic_count",
        "same_residue_benign_count",
        "same_aa_change_known_count",
        "same_aa_change_pathogenic_count",
        "same_aa_change_benign_count",
        "feature_known_active_count",
        "feature_known_pathogenic_count",
        "feature_known_benign_count",
        "feature_known_collagen_gly_pathogenic_count",
        "feature_known_structural_pathogenic_count",
    ]
    for col in count_cols:
        annotated[col] = annotated[col].fillna(0).astype(int)
    for col in [
        "same_residue_pathogenic_pchanges",
        "same_residue_benign_pchanges",
        "same_residue_pathogenic_coords",
        "same_residue_benign_coords",
        "same_aa_change_pathogenic_coords",
        "same_aa_change_benign_coords",
        "feature_known_top_pathogenic_genes",
        "feature_known_top_pathogenic_pchanges",
        "exact_known_pchanges",
    ]:
        if col in annotated:
            annotated[col] = annotated[col].fillna("")

    annotated["clinvar_evidence_level"] = annotated.apply(evidence_level, axis=1)

    summary_rows = []
    for group_name, mask in {
        "all_candidates": np.ones(len(annotated), dtype=bool),
        "tier1_collagen_gly": annotated["candidate_tier"].eq("T1_collagen_gly_with_collagen_gly_feature"),
        "strict_collagen_gly": annotated["is_collagen_gene"].astype(bool) & annotated["is_gly_sub_strict"].astype(bool),
    }.items():
        sub = annotated[mask]
        summary_rows.append(
            {
                "group": group_name,
                "n": len(sub),
                "same_residue_pathogenic": int(sub["same_residue_pathogenic_count"].gt(0).sum()),
                "same_residue_benign": int(sub["same_residue_benign_count"].gt(0).sum()),
                "same_aa_change_pathogenic": int(sub["same_aa_change_pathogenic_count"].gt(0).sum()),
                "exact_known_overlap": int(sub["exact_known_count"].gt(0).sum()),
                "feature_path_rate_ge_0_85": int(sub["feature_known_path_rate"].fillna(0).ge(0.85).sum()),
                "feature_collagen_gly_path_ge_10": int(sub["feature_known_collagen_gly_pathogenic_count"].ge(10).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    annotated.to_csv(out_dir / "vus_candidate_clinvar_residue_evidence.csv", index=False)
    summary.to_csv(out_dir / "vus_candidate_clinvar_residue_summary.csv", index=False)
    feature_evidence.to_csv(out_dir / "vus_candidate_feature_known_evidence.csv", index=False)
    write_report(
        out_dir / "vus_candidate_clinvar_residue_evidence.md",
        annotated,
        feature_evidence,
        args.top_n,
    )

    print(f"Wrote VUS candidate ClinVar evidence outputs to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
