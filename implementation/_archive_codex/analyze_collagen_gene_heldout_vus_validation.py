#!/usr/bin/env python3
"""Apply gene-heldout collagen sparse features to VUS review prioritization."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OUT_DIR = Path("results/interpretability_applications")
DEFAULT_FOCUS_GENES = ("COL4A3", "COL4A5", "COL1A2", "COL3A1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=repo_root / OUT_DIR)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--min-top", type=int, default=10)
    parser.add_argument("--top-candidates-per-gene", type=int, default=25)
    parser.add_argument("--n-random-sets", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=630)
    parser.add_argument("--focus-genes", nargs="*", default=list(DEFAULT_FOCUS_GENES))
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def strict_gly_substitution(value: object) -> bool:
    text = "" if pd.isna(value) else str(value)
    return bool(re.match(r"^Gly\d+[A-Z][a-z]{2}$", text))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.where(series.notna(), "").astype(str).str.lower().isin({"true", "1", "yes", "y"})


def count_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def build_variant_id(df: pd.DataFrame) -> pd.Series:
    chrom = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["pos"], errors="coerce").astype("Int64").astype(str)
    ref = df["ref"].astype(str)
    alt = df["alt"].astype(str)
    return "chr" + chrom + ":" + pos + ":" + ref + ">" + alt


def score_features(acts: np.ndarray, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(acts.shape[0], dtype=float)
    return np.abs(acts[:, features]).sum(axis=1)


def fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def empirical_ge(observed: float, null_values: list[float]) -> float:
    arr = np.asarray(null_values, dtype=float)
    finite = np.isfinite(arr)
    if not math.isfinite(observed) or not finite.any():
        return float("nan")
    return float((1.0 + np.sum(arr[finite] >= observed)) / (1.0 + np.sum(finite)))


def top_mask(scores: pd.Series, top_n: int) -> pd.Series:
    top_n = int(min(max(top_n, 0), len(scores)))
    mask = pd.Series(False, index=scores.index)
    if top_n == 0:
        return mask
    top_idx = scores.sort_values(ascending=False, kind="mergesort").head(top_n).index
    mask.loc[top_idx] = True
    return mask


def fisher_top_vs_rest(df: pd.DataFrame, top: pd.Series, column: str) -> tuple[float, float, int, int]:
    hook = bool_series(df[column])
    top_hook = int((top & hook).sum())
    top_nonhook = int((top & ~hook).sum())
    rest_hook = int((~top & hook).sum())
    rest_nonhook = int((~top & ~hook).sum())
    if top.sum() == 0 or (~top).sum() == 0:
        return float("nan"), float("nan"), top_hook, rest_hook
    odds_ratio, pvalue = fisher_exact(
        [[top_hook, top_nonhook], [rest_hook, rest_nonhook]],
        alternative="greater",
    )
    return float(odds_ratio), float(pvalue), top_hook, rest_hook


def matched_random_sets(
    selected: np.ndarray,
    cards: pd.DataFrame,
    n_features: int,
    n_sets: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    meta = pd.DataFrame({"feature": np.arange(n_features, dtype=int)})
    cards_small = cards[["feature", "n_active", "path_rate"]].copy()
    meta = meta.merge(cards_small, on="feature", how="left")
    meta["n_active"] = meta["n_active"].fillna(meta["n_active"].median())
    meta["path_rate"] = meta["path_rate"].fillna(meta["path_rate"].median())
    meta["active_bin"] = pd.qcut(
        meta["n_active"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )
    meta["path_bin"] = pd.qcut(
        meta["path_rate"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )
    selected_set = set(int(x) for x in selected)
    rows_by_feature = meta.set_index("feature")
    all_pool = meta.loc[~meta["feature"].isin(selected_set), "feature"].astype(int).to_numpy()

    random_sets: list[np.ndarray] = []
    for _ in range(n_sets):
        chosen: list[int] = []
        used = set(selected_set)
        for feature in selected:
            row = rows_by_feature.loc[int(feature)]
            pool = meta[
                (meta["active_bin"].eq(row["active_bin"]))
                & (meta["path_bin"].eq(row["path_bin"]))
                & ~meta["feature"].isin(used)
            ]["feature"].astype(int).to_numpy()
            if len(pool) == 0:
                pool = np.array([x for x in all_pool if x not in used], dtype=int)
            if len(pool) == 0:
                pool = all_pool
            pick = int(rng.choice(pool))
            chosen.append(pick)
            used.add(pick)
        random_sets.append(np.asarray(chosen, dtype=int))
    return random_sets


def read_vus(repo: Path) -> tuple[pd.DataFrame, np.ndarray]:
    vus = pd.read_csv(require_file(repo / "results/vus_pilot/vus_mechanism_profiles.csv"), dtype={"chrom": str})
    acts = np.load(require_file(repo / "results/vus_pilot/vus_sae_acts.npz"))["acts"].astype(np.float32, copy=False)
    if len(vus) != acts.shape[0]:
        raise ValueError(f"VUS table and activation rows differ: {len(vus)} vs {acts.shape[0]}")
    vus = vus.copy()
    vus["variant_id"] = build_variant_id(vus)
    vus["gene"] = vus["gene"].fillna("")
    vus["pchange"] = vus["pchange"].fillna("")
    vus["is_gly_sub_strict"] = vus["pchange"].apply(strict_gly_substitution)
    if "is_gly_sub" in vus.columns:
        vus["source_is_gly_sub"] = bool_series(vus["is_gly_sub"])
    else:
        vus["source_is_gly_sub"] = False
    return vus, acts


def read_evidence(repo: Path) -> pd.DataFrame:
    path = require_file(repo / OUT_DIR / "vus_application_readiness_candidates.csv")
    evidence = pd.read_csv(path, dtype={"chrom": str})
    keep = [
        "variant_id",
        "rank",
        "candidate_tier",
        "collagen_gly_feature_score",
        "n_collagen_gly_features_active",
        "top_structural_feature",
        "top_structural_feature_activation",
        "top_structural_feature_path_rate",
        "candidate_context_strength",
        "has_lsdb_link",
        "has_gene_disease_id",
        "same_residue_pathogenic_count",
        "same_residue_benign_count",
        "same_aa_change_pathogenic_count",
        "same_aa_change_benign_count",
        "feature_known_pathogenic_count",
        "feature_known_benign_count",
        "feature_known_path_rate",
        "clinvar_evidence_level",
        "application_evidence_level",
        "is_actionable_review_hook",
        "has_common_negative_warning",
    ]
    keep = [col for col in keep if col in evidence.columns]
    evidence = evidence[keep].drop_duplicates("variant_id", keep="first").copy()
    return evidence


def add_hook_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "rank": np.nan,
        "candidate_tier": "",
        "collagen_gly_feature_score": 0.0,
        "n_collagen_gly_features_active": 0,
        "top_structural_feature": np.nan,
        "top_structural_feature_activation": 0.0,
        "top_structural_feature_path_rate": 0.0,
        "candidate_context_strength": 0,
        "has_lsdb_link": False,
        "has_gene_disease_id": False,
        "same_residue_pathogenic_count": 0,
        "same_residue_benign_count": 0,
        "same_aa_change_pathogenic_count": 0,
        "same_aa_change_benign_count": 0,
        "feature_known_pathogenic_count": 0,
        "feature_known_benign_count": 0,
        "feature_known_path_rate": 0.0,
        "clinvar_evidence_level": "",
        "application_evidence_level": "",
        "is_actionable_review_hook": False,
        "has_common_negative_warning": False,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        else:
            out[col] = out[col].where(out[col].notna(), value)

    for col in [
        "same_residue_pathogenic_count",
        "same_residue_benign_count",
        "same_aa_change_pathogenic_count",
        "same_aa_change_benign_count",
        "feature_known_pathogenic_count",
        "feature_known_benign_count",
        "candidate_context_strength",
        "n_collagen_gly_features_active",
    ]:
        out[col] = count_series(out[col])
    for col in ["is_actionable_review_hook", "has_lsdb_link", "has_gene_disease_id", "has_common_negative_warning"]:
        out[col] = bool_series(out[col])
    for col in ["collagen_gly_feature_score", "top_structural_feature_activation", "top_structural_feature_path_rate", "feature_known_path_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["clinvar_residue_hook"] = (
        (
            out["same_residue_pathogenic_count"].gt(0)
            & out["same_residue_benign_count"].eq(0)
        )
        | (
            out["same_aa_change_pathogenic_count"].gt(0)
            & out["same_aa_change_benign_count"].eq(0)
        )
    )
    out["feature_known_pathogenic_support"] = (
        out["feature_known_pathogenic_count"].ge(5)
        & out["feature_known_path_rate"].ge(0.80)
        & out["feature_known_pathogenic_count"].gt(out["feature_known_benign_count"])
    )
    out["gene_context_hook"] = (
        out["candidate_context_strength"].ge(3)
        | out["has_lsdb_link"]
        | out["has_gene_disease_id"]
    )
    out["application_hook"] = (
        out["is_actionable_review_hook"]
        | out["clinvar_residue_hook"]
        | out["gene_context_hook"]
    )
    return out


def load_known_loo(repo: Path) -> pd.DataFrame:
    path = repo / OUT_DIR / "collagen_leave_one_gene_out_primary_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["holdout_gene", "known_loo_pass", "observed_auroc", "empirical_p_random_auroc_ge_observed"])
    df = pd.read_csv(path)
    primary = df[df["control"].eq("pathogenic_nonstructural_missense")].copy()
    primary["known_loo_pass"] = (
        primary["observed_auroc"].ge(0.60)
        & primary["empirical_p_random_auroc_ge_observed"].le(0.05)
    )
    return primary[["holdout_gene", "known_loo_pass", "observed_auroc", "empirical_p_random_auroc_ge_observed"]]


def evaluate_gene(
    gene_df: pd.DataFrame,
    gene_acts: np.ndarray,
    selected_features: np.ndarray,
    random_sets: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    observed_scores = score_features(gene_acts, selected_features)
    gene_df = gene_df.copy()
    gene_df["heldout_feature_score"] = observed_scores
    gene_df["heldout_feature_rank_in_gene"] = gene_df["heldout_feature_score"].rank(method="first", ascending=False).astype(int)
    gene_df["heldout_feature_percentile_in_gene"] = gene_df["heldout_feature_score"].rank(pct=True, ascending=True)

    n_vus = len(gene_df)
    top_n = min(n_vus, max(args.min_top, int(math.ceil(args.top_frac * n_vus))))
    top = top_mask(gene_df["heldout_feature_score"], top_n)
    gene_df["heldout_feature_top_set"] = top

    endpoints = [
        "is_gly_sub_strict",
        "clinvar_residue_hook",
        "is_actionable_review_hook",
        "feature_known_pathogenic_support",
        "gene_context_hook",
        "application_hook",
    ]
    row: dict[str, object] = {
        "holdout_gene": str(gene_df["gene"].iloc[0]),
        "n_vus": n_vus,
        "n_selected_features": len(selected_features),
        "top_frac": args.top_frac,
        "top_n": top_n,
        "n_gly_vus": int(bool_series(gene_df["is_gly_sub_strict"]).sum()),
        "observed_top_score_min": float(gene_df.loc[top, "heldout_feature_score"].min()) if top.any() else float("nan"),
        "observed_score_median": float(gene_df["heldout_feature_score"].median()),
    }

    for endpoint in endpoints:
        hook = bool_series(gene_df[endpoint])
        top_rate = float(hook[top].mean()) if top.any() else float("nan")
        rest_rate = float(hook[~top].mean()) if (~top).any() else float("nan")
        odds_ratio, fisher_p, top_hook, rest_hook = fisher_top_vs_rest(gene_df, top, endpoint)
        row[f"{endpoint}_top_n"] = top_hook
        row[f"{endpoint}_rest_n"] = rest_hook
        row[f"{endpoint}_top_rate"] = top_rate
        row[f"{endpoint}_rest_rate"] = rest_rate
        row[f"{endpoint}_odds_ratio"] = odds_ratio
        row[f"{endpoint}_fisher_p"] = fisher_p

    random_rows = []
    for i, features in enumerate(random_sets):
        random_scores = score_features(gene_acts, features)
        random_top = top_mask(pd.Series(random_scores, index=gene_df.index), top_n)
        random_row: dict[str, object] = {
            "holdout_gene": row["holdout_gene"],
            "random_set": i,
            "random_top_score_min": float(np.min(random_scores[random_top.to_numpy()])) if random_top.any() else float("nan"),
        }
        for endpoint in endpoints:
            hook = bool_series(gene_df[endpoint])
            random_row[f"{endpoint}_top_rate"] = float(hook[random_top].mean()) if random_top.any() else float("nan")
        random_rows.append(random_row)
    random_null = pd.DataFrame(random_rows)
    for endpoint in endpoints:
        null_col = f"{endpoint}_top_rate"
        observed = float(row[f"{endpoint}_top_rate"])
        null_values = random_null[null_col].tolist() if not random_null.empty else []
        row[f"{endpoint}_random_top_rate_mean"] = float(np.nanmean(random_null[null_col])) if not random_null.empty else float("nan")
        row[f"{endpoint}_random_top_rate_sd"] = float(np.nanstd(random_null[null_col])) if not random_null.empty else float("nan")
        row[f"{endpoint}_random_ge_observed_p"] = empirical_ge(observed, null_values)

    return row, random_null, gene_df


def write_report(
    path: Path,
    summary: pd.DataFrame,
    top_candidates: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    focus = summary[summary["holdout_gene"].isin(args.focus_genes)].copy()
    known_pass = summary[summary["known_loo_pass"].fillna(False)].copy() if "known_loo_pass" in summary.columns else pd.DataFrame()
    primary_cols = [
        "holdout_gene",
        "known_loo_pass",
        "known_loo_auroc",
        "n_vus",
        "top_n",
        "is_gly_sub_strict_top_rate",
        "clinvar_residue_hook_top_rate",
        "clinvar_residue_hook_rest_rate",
        "clinvar_residue_hook_fisher_p",
        "clinvar_residue_hook_random_ge_observed_p",
        "is_actionable_review_hook_top_rate",
        "is_actionable_review_hook_rest_rate",
        "is_actionable_review_hook_random_ge_observed_p",
        "application_hook_top_rate",
        "application_hook_rest_rate",
        "application_hook_random_ge_observed_p",
    ]
    primary_cols = [col for col in primary_cols if col in summary.columns]

    lines = [
        "# Collagen Gene-Heldout VUS Validation",
        "",
        "## Purpose",
        "",
        "This analysis applies each collagen gene's leave-one-gene-out sparse features to VUS in the same held-out gene. Feature selection was performed without that gene's known pathogenic Gly-X-Y variants, so this is a stricter VUS prioritization check than pooled collagen-gly scoring.",
        "",
        "## Configuration",
        "",
        f"- top fraction per gene: {args.top_frac:.2f}",
        f"- minimum top-set size: {args.min_top}",
        f"- matched random feature sets: {args.n_random_sets}",
        f"- focus genes: {', '.join(args.focus_genes)}",
        "",
        "## Main Readout",
        "",
        f"- genes evaluated: {len(summary)}",
        f"- known-label LOO pass genes with VUS: {len(known_pass)}",
    ]
    if not known_pass.empty:
        lines.append(
            "- known-label LOO pass genes median top-set ClinVar-residue-hook rate: "
            f"{fmt(known_pass['clinvar_residue_hook_top_rate'].median())}"
        )
        lines.append(
            "- known-label LOO pass genes median top-set actionable-review-hook rate: "
            f"{fmt(known_pass['is_actionable_review_hook_top_rate'].median())}"
        )
    lines.extend(["", "## Per-Gene Summary", ""])
    if summary.empty:
        lines.append("_No evaluable genes._")
    else:
        lines.append(
            summary[primary_cols]
            .sort_values(["known_loo_pass", "application_hook_top_rate", "clinvar_residue_hook_top_rate"], ascending=[False, False, False])
            .to_markdown(index=False)
        )

    lines.extend(["", "## Focus-Gene Candidates", ""])
    if top_candidates.empty:
        lines.append("_No top candidates exported._")
    else:
        cols = [
            "holdout_gene",
            "heldout_feature_rank_in_gene",
            "variant_id",
            "pchange",
            "heldout_feature_score",
            "clinvar_residue_hook",
            "is_actionable_review_hook",
            "application_evidence_level",
        ]
        cols = [col for col in cols if col in top_candidates.columns]
        lines.append(
            top_candidates[top_candidates["holdout_gene"].isin(args.focus_genes)]
            .sort_values(["holdout_gene", "heldout_feature_rank_in_gene"])
            .head(40)[cols]
            .to_markdown(index=False)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A positive row means the gene-heldout feature set ranks VUS with external residue or review hooks above the rest of that gene's VUS and above matched random feature sets.",
            "- This is still a prioritization/review signal. It is not a temporal reclassification endpoint, assay outcome, or clinical reclassification claim.",
            "- Genes with weak random-null p-values should remain supplementary boundary cases even when their top variants look biologically plausible.",
            "",
            "## Outputs",
            "",
            "- `collagen_gene_heldout_vus_validation_summary.csv`",
            "- `collagen_gene_heldout_vus_validation_random_null.csv`",
            "- `collagen_gene_heldout_vus_validation_variant_scores.csv`",
            "- `collagen_gene_heldout_vus_validation_top_candidates.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    selected = pd.read_csv(require_file(repo / OUT_DIR / "collagen_leave_one_gene_out_selected_features.csv"))
    cards = pd.read_csv(require_file(repo / "results/sae_genomewide/sae_genomewide_cards_deep.csv"))
    vus, acts = read_vus(repo)
    evidence = read_evidence(repo)
    known_loo = load_known_loo(repo)

    merged = vus.merge(evidence, on="variant_id", how="left")
    merged = add_hook_columns(merged)
    selected_by_gene = {
        gene: sub["feature"].astype(int).to_numpy()
        for gene, sub in selected.groupby("holdout_gene")
    }

    summary_rows: list[dict[str, object]] = []
    null_rows: list[pd.DataFrame] = []
    variant_rows: list[pd.DataFrame] = []
    for gene, features in sorted(selected_by_gene.items()):
        gene_mask = merged["gene"].eq(gene).to_numpy()
        if not gene_mask.any():
            continue
        gene_df = merged.loc[gene_mask].copy()
        gene_acts = acts[gene_mask]
        random_sets = matched_random_sets(features, cards, acts.shape[1], args.n_random_sets, rng)
        row, random_null, scored_gene_df = evaluate_gene(gene_df, gene_acts, features, random_sets, args)
        summary_rows.append(row)
        null_rows.append(random_null)
        scored_gene_df["holdout_gene"] = gene
        scored_gene_df["n_selected_features"] = len(features)
        variant_rows.append(scored_gene_df)

    summary = pd.DataFrame(summary_rows)
    random_null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    variants = pd.concat(variant_rows, ignore_index=True) if variant_rows else pd.DataFrame()

    if not summary.empty and not known_loo.empty:
        known_small = known_loo.rename(
            columns={
                "observed_auroc": "known_loo_auroc",
                "empirical_p_random_auroc_ge_observed": "known_loo_random_p",
            }
        )
        summary = summary.merge(known_small, on="holdout_gene", how="left")
        summary["known_loo_pass"] = summary["known_loo_pass"].fillna(False)
    elif not summary.empty:
        summary["known_loo_pass"] = False
        summary["known_loo_auroc"] = np.nan
        summary["known_loo_random_p"] = np.nan

    top_candidates = variants[variants["heldout_feature_rank_in_gene"].le(args.top_candidates_per_gene)].copy() if not variants.empty else pd.DataFrame()
    if not top_candidates.empty:
        export_cols = [
            "holdout_gene",
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "pchange",
            "clin_sig",
            "heldout_feature_rank_in_gene",
            "heldout_feature_percentile_in_gene",
            "heldout_feature_score",
            "is_gly_sub_strict",
            "candidate_tier",
            "collagen_gly_feature_score",
            "clinvar_residue_hook",
            "same_residue_pathogenic_count",
            "same_residue_benign_count",
            "same_aa_change_pathogenic_count",
            "same_aa_change_benign_count",
            "feature_known_pathogenic_support",
            "feature_known_pathogenic_count",
            "feature_known_benign_count",
            "feature_known_path_rate",
            "gene_context_hook",
            "is_actionable_review_hook",
            "application_hook",
            "clinvar_evidence_level",
            "application_evidence_level",
            "has_common_negative_warning",
        ]
        export_cols = [col for col in export_cols if col in top_candidates.columns]
        top_candidates = top_candidates[export_cols].sort_values(["holdout_gene", "heldout_feature_rank_in_gene"])

    summary.to_csv(out / "collagen_gene_heldout_vus_validation_summary.csv", index=False)
    random_null.to_csv(out / "collagen_gene_heldout_vus_validation_random_null.csv", index=False)
    variants.to_csv(out / "collagen_gene_heldout_vus_validation_variant_scores.csv", index=False)
    top_candidates.to_csv(out / "collagen_gene_heldout_vus_validation_top_candidates.csv", index=False)
    write_report(out / "collagen_gene_heldout_vus_validation.md", summary, top_candidates, args)

    print(f"wrote {out / 'collagen_gene_heldout_vus_validation_summary.csv'}")
    print(f"wrote {out / 'collagen_gene_heldout_vus_validation_random_null.csv'}")
    print(f"wrote {out / 'collagen_gene_heldout_vus_validation_variant_scores.csv'}")
    print(f"wrote {out / 'collagen_gene_heldout_vus_validation_top_candidates.csv'}")
    print(f"wrote {out / 'collagen_gene_heldout_vus_validation.md'}")


if __name__ == "__main__":
    main()
