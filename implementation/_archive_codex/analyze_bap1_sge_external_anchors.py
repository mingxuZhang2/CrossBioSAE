#!/usr/bin/env python3
"""Analyze clinical/population anchors for the BAP1 SGE third-gene benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("results/interpretability_applications")
PLOF_CLASSES = {
    "frameshift",
    "stop_gained",
    "splice_acceptor",
    "splice_donor",
    "start_lost",
    "stop_lost",
}
TOLERATED_CLASSES = {"synonymous", "UTR"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--variants", type=Path, default=repo_root / OUT_DIR / "bap1_sge_variants.csv")
    parser.add_argument("--output-prefix", default="bap1_sge_external_anchors")
    return parser.parse_args()


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def as_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if text in {"", "-", "."}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def binary_label(df: pd.DataFrame) -> pd.Series:
    return df["functional_classification"].map({"depleted": 1, "unchanged": 0})


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


def group_row(name: str, group: pd.DataFrame) -> dict[str, Any]:
    labels = binary_label(group)
    scores = -pd.to_numeric(group["function_score"], errors="coerce")
    return {
        "group": name,
        "n_total": int(len(group)),
        "n_binary": int(labels.notna().sum()),
        "n_depleted": int(labels.eq(1).sum()),
        "n_unchanged": int(labels.eq(0).sum()),
        "n_enriched": int(group["functional_classification"].eq("enriched").sum()),
        "depleted_rate_binary": float(labels.dropna().mean()) if labels.notna().any() else float("nan"),
        "mean_external_lof_score": float(scores.mean()) if scores.notna().any() else float("nan"),
        "median_external_lof_score": float(scores.median()) if scores.notna().any() else float("nan"),
    }


def build_anchors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["binary_depleted"] = binary_label(df)
    df["external_lof_score"] = -pd.to_numeric(df["function_score"], errors="coerce")
    df["gnomad_af_v3_numeric"] = df["gnomAD_AF_v3"].map(as_float) if "gnomAD_AF_v3" in df else np.nan
    slim = df["clinvar_clinical_significance_slim"].fillna("Unobserved").astype(str)
    p_lp = df[slim.eq("Pathogenic/Likely pathogenic")].copy()
    b_lb = df[slim.eq("Benign/Likely benign")].copy()
    vus = df[slim.eq("Uncertain significance")].copy()
    conflict = df[slim.eq("Conflicting interpretation")].copy()
    unobserved = df[slim.eq("Unobserved")].copy()
    gnomad = df[df["is_in_gnomad"].astype(str).str.upper().eq("Y")].copy()
    not_gnomad = df[df["is_in_gnomad"].astype(str).str.upper().ne("Y")].copy()
    plof = df[df["consequence"].isin(PLOF_CLASSES)].copy()
    tolerated = df[df["consequence"].isin(TOLERATED_CLASSES)].copy()
    missense = df[df["consequence"].eq("missense")].copy()

    group_summary = pd.DataFrame(
        [
            group_row("clinvar_pathogenic_likely_pathogenic", p_lp),
            group_row("clinvar_benign_likely_benign", b_lb),
            group_row("clinvar_vus", vus),
            group_row("clinvar_conflicting", conflict),
            group_row("clinvar_unobserved", unobserved),
            group_row("gnomad_observed", gnomad),
            group_row("gnomad_unobserved", not_gnomad),
            group_row("predicted_plof_consequences", plof),
            group_row("synonymous_or_utr_controls", tolerated),
            group_row("missense", missense),
        ]
    )

    tests = []
    odds, pval = fisher_binary(p_lp["binary_depleted"], b_lb["binary_depleted"], "greater")
    tests.append(
        {
            "test": "clinvar_pathogenic_enriched_for_depleted_vs_benign",
            "first_group": "clinvar_pathogenic_likely_pathogenic",
            "second_group": "clinvar_benign_likely_benign",
            "first_n_binary": int(p_lp["binary_depleted"].notna().sum()),
            "first_n_depleted": int(p_lp["binary_depleted"].eq(1).sum()),
            "second_n_binary": int(b_lb["binary_depleted"].notna().sum()),
            "second_n_depleted": int(b_lb["binary_depleted"].eq(1).sum()),
            "fisher_oddsratio": odds,
            "fisher_p": pval,
            "external_lof_auroc": safe_auc(p_lp["external_lof_score"], b_lb["external_lof_score"]),
            "mannwhitney_p_greater": float(
                mannwhitneyu(
                    p_lp["external_lof_score"].dropna().astype(float),
                    b_lb["external_lof_score"].dropna().astype(float),
                    alternative="greater",
                ).pvalue
            )
            if len(p_lp["external_lof_score"].dropna()) >= 2 and len(b_lb["external_lof_score"].dropna()) >= 2
            else float("nan"),
        }
    )
    odds, pval = fisher_binary(plof["binary_depleted"], tolerated["binary_depleted"], "greater")
    tests.append(
        {
            "test": "plof_consequences_enriched_for_depleted_vs_synonymous_utr",
            "first_group": "predicted_plof_consequences",
            "second_group": "synonymous_or_utr_controls",
            "first_n_binary": int(plof["binary_depleted"].notna().sum()),
            "first_n_depleted": int(plof["binary_depleted"].eq(1).sum()),
            "second_n_binary": int(tolerated["binary_depleted"].notna().sum()),
            "second_n_depleted": int(tolerated["binary_depleted"].eq(1).sum()),
            "fisher_oddsratio": odds,
            "fisher_p": pval,
            "external_lof_auroc": safe_auc(plof["external_lof_score"], tolerated["external_lof_score"]),
            "mannwhitney_p_greater": float(
                mannwhitneyu(
                    plof["external_lof_score"].dropna().astype(float),
                    tolerated["external_lof_score"].dropna().astype(float),
                    alternative="greater",
                ).pvalue
            )
            if len(plof["external_lof_score"].dropna()) >= 2 and len(tolerated["external_lof_score"].dropna()) >= 2
            else float("nan"),
        }
    )
    odds, pval = fisher_binary(gnomad["binary_depleted"], not_gnomad["binary_depleted"], "less")
    tests.append(
        {
            "test": "gnomad_observed_depleted_less_than_unobserved",
            "first_group": "gnomad_observed",
            "second_group": "gnomad_unobserved",
            "first_n_binary": int(gnomad["binary_depleted"].notna().sum()),
            "first_n_depleted": int(gnomad["binary_depleted"].eq(1).sum()),
            "second_n_binary": int(not_gnomad["binary_depleted"].notna().sum()),
            "second_n_depleted": int(not_gnomad["binary_depleted"].eq(1).sum()),
            "fisher_oddsratio": odds,
            "fisher_p": pval,
            "external_lof_auroc": safe_auc(not_gnomad["external_lof_score"], gnomad["external_lof_score"]),
            "mannwhitney_p_greater": float(
                mannwhitneyu(
                    not_gnomad["external_lof_score"].dropna().astype(float),
                    gnomad["external_lof_score"].dropna().astype(float),
                    alternative="greater",
                ).pvalue
            )
            if len(not_gnomad["external_lof_score"].dropna()) >= 2 and len(gnomad["external_lof_score"].dropna()) >= 2
            else float("nan"),
        }
    )
    tests_df = pd.DataFrame(tests)

    vus_candidates = vus.copy()
    vus_candidates = vus_candidates[
        vus_candidates["functional_classification"].isin(["depleted", "enriched"])
        | (
            vus_candidates["is_in_gnomad"].astype(str).str.upper().ne("Y")
            & vus_candidates["binary_depleted"].eq(1)
        )
    ].copy()
    vus_candidates = vus_candidates.sort_values(
        ["functional_classification", "external_lof_score"],
        ascending=[True, False],
    )
    candidate_cols = [
        "id",
        "HGVSc",
        "HGVSp",
        "consequence",
        "functional_classification",
        "function_score",
        "external_lof_score",
        "is_in_gnomad",
        "gnomad_af_v3_numeric",
        "clinvar_clinical_significance",
        "variation_id",
        "domains",
    ]
    vus_candidates = vus_candidates[[col for col in candidate_cols if col in vus_candidates.columns]].head(200)

    consequence = (
        df.groupby(["consequence", "functional_classification"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["depleted", "unchanged", "enriched"]:
        if col not in consequence.columns:
            consequence[col] = 0
    consequence["binary_n"] = consequence["depleted"] + consequence["unchanged"]
    consequence["depleted_rate_binary"] = consequence["depleted"] / consequence["binary_n"].replace(0, np.nan)
    consequence = consequence.sort_values(["depleted_rate_binary", "binary_n"], ascending=[False, False])
    return group_summary, tests_df, vus_candidates, consequence


def write_report(
    out_dir: Path,
    prefix: str,
    group_summary: pd.DataFrame,
    tests: pd.DataFrame,
    vus_candidates: pd.DataFrame,
    consequence: pd.DataFrame,
) -> None:
    clinvar_test = tests[tests["test"].eq("clinvar_pathogenic_enriched_for_depleted_vs_benign")].iloc[0]
    gnomad_test = tests[tests["test"].eq("gnomad_observed_depleted_less_than_unobserved")].iloc[0]
    plof_test = tests[tests["test"].eq("plof_consequences_enriched_for_depleted_vs_synonymous_utr")].iloc[0]
    lines = [
        "# BAP1 SGE External Clinical And Population Anchors",
        "",
        "## Purpose",
        "",
        "This analysis checks that the BAP1 SGE third-gene benchmark has the clinical, population, and consequence structure expected from a credible variant-effect functional map.",
        "",
        "## Main Results",
        "",
        f"- ClinVar P/LP rows are depleted more often than ClinVar B/LB rows: {int(clinvar_test['first_n_depleted'])}/{int(clinvar_test['first_n_binary'])} vs {int(clinvar_test['second_n_depleted'])}/{int(clinvar_test['second_n_binary'])}; Fisher p={fmt(clinvar_test['fisher_p'], sci=True)}; external-LOF AUROC={fmt(clinvar_test['external_lof_auroc'], 4)}.",
        f"- Predicted protein-truncating/splice-disrupting consequences are depleted more often than synonymous/UTR controls: {int(plof_test['first_n_depleted'])}/{int(plof_test['first_n_binary'])} vs {int(plof_test['second_n_depleted'])}/{int(plof_test['second_n_binary'])}; Fisher p={fmt(plof_test['fisher_p'], sci=True)}.",
        f"- gnomAD-observed variants are depleted less often than unobserved variants: {int(gnomad_test['first_n_depleted'])}/{int(gnomad_test['first_n_binary'])} vs {int(gnomad_test['second_n_depleted'])}/{int(gnomad_test['second_n_binary'])}; Fisher p={fmt(gnomad_test['fisher_p'], sci=True)}.",
        f"- VUS triage candidates exported: {len(vus_candidates)}.",
        "",
        "## Group Summary",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Statistical Tests",
        "",
        tests.to_markdown(index=False),
        "",
        "## Consequence Summary",
        "",
        consequence.head(20).to_markdown(index=False),
        "",
        "## Claim Boundary",
        "",
        "These results validate BAP1 SGE as a credible third-gene functional-map benchmark. They do not yet show that CrossBioSAE predicts or explains BAP1; that requires the pending Evo2/ESM/CrossBioSAE checkpoint outputs.",
        "",
        "## Outputs",
        "",
        f"- {prefix}_group_summary.csv",
        f"- {prefix}_tests.csv",
        f"- {prefix}_vus_candidates.csv",
        f"- {prefix}_consequence_summary.csv",
    ]
    (out_dir / f"{prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out_dir = repo / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(args.variants, low_memory=False)
    group_summary, tests, vus_candidates, consequence = build_anchors(variants)
    prefix = args.output_prefix
    group_summary.to_csv(out_dir / f"{prefix}_group_summary.csv", index=False)
    tests.to_csv(out_dir / f"{prefix}_tests.csv", index=False)
    vus_candidates.to_csv(out_dir / f"{prefix}_vus_candidates.csv", index=False)
    consequence.to_csv(out_dir / f"{prefix}_consequence_summary.csv", index=False)
    write_report(out_dir, prefix, group_summary, tests, vus_candidates, consequence)
    print(f"wrote {out_dir / (prefix + '.md')}")
    print(f"wrote {out_dir / (prefix + '_tests.csv')}")


if __name__ == "__main__":
    main()
