#!/usr/bin/env python3
"""Generic DNA-vs-protein checkpoint discordance analysis.

This script turns the variant-level output from run_brca2_llr_esm_checkpoint.py
into an interpretation-oriented score map. It is intentionally generic so the
same evidence ladder can be applied to BAP1, BRCA2, or future SGE genes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-scores", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=root / "results" / "interpretability_applications")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--gene-name", default="gene")
    p.add_argument("--quantile", type=float, default=0.20)
    p.add_argument("--domain-col", default="")
    p.add_argument("--top-examples", type=int, default=40)
    return p.parse_args()


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    values = pvals.to_numpy(dtype=float)
    finite = np.isfinite(values)
    q = np.full(len(values), np.nan, dtype=float)
    if not finite.any():
        return pd.Series(q, index=pvals.index)
    idx = np.where(finite)[0]
    order = idx[np.argsort(values[finite])]
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=pvals.index)


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    try:
        return show.to_markdown(index=False)
    except ImportError:
        return show.to_csv(index=False)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def fisher_vs_rest(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    local = df.loc[mask]
    rest = df.loc[~mask]
    if local.empty or rest.empty:
        return float("nan"), float("nan")
    tab = [
        [int(local["label"].sum()), int((1 - local["label"]).sum())],
        [int(rest["label"].sum()), int((1 - rest["label"]).sum())],
    ]
    odds, p = fisher_exact(tab, alternative="two-sided")
    return float(odds), float(p)


def clean_string_col(df: pd.DataFrame, col: str, default: str = "unavailable") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    out = df[col].fillna(default).astype(str)
    return out.mask(out.str.strip().eq("") | out.str.lower().eq("nan"), default)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    required = ["evo2_llr_zero_shot", "esm_cv_pred", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing required checkpoint columns: {missing}")
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out[out["label"].isin([0, 1])].copy()
    out["label"] = out["label"].astype(int)
    out["dna_score"] = pd.to_numeric(out["evo2_llr_zero_shot"], errors="coerce")
    out["protein_score"] = pd.to_numeric(out["esm_cv_pred"], errors="coerce")
    if "function_score" in out.columns:
        out["function_score"] = pd.to_numeric(out["function_score"], errors="coerce")
    return out.reset_index(drop=True)


def category_summary(missense: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_lof = float(missense["label"].mean())
    for category, sub in missense.groupby("discordance_category", sort=False):
        mask = missense["discordance_category"].eq(category)
        odds, p = fisher_vs_rest(missense, mask)
        rows.append(
            {
                "category": category,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "lof_rate_vs_all_missense": safe_ratio(float(sub["label"].mean()), global_lof),
                "mean_function_score": float(sub["function_score"].mean())
                if "function_score" in sub.columns
                else float("nan"),
                "mean_dna_score": float(sub["dna_score"].mean()),
                "mean_protein_score": float(sub["protein_score"].mean()),
                "mean_dna_percentile": float(sub["dna_percentile"].mean()),
                "mean_protein_percentile": float(sub["protein_percentile"].mean()),
                "fisher_odds_vs_rest": odds,
                "fisher_p_vs_rest": p,
            }
        )
    out = pd.DataFrame(rows)
    out["fisher_q_vs_rest"] = bh_qvalues(out["fisher_p_vs_rest"])
    order = {"both_high": 0, "protein_high_dna_low": 1, "dna_high_protein_low": 2, "both_low": 3, "other": 4}
    out["_order"] = out["category"].map(order).fillna(99)
    return out.sort_values(["_order", "category"]).drop(columns="_order").reset_index(drop=True)


def grouped_summary(missense: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in missense.columns:
        return pd.DataFrame()
    rows = []
    cats = ["both_high", "protein_high_dna_low", "dna_high_protein_low", "both_low"]
    keep = missense[missense["discordance_category"].isin(cats)].copy()
    for (group, category), sub in keep.groupby([group_col, "discordance_category"], sort=True):
        rows.append(
            {
                group_col: group,
                "category": category,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "mean_function_score": float(sub["function_score"].mean())
                if "function_score" in sub.columns
                else float("nan"),
                "mean_dna_percentile": float(sub["dna_percentile"].mean()),
                "mean_protein_percentile": float(sub["protein_percentile"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["category", "n"], ascending=[True, False]).reset_index(drop=True)


def anchor_summary(missense: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, pd.Series]] = []
    if "clinvar_clinical_significance_slim" in missense.columns:
        clin = clean_string_col(missense, "clinvar_clinical_significance_slim")
        groups.extend(
            [
                ("clinvar_pathogenic_likely_pathogenic", clin.str.contains("Pathogenic", case=False, na=False)),
                ("clinvar_benign_likely_benign", clin.str.contains("Benign", case=False, na=False)),
                ("clinvar_vus", clin.str.contains("Uncertain", case=False, na=False)),
            ]
        )
    elif "clinvar_simple" in missense.columns:
        clin = clean_string_col(missense, "clinvar_simple")
        groups.extend(
            [
                ("clinvar_pathogenic_likely_pathogenic", clin.str.contains("Pathogenic", case=False, na=False)),
                ("clinvar_benign_likely_benign", clin.str.contains("Benign", case=False, na=False)),
                ("clinvar_vus", clin.str.contains("Uncertain", case=False, na=False)),
            ]
        )
    if "is_in_gnomad" in missense.columns:
        gnomad = clean_string_col(missense, "is_in_gnomad").str.upper()
        groups.extend([("gnomad_observed", gnomad.eq("Y")), ("gnomad_unobserved", ~gnomad.eq("Y"))])
    rows = []
    for group, mask in groups:
        sub = missense.loc[mask]
        if sub.empty:
            continue
        for category, cat_sub in sub.groupby("discordance_category", sort=True):
            rows.append(
                {
                    "anchor_group": group,
                    "category": category,
                    "n": int(len(cat_sub)),
                    "n_lof": int(cat_sub["label"].sum()),
                    "lof_rate": float(cat_sub["label"].mean()),
                    "mean_dna_percentile": float(cat_sub["dna_percentile"].mean()),
                    "mean_protein_percentile": float(cat_sub["protein_percentile"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["anchor_group", "category"]).reset_index(drop=True) if rows else pd.DataFrame()


def example_rows(missense: pd.DataFrame, n: int, domain_col: str) -> pd.DataFrame:
    cols = [
        "discordance_category",
        "id",
        "label",
        "functional_classification",
        "func_class",
        "function_score",
        "consequence",
        "HGVSp",
        "AA.change",
        "aa_pos",
        domain_col,
        "dna_score",
        "protein_score",
        "dna_percentile",
        "protein_percentile",
        "clinvar_clinical_significance_slim",
        "clinvar_simple",
        "is_in_gnomad",
        "dbSNP.ID",
    ]
    cols = [col for col in cols if col and col in missense.columns]
    pieces = []
    specs = [
        ("both_high", ["label", "protein_percentile", "dna_percentile"]),
        ("protein_high_dna_low", ["label", "protein_percentile", "dna_percentile"]),
        ("dna_high_protein_low", ["label", "dna_percentile", "protein_percentile"]),
        ("both_low", ["label", "function_score" if "function_score" in missense.columns else "dna_percentile"]),
    ]
    for category, sort_cols in specs:
        sub = missense[missense["discordance_category"].eq(category)].copy()
        if sub.empty:
            continue
        sort_cols = [col for col in sort_cols if col in sub.columns]
        ascending = [False] * len(sort_cols)
        if "function_score" in sort_cols:
            ascending = [col == "function_score" for col in sort_cols]
        pieces.append(sub.sort_values(sort_cols, ascending=ascending).head(n)[cols])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=cols)


def nonmissense_summary(df: pd.DataFrame, q: float) -> pd.DataFrame:
    non = df[(~df["is_missense"].astype(bool)) & np.isfinite(df["dna_score"])].copy()
    if non.empty:
        return pd.DataFrame()
    non["dna_percentile_nonmissense"] = non["dna_score"].rank(pct=True, method="average")
    non["dna_bin"] = np.select(
        [non["dna_percentile_nonmissense"].ge(1.0 - q), non["dna_percentile_nonmissense"].le(q)],
        ["top_dna_quantile", "bottom_dna_quantile"],
        default="middle",
    )
    rows = []
    group_cols = ["consequence", "dna_bin"] if "consequence" in non.columns else ["dna_bin"]
    for key, sub in non.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "mean_dna_score": float(sub["dna_score"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be between 0 and 0.5")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = add_scores(pd.read_csv(args.checkpoint_scores))
    if "is_missense" not in df.columns:
        raise KeyError("checkpoint scores must include is_missense")
    missense = df[df["is_missense"].astype(bool) & np.isfinite(df["dna_score"]) & np.isfinite(df["protein_score"])].copy()
    if missense.empty:
        raise RuntimeError("no finite missense rows available")
    missense["dna_percentile"] = missense["dna_score"].rank(pct=True, method="average")
    missense["protein_percentile"] = missense["protein_score"].rank(pct=True, method="average")
    lo = args.quantile
    hi = 1.0 - args.quantile
    missense["discordance_category"] = np.select(
        [
            missense["dna_percentile"].ge(hi) & missense["protein_percentile"].ge(hi),
            missense["protein_percentile"].ge(hi) & missense["dna_percentile"].le(lo),
            missense["dna_percentile"].ge(hi) & missense["protein_percentile"].le(lo),
            missense["dna_percentile"].le(lo) & missense["protein_percentile"].le(lo),
        ],
        ["both_high", "protein_high_dna_low", "dna_high_protein_low", "both_low"],
        default="other",
    )
    df = df.merge(
        missense[["id", "dna_percentile", "protein_percentile", "discordance_category"]],
        on="id",
        how="left",
    )

    domain_col = args.domain_col if args.domain_col else ""
    if not domain_col:
        for candidate in ["domains", "brca2_domain", "consequence"]:
            if candidate in missense.columns:
                domain_col = candidate
                break

    cats = category_summary(missense)
    groups = grouped_summary(missense, domain_col) if domain_col else pd.DataFrame()
    anchors = anchor_summary(missense)
    examples = example_rows(missense, args.top_examples, domain_col)
    nonmissense = nonmissense_summary(df, args.quantile)

    prefix = args.output_prefix
    cats.to_csv(out / f"{prefix}_category_summary.csv", index=False)
    groups.to_csv(out / f"{prefix}_group_summary.csv", index=False)
    anchors.to_csv(out / f"{prefix}_anchor_summary.csv", index=False)
    examples.to_csv(out / f"{prefix}_examples.csv", index=False)
    nonmissense.to_csv(out / f"{prefix}_nonmissense_dna_summary.csv", index=False)
    missense.to_csv(out / f"{prefix}_missense_scores.csv", index=False)

    both_high = cats.loc[cats["category"].eq("both_high")]
    both_low = cats.loc[cats["category"].eq("both_low")]
    headline = "No both-high category was produced."
    if not both_high.empty:
        b = both_high.iloc[0]
        headline = (
            f"Both-high missense variants: n={int(b['n'])}, LOF/depleted rate={b['lof_rate']:.3f}, "
            f"{b['lof_rate_vs_all_missense']:.3f}x missense background, q={b['fisher_q_vs_rest']:.3g}."
        )
    if not both_low.empty:
        low = both_low.iloc[0]
        headline += f" Both-low rate={low['lof_rate']:.3f}."

    report = [
        f"# {args.gene_name} ESM-vs-Evo2 LLR Discordance",
        "",
        "## Purpose",
        "",
        "Map DNA-side Evo2 LLR and protein-side ESM checkpoint scores into concordant and discordant missense strata. This is a mechanism-stratification analysis, not a native-SAE causal intervention.",
        "",
        "## Definitions",
        "",
        f"- Scope: finite binary-label missense rows, n={len(missense):,}.",
        "- DNA score: `evo2_llr_zero_shot`, higher means more LOF/depleted-like.",
        "- Protein score: `esm_cv_pred`, higher means more LOF/depleted-like.",
        f"- High/low groups use top/bottom {args.quantile:.0%} quantiles within finite missense rows.",
        f"- Group column: `{domain_col}`." if domain_col else "- Group column: none available.",
        "",
        "## Main Result",
        "",
        f"- {headline}",
        "",
        "## Category Summary",
        "",
        table(cats),
        "",
        "## Domain Or Consequence Summary",
        "",
        table(groups, max_rows=60),
        "",
        "## Clinical And Population Anchors",
        "",
        table(anchors, max_rows=80),
        "",
        "## Example Variants",
        "",
        table(examples, max_rows=80),
        "",
        "## Non-Missense DNA-Side Check",
        "",
        table(nonmissense, max_rows=80),
        "",
        "## Claim Boundary",
        "",
        "- A positive both-high enrichment supports a mechanism/application stratum.",
        "- It does not prove sparse-feature causality; that requires a native-SAE ablation/rescue experiment with matched random and label-permutation controls.",
        "- If both-high is weak or nonsignificant, this gene should remain a boundary/negative checkpoint or be replaced by a better third-gene SGE route.",
        "",
    ]
    (out / f"{prefix}.md").write_text("\n".join(report), encoding="utf-8")
    print(cats.to_string(index=False))


if __name__ == "__main__":
    main()
