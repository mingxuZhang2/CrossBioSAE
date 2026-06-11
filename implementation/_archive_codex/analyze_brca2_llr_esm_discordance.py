#!/usr/bin/env python3
"""Analyze BRCA2 protein-vs-DNA checkpoint discordance.

This is an interpretation-oriented checkpoint analysis. It does not replace the
full Evo2 embedding / cross-modal / native-SAE run; it uses the completed Evo2
LLR scalar and ESM missense CV score to identify where protein-side and DNA-side
signals agree or disagree in the independent BRCA2 SGE table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-scores",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_llr_esm_checkpoint_variant_scores.csv",
    )
    parser.add_argument(
        "--esm-scores",
        type=Path,
        default=root / "results" / "interpretability_applications" / "brca2_esm_only_replication_variant_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "interpretability_applications",
    )
    parser.add_argument("--output-prefix", default="brca2_llr_esm_discordance")
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--top-examples", type=int, default=20)
    return parser.parse_args()


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


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a / b)


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy() if max_rows else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def fisher_vs_rest(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    local = df.loc[mask]
    rest = df.loc[~mask]
    if local.empty or rest.empty:
        return float("nan"), float("nan")
    table2 = [
        [int(local["label"].sum()), int((1 - local["label"]).sum())],
        [int(rest["label"].sum()), int((1 - rest["label"]).sum())],
    ]
    odds, p = fisher_exact(table2, alternative="two-sided")
    return float(odds), float(p)


def summarize_categories(missense: pd.DataFrame) -> pd.DataFrame:
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
                "mean_function_score": float(sub["function_score"].mean()),
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
    order = {
        "both_high": 0,
        "protein_high_dna_low": 1,
        "dna_high_protein_low": 2,
        "both_low": 3,
        "other": 4,
    }
    out["_order"] = out["category"].map(order).fillna(99)
    return out.sort_values(["_order", "category"]).drop(columns="_order").reset_index(drop=True)


def summarize_domains(missense: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = ["both_high", "protein_high_dna_low", "dna_high_protein_low", "both_low"]
    for (domain, category), sub in missense[missense["discordance_category"].isin(categories)].groupby(
        ["brca2_domain", "discordance_category"],
        sort=True,
    ):
        rows.append(
            {
                "brca2_domain": domain,
                "category": category,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "mean_function_score": float(sub["function_score"].mean()),
                "mean_dna_percentile": float(sub["dna_percentile"].mean()),
                "mean_protein_percentile": float(sub["protein_percentile"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "n"], ascending=[True, False]).reset_index(drop=True)


def summarize_hotspots(missense: pd.DataFrame) -> pd.DataFrame:
    keep = missense[missense["discordance_category"].ne("other") & missense["aa_pos"].notna()].copy()
    rows = []
    for (category, aa_pos, domain), sub in keep.groupby(["discordance_category", "aa_pos", "brca2_domain"], sort=False):
        if len(sub) < 2:
            continue
        rows.append(
            {
                "category": category,
                "aa_pos": int(aa_pos),
                "brca2_domain": domain,
                "n_variants": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "aa_changes": ";".join(sorted(sub["AA.change"].dropna().astype(str).unique())[:8]),
                "mean_function_score": float(sub["function_score"].mean()),
                "mean_dna_score": float(sub["dna_score"].mean()),
                "mean_protein_score": float(sub["protein_score"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["category", "n_lof", "n_variants", "lof_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def example_rows(missense: pd.DataFrame, top_n: int) -> pd.DataFrame:
    pieces = []
    specs = [
        ("both_high", ["label", "protein_percentile", "dna_percentile"]),
        ("protein_high_dna_low", ["label", "protein_percentile", "dna_percentile"]),
        ("dna_high_protein_low", ["label", "dna_percentile", "protein_percentile"]),
        ("both_low", ["label", "function_score"]),
    ]
    cols = [
        "discordance_category",
        "id",
        "AA.change",
        "brca2_domain",
        "label",
        "func_class",
        "function_score",
        "dna_score",
        "protein_score",
        "dna_percentile",
        "protein_percentile",
        "esm_delta_max_abs",
        "clinvar_simple",
    ]
    for category, sort_cols in specs:
        sub = missense[missense["discordance_category"].eq(category)].copy()
        if sub.empty:
            continue
        ascending = [False] * len(sort_cols)
        if "function_score" in sort_cols:
            ascending = [True if c == "function_score" else False for c in sort_cols]
        pieces.append(sub.sort_values(sort_cols, ascending=ascending).head(top_n)[cols])
    if not pieces:
        return pd.DataFrame(columns=cols)
    return pd.concat(pieces, ignore_index=True)


def nonmissense_dna_summary(df: pd.DataFrame, q: float) -> pd.DataFrame:
    non = df[~df["is_missense"].astype(bool) & np.isfinite(df["dna_score"])].copy()
    if non.empty:
        return pd.DataFrame()
    non["dna_percentile_nonmissense"] = non["dna_score"].rank(pct=True, method="average")
    non["dna_bin"] = np.select(
        [
            non["dna_percentile_nonmissense"].ge(1.0 - q),
            non["dna_percentile_nonmissense"].le(q),
        ],
        ["top_dna_quintile", "bottom_dna_quintile"],
        default="middle",
    )
    rows = []
    for (consequence, dna_bin), sub in non.groupby(["consequence", "dna_bin"], sort=True):
        rows.append(
            {
                "consequence": consequence,
                "dna_bin": dna_bin,
                "n": int(len(sub)),
                "n_lof": int(sub["label"].sum()),
                "lof_rate": float(sub["label"].mean()),
                "mean_function_score": float(sub["function_score"].mean()),
                "mean_dna_score": float(sub["dna_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dna_bin", "consequence"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be between 0 and 0.5")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = pd.read_csv(args.checkpoint_scores)
    esm = pd.read_csv(args.esm_scores)
    esm_cols = ["id", "esm_only_missense_cv_pred", "esm_delta_max_abs", "esm_delta_l2"]
    missing = [col for col in esm_cols if col not in esm.columns]
    if missing:
        raise KeyError(f"missing columns in ESM scores: {missing}")
    df = checkpoint.merge(esm[esm_cols], on="id", how="left", validate="one_to_one")
    df["dna_score"] = df["evo2_llr_zero_shot"].astype(float)
    df["protein_score"] = df["esm_only_missense_cv_pred"].astype(float)

    missense = df[df["is_missense"].astype(bool) & np.isfinite(df["dna_score"]) & np.isfinite(df["protein_score"])].copy()
    if missense.empty:
        raise RuntimeError("no finite missense rows available")
    missense["dna_percentile"] = missense["dna_score"].rank(pct=True, method="average")
    missense["protein_percentile"] = missense["protein_score"].rank(pct=True, method="average")
    lo = args.quantile
    hi = 1.0 - args.quantile
    both_high = missense["dna_percentile"].ge(hi) & missense["protein_percentile"].ge(hi)
    protein_high_dna_low = missense["protein_percentile"].ge(hi) & missense["dna_percentile"].le(lo)
    dna_high_protein_low = missense["dna_percentile"].ge(hi) & missense["protein_percentile"].le(lo)
    both_low = missense["dna_percentile"].le(lo) & missense["protein_percentile"].le(lo)
    missense["discordance_category"] = np.select(
        [both_high, protein_high_dna_low, dna_high_protein_low, both_low],
        ["both_high", "protein_high_dna_low", "dna_high_protein_low", "both_low"],
        default="other",
    )

    category_summary = summarize_categories(missense)
    domain_summary = summarize_domains(missense)
    hotspots = summarize_hotspots(missense)
    examples = example_rows(missense, args.top_examples)
    nonmissense = nonmissense_dna_summary(df.assign(dna_score=df["dna_score"]), args.quantile)

    prefix = args.output_prefix
    category_summary.to_csv(out / f"{prefix}_category_summary.csv", index=False)
    domain_summary.to_csv(out / f"{prefix}_domain_summary.csv", index=False)
    hotspots.to_csv(out / f"{prefix}_residue_hotspots.csv", index=False)
    examples.to_csv(out / f"{prefix}_examples.csv", index=False)
    nonmissense.to_csv(out / f"{prefix}_nonmissense_dna_summary.csv", index=False)
    missense.to_csv(out / f"{prefix}_missense_scores.csv", index=False)

    report = [
        "# BRCA2 ESM-vs-Evo2 LLR Discordance",
        "",
        "## Purpose",
        "",
        "Use the completed BRCA2 checkpoint to separate missense variants where protein-side ESM and DNA-side Evo2 LLR signals agree from variants where they disagree. This is a pre-native-SAE mechanism map, not the final BRCA2 cross-modal proof.",
        "",
        "## Definitions",
        "",
        f"- Scope: finite BRCA2 missense variants, n={len(missense):,}.",
        f"- DNA score: Evo2 LLR zero-shot pathogenicity score, higher means more SGE-LOF-like.",
        f"- Protein score: dedicated BRCA2 missense-only ESM CV prediction, higher means more SGE-LOF-like.",
        f"- High and low groups use the top/bottom {args.quantile:.0%} score quantiles within finite missense rows.",
        "",
        "## Category Summary",
        "",
        table(category_summary),
        "",
        "## Domain Summary",
        "",
        table(domain_summary),
        "",
        "## Recurrent Residue Hotspots",
        "",
        table(hotspots, max_rows=40),
        "",
        "## Example Variants",
        "",
        table(examples, max_rows=80),
        "",
        "## Non-Missense DNA-Side Check",
        "",
        table(nonmissense),
        "",
        "## Interpretation",
        "",
        "- Concordant high missense variants are the strongest current BRCA2 mechanism candidates before full Evo2 embeddings finish; they have much higher SGE-LOF rate than the missense background.",
        "- Protein-high/DNA-low variants identify protein-side mechanisms that a scalar Evo2 LLR under-ranks; these are natural candidates for later cross-modal discordance and native-SAE inspection.",
        "- DNA-high/protein-low variants identify variants where local nucleotide context or Evo2 DNA signal is strong despite weak ESM missense prediction; these should be treated as a separate mechanism channel rather than folded into a generic fusion-gain claim.",
        "- The current checkpoint supports mechanism stratification, not clinical interpretation and not a final BRCA2 SAE replication. The final test remains the dependent BRCA2 downstream and native-SAE jobs.",
        "",
    ]
    (out / f"{prefix}.md").write_text("\n".join(report), encoding="utf-8")
    print(category_summary.to_string(index=False))


if __name__ == "__main__":
    main()
