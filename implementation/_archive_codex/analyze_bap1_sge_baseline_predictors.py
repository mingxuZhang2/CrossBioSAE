#!/usr/bin/env python3
"""Evaluate public missense-effect baselines on BAP1 SGE labels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--variants", type=Path, default=repo_root / OUT_DIR / "bap1_sge_variants.csv")
    parser.add_argument("--output-prefix", default="bap1_sge_baseline_predictors")
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


def metric_row(df: pd.DataFrame, score_col: str, label: str) -> dict[str, Any]:
    y = df["binary_depleted"].astype(int)
    score = pd.to_numeric(df[score_col], errors="coerce")
    ok = score.notna()
    yy = y[ok].to_numpy()
    ss = score[ok].astype(float).to_numpy()
    if len(ss) < 20 or len(np.unique(yy)) < 2 or len(np.unique(ss)) < 2:
        auroc = auprc = pval = float("nan")
    else:
        auroc = float(roc_auc_score(yy, ss))
        auprc = float(average_precision_score(yy, ss))
        pval = float(mannwhitneyu(ss[yy == 1], ss[yy == 0], alternative="greater").pvalue)
    return {
        "baseline": label,
        "score_col": score_col,
        "n": int(ok.sum()),
        "n_depleted": int(y[ok].sum()),
        "auroc_for_sge_depleted": auroc,
        "auprc_for_sge_depleted": auprc,
        "mannwhitney_p_greater": pval,
    }


def binary_rule_row(df: pd.DataFrame, pred_col: str, label: str) -> dict[str, Any]:
    sub = df[df[pred_col].notna()].copy()
    pred = sub[pred_col].astype(int)
    y = sub["binary_depleted"].astype(int)
    if len(sub) == 0:
        odds = pval = float("nan")
    else:
        table = [
            [int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())],
            [int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())],
        ]
        odds, pval = fisher_exact(table, alternative="greater")
    return {
        "baseline": label,
        "n": int(len(sub)),
        "n_pred_positive": int(pred.eq(1).sum()) if len(sub) else 0,
        "n_pred_negative": int(pred.eq(0).sum()) if len(sub) else 0,
        "n_depleted_among_pred_positive": int(((pred == 1) & (y == 1)).sum()) if len(sub) else 0,
        "n_unchanged_among_pred_positive": int(((pred == 1) & (y == 0)).sum()) if len(sub) else 0,
        "n_depleted_among_pred_negative": int(((pred == 0) & (y == 1)).sum()) if len(sub) else 0,
        "n_unchanged_among_pred_negative": int(((pred == 0) & (y == 0)).sum()) if len(sub) else 0,
        "fisher_oddsratio": float(odds),
        "fisher_p_greater": float(pval),
    }


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["binary_depleted"] = binary_label(data)
    data = data[data["binary_depleted"].notna() & data["is_missense"].astype(str).str.lower().eq("true")].copy()
    data["EVE_score_numeric"] = data["EVE_score"].map(as_float)
    data["SIFT_deleterious_score"] = -data["SIFT"].map(as_float)
    poly_map = {"benign": 0.0, "possibly_damaging": 0.5, "probably_damaging": 1.0}
    data["PolyPhen_damage_score"] = data["PolyPhen"].astype(str).map(poly_map)
    data["SGE_external_lof_score"] = -pd.to_numeric(data["function_score"], errors="coerce")
    data["EVE_pathogenic_call"] = data["EVE_class"].astype(str).map({"Pathogenic": 1, "Benign": 0})
    data["PolyPhen_damaging_call"] = data["PolyPhen"].astype(str).map(
        {"probably_damaging": 1, "possibly_damaging": 1, "benign": 0}
    )
    data["SIFT_deleterious_call"] = data["SIFT"].map(as_float).map(lambda x: np.nan if pd.isna(x) else int(x <= 0.05))

    metrics = pd.DataFrame(
        [
            metric_row(data, "EVE_score_numeric", "EVE score"),
            metric_row(data, "SIFT_deleterious_score", "SIFT deleterious (-score)"),
            metric_row(data, "PolyPhen_damage_score", "PolyPhen ordinal damage"),
            metric_row(data, "SGE_external_lof_score", "BAP1 SGE functional score sanity"),
        ]
    )
    calls = pd.DataFrame(
        [
            binary_rule_row(data, "EVE_pathogenic_call", "EVE Pathogenic vs Benign"),
            binary_rule_row(data, "SIFT_deleterious_call", "SIFT <= 0.05"),
            binary_rule_row(data, "PolyPhen_damaging_call", "PolyPhen damaging"),
        ]
    )
    return metrics, calls


def write_report(out: Path, prefix: str, metrics: pd.DataFrame, calls: pd.DataFrame) -> None:
    top_rows = metrics[metrics["baseline"].ne("BAP1 SGE functional score sanity")].sort_values(
        "auroc_for_sge_depleted",
        ascending=False,
    )
    best = top_rows.iloc[0].to_dict() if not top_rows.empty else {}
    lines = [
        "# BAP1 SGE Public Baseline Predictors",
        "",
        "## Purpose",
        "",
        "This evaluates public missense-effect annotations included in the BAP1 SGE table. These are baseline comparators for the pending CrossBioSAE/Evo2/ESM BAP1 checkpoint.",
        "",
        "## Main Result",
        "",
        f"- Best public missense baseline among EVE/SIFT/PolyPhen: {best.get('baseline', 'NA')} with AUROC {fmt(best.get('auroc_for_sge_depleted'), 4)} and AUPRC {fmt(best.get('auprc_for_sge_depleted'), 4)}.",
        "- These baselines are not CrossBioSAE results; they set the comparator context for the third-gene benchmark.",
        "",
        "## Continuous Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Binary Call Tests",
        "",
        calls.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- {prefix}_metric_summary.csv",
        f"- {prefix}_binary_calls.csv",
    ]
    (out / f"{prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(args.variants, low_memory=False)
    metrics, calls = build(variants)
    metrics.to_csv(out / f"{args.output_prefix}_metric_summary.csv", index=False)
    calls.to_csv(out / f"{args.output_prefix}_binary_calls.csv", index=False)
    write_report(out, args.output_prefix, metrics, calls)
    print(f"wrote {out / (args.output_prefix + '.md')}")
    print(f"wrote {out / (args.output_prefix + '_metric_summary.csv')}")


if __name__ == "__main__":
    main()
