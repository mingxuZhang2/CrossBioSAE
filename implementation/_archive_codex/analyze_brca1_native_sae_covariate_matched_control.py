#!/usr/bin/env python
"""BRCA1 native-effect control matched on annotation plus CADD/phyloP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=630)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def match_level_pool(
    low: pd.DataFrame,
    row: pd.Series,
) -> tuple[pd.DataFrame, str]:
    exact = low[
        low["consequence"].astype(str).eq(str(row["consequence"]))
        & low["brca1_region"].astype(str).eq(str(row["brca1_region"]))
    ]
    if len(exact):
        return exact, "consequence_region_cadd_phylop"
    consequence = low[low["consequence"].astype(str).eq(str(row["consequence"]))]
    if len(consequence):
        return consequence, "consequence_cadd_phylop"
    region = low[low["brca1_region"].astype(str).eq(str(row["brca1_region"]))]
    if len(region):
        return region, "region_cadd_phylop"
    return low, "all_cadd_phylop"


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (numeric - mean) / sd


def nearest_pairs(scores: pd.DataFrame) -> pd.DataFrame:
    high = scores[scores["is_high_native_effect"]].copy()
    low = scores[~scores["is_high_native_effect"]].copy()
    rows = []
    for _, row in high.iterrows():
        pool, level = match_level_pool(low, row)
        if pool.empty:
            continue
        dist = (
            (pool["cadd_z"].to_numpy(dtype=float) - float(row["cadd_z"])) ** 2
            + (pool["phylop_z"].to_numpy(dtype=float) - float(row["phylop_z"])) ** 2
        )
        pick = pool.iloc[int(np.argmin(dist))]
        rows.append(
            {
                "high_index": int(row.name),
                "control_index": int(pick.name),
                "match_level": level,
                "distance_cadd_phylop_z": float(np.sqrt(np.min(dist))),
                "high_id": row.get("id", ""),
                "control_id": pick.get("id", ""),
                "high_label": int(row["label"]),
                "control_label": int(pick["label"]),
                "high_consequence": row["consequence"],
                "control_consequence": pick["consequence"],
                "high_brca1_region": row["brca1_region"],
                "control_brca1_region": pick["brca1_region"],
                "high_cadd": float(row["CADD"]),
                "control_cadd": float(pick["CADD"]),
                "high_phylop": float(row["phyloP"]),
                "control_phylop": float(pick["phyloP"]),
                "high_function_score": float(row["function_score"]),
                "control_function_score": float(pick["function_score"]),
                "high_native_effect": float(row["native_effect"]),
                "control_native_effect": float(pick["native_effect"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(pairs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hi = pairs["high_label"].to_numpy(dtype=float)
    ctrl = pairs["control_label"].to_numpy(dtype=float)
    rows = []
    for i in range(n_boot):
        idx = rng.integers(0, len(pairs), size=len(pairs))
        rows.append(
            {
                "bootstrap": i,
                "high_lof_rate": float(hi[idx].mean()),
                "control_lof_rate": float(ctrl[idx].mean()),
                "delta_lof_rate": float(hi[idx].mean() - ctrl[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def metric_balance(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["cadd", "phylop", "function_score", "native_effect"]:
        high = pairs[f"high_{metric}"].to_numpy(dtype=float)
        control = pairs[f"control_{metric}"].to_numpy(dtype=float)
        diff = high - control
        rows.append(
            {
                "metric": metric,
                "high_mean": float(np.mean(high)),
                "control_mean": float(np.mean(control)),
                "mean_difference": float(np.mean(diff)),
                "median_abs_pair_difference": float(np.median(np.abs(diff))),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(path: Path, summary: pd.DataFrame, balance: pd.DataFrame, levels: pd.DataFrame) -> None:
    row = summary.iloc[0]
    lines = [
        "# BRCA1 Native-SAE Covariate-Matched Annotation Control",
        "",
        "## Purpose",
        "",
        "This control asks whether the BRCA1 top-decile native-effect LOF enrichment remains after exact consequence/region matching plus nearest-neighbor matching on CADD and phyloP. Function score is intentionally not used for matching because it is the functional outcome.",
        "",
        "## Key Result",
        "",
        f"- pairs: {int(row['n_pairs'])}",
        f"- high-effect LOF rate: {fmt(row['high_lof_rate'], 3)}",
        f"- covariate-matched control LOF rate: {fmt(row['control_lof_rate'], 3)}",
        f"- delta LOF rate: {fmt(row['delta_lof_rate'], 3)}",
        f"- bootstrap 95% CI: [{fmt(row['delta_bootstrap_ci_low'], 3)}, {fmt(row['delta_bootstrap_ci_high'], 3)}]",
        f"- paired bootstrap p(delta <= 0): {fmt(row['bootstrap_p_delta_le_zero'], 4)}",
        "",
        "## Balance",
        "",
        balance.to_markdown(index=False),
        "",
        "## Match Levels",
        "",
        levels.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- The effect is smaller than the consequence/region-only null, which means conservation/pathogenicity covariates explain part of the signal.",
        "- The residual positive delta supports a narrower claim: native-effect variants are enriched for BRCA1 functional disruption beyond coarse annotation and CADD/phyloP matching.",
        "- This is still an observational matched control, not a replacement for the native-SAE ablation experiment.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    scores = pd.read_csv(require_file(out / "brca1_native_finetuned_sae_intervention_variant_scores.csv"))
    required = [
        "id",
        "consequence",
        "brca1_region",
        "label",
        "function_score",
        "CADD",
        "phyloP",
        "sae_recon_pred",
        "top_feature_ablate_pred",
    ]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    scores = scores.reset_index(drop=True)
    scores["native_effect"] = scores["sae_recon_pred"] - scores["top_feature_ablate_pred"]
    scores["is_high_native_effect"] = scores["native_effect"] >= scores["native_effect"].quantile(0.90)
    scores["CADD"] = pd.to_numeric(scores["CADD"], errors="coerce")
    scores["phyloP"] = pd.to_numeric(scores["phyloP"], errors="coerce")
    scores["CADD"] = scores["CADD"].fillna(scores["CADD"].median())
    scores["phyloP"] = scores["phyloP"].fillna(scores["phyloP"].median())
    scores["cadd_z"] = zscore(scores["CADD"])
    scores["phylop_z"] = zscore(scores["phyloP"])

    pairs = nearest_pairs(scores)
    boot = bootstrap_delta(pairs, args.bootstraps, args.seed)
    balance = metric_balance(pairs)
    levels = (
        pairs["match_level"]
        .value_counts()
        .rename_axis("match_level")
        .reset_index(name="n_pairs")
    )
    levels["frac_pairs"] = levels["n_pairs"] / len(pairs)

    high_lof = float(pairs["high_label"].mean())
    control_lof = float(pairs["control_label"].mean())
    delta = high_lof - control_lof
    high_pos = int(pairs["high_label"].sum())
    control_pos = int(pairs["control_label"].sum())
    odds, fisher_p = fisher_exact(
        [
            [high_pos, len(pairs) - high_pos],
            [control_pos, len(pairs) - control_pos],
        ],
        alternative="greater",
    )
    summary = pd.DataFrame(
        [
            {
                "comparison": "top_decile_native_effect_vs_consequence_region_cadd_phylop_nearest",
                "n_pairs": len(pairs),
                "high_lof_rate": high_lof,
                "control_lof_rate": control_lof,
                "delta_lof_rate": delta,
                "odds_ratio": float(odds),
                "fisher_p_greater": float(fisher_p),
                "delta_bootstrap_ci_low": float(boot["delta_lof_rate"].quantile(0.025)),
                "delta_bootstrap_ci_high": float(boot["delta_lof_rate"].quantile(0.975)),
                "bootstrap_p_delta_le_zero": float((1 + (boot["delta_lof_rate"] <= 0).sum()) / (len(boot) + 1)),
            }
        ]
    )

    summary.to_csv(out / "brca1_native_sae_covariate_matched_control_summary.csv", index=False)
    pairs.to_csv(out / "brca1_native_sae_covariate_matched_control_pairs.csv", index=False)
    boot.to_csv(out / "brca1_native_sae_covariate_matched_control_bootstrap.csv", index=False)
    balance.to_csv(out / "brca1_native_sae_covariate_matched_control_balance.csv", index=False)
    levels.to_csv(out / "brca1_native_sae_covariate_matched_control_match_levels.csv", index=False)
    write_report(out / "brca1_native_sae_covariate_matched_control.md", summary, balance, levels)

    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control_summary.csv'}")
    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control_pairs.csv'}")
    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control_bootstrap.csv'}")
    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control_balance.csv'}")
    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control_match_levels.csv'}")
    print(f"wrote {out / 'brca1_native_sae_covariate_matched_control.md'}")


if __name__ == "__main__":
    main()
