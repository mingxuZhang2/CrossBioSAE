#!/usr/bin/env python3
"""Summarize BRCA2 native-SAE original-style stratum control runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="brca2_native_sae_stratum_control_pilot")
    return parser.parse_args()


def table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    widths = [max(len(str(c)), *(len(str(v)) for v in show[c])) for c in show.columns]
    lines = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(show.columns, widths)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = args.repo_root.resolve() / OUT_DIR
    prefix = args.prefix
    summary_path = out / f"{prefix}_summary.csv"
    folds_path = out / f"{prefix}_folds.csv"
    controls_path = out / f"{prefix}_controls.csv"
    if not summary_path.exists() or not folds_path.exists() or not controls_path.exists():
        raise FileNotFoundError(f"missing {prefix} CSV outputs in {out}")

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(folds_path)
    controls = pd.read_csv(controls_path)
    tested = folds[folds["status"].eq("tested")].copy()

    lines = [
        "# BRCA2 Native-SAE Original-Style Stratum Control Pilot",
        "",
        "## Purpose",
        "",
        "The fixed-prediction focused-strata screen is descriptive. This pilot reruns the stricter native-SAE control logic inside candidate strata: per-fold SAE training, train-subset feature selection, targeted ablation on the stratum test subset, matched random features, and label-permuted feature selection.",
        "",
        "This pilot used reduced compute settings and is not a final publication-grade rerun. It is meant to decide whether any candidate stratum is worth a full GPU rerun.",
        "",
        "## Summary",
        "",
        table(
            summary,
            [
                "candidate",
                "n_total",
                "n_tested_folds",
                "mean_fold_delta_auc_recon_minus_top_ablate",
                "mean_random_delta_auc",
                "empirical_p_random_delta_ge_top_mean",
                "mean_label_permuted_delta_auc",
                "empirical_p_label_permuted_delta_ge_top_mean",
                "passes_original_style_gate",
            ],
        ),
        "",
        "## Fold Details",
        "",
        table(
            folds,
            [
                "candidate",
                "fold",
                "status",
                "n_train",
                "n_train_lof",
                "n_test",
                "n_test_lof",
                "native_sae_recon_auc",
                "top_feature_ablate_auc",
                "delta_auc_recon_minus_top_ablate",
            ],
            max_rows=30,
        ),
        "",
        "## Interpretation",
        "",
    ]

    passed = summary[summary["passes_original_style_gate"].astype(str).str.lower().eq("true")]
    if passed.empty:
        lines.extend(
            [
                "- No tested candidate passes the original-style gate in this pilot.",
                "- `protein_high_dna_low` has negative mean selected-ablation delta and is worse than random controls.",
                "- `ob1_both_high` is too small/label-imbalanced for stable fold-level AUROC in this setup.",
                "- `tier3_vus_conflicting` has a small positive AUROC delta and beats random controls in the pilot, but it does not beat label-permuted feature selection and has negative mean AUPRC delta.",
                "- Therefore this pilot reinforces the current manuscript stance: BRCA2 native-SAE should stay in error analysis, while BRCA2's main value remains mechanism stratification and VUS/assay triage.",
            ]
        )
    else:
        lines.append("- At least one candidate passed the pilot gate; rerun it with full GPU settings before upgrading the manuscript claim.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{summary_path.name}`",
            f"- `{folds_path.name}`",
            f"- `{controls_path.name}`",
            f"- `{prefix}_selected_features.csv`",
            "",
            f"Tested fold rows: {len(tested)}; control rows: {len(controls)}.",
            "",
        ]
    )

    report_path = out / f"{prefix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
