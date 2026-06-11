#!/usr/bin/env python3
"""Focused BRCA2 native-SAE error analysis.

The global BRCA2 native-SAE intervention has a positive targeted-ablation
effect, but it fails the label-permutation gate. This script asks whether a
more precise BRCA2 stratum is worth follow-up. The screen here is intentionally
descriptive: it does not replace the original fold-level label-permuted feature
selection control used by the native-SAE intervention script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


OUT_DIR = Path("results/interpretability_applications")
SCORES = "brca2_native_finetuned_sae_intervention_variant_scores.csv"
DISCORDANCE = "brca2_llr_esm_discordance_missense_scores.csv"
CANDIDATES = "brca2_clinvar_interpretability_candidates.csv"
PANEL = "brca2_prospective_followup_panel.csv"


@dataclass(frozen=True)
class Stratum:
    name: str
    group: str
    value: str
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], score[ok]))


def score_delta(y: np.ndarray, recon: np.ndarray, ablate: np.ndarray) -> float:
    return safe_auroc(y, recon) - safe_auroc(y, ablate)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    sub = df.loc[:, columns].head(max_rows).copy()
    if sub.empty:
        return "_No rows._"
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    widths = [max(len(str(c)), *(len(str(v)) for v in sub[c])) for c in sub.columns]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(sub.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = [
        "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
        for row in sub.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + rows)


def attach_context(repo: Path) -> pd.DataFrame:
    out = repo / OUT_DIR
    scores = read_csv(out / SCORES)

    disc = read_csv(out / DISCORDANCE)
    disc_cols = ["id", "discordance_category", "dna_percentile", "protein_percentile"]
    scores = scores.merge(disc.loc[:, disc_cols].drop_duplicates("id"), how="left", on="id")
    scores["discordance_category"] = scores["discordance_category"].fillna("not_missense_or_unscored")

    candidates = read_csv(out / CANDIDATES)
    cand_cols = ["id", "review_tier", "review_direction"]
    scores = scores.merge(candidates.loc[:, cand_cols].drop_duplicates("id"), how="left", on="id")
    scores["review_tier"] = scores["review_tier"].fillna("not_clinvar_review_candidate")
    scores["review_direction"] = scores["review_direction"].fillna("not_clinvar_review_candidate")

    panel = read_csv(out / PANEL)
    panel_cols = ["id", "panel_arm"]
    scores = scores.merge(panel.loc[:, panel_cols].drop_duplicates("id"), how="left", on="id")
    scores["panel_arm"] = scores["panel_arm"].fillna("not_in_prospective_panel")

    return scores


def make_strata(df: pd.DataFrame) -> list[Stratum]:
    strata: list[Stratum] = []
    n = len(df)

    def add(name: str, group: str, value: str, mask: pd.Series | np.ndarray) -> None:
        arr = np.asarray(mask, dtype=bool)
        if arr.shape == (n,) and arr.any():
            strata.append(Stratum(name=name, group=group, value=value, mask=arr))

    add("all", "global", "all", np.ones(n, dtype=bool))
    add("coding", "vtype", "coding", df["vtype"].eq("coding"))
    add("noncoding", "vtype", "noncoding", df["vtype"].eq("noncoding"))
    add("missense", "variant_class", "missense", df["is_missense"].astype(bool))
    add("non_missense", "variant_class", "non_missense", ~df["is_missense"].astype(bool))

    for col in ["consequence", "brca2_domain", "discordance_category", "review_tier", "review_direction", "panel_arm"]:
        for value in sorted(str(v) for v in df[col].dropna().unique()):
            add(f"{col}:{value}", col, value, df[col].astype(str).eq(value))

    split = df["discordance_category"].isin(["protein_high_dna_low", "dna_high_protein_low"])
    add("discordance_category:split_mechanism", "discordance_category", "split_mechanism", split)
    extreme = df["discordance_category"].isin(["both_high", "both_low"])
    add("discordance_category:concordant_extreme", "discordance_category", "concordant_extreme", extreme)
    actionable = df["review_direction"].isin(["pathogenic_review", "benign_review"])
    add("review_direction:actionable_review_candidate", "review_direction", "actionable_review_candidate", actionable)

    for domain in sorted(str(v) for v in df["brca2_domain"].dropna().unique()):
        domain_mask = df["brca2_domain"].astype(str).eq(domain)
        for cat in ["both_high", "both_low", "protein_high_dna_low", "dna_high_protein_low"]:
            add(
                f"domain_x_discordance:{domain}|{cat}",
                "domain_x_discordance",
                f"{domain}|{cat}",
                domain_mask & df["discordance_category"].eq(cat),
            )

    return strata


def bootstrap_delta(
    y: np.ndarray,
    recon: np.ndarray,
    ablate: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> np.ndarray:
    unique_clusters = np.unique(clusters)
    group_indices = {c: np.flatnonzero(clusters == c) for c in unique_clusters}
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([group_indices[c] for c in sampled_clusters])
        if len(np.unique(y[idx])) < 2:
            continue
        delta = score_delta(y[idx], recon[idx], ablate[idx])
        if np.isfinite(delta):
            deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def permuted_delta(
    y: np.ndarray,
    recon: np.ndarray,
    ablate: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> np.ndarray:
    deltas: list[float] = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        delta = score_delta(yp, recon, ablate)
        if np.isfinite(delta):
            deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def summarize_stratum(
    df: pd.DataFrame,
    stratum: Stratum,
    rng: np.random.Generator,
    min_n: int,
    n_boot: int,
    n_perm: int,
) -> dict[str, object]:
    sub = df.loc[stratum.mask].copy()
    y = sub["label"].astype(int).to_numpy()
    recon = sub["sae_recon_pred"].astype(float).to_numpy()
    ablate = sub["top_feature_ablate_pred"].astype(float).to_numpy()
    original = sub["original_pred"].astype(float).to_numpy()
    contribution = recon - ablate

    row: dict[str, object] = {
        "stratum": stratum.name,
        "group": stratum.group,
        "value": stratum.value,
        "n": int(len(sub)),
        "n_lof": int(y.sum()),
        "lof_rate": float(y.mean()) if len(y) else float("nan"),
        "mean_feature_contribution": float(np.nanmean(contribution)),
        "median_feature_contribution": float(np.nanmedian(contribution)),
        "mean_original_pred": float(np.nanmean(original)),
        "mean_sae_recon_pred": float(np.nanmean(recon)),
        "mean_top_feature_ablate_pred": float(np.nanmean(ablate)),
        "mean_original_gate_protein": float(np.nanmean(sub["original_gate_protein"])),
        "mean_gate_protein_shift_recon_minus_ablate": float(
            np.nanmean(sub["sae_recon_gate_protein"] - sub["top_feature_ablate_gate_protein"])
        ),
    }

    if len(sub) < min_n:
        row["analysis_status"] = "too_small_for_auc_gate"
        return row
    if len(np.unique(y)) < 2:
        row["analysis_status"] = "single_label_descriptive_only"
        return row

    obs_delta = score_delta(y, recon, ablate)
    boot = bootstrap_delta(
        y=y,
        recon=recon,
        ablate=ablate,
        clusters=sub["pos_hg38"].fillna(sub["id"]).astype(str).to_numpy(),
        rng=rng,
        n_boot=n_boot,
    )
    perm = permuted_delta(y=y, recon=recon, ablate=ablate, rng=rng, n_perm=n_perm)

    ci_lo = float(np.nanpercentile(boot, 2.5)) if len(boot) else float("nan")
    ci_hi = float(np.nanpercentile(boot, 97.5)) if len(boot) else float("nan")
    perm_p = float(((perm >= obs_delta).sum() + 1) / (len(perm) + 1)) if len(perm) else float("nan")
    passes_screen = bool(obs_delta > 0 and ci_lo > 0 and perm_p <= 0.05)

    row.update(
        {
            "analysis_status": "tested",
            "auroc_original": safe_auroc(y, original),
            "auroc_sae_recon": safe_auroc(y, recon),
            "auroc_top_feature_ablate": safe_auroc(y, ablate),
            "delta_auroc_recon_minus_top_ablate": obs_delta,
            "auprc_original": safe_auprc(y, original),
            "auprc_sae_recon": safe_auprc(y, recon),
            "auprc_top_feature_ablate": safe_auprc(y, ablate),
            "delta_auprc_recon_minus_top_ablate": safe_auprc(y, recon) - safe_auprc(y, ablate),
            "bootstrap_ci_lo_delta_auroc": ci_lo,
            "bootstrap_ci_hi_delta_auroc": ci_hi,
            "n_bootstrap_valid": int(len(boot)),
            "fixed_prediction_label_permutation_mean_delta_auroc": float(np.nanmean(perm)) if len(perm) else float("nan"),
            "fixed_prediction_label_permutation_p_delta_ge_observed": perm_p,
            "n_permutation_valid": int(len(perm)),
            "passes_focused_screen": passes_screen,
            "supports_native_sae_replication_claim": False,
        }
    )
    return row


def write_report(
    repo: Path,
    summary: pd.DataFrame,
    application: pd.DataFrame,
    min_n: int,
    n_boot: int,
    n_perm: int,
) -> None:
    out = repo / OUT_DIR
    tested = summary[summary["analysis_status"].eq("tested")].copy()
    screen_positive = tested[tested["passes_focused_screen"].astype(bool)] if not tested.empty else tested
    positive = tested.sort_values("delta_auroc_recon_minus_top_ablate", ascending=False) if not tested.empty else tested
    descriptive = application.sort_values(["group", "n"], ascending=[True, False])

    lines: list[str] = []
    lines.append("# BRCA2 Native-SAE Focused Strata Error Analysis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "The global BRCA2 native-SAE intervention is positive against random-feature controls "
        "but does not pass the label-permutation gate. This analysis tests whether a narrower "
        "domain, discordance, ClinVar-review, or prospective-panel stratum supports a bounded "
        "mechanistic claim."
    )
    lines.append("")
    lines.append("## Screen")
    lines.append("")
    lines.append(
        f"A stratum is tested only when n>={min_n} and both labels are present. The focused screen "
        f"requires delta AUROC(recon minus top-feature ablation)>0, paired cluster-bootstrap "
        f"95% CI lower bound>0, and fixed-prediction label-permutation p<=0.05. Bootstrap sets={n_boot}; "
        f"label permutations={n_perm}; clusters are genomic positions. This is weaker than the "
        "native-SAE fold-level label-permuted feature-selection control and must not be used as a "
        "standalone mechanistic replication gate."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- Tested strata: {len(tested)}")
    lines.append(f"- Focused-screen positive strata: {len(screen_positive)}")
    if len(screen_positive):
        lines.append(
            "- Interpretation: some broad strata have stable positive targeted-ablation effects, "
            "but this does not overturn the original BRCA2 native-SAE label-permutation failure."
        )
    else:
        lines.append(
            "- Interpretation: no focused stratum passes the screen; BRCA2 native-SAE should "
            "remain an error-analysis result rather than a mechanistic replication claim."
        )
    lines.append("")
    lines.append("## Screen-Positive Strata")
    lines.append("")
    lines.append(
        markdown_table(
            screen_positive.sort_values("delta_auroc_recon_minus_top_ablate", ascending=False),
            [
                "stratum",
                "n",
                "n_lof",
                "delta_auroc_recon_minus_top_ablate",
                "bootstrap_ci_lo_delta_auroc",
                "bootstrap_ci_hi_delta_auroc",
                "fixed_prediction_label_permutation_p_delta_ge_observed",
            ],
        )
    )
    lines.append("")
    lines.append("## Largest Positive Tested Effects")
    lines.append("")
    lines.append(
        markdown_table(
            positive,
            [
                "stratum",
                "n",
                "n_lof",
                "auroc_sae_recon",
                "auroc_top_feature_ablate",
                "delta_auroc_recon_minus_top_ablate",
                "bootstrap_ci_lo_delta_auroc",
                "fixed_prediction_label_permutation_p_delta_ge_observed",
                "passes_focused_screen",
            ],
        )
    )
    lines.append("")
    lines.append("## Application Strata Descriptives")
    lines.append("")
    lines.append(
        "ClinVar-review and prospective-panel arms are often single-label by design, so AUROC is "
        "not the right validation statistic. The table below reports feature contribution "
        "descriptively for review and assay handoff use."
    )
    lines.append("")
    lines.append(
        markdown_table(
            descriptive,
            [
                "stratum",
                "analysis_status",
                "n",
                "n_lof",
                "lof_rate",
                "mean_feature_contribution",
                "median_feature_contribution",
                "mean_original_pred",
            ],
            max_rows=20,
        )
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `brca2_native_sae_focused_strata_summary.csv`")
    lines.append("- `brca2_native_sae_focused_strata_application_descriptives.csv`")
    lines.append("- `brca2_native_sae_focused_strata_screen_positive.csv`")
    lines.append("")

    (out / "brca2_native_sae_focused_strata.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    rng = np.random.default_rng(args.seed)

    df = attach_context(repo)
    strata = make_strata(df)
    rows = [
        summarize_stratum(
            df=df,
            stratum=stratum,
            rng=rng,
            min_n=args.min_n,
            n_boot=args.bootstrap,
            n_perm=args.permutations,
        )
        for stratum in strata
    ]
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["analysis_status", "group", "n"], ascending=[True, True, False])

    app_groups = {"review_tier", "review_direction", "panel_arm"}
    application = summary[summary["group"].isin(app_groups)].copy()
    screen_source = summary.get("passes_focused_screen", pd.Series(False, index=summary.index))
    screen_col = screen_source.eq(True)
    screen_positive = summary[screen_col].copy()

    summary.to_csv(out / "brca2_native_sae_focused_strata_summary.csv", index=False)
    application.to_csv(out / "brca2_native_sae_focused_strata_application_descriptives.csv", index=False)
    screen_positive.to_csv(out / "brca2_native_sae_focused_strata_screen_positive.csv", index=False)
    write_report(repo, summary, application, args.min_n, args.bootstrap, args.permutations)

    tested = int(summary["analysis_status"].eq("tested").sum())
    passed = int(screen_positive.shape[0])
    print(f"wrote {out / 'brca2_native_sae_focused_strata_summary.csv'}")
    print(f"tested_strata={tested} focused_screen_positive={passed}")


if __name__ == "__main__":
    main()
