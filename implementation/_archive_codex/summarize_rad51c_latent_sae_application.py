#!/usr/bin/env python
"""Build RAD51C latent-SAE application artifacts.

This report turns the checkpoint-level RAD51C latent-SAE intervention into a
downstream review/assay design. It does not introduce new model training; it
joins existing RAD51C SGE, ClinVar anchor, checkpoint-discordance, and
latent-SAE intervention outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return path


def fmt(x: float | int | str | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, str):
        return x
    return f"{float(x):.{digits}g}"


def percentile_rank(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return x.rank(method="average", pct=True).fillna(0.0)


def normalize_clinvar(s: pd.Series) -> pd.Series:
    return s.fillna("Unobserved").astype(str).replace({"nan": "Unobserved", "": "Unobserved"})


def first_existing(df: pd.DataFrame, cols: Iterable[str]) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def add_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sparse_necessity_drop"] = out["sae_recon_pred"] - out["top_feature_ablate_pred"]
    out["sparse_sufficiency_lift"] = out["selected_feature_only_pred"] - out["bias_only_pred"]
    out["recon_minus_original"] = out["sae_recon_pred"] - out["original_pred"]
    out["clinvar_simple"] = normalize_clinvar(out.get("clinvar_simple", pd.Series(index=out.index, dtype=object)))
    out["discordance_category"] = out.get("discordance_category", "not_missense")
    out["is_review_candidate_status"] = out["clinvar_simple"].isin(
        ["Uncertain", "Uncertain significance", "Conflicting", "Conflicting/Likely_pathogenic", "Unobserved"]
    )
    missense = out["is_missense"].astype(bool) if "is_missense" in out.columns else pd.Series(False, index=out.index)
    for col in ["sparse_necessity_drop", "sparse_sufficiency_lift", "sae_recon_pred", "external_lof_score"]:
        score_col = f"{col}_pct_missense"
        out[score_col] = 0.0
        out.loc[missense, score_col] = percentile_rank(out.loc[missense, col])
    out["latent_application_score"] = (
        0.35 * out["sparse_necessity_drop_pct_missense"]
        + 0.25 * out["sparse_sufficiency_lift_pct_missense"]
        + 0.20 * out["sae_recon_pred_pct_missense"]
        + 0.20 * out["external_lof_score_pct_missense"]
    )
    out.loc[~missense, "latent_application_score"] = (
        0.50 * percentile_rank(out.loc[~missense, "sparse_necessity_drop"])
        + 0.25 * percentile_rank(out.loc[~missense, "sae_recon_pred"])
        + 0.25 * percentile_rank(out.loc[~missense, "external_lof_score"])
    )
    return out


def choose_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    used: set[str] = set()
    arms = [
        (
            "both_high_sparse_lof_review",
            16,
            (
                df["is_missense"].astype(bool)
                & (df["label"].astype(int) == 1)
                & (df["discordance_category"] == "both_high")
                & df["is_review_candidate_status"]
            ),
            "Primary RAD51C sparse-mechanism review/assay candidates.",
        ),
        (
            "protein_high_dna_low_sparse_lof",
            12,
            (
                df["is_missense"].astype(bool)
                & (df["label"].astype(int) == 1)
                & (df["discordance_category"] == "protein_high_dna_low")
            ),
            "Protein-side rescue arm: sparse ESM mechanism finds LOF with low DNA score.",
        ),
        (
            "sparse_high_sge_functional_conflict",
            12,
            (
                df["is_missense"].astype(bool)
                & (df["label"].astype(int) == 0)
                & (df["sparse_necessity_drop_pct_missense"] >= 0.90)
            ),
            "Error-analysis arm: high sparse support but SGE functional.",
        ),
        (
            "both_low_functional_control",
            12,
            (
                df["is_missense"].astype(bool)
                & (df["label"].astype(int) == 0)
                & (df["discordance_category"] == "both_low")
            ),
            "Matched low-risk missense negative controls.",
        ),
        (
            "synonymous_utr_control",
            12,
            (
                (df["label"].astype(int) == 0)
                & (~df["is_missense"].astype(bool))
                & (df["consequence"].astype(str).isin(["Synonymous", "UTR", "synonymous", "UTR"]))
            ),
            "Non-missense negative controls where protein latent-SAE should not drive signal.",
        ),
    ]
    rows = []
    summaries = []
    rank = 1
    for arm, target_n, mask, rationale in arms:
        cand = df.loc[mask & ~df["id"].isin(used)].copy()
        cand = cand.sort_values(
            ["latent_application_score", "sparse_necessity_drop", "external_lof_score"],
            ascending=[False, False, False],
        )
        take = cand.head(target_n).copy()
        used.update(take["id"].tolist())
        take.insert(0, "panel_rank", range(rank, rank + len(take)))
        take.insert(0, "panel_arm", arm)
        take["panel_rationale"] = rationale
        rank += len(take)
        rows.append(take)
        summaries.append(
            {
                "panel_arm": arm,
                "target_n": target_n,
                "selected_n": int(len(take)),
                "candidate_pool_n": int(len(cand)),
                "lof_rate": float(take["label"].mean()) if len(take) else np.nan,
                "median_sparse_necessity_drop": float(take["sparse_necessity_drop"].median()) if len(take) else np.nan,
                "median_sufficiency_lift": float(take["sparse_sufficiency_lift"].median()) if len(take) else np.nan,
                "median_sae_recon_pred": float(take["sae_recon_pred"].median()) if len(take) else np.nan,
                "clinvar_status_counts": "; ".join(
                    f"{k}:{v}" for k, v in take["clinvar_simple"].value_counts(dropna=False).sort_index().items()
                ),
                "rationale": rationale,
            }
        )
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return panel, pd.DataFrame(summaries)


def feature_fold_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, sub in selected.groupby("fold"):
        top = sub.sort_values("train_abs_delta", ascending=False).head(5)
        rows.append(
            {
                "fold": int(fold),
                "n_selected_features": int(len(sub)),
                "median_train_active": float(sub["train_active"].median()),
                "median_train_abs_delta": float(sub["train_abs_delta"].median()),
                "max_train_abs_delta": float(sub["train_abs_delta"].max()),
                "top5_fold_local_features": ";".join(top["feature"].astype(str).tolist()),
                "top5_abs_deltas": ";".join(fmt(v, 4) for v in top["train_abs_delta"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def utility_tests(summary: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    s = summary.iloc[0].to_dict()
    miss = strata.loc[strata["stratum"].astype(str).eq("missense")]
    syn = strata.loc[strata["stratum"].astype(str).eq("synonymous")]
    nonsense = strata.loc[strata["stratum"].astype(str).eq("nonsense")]
    miss_delta = float(miss.iloc[0]["delta_auroc_recon_minus_top_ablate"]) if not miss.empty else np.nan
    syn_delta = float(syn.iloc[0]["delta_auroc_recon_minus_top_ablate"]) if not syn.empty else np.nan
    nonsense_delta = float(nonsense.iloc[0]["delta_auroc_recon_minus_top_ablate"]) if not nonsense.empty else np.nan
    doses = [
        float(s.get("mean_dose_8_delta_auc", np.nan)),
        float(s.get("mean_dose_16_delta_auc", np.nan)),
        float(s.get("mean_dose_32_delta_auc", np.nan)),
        float(s.get("mean_dose_64_delta_auc", np.nan)),
    ]
    dose_monotone = all(np.isfinite(doses[i]) and np.isfinite(doses[i + 1]) and doses[i] <= doses[i + 1] for i in range(3))
    rows = [
        {
            "test_name": "top_sparse_feature_necessity",
            "scope": "all",
            "observed": float(s.get("delta_auroc_recon_minus_top_ablate", np.nan)),
            "comparator": "positive_delta",
            "comparator_value": 0.0,
            "p_value": np.nan,
            "pass": bool(float(s.get("delta_auroc_recon_minus_top_ablate", 0.0)) > 0),
            "interpretation": "Top selected sparse features reduce held-out LOF AUROC when ablated.",
        },
        {
            "test_name": "matched_random_control",
            "scope": "all",
            "observed": float(s.get("mean_fold_delta_auc_recon_minus_top_ablate", np.nan)),
            "comparator": "mean_random_delta_auc",
            "comparator_value": float(s.get("mean_random_delta_auc", np.nan)),
            "p_value": float(s.get("empirical_p_random_delta_ge_top_mean", np.nan)),
            "pass": bool(float(s.get("empirical_p_random_delta_ge_top_mean", 1.0)) <= 0.05),
            "interpretation": "Selected sparse features outperform matched random features.",
        },
        {
            "test_name": "label_permutation_control",
            "scope": "all",
            "observed": float(s.get("mean_fold_delta_auc_recon_minus_top_ablate", np.nan)),
            "comparator": "mean_label_permuted_delta_auc",
            "comparator_value": float(s.get("mean_label_permuted_delta_auc", np.nan)),
            "p_value": float(s.get("empirical_p_label_permuted_delta_ge_top_mean", np.nan)),
            "pass": bool(float(s.get("empirical_p_label_permuted_delta_ge_top_mean", 1.0)) <= 0.05),
            "interpretation": "Feature selection depends on true labels rather than arbitrary sparse axes.",
        },
        {
            "test_name": "common_background_rescue",
            "scope": "all",
            "observed": float(s.get("mean_rescue_delta_auc_selected_minus_random", np.nan)),
            "comparator": "zero_or_negative_rescue_delta",
            "comparator_value": 0.0,
            "p_value": float(s.get("empirical_p_rescue_delta_auc_le_zero", np.nan)),
            "pass": bool(
                float(s.get("mean_rescue_delta_auc_selected_minus_random", 0.0)) > 0
                and float(s.get("empirical_p_rescue_delta_auc_le_zero", 1.0)) <= 0.05
            ),
            "interpretation": "Adding selected features back restores prediction more than random features.",
        },
        {
            "test_name": "feature_only_sufficiency_boundary",
            "scope": "all",
            "observed": float(s.get("selected_feature_only_auroc", np.nan)),
            "comparator": "bias_only_auroc",
            "comparator_value": float(s.get("bias_only_auroc", np.nan)),
            "p_value": float(s.get("empirical_p_random_feature_only_auc_ge_selected_mean", np.nan)),
            "pass": bool(
                float(s.get("selected_feature_only_auroc", 0.0)) > float(s.get("bias_only_auroc", 1.0))
                and float(s.get("empirical_p_random_feature_only_auc_ge_selected_mean", 1.0)) <= 0.05
            ),
            "interpretation": "Selected sparse features alone carry more held-out signal than bias or random features.",
        },
        {
            "test_name": "dose_response",
            "scope": "all",
            "observed": doses[-1],
            "comparator": "dose_8_to_64_delta_auc",
            "comparator_value": doses[0],
            "p_value": np.nan,
            "pass": bool(dose_monotone),
            "interpretation": "Ablation effect increases from top-8 to top-64 sparse features.",
        },
        {
            "test_name": "missense_localization",
            "scope": "missense_vs_nonmissense_controls",
            "observed": miss_delta,
            "comparator": "synonymous_and_nonsense_delta_auc",
            "comparator_value": max([v for v in [syn_delta, nonsense_delta] if np.isfinite(v)], default=np.nan),
            "p_value": np.nan,
            "pass": bool(miss_delta > 0 and abs(syn_delta) < 1e-9 and abs(nonsense_delta) < 1e-9),
            "interpretation": "Protein latent-SAE effect localizes to missense/coding variants, not synonymous or nonsense controls.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No rows.\n"
    show = df.head(max_rows).copy()
    return show.to_markdown(index=False)


def write_report(
    out_path: Path,
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    arm_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    utility: pd.DataFrame,
    priorities: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    s = summary.iloc[0].to_dict()
    missense = strata.loc[strata["stratum"].astype(str).eq("missense")]
    miss = missense.iloc[0].to_dict() if not missense.empty else {}
    both_high = priorities.loc[
        (priorities["discordance_category"] == "both_high") & (priorities["is_missense"].astype(bool))
    ]
    protein_high = priorities.loc[
        (priorities["discordance_category"] == "protein_high_dna_low") & (priorities["is_missense"].astype(bool))
    ]
    lines = [
        "# RAD51C Latent-SAE Application Report",
        "",
        "## Main Finding",
        "",
        (
            "RAD51C checkpoint-level ESM latent sparse features are necessary for held-out SGE LOF prediction: "
            f"SAE-reconstructed AUROC {fmt(s.get('sae_recon_auroc'))} vs top-feature-ablation AUROC "
            f"{fmt(s.get('top_feature_ablate_auroc'))}, delta {fmt(s.get('delta_auroc_recon_minus_top_ablate'))}. "
            f"Matched random-feature empirical p={fmt(s.get('empirical_p_random_delta_ge_top_mean'))}; "
            f"label-permuted feature-selection p={fmt(s.get('empirical_p_label_permuted_delta_ge_top_mean'))}; "
            f"addback/rescue p(delta<=0)={fmt(s.get('empirical_p_rescue_delta_auc_le_zero'))}."
        ),
        "",
        (
            "The effect localizes to protein-changing RAD51C variants: missense delta AUROC "
            f"{fmt(miss.get('delta_auroc_recon_minus_top_ablate'))}, while synonymous and nonsense strata have "
            "zero protein-latent ablation effect in the current output."
        ),
        "",
        "## Downstream Use",
        "",
        (
            "Use this result to build a RAD51C review/assay panel that tests whether sparse protein-side "
            "mechanism support enriches for SGE-depleted variants, especially in both-high and "
            "protein-high/DNA-low missense strata. This is an application of checkpoint-level sparse "
            "mechanism evidence, not a native CrossBioSAE causal replication."
        ),
        "",
        "## Panel Arms",
        "",
        markdown_table(arm_summary),
        "",
        "## Feature Fold Summary",
        "",
        markdown_table(feature_summary),
        "",
        "## Utility Tests",
        "",
        markdown_table(utility),
        "",
        "## Top Priority Variants",
        "",
        markdown_table(
            priorities[
                [
                    "id",
                    "HGVSc",
                    "HGVSp",
                    "clinvar_simple",
                    "discordance_category",
                    "label",
                    "function_score",
                    "latent_application_score",
                    "sparse_necessity_drop",
                    "sparse_sufficiency_lift",
                    "sae_recon_pred",
                    "external_lof_score",
                ]
            ].head(25)
        ),
        "",
        "## Mechanism Stratum Notes",
        "",
        f"- both_high missense candidates in the priority table: {len(both_high)}.",
        f"- protein_high_dna_low missense candidates in the priority table: {len(protein_high)}.",
        (
            "- Candidate priority is a pre-specified rank combining sparse necessity drop, selected-feature-only "
            "lift, SAE-reconstructed LOF probability, and SGE external LOF score within missense variants."
        ),
        "- Panel rows are intended for review/assay prioritization, not clinical reclassification.",
        "",
        "## Output Files",
        "",
        "- rad51c_latent_sae_application_variant_priorities.csv",
        "- rad51c_latent_sae_application_panel.csv",
        "- rad51c_latent_sae_application_arm_summary.csv",
        "- rad51c_latent_sae_application_feature_fold_summary.csv",
        "- rad51c_latent_sae_application_utility_tests.csv",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR

    summary = pd.read_csv(require(out / "rad51c_latent_sae_intervention_summary.csv"))
    strata = pd.read_csv(require(out / "rad51c_latent_sae_intervention_stratum_summary.csv"))
    selected = pd.read_csv(require(out / "rad51c_latent_sae_intervention_selected_features.csv"))
    latent = pd.read_csv(require(out / "rad51c_latent_sae_intervention_variant_scores.csv"))
    discord = pd.read_csv(require(out / "rad51c_llr_esm_discordance_missense_scores.csv"))
    anchors = pd.read_csv(require(out / "rad51c_sge_external_anchors_annotated.csv"))

    keep_discord = ["id", "dna_score", "protein_score", "dna_percentile", "protein_percentile", "discordance_category"]
    discord = discord[[c for c in keep_discord if c in discord.columns]].drop_duplicates("id")
    keep_anchor = [
        "id",
        "variant_key",
        "external_lof_score",
        "clinvar_matched",
        "clinvar_simple",
        "clinvar_clnsig",
        "clinvar_review_status",
        "clinvar_rs",
        "clinvar_disease",
        "clinvar_snapshot_date",
    ]
    anchors = anchors[[c for c in keep_anchor if c in anchors.columns]].drop_duplicates("id")
    df = latent.merge(discord, on="id", how="left").merge(anchors, on="id", how="left")
    if "external_lof_score" not in df.columns:
        score_col = first_existing(df, ["function_score", "score"])
        df["external_lof_score"] = -pd.to_numeric(df[score_col], errors="coerce") if score_col else np.nan
    df["discordance_category"] = df["discordance_category"].fillna("not_missense")
    df = add_priority_scores(df)
    priorities = df.sort_values(
        ["latent_application_score", "sparse_necessity_drop", "external_lof_score"],
        ascending=[False, False, False],
    )

    panel, arm_summary = choose_panel(priorities)
    feature_summary = feature_fold_summary(selected)
    utility = utility_tests(summary, strata)

    priority_cols = [
        "id",
        "variant_key",
        "HGVSc",
        "HGVSp",
        "consequence",
        "domains",
        "clinvar_simple",
        "clinvar_clnsig",
        "clinvar_review_status",
        "clinvar_rs",
        "clinvar_disease",
        "label",
        "functional_classification",
        "function_score",
        "external_lof_score",
        "evo2_llr_zero_shot",
        "evo2_llr_cv_pred",
        "esm_cv_pred",
        "esm_llr_cv_pred",
        "original_pred",
        "sae_recon_pred",
        "top_feature_ablate_pred",
        "selected_feature_only_pred",
        "bias_only_pred",
        "sparse_necessity_drop",
        "sparse_sufficiency_lift",
        "recon_minus_original",
        "dna_score",
        "protein_score",
        "dna_percentile",
        "protein_percentile",
        "discordance_category",
        "latent_application_score",
        "is_review_candidate_status",
    ]
    priorities[[c for c in priority_cols if c in priorities.columns]].to_csv(
        out / "rad51c_latent_sae_application_variant_priorities.csv", index=False
    )
    panel_cols = ["panel_arm", "panel_rank"] + [c for c in priority_cols if c in panel.columns] + ["panel_rationale"]
    panel[panel_cols].to_csv(out / "rad51c_latent_sae_application_panel.csv", index=False)
    arm_summary.to_csv(out / "rad51c_latent_sae_application_arm_summary.csv", index=False)
    feature_summary.to_csv(out / "rad51c_latent_sae_application_feature_fold_summary.csv", index=False)
    utility.to_csv(out / "rad51c_latent_sae_application_utility_tests.csv", index=False)
    write_report(
        out / "rad51c_latent_sae_application.md",
        summary,
        strata,
        arm_summary,
        feature_summary,
        utility,
        priorities,
        panel,
    )
    print(f"wrote {out / 'rad51c_latent_sae_application.md'}")


if __name__ == "__main__":
    main()
