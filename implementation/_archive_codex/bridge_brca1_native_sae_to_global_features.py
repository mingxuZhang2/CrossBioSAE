#!/usr/bin/env python
"""Bridge BRCA1 fold-native SAE necessity to named global SAE mechanisms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    n = len(ranked)
    vals = ranked * n / np.arange(1, n + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return q


def safe_mwu(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return np.nan, np.nan
    r, p = spearmanr(a[ok], b[ok])
    return float(r), float(p)


def add_odds_ratio(table: np.ndarray) -> tuple[float, float]:
    try:
        odds, p = fisher_exact(table)
        return float(odds), float(p)
    except Exception:
        return np.nan, np.nan


def load_cards(repo_root: Path) -> pd.DataFrame:
    deep = repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv"
    if deep.exists():
        cards = pd.read_csv(deep)
    else:
        cards = pd.read_csv(repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards.csv")
    sge_path = repo_root / "results" / "interpretability_applications" / "brca1_sge_sae_feature_summary.csv"
    if sge_path.exists():
        sge = pd.read_csv(sge_path)
        keep = [
            "feature",
            "n_active_brca1",
            "active_frac_brca1",
            "lof_rate_active",
            "lof_rate_inactive",
            "delta_abs_act_lof_minus_func",
            "auroc_for_sge_lof",
            "auprc_for_sge_lof",
            "mannwhitney_lof_gt_func_p",
            "modality",
            "concept_v2",
            "path_rate",
            "path_enrich",
        ]
        keep = [c for c in keep if c in sge.columns]
        cards = cards.merge(sge[keep], on="feature", how="left", suffixes=("", "_brca1_sge"))
        for col in ["modality", "concept_v2", "path_rate", "path_enrich"]:
            alt = f"{col}_brca1_sge"
            if col in cards.columns and alt in cards.columns:
                cards[col] = cards[col].combine_first(cards[alt])
    return cards


def feature_bridge(
    acts: np.ndarray,
    native_effect: np.ndarray,
    label: np.ndarray,
    high_effect: np.ndarray,
    cards: pd.DataFrame,
    min_active: int,
) -> pd.DataFrame:
    rows = []
    abs_acts = np.abs(acts)
    for j in range(abs_acts.shape[1]):
        x = abs_acts[:, j]
        active = x > 0
        n_active = int(active.sum())
        if n_active < min_active or n_active >= len(x) - min_active:
            continue
        rho, rho_p = safe_spearman(x, native_effect)
        effect_active = native_effect[active]
        effect_inactive = native_effect[~active]
        active_high = int((active & high_effect).sum())
        inactive_high = int((~active & high_effect).sum())
        active_low = int((active & ~high_effect).sum())
        inactive_low = int((~active & ~high_effect).sum())
        odds, fisher_p = add_odds_ratio(np.array([[active_high, active_low], [inactive_high, inactive_low]]))
        rows.append(
            {
                "feature": j,
                "n_active_bridge": n_active,
                "active_frac_bridge": n_active / len(x),
                "spearman_abs_act_vs_native_effect": rho,
                "spearman_p": rho_p,
                "mean_native_effect_active": float(np.nanmean(effect_active)),
                "mean_native_effect_inactive": float(np.nanmean(effect_inactive)),
                "delta_native_effect_active_minus_inactive": float(
                    np.nanmean(effect_active) - np.nanmean(effect_inactive)
                ),
                "mwu_active_effect_gt_inactive_p": safe_mwu(effect_active, effect_inactive, alternative="greater"),
                "high_effect_active_rate": active_high / max(n_active, 1),
                "high_effect_inactive_rate": inactive_high / max(int((~active).sum()), 1),
                "high_effect_odds_ratio": odds,
                "high_effect_fisher_p": fisher_p,
                "lof_rate_active_bridge": float(np.nanmean(label[active])),
                "lof_rate_inactive_bridge": float(np.nanmean(label[~active])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_spearman"] = bh_qvalues(out["spearman_p"].to_numpy())
    out["q_active_effect"] = bh_qvalues(out["mwu_active_effect_gt_inactive_p"].to_numpy())
    out["q_high_effect"] = bh_qvalues(out["high_effect_fisher_p"].to_numpy())
    out = out.merge(cards, on="feature", how="left")
    return out.sort_values(
        [
            "q_active_effect",
            "delta_native_effect_active_minus_inactive",
            "spearman_abs_act_vs_native_effect",
        ],
        ascending=[True, False, False],
    )


def stratum_bridge(vs: pd.DataFrame, high_effect: np.ndarray) -> pd.DataFrame:
    rows = []
    total_high = int(high_effect.sum())
    total_low = int((~high_effect).sum())
    for kind, col in [
        ("brca1_region", "brca1_region"),
        ("vtype", "vtype"),
        ("consequence", "consequence"),
        ("missense_status", "is_missense"),
        ("label", "label"),
    ]:
        if col not in vs.columns:
            continue
        ser = vs[col].fillna("NA").astype(str)
        for val, mask in ser.groupby(ser).groups.items():
            mask = np.asarray(list(mask), dtype=int)
            in_group = np.zeros(len(vs), dtype=bool)
            in_group[mask] = True
            n_group = int(in_group.sum())
            if n_group < 10:
                continue
            high_in = int((in_group & high_effect).sum())
            low_in = int((in_group & ~high_effect).sum())
            high_out = total_high - high_in
            low_out = total_low - low_in
            odds, p = add_odds_ratio(np.array([[high_in, low_in], [high_out, low_out]]))
            rows.append(
                {
                    "group_kind": kind,
                    "group": val,
                    "n_group": n_group,
                    "n_high_effect": high_in,
                    "high_effect_rate": high_in / max(n_group, 1),
                    "rest_high_effect_rate": high_out / max(len(vs) - n_group, 1),
                    "high_effect_odds_ratio": odds,
                    "fisher_p": p,
                    "mean_native_effect": float(vs.loc[in_group, "native_effect"].mean()),
                    "median_native_effect": float(vs.loc[in_group, "native_effect"].median()),
                    "lof_rate": float(vs.loc[in_group, "label"].mean()) if "label" in vs.columns else np.nan,
                    "mean_function_score": float(vs.loc[in_group, "function_score"].mean())
                    if "function_score" in vs.columns
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_fisher"] = bh_qvalues(out["fisher_p"].to_numpy())
    return out.sort_values(["q_fisher", "high_effect_odds_ratio"], ascending=[True, False])


def make_report(
    out_path: Path,
    summary: dict[str, float],
    feature_df: pd.DataFrame,
    stratum_df: pd.DataFrame,
    top_variants: pd.DataFrame,
) -> None:
    def tbl(df: pd.DataFrame, n: int, cols: list[str]) -> str:
        if df.empty:
            return "No rows.\n"
        cols = [c for c in cols if c in df.columns]
        return df.head(n)[cols].to_markdown(index=False) + "\n"

    lines = [
        "# BRCA1 Native-SAE Biological Bridge",
        "",
        "## Purpose",
        "",
        "The fold-native `z_dna` SAE intervention is the strongest model-internal",
        "necessity test, but its selected feature IDs are local to each fold. This",
        "analysis bridges that local necessity signal back to named biology by",
        "asking whether variants most affected by native-feature ablation also",
        "co-activate pretrained/genome-wide SAE features and BRCA1 functional strata.",
        "",
        "## Native Effect Definition",
        "",
        "`native_effect = native_sae_recon_pred - top_feature_ablate_pred`; positive",
        "values mean that removing selected fold-native features lowers the predicted",
        "LOF/pathogenic probability for that variant.",
        "",
        "## Summary",
        "",
        f"- Variants analyzed: {int(summary['n_variants'])}.",
        f"- Mean native effect: {summary['mean_native_effect']:.5f}.",
        f"- Median native effect: {summary['median_native_effect']:.5f}.",
        f"- Top-decile threshold: {summary['top_decile_threshold']:.5f}.",
        f"- Top-decile variants: {int(summary['n_high_effect'])}.",
        f"- LOF rate in top-decile variants: {summary['lof_rate_high_effect']:.3f}.",
        f"- LOF rate outside top decile: {summary['lof_rate_rest']:.3f}.",
        "",
        "## Global SAE Features Tracking Native Effect",
        "",
        tbl(
            feature_df,
            20,
            [
                "feature",
                "n_active_bridge",
                "delta_native_effect_active_minus_inactive",
                "spearman_abs_act_vs_native_effect",
                "q_active_effect",
                "q_spearman",
                "high_effect_odds_ratio",
                "q_high_effect",
                "auroc_for_sge_lof",
                "delta_abs_act_lof_minus_func",
                "modality",
                "concept_v2",
                "path_rate",
                "path_enrich",
            ],
        ),
        "## Strata Enriched for High Native Effect",
        "",
        tbl(
            stratum_df,
            30,
            [
                "group_kind",
                "group",
                "n_group",
                "n_high_effect",
                "high_effect_rate",
                "rest_high_effect_rate",
                "high_effect_odds_ratio",
                "q_fisher",
                "mean_native_effect",
                "median_native_effect",
                "lof_rate",
                "mean_function_score",
            ],
        ),
        "## Highest Native-Effect Variants",
        "",
        tbl(
            top_variants,
            30,
            [
                "id",
                "brca1_region",
                "consequence",
                "aa_pos",
                "aa_ref",
                "aa_alt",
                "label",
                "function_score",
                "native_effect",
                "sae_recon_pred",
                "top_feature_ablate_pred",
                "sae_top_sge_feature_score",
                "sae_probe_pred",
                "shared_cosine",
                "discordance",
                "CADD",
                "phyloP",
            ],
        ),
        "## Interpretation",
        "",
        "This is not a one-to-one identity map from fold-native SAE features to",
        "stable genome-wide SAE cards. It is a variant-level bridge: the variants",
        "whose predictions depend most on fold-native sparse directions are tested",
        "for co-activation of named global features and enrichment in known BRCA1",
        "functional strata.",
        "",
        "A positive bridge strengthens the paper story because the strongest",
        "necessity evidence no longer stands alone as an unnamed local-basis result;",
        "it can be paired with genome-wide feature cards, BRCA1 SGE feature",
        "calibration, and domain-level localization. The remaining top-journal",
        "requirement is still independent replication, with BRCA2 as the current",
        "mainline, plus a rescue or feature-addback control for the native SAE",
        "intervention.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--top-quantile", type=float, default=0.9)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir or repo / "results" / "interpretability_applications"
    out_dir.mkdir(parents=True, exist_ok=True)

    native_path = out_dir / "brca1_native_finetuned_sae_intervention_variant_scores.csv"
    domain_path = out_dir / "brca1_domain_sae_variant_scores.csv"
    acts_path = out_dir / "brca1_sge_sae_acts.npz"

    native = pd.read_csv(native_path)
    domain = pd.read_csv(domain_path)
    acts = np.load(acts_path)["acts"]
    if len(native) != acts.shape[0]:
        raise ValueError(f"native rows {len(native)} != acts rows {acts.shape[0]}")

    native["native_effect"] = native["sae_recon_pred"] - native["top_feature_ablate_pred"]
    merge_cols = [
        "id",
        "variant_class",
        "fusion_pred",
        "shared_cosine",
        "discordance",
        "sae_all_abs_score",
        "sae_top_sge_feature_score",
        "sae_probe_pred",
    ]
    merge_cols = [c for c in merge_cols if c in domain.columns]
    vs = native.merge(domain[merge_cols], on="id", how="left", suffixes=("", "_domain"))

    effect = vs["native_effect"].to_numpy(dtype=float)
    threshold = float(np.nanquantile(effect, args.top_quantile))
    high_effect = effect >= threshold
    label = vs["label"].to_numpy(dtype=float)
    cards = load_cards(repo)

    feature_df = feature_bridge(acts, effect, label, high_effect, cards, args.min_active)
    stratum_df = stratum_bridge(vs, high_effect)
    top_variants = vs.sort_values("native_effect", ascending=False).head(200)

    feature_df.to_csv(out_dir / "brca1_native_sae_biological_bridge_feature_correlations.csv", index=False)
    stratum_df.to_csv(out_dir / "brca1_native_sae_biological_bridge_strata.csv", index=False)
    top_variants.to_csv(out_dir / "brca1_native_sae_biological_bridge_top_variants.csv", index=False)

    summary = {
        "n_variants": float(len(vs)),
        "mean_native_effect": float(np.nanmean(effect)),
        "median_native_effect": float(np.nanmedian(effect)),
        "top_decile_threshold": threshold,
        "n_high_effect": float(high_effect.sum()),
        "lof_rate_high_effect": float(np.nanmean(label[high_effect])),
        "lof_rate_rest": float(np.nanmean(label[~high_effect])),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "brca1_native_sae_biological_bridge_summary.csv", index=False)
    make_report(
        out_dir / "brca1_native_sae_biological_bridge.md",
        summary,
        feature_df,
        stratum_df,
        top_variants,
    )


if __name__ == "__main__":
    main()
