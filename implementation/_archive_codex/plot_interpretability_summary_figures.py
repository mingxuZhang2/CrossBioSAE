#!/usr/bin/env python
"""Plot current manuscript-draft summary figures for interpretability results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")
FIG_DIR = OUT_DIR / "figures"


COLORS = {
    "blue": "#31688e",
    "green": "#35b779",
    "orange": "#f89540",
    "red": "#cc4778",
    "purple": "#6f4c9b",
    "gray": "#6c757d",
    "light_gray": "#dee2e6",
}


def read_csv(repo: Path, name: str) -> pd.DataFrame:
    path = repo / OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("empty dataframe")
    return df.iloc[0]


def save(fig: plt.Figure, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    svg = out_base.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", linewidth=0.8)
    ax.set_axisbelow(True)


def row_by_test(df: pd.DataFrame, test: str) -> pd.Series:
    hit = df.loc[df["test"].eq(test)]
    if hit.empty:
        raise ValueError(f"missing test row: {test}")
    return hit.iloc[0]


def depleted_rate(row: pd.Series, prefix: str) -> float:
    return float(row[f"{prefix}_n_depleted"]) / float(row[f"{prefix}_n_binary"])


def count_label(row: pd.Series, prefix: str) -> str:
    return f"{int(row[f'{prefix}_n_depleted'])}/{int(row[f'{prefix}_n_binary'])}"


def plot_brca1(repo: Path) -> list[Path]:
    summary = first(read_csv(repo, "brca1_native_finetuned_sae_intervention_summary.csv"))
    folds = read_csv(repo, "brca1_native_finetuned_sae_intervention_folds.csv")
    raw = first(read_csv(repo, "brca1_raw_zdna_dim_control_summary.csv"))
    rand = read_csv(repo, "brca1_native_finetuned_sae_intervention_random_ablation.csv")
    dose = read_csv(repo, "brca1_native_finetuned_sae_intervention_dose_response.csv")
    rescue_summary = first(read_csv(repo, "brca1_native_finetuned_sae_rescue_summary.csv"))
    rescue = read_csv(repo, "brca1_native_finetuned_sae_rescue_rescue_controls.csv")
    sufficiency = read_csv(repo, "brca1_native_finetuned_sae_rescue_sufficiency_controls.csv")

    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6))
    fig.suptitle(
        "BRCA1 native-SAE necessity, rescue, and claim-boundary controls",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ax = axs[0, 0]
    labels = ["Original", "SAE recon", "Top-feature\nablation"]
    vals = [
        summary["original_auroc"],
        summary["native_sae_recon_auroc"],
        summary["top_feature_ablate_auroc"],
    ]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["green"], COLORS["red"]], width=0.65)
    ax.set_ylabel("AUROC")
    ax.set_ylim(max(0.84, min(vals) - 0.02), min(1.0, max(vals) + 0.02))
    ax.set_title("Native SAE preserves prediction; selected-feature ablation reduces it")
    ax.text(
        1.0,
        min(1.0, max(vals) + 0.015),
        f"Delta={summary['delta_auroc_recon_minus_top_ablate']:.3f}\n"
        f"random p={summary['empirical_p_random_delta_ge_top_mean']:.4f}\n"
        f"perm p={summary['empirical_p_label_permuted_delta_ge_top_mean']:.4f}",
        ha="center",
        va="top",
        fontsize=9,
    )
    clean_axis(ax)

    ax = axs[0, 1]
    x = np.arange(3)
    for _, row in folds.iterrows():
        y = [row["original_auc"], row["native_sae_recon_auc"], row["top_feature_ablate_auc"]]
        ax.plot(x, y, color=COLORS["light_gray"], linewidth=1.3, marker="o", markersize=3)
    ax.plot(x, vals, color=COLORS["red"], linewidth=2.2, marker="o", label="Pooled")
    ax.set_xticks(x)
    ax.set_xticklabels(["Original", "Recon", "Ablate"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(max(0.84, min(folds["top_feature_ablate_auc"].min(), min(vals)) - 0.02), min(1.0, max(folds["original_auc"].max(), max(vals)) + 0.02))
    fold_delta = folds["delta_auc_recon_minus_top_ablate"].to_numpy(dtype=float)
    ci_lo, ci_hi = np.quantile(fold_delta, [0.025, 0.975])
    ax.set_title("Held-out folds show paired degradation")
    ax.text(
        0.03,
        0.05,
        f"fold mean delta={fold_delta.mean():.3f}\nfold range={fold_delta.min():.3f}-{fold_delta.max():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    clean_axis(ax)

    ax = axs[0, 2]
    null = rand["delta_auc_random_ablate"].dropna()
    observed = summary["mean_fold_delta_auc_recon_minus_top_ablate"]
    ax.hist(null, bins=30, color=COLORS["light_gray"], edgecolor="white")
    ax.axvline(observed, color=COLORS["red"], linewidth=2, label="Observed top features")
    ax.axvline(raw["delta_auroc_original_minus_top_raw_dim_ablate"], color=COLORS["blue"], linewidth=2, linestyle="--", label="Raw dims")
    ax.set_xlabel("Delta AUROC")
    ax.set_ylabel("Random sets")
    ax.set_title("Matched-feature null control")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axs[1, 0]
    dose_summary = dose.groupby("dose_k", as_index=False)["delta_auc_recon_minus_ablate"].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        dose_summary["dose_k"],
        dose_summary["mean"],
        yerr=dose_summary["std"].fillna(0.0),
        marker="o",
        color=COLORS["purple"],
        linewidth=2,
        capsize=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dose_summary["dose_k"])
    ax.set_xticklabels([str(int(x)) for x in dose_summary["dose_k"]])
    ax.set_xlabel("Top SAE features ablated")
    ax.set_ylabel("Delta AUROC")
    ax.set_title("Dose response")
    clean_axis(ax)

    ax = axs[1, 1]
    rescue_delta = rescue["delta_auc_selected_rescue_minus_random_rescue"].dropna()
    ax.hist(rescue_delta, bins=30, color=COLORS["light_gray"], edgecolor="white")
    ax.axvline(
        rescue_summary["mean_rescue_delta_auc_selected_minus_random"],
        color=COLORS["green"],
        linewidth=2,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Selected addback minus random addback AUROC")
    ax.set_ylabel("Matched random sets")
    ax.set_title("Common-background addback rescue")
    ax.text(
        0.98,
        0.93,
        f"mean={rescue_summary['mean_rescue_delta_auc_selected_minus_random']:.3f}\n"
        f"p(delta<=0)={rescue_summary['empirical_p_rescue_delta_auc_le_zero']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    clean_axis(ax)

    ax = axs[1, 2]
    random_only = sufficiency.loc[
        sufficiency["control"].eq("random_active_matched_only"),
        "feature_only_auc",
    ].dropna()
    bottom_only = sufficiency.loc[
        sufficiency["control"].eq("bottom_abs_delta_active_matched_only"),
        "feature_only_auc",
    ].dropna()
    feature_labels = ["Selected\nfeatures", "Random\nfeatures", "Bottom\nfeatures", "Decoder\nbias"]
    feature_vals = [
        rescue_summary["selected_feature_only_auroc"],
        random_only.mean(),
        bottom_only.mean(),
        rescue_summary["bias_only_auroc"],
    ]
    feature_err = [
        0.0,
        random_only.std(ddof=1),
        bottom_only.std(ddof=1),
        0.0,
    ]
    ax.bar(feature_labels, feature_vals, yerr=feature_err, capsize=3, color=[COLORS["purple"], COLORS["light_gray"], COLORS["gray"], COLORS["blue"]])
    ax.set_ylabel("Feature-only AUROC")
    ax.set_ylim(max(0.84, min(feature_vals) - 0.025), min(0.94, max(feature_vals) + 0.025))
    ax.set_title("Feature-only decoding bounds the claim")
    ax.text(
        0.5,
        0.06,
        f"random >= selected p={rescue_summary['empirical_p_random_feature_only_auc_ge_selected_mean']:.3f}\n"
        "necessary, not sufficient alone",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    clean_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig2_brca1_native_sae")


def plot_brca1_biological_bridge(repo: Path) -> list[Path]:
    summary = first(read_csv(repo, "brca1_native_sae_biological_bridge_summary.csv"))
    matched = read_csv(repo, "brca1_native_sae_matched_annotation_control_summary.csv")
    matched_row = first(
        matched.loc[
            matched["comparison"].eq("top_decile_native_effect_vs_consequence_region_matched_null")
        ]
    )
    cov_row = first(read_csv(repo, "brca1_native_sae_covariate_matched_control_summary.csv"))
    cov_balance = read_csv(repo, "brca1_native_sae_covariate_matched_control_balance.csv")
    strata = read_csv(repo, "brca1_native_sae_biological_bridge_strata.csv")
    bridge = read_csv(repo, "brca1_global_sae_native_effect_bridge_summary.csv")
    features = read_csv(repo, "brca1_global_feature_cards.csv")
    variants = read_csv(repo, "brca1_native_effect_variant_cards.csv")
    top_variants = read_csv(repo, "brca1_native_sae_biological_bridge_top_variants.csv")

    fig, axs = plt.subplots(2, 3, figsize=(17, 8.8))
    fig.suptitle("BRCA1 native-SAE biological localization and matched-control support", fontsize=16, fontweight="bold", y=0.98)

    ax = axs[0, 0]
    labels = [
        "Top effect",
        "Region null",
        "CADD/phyloP\nmatched",
        "Rest",
    ]
    vals = [
        summary["lof_rate_high_effect"],
        matched_row["rest_or_null_mean_lof_rate"],
        cov_row["control_lof_rate"],
        summary["lof_rate_rest"],
    ]
    ax.bar(labels, vals, color=[COLORS["red"], COLORS["light_gray"], COLORS["orange"], COLORS["gray"]], width=0.62)
    ax.errorbar(
        [1],
        [matched_row["rest_or_null_mean_lof_rate"]],
        yerr=[matched_row["rest_or_null_sd_lof_rate"]],
        color="black",
        capsize=3,
        linewidth=1,
    )
    ax.set_ylabel("SGE-LOF rate")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("High native-effect variants exceed matched nulls")
    ax.tick_params(axis="x", labelrotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, val in enumerate(vals):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.50,
        0.94,
        f"n={int(summary['n_high_effect'])}; region p={matched_row['p_value']:.4f}\n"
        f"CADD/phyloP p={cov_row['fisher_p_greater']:.4f}, CI=[{cov_row['delta_bootstrap_ci_low']:.3f}, {cov_row['delta_bootstrap_ci_high']:.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["light_gray"], "alpha": 0.9},
    )
    clean_axis(ax)

    ax = axs[0, 1]
    lollipop = top_variants.loc[top_variants["aa_pos"].notna()].copy()
    lollipop = lollipop.sort_values("native_effect", ascending=False).head(75)
    domains = [
        ("RING", 24, 65, COLORS["blue"]),
        ("BRCT1", 1642, 1736, COLORS["green"]),
        ("BRCT2", 1756, 1855, COLORS["purple"]),
    ]
    for name, start, end, color in domains:
        ax.axvspan(start, end, color=color, alpha=0.10)
        ax.text((start + end) / 2, 0.94, name, ha="center", va="top", fontsize=8, color=color)
    lof = lollipop["label"].astype(int).eq(1)
    colors = np.where(lof, COLORS["red"], COLORS["blue"])
    ax.vlines(lollipop["aa_pos"], 0, lollipop["native_effect"], color=colors, alpha=0.45, linewidth=1)
    ax.scatter(lollipop["aa_pos"], lollipop["native_effect"], c=colors, s=22, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, max(0.95, lollipop["native_effect"].max() * 1.08))
    ax.set_xlabel("BRCA1 amino-acid position")
    ax.set_ylabel("Native-effect score")
    ax.set_title("Protein-coordinate examples of high native effect")
    ax.text(
        0.02,
        0.08,
        "red=SGE-LOF; blue=functional\nshaded=RING/BRCT domains",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
    )
    clean_axis(ax)

    ax = axs[0, 2]
    show = strata.loc[
        strata["group_kind"].isin(["consequence", "brca1_region"])
        & strata["group"].isin(["Nonsense", "Canonical splice", "Splice region", "Synonymous", "splice_or_noncoding", "RING_24_65", "BRCT1_1642_1736", "BRCT2_1756_1855"])
    ].copy()
    show = show.sort_values("high_effect_odds_ratio", ascending=True)
    show["label"] = (
        show["group"]
        .str.replace("_", " ", regex=False)
        .str.replace("Canonical splice", "Canonical\nsplice", regex=False)
        .str.replace("splice or noncoding", "splice/noncoding", regex=False)
    )
    colors = [COLORS["red"] if x > 1 else COLORS["blue"] for x in show["high_effect_odds_ratio"]]
    ax.barh(show["label"], np.log2(show["high_effect_odds_ratio"].astype(float)), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log2 OR for high native-effect")
    ax.set_title("Localization by variant class and BRCA1 region")
    clean_axis(ax)

    ax = axs[1, 0]
    balance = cov_balance.loc[cov_balance["metric"].isin(["cadd", "phylop"])].copy()
    balance["label"] = balance["metric"].map({"cadd": "CADD", "phylop": "phyloP"})
    y = np.arange(len(balance))
    ax.barh(y, balance["mean_difference"], color=[COLORS["orange"], COLORS["green"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(balance["label"])
    ax.set_xlabel("High-effect minus matched-control mean")
    ax.set_title("CADD/phyloP balance after nearest-neighbor matching")
    ax.set_xlim(-0.055, 0.022)
    for yi, (_, row) in enumerate(balance.iterrows()):
        ax.text(
            0.002,
            yi,
            f"diff={row['mean_difference']:.3f}; median pair |diff|={row['median_abs_pair_difference']:.3f}",
            ha="left",
            va="center",
            fontsize=8.2,
        )
    clean_axis(ax)

    ax = axs[1, 1]
    metrics = ["high_effect_auroc", "native_effect_spearman"]
    labels = ["High-effect\nAUROC", "Native-effect\nSpearman"]
    x = np.arange(len(metrics))
    width = 0.18
    series = [
        ("Global SAE", "global_sae_selected_mean", COLORS["purple"]),
        ("Annotation", "annotation_baseline_mean", COLORS["green"]),
        ("Random", "random_feature_mean", COLORS["light_gray"]),
        ("Label perm", "label_permuted_mean", COLORS["gray"]),
    ]
    for i, (name, col, color) in enumerate(series):
        vals = []
        for metric in metrics:
            row = bridge.loc[bridge["metric"].eq(metric)].iloc[0]
            vals.append(float(row[col]))
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=name, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Global-SAE bridge is supportive, not causal")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    clean_axis(ax)

    ax = axs[1, 2]
    ax.axis("off")
    feat = features.sort_values("q_high_effect").head(4)
    var = variants.sort_values("native_effect", ascending=False).head(3)
    feat_lines = [
        f"Feature {int(row.feature)}: OR={row.high_effect_odds_ratio:.2f}, q={row.q_high_effect:.1e}, {row.concept_v2}"
        for _, row in feat.iterrows()
    ]
    var_lines = [
        f"{row.id}: {row.brca1_region}, {row.consequence}, effect={row.native_effect:.3f}"
        for _, row in var.iterrows()
    ]
    text = (
        "Feature-card examples\n"
        + "\n".join(feat_lines)
        + "\n\nHigh-effect variant examples\n"
        + "\n".join(var_lines)
        + "\n\nClaim boundary:\nfeature cards support naming/localization;\ncausal proof remains Figure 2 intervention."
    )
    ax.text(0.01, 0.98, text, ha="left", va="top", fontsize=9.2, linespacing=1.35)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig3_brca1_biological_bridge")


def plot_brca2_checkpoint(repo: Path) -> list[Path]:
    metric = read_csv(repo, "brca2_llr_esm_checkpoint_metric_summary.csv")
    delta = read_csv(repo, "brca2_llr_esm_checkpoint_bootstrap_delta.csv")
    disc = read_csv(repo, "brca2_llr_esm_discordance_category_summary.csv")
    dom = read_csv(repo, "brca2_llr_esm_discordance_domain_summary.csv")
    domain_matched = first(read_csv(repo, "brca2_discordance_domain_matched_control_summary.csv"))
    proxy = read_csv(repo, "brca2_external_proxy_validation_known_clinvar.csv")
    mavedb_category = read_csv(repo, "brca2_mavedb_external_assay_category_tests.csv")

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle("BRCA2 independent SGE checkpoint and mechanism stratification", fontsize=16, fontweight="bold", y=0.98)

    ax = axs[0, 0]
    scopes = ["all", "missense", "non_missense"]
    methods = ["Evo2 LLR zero-shot", "ESM-only CV logistic", "ESM+Evo2 LLR CV logistic"]
    x = np.arange(len(scopes))
    width = 0.24
    for i, method in enumerate(methods):
        vals = []
        for scope in scopes:
            hit = metric.loc[(metric["scope"] == scope) & (metric["metric"] == method)]
            vals.append(float(hit["auroc_for_sge_lof"].iloc[0]) if not hit.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Missense", "Non-missense"])
    ax.set_ylabel("AUROC for SGE-LOF")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Checkpoint prediction metrics")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axs[0, 1]
    keep = delta.loc[
        (delta["comparison"] == "ESM+LLR minus LLR-CV")
        & (delta["scope"].isin(["all", "missense", "non_missense"]))
    ].copy()
    keep["label"] = keep["scope"].map({"all": "All", "missense": "Missense", "non_missense": "Non-missense"})
    y = np.arange(len(keep))
    ax.errorbar(
        keep["median_delta_auroc"],
        y,
        xerr=[keep["median_delta_auroc"] - keep["ci_lo"], keep["ci_hi"] - keep["median_delta_auroc"]],
        fmt="o",
        color=COLORS["red"],
        ecolor=COLORS["gray"],
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(keep["label"])
    ax.set_xlabel("Delta AUROC")
    ax.set_title("Fusion does not beat Evo2 LLR overall")
    clean_axis(ax)

    ax = axs[1, 0]
    order = ["both_high", "protein_high_dna_low", "dna_high_protein_low", "both_low", "other"]
    plot = disc.set_index("category").loc[[x for x in order if x in set(disc["category"])]].reset_index()
    ax.bar(plot["category"], plot["lof_rate"], color=[COLORS["red"], COLORS["orange"], COLORS["blue"], COLORS["green"], COLORS["gray"]][: len(plot)])
    ax.set_ylabel("SGE-LOF rate")
    ax.set_title("DNA/protein concordance quadrants")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    clean_axis(ax)

    ax = axs[0, 2]
    order = ["both_high", "both_low", "dna_high_protein_low", "protein_high_dna_low", "other"]
    clinvar = proxy.set_index("discordance_category").loc[
        [x for x in order if x in set(proxy["discordance_category"])]
    ].reset_index()
    colors = [
        COLORS["red"] if x == "both_high" else COLORS["green"] if x == "both_low" else COLORS["gray"]
        for x in clinvar["discordance_category"]
    ]
    ax.bar(clinvar["discordance_category"], clinvar["pathogenic_rate"], color=colors)
    ax.axhline(clinvar.loc[clinvar["discordance_category"] != "both_high", "pathogenic_rate"].mean(), color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pathogenic ClinVar rate")
    ax.set_title("Exact current ClinVar known-label proxy")
    ax.text(
        0,
        max(0.08, float(clinvar.loc[clinvar["discordance_category"].eq("both_high"), "pathogenic_rate"].iloc[0]) - 0.04),
        "both-high\nq="
        + f"{float(clinvar.loc[clinvar['discordance_category'].eq('both_high'), 'fisher_q_pathogenic_enrichment_vs_rest'].iloc[0]):.3g}",
        ha="center",
        va="top",
        fontsize=8,
        color="white",
    )
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    clean_axis(ax)

    ax = axs[1, 1]
    both_high = dom.loc[dom["category"] == "both_high"].copy()
    both_high = both_high.sort_values("lof_rate", ascending=True)
    both_high["domain"] = (
        both_high["brca2_domain"]
        .str.replace("CTDB_", "", regex=False)
        .str.replace("_2479_2668", "", regex=False)
        .str.replace("_2682_2794", "", regex=False)
        .str.replace("_2804_3054", "", regex=False)
        .str.replace("_3073_3167", "", regex=False)
        .str.replace("_2479_3216", "", regex=False)
    )
    ax.barh(both_high["domain"], both_high["lof_rate"], color=COLORS["purple"])
    ax.set_xlabel("LOF rate in both-high group")
    ax.set_title("Both-high enrichment by BRCA2 CTDB domain")
    clean_axis(ax)

    ax = axs[1, 2]
    ax.axis("off")
    both_high_proxy = proxy.loc[proxy["discordance_category"].eq("both_high")].iloc[0]
    hap1_bh = mavedb_category[
        mavedb_category["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
        & mavedb_category["comparison"].eq("both_high_vs_both_low")
    ]
    vc8_bh = mavedb_category[
        mavedb_category["mavedb_urn"].eq("urn:mavedb:00001224-a-1")
        & mavedb_category["comparison"].eq("both_high_vs_both_low")
    ]
    mavedb_line = ""
    if not hap1_bh.empty and not vc8_bh.empty:
        hap1_row = hap1_bh.iloc[0]
        vc8_row = vc8_bh.iloc[0]
        mavedb_line = (
            "\nPublic MaveDB assays:\n"
            f"HAP1 AUROC={float(hap1_row['external_lof_auroc']):.3f}, p={float(hap1_row['mannwhitney_p_greater']):.1e}\n"
            f"VC-8 AUROC={float(vc8_row['external_lof_auroc']):.3f}, p={float(vc8_row['mannwhitney_p_greater']):.3g}\n\n"
        )
    text = (
        "Mechanism-stratification evidence\n\n"
        f"SGE both-high: n={int(disc.loc[disc['category'].eq('both_high'), 'n'].iloc[0])}, "
        f"LOF rate={float(disc.loc[disc['category'].eq('both_high'), 'lof_rate'].iloc[0]):.3f}\n"
        f"SGE OR={float(disc.loc[disc['category'].eq('both_high'), 'fisher_odds_vs_rest'].iloc[0]):.2f}, "
        f"q={float(disc.loc[disc['category'].eq('both_high'), 'fisher_q_vs_rest'].iloc[0]):.2e}\n\n"
        f"Domain-matched null: {domain_matched['matched_control_mean_lof_rate']:.3f}\n"
        f"delta={domain_matched['delta_lof_rate']:.3f}, "
        f"CI=[{domain_matched['delta_bootstrap_ci_low']:.3f}, {domain_matched['delta_bootstrap_ci_high']:.3f}]\n\n"
        f"Exact ClinVar known proxy: {int(both_high_proxy['n_pathogenic_or_likely_pathogenic'])}/"
        f"{int(both_high_proxy['n_exact_known_clinvar'])} pathogenic\n"
        f"proxy q={float(both_high_proxy['fisher_q_pathogenic_enrichment_vs_rest']):.3g}\n\n"
        + mavedb_line
        +
        "Claim boundary:\n"
        "proxy supports mechanism strata;\n"
        "it is not VUS reclassification."
    )
    ax.text(0.02, 0.98, text, ha="left", va="top", fontsize=9.4, linespacing=1.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig4_brca2_checkpoint")


def plot_brca2_application(repo: Path) -> list[Path]:
    tiers = read_csv(repo, "brca2_clinvar_interpretability_candidates_tier_summary.csv")
    panel = read_csv(repo, "brca2_prospective_followup_panel_summary.csv")
    fisher = read_csv(repo, "brca2_assay_statistical_plan_fisher_thresholds.csv")
    validation = first(read_csv(repo, "brca2_blinded_validation_status.csv"))
    proxy_panel = read_csv(repo, "brca2_external_proxy_validation_panel_same_residue.csv")
    proxy_tests = read_csv(repo, "brca2_external_proxy_validation_panel_proxy_tests.csv")
    temporal_candidate = first(read_csv(repo, "brca2_clinvar_temporal_proxy_validation_candidate_tests.csv"))
    temporal_panel = first(read_csv(repo, "brca2_clinvar_temporal_proxy_validation_panel_tests.csv"))
    mavedb_panel = read_csv(repo, "brca2_mavedb_external_assay_panel_tests.csv")
    hap1_endpoint = read_csv(repo, "brca2_public_mavedb_hap1_matched_subset_endpoints.csv")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.suptitle("BRCA2 VUS triage and validation-ready assay design", fontsize=16, fontweight="bold", y=0.98)

    ax = axs[0, 0]
    tiers = tiers.copy()
    tiers["label"] = tiers["review_tier"].str.replace("tier1_", "T1 ", regex=False).str.replace("tier2_", "T2 ", regex=False).str.replace("_", " ")
    show = tiers.head(5)
    ax.barh(show["label"], show["n"], color=COLORS["blue"])
    ax.set_xlabel("Candidate count")
    ax.set_title("ClinVar unresolved/conflicting tiers")
    clean_axis(ax)

    ax = axs[0, 1]
    panel = panel.copy()
    panel["arm"] = panel["panel_arm"].str.replace("prospective_", "", regex=False).str.replace("_", " ")
    x = np.arange(len(panel))
    ax.bar(x, panel["n"], color=COLORS["gray"], label="Total")
    ax.bar(x, panel["n_lof"], color=COLORS["red"], label="SGE LOF")
    ax.set_xticks(x)
    ax.set_xticklabels(panel["arm"], rotation=35, ha="right")
    ax.set_ylabel("Variants")
    ax.set_title("Prospective panel arms")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axs[1, 0]
    show = fisher.loc[fisher["benign_control_lof_count"].isin([0, 1, 2, 3, 4])].copy()
    ax.plot(
        show["benign_control_lof_count"],
        show["min_pathogenic_lof_for_p_lt_0.05"],
        marker="o",
        label="p < 0.05",
        color=COLORS["green"],
    )
    ax.plot(
        show["benign_control_lof_count"],
        show["min_pathogenic_lof_for_p_lt_0.01"],
        marker="o",
        label="p < 0.01",
        color=COLORS["purple"],
    )
    ax.set_xlabel("LOF count in 16 benign controls")
    ax.set_ylabel("Required LOF count in 24 pathogenic targets")
    ax.set_title("Assay statistical threshold")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axs[1, 1]
    rows = [
        ("Review binary", validation["review_rows_with_binary_classification"], validation["review_rows_total"]),
        ("Assay binary", validation["assay_rows_with_binary_call"], validation["assay_rows_total"]),
        ("Assay mean", validation["assay_rows_with_mean_function_score"], validation["assay_rows_total"]),
    ]
    labels = [x[0] for x in rows]
    counts = [float(x[1]) for x in rows]
    totals = [float(x[2]) for x in rows]
    ax.bar(labels, totals, color=COLORS["light_gray"], label="Required rows")
    ax.bar(labels, counts, color=COLORS["blue"], label="Locked outcomes")
    ax.set_ylim(0, max(totals) * 1.15)
    ax.set_ylabel("Rows")
    ax.set_title("Blinded validation readiness")
    ax.legend(frameon=False, fontsize=8)
    for i, (count, total) in enumerate(zip(counts, totals, strict=False)):
        ax.text(i, total + 1.5, f"{int(count)}/{int(total)}", ha="center", va="bottom", fontsize=8)
    clean_axis(ax)

    proxy_order = ["prospective_pathogenic_review", "prospective_benign_controls"]
    show = proxy_panel.set_index("panel_arm").loc[proxy_order].reset_index()
    path_test = proxy_tests.loc[
        proxy_tests["proxy_test"].eq("pathogenic_review_enriched_for_same_residue_pathogenic_no_benign")
    ].iloc[0]
    hap1_panel = mavedb_panel[
        mavedb_panel["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
        & mavedb_panel["comparison"].eq("pathogenic_review_vs_benign_controls")
    ]
    hap1_text = ""
    if not hap1_panel.empty:
        hap1 = hap1_panel.iloc[0]
        hap1_text = (
            "\npublic HAP1 sGE assay:\n"
            f"{int(hap1['n_pathogenic_review_matches'])} vs {int(hap1['n_control_matches'])} matches; "
            f"AUROC={float(hap1['external_lof_auroc']):.3f}; "
            f"p={float(hap1['mannwhitney_p_greater']):.1e}"
        )
    hap1_binary = hap1_endpoint[hap1_endpoint["comparison"].eq("pathogenic_review_vs_benign_controls")]
    if not hap1_binary.empty:
        row = hap1_binary.iloc[0]
        hap1_text += (
            "\nHAP1 binary readout: "
            f"{int(row['positive_binary_lof'])}/{int(row['n_positive_group'])} vs "
            f"{int(row['negative_binary_lof'])}/{int(row['n_negative_group'])}; "
            f"p={float(row['fisher_p_greater']):.1e}"
        )
    ax.text(
        0.53,
        0.82,
        "Supportive proxies, not locked validation\n"
        f"exact known: {int(proxy_panel['n_current_exact_known'].sum())}/64\n"
        "same-residue pathogenic/no-benign:\n"
        f"{int(show.loc[show['panel_arm'].eq('prospective_pathogenic_review'), 'n_same_residue_pathogenic_no_benign'].iloc[0])}/24 vs "
        f"{int(show.loc[show['panel_arm'].eq('prospective_benign_controls'), 'n_same_residue_pathogenic_no_benign'].iloc[0])}/16; "
        f"p={float(path_test['fisher_p_greater']):.3f}\n"
        "archive-current tier proxy:\n"
        f"{int(temporal_candidate['positive_current_pathogenic_known_only'])}/"
        f"{int(temporal_candidate['n_positive_current_exact_known'])} vs "
        f"{int(temporal_candidate['negative_current_pathogenic_known_only'])}/"
        f"{int(temporal_candidate['n_negative_current_exact_known'])} P/LP; "
        f"p={float(temporal_candidate['known_only_fisher_p_greater']):.3f}\n"
        "panel old-unresolved sensitivity:\n"
        f"{int(temporal_panel['positive_current_pathogenic_missing_as_not_pathogenic'])}/"
        f"{int(temporal_panel['n_positive_old_unresolved'])} vs "
        f"{int(temporal_panel['negative_current_pathogenic_missing_as_not_pathogenic'])}/"
        f"{int(temporal_panel['n_negative_old_unresolved'])}; "
        f"p={float(temporal_panel['missing_as_not_pathogenic_fisher_p_greater']):.3f}"
        + hap1_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig5_brca2_application")


def plot_brca2_native_sae(repo: Path) -> list[Path]:
    summary = first(read_csv(repo, "brca2_native_finetuned_sae_intervention_summary.csv"))
    rand = read_csv(repo, "brca2_native_finetuned_sae_intervention_random_ablation.csv")
    dose = read_csv(repo, "brca2_native_finetuned_sae_intervention_dose_response.csv")
    strata = read_csv(repo, "brca2_native_finetuned_sae_intervention_stratum_summary.csv")

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("BRCA2 native-SAE mixed replication result", fontsize=16, fontweight="bold", y=0.98)

    ax = axs[0, 0]
    labels = ["Original", "SAE recon", "Top-feature\nablation"]
    vals = [
        summary["original_auroc"],
        summary["native_sae_recon_auroc"],
        summary["top_feature_ablate_auroc"],
    ]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["green"], COLORS["orange"]], width=0.65)
    ax.set_ylabel("AUROC")
    ax.set_ylim(max(0.72, min(vals) - 0.02), min(0.82, max(vals) + 0.02))
    ax.set_title("Positive ablation effect, but not strict replication")
    ax.text(
        1.0,
        min(0.82, max(vals) + 0.015),
        f"Delta={summary['delta_auroc_recon_minus_top_ablate']:.3f}\n"
        f"random p={summary['empirical_p_random_delta_ge_top_mean']:.4f}\n"
        f"perm p={summary['empirical_p_label_permuted_delta_ge_top_mean']:.4f}",
        ha="center",
        va="top",
        fontsize=9,
    )
    clean_axis(ax)

    ax = axs[0, 1]
    null = rand["delta_auc_random_ablate"].dropna()
    observed = summary["mean_fold_delta_auc_recon_minus_top_ablate"]
    ax.hist(null, bins=30, color=COLORS["light_gray"], edgecolor="white")
    ax.axvline(observed, color=COLORS["orange"], linewidth=2, label="Observed top features")
    ax.axvline(summary["mean_label_permuted_delta_auc"], color=COLORS["purple"], linewidth=2, linestyle="--", label="Label-perm mean")
    ax.set_xlabel("Delta AUROC")
    ax.set_ylabel("Random sets")
    ax.set_title("Random-feature control passes; label-permutation gate does not")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axs[1, 0]
    dose_summary = dose.groupby("dose_k", as_index=False)["delta_auc_recon_minus_ablate"].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        dose_summary["dose_k"],
        dose_summary["mean"],
        yerr=dose_summary["std"].fillna(0.0),
        marker="o",
        color=COLORS["blue"],
        linewidth=2,
        capsize=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dose_summary["dose_k"])
    ax.set_xticklabels([str(int(x)) for x in dose_summary["dose_k"]])
    ax.set_xlabel("Top SAE features ablated")
    ax.set_ylabel("Delta AUROC")
    ax.set_title("Dose response is positive but small")
    clean_axis(ax)

    ax = axs[1, 1]
    keep = strata.copy()
    keep = keep.sort_values("delta_auroc_recon_minus_top_ablate", ascending=True).tail(8)
    keep["label"] = (
        keep["stratum"]
        .str.replace("brca2_domain:", "", regex=False)
        .str.replace("consequence:", "", regex=False)
        .str.replace("CTDB_", "", regex=False)
        .str.replace("_", " ")
    )
    ax.barh(keep["label"], keep["delta_auroc_recon_minus_top_ablate"], color=COLORS["orange"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta AUROC")
    ax.set_title("Strata for focused follow-up")
    clean_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig6_brca2_native_sae_mixed")


def plot_third_gene_benchmarks(repo: Path) -> list[Path]:
    bap1 = read_csv(repo, "bap1_sge_external_anchors_tests.csv")
    bap1_baseline = read_csv(repo, "bap1_sge_baseline_predictors_metric_summary.csv")
    rad51c = read_csv(repo, "rad51c_sge_external_anchors_tests.csv")
    bap1_metrics_path = repo / OUT_DIR / "bap1_llr_esm_checkpoint_metric_summary.csv"
    bap1_disc_path = repo / OUT_DIR / "bap1_llr_esm_discordance_category_summary.csv"
    rad_metrics_path = repo / OUT_DIR / "rad51c_llr_esm_checkpoint_metric_summary.csv"
    rad_disc_path = repo / OUT_DIR / "rad51c_llr_esm_discordance_category_summary.csv"
    bap1_metrics = pd.read_csv(bap1_metrics_path) if bap1_metrics_path.exists() else pd.DataFrame()
    bap1_disc = pd.read_csv(bap1_disc_path) if bap1_disc_path.exists() else pd.DataFrame()
    rad_metrics = pd.read_csv(rad_metrics_path) if rad_metrics_path.exists() else pd.DataFrame()
    rad_disc = pd.read_csv(rad_disc_path) if rad_disc_path.exists() else pd.DataFrame()

    bap1_clinvar = row_by_test(bap1, "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    bap1_plof = row_by_test(bap1, "plof_consequences_enriched_for_depleted_vs_synonymous_utr")
    rad_clinvar = row_by_test(rad51c, "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    rad_nonsense = row_by_test(rad51c, "nonsense_enriched_for_depleted_vs_synonymous")
    rad_missense = row_by_test(rad51c, "missense_enriched_for_depleted_vs_synonymous")
    rad_missense_utr = row_by_test(rad51c, "missense_enriched_for_depleted_vs_utr")

    fig, axs = plt.subplots(2, 2, figsize=(13.8, 8.2))
    fig.suptitle(
        "BAP1/RAD51C third-gene functional-map benchmark anchors",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ax = axs[0, 0]
    labels = ["BAP1\nP/LP", "BAP1\nB/LB", "RAD51C\nP/LP", "RAD51C\nB/LB"]
    vals = [
        depleted_rate(bap1_clinvar, "first"),
        depleted_rate(bap1_clinvar, "second"),
        depleted_rate(rad_clinvar, "first"),
        depleted_rate(rad_clinvar, "second"),
    ]
    counts = [
        count_label(bap1_clinvar, "first"),
        count_label(bap1_clinvar, "second"),
        count_label(rad_clinvar, "first"),
        count_label(rad_clinvar, "second"),
    ]
    colors = [COLORS["red"], COLORS["blue"], COLORS["red"], COLORS["blue"]]
    ax.bar(labels, vals, color=colors, width=0.66)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("SGE-depleted rate")
    ax.set_title("Exact ClinVar anchors separate pathogenic from benign")
    for i, (val, label) in enumerate(zip(vals, counts, strict=False)):
        ax.text(i, val + 0.03, label, ha="center", va="bottom", fontsize=9)
    ax.text(
        0.50,
        0.53,
        f"BAP1 AUROC={float(bap1_clinvar['external_lof_auroc']):.3f}, p={float(bap1_clinvar['fisher_p']):.1e}\n"
        f"RAD51C AUROC={float(rad_clinvar['external_lof_auroc']):.3f}, p={float(rad_clinvar['fisher_p_greater']):.1e}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["light_gray"], "alpha": 0.92},
    )
    clean_axis(ax)

    ax = axs[0, 1]
    labels = [
        "BAP1\npLOF",
        "BAP1\nsyn/UTR",
        "RAD51C\nnonsense",
        "RAD51C\nmissense",
        "RAD51C\nsyn",
        "RAD51C\nUTR",
    ]
    vals = [
        depleted_rate(bap1_plof, "first"),
        depleted_rate(bap1_plof, "second"),
        depleted_rate(rad_nonsense, "first"),
        depleted_rate(rad_missense, "first"),
        depleted_rate(rad_nonsense, "second"),
        depleted_rate(rad_missense_utr, "second"),
    ]
    counts = [
        count_label(bap1_plof, "first"),
        count_label(bap1_plof, "second"),
        count_label(rad_nonsense, "first"),
        count_label(rad_missense, "first"),
        count_label(rad_nonsense, "second"),
        count_label(rad_missense_utr, "second"),
    ]
    colors = [
        COLORS["red"],
        COLORS["gray"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["gray"],
        COLORS["gray"],
    ]
    ax.bar(labels, vals, color=colors, width=0.68)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("SGE-depleted rate")
    ax.set_title("Expected consequence controls validate both assays")
    ax.tick_params(axis="x", labelrotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, (val, label) in enumerate(zip(vals, counts, strict=False)):
        ax.text(i, val + 0.025, label, ha="center", va="bottom", fontsize=8.2)
    ax.text(
        0.98,
        0.72,
        f"BAP1 pLOF AUROC={float(bap1_plof['external_lof_auroc']):.3f}\n"
        f"RAD51C nonsense AUROC={float(rad_nonsense['external_lof_auroc']):.3f}\n"
        f"RAD51C missense AUROC={float(rad_missense['external_lof_auroc']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["light_gray"], "alpha": 0.92},
    )
    clean_axis(ax)

    ax = axs[1, 0]
    public = bap1_baseline.loc[
        bap1_baseline["baseline"].isin(
            ["EVE score", "SIFT deleterious (-score)", "PolyPhen ordinal damage"]
        )
    ].copy()
    public["label"] = public["baseline"].map(
        {
            "EVE score": "EVE",
            "SIFT deleterious (-score)": "SIFT",
            "PolyPhen ordinal damage": "PolyPhen",
        }
    )
    public = public.sort_values("auroc_for_sge_depleted", ascending=True)
    ax.barh(public["label"], public["auroc_for_sge_depleted"], color=COLORS["green"])
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(0.45, 0.84)
    ax.set_xlabel("AUROC for BAP1 SGE-depleted label")
    ax.set_title("BAP1 public missense baselines set the comparator bar")
    for i, (_, row) in enumerate(public.iterrows()):
        ax.text(
            float(row["auroc_for_sge_depleted"]) + 0.008,
            i,
            f"{float(row['auroc_for_sge_depleted']):.3f}",
            va="center",
            fontsize=8.8,
        )
    clean_axis(ax)

    ax = axs[1, 1]
    ax.axis("off")
    status_checks = [
        ("BAP1 ESM delta", repo / "results/variant/bap1_esm_delta.npz"),
        ("BAP1 Evo2 LLR", repo / "results/variant/bap1_evo2_llr.npz"),
        ("BAP1 checkpoint", repo / OUT_DIR / "bap1_llr_esm_checkpoint_metric_summary.csv"),
        ("RAD51C ESM delta", repo / "results/variant/rad51c_esm_delta.npz"),
        ("RAD51C Evo2 LLR", repo / "results/variant/rad51c_evo2_llr.npz"),
        ("RAD51C checkpoint", repo / OUT_DIR / "rad51c_llr_esm_checkpoint_metric_summary.csv"),
    ]
    status_lines = [f"{name}: {'present' if path.exists() else 'pending'}" for name, path in status_checks]
    checkpoint_lines: list[str] = []
    if not bap1_metrics.empty:
        bap1_all_llr = bap1_metrics.loc[
            bap1_metrics["scope"].eq("all") & bap1_metrics["metric"].eq("Evo2 LLR zero-shot")
        ]
        bap1_all_fusion = bap1_metrics.loc[
            bap1_metrics["scope"].eq("all") & bap1_metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
        ]
        bap1_missense_fusion = bap1_metrics.loc[
            bap1_metrics["scope"].eq("missense") & bap1_metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
        ]
        if not bap1_all_llr.empty and not bap1_all_fusion.empty and not bap1_missense_fusion.empty:
            checkpoint_lines.append(
                "BAP1 finite checkpoint: all AUROC "
                f"{float(bap1_all_llr.iloc[0]['auroc_for_sge_lof']):.3f} -> "
                f"{float(bap1_all_fusion.iloc[0]['auroc_for_sge_lof']):.3f}; "
                "missense fusion "
                f"{float(bap1_missense_fusion.iloc[0]['auroc_for_sge_lof']):.3f}"
            )
    if not bap1_disc.empty:
        both_high = bap1_disc.loc[bap1_disc["category"].eq("both_high")]
        both_low = bap1_disc.loc[bap1_disc["category"].eq("both_low")]
        if not both_high.empty and not both_low.empty:
            checkpoint_lines.append(
                "BAP1 both-high: "
                f"n={int(both_high.iloc[0]['n'])}, LOF={float(both_high.iloc[0]['lof_rate']):.3f}; "
                f"both-low={float(both_low.iloc[0]['lof_rate']):.3f}"
            )
    if not rad_metrics.empty:
        rad_all_llr = rad_metrics.loc[
            rad_metrics["scope"].eq("all") & rad_metrics["metric"].eq("Evo2 LLR zero-shot")
        ]
        rad_all_fusion = rad_metrics.loc[
            rad_metrics["scope"].eq("all") & rad_metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
        ]
        rad_missense_fusion = rad_metrics.loc[
            rad_metrics["scope"].eq("missense") & rad_metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
        ]
        if not rad_all_llr.empty and not rad_all_fusion.empty and not rad_missense_fusion.empty:
            checkpoint_lines.append(
                "RAD51C checkpoint: all AUROC "
                f"{float(rad_all_llr.iloc[0]['auroc_for_sge_lof']):.3f} -> "
                f"{float(rad_all_fusion.iloc[0]['auroc_for_sge_lof']):.3f}; "
                "missense fusion "
                f"{float(rad_missense_fusion.iloc[0]['auroc_for_sge_lof']):.3f}"
            )
    if not rad_disc.empty:
        both_high = rad_disc.loc[rad_disc["category"].eq("both_high")]
        both_low = rad_disc.loc[rad_disc["category"].eq("both_low")]
        if not both_high.empty and not both_low.empty:
            checkpoint_lines.append(
                "RAD51C both-high: "
                f"n={int(both_high.iloc[0]['n'])}, LOF={float(both_high.iloc[0]['lof_rate']):.3f}; "
                f"both-low={float(both_low.iloc[0]['lof_rate']):.3f}"
            )
    text = (
        "Interpretation boundary\n\n"
        "What is supported now:\n"
        "- BAP1 and RAD51C are credible third-gene functional-map benchmarks.\n"
        "- Both assays recover expected ClinVar and consequence anchors.\n"
        "- BAP1 has public EVE/SIFT/PolyPhen comparator baselines.\n"
        + ("\n".join(f"- {line}" for line in checkpoint_lines) + "\n\n" if checkpoint_lines else "\n")
        + "What is not yet supported:\n"
        "- Third-gene native-SAE causal replication.\n"
        "- Any VUS reclassification claim from these anchors alone.\n\n"
        "Checkpoint state:\n"
        + "\n".join(status_lines)
    )
    ax.text(0.02, 0.98, text, ha="left", va="top", fontsize=8.4, linespacing=1.22)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, repo / FIG_DIR / "fig_s2_third_gene_benchmarks")


def write_index(repo: Path, paths: list[Path]) -> Path:
    out = repo / FIG_DIR / "interpretability_figure_outputs.md"
    lines = [
        "# Interpretability Figure Outputs",
        "",
        "Current manuscript-draft figures generated from local result tables.",
        "",
    ]
    for path in paths:
        if path.suffix == ".png":
            lines.append(f"- `{path.relative_to(repo)}`")
    lines.extend(
        [
            "",
            "Generated figures are drafts for scientific review. Figure 2 now includes BRCA1 native-SAE deletion necessity, fold-paired degradation, matched random/raw controls, dose response, common-background addback rescue, and feature-only insufficiency boundary. Figure 3 includes BRCA1 native-effect biological localization, protein-coordinate examples, consequence/region enrichment, CADD/phyloP-nearest matched control, covariate balance, and feature-card examples. Figure 4 includes BRCA2 both-high mechanism stratification, domain-matched non-both-high null support, exact current ClinVar known-label proxy support, and public MaveDB external-assay stratum support. Figure 5 includes blinded validation readiness, same-residue proxy support, archive-current ClinVar temporal-proxy support, and HAP1 public functional-assay panel support. Figure 6 is an error-analysis figure: BRCA2 native-SAE has a positive targeted-ablation effect but does not pass the strict label-permutation replication gate. Supplementary Figure S2 summarizes BAP1/RAD51C third-gene functional-map benchmark anchors plus the positive RAD51C checkpoint and finite-SNV BAP1 checkpoint while keeping third-gene native-SAE causal replication pending.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    all_paths: list[Path] = []
    all_paths.extend(plot_brca1(repo))
    all_paths.extend(plot_brca1_biological_bridge(repo))
    all_paths.extend(plot_brca2_checkpoint(repo))
    all_paths.extend(plot_brca2_application(repo))
    third_gene_inputs = [
        repo / OUT_DIR / "bap1_sge_external_anchors_tests.csv",
        repo / OUT_DIR / "bap1_sge_baseline_predictors_metric_summary.csv",
        repo / OUT_DIR / "rad51c_sge_external_anchors_tests.csv",
    ]
    if all(path.exists() for path in third_gene_inputs):
        all_paths.extend(plot_third_gene_benchmarks(repo))
    if (repo / OUT_DIR / "brca2_native_finetuned_sae_intervention_summary.csv").exists():
        all_paths.extend(plot_brca2_native_sae(repo))
    index = write_index(repo, all_paths)
    for path in all_paths:
        print(f"wrote {path}")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
