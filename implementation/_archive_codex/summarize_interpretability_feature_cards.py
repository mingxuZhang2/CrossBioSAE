#!/usr/bin/env python
"""Generate feature and hotspot cards for the interpretability manuscript."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def read_csv(repo: Path, name: str) -> pd.DataFrame:
    path = repo / OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(x: object, digits: int = 3, sci: bool = False) -> str:
    try:
        if pd.isna(x):
            return "NA"
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if sci:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def aa_pos_from_change(change: object) -> float:
    if pd.isna(change):
        return float("nan")
    match = re.search(r"(\d+)", str(change))
    return float(match.group(1)) if match else float("nan")


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_brca1_global_feature_cards(repo: Path, n: int = 12) -> pd.DataFrame:
    features = read_csv(repo, "brca1_native_sae_biological_bridge_feature_correlations.csv")
    cols = [
        "feature",
        "concept_v2",
        "modality",
        "n_active_bridge",
        "delta_native_effect_active_minus_inactive",
        "spearman_abs_act_vs_native_effect",
        "high_effect_odds_ratio",
        "q_high_effect",
        "auroc_for_sge_lof",
        "path_rate",
        "path_enrich",
    ]
    for col in cols:
        if col not in features.columns:
            features[col] = pd.NA
    features = features.loc[
        features["concept_v2"].notna()
        & features["modality"].notna()
        & (features["delta_native_effect_active_minus_inactive"] > 0)
        & (features["high_effect_odds_ratio"] > 1)
    ].copy()
    cards = features.sort_values(
        ["q_high_effect", "spearman_abs_act_vs_native_effect"],
        ascending=[True, False],
        na_position="last",
    ).head(n)[cols].copy()
    cards["card_name"] = cards.apply(
        lambda r: f"global_feature_{int(r['feature'])}_{str(r['concept_v2'])}",
        axis=1,
    )
    cards["evidence_sentence"] = cards.apply(
        lambda r: (
            f"Feature {int(r['feature'])} ({r['concept_v2']}, {r['modality']}) tracks BRCA1 native-SAE effect: "
            f"active-minus-inactive native effect {fmt(r['delta_native_effect_active_minus_inactive'])}, "
            f"Spearman {fmt(r['spearman_abs_act_vs_native_effect'])}, "
            f"high-effect OR {fmt(r['high_effect_odds_ratio'])}, q={fmt(r['q_high_effect'], sci=True)}."
        ),
        axis=1,
    )
    cards["paper_use"] = "feature-card support for BRCA1 native-effect biology; bridge evidence, not causal sufficiency"
    return cards


def build_brca1_variant_cards(repo: Path, n: int = 20) -> pd.DataFrame:
    variants = read_csv(repo, "brca1_native_sae_biological_bridge_top_variants.csv")
    keep_cols = [
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
        "CADD",
        "phyloP",
        "clinvar_simple",
    ]
    for col in keep_cols:
        if col not in variants.columns:
            variants[col] = pd.NA
    cards = variants.sort_values("native_effect", ascending=False).head(n)[keep_cols].copy()
    cards["aa_change"] = cards.apply(
        lambda r: "NA"
        if pd.isna(r["aa_pos"]) or pd.isna(r["aa_ref"]) or pd.isna(r["aa_alt"])
        else f"{r['aa_ref']}{int(r['aa_pos'])}{r['aa_alt']}",
        axis=1,
    )
    cards["evidence_sentence"] = cards.apply(
        lambda r: (
            f"{r['id']} ({r['brca1_region']}, {r['consequence']}, {r['aa_change']}) has native effect "
            f"{fmt(r['native_effect'])}; SAE recon prediction {fmt(r['sae_recon_pred'])} drops to "
            f"{fmt(r['top_feature_ablate_pred'])} after top-feature ablation."
        ),
        axis=1,
    )
    cards["paper_use"] = "variant examples for BRCA1 causal SAE intervention and feature-card panels"
    return cards


def build_brca2_hotspot_cards(repo: Path, n: int = 20) -> pd.DataFrame:
    hotspots = read_csv(repo, "brca2_llr_esm_discordance_residue_hotspots.csv")
    panel = read_csv(repo, "brca2_prospective_followup_panel.csv")
    candidates = read_csv(repo, "brca2_clinvar_interpretability_candidates_top_candidates.csv")

    for frame in [panel, candidates]:
        frame["aa_pos"] = frame["AA.change"].map(aa_pos_from_change)

    rows: list[dict[str, object]] = []
    both_high = hotspots.loc[hotspots["category"].eq("both_high")].copy()
    both_high = both_high.sort_values(["lof_rate", "n_lof", "n_variants"], ascending=[False, False, False]).head(n)
    for _, h in both_high.iterrows():
        aa_pos = h["aa_pos"]
        panel_hits = panel.loc[panel["aa_pos"].eq(aa_pos)].copy()
        cand_hits = candidates.loc[candidates["aa_pos"].eq(aa_pos)].copy()
        pathogenic_review = panel_hits.loc[panel_hits["panel_arm"].astype(str).str.contains("pathogenic", na=False)]
        split_review = panel_hits.loc[panel_hits["panel_arm"].astype(str).str.contains("split", na=False)]
        benign_or_conflict = panel_hits.loc[
            panel_hits["panel_arm"].astype(str).str.contains("benign|conflict", case=False, regex=True, na=False)
        ]
        rows.append(
            {
                "aa_pos": int(aa_pos),
                "domain": h["brca2_domain"],
                "n_variants": int(h["n_variants"]),
                "n_lof": int(h["n_lof"]),
                "lof_rate": h["lof_rate"],
                "aa_changes": h["aa_changes"],
                "mean_function_score": h["mean_function_score"],
                "mean_dna_score": h["mean_dna_score"],
                "mean_protein_score": h["mean_protein_score"],
                "panel_hits": len(panel_hits),
                "pathogenic_review_hits": len(pathogenic_review),
                "split_mechanism_hits": len(split_review),
                "benign_or_conflict_hits": len(benign_or_conflict),
                "candidate_hits": len(cand_hits),
                "panel_variants": ";".join(panel_hits["AA.change"].dropna().astype(str).tolist()[:8]),
                "candidate_variants": ";".join(cand_hits["AA.change"].dropna().astype(str).tolist()[:8]),
            }
        )
    cards = pd.DataFrame(rows)
    cards["evidence_sentence"] = cards.apply(
        lambda r: (
            f"BRCA2 residue {int(r['aa_pos'])} in {r['domain']} is a both-high hotspot: "
            f"{int(r['n_lof'])}/{int(r['n_variants'])} assayed substitutions are SGE-LOF "
            f"(rate {fmt(r['lof_rate'])}); DNA/protein means {fmt(r['mean_dna_score'])}/"
            f"{fmt(r['mean_protein_score'])}; prospective-panel hits {int(r['panel_hits'])}."
        ),
        axis=1,
    )
    cards["paper_use"] = "residue-level BRCA2 mechanism cards for VUS triage and assay-panel rationale"
    return cards


def build_brca2_panel_cards(repo: Path) -> pd.DataFrame:
    panel = read_csv(repo, "brca2_prospective_followup_panel.csv").copy()
    panel["aa_pos"] = panel["AA.change"].map(aa_pos_from_change)
    priority = {
        "prospective_pathogenic_review": 0,
        "prospective_split_mechanism_tests": 1,
        "prospective_benign_controls": 2,
        "prospective_model_conflict_controls": 3,
    }
    panel["arm_order"] = panel["panel_arm"].map(priority).fillna(9)
    panel = panel.sort_values(["arm_order", "panel_rank"])
    cols = [
        "panel_rank",
        "panel_arm",
        "id",
        "AA.change",
        "brca2_domain",
        "clinvar_simple",
        "old_status_class",
        "func_class",
        "function_score",
        "discordance_category",
        "dna_percentile",
        "protein_percentile",
        "hotspot_n_variants",
        "hotspot_n_lof",
        "same_residue_other_known_pathogenic",
        "same_residue_other_known_benign",
        "c.nom",
        "g.nom",
    ]
    for col in cols:
        if col not in panel.columns:
            panel[col] = pd.NA
    cards = panel[cols].copy()
    cards["evidence_sentence"] = cards.apply(
        lambda r: (
            f"{r['AA.change']} ({r['panel_arm']}) is {r['old_status_class']} with SGE class {r['func_class']}, "
            f"DNA/protein percentiles {fmt(r['dna_percentile'])}/{fmt(r['protein_percentile'])}, "
            f"and hotspot support {fmt(r['hotspot_n_lof'], 0)}/{fmt(r['hotspot_n_variants'], 0)}."
        ),
        axis=1,
    )
    cards["paper_use"] = "variant-card manifest for blinded review or mini-assay follow-up"
    return cards


def write_report(
    repo: Path,
    brca1_features: pd.DataFrame,
    brca1_variants: pd.DataFrame,
    brca2_hotspots: pd.DataFrame,
    brca2_panel: pd.DataFrame,
) -> Path:
    out = repo / OUT_DIR / "interpretability_feature_hotspot_cards.md"
    lines = [
        "# Interpretability Feature And Hotspot Cards",
        "",
        "## Purpose",
        "",
        "This report converts the current interpretability evidence into concrete feature, variant, residue-hotspot, and review-panel cards. These are manuscript-facing examples; they are not independent clinical classifications.",
        "",
        "## BRCA1 Global Feature Cards",
        "",
        md_table(
            brca1_features,
            [
                "feature",
                "concept_v2",
                "modality",
                "n_active_bridge",
                "delta_native_effect_active_minus_inactive",
                "spearman_abs_act_vs_native_effect",
                "high_effect_odds_ratio",
                "q_high_effect",
                "paper_use",
            ],
        ),
        "",
        "## BRCA1 Highest Native-Effect Variant Cards",
        "",
        md_table(
            brca1_variants,
            [
                "id",
                "brca1_region",
                "consequence",
                "aa_change",
                "label",
                "function_score",
                "native_effect",
                "sae_recon_pred",
                "top_feature_ablate_pred",
            ],
        ),
        "",
        "## BRCA2 Both-High Residue Hotspot Cards",
        "",
        md_table(
            brca2_hotspots,
            [
                "aa_pos",
                "domain",
                "n_variants",
                "n_lof",
                "lof_rate",
                "aa_changes",
                "panel_hits",
                "pathogenic_review_hits",
                "split_mechanism_hits",
                "candidate_variants",
            ],
        ),
        "",
        "## BRCA2 Prospective Panel Variant Cards",
        "",
        md_table(
            brca2_panel,
            [
                "panel_rank",
                "panel_arm",
                "AA.change",
                "brca2_domain",
                "old_status_class",
                "func_class",
                "function_score",
                "discordance_category",
                "dna_percentile",
                "protein_percentile",
            ],
        ),
        "",
        "## Interpretation Guardrails",
        "",
        "- BRCA1 cards support biological naming and example selection for the already-complete native-SAE causal intervention.",
        "- BRCA2 hotspot cards support mechanism stratification and assay-panel rationale before full BRCA2 native-SAE replication.",
        "- These cards should be paired with intervention statistics; they should not be used as standalone proof.",
        "- VUS and panel rows are review/assay targets, not clinical reclassifications.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    brca1_features = build_brca1_global_feature_cards(repo)
    brca1_variants = build_brca1_variant_cards(repo)
    brca2_hotspots = build_brca2_hotspot_cards(repo)
    brca2_panel = build_brca2_panel_cards(repo)

    paths = [
        (out / "brca1_global_feature_cards.csv", brca1_features),
        (out / "brca1_native_effect_variant_cards.csv", brca1_variants),
        (out / "brca2_hotspot_cards.csv", brca2_hotspots),
        (out / "brca2_panel_variant_cards.csv", brca2_panel),
    ]
    for path, frame in paths:
        frame.to_csv(path, index=False)
        print(f"wrote {path}")
    report = write_report(repo, brca1_features, brca1_variants, brca2_hotspots, brca2_panel)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
