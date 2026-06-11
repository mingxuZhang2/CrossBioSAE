#!/usr/bin/env python
"""Build a submission-facing evidence matrix for local interpretability claims."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


@dataclass
class Claim:
    claim_id: str
    manuscript_claim: str
    dataset: str
    evidence_type: str
    key_result: str
    controls_or_comparators: str
    downstream_use: str
    readiness: str
    missing_for_top_journal: str
    priority_next_action: str
    key_artifacts: str


def fmt(x: Any, digits: int = 3, sci: bool = False) -> str:
    if x is None or pd.isna(x):
        return "NA"
    x = float(x)
    if sci:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_row(df: pd.DataFrame, **filters: Any) -> dict[str, Any]:
    if df.empty:
        return {}
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            return {}
        mask &= df[col].astype(str).eq(str(val))
    if not mask.any():
        return {}
    return df.loc[mask].iloc[0].to_dict()


def artifacts_exist(*paths: Path) -> str:
    parts = []
    for path in paths:
        parts.append(f"{path.name}={'yes' if path.exists() else 'no'}")
    return "; ".join(parts)


def build_claims(repo: Path) -> list[Claim]:
    out = repo / OUT_DIR

    native = first_row(read_csv(out / "brca1_native_finetuned_sae_intervention_summary.csv"))
    native_rescue = first_row(read_csv(out / "brca1_native_finetuned_sae_rescue_summary.csv"))
    raw = first_row(read_csv(out / "brca1_raw_zdna_dim_control_summary.csv"))
    bridge = first_row(read_csv(out / "brca1_native_sae_biological_bridge_summary.csv"))
    global_bridge = read_csv(out / "brca1_global_sae_native_effect_bridge_summary.csv")
    global_auroc = get_row(global_bridge, metric="high_effect_auroc")
    global_spearman = get_row(global_bridge, metric="native_effect_spearman")
    matched_control = read_csv(out / "brca1_native_sae_matched_annotation_control_summary.csv")
    matched_control_row = get_row(
        matched_control,
        comparison="top_decile_native_effect_vs_consequence_region_matched_null",
    )
    covariate_matched = first_row(read_csv(out / "brca1_native_sae_covariate_matched_control_summary.csv"))
    brca2_llr = read_csv(out / "brca2_llr_esm_checkpoint_metric_summary.csv")
    brca2_delta = read_csv(out / "brca2_llr_esm_checkpoint_bootstrap_delta.csv")
    brca2_disc = read_csv(out / "brca2_llr_esm_discordance_category_summary.csv")
    brca2_domain_matched = first_row(read_csv(out / "brca2_discordance_domain_matched_control_summary.csv"))
    brca2_tiers = read_csv(out / "brca2_clinvar_interpretability_candidates_tier_summary.csv")
    brca2_panel = read_csv(out / "brca2_prospective_followup_panel_summary.csv")
    brca2_dossier_arms = read_csv(out / "brca2_review_evidence_dossier_arm_summary.csv")
    brca2_validation = first_row(read_csv(out / "brca2_blinded_validation_status.csv"))
    brca2_proxy_known = read_csv(out / "brca2_external_proxy_validation_known_clinvar.csv")
    brca2_proxy_summary = read_csv(out / "brca2_external_proxy_validation_summary.csv")
    brca2_proxy_tests = read_csv(out / "brca2_external_proxy_validation_panel_proxy_tests.csv")
    brca2_temporal_candidate = first_row(read_csv(out / "brca2_clinvar_temporal_proxy_validation_candidate_tests.csv"))
    brca2_temporal_panel = first_row(read_csv(out / "brca2_clinvar_temporal_proxy_validation_panel_tests.csv"))
    brca2_mavedb_panel_tests = read_csv(out / "brca2_mavedb_external_assay_panel_tests.csv")
    brca2_mavedb_category_tests = read_csv(out / "brca2_mavedb_external_assay_category_tests.csv")
    brca2_hap1_public_endpoints = read_csv(out / "brca2_public_mavedb_hap1_matched_subset_endpoints.csv")
    brca2_fisher = read_csv(out / "brca2_assay_statistical_plan_fisher_thresholds.csv")
    brca2_metrics = read_csv(out / "brca2_sge_metric_summary.csv")
    vus_counts = read_csv(out / "vus_application_readiness_counts.csv")
    collagen_known = first_row(read_csv(out / "collagen_gly_random_feature_null_summary.csv"))
    collagen_vus = first_row(read_csv(out / "collagen_gly_vus_random_feature_summary.csv"))
    collagen_logo = read_csv(out / "collagen_leave_one_gene_out_primary_summary.csv")
    collagen_gene_vus = read_csv(out / "collagen_gene_heldout_vus_validation_summary.csv")
    temporal = first_row(read_csv(out / "clinvar_pseudo_temporal_2025-01_metric_summary.csv"))
    brca2_native = first_row(read_csv(out / "brca2_native_finetuned_sae_intervention_summary.csv"))
    brca2_focused = read_csv(out / "brca2_native_sae_focused_strata_summary.csv")
    brca2_stratum_pilot = read_csv(out / "brca2_native_sae_stratum_control_pilot_summary.csv")
    esol_summary = read_csv(out / "esol_crossmodal_functional_probe_summary.csv")
    esol_split = read_csv(out / "esol_crossmodal_functional_probe_official_split.csv")
    esol_deltas = read_csv(out / "esol_crossmodal_functional_probe_deltas.csv")
    dms_tests = read_csv(out / "dms_clean_functional_validation_tests.csv")
    dms_summary = read_csv(out / "dms_clean_functional_validation_summary.csv")
    bap1_variants = read_csv(out / "bap1_sge_variants.csv")
    bap1_metrics = read_csv(out / "bap1_llr_esm_checkpoint_metric_summary.csv")
    bap1_delta = read_csv(out / "bap1_llr_esm_checkpoint_bootstrap_delta.csv")
    bap1_disc = read_csv(out / "bap1_llr_esm_discordance_category_summary.csv")
    bap1_anchor_tests = read_csv(out / "bap1_sge_external_anchors_tests.csv")
    bap1_baseline_metrics = read_csv(out / "bap1_sge_baseline_predictors_metric_summary.csv")
    rad51c_mapping = first_row(read_csv(out / "rad51c_grch38_mapping_summary.csv"))
    rad51c_anchor_tests = read_csv(out / "rad51c_sge_external_anchors_tests.csv")
    rad51c_metrics = read_csv(out / "rad51c_llr_esm_checkpoint_metric_summary.csv")
    rad51c_delta = read_csv(out / "rad51c_llr_esm_checkpoint_bootstrap_delta.csv")
    rad51c_disc = read_csv(out / "rad51c_llr_esm_discordance_category_summary.csv")
    rad51c_latent = first_row(read_csv(out / "rad51c_latent_sae_intervention_summary.csv"))
    rad51c_latent_strata = read_csv(out / "rad51c_latent_sae_intervention_stratum_summary.csv")
    rad51c_latent_app_arms = read_csv(out / "rad51c_latent_sae_application_arm_summary.csv")

    llr_all = get_row(brca2_llr, scope="all", metric="Evo2 LLR zero-shot")
    llr_missense = get_row(brca2_llr, scope="missense", metric="Evo2 LLR zero-shot")
    llr_nonmissense = get_row(brca2_llr, scope="non_missense", metric="Evo2 LLR zero-shot")
    esm_all = get_row(brca2_llr, scope="all", metric="ESM-only CV logistic")
    fusion_delta = get_row(brca2_delta, comparison="ESM+LLR minus LLR-CV", scope="all")
    both_high = get_row(brca2_disc, category="both_high")
    both_low = get_row(brca2_disc, category="both_low")
    both_high_proxy = get_row(brca2_proxy_known, discordance_category="both_high")
    brca2_tier1_path = get_row(
        brca2_tiers,
        review_tier="tier1_functional_lof_concordant_high",
        review_direction="pathogenic_review",
    )
    brca2_tier1_benign = get_row(
        brca2_tiers,
        review_tier="tier1_functional_benign_concordant_low",
        review_direction="benign_review",
    )
    panel_path = get_row(brca2_panel, panel_arm="prospective_pathogenic_review")
    panel_benign = get_row(brca2_panel, panel_arm="prospective_benign_controls")
    panel_split = get_row(brca2_panel, panel_arm="prospective_split_mechanism_tests")
    dossier_path = get_row(brca2_dossier_arms, panel_arm="prospective_pathogenic_review")
    dossier_benign = get_row(brca2_dossier_arms, panel_arm="prospective_benign_controls")
    fisher_zero = get_row(brca2_fisher, benign_control_lof_count=0)
    vus_tier1 = get_row(vus_counts, group="tier1_collagen_gly")
    panel_proxy_exact = get_row(brca2_proxy_summary, summary_item="prospective_panel_exact_current_clinvar_known")
    panel_proxy_path = get_row(
        brca2_proxy_tests,
        proxy_test="pathogenic_review_enriched_for_same_residue_pathogenic_no_benign",
    )
    mavedb_hap1_panel = get_row(
        brca2_mavedb_panel_tests,
        mavedb_urn="urn:mavedb:00001225-a-1",
        comparison="pathogenic_review_vs_benign_controls",
    )
    mavedb_hap1_both_high = get_row(
        brca2_mavedb_category_tests,
        mavedb_urn="urn:mavedb:00001225-a-1",
        comparison="both_high_vs_both_low",
    )
    mavedb_vc8_both_high = get_row(
        brca2_mavedb_category_tests,
        mavedb_urn="urn:mavedb:00001224-a-1",
        comparison="both_high_vs_both_low",
    )
    hap1_public_primary = get_row(
        brca2_hap1_public_endpoints,
        comparison="pathogenic_review_vs_benign_controls",
    )
    esol_cv_protein = get_row(esol_summary, eval="cv5", rep="Protein-only (ESM2)")
    esol_cv_dna = get_row(esol_summary, eval="cv5", rep="DNA-only (NT)")
    esol_cv_concat = get_row(esol_summary, eval="cv5", rep="Concat")
    esol_split_protein = get_row(esol_summary, eval="official_split", rep="Protein-only (ESM2)")
    esol_split_dna = get_row(esol_summary, eval="official_split", rep="DNA-only (NT)")
    esol_split_concat = get_row(esol_summary, eval="official_split", rep="Concat")
    esol_cv_delta_protein = get_row(esol_deltas, eval="cv5", comparison="Concat minus Protein-only (spearman_mean)")
    esol_split_delta_protein = get_row(
        esol_deltas,
        eval="official_split",
        comparison="Concat minus Protein-only (spearman_mean)",
    )
    esol_split_delta_shuffle = get_row(
        esol_deltas,
        eval="official_split",
        comparison="Concat minus shuffled-DNA (spearman_mean)",
    )
    esol_n = int(esol_split["n_train"].iloc[0] + esol_split["n_test"].iloc[0]) if not esol_split.empty else 0
    dms_group = get_row(dms_tests, test="A_groupcv")
    dms_syn = get_row(dms_tests, test="B_within_synonymous")
    dms_usable = get_row(dms_summary, summary_item="usable_variants")
    dms_proteins = get_row(dms_summary, summary_item="distinct_proteins")
    dms_syn_groups = get_row(dms_summary, summary_item="synonymous_groups_min3")
    dms_syn_variants = get_row(dms_summary, summary_item="synonymous_group_variants")
    bap1_n = len(bap1_variants)
    bap1_binary_n = int(bap1_variants["label"].notna().sum()) if "label" in bap1_variants else 0
    bap1_depleted_n = (
        int(bap1_variants["functional_classification"].eq("depleted").sum())
        if "functional_classification" in bap1_variants
        else 0
    )
    bap1_unchanged_n = (
        int(bap1_variants["functional_classification"].eq("unchanged").sum())
        if "functional_classification" in bap1_variants
        else 0
    )
    bap1_enriched_n = (
        int(bap1_variants["functional_classification"].eq("enriched").sum())
        if "functional_classification" in bap1_variants
        else 0
    )
    bap1_missense_n = int(bap1_variants["is_missense"].astype(str).str.lower().eq("true").sum()) if "is_missense" in bap1_variants else 0
    bap1_all_llr = get_row(bap1_metrics, scope="all", metric="Evo2 LLR zero-shot")
    bap1_all_fusion = get_row(bap1_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    bap1_missense_llr = get_row(bap1_metrics, scope="missense", metric="Evo2 LLR zero-shot")
    bap1_missense_esm = get_row(bap1_metrics, scope="missense", metric="ESM-only CV logistic")
    bap1_missense_fusion = get_row(bap1_metrics, scope="missense", metric="ESM+Evo2 LLR CV logistic")
    bap1_nonmissense_llr = get_row(bap1_metrics, scope="non_missense", metric="Evo2 LLR zero-shot")
    bap1_nonmissense_esm = get_row(bap1_metrics, scope="non_missense", metric="ESM-only CV logistic")
    bap1_all_delta = get_row(bap1_delta, comparison="ESM+LLR minus LLR-CV", scope="all")
    bap1_missense_delta = get_row(bap1_delta, comparison="ESM+LLR minus LLR-CV", scope="missense")
    bap1_both_high = get_row(bap1_disc, category="both_high")
    bap1_both_low = get_row(bap1_disc, category="both_low")
    bap1_clinvar_anchor = get_row(
        bap1_anchor_tests,
        test="clinvar_pathogenic_enriched_for_depleted_vs_benign",
    )
    bap1_gnomad_anchor = get_row(
        bap1_anchor_tests,
        test="gnomad_observed_depleted_less_than_unobserved",
    )
    bap1_eve_baseline = get_row(bap1_baseline_metrics, baseline="EVE score")
    bap1_sift_baseline = get_row(bap1_baseline_metrics, baseline="SIFT deleterious (-score)")
    bap1_polyphen_baseline = get_row(bap1_baseline_metrics, baseline="PolyPhen ordinal damage")
    bap1_llr_note = (
        "BAP1 checkpoint has run on a finite SNV-subset/imputed-LLR design: finite zero-shot LLR rows n="
        + fmt(bap1_all_llr.get("n"), 0)
        + ", LLR AUROC/AUPRC "
        + fmt(bap1_all_llr.get("auroc_for_sge_lof"))
        + "/"
        + fmt(bap1_all_llr.get("auprc_for_sge_lof"))
        + "; ESM+LLR CV all AUROC/AUPRC "
        + fmt(bap1_all_fusion.get("auroc_for_sge_lof"))
        + "/"
        + fmt(bap1_all_fusion.get("auprc_for_sge_lof"))
        + ", delta vs LLR-CV "
        + fmt(bap1_all_delta.get("median_delta_auroc"))
        + " CI ["
        + fmt(bap1_all_delta.get("ci_lo"))
        + ", "
        + fmt(bap1_all_delta.get("ci_hi"))
        + "]; missense LLR "
        + fmt(bap1_missense_llr.get("auroc_for_sge_lof"))
        + ", ESM-only "
        + fmt(bap1_missense_esm.get("auroc_for_sge_lof"))
        + ", ESM+LLR "
        + fmt(bap1_missense_fusion.get("auroc_for_sge_lof"))
        + ", delta "
        + fmt(bap1_missense_delta.get("median_delta_auroc"))
        + " CI ["
        + fmt(bap1_missense_delta.get("ci_lo"))
        + ", "
        + fmt(bap1_missense_delta.get("ci_hi"))
        + "]; non-missense LLR "
        + fmt(bap1_nonmissense_llr.get("auroc_for_sge_lof"))
        + " vs ESM-only "
        + fmt(bap1_nonmissense_esm.get("auroc_for_sge_lof"))
        + "."
        if bap1_all_llr and bap1_all_fusion and bap1_all_delta and bap1_missense_delta
        else "BAP1 data are prepared; Evo2/ESM checkpoint jobs are pending."
    )
    bap1_discordance_note = (
        " BAP1 DNA/protein mechanism strata are strong within finite missense rows: both-high n="
        + fmt(bap1_both_high.get("n"), 0)
        + ", LOF/depleted rate "
        + fmt(bap1_both_high.get("lof_rate"))
        + ", "
        + fmt(bap1_both_high.get("lof_rate_vs_all_missense"))
        + "x missense background, q="
        + fmt(bap1_both_high.get("fisher_q_vs_rest"), sci=True)
        + "; both-low LOF rate "
        + fmt(bap1_both_low.get("lof_rate"))
        + "."
        if bap1_both_high and bap1_both_low
        else ""
    )
    bap1_anchor_note = (
        " External anchors validate the benchmark: ClinVar P/LP depleted "
        + fmt(bap1_clinvar_anchor.get("first_n_depleted"), 0)
        + "/"
        + fmt(bap1_clinvar_anchor.get("first_n_binary"), 0)
        + " vs B/LB "
        + fmt(bap1_clinvar_anchor.get("second_n_depleted"), 0)
        + "/"
        + fmt(bap1_clinvar_anchor.get("second_n_binary"), 0)
        + ", AUROC "
        + fmt(bap1_clinvar_anchor.get("external_lof_auroc"), 4)
        + ", Fisher p="
        + fmt(bap1_clinvar_anchor.get("fisher_p"), sci=True)
        + "; gnomAD-observed depleted "
        + fmt(bap1_gnomad_anchor.get("first_n_depleted"), 0)
        + "/"
        + fmt(bap1_gnomad_anchor.get("first_n_binary"), 0)
        + " vs unobserved "
        + fmt(bap1_gnomad_anchor.get("second_n_depleted"), 0)
        + "/"
        + fmt(bap1_gnomad_anchor.get("second_n_binary"), 0)
        + ", p="
        + fmt(bap1_gnomad_anchor.get("fisher_p"), sci=True)
        + "."
        if bap1_clinvar_anchor and bap1_gnomad_anchor
        else ""
    )
    bap1_baseline_note = (
        " Public missense baselines on BAP1 SGE: EVE AUROC "
        + fmt(bap1_eve_baseline.get("auroc_for_sge_depleted"), 4)
        + ", SIFT AUROC "
        + fmt(bap1_sift_baseline.get("auroc_for_sge_depleted"), 4)
        + ", PolyPhen AUROC "
        + fmt(bap1_polyphen_baseline.get("auroc_for_sge_depleted"), 4)
        + "."
        if bap1_eve_baseline and bap1_sift_baseline and bap1_polyphen_baseline
        else ""
    )
    rad51c_mapping_note = (
        " RAD51C fallback is now GRCh38-checkpoint ready: direct cDNA SNVs mapped n="
        + fmt(rad51c_mapping.get("mapped_ref_match_rows"), 0)
        + " with binary rows n="
        + fmt(rad51c_mapping.get("output_binary_mapped_rows"), 0)
        + ", missense rows n="
        + fmt(rad51c_mapping.get("output_missense_rows"), 0)
        + ", LOF/depleted rows n="
        + fmt(rad51c_mapping.get("output_lof_rows"), 0)
        + "; genome and protein reference checks have 0 mismatches."
        if rad51c_mapping
        else ""
    )
    rad51c_clinvar_anchor = get_row(
        rad51c_anchor_tests,
        test="clinvar_pathogenic_enriched_for_depleted_vs_benign",
    )
    rad51c_nonsense_anchor = get_row(
        rad51c_anchor_tests,
        test="nonsense_enriched_for_depleted_vs_synonymous",
    )
    rad51c_missense_anchor = get_row(
        rad51c_anchor_tests,
        test="missense_enriched_for_depleted_vs_synonymous",
    )
    rad51c_anchor_note = (
        " RAD51C external anchors validate the fallback benchmark: ClinVar P/LP depleted "
        + fmt(rad51c_clinvar_anchor.get("first_n_depleted"), 0)
        + "/"
        + fmt(rad51c_clinvar_anchor.get("first_n_binary"), 0)
        + " vs B/LB "
        + fmt(rad51c_clinvar_anchor.get("second_n_depleted"), 0)
        + "/"
        + fmt(rad51c_clinvar_anchor.get("second_n_binary"), 0)
        + ", AUROC "
        + fmt(rad51c_clinvar_anchor.get("external_lof_auroc"), 4)
        + ", Fisher p="
        + fmt(rad51c_clinvar_anchor.get("fisher_p_greater"), sci=True)
        + "; nonsense depleted "
        + fmt(rad51c_nonsense_anchor.get("first_n_depleted"), 0)
        + "/"
        + fmt(rad51c_nonsense_anchor.get("first_n_binary"), 0)
        + " vs synonymous "
        + fmt(rad51c_nonsense_anchor.get("second_n_depleted"), 0)
        + "/"
        + fmt(rad51c_nonsense_anchor.get("second_n_binary"), 0)
        + ", p="
        + fmt(rad51c_nonsense_anchor.get("fisher_p_greater"), sci=True)
        + "; missense vs synonymous AUROC "
        + fmt(rad51c_missense_anchor.get("external_lof_auroc"), 4)
        + "."
        if rad51c_clinvar_anchor and rad51c_nonsense_anchor and rad51c_missense_anchor
        else ""
    )
    rad51c_all_llr = get_row(rad51c_metrics, scope="all", metric="Evo2 LLR zero-shot")
    rad51c_all_fusion = get_row(rad51c_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    rad51c_missense_llr = get_row(rad51c_metrics, scope="missense", metric="Evo2 LLR zero-shot")
    rad51c_missense_esm = get_row(rad51c_metrics, scope="missense", metric="ESM-only CV logistic")
    rad51c_missense_fusion = get_row(rad51c_metrics, scope="missense", metric="ESM+Evo2 LLR CV logistic")
    rad51c_nonmissense_llr = get_row(rad51c_metrics, scope="non_missense", metric="Evo2 LLR zero-shot")
    rad51c_nonmissense_esm = get_row(rad51c_metrics, scope="non_missense", metric="ESM-only CV logistic")
    rad51c_all_delta = get_row(rad51c_delta, comparison="ESM+LLR minus LLR-CV", scope="all")
    rad51c_missense_delta = get_row(rad51c_delta, comparison="ESM+LLR minus LLR-CV", scope="missense")
    rad51c_both_high = get_row(rad51c_disc, category="both_high")
    rad51c_both_low = get_row(rad51c_disc, category="both_low")
    rad51c_protein_high_dna_low = get_row(rad51c_disc, category="protein_high_dna_low")
    rad51c_latent_missense = get_row(rad51c_latent_strata, stratum="missense")
    rad51c_latent_both_high_arm = get_row(rad51c_latent_app_arms, panel_arm="both_high_sparse_lof_review")
    rad51c_latent_protein_arm = get_row(rad51c_latent_app_arms, panel_arm="protein_high_dna_low_sparse_lof")
    rad51c_checkpoint_note = (
        " RAD51C checkpoint is positive: all LLR AUROC/AUPRC "
        + fmt(rad51c_all_llr.get("auroc_for_sge_lof"))
        + "/"
        + fmt(rad51c_all_llr.get("auprc_for_sge_lof"))
        + " vs ESM+LLR "
        + fmt(rad51c_all_fusion.get("auroc_for_sge_lof"))
        + "/"
        + fmt(rad51c_all_fusion.get("auprc_for_sge_lof"))
        + ", delta AUROC "
        + fmt(rad51c_all_delta.get("median_delta_auroc"))
        + " CI ["
        + fmt(rad51c_all_delta.get("ci_lo"))
        + ", "
        + fmt(rad51c_all_delta.get("ci_hi"))
        + "]; missense LLR "
        + fmt(rad51c_missense_llr.get("auroc_for_sge_lof"))
        + ", ESM-only "
        + fmt(rad51c_missense_esm.get("auroc_for_sge_lof"))
        + ", ESM+LLR "
        + fmt(rad51c_missense_fusion.get("auroc_for_sge_lof"))
        + ", delta "
        + fmt(rad51c_missense_delta.get("median_delta_auroc"))
        + " CI ["
        + fmt(rad51c_missense_delta.get("ci_lo"))
        + ", "
        + fmt(rad51c_missense_delta.get("ci_hi"))
        + "]; non-missense LLR "
        + fmt(rad51c_nonmissense_llr.get("auroc_for_sge_lof"))
        + " vs ESM-only "
        + fmt(rad51c_nonmissense_esm.get("auroc_for_sge_lof"))
        + "."
        if rad51c_all_llr and rad51c_all_fusion and rad51c_all_delta and rad51c_missense_delta
        else " RAD51C checkpoint metrics are pending."
    )
    rad51c_discordance_note = (
        " RAD51C DNA/protein mechanism strata are strong: both-high missense n="
        + fmt(rad51c_both_high.get("n"), 0)
        + ", LOF/depleted rate "
        + fmt(rad51c_both_high.get("lof_rate"))
        + ", "
        + fmt(rad51c_both_high.get("lof_rate_vs_all_missense"))
        + "x missense background, q="
        + fmt(rad51c_both_high.get("fisher_q_vs_rest"), sci=True)
        + "; protein-high/DNA-low n="
        + fmt(rad51c_protein_high_dna_low.get("n"), 0)
        + ", LOF rate "
        + fmt(rad51c_protein_high_dna_low.get("lof_rate"))
        + "; both-low LOF rate "
        + fmt(rad51c_both_low.get("lof_rate"))
        + "."
        if rad51c_both_high and rad51c_both_low
        else ""
    )
    rad51c_latent_note = (
        " RAD51C checkpoint-level latent-SAE intervention is positive: SAE recon AUROC "
        + fmt(rad51c_latent.get("sae_recon_auroc"))
        + " vs top-feature ablation "
        + fmt(rad51c_latent.get("top_feature_ablate_auroc"))
        + ", delta "
        + fmt(rad51c_latent.get("delta_auroc_recon_minus_top_ablate"))
        + "; matched-random p="
        + fmt(rad51c_latent.get("empirical_p_random_delta_ge_top_mean"), 4)
        + "; label-permutation p="
        + fmt(rad51c_latent.get("empirical_p_label_permuted_delta_ge_top_mean"), 4)
        + "; rescue p(delta<=0)="
        + fmt(rad51c_latent.get("empirical_p_rescue_delta_auc_le_zero"), 4)
        + "; missense delta "
        + fmt(rad51c_latent_missense.get("delta_auroc_recon_minus_top_ablate"))
        + ". Application panel arms are filled: both-high sparse LOF selected n="
        + fmt(rad51c_latent_both_high_arm.get("selected_n"), 0)
        + " with LOF rate "
        + fmt(rad51c_latent_both_high_arm.get("lof_rate"))
        + "; protein-high/DNA-low sparse LOF selected n="
        + fmt(rad51c_latent_protein_arm.get("selected_n"), 0)
        + " with LOF rate "
        + fmt(rad51c_latent_protein_arm.get("lof_rate"))
        + "."
        if rad51c_latent and rad51c_latent_missense
        else ""
    )
    rad51c_checkpoint_ready = bool(rad51c_all_llr and rad51c_all_fusion and rad51c_both_high)
    rad51c_latent_ready = bool(rad51c_latent and float(rad51c_latent.get("delta_auroc_recon_minus_top_ablate", 0)) > 0)
    logo_tested = len(collagen_logo)
    logo_passed = 0
    logo_median_auc = float("nan")
    logo_median_delta = float("nan")
    if not collagen_logo.empty:
        logo_passed = int(
            (
                (collagen_logo["observed_auroc"].astype(float) >= 0.60)
                & (collagen_logo["empirical_p_random_auroc_ge_observed"].astype(float) <= 0.05)
            ).sum()
        )
        logo_median_auc = float(collagen_logo["observed_auroc"].median())
        logo_median_delta = float(collagen_logo["observed_minus_random_auroc_mean"].median())
    gene_vus_known_count = 0
    gene_vus_app_pass = 0
    gene_vus_residue_pass = 0
    col3a1_gene_vus = {}
    if not collagen_gene_vus.empty:
        if "known_loo_pass" in collagen_gene_vus.columns:
            known_gene_mask = collagen_gene_vus["known_loo_pass"].astype(str).str.lower().isin({"true", "1"})
        else:
            known_gene_mask = pd.Series(False, index=collagen_gene_vus.index)
        known_gene_vus = collagen_gene_vus.loc[known_gene_mask].copy()
        gene_vus_known_count = len(known_gene_vus)
        if not known_gene_vus.empty:
            gene_vus_app_pass = int((known_gene_vus["application_hook_random_ge_observed_p"].astype(float) <= 0.05).sum())
            gene_vus_residue_pass = int((known_gene_vus["clinvar_residue_hook_random_ge_observed_p"].astype(float) <= 0.05).sum())
        col3a1_gene_vus = get_row(collagen_gene_vus, holdout_gene="COL3A1")
    collagen_gene_vus_note = (
        "; gene-heldout VUS test: known-label-pass genes with VUS "
        + str(gene_vus_known_count)
        + ", application-hook random-null pass "
        + str(gene_vus_app_pass)
        + "/"
        + str(gene_vus_known_count)
        + ", ClinVar-residue-hook random-null pass "
        + str(gene_vus_residue_pass)
        + "/"
        + str(gene_vus_known_count)
        + (
            "; COL3A1 application-hook top "
            + fmt(col3a1_gene_vus.get("application_hook_top_rate"))
            + " vs rest "
            + fmt(col3a1_gene_vus.get("application_hook_rest_rate"))
            + ", random p="
            + fmt(col3a1_gene_vus.get("application_hook_random_ge_observed_p"), 4)
            if col3a1_gene_vus
            else ""
        )
        if not collagen_gene_vus.empty
        else ""
    )
    known_proxy_note = (
        "; exact current ClinVar known-label proxy both-high pathogenic rate "
        + fmt(both_high_proxy.get("pathogenic_rate"))
        + " (n="
        + fmt(both_high_proxy.get("n_exact_known_clinvar"), 0)
        + "), q="
        + fmt(both_high_proxy.get("fisher_q_pathogenic_enrichment_vs_rest"), 4)
        if both_high_proxy
        else ""
    )
    panel_proxy_note = (
        " Proxy layer: exact current ClinVar known outcomes in panel "
        + str(panel_proxy_exact.get("value", "NA"))
        + "; same-residue pathogenic/no-benign support "
        + fmt(panel_proxy_path.get("first_arm_n_proxy_positive"), 0)
        + "/"
        + fmt(panel_proxy_path.get("first_arm_n"), 0)
        + " vs "
        + fmt(panel_proxy_path.get("second_arm_n_proxy_positive"), 0)
        + "/"
        + fmt(panel_proxy_path.get("second_arm_n"), 0)
        + ", Fisher p="
        + fmt(panel_proxy_path.get("fisher_p_greater"), 4)
        + "."
        if panel_proxy_exact and panel_proxy_path
        else ""
    )
    mavedb_category_note = (
        "; public MaveDB external assays support both-high: HAP1 both-high vs both-low AUROC "
        + fmt(mavedb_hap1_both_high.get("external_lof_auroc"))
        + ", p="
        + fmt(mavedb_hap1_both_high.get("mannwhitney_p_greater"), sci=True)
        + "; VC-8 AUROC "
        + fmt(mavedb_vc8_both_high.get("external_lof_auroc"))
        + ", p="
        + fmt(mavedb_vc8_both_high.get("mannwhitney_p_greater"), 4)
        if mavedb_hap1_both_high and mavedb_vc8_both_high
        else ""
    )
    mavedb_panel_note = (
        "; public HAP1 sGE MaveDB external assay endpoint: pathogenic-review arm "
        + fmt(mavedb_hap1_panel.get("n_pathogenic_review_matches"), 0)
        + " matched variants vs benign controls "
        + fmt(mavedb_hap1_panel.get("n_control_matches"), 0)
        + ", external-lof AUROC "
        + fmt(mavedb_hap1_panel.get("external_lof_auroc"))
        + ", p="
        + fmt(mavedb_hap1_panel.get("mannwhitney_p_greater"), sci=True)
        if mavedb_hap1_panel
        else ""
    )
    hap1_public_binary_note = (
        "; public HAP1 matched-subset binary assay-readout endpoint: "
        + fmt(hap1_public_primary.get("positive_binary_lof"), 0)
        + "/"
        + fmt(hap1_public_primary.get("n_positive_group"), 0)
        + " pathogenic-review LOF vs "
        + fmt(hap1_public_primary.get("negative_binary_lof"), 0)
        + "/"
        + fmt(hap1_public_primary.get("n_negative_group"), 0)
        + " benign-control LOF, Fisher p="
        + fmt(hap1_public_primary.get("fisher_p_greater"), 4)
        if hap1_public_primary
        else ""
    )
    validation_status = str(brca2_validation.get("validation_status", "not_run"))
    validation_review_binary = fmt(brca2_validation.get("review_rows_with_binary_classification", 0), 0)
    validation_review_total = fmt(brca2_validation.get("review_rows_total", 0), 0)
    validation_assay_binary = fmt(brca2_validation.get("assay_rows_with_binary_call", 0), 0)
    validation_assay_mean = fmt(brca2_validation.get("assay_rows_with_mean_function_score", 0), 0)
    validation_assay_total = fmt(brca2_validation.get("assay_rows_total", 0), 0)
    validation_note = (
        " Blinded validation analyzer status="
        + validation_status
        + "; review binary outcomes "
        + validation_review_binary
        + "/"
        + validation_review_total
        + "; assay binary outcomes "
        + validation_assay_binary
        + "/"
        + validation_assay_total
        + "; assay mean scores "
        + validation_assay_mean
        + "/"
        + validation_assay_total
        + "."
        if brca2_validation
        else " Blinded validation analyzer has not been run yet."
    )

    full_brca2 = repo / "results/variant/brca2_evo2.npz"
    full_brca2_done = full_brca2.exists()
    brca2_gate_done = (repo / "results/brca2_gate_analysis/brca2_gate_analysis.npz").exists()
    brca2_downstream_done = brca2_gate_done and not brca2_metrics.empty
    brca2_native_done = bool(brca2_native)
    brca2_native_delta = float(brca2_native.get("delta_auroc_recon_minus_top_ablate", float("nan")))
    brca2_native_random_p = float(brca2_native.get("empirical_p_random_delta_ge_top_mean", float("nan")))
    brca2_native_perm_p = float(brca2_native.get("empirical_p_label_permuted_delta_ge_top_mean", float("nan")))
    brca2_native_pass = (
        brca2_native_done
        and brca2_native_delta > 0
        and brca2_native_random_p <= 0.05
        and brca2_native_perm_p <= 0.05
    )
    focused_tested = 0
    focused_screen_positive = 0
    focused_biological_screen_positive = 0
    if not brca2_focused.empty:
        focused_tested = int(brca2_focused["analysis_status"].eq("tested").sum())
        if "passes_focused_screen" in brca2_focused.columns:
            screen = brca2_focused["passes_focused_screen"].astype(str).str.lower().eq("true")
            focused_screen_positive = int(screen.sum())
            broad_names = {"all", "coding"}
            broad_markers = ("not_clinvar_review_candidate", "not_in_prospective_panel")
            focused_biological_screen_positive = int(
                (
                    screen
                    & ~brca2_focused["stratum"].isin(broad_names)
                    & ~brca2_focused["stratum"].astype(str).str.contains("|".join(broad_markers), regex=True)
                ).sum()
            )
    pilot_tested = 0
    pilot_passed = 0
    pilot_note = ""
    if not brca2_stratum_pilot.empty:
        pilot_tested = int((brca2_stratum_pilot["n_tested_folds"].fillna(0).astype(float) > 0).sum())
        pilot_passed = int(
            brca2_stratum_pilot["passes_original_style_gate"].astype(str).str.lower().eq("true").sum()
        )
        pieces = []
        for _, row in brca2_stratum_pilot.iterrows():
            cand = str(row.get("candidate", ""))
            delta = row.get("mean_fold_delta_auc_recon_minus_top_ablate", float("nan"))
            perm_p = row.get("empirical_p_label_permuted_delta_ge_top_mean", float("nan"))
            if pd.notna(delta):
                pieces.append(f"{cand} delta={float(delta):.4g}, label_perm_p={float(perm_p):.4g}")
            else:
                pieces.append(f"{cand} not evaluable")
        pilot_note = "; ".join(pieces)
    if brca2_native_pass:
        c3_gap = "BRCA2 native-SAE intervention is complete; remaining top-journal gap is external review or assay outcome."
        c3_next = "Promote BRCA2 native-SAE to a replication figure and keep clinical claims separate."
    elif brca2_native_done:
        c3_gap = "BRCA2 downstream is complete, but native-SAE did not pass the strict replication gate."
        c3_next = "Use BRCA2 as mechanism stratification/application evidence; focused native-SAE screen did not identify a narrow biological replication claim."
    elif brca2_downstream_done:
        c3_gap = "BRCA2 downstream gate analysis is complete; BRCA2 native-SAE intervention is still pending."
        c3_next = "Run/inspect the BRCA2 native-SAE job and regenerate summary."
    elif full_brca2_done:
        c3_gap = "Downstream gate analysis and BRCA2 native-SAE intervention are still pending."
        c3_next = "Let the queued BRCA2 downstream/native-SAE jobs finish; then regenerate summary."
    else:
        c3_gap = "Full Evo2 embeddings, downstream gate analysis, and BRCA2 native-SAE intervention are still pending."
        c3_next = "Let the queued BRCA2 full Evo2/downstream/native-SAE jobs finish; then regenerate summary."
    c9_gap = (
        "Await downstream gate/fusion artifacts and BRCA2 native-SAE intervention outputs."
        if full_brca2_done
        else "Await full Evo2 embedding, downstream gate/fusion artifacts, and BRCA2 native-SAE intervention outputs."
    )
    c9_next = (
        "Monitor BRCA2 downstream and native-SAE jobs; regenerate evidence matrix when native-SAE outputs appear."
        if full_brca2_done
        else "Monitor SLURM chain and validate brca2_evo2.npz as soon as it appears."
    )
    if brca2_native_pass:
        c9_key = (
            "SAE recon AUROC "
            + fmt(brca2_native.get("native_sae_recon_auroc"))
            + " vs top-feature ablation "
            + fmt(brca2_native.get("top_feature_ablate_auroc"))
            + "; delta "
            + fmt(brca2_native.get("delta_auroc_recon_minus_top_ablate"))
            + "; random p="
            + fmt(brca2_native.get("empirical_p_random_delta_ge_top_mean"), 4)
            + "; label-permutation p="
            + fmt(brca2_native.get("empirical_p_label_permuted_delta_ge_top_mean"), 4)
        )
        c9_readiness = "complete_strong_replication"
        c9_gap = "External blinded review or assay outcome is still required for a clinical-utility claim."
        c9_next = "Promote BRCA2 native-SAE to the decisive replication figure and rerun figures/manuscript package."
        c9_evidence_type = "Native sparse-feature intervention replication"
    elif brca2_native_done:
        c9_key = (
            "BRCA2 native-SAE output exists but did not pass the current replication gate: SAE recon AUROC "
            + fmt(brca2_native.get("native_sae_recon_auroc"))
            + " vs top-feature ablation "
            + fmt(brca2_native.get("top_feature_ablate_auroc"))
            + "; delta "
            + fmt(brca2_native.get("delta_auroc_recon_minus_top_ablate"))
            + "; random p="
            + fmt(brca2_native.get("empirical_p_random_delta_ge_top_mean"), 4)
            + "; label-permutation p="
            + fmt(brca2_native.get("empirical_p_label_permuted_delta_ge_top_mean"), 4)
        )
        if focused_tested:
            c9_key += (
                ". Focused-strata screen tested "
                + str(focused_tested)
                + " strata; "
                + str(focused_screen_positive)
                + " broad strata were screen-positive, but "
                + str(focused_biological_screen_positive)
                + " narrow biological/domain/discordance/review strata were screen-positive."
            )
        if pilot_tested or pilot_note:
            c9_key += (
                " Original-style stratum-control pilot tested "
                + str(pilot_tested)
                + " evaluable candidates; "
                + str(pilot_passed)
                + " passed. "
                + pilot_note
                + "."
            )
        c9_readiness = "mixed_or_negative_replication"
        c9_gap = "BRCA2 native-SAE needs an improved intervention result or external assay/review outcome before it can support replication; current original-style stratum-control pilot did not pass."
        c9_next = "Do not promote C9 as replication; prioritize external review/assay validation or redesign BRCA2 native-SAE rather than adding more descriptive strata."
        c9_evidence_type = "Native sparse-feature intervention did not pass preset gate"
        c9_artifacts = "results/variant/brca2_evo2.npz; brca2_native_finetuned_sae_intervention_summary.csv; brca2_native_sae_focused_strata.md; brca2_native_sae_stratum_control_pilot.md"
    else:
        c9_key = artifacts_exist(full_brca2)
        c9_readiness = "pending_decisive"
        c9_evidence_type = "Pending native-SAE intervention"
        c9_artifacts = "results/variant/brca2_evo2.npz; brca2_native_sae pending outputs"
    if brca2_native_pass:
        c9_artifacts = "results/variant/brca2_evo2.npz; brca2_native_finetuned_sae_intervention_summary.csv"

    claims = [
        Claim(
            claim_id="C1",
            manuscript_claim="Fold-native SAE features are causally necessary for BRCA1 functional prediction.",
            dataset="BRCA1 SGE variants, n=" + fmt(native.get("n"), 0),
            evidence_type="Native sparse-feature intervention",
            key_result=(
                "SAE recon AUROC "
                + fmt(native.get("native_sae_recon_auroc"))
                + " vs top-feature ablation "
                + fmt(native.get("top_feature_ablate_auroc"))
                + "; delta "
                + fmt(native.get("delta_auroc_recon_minus_top_ablate"))
                + "; random p="
                + fmt(native.get("empirical_p_random_delta_ge_top_mean"), 4)
                + "; label-permutation p="
                + fmt(native.get("empirical_p_label_permuted_delta_ge_top_mean"), 4)
                + (
                    "; common-background addback rescue delta AUROC "
                    + fmt(native_rescue.get("mean_rescue_delta_auc_selected_minus_random"))
                    + ", p(delta<=0)="
                    + fmt(native_rescue.get("empirical_p_rescue_delta_auc_le_zero"), 4)
                    + "; selected-feature-only does not beat matched random-only p="
                    + fmt(native_rescue.get("empirical_p_random_feature_only_auc_ge_selected_mean"), 4)
                    if native_rescue
                    else ""
                )
            ),
            controls_or_comparators=(
                "Random matched features; bottom features; label permutation; dose response; common-background addback rescue; feature-only insufficiency boundary; raw z_dna dimension ablation delta "
                + fmt(raw.get("delta_auroc_original_minus_top_raw_dim_ablate"))
            ),
            downstream_use="Main mechanistic proof: explanations change model behavior when ablated.",
            readiness="complete_strong",
            missing_for_top_journal="Core computation and Figure 2 controls are complete; optional 3D structural interface mapping would improve biological presentation.",
            priority_next_action="Use as primary Figure 2/3 evidence and benchmark all replications against this standard.",
            key_artifacts=(
                "brca1_native_finetuned_sae_intervention_summary.csv; "
                "brca1_native_finetuned_sae_rescue_summary.csv; "
                "brca1_raw_zdna_dim_control_summary.csv"
            ),
        ),
        Claim(
            claim_id="C2",
            manuscript_claim="BRCA1 native-SAE effects localize to biology rather than arbitrary latent axes.",
            dataset="BRCA1 SGE variants with consequence/domain labels",
            evidence_type="Biological enrichment and bridge analysis",
            key_result=(
                "Top-decile native-effect variants LOF rate "
                + fmt(bridge.get("lof_rate_high_effect"))
                + " vs rest "
                + fmt(bridge.get("lof_rate_rest"))
                + "; global-SAE bridge high-effect AUROC "
                + fmt(global_auroc.get("global_sae_selected_mean"))
                + " and Spearman "
                + fmt(global_spearman.get("global_sae_selected_mean"))
                + (
                    "; consequence+region matched null LOF rate "
                    + fmt(matched_control_row.get("rest_or_null_mean_lof_rate"))
                    + ", observed-minus-null "
                    + fmt(matched_control_row.get("observed_minus_rest_or_null"))
                    + ", p="
                    + fmt(matched_control_row.get("p_value"), 4)
                    if matched_control_row
                    else ""
                )
                + (
                    "; CADD/phyloP-nearest matched control LOF rate "
                    + fmt(covariate_matched.get("control_lof_rate"))
                    + ", delta "
                    + fmt(covariate_matched.get("delta_lof_rate"))
                    + ", bootstrap CI ["
                    + fmt(covariate_matched.get("delta_bootstrap_ci_low"))
                    + ", "
                    + fmt(covariate_matched.get("delta_bootstrap_ci_high"))
                    + "]"
                    if covariate_matched
                    else ""
                )
            ),
            controls_or_comparators=(
                "Random global features p="
                + fmt(global_auroc.get("empirical_p_random_ge_observed"), 4)
                + "; label-permuted p="
                + fmt(global_auroc.get("empirical_p_permuted_ge_observed"), 4)
                + "; annotation baseline is stronger and should be reported as a comparator; consequence+region and CADD/phyloP covariate-matched native-effect nulls added."
            ),
            downstream_use="Links mechanistic features to variant classes and supports interpretable feature naming.",
            readiness="complete_moderate",
            missing_for_top_journal="Figure-ready domain/protein-coordinate panels and feature-card examples are now present; optional 3D structural interface mapping would further strengthen biological presentation. Keep bridge wording narrower than causal sufficiency.",
            priority_next_action="Use Figure 3 as the primary C2 localization figure; avoid claiming global-SAE sufficiency.",
            key_artifacts=(
                "brca1_native_sae_biological_bridge_summary.csv; "
                "brca1_global_sae_native_effect_bridge_summary.csv; "
                "brca1_native_sae_matched_annotation_control_summary.csv; "
                "brca1_native_sae_covariate_matched_control_summary.csv; "
                "brca1_native_effect_variant_cards.csv; "
                "brca1_global_feature_cards.csv; "
                "fig3_brca1_biological_bridge.png"
            ),
        ),
        Claim(
            claim_id="C3",
            manuscript_claim="BRCA2 provides an independent SGE replication target, but current checkpoint supports mechanism stratification more than generic fusion accuracy.",
            dataset="BRCA2 Sahu et al. SGE variants, n=6198",
            evidence_type="DNA-side Evo2 LLR and protein-side ESM checkpoint",
            key_result=(
                "Evo2 LLR overall AUROC/AUPRC "
                + fmt(llr_all.get("auroc_for_sge_lof"))
                + "/"
                + fmt(llr_all.get("auprc_for_sge_lof"))
                + "; missense "
                + fmt(llr_missense.get("auroc_for_sge_lof"))
                + "/"
                + fmt(llr_missense.get("auprc_for_sge_lof"))
                + "; non-missense "
                + fmt(llr_nonmissense.get("auroc_for_sge_lof"))
                + "/"
                + fmt(llr_nonmissense.get("auprc_for_sge_lof"))
                + "; ESM-only overall AUROC "
                + fmt(esm_all.get("auroc_for_sge_lof"))
            ),
            controls_or_comparators=(
                "ESM+LLR vs LLR-CV delta AUROC "
                + fmt(fusion_delta.get("median_delta_auroc"))
                + " CI ["
                + fmt(fusion_delta.get("ci_lo"))
                + ", "
                + fmt(fusion_delta.get("ci_hi"))
                + "]"
            ),
            downstream_use="Negative predictive-gain control: paper should emphasize interpretation/mechanism, not raw fusion SOTA.",
            readiness="complete_checkpoint",
            missing_for_top_journal=c3_gap,
            priority_next_action=c3_next,
            key_artifacts="brca2_llr_esm_checkpoint_metric_summary.csv; brca2_llr_esm_checkpoint_bootstrap_delta.csv",
        ),
        Claim(
            claim_id="C4",
            manuscript_claim="BRCA2 DNA/protein concordance separates high-risk missense mechanisms from low-risk controls.",
            dataset="Finite BRCA2 missense rows, n=4270",
            evidence_type="Evo2 LLR vs ESM discordance map",
            key_result=(
                "Both-high group n="
                + fmt(both_high.get("n"), 0)
                + ", LOF rate "
                + fmt(both_high.get("lof_rate"))
                + ", "
                + fmt(both_high.get("lof_rate_vs_all_missense"))
                + "x missense background, OR "
                + fmt(both_high.get("fisher_odds_vs_rest"))
                + ", q="
                + fmt(both_high.get("fisher_q_vs_rest"), sci=True)
                + "; both-low LOF rate "
                + fmt(both_low.get("lof_rate"))
                + (
                    "; domain-matched non-both-high null LOF rate "
                    + fmt(brca2_domain_matched.get("matched_control_mean_lof_rate"))
                    + ", delta "
                    + fmt(brca2_domain_matched.get("delta_lof_rate"))
                    + ", bootstrap CI ["
                    + fmt(brca2_domain_matched.get("delta_bootstrap_ci_low"))
                    + ", "
                    + fmt(brca2_domain_matched.get("delta_bootstrap_ci_high"))
                    + "]"
                    if brca2_domain_matched
                    else ""
                )
                + known_proxy_note
                + mavedb_category_note
            ),
            controls_or_comparators="Both-low and discordant quadrants; domain and residue-hotspot summaries; BRCA2-domain matched non-both-high null; public MaveDB HAP1/VC-8 functional assays.",
            downstream_use="Prioritizes BRCA2 VUS review, split-mechanism assay arms, and future native-SAE feature inspection.",
            readiness="complete_strong_checkpoint",
            missing_for_top_journal="Needs full native-SAE causal replication to convert stratification into model-internal mechanism proof.",
            priority_next_action="Use both-high as BRCA2 positive mechanism candidates and discordant groups as stress tests.",
            key_artifacts=(
                "brca2_llr_esm_discordance_category_summary.csv; "
                "brca2_llr_esm_discordance_residue_hotspots.csv; "
                "brca2_external_proxy_validation_known_clinvar.csv; "
                "brca2_discordance_domain_matched_control_summary.csv; "
                "brca2_mavedb_external_assay_validation.md"
            ),
        ),
        Claim(
            claim_id="C5",
            manuscript_claim="BRCA2 interpretability can produce a reviewable VUS triage table, not automatic clinical reclassification.",
            dataset="ClinVar unresolved/conflicting BRCA2 missense candidates",
            evidence_type="ClinVar/SGE/model-concordance candidate stratification",
            key_result=(
                "Tier1 pathogenic-review n="
                + fmt(brca2_tier1_path.get("n"), 0)
                + " with SGE LOF rate "
                + fmt(brca2_tier1_path.get("lof_rate"))
                + "; Tier1 benign-review n="
                + fmt(brca2_tier1_benign.get("n"), 0)
                + " with LOF rate "
                + fmt(brca2_tier1_benign.get("lof_rate"))
                + "; same-residue pathogenic/no-benign hooks "
                + fmt(brca2_tier1_path.get("n_same_residue_known_pathogenic_no_benign"), 0)
                + (
                    "; archive-current old-unresolved tier1 proxy "
                    + fmt(brca2_temporal_candidate.get("positive_current_pathogenic_known_only"), 0)
                    + "/"
                    + fmt(brca2_temporal_candidate.get("n_positive_current_exact_known"), 0)
                    + " vs "
                    + fmt(brca2_temporal_candidate.get("negative_current_pathogenic_known_only"), 0)
                    + "/"
                    + fmt(brca2_temporal_candidate.get("n_negative_current_exact_known"), 0)
                    + " current P/LP, known-only p="
                    + fmt(brca2_temporal_candidate.get("known_only_fisher_p_greater"), 4)
                    + ", missing-as-not-pathogenic p="
                    + fmt(brca2_temporal_candidate.get("missing_as_not_pathogenic_fisher_p_greater"), 4)
                    if brca2_temporal_candidate
                    else ""
                )
            ),
            controls_or_comparators="Benign concordant-low tier, model-high/SGE-benign conflicts, same-residue benign evidence, and archive-current ClinVar pseudo-temporal proxy.",
            downstream_use="Clinical-review queue and assay target selection.",
            readiness="complete_application_triage",
            missing_for_top_journal="Independent manual review outcome or prospective assay results are still required for a clinical-utility claim.",
            priority_next_action="Keep language as triage/review prioritization; do not state reclassification.",
            key_artifacts="brca2_clinvar_interpretability_candidates_tier_summary.csv",
        ),
        Claim(
            claim_id="C6",
            manuscript_claim="A prospective-style BRCA2 assay panel is executable with positive, negative, split-mechanism, and error-analysis arms.",
            dataset="Archived-ClinVar-filtered BRCA2 follow-up panel",
            evidence_type="Prospective panel design and power/statistical plan",
            key_result=(
                "Panel has pathogenic targets n="
                + fmt(panel_path.get("n"), 0)
                + " all SGE LOF; benign controls n="
                + fmt(panel_benign.get("n"), 0)
                + " all SGE functional; split-mechanism tests n="
                + fmt(panel_split.get("n"), 0)
                + ". Dossier exports blinded reviewer packet and internal evidence table; primary targets old-unresolved/absent n="
                + fmt(dossier_path.get("n_old_unresolved_or_absent"), 0)
                + ", benign controls old-unresolved/absent n="
                + fmt(dossier_benign.get("n_old_unresolved_or_absent"), 0)
                + ". If benign controls have 0 LOF, pathogenic arm needs "
                + fmt(fisher_zero.get("min_pathogenic_lof_for_p_lt_0.05"), 0)
                + "/24 for p<0.05 and "
                + fmt(fisher_zero.get("min_pathogenic_lof_for_p_lt_0.01"), 0)
                + "/24 for p<0.01."
                + validation_note
                + panel_proxy_note
                + (
                    "; panel archive-current proxy old-unresolved pathogenic arm current P/LP "
                    + fmt(brca2_temporal_panel.get("positive_current_pathogenic_missing_as_not_pathogenic"), 0)
                    + "/"
                    + fmt(brca2_temporal_panel.get("n_positive_old_unresolved"), 0)
                    + " vs benign arm "
                    + fmt(brca2_temporal_panel.get("negative_current_pathogenic_missing_as_not_pathogenic"), 0)
                    + "/"
                    + fmt(brca2_temporal_panel.get("n_negative_old_unresolved"), 0)
                    + ", p="
                    + fmt(brca2_temporal_panel.get("missing_as_not_pathogenic_fisher_p_greater"), 4)
                    if brca2_temporal_panel
                    else ""
                )
                + mavedb_panel_note
                + hap1_public_binary_note
            ),
            controls_or_comparators="Benign controls and model-conflict controls are included in the same manifest; public HAP1 sGE MaveDB assay is used as an external functional proxy.",
            downstream_use="Wet-lab or expert-review validation plan that can turn interpretation into a top-journal application.",
            readiness="complete_design_not_validated",
            missing_for_top_journal="Actual locked blinded review or assay readout is not available yet; public MaveDB functional support is positive but not a locked blinded clinical endpoint.",
            priority_next_action="Fill the blinded review or assay template with locked external outcomes, rerun analyze_brca2_blinded_review_assay_results.py, and evaluate the one-sided Fisher primary endpoint.",
            key_artifacts=(
                "brca2_prospective_followup_panel_summary.csv; "
                "brca2_review_assay_protocol_manifest.csv; "
                "brca2_assay_statistical_plan_fisher_thresholds.csv; "
                "brca2_review_evidence_dossier.md; "
                "brca2_blinded_validation_status.csv; "
                "brca2_blinded_validation_report.md; "
                "brca2_external_proxy_validation.md; "
                "brca2_clinvar_temporal_proxy_validation.md; "
                "brca2_mavedb_external_assay_validation.md; "
                "brca2_public_mavedb_hap1_assay_readout.md"
            ),
        ),
        Claim(
            claim_id="C7",
            manuscript_claim="Collagen-glycine sparse features show biological specificity on known pathogenic variants but are not yet significant for unsupervised VUS enrichment.",
            dataset="ClinVar/structural-gene VUS and known collagen-glycine pathogenic controls",
            evidence_type="Mechanism-specific feature controls",
            key_result=(
                "Known collagen-gly controls delta mean score "
                + fmt(collagen_known.get("observed_delta_mean_score"))
                + " vs random-feature null p="
                + fmt(collagen_known.get("empirical_p_delta_mean_score"), 4)
                + "; leave-one-gene-out transfer passed "
                + str(logo_passed)
                + "/"
                + str(logo_tested)
                + " collagen genes, median AUROC "
                + fmt(logo_median_auc)
                + ", median observed-minus-random AUROC "
                + fmt(logo_median_delta)
                + "; Tier1 collagen-gly VUS n="
                + fmt(vus_tier1.get("n"), 0)
                + " with actionable review hooks "
                + fmt(vus_tier1.get("n_actionable_review_hooks"), 0)
                + "; VUS set-level enrichment p="
                + fmt(collagen_vus.get("empirical_p_score_enrichment"), 4)
                + collagen_gene_vus_note
            ),
            controls_or_comparators="Matched controls, random feature nulls, leave-one-gene-out transfer, same-residue ClinVar hooks.",
            downstream_use="Secondary VUS case study and review-hook generator.",
            readiness="partial_hypothesis_generating",
            missing_for_top_journal="Needs temporal reclassification, external disease database review, or assay evidence before becoming a central claim.",
            priority_next_action="Use as supplementary application unless stronger external validation is added.",
            key_artifacts=(
                "vus_application_readiness_counts.csv; collagen_gly_random_feature_null_summary.csv; "
                "collagen_gly_vus_random_feature_summary.csv; collagen_leave_one_gene_out.md; "
                "collagen_gene_heldout_vus_validation.md"
            ),
        ),
        Claim(
            claim_id="C8",
            manuscript_claim="ClinVar temporal analysis is useful as an external application pattern, but current broad temporal result is not specific enough to prove CrossBioSAE clinical utility.",
            dataset="Old ClinVar VUS to current known labels",
            evidence_type="Pseudo-temporal reclassification audit",
            key_result=(
                "Old-VUS-to-current-known n="
                + fmt(temporal.get("n"), 0)
                + ", current pathogenic rate "
                + fmt(temporal.get("current_path_rate"))
                + "; novel path score AUROC/AUPRC "
                + fmt(temporal.get("auroc_novel_path_score"))
                + "/"
                + fmt(temporal.get("auprc_novel_path_score"))
            ),
            controls_or_comparators="CADD, ESM-1b, GPN-MSA; collagen-specific strata are too small in current snapshot.",
            downstream_use="Defines the kind of external validation reviewers will expect.",
            readiness="partial_context_only",
            missing_for_top_journal="Need a larger, mechanism-specific temporal holdout or disease-gene-specific evaluation.",
            priority_next_action="Use BRCA2 archived-status panel as the cleaner temporal-style application for now.",
            key_artifacts="clinvar_pseudo_temporal_2025-01_metric_summary.csv",
        ),
        Claim(
            claim_id="C9",
            manuscript_claim=(
                "BRCA2 native-SAE causal replication supports the independent mechanistic claim."
                if brca2_native_pass
                else "BRCA2 native-SAE causal replication is the next decisive experiment."
            ),
            dataset="BRCA2 full Evo2 embeddings and dependent downstream jobs",
            evidence_type=c9_evidence_type,
            key_result=c9_key,
            controls_or_comparators="Should mirror BRCA1: reconstruction preservation, top-feature ablation, random/bottom features, label permutation, raw-dimension control if possible.",
            downstream_use="Turns BRCA2 from checkpoint/application evidence into independent mechanistic replication.",
            readiness=c9_readiness,
            missing_for_top_journal=c9_gap,
            priority_next_action=c9_next,
            key_artifacts=c9_artifacts,
        ),
        Claim(
            claim_id="C10",
            manuscript_claim="eSOL solubility is currently a negative functional-phenotype control for simple cross-modal concatenation.",
            dataset="eSOL verified protein-CDS rows, n=" + (str(esol_n) if esol_n else "NA"),
            evidence_type="Functional phenotype linear probe with shuffled-DNA control",
            key_result=(
                "5-fold CV Spearman protein "
                + fmt(esol_cv_protein.get("spearman_mean"))
                + ", DNA "
                + fmt(esol_cv_dna.get("spearman_mean"))
                + ", concat "
                + fmt(esol_cv_concat.get("spearman_mean"))
                + "; official split protein "
                + fmt(esol_split_protein.get("spearman_mean"))
                + ", DNA "
                + fmt(esol_split_dna.get("spearman_mean"))
                + ", concat "
                + fmt(esol_split_concat.get("spearman_mean"))
                + "; concat-minus-protein Spearman delta CV "
                + fmt(esol_cv_delta_protein.get("delta"))
                + ", official "
                + fmt(esol_split_delta_protein.get("delta"))
            ),
            controls_or_comparators=(
                "Official train/test split plus shuffled-DNA concat control; official concat-minus-shuffled Spearman delta "
                + fmt(esol_split_delta_shuffle.get("delta"))
            ),
            downstream_use="Boundary evidence: do not use eSOL as the current downstream application claim; focus applications on BRCA2 triage and collagen/DMS follow-up.",
            readiness="negative_control_context",
            missing_for_top_journal="A positive eSOL claim would need locked nonlinear GPU probe repeats and feature-level mechanism analysis; current linear result is negative.",
            priority_next_action="Keep as a negative/control assay context; run avGFP DMS clean validation on GPU for a stronger functional-assay method check.",
            key_artifacts=(
                "esol_crossmodal_functional_probe_summary.csv; "
                "esol_crossmodal_functional_probe_deltas.csv; "
                "esol_crossmodal_functional_probe.md"
            ),
        ),
        Claim(
            claim_id="C11",
            manuscript_claim="avGFP DMS provides weak functional-assay method context, but not a strong codon-level interpretability claim.",
            dataset=(
                "avGFP DMS reliable variants, usable n="
                + fmt(dms_usable.get("value"), 0)
                + "; protein groups n="
                + fmt(dms_proteins.get("value"), 0)
            ),
            evidence_type="Reliability-filtered GroupKFold functional assay and synonymous-group leakage test",
            key_result=(
                "GroupKFold Spearman concat "
                + fmt(dms_group.get("concat_spearman"))
                + " vs protein "
                + fmt(dms_group.get("protein_spearman"))
                + " and DNA "
                + fmt(dms_group.get("dna_spearman"))
                + "; concat-minus-protein "
                + fmt(dms_group.get("concat_minus_protein"))
                + "; within-synonymous DNA "
                + fmt(dms_syn.get("dna_spearman"))
                + ", concat "
                + fmt(dms_syn.get("concat_spearman"))
            ),
            controls_or_comparators=(
                "Reliability filter >=2 barcodes; GroupKFold by protein; synonymous groups n="
                + fmt(dms_syn_groups.get("value"), 0)
                + " / variants "
                + fmt(dms_syn_variants.get("value"), 0)
            ),
            downstream_use="Functional-assay method context only; main publishable application remains BRCA2 review/assay triage.",
            readiness="partial_method_support",
            missing_for_top_journal="Needs seed repeats, stronger baseline comparison, and feature-level intervention before becoming a main method claim.",
            priority_next_action="Keep DMS as supplementary method context; do not claim strong codon-level mechanism from the current synonymous test.",
            key_artifacts=(
                "dms_clean_functional_validation_summary.csv; "
                "dms_clean_functional_validation_tests.csv; "
                "dms_clean_functional_validation.md"
            ),
        ),
        Claim(
            claim_id="C12",
            manuscript_claim=(
                "RAD51C provides a positive third-gene checkpoint plus latent-SAE sparse-mechanism screen for variant stratification."
                if rad51c_checkpoint_ready and rad51c_latent_ready
                else "RAD51C provides a positive third-gene Evo2/ESM checkpoint for mechanism-aware variant stratification."
                if rad51c_checkpoint_ready
                else "BAP1/RAD51C are prepared as independent third-gene SGE benchmarks for CrossBioSAE interpretability."
            ),
            dataset=(
                "BAP1 SGE variants, n="
                + str(bap1_n if bap1_n else "NA")
                + "; binary depleted-vs-unchanged n="
                + str(bap1_binary_n if bap1_binary_n else "NA")
                + (
                    "; RAD51C direct-SNV fallback n="
                    + fmt(rad51c_mapping.get("output_binary_mapped_rows"), 0)
                    if rad51c_mapping
                    else ""
                )
            ),
            evidence_type=(
                "Third-gene MaveDB functional map with Evo2/ESM checkpoint, DNA/protein discordance strata, and checkpoint-level latent-SAE intervention."
                if rad51c_checkpoint_ready and rad51c_latent_ready
                else "Third-gene MaveDB functional map with Evo2/ESM checkpoint and DNA/protein discordance strata."
                if rad51c_checkpoint_ready
                else "Public Nature Genetics/MaveDB functional-map preparation; Evo2/ESM checkpoint pending or available."
            ),
            key_result=(
                "Prepared BAP1 SGE table with depleted LOF rows n="
                + str(bap1_depleted_n if bap1_depleted_n else "NA")
                + ", unchanged functional rows n="
                + str(bap1_unchanged_n if bap1_unchanged_n else "NA")
                + ", enriched rows excluded from binary n="
                + str(bap1_enriched_n if bap1_enriched_n else "NA")
                + ", missense rows n="
                + str(bap1_missense_n if bap1_missense_n else "NA")
                + ". "
                + bap1_anchor_note
                + bap1_baseline_note
                + " "
                + bap1_llr_note
                + bap1_discordance_note
                + rad51c_mapping_note
                + rad51c_anchor_note
                + rad51c_checkpoint_note
                + rad51c_discordance_note
                + rad51c_latent_note
            ),
            controls_or_comparators="Binary benchmark excludes enriched variants; ClinVar B/LB, gnomAD-observed variants, and synonymous/UTR rows serve as negative controls; EVE/SIFT/PolyPhen are public missense baselines for BAP1; RAD51C has current ClinVar exact-match and consequence-class anchors but still uses direct cDNA SNVs only and excludes delins/intronic edits; GroupKFold should be by genomic position; compare Evo2 LLR, ESM, and ESM+LLR before any native-SAE claim.",
            downstream_use="Third independent cancer-gene functional-map benchmark; can test whether mechanism/application conclusions generalize beyond BRCA1/BRCA2.",
            readiness=(
                "complete_latent_sae_checkpoint_screen"
                if rad51c_checkpoint_ready and rad51c_latent_ready
                else "complete_checkpoint_third_gene"
                if rad51c_checkpoint_ready
                else "pending_third_gene_benchmark"
            ),
            missing_for_top_journal=(
                "RAD51C checkpoint and checkpoint-level latent-SAE sparse-feature intervention are positive, but top-journal causal interpretability replication still needs the stricter native gate-SAE ablation/rescue chain, with missense-only z_prot as the primary RAD51C native-SAE check; BAP1 must be described as finite-SNV/imputed-LLR checkpoint evidence because many BAP1 variants are indel or multibase rows with NaN zero-shot LLR."
                if rad51c_checkpoint_ready and rad51c_latent_ready
                else "RAD51C and BAP1 checkpoint/mechanism strata are positive, but a top-journal causal interpretability replication still needs native-SAE ablation/rescue with matched random and label-permutation controls; BAP1 must be described as finite-SNV/imputed-LLR checkpoint evidence because many BAP1 variants are indel or multibase rows with NaN zero-shot LLR."
                if rad51c_checkpoint_ready
                else "Needs completed BAP1 or RAD51C Evo2/ESM/CrossBioSAE scores and preferably feature-level intervention or mechanism localization before it can be promoted; external benchmark anchors are now available for both BAP1 and RAD51C."
            ),
            priority_next_action=(
                "Promote RAD51C as the clean third-gene checkpoint plus latent-SAE mechanism-screen result; use rad51c_latent_sae_application_panel.csv for review/assay planning while waiting for the stricter full-native RAD51C SAE SLURM chain, including the missense-only z_prot run."
                if rad51c_checkpoint_ready and rad51c_latent_ready
                else "Promote RAD51C as the clean third-gene checkpoint/mechanism-stratification result and BAP1 as a finite-SNV checkpoint/boundary result; inspect the submitted RAD51C latent-SAE/full-native-SAE SLURM chain when complete, then use passing both-high/native-SAE strata for locked review/assay endpoints."
                if rad51c_checkpoint_ready
                else "Wait for BAP1/RAD51C Evo2 LLR SLURM jobs and run the dependent checkpoints; then decide whether third-gene evidence is positive validation or a boundary result."
            ),
            key_artifacts=(
                "bap1_sge_data_preparation.md; "
                "bap1_sge_external_anchors.md; "
                "bap1_sge_baseline_predictors.md; "
                "third_gene_interpretability_application_plan.md; "
                "third_gene_application_decision_gates.csv; "
                "third_gene_application_blueprint.csv; "
                "third_gene_review_assay_panel.md; "
                "third_gene_review_assay_panel_blinded.csv; "
                "third_gene_review_assay_panel_assay_template.csv; "
                "third_gene_assay_statistical_plan.md; "
                "third_gene_assay_statistical_thresholds.csv; "
                "bap1_sge_variants.csv; "
                "bap1_llr_esm_checkpoint.md; "
                "bap1_llr_esm_checkpoint_metric_summary.csv; "
                "bap1_llr_esm_checkpoint_bootstrap_delta.csv; "
                "bap1_llr_esm_discordance.md; "
                "bap1_llr_esm_discordance_category_summary.csv; "
                "data/variant/bap1/bap1_variants.csv; "
                "scripts/slurm_bap1_evo2_llr.sh; "
                "scripts/slurm_bap1_esm.sh; "
                "scripts/slurm_bap1_checkpoint.sh; "
                "rad51c_grch38_mapping.md; "
                "rad51c_sge_external_anchors.md; "
                "rad51c_llr_esm_checkpoint.md; "
                "rad51c_llr_esm_checkpoint_metric_summary.csv; "
                "rad51c_llr_esm_checkpoint_bootstrap_delta.csv; "
                "rad51c_llr_esm_discordance.md; "
                "rad51c_llr_esm_discordance_category_summary.csv; "
                "scripts/run_third_gene_latent_sae_intervention.py; "
                "scripts/slurm_rad51c_latent_sae.sh; "
                "scripts/slurm_rad51c_evo2_full.sh; "
                "scripts/slurm_rad51c_gate_analysis.sh; "
                "scripts/slurm_rad51c_native_sae.sh; "
                "scripts/slurm_rad51c_native_sae_missense.sh; "
                "rad51c_latent_sae_intervention_summary.csv; "
                "rad51c_latent_sae_application.md; "
                "rad51c_latent_sae_application_panel.csv; "
                "rad51c_latent_sae_application_arm_summary.csv; "
                "rad51c_latent_sae_application_utility_tests.csv; "
                "rad51c_latent_sae_intervention_smoke_summary.csv; "
                "data/variant/rad51c/rad51c_grch38_variants.csv; "
                "scripts/slurm_rad51c_evo2_llr.sh; "
                "scripts/slurm_rad51c_esm.sh; "
                "scripts/slurm_rad51c_checkpoint.sh"
            ),
        ),
    ]
    return claims


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = []
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows.extend([header, sep])
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            text = str(row[col]).replace("\n", " ").replace("|", "\\|")
            vals.append(text)
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    ready_order = [
        "complete_strong",
        "complete_strong_replication",
        "complete_strong_checkpoint",
        "complete_moderate",
        "complete_checkpoint",
        "complete_latent_sae_checkpoint_screen",
        "complete_checkpoint_third_gene",
        "complete_application_triage",
        "complete_design_not_validated",
        "partial_method_support",
        "partial_hypothesis_generating",
        "partial_context_only",
        "negative_control_context",
        "pending_decisive",
        "mixed_or_negative_replication",
    ]
    counts = df["readiness"].value_counts().reindex(ready_order).dropna().astype(int)
    complete_mask = df["readiness"].str.startswith("complete")
    pending_mask = df["readiness"].str.contains("pending|partial|mixed|negative", regex=True)
    c9 = df.loc[df["claim_id"].eq("C9")]
    full_brca2_done = False
    c9_mixed = False
    if not c9.empty:
        full_brca2_done = "brca2_evo2.npz=yes" in str(c9.iloc[0]["key_result"])
        c9_mixed = "mixed" in str(c9.iloc[0]["readiness"]) or "negative" in str(c9.iloc[0]["readiness"])
    decisive_missing = (
        "BRCA2 native-SAE did not pass the strict replication gate; the fixed-prediction focused screen did not identify a narrow biological rescue, so an original-style per-stratum control rerun, improved intervention, or external assay/review evidence is needed."
        if c9_mixed
        else (
        "BRCA2 native-SAE intervention after the downstream gate/fusion artifacts finish."
        if full_brca2_done
        else "BRCA2 full native-SAE intervention after the full Evo2 embedding job finishes."
        )
    )

    lines = [
        "# CrossBioSAE Interpretability Submission Evidence Matrix",
        "",
        "## Purpose",
        "",
        "This report maps each publishable interpretability claim to the exact local evidence, controls, downstream application, and remaining top-journal gap. It is intentionally stricter than a results summary: a claim is marked ready only when the current artifacts support the scope of the statement.",
        "",
        "## Submission Stance",
        "",
        "- Current strongest story: BRCA1 native-SAE causal necessity plus BRCA2 independent mechanism stratification and VUS/assay-panel design.",
        "- Current non-story: generic DNA+protein fusion SOTA. The BRCA2 checkpoint shows no overall gain from simple ESM+LLR fusion over Evo2 LLR.",
        "- Current decisive missing piece: " + decisive_missing,
        "",
        "## Readiness Counts",
        "",
    ]
    for readiness, count in counts.items():
        lines.append(f"- {readiness}: {count}")
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            markdown_table(
                df,
                [
                    "claim_id",
                    "readiness",
                    "manuscript_claim",
                    "dataset",
                    "evidence_type",
                    "key_result",
                    "missing_for_top_journal",
                    "priority_next_action",
                ],
            ),
            "",
            "## Complete Claims That Can Anchor The Paper",
            "",
        ]
    )
    for _, row in df.loc[complete_mask].iterrows():
        lines.append(f"- {row['claim_id']}: {row['manuscript_claim']}")
    lines.extend(["", "## Incomplete Or Context-Only Claims", ""])
    for _, row in df.loc[pending_mask].iterrows():
        lines.append(f"- {row['claim_id']}: {row['missing_for_top_journal']}")
    lines.extend(
        [
            "",
            "## Top-Journal Gate",
            "",
            "The paper is strongest if the main claims are limited to what has hard evidence now: BRCA1 causal SAE intervention, BRCA2 checkpoint mechanism stratification, and BRCA2 review/assay design. A full BRCA2 native-SAE mechanistic replication should wait for an original-style per-stratum control rerun, improved intervention result, or external assay/review outcome; a clinical-utility claim should wait for blinded review, temporal validation, or assay outcomes.",
            "",
            "## Artifact Index",
            "",
            markdown_table(df, ["claim_id", "controls_or_comparators", "downstream_use", "key_artifacts"]),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    claims = build_claims(repo)
    df = pd.DataFrame([asdict(c) for c in claims])
    csv_path = out / "interpretability_submission_evidence_matrix.csv"
    md_path = out / "interpretability_submission_evidence_matrix.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
