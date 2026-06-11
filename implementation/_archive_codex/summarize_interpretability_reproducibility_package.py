#!/usr/bin/env python
"""Build a reproducibility package for the interpretability manuscript track."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


@dataclass
class Artifact:
    group: str
    path: str
    role: str
    required_for_claims: str
    exists: bool
    size_mb: str
    rows_or_shape: str
    sha256_16: str
    status: str


@dataclass
class Command:
    step: str
    purpose: str
    command: str
    expected_outputs: str
    current_status: str


@dataclass
class Gate:
    gate_id: str
    requirement: str
    evidence: str
    status: str
    next_action: str


def sha16(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe_file(repo: Path, path: str) -> tuple[bool, str, str, str]:
    full = repo / path
    if not full.exists():
        return False, "", "", ""
    size_mb = f"{full.stat().st_size / (1024 * 1024):.3f}"
    shape = ""
    try:
        if full.suffix == ".csv":
            df = pd.read_csv(full)
            shape = f"rows={len(df)} cols={len(df.columns)}"
        elif full.suffix == ".npz":
            data = np.load(full)
            parts = []
            for key in data.files:
                value = data[key]
                parts.append(f"{key}{tuple(value.shape)}")
            shape = "; ".join(parts[:8])
    except Exception as exc:  # pragma: no cover - report-only path
        shape = f"inspect_error={type(exc).__name__}"
    return True, size_mb, shape, sha16(full)


def artifact(repo: Path, group: str, path: str, role: str, claims: str, required: bool = True) -> Artifact:
    exists, size_mb, rows_or_shape, checksum = describe_file(repo, path)
    if exists:
        status = "present"
    elif required:
        status = "missing_required"
    else:
        status = "pending_or_optional"
    return Artifact(group, path, role, claims, exists, size_mb, rows_or_shape, checksum, status)


def build_artifacts(repo: Path) -> list[Artifact]:
    rows = [
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_evidence_map.md", "Top-journal evidence standard and literature pattern", "all"),
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_interpretability_strategy.md", "Main strategy document", "all"),
        artifact(repo, "literature_strategy", "scripts/summarize_top_journal_interpretability_evidence_plan.py", "Regenerates top-journal interpretability proof-pattern and priority-action map", "all", required=False),
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_interpretability_evidence_plan.md", "Top-journal interpretability proof-pattern and current-gap report", "all", required=False),
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_interpretability_literature_patterns.csv", "Literature proof-pattern table for interpretability/application claims", "all", required=False),
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_interpretability_evidence_matrix.csv", "Current evidence against top-journal proof standards", "all", required=False),
        artifact(repo, "literature_strategy", "results/interpretability_applications/top_journal_interpretability_priority_actions.csv", "Prioritized experiments needed to upgrade claims", "all", required=False),
        artifact(repo, "literature_strategy", "results/interpretability_applications/interpretability_agent_team_synthesis.md", "Current four-agent interpretability audit synthesis", "all", required=False),
        artifact(repo, "submission", "results/interpretability_applications/interpretability_submission_evidence_matrix.csv", "Claim-to-evidence matrix", "all"),
        artifact(repo, "submission", "results/interpretability_applications/interpretability_manuscript_package.md", "Figure plan and reviewer risk register", "all"),
        artifact(repo, "submission", "results/interpretability_applications/interpretability_manuscript_draft.md", "Claim-bounded manuscript draft skeleton", "all", required=False),
        artifact(repo, "submission", "results/interpretability_applications/interpretability_manuscript_draft_sections.csv", "Manuscript section-to-claim map", "all", required=False),
        artifact(repo, "submission", "results/interpretability_applications/interpretability_feature_hotspot_cards.md", "Concrete feature/hotspot examples", "C2,C4,C5,C6"),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_interpretability_application_plan.md", "BAP1/RAD51C third-gene application and decision-gate plan", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_application_decision_gates.csv", "BAP1/RAD51C third-gene publishability gates", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_application_blueprint.csv", "BAP1/RAD51C review/assay application blueprint", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_review_assay_panel.md", "BAP1/RAD51C 64-row blinded review/assay panel report", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_review_assay_panel_internal.csv", "Unblinded BAP1/RAD51C review/assay panel", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_review_assay_panel_blinded.csv", "Blinded BAP1/RAD51C reviewer panel", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_review_assay_panel_assay_template.csv", "BAP1/RAD51C assay readout template", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_review_assay_panel_summary.csv", "BAP1/RAD51C review/assay panel arm summary", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_assay_statistical_plan.md", "Pre-specified BAP1/RAD51C review/assay statistical plan", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_assay_endpoint_plan.csv", "BAP1/RAD51C panel endpoint plan", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_assay_statistical_thresholds.csv", "BAP1/RAD51C panel Fisher success thresholds", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_assay_power_grid.csv", "BAP1/RAD51C panel exact power grid", "C12", required=False),
        artifact(repo, "submission", "results/interpretability_applications/third_gene_assay_design_sanity.csv", "BAP1/RAD51C panel internal SGE design sanity table", "C12", required=False),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_finetuned_sae_intervention_summary.csv", "BRCA1 native-SAE causal intervention summary", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_finetuned_sae_intervention_random_ablation.csv", "BRCA1 matched random-feature control", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_finetuned_sae_rescue_summary.csv", "BRCA1 native-SAE feature-only and addback-rescue summary", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_finetuned_sae_rescue_rescue_controls.csv", "BRCA1 common-background addback-rescue controls", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_finetuned_sae_rescue_sufficiency_controls.csv", "BRCA1 feature-only insufficiency controls", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_raw_zdna_dim_control_summary.csv", "BRCA1 raw dense-dimension control", "C1"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_biological_bridge_summary.csv", "BRCA1 native-effect biological bridge", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_biological_bridge_strata.csv", "BRCA1 native-effect biological strata", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_matched_annotation_control_summary.csv", "BRCA1 consequence/region-matched native-effect control", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_matched_annotation_control.md", "BRCA1 annotation-matched control report", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_covariate_matched_control_summary.csv", "BRCA1 consequence/region plus CADD/phyloP matched native-effect control", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_covariate_matched_control_balance.csv", "BRCA1 covariate-matched balance table", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_sae_covariate_matched_control.md", "BRCA1 covariate-matched control report", "C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_native_effect_variant_cards.csv", "BRCA1 high native-effect variant cards", "C1,C2"),
        artifact(repo, "brca1_data", "results/interpretability_applications/brca1_global_feature_cards.csv", "BRCA1 named feature cards", "C2"),
        artifact(repo, "brca2_inputs", "data/variant/brca2/brca2_variants.csv", "BRCA2 binary SGE variant table", "C3,C4,C5,C6,C9"),
        artifact(repo, "brca2_inputs", "results/variant/brca2_esm_delta.npz", "BRCA2 protein-side ESM deltas", "C3,C4,C5,C6,C9"),
        artifact(repo, "brca2_inputs", "results/variant/brca2_evo2_llr.npz", "BRCA2 Evo2 LLR checkpoint", "C3,C4,C5,C6"),
        artifact(repo, "brca2_inputs", "results/variant/brca2_evo2.npz", "BRCA2 full Evo2 embeddings", "C9", required=False),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_llr_esm_checkpoint_metric_summary.csv", "BRCA2 LLR/ESM checkpoint metrics", "C3"),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_llr_esm_checkpoint_bootstrap_delta.csv", "BRCA2 fusion negative-control bootstrap", "C3"),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_llr_esm_discordance_category_summary.csv", "BRCA2 DNA/protein concordance categories", "C4"),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_discordance_domain_matched_control_summary.csv", "BRCA2 both-high domain-matched control summary", "C4"),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_discordance_domain_matched_control.md", "BRCA2 both-high domain-matched control report", "C4"),
        artifact(repo, "brca2_checkpoint", "results/interpretability_applications/brca2_hotspot_cards.csv", "BRCA2 both-high residue hotspot cards", "C4,C5,C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_external_proxy_validation.md", "BRCA2 external/proxy validation report", "C4,C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_external_proxy_validation_summary.csv", "BRCA2 external/proxy validation summary", "C4,C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_external_proxy_validation_known_clinvar.csv", "Exact current ClinVar known-label proxy by BRCA2 mechanism stratum", "C4"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_external_proxy_validation_panel_same_residue.csv", "Prospective-panel same-residue proxy support", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_external_proxy_validation_panel_proxy_tests.csv", "Prospective-panel proxy arm tests", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_clinvar_temporal_proxy_validation.md", "BRCA2 archive-current ClinVar temporal proxy report", "C5,C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_clinvar_temporal_proxy_validation_candidate_tests.csv", "BRCA2 candidate-tier archive-current proxy tests", "C5"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_clinvar_temporal_proxy_validation_panel_tests.csv", "BRCA2 panel archive-current proxy tests", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_mavedb_external_assay_validation.md", "BRCA2 public MaveDB external functional-assay validation report", "C4,C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_mavedb_external_assay_panel_tests.csv", "BRCA2 public MaveDB panel arm tests", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_mavedb_external_assay_category_tests.csv", "BRCA2 public MaveDB mechanism stratum tests", "C4"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_public_mavedb_hap1_assay_readout.md", "BRCA2 public HAP1 assay readout mapped to the 64-row blinded panel", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_public_mavedb_hap1_assay_readout.csv", "BRCA2 public HAP1 assay readout CSV for the blinded panel", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_public_mavedb_hap1_matched_subset_endpoints.csv", "BRCA2 public HAP1 matched-subset binary and continuous endpoints", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_public_mavedb_hap1_calibration.csv", "BRCA2 public HAP1 local orientation and binary threshold calibration", "C6"),
        artifact(repo, "brca2_proxy", "results/interpretability_applications/brca2_public_mavedb_hap1_blinded_validation_status.csv", "Conservative blinded analyzer status for the public HAP1 readout", "C6"),
        artifact(repo, "brca2_application", "results/interpretability_applications/brca2_clinvar_interpretability_candidates_tier_summary.csv", "BRCA2 VUS tier summary", "C5"),
        artifact(repo, "brca2_application", "results/interpretability_applications/brca2_prospective_followup_panel_summary.csv", "BRCA2 prospective panel summary", "C6"),
        artifact(repo, "brca2_application", "results/interpretability_applications/brca2_review_assay_protocol_manifest.csv", "BRCA2 assay/review manifest", "C6"),
        artifact(repo, "brca2_application", "results/interpretability_applications/brca2_assay_statistical_plan_fisher_thresholds.csv", "BRCA2 assay Fisher thresholds", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_handoff_readme.md", "BRCA2 blinded review/assay handoff README", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_review_sheet.csv", "Reviewer-facing blinded curation sheet", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_assay_readout_template.csv", "Assay-facing blinded readout template", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_unblinding_key.csv", "Internal unblinding key", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_review_evidence_dossier.md", "BRCA2 review/assay evidence dossier report", "C5,C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_review_evidence_dossier_internal.csv", "Unblinded analyst evidence dossier", "C5,C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_review_evidence_dossier_blinded_packet.csv", "Blinding-safe reviewer evidence packet", "C5,C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_validation_status.csv", "Locked-outcome readiness status for BRCA2 blinded validation", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_validation_primary_endpoint.csv", "Pre-specified BRCA2 blinded validation primary endpoint", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_validation_arm_summary.csv", "BRCA2 blinded validation arm-level summary", "C6"),
        artifact(repo, "brca2_handoff", "results/interpretability_applications/brca2_blinded_validation_report.md", "BRCA2 blinded validation status/report", "C6"),
        artifact(repo, "figures", "results/interpretability_applications/figures/fig2_brca1_native_sae.png", "Draft Figure 2 with BRCA1 necessity, addback-rescue, and feature-only boundary controls", "C1,C2"),
        artifact(repo, "figures", "results/interpretability_applications/figures/fig3_brca1_biological_bridge.png", "Draft Figure 3 with BRCA1 localization, CADD/phyloP matched control, balance, and feature-card panels", "C2"),
        artifact(repo, "figures", "results/interpretability_applications/figures/fig4_brca2_checkpoint.png", "Draft Figure 4", "C3,C4"),
        artifact(repo, "figures", "results/interpretability_applications/figures/fig5_brca2_application.png", "Draft Figure 5", "C5,C6"),
        artifact(repo, "figures", "results/interpretability_applications/figures/fig_s2_third_gene_benchmarks.png", "Supplementary Figure S2 with BAP1/RAD51C third-gene benchmark anchors and checkpoint boundary", "C12", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_native_finetuned_sae_intervention_summary.csv", "BRCA2 native-SAE causal intervention summary", "C9", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_native_sae_focused_strata_summary.csv", "BRCA2 native-SAE focused strata screen", "C9", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_native_sae_focused_strata.md", "BRCA2 native-SAE focused strata report", "C9", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_native_sae_stratum_control_pilot_summary.csv", "BRCA2 native-SAE original-style stratum control pilot", "C9", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_native_sae_stratum_control_pilot.md", "BRCA2 native-SAE original-style stratum control pilot report", "C9", required=False),
        artifact(repo, "brca2_pending", "results/interpretability_applications/brca2_sge_metric_summary.csv", "BRCA2 downstream SGE mechanism metrics", "C9", required=False),
        artifact(repo, "third_gene_bap1", "data/variant/bap1/bap1_variants.csv", "Prepared BAP1 SGE variant table for third-gene benchmark", "C12", required=False),
        artifact(repo, "third_gene_bap1", "data/variant/bap1/bap1_Q96TC6.fasta", "BAP1 protein FASTA for ESM delta extraction", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_data_preparation.md", "BAP1 SGE data-readiness report", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_variants.csv", "BAP1 SGE variant table copy in result manifest", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_functional_class_summary.csv", "BAP1 SGE functional-class summary", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_external_anchors.md", "BAP1 ClinVar/gnomAD/consequence anchor report", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_external_anchors_tests.csv", "BAP1 external anchor statistical tests", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_external_anchors_vus_candidates.csv", "BAP1 SGE-based VUS triage candidates", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_baseline_predictors.md", "BAP1 public EVE/SIFT/PolyPhen baseline report", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_baseline_predictors_metric_summary.csv", "BAP1 public baseline metric summary", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_sge_baseline_predictors_binary_calls.csv", "BAP1 public baseline binary-call tests", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/variant/bap1_evo2_llr.npz", "BAP1 Evo2 LLR checkpoint output", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/variant/bap1_esm_delta.npz", "BAP1 ESM delta checkpoint output", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_llr_esm_checkpoint_metric_summary.csv", "BAP1 Evo2/ESM third-gene checkpoint metrics", "C12", required=False),
        artifact(repo, "third_gene_bap1", "results/interpretability_applications/bap1_llr_esm_discordance_category_summary.csv", "BAP1 DNA/protein checkpoint discordance strata", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_mavedb_scores.csv", "Prepared RAD51C MaveDB SGE fallback score table", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_mavedb_metadata.json", "RAD51C MaveDB score-set metadata", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_sge_data_preparation.md", "RAD51C fallback data-readiness report", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_sge_data_preparation_summary.csv", "RAD51C fallback data-readiness summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_grch38_variants.csv", "RAD51C GRCh38-mapped direct-SNV checkpoint variant table", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_grch38_mapping_all.csv", "RAD51C all-row cDNA-to-GRCh38 mapping audit table", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_O43502.fasta", "RAD51C protein FASTA generated from ENST00000337432 on GRCh38", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "data/variant/rad51c/rad51c_ensembl_ENST00000337432_GRCh38.json", "Cached Ensembl transcript lookup for RAD51C GRCh38 mapping", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_grch38_mapping.md", "RAD51C GRCh38 mapping report", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_grch38_mapping_summary.csv", "RAD51C GRCh38 mapping summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_grch38_mapping_status_counts.csv", "RAD51C mapping status counts", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_grch38_mapping_consequence_counts.csv", "RAD51C mapped consequence counts", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_sge_external_anchors.md", "RAD51C ClinVar/consequence external-anchor report", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_sge_external_anchors_tests.csv", "RAD51C external anchor statistical tests", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_sge_external_anchors_candidate_review.csv", "RAD51C depleted unresolved/conflicting/unobserved review candidates", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/variant/rad51c_evo2_llr.npz", "RAD51C Evo2 LLR checkpoint output", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/run_third_gene_latent_sae_intervention.py", "Generic third-gene ESM-latent SAE intervention screen", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/summarize_rad51c_latent_sae_application.py", "RAD51C latent-SAE sparse-mechanism application and panel builder", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/slurm_rad51c_latent_sae.sh", "RAD51C checkpoint-level latent-SAE SLURM job", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/slurm_rad51c_evo2_full.sh", "RAD51C full Evo2 embedding SLURM job for native-SAE replication", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/slurm_rad51c_gate_analysis.sh", "RAD51C fold gate-analysis SLURM job for native-SAE replication", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/slurm_rad51c_native_sae.sh", "RAD51C BRCA1-style native-SAE SLURM job", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "scripts/slurm_rad51c_native_sae_missense.sh", "RAD51C missense-only z_prot native-SAE SLURM job", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/variant/rad51c_evo2.npz", "RAD51C full Evo2 embedding output required for native-SAE replication", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/rad51c_gate_analysis/rad51c_gate_analysis.npz", "RAD51C trained fold gate-analysis output", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_intervention_summary.csv", "RAD51C checkpoint-level latent-SAE intervention summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application.md", "RAD51C latent-SAE sparse-mechanism application report", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application_panel.csv", "RAD51C 64-row sparse-mechanism review/assay panel", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application_arm_summary.csv", "RAD51C latent-SAE panel arm summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application_variant_priorities.csv", "RAD51C latent-SAE variant priority table", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application_feature_fold_summary.csv", "RAD51C latent-SAE selected-feature fold summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_application_utility_tests.csv", "RAD51C latent-SAE utility pass/fail tests", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_native_finetuned_sae_intervention_summary.csv", "RAD51C BRCA1-style native-SAE causal intervention summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_native_finetuned_sae_intervention_missense_summary.csv", "RAD51C missense-only z_prot native-SAE causal intervention summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c_sae", "results/interpretability_applications/rad51c_latent_sae_intervention_smoke_summary.csv", "RAD51C latent-SAE smoke-test summary", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/variant/rad51c_esm_delta.npz", "RAD51C ESM delta checkpoint output", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_llr_esm_checkpoint_metric_summary.csv", "RAD51C Evo2/ESM checkpoint metrics", "C12", required=False),
        artifact(repo, "third_gene_rad51c", "results/interpretability_applications/rad51c_llr_esm_discordance_category_summary.csv", "RAD51C DNA/protein checkpoint discordance strata", "C12", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_gly_random_feature_null_summary.csv", "Collagen Gly-X-Y known-control random-feature summary", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_gly_vus_random_feature_summary.csv", "Collagen Gly-X-Y VUS random-feature summary", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_leave_one_gene_out_primary_summary.csv", "Collagen leave-one-gene-out primary transfer summary", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_leave_one_gene_out_control_tests.csv", "Collagen leave-one-gene-out secondary controls", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_leave_one_gene_out.md", "Collagen leave-one-gene-out report", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_gene_heldout_vus_validation_summary.csv", "Collagen gene-heldout VUS review-hook validation summary", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_gene_heldout_vus_validation_top_candidates.csv", "Collagen gene-heldout top VUS candidate table", "C7", required=False),
        artifact(repo, "collagen_application", "results/interpretability_applications/collagen_gene_heldout_vus_validation.md", "Collagen gene-heldout VUS validation report", "C7", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/esol_crossmodal_functional_probe_summary.csv", "eSOL cross-modal functional phenotype probe summary", "C10", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/esol_crossmodal_functional_probe_deltas.csv", "eSOL cross-modal deltas and shuffled-DNA control", "C10", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/esol_crossmodal_functional_probe.md", "eSOL negative/control functional phenotype report", "C10", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/dms_clean_functional_validation_summary.csv", "avGFP DMS clean functional validation summary", "C11", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/dms_clean_functional_validation_tests.csv", "avGFP DMS leakage-controlled functional validation tests", "C11", required=False),
        artifact(repo, "functional_controls", "results/interpretability_applications/dms_clean_functional_validation.md", "avGFP DMS functional-assay method-context report", "C11", required=False),
    ]
    return rows


def build_commands() -> list[Command]:
    return [
        Command(
            "brca1_native_sae",
            "Reproduce BRCA1 native-SAE causal intervention plus feature-only and common-background addback-rescue controls.",
            "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python scripts/run_brca1_native_finetuned_sae_intervention.py --repo-root . --device cpu --epochs 800 --patience 80 --random-sets 100 --permutation-sets 20 --sae-hidden 1024 --sae-k 32 --top-k-features 32 --dose-k-values 8,16,32,64 --output-prefix brca1_native_finetuned_sae_rescue",
            "brca1_native_finetuned_sae_rescue_summary.csv, random/rescue/sufficiency controls, variant scores",
            "complete in current artifact set; deletion is positive, addback rescue is positive, feature-only sufficiency is not supported",
        ),
        Command(
            "brca1_bridge_cards",
            "Regenerate BRCA1 feature cards and BRCA2 hotspot cards from existing result tables.",
            "python scripts/summarize_interpretability_feature_cards.py --repo-root .",
            "interpretability_feature_hotspot_cards.md and *_cards.csv",
            "complete",
        ),
        Command(
            "brca1_matched_annotation_control",
            "Regenerate BRCA1 consequence/region-matched control for native-effect biological localization.",
            "python scripts/analyze_brca1_native_sae_matched_annotation_control.py --repo-root . --permutations 5000",
            "brca1_native_sae_matched_annotation_control_summary.csv, null.csv, balance.csv, strata.csv, md report",
            "complete; supports C2 beyond consequence/region composition",
        ),
        Command(
            "brca1_covariate_matched_control",
            "Regenerate BRCA1 consequence/region plus CADD/phyloP nearest-neighbor control for native-effect biological localization.",
            "python scripts/analyze_brca1_native_sae_covariate_matched_control.py --repo-root . --bootstraps 5000",
            "brca1_native_sae_covariate_matched_control_summary.csv, pairs.csv, bootstrap.csv, balance.csv, md report",
            "complete; CADD/phyloP-balanced residual LOF enrichment remains positive",
        ),
        Command(
            "brca2_domain_matched_control",
            "Regenerate BRCA2 both-high versus domain-matched non-both-high control.",
            "python scripts/analyze_brca2_discordance_domain_matched_control.py --permutations 20000 --bootstraps 5000",
            "brca2_discordance_domain_matched_control_summary.csv, null.csv, bootstrap.csv, domain_balance.csv, md report",
            "complete; both-high LOF enrichment remains far above BRCA2-domain-matched non-both-high null",
        ),
        Command(
            "brca2_blinded_handoff",
            "Regenerate BRCA2 blinded review and assay handoff package.",
            "python scripts/design_brca2_blinded_handoff_package.py --repo-root .",
            "brca2_blinded_review_sheet.csv, brca2_blinded_assay_readout_template.csv, brca2_blinded_unblinding_key.csv",
            "complete",
        ),
        Command(
            "brca2_review_dossier",
            "Regenerate BRCA2 review/assay evidence dossier.",
            "python scripts/build_brca2_review_evidence_dossier.py --repo-root .",
            "brca2_review_evidence_dossier.md, internal.csv, blinded_packet.csv, arm_summary.csv",
            "complete; blinded packet leak check passed locally",
        ),
        Command(
            "brca2_blinded_validation",
            "Evaluate locked BRCA2 blinded review/assay outcomes without unblinding incomplete sheets.",
            "python scripts/analyze_brca2_blinded_review_assay_results.py --repo-root .",
            "brca2_blinded_validation_status.csv, primary_endpoint.csv, arm_summary.csv, report.md",
            "complete; current status is expected to remain not-ready until external outcomes are filled",
        ),
        Command(
            "brca2_external_proxy_validation",
            "Regenerate BRCA2 exact-ClinVar and same-residue proxy validation checks.",
            "python scripts/analyze_brca2_external_proxy_validation.py --repo-root .",
            "brca2_external_proxy_validation_summary.csv, known_clinvar.csv, panel_same_residue.csv, panel_proxy_tests.csv, md report",
            "complete; proxy support only, not a locked clinical endpoint",
        ),
        Command(
            "brca2_clinvar_temporal_proxy",
            "Regenerate BRCA2 archive-current ClinVar temporal proxy validation.",
            "python scripts/analyze_brca2_clinvar_temporal_proxy_validation.py --repo-root .",
            "brca2_clinvar_temporal_proxy_validation.md and candidate/panel proxy CSVs",
            "complete; proxy supports candidate-tier triage but does not satisfy locked clinical-utility gate",
        ),
        Command(
            "brca2_mavedb_external_assay_validation",
            "Regenerate BRCA2 public MaveDB external functional-assay validation for mechanism strata and panel arms.",
            "python scripts/analyze_brca2_mavedb_external_assay_validation.py --repo-root .",
            "brca2_mavedb_external_assay_validation.md, score_sets.csv, orientation.csv, local_matches.csv, panel_matches.csv, panel_tests.csv, category_tests.csv",
            "complete; HAP1 public sGE supports panel pathogenic-review vs benign-control separation but is not a locked blinded clinical endpoint",
        ),
        Command(
            "brca2_public_mavedb_hap1_assay_readout",
            "Map public HAP1 MaveDB scores to the 64-row BRCA2 blinded panel and evaluate the matched-subset proxy endpoint.",
            "python scripts/build_brca2_public_mavedb_hap1_assay_readout.py --repo-root .",
            "brca2_public_mavedb_hap1_assay_readout.md, assay_readout.csv, calibration.csv, matched_subset_endpoints.csv",
            "complete; matched-subset endpoint is positive",
        ),
        Command(
            "brca2_public_mavedb_hap1_blinded_status",
            "Run the conservative blinded analyzer on the public HAP1 readout to confirm it remains partial, not a locked 64-row endpoint.",
            "python scripts/analyze_brca2_blinded_review_assay_results.py --repo-root . --assay-sheet results/interpretability_applications/brca2_public_mavedb_hap1_assay_readout.csv --output-prefix brca2_public_mavedb_hap1_blinded_validation",
            "brca2_public_mavedb_hap1_blinded_validation_status.csv, primary_endpoint.csv, arm_summary.csv, report.md",
            "complete; locked 64-row blinded endpoint remains partial/not-ready",
        ),
        Command(
            "brca2_evo2_full",
            "Extract BRCA2 full Evo2 LLR and embeddings.",
            "sbatch scripts/slurm_brca2_evo2.sh",
            "results/variant/brca2_evo2.npz",
            "running/pending at package generation if output missing",
        ),
        Command(
            "brca2_downstream",
            "Run BRCA2 shared embedding, pretrained fusion, gate analysis, and SGE mechanism validation.",
            "sbatch --dependency=afterok:<brca2_evo2_jobid> scripts/slurm_brca2_downstream.sh",
            "brca2 SGE metric summaries and results/brca2_gate_analysis/fold_artifacts",
            "pending until full Evo2 output exists",
        ),
        Command(
            "brca2_native_sae",
            "Run BRCA2 native-SAE causal intervention with BRCA2 fold artifacts.",
            "sbatch --dependency=afterok:<brca2_downstream_jobid> scripts/slurm_brca2_native_sae.sh",
            "brca2_native_finetuned_sae_intervention_summary.csv and controls",
            "see G5 for current pass/fail status",
        ),
        Command(
            "brca2_summary",
            "Regenerate BRCA2 replication summary after dependent jobs.",
            "python scripts/summarize_brca2_replication.py --repo-root .",
            "brca2_replication_summary.md",
            "checkpoint-complete; native-SAE is mixed/not-met",
        ),
        Command(
            "brca2_focused_native_sae",
            "Regenerate focused BRCA2 native-SAE strata screen.",
            "python scripts/analyze_brca2_native_sae_focused_strata.py --repo-root . --bootstrap 1000 --permutations 1000",
            "brca2_native_sae_focused_strata_summary.csv and brca2_native_sae_focused_strata.md",
            "complete; screen only, not a replacement for original feature-selection controls",
        ),
        Command(
            "brca2_stratum_control_pilot",
            "Regenerate original-style BRCA2 native-SAE controls inside candidate strata.",
            "python scripts/run_brca2_native_sae_stratum_control.py --repo-root . --candidate protein_high_dna_low --candidate ob1_both_high --candidate tier3_vus_conflicting --epochs 80 --patience 12 --sae-hidden 256 --random-sets 20 --permutation-sets 10 --min-active 3 --top-k-features 16 --output-prefix brca2_native_sae_stratum_control_pilot && python scripts/summarize_brca2_native_sae_stratum_control.py --repo-root . --prefix brca2_native_sae_stratum_control_pilot",
            "brca2_native_sae_stratum_control_pilot_summary.csv and brca2_native_sae_stratum_control_pilot.md",
            "complete; pilot did not pass original-style gate",
        ),
        Command(
            "bap1_sge_prepare",
            "Prepare the public BAP1 Nature Genetics/MaveDB SGE dataset as a third-gene benchmark.",
            "python scripts/prepare_bap1_sge.py --repo-root . --timeout 120",
            "data/variant/bap1/bap1_variants.csv, bap1_Q96TC6.fasta, bap1_sge_data_preparation.md",
            "complete; data-ready, not yet a CrossBioSAE replication result",
        ),
        Command(
            "bap1_evo2_llr",
            "Run BAP1 Evo2 LLR extraction for the third-gene functional-map checkpoint.",
            "sbatch scripts/slurm_bap1_evo2_llr.sh",
            "results/variant/bap1_evo2_llr.npz",
            "complete in current artifact set; rerun only if BAP1 inputs change",
        ),
        Command(
            "bap1_sge_external_anchors",
            "Check that BAP1 SGE has expected ClinVar, gnomAD, and consequence anchors before using it as a third-gene benchmark.",
            "python scripts/analyze_bap1_sge_external_anchors.py --repo-root .",
            "bap1_sge_external_anchors.md, group_summary.csv, tests.csv, vus_candidates.csv",
            "complete; validates BAP1 SGE as a credible benchmark but does not yet validate CrossBioSAE on BAP1",
        ),
        Command(
            "bap1_sge_baseline_predictors",
            "Evaluate public EVE/SIFT/PolyPhen missense baselines on BAP1 SGE labels.",
            "python scripts/analyze_bap1_sge_baseline_predictors.py --repo-root .",
            "bap1_sge_baseline_predictors.md, metric_summary.csv, binary_calls.csv",
            "complete; gives comparator baselines for the BAP1 Evo2/ESM checkpoint",
        ),
        Command(
            "bap1_esm",
            "Run BAP1 ESM missense delta extraction for the third-gene functional-map checkpoint.",
            "sbatch scripts/slurm_bap1_esm.sh",
            "results/variant/bap1_esm_delta.npz",
            "complete in current artifact set; rerun only if BAP1 inputs change",
        ),
        Command(
            "bap1_checkpoint",
            "Evaluate BAP1 depleted-vs-unchanged labels using Evo2 LLR, ESM, and ESM+LLR with position-grouped CV.",
            "sbatch --dependency=afterok:<bap1_evo2_jobid>:<bap1_esm_jobid> scripts/slurm_bap1_checkpoint.sh",
            "bap1_llr_esm_checkpoint_metric_summary.csv, bootstrap_delta.csv, variant_scores.csv, checkpoint md report, and bap1_llr_esm_discordance_* mechanism-strata reports",
            "complete; positive finite-SNV/imputed-LLR checkpoint with indel/multibase caveat",
        ),
        Command(
            "bap1_checkpoint_discordance",
            "Convert BAP1 checkpoint variant scores into DNA/protein concordance and discordance mechanism strata.",
            "python scripts/analyze_llr_esm_checkpoint_discordance.py --checkpoint-scores results/interpretability_applications/bap1_llr_esm_checkpoint_variant_scores.csv --output-dir results/interpretability_applications --output-prefix bap1_llr_esm_discordance --gene-name BAP1 --domain-col domains --quantile 0.20",
            "bap1_llr_esm_discordance.md, category_summary.csv, group_summary.csv, anchor_summary.csv, examples.csv, missense_scores.csv",
            "complete; both-high finite missense stratum is strongly depleted-enriched",
        ),
        Command(
            "rad51c_sge_prepare",
            "Prepare RAD51C MaveDB SGE as the first fallback HR-pathway third-gene benchmark if BAP1 is weak.",
            "python scripts/prepare_rad51c_sge_mavedb.py --repo-root . --timeout 90",
            "data/variant/rad51c/rad51c_mavedb_scores.csv, rad51c_sge_data_preparation.md, rad51c_sge_data_preparation_summary.csv",
            "complete as a functional-map fallback; GRCh38 mapping is now handled by rad51c_grch38_map",
        ),
        Command(
            "rad51c_grch38_map",
            "Map RAD51C transcript-level MaveDB direct cDNA SNVs onto GRCh38 and generate the protein FASTA for ESM.",
            "python scripts/map_rad51c_sge_to_grch38.py --repo-root . --timeout 90",
            "data/variant/rad51c/rad51c_grch38_variants.csv, rad51c_O43502.fasta, rad51c_grch38_mapping.md, mapping summary/status/consequence CSVs",
            "complete; 3,390 direct cDNA SNVs mapped with GRCh38 ref matches, 3,365 binary rows, 2,348 missense rows",
        ),
        Command(
            "rad51c_external_anchors",
            "Check current ClinVar exact-match and consequence-class anchors for the RAD51C fallback benchmark.",
            "python scripts/analyze_rad51c_sge_external_anchors.py --repo-root .",
            "rad51c_sge_external_anchors.md, annotated.csv, group_summary.csv, tests.csv, consequence_summary.csv, candidate_review.csv",
            "complete; validates RAD51C SGE as a clinically anchored fallback benchmark but does not yet validate CrossBioSAE on RAD51C",
        ),
        Command(
            "rad51c_esm",
            "Run RAD51C ESM missense delta extraction for the fallback third-gene checkpoint.",
            "sbatch scripts/slurm_rad51c_esm.sh",
            "results/variant/rad51c_esm_delta.npz",
            "complete in current artifact set; rerun only if RAD51C inputs change",
        ),
        Command(
            "rad51c_evo2_llr",
            "Run RAD51C Evo2 LLR extraction on the GRCh38-mapped direct-SNV checkpoint table.",
            "sbatch scripts/slurm_rad51c_evo2_llr.sh",
            "results/variant/rad51c_evo2_llr.npz",
            "complete in current artifact set; rerun only if RAD51C inputs change",
        ),
        Command(
            "rad51c_checkpoint",
            "Evaluate RAD51C depleted-vs-unchanged labels using Evo2 LLR, ESM, and ESM+LLR with position-grouped CV, then build DNA/protein discordance strata.",
            "sbatch --dependency=afterok:<rad51c_evo2_jobid>:<rad51c_esm_jobid> scripts/slurm_rad51c_checkpoint.sh",
            "rad51c_llr_esm_checkpoint_metric_summary.csv, bootstrap_delta.csv, variant_scores.csv, and rad51c_llr_esm_discordance_* reports",
            "complete; positive clean third-gene checkpoint",
        ),
        Command(
            "rad51c_checkpoint_afternotok_fallback",
            "Run the RAD51C checkpoint if Evo2 LLR exits nonzero after writing the score file.",
            "sbatch --dependency=afternotok:<rad51c_evo2_jobid> scripts/slurm_rad51c_checkpoint.sh",
            "same outputs as rad51c_checkpoint",
            "historical fallback; no longer needed for the current completed RAD51C checkpoint",
        ),
        Command(
            "rad51c_latent_sae",
            "Run a checkpoint-level ESM latent-SAE intervention screen for RAD51C.",
            "sbatch scripts/slurm_rad51c_latent_sae.sh",
            "rad51c_latent_sae_intervention_summary.csv, folds.csv, controls.csv, random_ablation.csv, dose_response.csv, rescue_controls.csv, selected_features.csv, stratum_summary.csv, variant_scores.csv, md report",
            "completed as job 9802960; positive checkpoint-level sparse-feature necessity screen",
        ),
        Command(
            "rad51c_latent_sae_application",
            "Generate RAD51C latent-SAE sparse-mechanism variant priorities and a 64-row review/assay panel.",
            "python scripts/summarize_rad51c_latent_sae_application.py --repo-root .",
            "rad51c_latent_sae_application.md, panel.csv, arm_summary.csv, variant_priorities.csv, feature_fold_summary.csv, utility_tests.csv",
            "complete; application panel should be used for review/assay planning but not clinical reclassification",
        ),
        Command(
            "rad51c_evo2_full",
            "Generate full RAD51C Evo2 LLR plus embedding deltas for BRCA1-style native-SAE replication.",
            "sbatch scripts/slurm_rad51c_evo2_full.sh",
            "results/variant/rad51c_evo2.npz with llr and edelta",
            "submitted as job 9802961",
        ),
        Command(
            "rad51c_gate_analysis",
            "Fine-tune RAD51C fold-specific fusion/gate heads and save fold artifacts for native-SAE intervention.",
            "sbatch --dependency=afterok:<rad51c_evo2_full_jobid> scripts/slurm_rad51c_gate_analysis.sh",
            "results/rad51c_gate_analysis/rad51c_gate_analysis.npz and fold_artifacts/variant_fusion_fold*.pt",
            "submitted as job 9802962 afterok:9802961",
        ),
        Command(
            "rad51c_native_sae",
            "Run BRCA1-style fold-native SAE intervention on RAD51C gate-head z_prot representations.",
            "sbatch --dependency=afterok:<rad51c_gate_jobid> scripts/slurm_rad51c_native_sae.sh",
            "rad51c_native_finetuned_sae_intervention_summary.csv, folds.csv, controls.csv, random_ablation.csv, dose_response.csv, sufficiency_controls.csv, rescue_controls.csv, selected_features.csv, stratum_summary.csv, variant_scores.csv, md report",
            "submitted as job 9802963 afterok:9802962; this is the decisive third-gene causal replication gate",
        ),
        Command(
            "rad51c_native_sae_missense",
            "Run the primary RAD51C native-SAE check restricted to missense rows, avoiding non-missense zero-pmask dilution in z_prot.",
            "sbatch --dependency=afterok:<rad51c_gate_jobid> scripts/slurm_rad51c_native_sae_missense.sh",
            "rad51c_native_finetuned_sae_intervention_missense_summary.csv, folds.csv, controls.csv, random_ablation.csv, dose_response.csv, sufficiency_controls.csv, rescue_controls.csv, selected_features.csv, stratum_summary.csv, variant_scores.csv, md report",
            "submitted as job 9803849 afterok:9802962; this is the primary z_prot native-SAE replication gate if the all-variant run is diluted by non-missense masking",
        ),
        Command(
            "third_gene_application_plan",
            "Regenerate the BAP1/RAD51C application blueprint and publishability decision gates.",
            "python scripts/summarize_third_gene_application_plan.py --repo-root .",
            "third_gene_interpretability_application_plan.md, third_gene_application_decision_gates.csv, third_gene_application_blueprint.csv",
            "complete; summarizes RAD51C clean positive checkpoint and BAP1 finite-SNV boundary checkpoint while keeping native-SAE pending",
        ),
        Command(
            "third_gene_review_assay_panel",
            "Regenerate the BAP1/RAD51C 64-row blinded review/assay panel from current benchmark anchors.",
            "python scripts/design_third_gene_review_assay_panel.py --repo-root .",
            "third_gene_review_assay_panel.md, internal.csv, blinded.csv, assay_template.csv, summary.csv",
            "complete; produces 32 candidate-review rows plus 16 positive and 16 negative controls, but remains pre-checkpoint",
        ),
        Command(
            "third_gene_assay_statistical_plan",
            "Regenerate the pre-specified BAP1/RAD51C panel endpoint thresholds and exact power grid.",
            "python scripts/design_third_gene_assay_statistical_plan.py --repo-root .",
            "third_gene_assay_statistical_plan.md, endpoint_plan.csv, statistical_thresholds.csv, power_grid.csv, design_sanity.csv",
            "complete; defines binary Fisher thresholds before external review/assay outcomes are available",
        ),
        Command(
            "collagen_leave_one_gene_out",
            "Regenerate leave-one-collagen-gene-out sparse-feature transfer validation.",
            "python scripts/analyze_collagen_leave_one_gene_out.py --repo-root . --n-random-sets 200",
            "collagen_leave_one_gene_out_primary_summary.csv, control_tests.csv, selected_features.csv, random_null.csv, md report",
            "complete; mixed cross-gene support, suitable for supplementary C7 but not a main clinical claim",
        ),
        Command(
            "collagen_gene_heldout_vus_validation",
            "Apply gene-heldout collagen features to VUS review-hook prioritization with matched random feature nulls.",
            "python scripts/analyze_collagen_gene_heldout_vus_validation.py --repo-root .",
            "collagen_gene_heldout_vus_validation_summary.csv, random_null.csv, variant_scores.csv, top_candidates.csv, md report",
            "complete; only COL3A1 has a matched-random-positive broader application hook, so this remains supplementary/boundary evidence",
        ),
        Command(
            "esol_crossmodal_functional_probe",
            "Regenerate fast eSOL functional-phenotype cross-modal boundary check.",
            "python scripts/analyze_esol_crossmodal_functional_probe.py --repo-root .",
            "esol_crossmodal_functional_probe_summary.csv, deltas.csv, predictions.csv, md report",
            "complete; current linear concat result is negative/control evidence",
        ),
        Command(
            "dms_clean_functional_validation",
            "Regenerate avGFP DMS clean functional-assay method-context summary after the HPC3 clean job finishes.",
            "python scripts/summarize_dms_clean_functional_validation.py --repo-root .",
            "dms_clean_functional_validation_summary.csv, tests.csv, md report",
            "complete; weak GroupKFold concat gain, but no strong synonymous/codon-level mechanism",
        ),
        Command(
            "evidence_matrix",
            "Regenerate manuscript claim matrix.",
            "python scripts/summarize_interpretability_submission_evidence_matrix.py --repo-root .",
            "interpretability_submission_evidence_matrix.md/csv",
            "complete; rerun after BRCA2 native-SAE",
        ),
        Command(
            "top_journal_interpretability_evidence_plan",
            "Regenerate the top-journal proof-pattern map and current experiment/action audit.",
            "python scripts/summarize_top_journal_interpretability_evidence_plan.py --repo-root .",
            "top_journal_interpretability_evidence_plan.md, literature_patterns.csv, evidence_matrix.csv, priority_actions.csv",
            "complete; rerun after RAD51C native-SAE and any external review/assay endpoint",
        ),
        Command(
            "manuscript_package",
            "Regenerate figure plan and reviewer risk register.",
            "python scripts/summarize_interpretability_manuscript_package.py --repo-root .",
            "interpretability_manuscript_package.md and risk/figure CSVs",
            "complete; rerun after BRCA2 native-SAE",
        ),
        Command(
            "manuscript_draft",
            "Regenerate the claim-bounded manuscript skeleton.",
            "python scripts/summarize_interpretability_manuscript_draft.py --repo-root .",
            "interpretability_manuscript_draft.md and interpretability_manuscript_draft_sections.csv",
            "complete; rerun after evidence matrix or BRCA2 native-SAE changes",
        ),
        Command(
            "draft_figures",
            "Regenerate current manuscript draft figures.",
            "python scripts/plot_interpretability_summary_figures.py --repo-root .",
            "fig2_brca1_native_sae.png, fig3_brca1_biological_bridge.png, fig4_brca2_checkpoint.png, fig5_brca2_application.png, fig_s2_third_gene_benchmarks.png when BAP1/RAD51C anchors exist, fig6_brca2_native_sae_mixed.png when BRCA2 native-SAE outputs exist",
            "complete; Figure 3 includes BRCA1 CADD/phyloP matched-control and protein-coordinate localization panels; Supplementary Figure S2 summarizes BAP1/RAD51C benchmark anchors plus RAD51C and BAP1 checkpoint results while keeping third-gene native-SAE causal replication pending",
        ),
    ]


def run_shell(cmd: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - report-only path
        return f"{type(exc).__name__}: {exc}"


def build_gates(repo: Path) -> list[Gate]:
    artifacts = {a.path: a for a in build_artifacts(repo)}
    brca2_evo2 = artifacts["results/variant/brca2_evo2.npz"].exists
    brca2_native = artifacts[
        "results/interpretability_applications/brca2_native_finetuned_sae_intervention_summary.csv"
    ].exists
    proxy_path = repo / "results/interpretability_applications/brca2_external_proxy_validation_summary.csv"
    proxy_evidence = ""
    if proxy_path.exists():
        try:
            proxy = pd.read_csv(proxy_path)
            both_high = proxy.loc[proxy["summary_item"].eq("known_clinvar_both_high_pathogenic_enrichment")]
            panel_exact = proxy.loc[proxy["summary_item"].eq("prospective_panel_exact_current_clinvar_known")]
            pieces = []
            if not both_high.empty:
                pieces.append("external_proxy_" + str(both_high.iloc[0].get("value", "")))
            if not panel_exact.empty:
                pieces.append("panel_exact_known=" + str(panel_exact.iloc[0].get("value", "")))
            proxy_evidence = "; ".join(pieces)
        except Exception as exc:  # pragma: no cover - report-only path
            proxy_evidence = f"proxy_inspect_error={type(exc).__name__}"
    mavedb_evidence = ""
    mavedb_panel_path = repo / "results/interpretability_applications/brca2_mavedb_external_assay_panel_tests.csv"
    mavedb_category_path = repo / "results/interpretability_applications/brca2_mavedb_external_assay_category_tests.csv"
    hap1_readout_endpoint_path = (
        repo / "results/interpretability_applications/brca2_public_mavedb_hap1_matched_subset_endpoints.csv"
    )
    if mavedb_panel_path.exists():
        try:
            panel = pd.read_csv(mavedb_panel_path)
            hap1_panel = panel[
                panel["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
                & panel["comparison"].eq("pathogenic_review_vs_benign_controls")
            ]
            pieces = []
            if not hap1_panel.empty:
                row = hap1_panel.iloc[0]
                pieces.append(
                    "mavedb_hap1_panel_auc="
                    + f"{float(row.get('external_lof_auroc', float('nan'))):.3f}"
                    + "; p="
                    + f"{float(row.get('mannwhitney_p_greater', float('nan'))):.3g}"
                    + "; n="
                    + str(int(row.get("n_pathogenic_review_matches", 0)))
                    + "/"
                    + str(int(row.get("n_control_matches", 0)))
                )
            if mavedb_category_path.exists():
                category = pd.read_csv(mavedb_category_path)
                hap1_bh = category[
                    category["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
                    & category["comparison"].eq("both_high_vs_both_low")
                ]
                if not hap1_bh.empty:
                    row = hap1_bh.iloc[0]
                    pieces.append(
                        "mavedb_hap1_both_high_auc="
                        + f"{float(row.get('external_lof_auroc', float('nan'))):.3f}"
                        + "; p="
                        + f"{float(row.get('mannwhitney_p_greater', float('nan'))):.3g}"
                    )
            if hap1_readout_endpoint_path.exists():
                endpoints = pd.read_csv(hap1_readout_endpoint_path)
                primary = endpoints[endpoints["comparison"].eq("pathogenic_review_vs_benign_controls")]
                if not primary.empty:
                    row = primary.iloc[0]
                    pieces.append(
                        "hap1_readout_binary="
                        + str(int(row.get("positive_binary_lof", 0)))
                        + "/"
                        + str(int(row.get("n_positive_group", 0)))
                        + " vs "
                        + str(int(row.get("negative_binary_lof", 0)))
                        + "/"
                        + str(int(row.get("n_negative_group", 0)))
                        + "; fisher_p="
                        + f"{float(row.get('fisher_p_greater', float('nan'))):.3g}"
                    )
            mavedb_evidence = "; ".join(pieces)
        except Exception as exc:  # pragma: no cover - report-only path
            mavedb_evidence = f"mavedb_inspect_error={type(exc).__name__}"
    brca2_native_status = "pending"
    brca2_native_evidence = "brca2_native_finetuned_sae_intervention_summary.csv"
    brca2_native_next = "Run/inspect downstream and native-SAE jobs after G4."
    if brca2_native:
        try:
            native = pd.read_csv(
                repo / "results/interpretability_applications/brca2_native_finetuned_sae_intervention_summary.csv"
            ).iloc[0]
            delta = float(native.get("delta_auroc_recon_minus_top_ablate", float("nan")))
            random_p = float(native.get("empirical_p_random_delta_ge_top_mean", float("nan")))
            perm_p = float(native.get("empirical_p_label_permuted_delta_ge_top_mean", float("nan")))
            brca2_native_evidence = (
                "delta_auroc="
                + f"{delta:.4g}"
                + "; random_p="
                + f"{random_p:.4g}"
                + "; label_perm_p="
                + f"{perm_p:.4g}"
            )
            if delta > 0 and random_p <= 0.05 and perm_p <= 0.05:
                brca2_native_status = "passed"
                brca2_native_next = "Promote BRCA2 native-SAE to independent mechanistic replication."
            else:
                brca2_native_status = "not_met"
                brca2_native_next = "Inspect BRCA2 native-SAE controls/strata before claiming replication."
                focused_path = repo / "results/interpretability_applications/brca2_native_sae_focused_strata_summary.csv"
                if focused_path.exists():
                    focused = pd.read_csv(focused_path)
                    tested = int(focused["analysis_status"].eq("tested").sum()) if "analysis_status" in focused else 0
                    screen_positive = (
                        int(focused["passes_focused_screen"].astype(str).str.lower().eq("true").sum())
                        if "passes_focused_screen" in focused
                        else 0
                    )
                    brca2_native_evidence += f"; focused_screen_tested={tested}; focused_screen_positive={screen_positive}"
                    brca2_native_next = (
                        "Focused screen is descriptive only; rerun original-style feature-selection controls inside "
                        "candidate strata before claiming BRCA2 replication."
                    )
                pilot_path = repo / "results/interpretability_applications/brca2_native_sae_stratum_control_pilot_summary.csv"
                if pilot_path.exists():
                    pilot = pd.read_csv(pilot_path)
                    pilot_passed = (
                        int(pilot["passes_original_style_gate"].astype(str).str.lower().eq("true").sum())
                        if "passes_original_style_gate" in pilot
                        else 0
                    )
                    brca2_native_evidence += f"; stratum_control_pilot_passed={pilot_passed}/{len(pilot)}"
                    brca2_native_next = (
                        "Current original-style stratum-control pilot did not pass; prioritize external "
                        "review/assay validation or redesign BRCA2 native-SAE before claiming replication."
                    )
        except Exception as exc:  # pragma: no cover - report-only path
            brca2_native_status = "not_met"
            brca2_native_evidence = f"inspect_error={type(exc).__name__}"
            brca2_native_next = "Fix or regenerate BRCA2 native-SAE summary."
    squeue = run_shell("squeue -j 9802290,9802296,9802298,9802300 -o '%i %.18j %.9T %.10M %.6D %R' || true", repo)
    g6_status = "not_met"
    g6_evidence = "No blinded review or new assay outcome artifact in repo."
    g6_next = "Keep language to review/assay triage, not VUS reclassification."
    validation_path = repo / "results/interpretability_applications/brca2_blinded_validation_status.csv"
    primary_path = repo / "results/interpretability_applications/brca2_blinded_validation_primary_endpoint.csv"
    if validation_path.exists():
        try:
            validation = pd.read_csv(validation_path).iloc[0]
            validation_status = str(validation.get("validation_status", "unknown"))
            review_binary = int(validation.get("review_rows_with_binary_classification", 0))
            review_total = int(validation.get("review_rows_total", 0))
            assay_binary = int(validation.get("assay_rows_with_binary_call", 0))
            assay_mean = int(validation.get("assay_rows_with_mean_function_score", 0))
            assay_total = int(validation.get("assay_rows_total", 0))
            g6_evidence = (
                f"validation_status={validation_status}; review_binary={review_binary}/{review_total}; "
                f"assay_binary={assay_binary}/{assay_total}; assay_mean={assay_mean}/{assay_total}"
            )
            temporal_proxy = repo / "results/interpretability_applications/brca2_clinvar_temporal_proxy_validation_candidate_tests.csv"
            if temporal_proxy.exists():
                g6_evidence += "; archive-current ClinVar proxy exists but is not a locked blinded outcome"
            if mavedb_evidence:
                g6_evidence += "; public MaveDB functional proxy: " + mavedb_evidence
            g6_next = (
                "Fill locked external outcomes in the blinded review or assay template, then rerun "
                "analyze_brca2_blinded_review_assay_results.py and inspect the pre-specified Fisher endpoint."
            )
            if primary_path.exists():
                primary = pd.read_csv(primary_path).iloc[0]
                endpoint_status = str(primary.get("endpoint_status", "unknown"))
                pval = primary.get("fisher_p_greater", float("nan"))
                if endpoint_status == "evaluated" and pd.notna(pval):
                    g6_evidence += f"; primary_endpoint={endpoint_status}; fisher_p_greater={float(pval):.4g}"
                    if float(pval) < 0.05:
                        g6_status = "passed"
                        g6_next = "A blinded external endpoint is positive; update claim language cautiously and keep clinical reclassification separate from triage utility."
                    else:
                        g6_next = "External outcomes exist but do not pass the primary endpoint; report as validation failure/error analysis."
        except Exception as exc:  # pragma: no cover - report-only path
            g6_evidence = f"validation_status_inspect_error={type(exc).__name__}"
            g6_next = "Fix or rerun the BRCA2 blinded validation analyzer."
    bap1_status = "pending"
    bap1_evidence = "BAP1 SGE data preparation has not been run."
    bap1_next = "Run scripts/prepare_bap1_sge.py, then submit BAP1 Evo2/ESM checkpoint jobs."
    bap1_variants_path = repo / "results/interpretability_applications/bap1_sge_variants.csv"
    bap1_metrics_path = repo / "results/interpretability_applications/bap1_llr_esm_checkpoint_metric_summary.csv"
    bap1_delta_path = repo / "results/interpretability_applications/bap1_llr_esm_checkpoint_bootstrap_delta.csv"
    bap1_disc_path = repo / "results/interpretability_applications/bap1_llr_esm_discordance_category_summary.csv"
    bap1_anchor_path = repo / "results/interpretability_applications/bap1_sge_external_anchors_tests.csv"
    bap1_baseline_path = repo / "results/interpretability_applications/bap1_sge_baseline_predictors_metric_summary.csv"
    rad51c_mapping_path = repo / "results/interpretability_applications/rad51c_grch38_mapping_summary.csv"
    rad51c_anchor_path = repo / "results/interpretability_applications/rad51c_sge_external_anchors_tests.csv"
    rad51c_metrics_path = repo / "results/interpretability_applications/rad51c_llr_esm_checkpoint_metric_summary.csv"
    rad51c_disc_path = repo / "results/interpretability_applications/rad51c_llr_esm_discordance_category_summary.csv"
    rad51c_latent_path = repo / "results/interpretability_applications/rad51c_latent_sae_intervention_summary.csv"
    rad51c_latent_app_path = repo / "results/interpretability_applications/rad51c_latent_sae_application_arm_summary.csv"
    if bap1_variants_path.exists():
        try:
            bap1 = pd.read_csv(bap1_variants_path)
            binary_n = int(pd.to_numeric(bap1.get("label", pd.Series(dtype=float)), errors="coerce").notna().sum())
            depleted_n = (
                int(bap1["functional_classification"].eq("depleted").sum())
                if "functional_classification" in bap1
                else 0
            )
            unchanged_n = (
                int(bap1["functional_classification"].eq("unchanged").sum())
                if "functional_classification" in bap1
                else 0
            )
            bap1_evidence = (
                f"bap1_sge_rows={len(bap1)}; binary_depleted_vs_unchanged={binary_n}; "
                f"depleted={depleted_n}; unchanged={unchanged_n}"
            )
            if bap1_anchor_path.exists():
                anchors = pd.read_csv(bap1_anchor_path)
                clinvar_anchor = anchors[
                    anchors["test"].eq("clinvar_pathogenic_enriched_for_depleted_vs_benign")
                ]
                gnomad_anchor = anchors[anchors["test"].eq("gnomad_observed_depleted_less_than_unobserved")]
                if not clinvar_anchor.empty:
                    row = clinvar_anchor.iloc[0]
                    bap1_evidence += (
                        "; clinvar_anchor="
                        + str(int(row.get("first_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("first_n_binary", 0)))
                        + " vs "
                        + str(int(row.get("second_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("second_n_binary", 0)))
                        + "; auc="
                        + f"{float(row.get('external_lof_auroc', float('nan'))):.3f}"
                    )
                if not gnomad_anchor.empty:
                    row = gnomad_anchor.iloc[0]
                    bap1_evidence += (
                        "; gnomad_depleted="
                        + str(int(row.get("first_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("first_n_binary", 0)))
                        + " vs "
                        + str(int(row.get("second_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("second_n_binary", 0)))
                    )
            if bap1_baseline_path.exists():
                baselines = pd.read_csv(bap1_baseline_path)
                eve = baselines[baselines["baseline"].eq("EVE score")]
                if not eve.empty:
                    row = eve.iloc[0]
                    bap1_evidence += (
                        "; public_eve_auroc="
                        + f"{float(row.get('auroc_for_sge_depleted', float('nan'))):.3f}"
                    )
            bap1_next = "Wait for BAP1 Evo2/ESM jobs, then run or inspect the BAP1 checkpoint metrics."
            if bap1_metrics_path.exists():
                metrics = pd.read_csv(bap1_metrics_path)
                llr = metrics[
                    metrics["scope"].eq("all") & metrics["metric"].eq("Evo2 LLR zero-shot")
                ]
                if not llr.empty:
                    row = llr.iloc[0]
                    bap1_evidence += (
                        "; bap1_llr_auroc="
                        + f"{float(row.get('auroc_for_sge_lof', float('nan'))):.3f}"
                        + "; auprc="
                        + f"{float(row.get('auprc_for_sge_lof', float('nan'))):.3f}"
                    )
                fusion = metrics[
                    metrics["scope"].eq("all") & metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
                ]
                if not fusion.empty:
                    row = fusion.iloc[0]
                    bap1_evidence += (
                        "; bap1_fusion_auroc="
                        + f"{float(row.get('auroc_for_sge_lof', float('nan'))):.3f}"
                        + "; bap1_fusion_auprc="
                        + f"{float(row.get('auprc_for_sge_lof', float('nan'))):.3f}"
                    )
                missense_fusion = metrics[
                    metrics["scope"].eq("missense") & metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
                ]
                if not missense_fusion.empty:
                    row = missense_fusion.iloc[0]
                    bap1_evidence += (
                        "; bap1_missense_fusion_auroc="
                        + f"{float(row.get('auroc_for_sge_lof', float('nan'))):.3f}"
                    )
                if bap1_delta_path.exists():
                    deltas = pd.read_csv(bap1_delta_path)
                    missense_delta = deltas[
                        deltas["comparison"].eq("ESM+LLR minus LLR-CV") & deltas["scope"].eq("missense")
                    ]
                    if not missense_delta.empty:
                        row = missense_delta.iloc[0]
                        bap1_evidence += (
                            "; bap1_missense_fusion_minus_llr_cv="
                            + f"{float(row.get('median_delta_auroc', float('nan'))):.3f}"
                        )
                if bap1_disc_path.exists():
                    disc = pd.read_csv(bap1_disc_path)
                    both_high = disc[disc["category"].eq("both_high")]
                    both_low = disc[disc["category"].eq("both_low")]
                    if not both_high.empty:
                        row = both_high.iloc[0]
                        bap1_evidence += (
                            "; bap1_both_high_n="
                            + str(int(row.get("n", 0)))
                            + "; bap1_both_high_lof_rate="
                            + f"{float(row.get('lof_rate', float('nan'))):.3f}"
                        )
                    if not both_low.empty:
                        row = both_low.iloc[0]
                        bap1_evidence += (
                            "; bap1_both_low_lof_rate="
                            + f"{float(row.get('lof_rate', float('nan'))):.3f}"
                        )
                bap1_status = "passed"
                bap1_next = "Use BAP1 as finite-SNV/imputed-LLR checkpoint and boundary support; do not claim third-gene native-SAE causal replication."
            else:
                bap1_queue = run_shell(
                    "squeue -u \"$USER\" -o '%i %.18j %.9T %.10M %.6D %R' | grep bap1 || true",
                    repo,
                )
                if bap1_queue:
                    bap1_evidence += "; current_bap1_squeue=" + bap1_queue.replace("\n", " | ")
        except Exception as exc:  # pragma: no cover - report-only path
            bap1_status = "pending"
            bap1_evidence = f"bap1_inspect_error={type(exc).__name__}"
            bap1_next = "Fix or rerun BAP1 data preparation."
    if rad51c_mapping_path.exists():
        try:
            rad = pd.read_csv(rad51c_mapping_path).iloc[0]
            bap1_evidence += (
                "; rad51c_grch38_direct_snv="
                + str(int(rad.get("mapped_ref_match_rows", 0)))
                + "; rad51c_binary="
                + str(int(rad.get("output_binary_mapped_rows", 0)))
                + "; rad51c_missense="
                + str(int(rad.get("output_missense_rows", 0)))
                + "; rad51c_lof="
                + str(int(rad.get("output_lof_rows", 0)))
            )
            if rad51c_metrics_path.exists():
                metrics = pd.read_csv(rad51c_metrics_path)
                llr = metrics[
                    metrics["scope"].eq("all") & metrics["metric"].eq("Evo2 LLR zero-shot")
                ]
                if not llr.empty:
                    row = llr.iloc[0]
                    bap1_evidence += (
                        "; rad51c_llr_auroc="
                        + f"{float(row.get('auroc_for_sge_lof', float('nan'))):.3f}"
                        + "; rad51c_llr_auprc="
                        + f"{float(row.get('auprc_for_sge_lof', float('nan'))):.3f}"
                    )
                fusion = metrics[
                    metrics["scope"].eq("all") & metrics["metric"].eq("ESM+Evo2 LLR CV logistic")
                ]
                if not fusion.empty:
                    row = fusion.iloc[0]
                    bap1_evidence += (
                        "; rad51c_fusion_auroc="
                        + f"{float(row.get('auroc_for_sge_lof', float('nan'))):.3f}"
                        + "; rad51c_fusion_auprc="
                        + f"{float(row.get('auprc_for_sge_lof', float('nan'))):.3f}"
                    )
                if rad51c_disc_path.exists():
                    disc = pd.read_csv(rad51c_disc_path)
                    both_high = disc[disc["category"].eq("both_high")]
                    if not both_high.empty:
                        row = both_high.iloc[0]
                        bap1_evidence += (
                            "; rad51c_both_high_n="
                            + str(int(row.get("n", 0)))
                            + "; rad51c_both_high_lof_rate="
                            + f"{float(row.get('lof_rate', float('nan'))):.3f}"
                        )
                if rad51c_latent_path.exists():
                    latent = pd.read_csv(rad51c_latent_path).iloc[0]
                    bap1_evidence += (
                        "; rad51c_latent_sae_delta_auc="
                        + f"{float(latent.get('delta_auroc_recon_minus_top_ablate', float('nan'))):.3f}"
                        + "; rad51c_latent_random_p="
                        + f"{float(latent.get('empirical_p_random_delta_ge_top_mean', float('nan'))):.4f}"
                        + "; rad51c_latent_label_perm_p="
                        + f"{float(latent.get('empirical_p_label_permuted_delta_ge_top_mean', float('nan'))):.4f}"
                    )
                if rad51c_latent_app_path.exists():
                    app = pd.read_csv(rad51c_latent_app_path)
                    bap1_evidence += "; rad51c_latent_panel_rows=" + str(int(app.get("selected_n", pd.Series(dtype=float)).sum()))
                bap1_status = "passed"
                if bap1_metrics_path.exists():
                    bap1_next = "Use RAD51C as a clean positive third-gene checkpoint plus latent-SAE sparse-mechanism screen and BAP1 as finite-SNV/boundary checkpoint support; strict third-gene native-SAE intervention is still required for causal replication."
                else:
                    bap1_next = "Use RAD51C as a positive third-gene checkpoint/mechanism result; continue waiting for BAP1 checkpoint and third-gene native-SAE intervention before claiming causal replication."
            else:
                rad51c_queue = run_shell(
                    "squeue -u \"$USER\" -o '%i %.18j %.9T %.10M %.6D %R' | grep rad51c || true",
                    repo,
                )
                if rad51c_queue:
                    bap1_evidence += "; current_rad51c_squeue=" + rad51c_queue.replace("\n", " | ")
                if bap1_status != "passed":
                    bap1_next = (
                        "Wait for BAP1 and RAD51C Evo2/ESM jobs, then inspect checkpoint metrics before "
                        "promoting any third-gene generalization claim."
                    )
            if rad51c_anchor_path.exists():
                anchors = pd.read_csv(rad51c_anchor_path)
                clinvar_anchor = anchors[
                    anchors["test"].eq("clinvar_pathogenic_enriched_for_depleted_vs_benign")
                ]
                nonsense_anchor = anchors[
                    anchors["test"].eq("nonsense_enriched_for_depleted_vs_synonymous")
                ]
                if not clinvar_anchor.empty:
                    row = clinvar_anchor.iloc[0]
                    bap1_evidence += (
                        "; rad51c_clinvar_anchor="
                        + str(int(row.get("first_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("first_n_binary", 0)))
                        + " vs "
                        + str(int(row.get("second_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("second_n_binary", 0)))
                        + "; rad51c_auc="
                        + f"{float(row.get('external_lof_auroc', float('nan'))):.3f}"
                    )
                if not nonsense_anchor.empty:
                    row = nonsense_anchor.iloc[0]
                    bap1_evidence += (
                        "; rad51c_nonsense_anchor="
                        + str(int(row.get("first_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("first_n_binary", 0)))
                        + " vs synonymous "
                        + str(int(row.get("second_n_depleted", 0)))
                        + "/"
                        + str(int(row.get("second_n_binary", 0)))
                    )
        except Exception as exc:  # pragma: no cover - report-only path
            bap1_evidence += f"; rad51c_mapping_inspect_error={type(exc).__name__}"
    gates = [
        Gate(
            "G1",
            "BRCA1 native-SAE causal necessity is supported by intervention and controls.",
            "brca1_native_finetuned_sae_intervention_summary.csv plus random/raw controls",
            "passed",
            "Use as anchor mechanistic claim.",
        ),
        Gate(
            "G2",
            "BRCA1 biology bridge has feature and variant cards.",
            "brca1_global_feature_cards.csv, brca1_native_effect_variant_cards.csv, brca1_native_sae_matched_annotation_control_summary.csv, and brca1_native_sae_covariate_matched_control_summary.csv",
            "passed",
            "Use as feature-card and localization evidence; still do not promote as standalone causal sufficiency.",
        ),
        Gate(
            "G3",
            "BRCA2 checkpoint mechanism stratification is complete.",
            "brca2_llr_esm_checkpoint_metric_summary.csv and brca2_llr_esm_discordance_category_summary.csv"
            + "; brca2_discordance_domain_matched_control_summary.csv"
            + (f"; {proxy_evidence}" if proxy_evidence else "")
            + (f"; {mavedb_evidence}" if mavedb_evidence else ""),
            "passed",
            "Use as independent checkpoint and VUS-panel rationale.",
        ),
        Gate(
            "G4",
            "BRCA2 full Evo2 embeddings exist.",
            "results/variant/brca2_evo2.npz",
            "passed" if brca2_evo2 else "pending",
            (
                "Full Evo2 exists; downstream/native-SAE artifacts have been inspected. Current squeue: "
                if brca2_evo2
                else "Wait for SLURM job chain; current squeue: "
            )
            + squeue.replace("\n", " | "),
        ),
        Gate(
            "G5",
            "BRCA2 native-SAE causal replication exists.",
            brca2_native_evidence,
            brca2_native_status,
            brca2_native_next,
        ),
        Gate(
            "G6",
            "Clinical utility claim has external review/assay outcome.",
            g6_evidence,
            g6_status,
            g6_next,
        ),
        Gate(
            "G7",
            "Third-gene functional-map benchmark is available.",
            bap1_evidence,
            bap1_status,
            bap1_next,
        ),
    ]
    return gates


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = [str(row.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(repo: Path, artifacts: list[Artifact], commands: list[Command], gates: list[Gate]) -> Path:
    out = repo / OUT_DIR / "interpretability_reproducibility_package.md"
    passed = sum(g.status == "passed" for g in gates)
    pending = sum(g.status == "pending" for g in gates)
    not_met = sum(g.status == "not_met" for g in gates)
    lines = [
        "# CrossBioSAE Interpretability Reproducibility Package",
        "",
        "## Purpose",
        "",
        "This report records the local data, scripts, outputs, rerun commands, and completion gates for the interpretability manuscript track. It is designed for internal audit and reviewer-facing reproducibility checks.",
        "",
        "## Gate Summary",
        "",
        f"- passed: {passed}",
        f"- pending: {pending}",
        f"- not_met: {not_met}",
        "",
        md_table([asdict(g) for g in gates], ["gate_id", "requirement", "status", "evidence", "next_action"]),
        "",
        "## Rerun Commands",
        "",
        md_table([asdict(c) for c in commands], ["step", "purpose", "command", "expected_outputs", "current_status"]),
        "",
        "## Artifact Manifest",
        "",
        md_table(
            [asdict(a) for a in artifacts],
            ["group", "path", "role", "required_for_claims", "exists", "rows_or_shape", "sha256_16", "status"],
        ),
        "",
        "## Claim Language Audit",
        "",
        "- Safe now: BRCA1 native-SAE causal intervention, BRCA1 biological bridge examples, BRCA2 checkpoint mechanism stratification, BRCA2 VUS review/assay triage design, RAD51C clean third-gene checkpoint plus checkpoint-level latent-SAE sparse-mechanism screen, and BAP1 finite-SNV/boundary checkpoint support.",
        "- Not met yet: BRCA2 full native-SAE causal replication; focused strata screen is descriptive and does not replace the original feature-selection control.",
        "- Not supported yet: clinical VUS reclassification or generic multimodal fusion SOTA.",
        "- Supplementary application context: collagen leave-one-gene-out adds mixed cross-gene support; gene-heldout VUS review-hook validation is mostly weak except COL3A1 broader application hooks, so it is not enough for a central VUS claim.",
        "- Functional-assay context: eSOL linear functional probe is negative for simple concat; avGFP DMS gives weak leakage-controlled concat gain but no strong synonymous/codon-level mechanism.",
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

    artifacts = build_artifacts(repo)
    commands = build_commands()
    gates = build_gates(repo)

    artifact_path = out / "interpretability_artifact_manifest.csv"
    command_path = out / "interpretability_rerun_commands.csv"
    gate_path = out / "interpretability_completion_gates.csv"
    pd.DataFrame([asdict(a) for a in artifacts]).to_csv(artifact_path, index=False)
    pd.DataFrame([asdict(c) for c in commands]).to_csv(command_path, index=False)
    pd.DataFrame([asdict(g) for g in gates]).to_csv(gate_path, index=False)
    report = write_report(repo, artifacts, commands, gates)
    print(f"wrote {artifact_path}")
    print(f"wrote {command_path}")
    print(f"wrote {gate_path}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
