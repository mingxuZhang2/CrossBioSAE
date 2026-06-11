#!/usr/bin/env python
"""Create a manuscript-facing figure plan and reviewer risk register."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def read_matrix(repo: Path) -> pd.DataFrame:
    path = repo / OUT_DIR / "interpretability_submission_evidence_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run summarize_interpretability_submission_evidence_matrix.py first."
        )
    return pd.read_csv(path)


def row(df: pd.DataFrame, claim_id: str) -> dict[str, str]:
    hit = df.loc[df["claim_id"].eq(claim_id)]
    if hit.empty:
        return {}
    return hit.iloc[0].fillna("").to_dict()


def build_figure_plan(claims: pd.DataFrame) -> pd.DataFrame:
    c = {claim_id: row(claims, claim_id) for claim_id in claims["claim_id"].tolist()}
    c9_ready = c.get("C9", {}).get("readiness", "").startswith("complete")
    c9_mixed = "mixed" in c.get("C9", {}).get("readiness", "") or "negative" in c.get("C9", {}).get("readiness", "")
    c9_key = c.get("C9", {}).get("key_result", "")
    c12_ready = "complete_checkpoint" in c.get("C12", {}).get("readiness", "")
    full_brca2_done = "brca2_evo2.npz=yes" in c9_key
    rows = [
        {
            "figure": "Figure 1",
            "panel": "1a",
            "title": "CrossBioSAE problem setup and evidence ladder",
            "primary_claims": "C1-C12 overview",
            "panel_message": "Top-journal interpretability requires functional labels, causal intervention, replication, controls, and downstream utility.",
            "data_or_artifact": "top_journal_evidence_map.md; interpretability_submission_evidence_matrix.csv",
            "required_plot_or_table": "Schematic plus compact evidence ladder",
            "status": "ready_for_drafting",
            "reviewer_question_answered": "Why is this more than a saliency visualization?",
            "next_action": "Draw schematic manually from the evidence matrix.",
        },
        {
            "figure": "Figure 2",
            "panel": "2a",
            "title": "BRCA1 native-SAE causal intervention and paired fold degradation",
            "primary_claims": "C1",
            "panel_message": c["C1"].get("key_result", ""),
            "data_or_artifact": "brca1_native_finetuned_sae_intervention_summary.csv; brca1_native_finetuned_sae_intervention_folds.csv",
            "required_plot_or_table": "Bar/point plot plus fold-paired lines: original, SAE recon, top-feature ablation AUROC/AUPRC",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Do sparse features causally matter to prediction?",
            "next_action": "Use fig2_brca1_native_sae.png as the current manuscript draft panel.",
        },
        {
            "figure": "Figure 2",
            "panel": "2b",
            "title": "BRCA1 intervention controls",
            "primary_claims": "C1",
            "panel_message": c["C1"].get("controls_or_comparators", ""),
            "data_or_artifact": "brca1_native_finetuned_sae_intervention_random_ablation.csv; brca1_raw_zdna_dim_control_summary.csv",
            "required_plot_or_table": "Null distribution with observed top-feature delta; raw-dimension control inset",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Is the drop just model fragility or arbitrary dimension removal?",
            "next_action": "Plot random-feature null and raw-dimension control side by side.",
        },
        {
            "figure": "Figure 2",
            "panel": "2c",
            "title": "BRCA1 dose response, addback rescue, and feature-only boundary",
            "primary_claims": "C1,C2",
            "panel_message": "Feature-ablation effect increases with top-k dose; selected addback beats matched-random addback, while selected-feature-only decoding does not beat matched random feature-only controls.",
            "data_or_artifact": "brca1_native_finetuned_sae_intervention_dose_response.csv; brca1_native_finetuned_sae_rescue_summary.csv; brca1_native_finetuned_sae_rescue_rescue_controls.csv; brca1_native_finetuned_sae_rescue_sufficiency_controls.csv",
            "required_plot_or_table": "Dose-response line, selected-minus-random addback-rescue null, and feature-only AUROC boundary bars",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Is the result specific and reversible without overclaiming feature-only sufficiency?",
            "next_action": "Use addback rescue as a reverse-direction specificity control and state feature-only insufficiency explicitly.",
        },
        {
            "figure": "Figure 3",
            "panel": "3a",
            "title": "BRCA1 native-effect biological localization",
            "primary_claims": "C2",
            "panel_message": c["C2"].get("key_result", ""),
            "data_or_artifact": "brca1_native_sae_biological_bridge_summary.csv; brca1_native_sae_biological_bridge_strata.csv; brca1_native_sae_matched_annotation_control_summary.csv; brca1_native_sae_covariate_matched_control_summary.csv",
            "required_plot_or_table": "Top-decile enrichment plot, consequence/domain strata, protein-coordinate examples, consequence-region matched null, CADD/phyloP covariate-matched control, and covariate balance panel",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Do high-effect SAE variants align with functional biology?",
            "next_action": "Use Figure 3 as the C2 localization figure; optional 3D structure mapping can be added if reviewers require it.",
        },
        {
            "figure": "Figure 3",
            "panel": "3b",
            "title": "Global-SAE to native-effect bridge",
            "primary_claims": "C2",
            "panel_message": "Global features predict native-effect above random and permutation controls but do not beat annotation baseline.",
            "data_or_artifact": "brca1_global_sae_native_effect_bridge_summary.csv",
            "required_plot_or_table": "Observed vs random/permuted AUROC and Spearman bars",
            "status": "ready_with_caveat",
            "reviewer_question_answered": "Can global interpretable features recover fold-local causal effects?",
            "next_action": "State explicitly as bridge evidence, not sufficiency.",
        },
        {
            "figure": "Figure 4",
            "panel": "4a",
            "title": "BRCA2 independent SGE checkpoint",
            "primary_claims": "C3",
            "panel_message": c["C3"].get("key_result", ""),
            "data_or_artifact": "brca2_llr_esm_checkpoint_metric_summary.csv",
            "required_plot_or_table": "Grouped bars: Evo2 LLR, ESM-only, ESM+LLR across all/missense/non-missense/domains",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Does the method transfer to independent BRCA2 SGE data?",
            "next_action": "Plot checkpoint metrics while clearly labeling native-SAE as pending.",
        },
        {
            "figure": "Figure 4",
            "panel": "4b",
            "title": "BRCA2 fusion negative control",
            "primary_claims": "C3",
            "panel_message": c["C3"].get("controls_or_comparators", ""),
            "data_or_artifact": "brca2_llr_esm_checkpoint_bootstrap_delta.csv",
            "required_plot_or_table": "Bootstrap delta AUROC forest plot",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Is this paper claiming generic multimodal prediction gain?",
            "next_action": "Use this as a guardrail: mechanism not SOTA fusion.",
        },
        {
            "figure": "Figure 4",
            "panel": "4c",
            "title": "BRCA2 DNA/protein concordance mechanism map",
            "primary_claims": "C4",
            "panel_message": c["C4"].get("key_result", ""),
            "data_or_artifact": "brca2_llr_esm_discordance_category_summary.csv; brca2_llr_esm_discordance_domain_summary.csv; brca2_external_proxy_validation_known_clinvar.csv; brca2_discordance_domain_matched_control_summary.csv; brca2_mavedb_external_assay_category_tests.csv",
            "required_plot_or_table": "2D DNA vs protein score scatter/hexbin with quadrant LOF rates, exact ClinVar proxy, domain facets, BRCA2-domain matched non-both-high null, and public MaveDB external-assay stratum support",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "What does interpretation add beyond scalar prediction?",
            "next_action": "Use the domain-matched null to show both-high enrichment is not just CTDB-domain composition.",
        },
        {
            "figure": "Figure 5",
            "panel": "5a",
            "title": "BRCA2 VUS triage tiers",
            "primary_claims": "C5",
            "panel_message": c["C5"].get("key_result", ""),
            "data_or_artifact": "brca2_clinvar_interpretability_candidates_tier_summary.csv; brca2_clinvar_temporal_proxy_validation_candidate_tests.csv",
            "required_plot_or_table": "Tier table: pathogenic-review, benign-review, split-mechanism, conflict controls, and archive-current ClinVar temporal-proxy comparison",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "What is the downstream use of interpretability?",
            "next_action": "Phrase as review/assay prioritization, not clinical reclassification.",
        },
        {
            "figure": "Figure 5",
            "panel": "5b",
            "title": "Prospective BRCA2 review/assay panel",
            "primary_claims": "C6",
            "panel_message": c["C6"].get("key_result", ""),
            "data_or_artifact": "brca2_prospective_followup_panel_summary.csv; brca2_assay_statistical_plan_fisher_thresholds.csv; brca2_blinded_validation_status.csv; brca2_external_proxy_validation_panel_proxy_tests.csv; brca2_clinvar_temporal_proxy_validation_panel_tests.csv; brca2_mavedb_external_assay_panel_tests.csv",
            "required_plot_or_table": "Panel-arm composition, Fisher threshold plot, locked-outcome readiness, same-residue proxy support, archive-current panel proxy sensitivity, and HAP1 public functional-assay support including matched-subset binary endpoint",
            "status": "ready_for_plotting",
            "reviewer_question_answered": "Can the interpretation produce an executable application?",
            "next_action": "Export manifest to collaborator or use it as a proposed validation panel.",
        },
        {
            "figure": "Supplementary Figure S1",
            "panel": "S1",
            "title": "Collagen-glycine feature case study",
            "primary_claims": "C7",
            "panel_message": c["C7"].get("key_result", ""),
            "data_or_artifact": "collagen_gly_random_feature_null_summary.csv; collagen_leave_one_gene_out_primary_summary.csv; vus_application_readiness_counts.csv; collagen_gene_heldout_vus_validation_summary.csv",
            "required_plot_or_table": "Known pathogenic control enrichment, leave-one-gene-out transfer table, VUS review-hook table, and gene-heldout VUS random-null summary",
            "status": "supplement_only_until_external_validation",
            "reviewer_question_answered": "Does the approach generalize to a recognizable structural grammar?",
            "next_action": "Keep as supplementary unless temporal/disease-database validation improves.",
        },
        {
            "figure": "Supplementary Figure S2",
            "panel": "S2",
            "title": "BAP1/RAD51C third-gene functional-map benchmarks",
            "primary_claims": "C12",
            "panel_message": c.get("C12", {}).get("key_result", ""),
            "data_or_artifact": (
                "fig_s2_third_gene_benchmarks.png; third_gene_interpretability_application_plan.md; third_gene_review_assay_panel.md; third_gene_review_assay_panel_blinded.csv; third_gene_assay_statistical_plan.md; third_gene_assay_statistical_thresholds.csv; third_gene_application_decision_gates.csv; bap1_sge_data_preparation.md; bap1_sge_external_anchors_tests.csv; bap1_sge_baseline_predictors_metric_summary.csv; bap1_llr_esm_checkpoint_metric_summary.csv; bap1_llr_esm_checkpoint_bootstrap_delta.csv; bap1_llr_esm_discordance_category_summary.csv; rad51c_grch38_mapping.md; rad51c_sge_external_anchors_tests.csv; rad51c_llr_esm_checkpoint_metric_summary.csv; rad51c_llr_esm_checkpoint_bootstrap_delta.csv; rad51c_llr_esm_discordance_category_summary.csv"
                if c12_ready
                else "fig_s2_third_gene_benchmarks.png; third_gene_interpretability_application_plan.md; third_gene_review_assay_panel.md; third_gene_review_assay_panel_blinded.csv; third_gene_assay_statistical_plan.md; third_gene_assay_statistical_thresholds.csv; third_gene_application_decision_gates.csv; bap1_sge_data_preparation.md; bap1_sge_external_anchors_tests.csv; bap1_sge_baseline_predictors_metric_summary.csv; rad51c_grch38_mapping.md; rad51c_sge_external_anchors_tests.csv; pending bap1/rad51c checkpoint metrics"
            ),
            "required_plot_or_table": (
                "BAP1/RAD51C ClinVar and consequence anchors; BAP1 public EVE/SIFT/PolyPhen missense-baseline AUROCs; RAD51C checkpoint metrics; BAP1 finite-SNV checkpoint metrics; both-high/both-low mechanism strata; explicit boundary note that third-gene native-SAE causal replication is pending."
                if c12_ready
                else "BAP1/RAD51C ClinVar and consequence anchors; BAP1 public EVE/SIFT/PolyPhen missense-baseline AUROCs; explicit pending-boundary note for Evo2/ESM/CrossBioSAE checkpoint table."
            ),
            "status": "third_gene_checkpoints_ready_native_sae_pending" if c12_ready else "anchor_figure_ready_checkpoint_pending",
            "reviewer_question_answered": "Does the application and mechanism-evidence ladder generalize beyond BRCA1/BRCA2?",
            "next_action": (
                "Use RAD51C as the clean third-gene checkpoint/mechanism-stratification panel and BAP1 as finite-SNV checkpoint/boundary support; keep third-gene native-SAE causal replication as the pending gate."
                if c12_ready
                else "Wait for BAP1/RAD51C Evo2 LLR SLURM outputs and run the dependent checkpoints before promoting this from benchmark readiness to CrossBioSAE validation."
            ),
        },
        {
            "figure": "Figure 6" if c9_ready else "Future decisive figure",
            "panel": "F6",
            "title": "BRCA2 native-SAE causal replication",
            "primary_claims": "C9",
            "panel_message": c["C9"].get("key_result", ""),
            "data_or_artifact": (
                "results/variant/brca2_evo2.npz; brca2_native_finetuned_sae_intervention_summary.csv"
                if (c9_ready or c9_mixed)
                else "results/variant/brca2_evo2.npz; pending BRCA2 native-SAE outputs"
            ),
            "required_plot_or_table": "Same layout as Figure 2 for BRCA2",
            "status": "ready_for_error_analysis" if c9_mixed else "ready_for_plotting" if c9_ready else "pending_gpu_chain",
            "reviewer_question_answered": "Does the causal SAE mechanism replicate outside BRCA1?",
            "next_action": (
                "Plot BRCA2 native-SAE intervention beside BRCA1 and update the replication narrative."
                if c9_ready
                else "Plot BRCA2 native-SAE as a mixed replication/error-analysis result; do not promote it as decisive proof."
                if c9_mixed
                else (
                    "Wait for downstream/native-SAE chain, then regenerate evidence matrix."
                    if full_brca2_done
                    else "Wait for full Evo2/downstream/native-SAE chain, then regenerate evidence matrix."
                )
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_risk_register(claims: pd.DataFrame) -> pd.DataFrame:
    c = {claim_id: row(claims, claim_id) for claim_id in claims["claim_id"].tolist()}
    c9_ready = c.get("C9", {}).get("readiness", "").startswith("complete")
    c9_mixed = "mixed" in c.get("C9", {}).get("readiness", "") or "negative" in c.get("C9", {}).get("readiness", "")
    c12_ready = "complete_checkpoint" in c.get("C12", {}).get("readiness", "")
    rows = [
        {
            "risk_id": "R1",
            "risk": "Reviewer says SAE explanation is post-hoc visualization.",
            "severity": "high",
            "current_answer": c["C1"].get("key_result", ""),
            "artifact": "brca1_native_finetuned_sae_intervention_summary.csv",
            "remaining_gap": "Need concise plots and exact intervention protocol in Methods.",
            "mitigation": "Lead with ablation/rescue/random/permutation controls, not feature cards.",
        },
        {
            "risk_id": "R2",
            "risk": "Reviewer asks whether BRCA1 result is gene-specific.",
            "severity": "high",
            "current_answer": c["C9"].get("key_result", "") if (c9_ready or c9_mixed) else c["C4"].get("key_result", ""),
            "artifact": (
                "brca2_native_finetuned_sae_intervention_summary.csv"
                if (c9_ready or c9_mixed)
                else "brca2_llr_esm_discordance_category_summary.csv"
            ),
            "remaining_gap": (
                "External review or assay outcome is still needed for clinical utility."
                if c9_ready
                else "Native-SAE shows a positive broad ablation effect but fails label-permutation significance; focused strata screen is descriptive and does not provide a narrow biological replication claim."
                if c9_mixed
                else "BRCA2 native-SAE causal intervention is still pending."
            ),
            "mitigation": (
                "Present BRCA2 as independent mechanistic replication, while keeping clinical claims separate."
                if c9_ready
                else "Present BRCA2 native-SAE as mixed/error-analysis evidence and keep BRCA2's main role as mechanism stratification plus VUS triage."
                if c9_mixed
                else "State BRCA2 checkpoint now; upgrade to replication only after C9 completes."
            ),
        },
        {
            "risk_id": "R3",
            "risk": "Reviewer expects prediction improvement from multimodal fusion.",
            "severity": "medium",
            "current_answer": c["C3"].get("controls_or_comparators", ""),
            "artifact": "brca2_llr_esm_checkpoint_bootstrap_delta.csv",
            "remaining_gap": "Need clear framing in abstract/results.",
            "mitigation": "Explicitly position as mechanism-aware interpretability, not generic fusion SOTA.",
        },
        {
            "risk_id": "R4",
            "risk": "Clinical reviewers object to VUS reclassification language.",
            "severity": "high",
            "current_answer": c["C5"].get("key_result", ""),
            "artifact": "brca2_clinvar_interpretability_candidates_tier_summary.csv",
            "remaining_gap": "No blinded expert review or new wet-lab assay outcome yet.",
            "mitigation": "Use review/assay triage language only; include benign and model-conflict controls.",
        },
        {
            "risk_id": "R5",
            "risk": "Application seems descriptive rather than actionable.",
            "severity": "medium",
            "current_answer": c["C6"].get("key_result", ""),
            "artifact": "brca2_review_assay_protocol_manifest.csv",
            "remaining_gap": "Assay/review execution is external.",
            "mitigation": "Show complete manifest, arm hypotheses, Fisher thresholds, and power grid.",
        },
        {
            "risk_id": "R6",
            "risk": "Collagen VUS case study fails set-level enrichment.",
            "severity": "medium",
            "current_answer": c["C7"].get("key_result", ""),
            "artifact": "collagen_gly_vus_random_feature_summary.csv",
            "remaining_gap": "VUS enrichment p-value is not significant.",
            "mitigation": "Demote collagen to supplement; emphasize known pathogenic controls and review hooks only.",
        },
        {
            "risk_id": "R7",
            "risk": "Temporal ClinVar validation may be inflated by skewed known-label set.",
            "severity": "medium",
            "current_answer": c["C8"].get("key_result", ""),
            "artifact": "clinvar_pseudo_temporal_2025-01_metric_summary.csv",
            "remaining_gap": "Current temporal set has high current pathogenic rate and weak specificity.",
            "mitigation": "Treat as context; prefer BRCA2 archived-status prospective panel for main application.",
        },
        {
            "risk_id": "R8",
            "risk": "SAE feature interpretability lacks feature-level biological names.",
            "severity": "medium",
            "current_answer": c["C2"].get("key_result", ""),
            "artifact": "brca1_native_sae_biological_bridge_feature_correlations.csv; brca1_global_feature_cards.csv; brca1_native_effect_variant_cards.csv; fig3_brca1_biological_bridge.png",
            "remaining_gap": "Figure-ready feature cards, domain panels, and protein-coordinate examples are present; remaining gap is optional 3D structural interface mapping.",
            "mitigation": "Pair feature cards with the causal intervention in Figure 2 and keep bridge wording as localization/naming support, not standalone proof.",
        },
        {
            "risk_id": "R9",
            "risk": "Third-gene benchmark is mistaken for completed CrossBioSAE validation.",
            "severity": "high",
            "current_answer": c.get("C12", {}).get("key_result", ""),
            "artifact": (
                "bap1_sge_external_anchors.md; bap1_sge_baseline_predictors.md; bap1_llr_esm_checkpoint_metric_summary.csv; bap1_llr_esm_discordance_category_summary.csv; rad51c_sge_external_anchors.md; rad51c_llr_esm_checkpoint_metric_summary.csv; rad51c_llr_esm_discordance_category_summary.csv"
                if c12_ready
                else "bap1_sge_external_anchors.md; bap1_sge_baseline_predictors.md; rad51c_sge_external_anchors.md; pending bap1/rad51c checkpoint metric summaries"
            ),
            "remaining_gap": (
                "RAD51C checkpoint is positive and BAP1 finite-SNV checkpoint is positive/boundary, but third-gene native-SAE causal intervention and locked review/assay outcomes are still missing."
                if c12_ready
                else "BAP1 and RAD51C data/anchor evidence are complete enough to justify the benchmarks, but Evo2/ESM/CrossBioSAE checkpoint metrics are still pending."
            ),
            "mitigation": (
                "Promote RAD51C as checkpoint/mechanism stratification and BAP1 as finite-SNV checkpoint/boundary evidence; reserve causal CrossBioSAE replication for a BRCA1-style native-SAE intervention."
                if c12_ready
                else "Keep BAP1/RAD51C in benchmark-ready/pending status until checkpoint metrics exist; promote only if third-gene checkpoints beat or usefully explain public baselines under position-grouped evaluation."
            ),
        },
    ]
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, item in df.iterrows():
        vals = [str(item[col]).replace("\n", " ").replace("|", "\\|") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(repo: Path, claims: pd.DataFrame, figures: pd.DataFrame, risks: pd.DataFrame) -> Path:
    out = repo / OUT_DIR / "interpretability_manuscript_package.md"
    ready_figs = figures.loc[~figures["status"].str.contains("pending", case=False, na=False)]
    pending_figs = figures.loc[figures["status"].str.contains("pending", case=False, na=False)]

    lines = [
        "# CrossBioSAE Interpretability Manuscript Package",
        "",
        "## Core Story",
        "",
        "CrossBioSAE should be written as a mechanism-aware interpretability and variant-review/assay-triage paper. The central evidence is not that DNA+protein fusion gives a generic prediction gain. The central evidence is that sparse, fold-native features can be causally intervened on in BRCA1, and that the same evidence ladder now produces BRCA2 mechanism strata and an executable VUS follow-up panel.",
        "",
        "## Figure Plan",
        "",
        md_table(
            figures,
            [
                "figure",
                "panel",
                "title",
                "primary_claims",
                "status",
                "panel_message",
                "data_or_artifact",
                "next_action",
            ],
        ),
        "",
        "## Ready Figures",
        "",
    ]
    for _, item in ready_figs.iterrows():
        lines.append(f"- {item['figure']} {item['panel']}: {item['title']}")
    lines.extend(["", "## Pending Decisive Figure", ""])
    for _, item in pending_figs.iterrows():
        lines.append(f"- {item['figure']} {item['panel']}: {item['next_action']}")
    lines.extend(
        [
            "",
            "## Reviewer Risk Register",
            "",
            md_table(
                risks,
                [
                    "risk_id",
                    "severity",
                    "risk",
                    "current_answer",
                    "remaining_gap",
                    "mitigation",
                ],
            ),
            "",
            "## Claim Guardrails",
            "",
            "- Safe main claim: BRCA1 sparse-feature interventions provide causal interpretability evidence for functional variant prediction.",
            "- Safe BRCA2 claim now: independent BRCA2 SGE checkpoint supports DNA/protein mechanism stratification and VUS assay-panel design.",
            "- Safe BAP1/RAD51C claim now: RAD51C provides a positive third-gene Evo2/ESM checkpoint and DNA/protein mechanism-stratification result; BAP1 provides positive finite-SNV checkpoint/boundary support; both remain credible benchmark/application anchors.",
            "- Unsafe until C9 completes: BRCA2 native-SAE causal replication.",
            "- Unsafe until third-gene native-SAE intervention completes: BAP1/RAD51C causal CrossBioSAE replication.",
            "- Unsafe without external review/assay: clinical reclassification of VUS.",
            "- Unsafe given current checkpoint: generic multimodal fusion SOTA.",
            "",
            "## Inputs",
            "",
            "- interpretability_submission_evidence_matrix.csv",
            "- interpretability_agent_team_synthesis.md",
            "- third_gene_interpretability_application_plan.md",
            "- third_gene_review_assay_panel.md",
            "- third_gene_assay_statistical_plan.md",
            "- top_journal_evidence_map.md",
            "- BRCA1 native-SAE intervention/control tables",
            "- BRCA2 LLR/ESM checkpoint, discordance, VUS tier, and assay-plan tables",
            "- BAP1 SGE preparation, external-anchor, public-baseline, checkpoint, and discordance tables",
            "- RAD51C SGE preparation, external-anchor, Evo2/ESM checkpoint, and discordance tables",
            "",
            "## Manuscript Draft",
            "",
            "The narrative manuscript skeleton is generated by",
            "`scripts/summarize_interpretability_manuscript_draft.py`:",
            "",
            "- `interpretability_manuscript_draft.md`",
            "- `interpretability_manuscript_draft_sections.csv`",
            "",
            "Use the draft as a claim-bounded writing scaffold. It should be regenerated",
            "after BRCA2 full Evo2/downstream/native-SAE completion, after third-gene",
            "native-SAE completion, and after any blinded review or assay readout is added.",
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
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    claims = read_matrix(repo)
    figures = build_figure_plan(claims)
    risks = build_risk_register(claims)

    fig_path = out / "interpretability_manuscript_figure_plan.csv"
    risk_path = out / "interpretability_reviewer_risk_register.csv"
    figures.to_csv(fig_path, index=False)
    risks.to_csv(risk_path, index=False)
    report_path = write_report(repo, claims, figures, risks)

    print(f"wrote {fig_path}")
    print(f"wrote {risk_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
