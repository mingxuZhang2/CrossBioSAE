#!/usr/bin/env python
"""Map top-journal interpretability proof patterns to current artifacts.

The report is intentionally conservative: it separates what current artifacts
prove from what would still be needed for a top-journal mechanism/application
claim. Literature entries are static exemplars; local evidence is read from the
current result tables so the report can be regenerated after queued GPU jobs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


@dataclass
class Exemplar:
    pattern_id: str
    exemplar: str
    venue_year: str
    source_url: str
    data_used: str
    proof_standard: str
    application_proof: str
    implication_for_crossbiosae: str


@dataclass
class EvidenceRow:
    axis: str
    top_journal_standard: str
    current_crossbiosae_evidence: str
    current_status: str
    missing_or_risk: str
    decisive_next_action: str
    local_artifacts: str


@dataclass
class Action:
    priority: int
    action_id: str
    objective: str
    success_rule: str
    why_it_matters: str
    commands_or_artifacts: str
    status: str


def fmt(x: Any, digits: int = 3, sci: bool = False) -> str:
    if x is None or pd.isna(x):
        return "NA"
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return "NA"
    if sci:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def read_csv(repo: Path, name: str) -> pd.DataFrame:
    path = repo / OUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row(df: pd.DataFrame) -> dict[str, Any]:
    return {} if df.empty else df.iloc[0].to_dict()


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


def file_state(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.exists():
        return f"{rel}=missing"
    return f"{rel}=present"


def exemplars() -> list[Exemplar]:
    return [
        Exemplar(
            "P1",
            "EVE disease-variant prediction",
            "Nature 2021",
            "https://www.nature.com/articles/s41586-021-04043-8",
            "ClinVar labels, MAVE/high-throughput functional assays, population-frequency resources, and 3,219 disease genes.",
            "Benchmark against clinical labels and functional experiments while avoiding direct label dependence; report broad gene-scale generalization.",
            "Prioritize VUS and provide independent evidence for variant interpretation at scale.",
            "CrossBioSAE needs external labels plus functional assays, not only internal feature association; VUS/prioritization claims should be tied to independent evidence.",
        ),
        Exemplar(
            "P2",
            "Enformer sequence-to-expression model",
            "Nature Methods 2021",
            "https://www.nature.com/articles/s41592-021-01252-x",
            "Large epigenomic expression tracks, GTEx eQTL summary statistics, fine-mapped eQTLs, and motif-level in silico mutagenesis.",
            "Show improved predictive accuracy, concordance with eQTL statistics, fine-mapped causal-variant discrimination, and motif/mechanism examples.",
            "Use model predictions to improve causal regulatory variant prioritization and explain a locus-level mechanism.",
            "A top-journal interpretability claim should include feature intervention, matched nulls, and at least one locus/gene case where the feature explains a known mechanism.",
        ),
        Exemplar(
            "P3",
            "DeepSEA noncoding variant effects",
            "Nature Methods 2015",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4768299/",
            "ENCODE/Roadmap chromatin profiles with regulatory mutations, eQTLs, and GWAS variants as external standards.",
            "Train on high-throughput functional genomic tracks and validate variant prioritization against independent regulatory/trait associations.",
            "Score noncoding variants by predicted effects on TF binding, chromatin accessibility, histone marks, eQTL, and GWAS relevance.",
            "For CrossBioSAE, DNA/protein feature effects should be externally anchored to ClinVar, SGE, gnomAD, or known mechanism strata.",
        ),
        Exemplar(
            "P4",
            "AlphaMissense and PrimateAI-3D",
            "Science 2023 / Nature Genetics 2023",
            "https://www.science.org/doi/10.1126/science.adg7492; https://www.nature.com/articles/s41588-023-01455-2",
            "ClinVar, de novo disease variants, population/common-variant depletion, protein structure/MSA context, MAVE and independent patient cohorts.",
            "Compare to many existing predictors and test generalization on independent clinical/experimental cohorts.",
            "Genome-scale pathogenicity scoring with calibrated categories or improved disease/benign discrimination.",
            "Our method needs public-predictor baselines per gene, especially AlphaMissense/EVE/CADD/REVEL-like comparisons for missense claims.",
        ),
        Exemplar(
            "P5",
            "BAP1 and VHL saturation genome editing studies",
            "Nature Genetics 2024",
            "https://www.nature.com/articles/s41588-024-01799-3; https://www.nature.com/articles/s41588-024-01800-z",
            "Endogenous SGE maps, ClinVar truth sets, gnomAD/cancer/pedigree context, consequence classes, and assay-calibrated thresholds.",
            "Quantify sensitivity/specificity, likelihood ratios/evidence strengths, clinical-class concordance, and disease-mechanism strata.",
            "Convert functional maps into variant-classification evidence and phenotype/cancer interpretation.",
            "RAD51C/BAP1 are right benchmark genes; CrossBioSAE should state assay-calibrated effect strengths and use blinded/locked review endpoints.",
        ),
        Exemplar(
            "P6",
            "MaveDB/ClinMAVE clinical MAVE infrastructure",
            "Genome Biology 2025 / clinical MAVE resources",
            "https://link.springer.com/article/10.1186/s13059-025-03476-y; https://pmc.ncbi.nlm.nih.gov/articles/PMC12807648/",
            "Curated MAVE/SGE assays across many genes, linked to ClinVar, gnomAD, and ACMG/AMP-style evidence frameworks.",
            "Standardize assay metadata, evidence calibration, clinical annotations, and reproducible exports.",
            "Make variant-effect maps usable for clinical interpretation and future curation.",
            "Our application panels need locked IDs, assay endpoints, evidence-strength thresholds, and provenance rather than only ranked examples.",
        ),
    ]


def build_evidence(repo: Path) -> list[EvidenceRow]:
    brca1_native = first_row(read_csv(repo, "brca1_native_finetuned_sae_intervention_summary.csv"))
    brca1_rescue = first_row(read_csv(repo, "brca1_native_finetuned_sae_rescue_summary.csv"))
    brca1_bridge = first_row(read_csv(repo, "brca1_native_sae_biological_bridge_summary.csv"))
    brca2_native = first_row(read_csv(repo, "brca2_native_finetuned_sae_intervention_summary.csv"))
    brca2_panel = read_csv(repo, "brca2_prospective_followup_panel_summary.csv")
    brca2_temporal = first_row(read_csv(repo, "brca2_clinvar_temporal_proxy_validation_panel_tests.csv"))
    bap1_metrics = read_csv(repo, "bap1_llr_esm_checkpoint_metric_summary.csv")
    bap1_delta = read_csv(repo, "bap1_llr_esm_checkpoint_bootstrap_delta.csv")
    rad_metrics = read_csv(repo, "rad51c_llr_esm_checkpoint_metric_summary.csv")
    rad_delta = read_csv(repo, "rad51c_llr_esm_checkpoint_bootstrap_delta.csv")
    rad_disc = read_csv(repo, "rad51c_llr_esm_discordance_category_summary.csv")
    rad_latent = first_row(read_csv(repo, "rad51c_latent_sae_intervention_summary.csv"))
    rad_latent_strata = read_csv(repo, "rad51c_latent_sae_intervention_stratum_summary.csv")
    rad_app = read_csv(repo, "rad51c_latent_sae_application_arm_summary.csv")

    rad_all = get_row(rad_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    rad_llr = get_row(rad_metrics, scope="all", metric="Evo2 LLR zero-shot")
    rad_missense_delta = get_row(rad_delta, comparison="ESM+LLR minus LLR-CV", scope="missense")
    rad_both_high = get_row(rad_disc, category="both_high")
    rad_missense_latent = get_row(rad_latent_strata, stratum="missense")
    rad_panel_n = int(pd.to_numeric(rad_app.get("selected_n", pd.Series(dtype=float)), errors="coerce").sum()) if not rad_app.empty else 0
    bap1_all = get_row(bap1_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    bap1_all_delta = get_row(bap1_delta, comparison="ESM+LLR minus LLR-CV", scope="all")
    if not brca2_panel.empty and "selected_n" in brca2_panel.columns:
        brca2_panel_n = int(pd.to_numeric(brca2_panel["selected_n"], errors="coerce").sum())
    elif not brca2_panel.empty and "n" in brca2_panel.columns:
        brca2_panel_n = int(pd.to_numeric(brca2_panel["n"], errors="coerce").sum())
    else:
        brca2_panel_n = 0
    temporal_known = (
        "positive known pathogenic "
        + fmt(brca2_temporal.get("positive_current_pathogenic_known_only"), 0)
        + "/"
        + fmt(brca2_temporal.get("n_positive_current_exact_known"), 0)
        + ", negative known pathogenic "
        + fmt(brca2_temporal.get("negative_current_pathogenic_known_only"), 0)
        + "/"
        + fmt(brca2_temporal.get("n_negative_current_exact_known"), 0)
        + ", missing-as-not-pathogenic Fisher p="
        + fmt(brca2_temporal.get("missing_as_not_pathogenic_fisher_p_greater"))
        if brca2_temporal
        else "temporal proxy pending"
    )

    brca1_claim = (
        "BRCA1 native SAE: recon AUROC "
        + fmt(brca1_native.get("native_sae_recon_auroc"))
        + ", top-feature ablation "
        + fmt(brca1_native.get("top_feature_ablate_auroc"))
        + ", delta "
        + fmt(brca1_native.get("delta_auroc_recon_minus_top_ablate"))
        + "; rescue delta "
        + fmt(brca1_rescue.get("mean_rescue_delta_auc_selected_minus_random"))
        + "; biological bridge high-effect LOF rate "
        + fmt(brca1_bridge.get("lof_rate_high_effect"))
        + " vs rest "
        + fmt(brca1_bridge.get("lof_rate_rest"))
        + "."
        if brca1_native
        else "BRCA1 native SAE summary missing."
    )

    brca2_claim = (
        "BRCA2 has mechanism/application artifacts and panel rows n="
        + str(brca2_panel_n)
        + "; temporal/proxy panel "
        + temporal_known
        + "; native-SAE strict replication delta "
        + fmt(brca2_native.get("delta_auroc_recon_minus_top_ablate"))
        + " with random p="
        + fmt(brca2_native.get("empirical_p_random_delta_ge_top_mean"))
        + ", label-permutation p="
        + fmt(brca2_native.get("empirical_p_label_permuted_delta_ge_top_mean"))
        + ", rescue p(delta<=0)="
        + fmt(brca2_native.get("empirical_p_rescue_delta_auc_le_zero"))
        + "."
        if brca2_native or brca2_panel_n
        else "BRCA2 application/native summaries missing."
    )

    rad_claim = (
        "RAD51C checkpoint: LLR AUROC "
        + fmt(rad_llr.get("auroc_for_sge_lof"))
        + " vs ESM+LLR "
        + fmt(rad_all.get("auroc_for_sge_lof"))
        + "; missense fusion delta "
        + fmt(rad_missense_delta.get("median_delta_auroc"))
        + " CI ["
        + fmt(rad_missense_delta.get("ci_lo"))
        + ", "
        + fmt(rad_missense_delta.get("ci_hi"))
        + "]; both-high LOF rate "
        + fmt(rad_both_high.get("lof_rate"))
        + "; latent-SAE delta "
        + fmt(rad_latent.get("delta_auroc_recon_minus_top_ablate"))
        + ", missense delta "
        + fmt(rad_missense_latent.get("delta_auroc_recon_minus_top_ablate"))
        + ", random p="
        + fmt(rad_latent.get("empirical_p_random_delta_ge_top_mean"), 4)
        + "; application panel n="
        + str(rad_panel_n)
        + "."
        if rad_latent
        else "RAD51C checkpoint/latent summaries missing."
    )

    return [
        EvidenceRow(
            "Predictive benchmark",
            "Independent labels or assays; compare to strong public baselines; use grouped/temporal/external splits.",
            rad_claim + " BAP1 checkpoint all ESM+LLR AUROC " + fmt(bap1_all.get("auroc_for_sge_lof")) + ", delta " + fmt(bap1_all_delta.get("median_delta_auroc")) + ".",
            "strong_for_RAD51C_checkpoint; bounded_for_BAP1_finite_SNV",
            "RAD51C still lacks AlphaMissense/EVE/CADD/REVEL-style baseline comparison; BAP1 zero-shot LLR is finite-SNV/imputed for many variants.",
            "Add RAD51C public missense baselines and report ESM+LLR gain under GroupKFold by position; keep BAP1 caveat explicit.",
            "rad51c_llr_esm_checkpoint_metric_summary.csv; bap1_llr_esm_checkpoint_metric_summary.csv",
        ),
        EvidenceRow(
            "Causal feature necessity",
            "Feature intervention must preserve reconstruction, then deletion of selected features should beat matched random, label-permuted, bottom-feature, and rescue controls.",
            brca1_claim + " " + rad_claim,
            "complete_for_BRCA1; positive_latent_screen_for_RAD51C; native_pending_for_RAD51C; failed_or_boundary_for_BRCA2",
            "RAD51C latent-SAE is checkpoint-level, not native gate-head SAE. BRCA2 native-SAE did not pass strict replication.",
            "Promote BRCA1 as current causal anchor; wait for RAD51C all-variant and missense-only native-SAE jobs before upgrading third-gene causal claim.",
            "brca1_native_finetuned_sae_intervention_summary.csv; rad51c_latent_sae_intervention_summary.csv; rad51c_native_finetuned_sae_intervention_missense_summary.csv",
        ),
        EvidenceRow(
            "Biological localization",
            "Mechanism should localize to known domains, motifs, consequence classes, or orthogonal biological axes and survive annotation/covariate controls.",
            "BRCA1 biological bridge and matched controls exist. RAD51C latent-SAE localizes to missense/coding with synonymous/nonsense/UTR deltas near zero; RAD51C both-high missense LOF rate " + fmt(rad_both_high.get("lof_rate")) + ".",
            "strong_for_BRCA1; strong_checkpoint_localization_for_RAD51C",
            "RAD51C native feature IDs are not yet available; biological feature cards require native or global-SAE mapping after strict replication.",
            "After RAD51C native output, build feature cards for top native features and domain/consequence/covariate controls.",
            "brca1_native_sae_biological_bridge_summary.csv; rad51c_latent_sae_intervention_stratum_summary.csv",
        ),
        EvidenceRow(
            "Downstream application",
            "Ranked variants should be converted into locked review/assay panels with positive, negative, mechanism-conflict, and control arms plus prespecified endpoints.",
            "RAD51C latent application panel n=" + str(rad_panel_n) + "; BRCA2 prospective panel n=" + str(brca2_panel_n) + ". " + brca2_claim,
            "panel_design_complete; external_readout_pending",
            "No locked blinded expert-review or new assay outcome has been returned yet, so clinical utility cannot be claimed.",
            "Freeze RAD51C/BRCA2 panel IDs and assay/review endpoints; run blinded review, temporal ClinVar update, or public MAVE holdout validation.",
            "rad51c_latent_sae_application_panel.csv; brca2_prospective_followup_panel_summary.csv; third_gene_assay_statistical_plan.md",
        ),
        EvidenceRow(
            "Generalization",
            "At least one independent gene or data modality should replicate the mechanism/application pattern under the same locked analysis.",
            "BRCA1 is the causal anchor; RAD51C is the cleanest third-gene checkpoint plus latent-SAE screen; BAP1 is a strong SGE benchmark with finite-SNV boundary; BRCA2 is application/mechanism evidence but not strict native replication.",
            "in_progress",
            "Top-journal third-gene causal replication remains unproven until RAD51C native-SAE passes or another public SGE gene is run with the same chain.",
            "Use RAD51C native missense as primary third-gene gate; if it fails, choose a next MAVE/SGE gene with dense missense + ClinVar anchors.",
            "interpretability_submission_evidence_matrix.csv; third_gene_application_decision_gates.csv",
        ),
    ]


def build_actions(repo: Path) -> list[Action]:
    rad_native_status = "; ".join(
        [
            file_state(repo, "results/variant/rad51c_evo2.npz"),
            file_state(repo, "results/rad51c_gate_analysis/rad51c_gate_analysis.npz"),
            file_state(repo, "results/interpretability_applications/rad51c_native_finetuned_sae_intervention_summary.csv"),
            file_state(repo, "results/interpretability_applications/rad51c_native_finetuned_sae_intervention_missense_summary.csv"),
        ]
    )
    return [
        Action(
            1,
            "RAD51C_NATIVE_GATE",
            "Finish strict RAD51C native gate-head SAE replication, with missense-only z_prot as the primary interpretation.",
            "Native recon preserves predictor and top selected features show positive ablation delta beating matched random and label-permuted controls; rescue/addback positive if available.",
            "This is the difference between a checkpoint-level sparse screen and a top-journal third-gene causal interpretability claim.",
            "Jobs 9802961 -> 9802962 -> 9802963 and 9803849; rad51c_native_finetuned_sae_intervention*_summary.csv",
            rad_native_status,
        ),
        Action(
            2,
            "PUBLIC_BASELINES_RAD51C",
            "Add public missense predictors for RAD51C and report whether CrossBioSAE adds value beyond them.",
            "On RAD51C missense SGE labels, compare ESM+LLR/latent-SAE priority against AlphaMissense/EVE/CADD/REVEL-like scores where available.",
            "Top-journal variant papers rarely accept a new predictor/application without strong public baselines.",
            "New script should output rad51c_public_baseline_predictors_metric_summary.csv and binary-call comparison tables.",
            "not_started_in_current_artifacts",
        ),
        Action(
            3,
            "LOCKED_REVIEW_ASSAY_ENDPOINT",
            "Freeze RAD51C and BRCA2 review/assay panels with blinded IDs and prespecified success thresholds.",
            "At least one panel endpoint separates pathogenic-review arms from benign/control arms under a locked Fisher/AUROC/sensitivity-specificity threshold.",
            "This turns interpretability into an application claim rather than a descriptive ranking table.",
            "rad51c_latent_sae_application_panel.csv; brca2_prospective_followup_panel_summary.csv; third_gene_assay_statistical_plan.md",
            "designed_but_external_readout_pending",
        ),
        Action(
            4,
            "FEATURE_CARD_UPGRADE",
            "Convert passing native/latent features into gene-specific feature cards with domain, consequence, ClinVar, gnomAD, and assay enrichment.",
            "Feature cards show stable biological localization and survive matched annotation/covariate nulls.",
            "This is the mechanism-discovery layer expected from interpretability work, not only prediction.",
            "Extend brca1_global_feature_cards.csv style to RAD51C after native outputs appear.",
            "BRCA1_complete; RAD51C_waiting_native_outputs",
        ),
        Action(
            5,
            "NEXT_GENE_BACKUP",
            "Prepare a backup SGE/MAVE gene if RAD51C native-SAE does not pass.",
            "Candidate must have dense missense functional map, ClinVar/gnomAD anchors, public predictor coverage, and enough rows for grouped CV plus SAE controls.",
            "Prevents the paper from depending on one third-gene native result.",
            "Candidate short list: VHL, LDLR, MSH2, PTEN, TP53, SCN5A, DDX3X/KCNQ4 depending on data availability.",
            "planned_only",
        ),
    ]


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    rows = [
        "| " + " | ".join(show.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in show.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(out: Path, ex: pd.DataFrame, evidence: pd.DataFrame, actions: pd.DataFrame) -> None:
    lines = [
        "# Top-Journal Interpretability Evidence Plan",
        "",
        "## Bottom Line",
        "",
        "The current paper-grade center is BRCA1 causal native-SAE interpretability plus BRCA2/RAD51C mechanism-aware application design. RAD51C is now the cleanest path to an independent third-gene upgrade, but only the checkpoint-level latent-SAE screen has passed so far; the stricter native gate-head SAE chain is still pending.",
        "",
        "Top bioinformatics/genomics papers usually prove usefulness with a ladder: independent predictive benchmarks, biological or clinical anchors, intervention or mechanistic localization, strong baselines, and a locked downstream use case. The rows below translate that ladder into current CrossBioSAE artifacts.",
        "",
        "## Literature Proof Patterns",
        "",
        md_table(ex),
        "",
        "## Current Evidence Against That Standard",
        "",
        md_table(evidence),
        "",
        "## Priority Actions",
        "",
        md_table(actions),
        "",
        "## Claim Discipline",
        "",
        "- Safe now: BRCA1 fold-native sparse features are necessary for the learned functional predictor; RAD51C provides a strong independent checkpoint and positive checkpoint-level sparse-feature necessity screen; RAD51C/BRCA2 panels are application designs.",
        "- Not safe yet: third-gene native-SAE causal replication, clinical reclassification, or prospective utility without locked external readout.",
        "- Upgrade condition: RAD51C missense native-SAE passes matched-random, label-permutation, dose/localization, and preferably rescue controls, or an external locked review/assay endpoint validates the panel.",
        "",
    ]
    (out / "top_journal_interpretability_evidence_plan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    ex = pd.DataFrame([asdict(x) for x in exemplars()])
    evidence = pd.DataFrame([asdict(x) for x in build_evidence(repo)])
    actions = pd.DataFrame([asdict(x) for x in build_actions(repo)])

    ex.to_csv(out / "top_journal_interpretability_literature_patterns.csv", index=False)
    evidence.to_csv(out / "top_journal_interpretability_evidence_matrix.csv", index=False)
    actions.to_csv(out / "top_journal_interpretability_priority_actions.csv", index=False)
    write_report(out, ex, evidence, actions)
    print(f"wrote {out / 'top_journal_interpretability_evidence_plan.md'}")
    print(f"wrote {out / 'top_journal_interpretability_evidence_matrix.csv'}")
    print(f"wrote {out / 'top_journal_interpretability_priority_actions.csv'}")


if __name__ == "__main__":
    main()
