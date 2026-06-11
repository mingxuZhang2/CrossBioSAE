#!/usr/bin/env python
"""Build a third-gene interpretability application plan for BAP1/RAD51C."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def read_csv(repo: Path, name: str) -> pd.DataFrame:
    path = repo / OUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def row_by(df: pd.DataFrame, col: str, value: str) -> dict[str, Any]:
    if df.empty or col not in df.columns:
        return {}
    hit = df.loc[df[col].astype(str).eq(str(value))]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def row_where(df: pd.DataFrame, **criteria: str) -> dict[str, Any]:
    if df.empty:
        return {}
    mask = pd.Series(True, index=df.index)
    for col, value in criteria.items():
        if col not in df.columns:
            return {}
        mask &= df[col].astype(str).eq(str(value))
    hit = df.loc[mask]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def fmt(x: Any, digits: int = 3, sci: bool = False) -> str:
    if x is None or x == "" or pd.isna(x):
        return "NA"
    value = float(x)
    if sci:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def count_pair(row: dict[str, Any], prefix: str) -> str:
    if not row:
        return "NA"
    return f"{int(row[f'{prefix}_n_depleted'])}/{int(row[f'{prefix}_n_binary'])}"


def rate(row: dict[str, Any], prefix: str) -> str:
    if not row:
        return "NA"
    total = float(row[f"{prefix}_n_binary"])
    if total == 0:
        return "NA"
    return fmt(float(row[f"{prefix}_n_depleted"]) / total)


def file_state(repo: Path, rel: str) -> str:
    path = repo / rel
    if path.exists():
        return "present"
    return "missing"


def squeue_snapshot() -> str:
    try:
        out = subprocess.run(
            [
                "bash",
                "-lc",
                "squeue -u \"$USER\" -o '%i %j %T %M %R' | grep -E 'rad51c|bap1' || true",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=8,
        )
    except Exception as exc:  # pragma: no cover - report-only path
        return f"unavailable: {type(exc).__name__}: {exc}"
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        return "no tracked BAP1/RAD51C SLURM jobs in squeue"
    active = [line for line in lines if "DependencyNeverSatisfied" not in line]
    if not active:
        return "no active tracked BAP1/RAD51C SLURM jobs in squeue; historical fallback dependency is unsatisfied"
    return "; ".join(active)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = [header, sep]
    for _, row in show.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in show.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def build_decision_gates(repo: Path) -> pd.DataFrame:
    bap1_tests = read_csv(repo, "bap1_sge_external_anchors_tests.csv")
    rad_tests = read_csv(repo, "rad51c_sge_external_anchors_tests.csv")
    bap1_base = read_csv(repo, "bap1_sge_baseline_predictors_metric_summary.csv")
    bap1_metrics = read_csv(repo, "bap1_llr_esm_checkpoint_metric_summary.csv")
    bap1_delta = read_csv(repo, "bap1_llr_esm_checkpoint_bootstrap_delta.csv")
    bap1_disc = read_csv(repo, "bap1_llr_esm_discordance_category_summary.csv")
    rad_metrics = read_csv(repo, "rad51c_llr_esm_checkpoint_metric_summary.csv")
    rad_delta = read_csv(repo, "rad51c_llr_esm_checkpoint_bootstrap_delta.csv")
    rad_disc = read_csv(repo, "rad51c_llr_esm_discordance_category_summary.csv")
    rad_latent = read_csv(repo, "rad51c_latent_sae_intervention_summary.csv")

    bap1_clin = row_by(bap1_tests, "test", "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    bap1_plof = row_by(bap1_tests, "test", "plof_consequences_enriched_for_depleted_vs_synonymous_utr")
    rad_clin = row_by(rad_tests, "test", "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    rad_nonsense = row_by(rad_tests, "test", "nonsense_enriched_for_depleted_vs_synonymous")
    eve = row_by(bap1_base, "baseline", "EVE score")

    bap1_llr = repo / "results/variant/bap1_evo2_llr.npz"
    rad_llr = repo / "results/variant/rad51c_evo2_llr.npz"
    bap1_ckpt = repo / OUT_DIR / "bap1_llr_esm_checkpoint_metric_summary.csv"
    rad_ckpt = repo / OUT_DIR / "rad51c_llr_esm_checkpoint_metric_summary.csv"
    bap1_all_llr = row_where(bap1_metrics, scope="all", metric="Evo2 LLR zero-shot")
    bap1_all_fusion = row_where(bap1_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    bap1_missense_fusion = row_where(bap1_metrics, scope="missense", metric="ESM+Evo2 LLR CV logistic")
    bap1_all_delta = row_where(bap1_delta, comparison="ESM+LLR minus LLR-CV", scope="all")
    bap1_missense_delta = row_where(bap1_delta, comparison="ESM+LLR minus LLR-CV", scope="missense")
    bap1_both_high = row_by(bap1_disc, "category", "both_high")
    bap1_both_low = row_by(bap1_disc, "category", "both_low")
    bap1_checkpoint_positive = bool(bap1_all_llr and bap1_all_fusion and bap1_missense_delta and bap1_both_high)
    rad_all_fusion = row_where(rad_metrics, scope="all", metric="ESM+Evo2 LLR CV logistic")
    rad_missense_delta = row_where(rad_delta, comparison="ESM+LLR minus LLR-CV", scope="missense")
    rad_both_high = row_by(rad_disc, "category", "both_high")
    rad_both_low = row_by(rad_disc, "category", "both_low")
    rad_checkpoint_positive = bool(rad_all_fusion and rad_missense_delta and rad_both_high)
    rad_latent_row = rad_latent.iloc[0].to_dict() if not rad_latent.empty else {}
    rad_latent_positive = bool(
        rad_latent_row and float(rad_latent_row.get("delta_auroc_recon_minus_top_ablate", 0.0)) > 0
    )

    rows = [
        {
            "gate_id": "G12.1",
            "question": "Are the third-gene functional maps credible external benchmarks?",
            "current_evidence": (
                f"BAP1 ClinVar P/LP {count_pair(bap1_clin, 'first')} depleted vs "
                f"B/LB {count_pair(bap1_clin, 'second')}, AUROC {fmt(bap1_clin.get('external_lof_auroc'))}; "
                f"BAP1 pLOF {count_pair(bap1_plof, 'first')} vs syn/UTR {count_pair(bap1_plof, 'second')}. "
                f"RAD51C ClinVar P/LP {count_pair(rad_clin, 'first')} vs B/LB {count_pair(rad_clin, 'second')}, "
                f"AUROC {fmt(rad_clin.get('external_lof_auroc'))}; nonsense "
                f"{count_pair(rad_nonsense, 'first')} vs synonymous {count_pair(rad_nonsense, 'second')}."
            ),
            "status": "passed_anchor_ready",
            "required_artifacts": "bap1_sge_external_anchors_tests.csv; rad51c_sge_external_anchors_tests.csv",
            "action_if_pass": "Use BAP1/RAD51C as Supplementary Figure S2 benchmark anchors.",
            "action_if_fail": "Do not use as third-gene benchmarks; select another SGE/MAVE map.",
        },
        {
            "gate_id": "G12.2",
            "question": "Is there a public comparator bar for third-gene model claims?",
            "current_evidence": f"BAP1 EVE missense baseline AUROC {fmt(eve.get('auroc_for_sge_depleted'))}; SIFT/PolyPhen are also tabulated.",
            "status": "passed_for_bap1_partial_for_rad51c",
            "required_artifacts": "bap1_sge_baseline_predictors_metric_summary.csv",
            "action_if_pass": "Compare BAP1 checkpoint against EVE/SIFT/PolyPhen before any model-performance claim.",
            "action_if_fail": "Restrict to mechanism/anchor claims, not performance generalization.",
        },
        {
            "gate_id": "G12.3",
            "question": "Have Evo2/ESM checkpoint inputs finished?",
            "current_evidence": (
                f"BAP1 ESM {file_state(repo, 'results/variant/bap1_esm_delta.npz')}, "
                f"BAP1 Evo2 LLR {file_state(repo, 'results/variant/bap1_evo2_llr.npz')}; "
                f"RAD51C ESM {file_state(repo, 'results/variant/rad51c_esm_delta.npz')}, "
                f"RAD51C Evo2 LLR {file_state(repo, 'results/variant/rad51c_evo2_llr.npz')}. "
                f"Tracked squeue: {squeue_snapshot()}"
            ),
            "status": (
                "passed_inputs_ready"
                if bap1_llr.exists() and rad_llr.exists()
                else "passed_rad51c_input_bap1_pending"
                if rad_llr.exists()
                else "pending_gpu_outputs"
            ),
            "required_artifacts": "results/variant/bap1_evo2_llr.npz; results/variant/rad51c_evo2_llr.npz",
            "action_if_pass": "Run or inspect dependent checkpoint jobs immediately.",
            "action_if_fail": "Keep C12 as benchmark readiness only.",
        },
        {
            "gate_id": "G12.4",
            "question": "Do BAP1/RAD51C support model generalization under position-grouped CV?",
            "current_evidence": (
                f"BAP1 checkpoint {file_state(repo, 'results/interpretability_applications/bap1_llr_esm_checkpoint_metric_summary.csv')}; "
                f"RAD51C checkpoint {file_state(repo, 'results/interpretability_applications/rad51c_llr_esm_checkpoint_metric_summary.csv')}."
                + (
                    " BAP1 finite-SNV checkpoint: zero-shot LLR all AUROC "
                    + fmt(bap1_all_llr.get("auroc_for_sge_lof"))
                    + "; ESM+LLR all AUROC "
                    + fmt(bap1_all_fusion.get("auroc_for_sge_lof"))
                    + "; all delta vs LLR-CV "
                    + fmt(bap1_all_delta.get("median_delta_auroc"))
                    + "; missense fusion AUROC "
                    + fmt(bap1_missense_fusion.get("auroc_for_sge_lof"))
                    + "; missense delta "
                    + fmt(bap1_missense_delta.get("median_delta_auroc"))
                    + " CI ["
                    + fmt(bap1_missense_delta.get("ci_lo"))
                    + ", "
                    + fmt(bap1_missense_delta.get("ci_hi"))
                    + "]; both-high LOF rate "
                    + fmt(bap1_both_high.get("lof_rate"))
                    + " vs both-low "
                    + fmt(bap1_both_low.get("lof_rate"))
                    + "."
                    if bap1_checkpoint_positive and bap1_both_low
                    else ""
                )
                + (
                    " RAD51C ESM+LLR all AUROC "
                    + fmt(rad_all_fusion.get("auroc_for_sge_lof"))
                    + "; missense delta vs LLR "
                    + fmt(rad_missense_delta.get("median_delta_auroc"))
                    + " CI ["
                    + fmt(rad_missense_delta.get("ci_lo"))
                    + ", "
                    + fmt(rad_missense_delta.get("ci_hi"))
                    + "]; both-high missense LOF rate "
                    + fmt(rad_both_high.get("lof_rate"))
                    + " vs both-low "
                    + fmt(rad_both_low.get("lof_rate"))
                    + "."
                    if rad_checkpoint_positive and rad_both_low
                    else ""
                )
            ),
            "status": (
                "passed_rad51c_and_bap1_checkpoint_native_sae_pending"
                if rad_checkpoint_positive and bap1_checkpoint_positive
                else "passed_rad51c_checkpoint_bap1_pending"
                if rad_checkpoint_positive and not bap1_ckpt.exists()
                else "inspect_metrics"
                if (bap1_ckpt.exists() or rad_ckpt.exists())
                else "pending_checkpoint_metrics"
            ),
            "required_artifacts": "bap1_llr_esm_checkpoint_metric_summary.csv or rad51c_llr_esm_checkpoint_metric_summary.csv",
            "action_if_pass": "Promote a third-gene validation figure only if metrics beat relevant baselines or reveal useful mechanism strata.",
            "action_if_fail": "Report as boundary result and keep BRCA1/BRCA2 as the main story.",
        },
        {
            "gate_id": "G12.5",
            "question": "Is there enough for a top-journal mechanism claim?",
            "current_evidence": (
                "BRCA1 has causal SAE intervention; RAD51C has a clean positive checkpoint/mechanism-stratification result plus a positive checkpoint-level latent-SAE intervention screen; BAP1 has a finite-SNV positive/boundary checkpoint; strict third-gene native-SAE intervention is still pending."
                if rad_checkpoint_positive and bap1_checkpoint_positive and rad_latent_positive
                else "BRCA1 has causal SAE intervention; RAD51C has a clean positive checkpoint/mechanism-stratification result; BAP1 has a finite-SNV positive/boundary checkpoint; neither third gene has native-SAE intervention or rescue."
                if rad_checkpoint_positive and bap1_checkpoint_positive
                else "BRCA1 has causal SAE intervention; RAD51C now has a positive checkpoint/mechanism-stratification result, but BAP1/RAD51C still have no native-SAE intervention or rescue."
                if rad_checkpoint_positive
                else "BRCA1 has causal SAE intervention; BAP1/RAD51C currently have external anchors but no native-SAE intervention or rescue."
            ),
            "status": "passed_latent_sae_screen_native_pending" if rad_latent_positive else "not_started_for_third_gene_sae",
            "required_artifacts": "rad51c_latent_sae_intervention_summary.csv; third-gene native-SAE ablation, random-feature null, label permutation, addback rescue",
            "action_if_pass": "Use third-gene native-SAE as a main-text replication mechanism figure.",
            "action_if_fail": "Keep third-gene evidence as benchmark/application readiness, not causal interpretability replication; current RAD51C latent/full-native SAE SLURM chain should be inspected when jobs finish.",
        },
    ]
    return pd.DataFrame(rows)


def build_application_blueprint(repo: Path) -> pd.DataFrame:
    bap1_candidates = read_csv(repo, "bap1_sge_external_anchors_vus_candidates.csv")
    rad_candidates = read_csv(repo, "rad51c_sge_external_anchors_candidate_review.csv")
    bap1_tests = read_csv(repo, "bap1_sge_external_anchors_tests.csv")
    rad_tests = read_csv(repo, "rad51c_sge_external_anchors_tests.csv")
    rad_disc = read_csv(repo, "rad51c_llr_esm_discordance_category_summary.csv")
    bap1_disc = read_csv(repo, "bap1_llr_esm_discordance_category_summary.csv")
    rad_latent = read_csv(repo, "rad51c_latent_sae_intervention_summary.csv")
    rad_latent_strata = read_csv(repo, "rad51c_latent_sae_intervention_stratum_summary.csv")
    rad_latent_app = read_csv(repo, "rad51c_latent_sae_application_arm_summary.csv")
    bap1_clin = row_by(bap1_tests, "test", "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    rad_clin = row_by(rad_tests, "test", "clinvar_pathogenic_enriched_for_depleted_vs_benign")
    rad_nonsense = row_by(rad_tests, "test", "nonsense_enriched_for_depleted_vs_synonymous")
    rad_both_high = row_by(rad_disc, "category", "both_high")
    rad_both_low = row_by(rad_disc, "category", "both_low")
    bap1_both_high = row_by(bap1_disc, "category", "both_high")
    bap1_both_low = row_by(bap1_disc, "category", "both_low")
    rad_latent_row = rad_latent.iloc[0].to_dict() if not rad_latent.empty else {}
    rad_latent_missense = row_by(rad_latent_strata, "stratum", "missense")
    rad_latent_both_high_arm = row_by(rad_latent_app, "panel_arm", "both_high_sparse_lof_review")
    rad_latent_protein_arm = row_by(rad_latent_app, "panel_arm", "protein_high_dna_low_sparse_lof")
    rad_latent_positive = bool(
        rad_latent_row and float(rad_latent_row.get("delta_auroc_recon_minus_top_ablate", 0.0)) > 0
    )

    rows = [
        {
            "application_arm": "BAP1_depleted_VUS_review",
            "gene": "BAP1",
            "current_n": int(len(bap1_candidates)),
            "selection_rule": "Top SGE-depleted ClinVar VUS or unresolved BAP1 rows, prioritizing not-in-gnomAD and domain-covered missense rows.",
            "endpoint": "Manual expert review or independent functional assay should enrich for likely pathogenic evidence versus benign controls.",
            "claim_if_positive": "CrossBioSAE/SGE-derived interpretation prioritizes reviewable BAP1 VUS.",
            "current_boundary": (
                "BAP1 checkpoint is positive within the finite-SNV/imputed-LLR design; keep review/assay language until locked outcomes exist."
                if bap1_both_high
                else "SGE anchor only until Evo2/ESM checkpoint and locked review/assay readout exist."
            ),
        },
        {
            "application_arm": "BAP1_benign_population_controls",
            "gene": "BAP1",
            "current_n": int(bap1_clin.get("second_n_binary", 0)),
            "selection_rule": "Exact ClinVar B/LB rows plus gnomAD-observed rows with unchanged SGE function.",
            "endpoint": "Negative-control depletion rate should stay low; current B/LB depleted rate is " + rate(bap1_clin, "second") + ".",
            "claim_if_positive": "Panel is calibrated against benign/population controls.",
            "current_boundary": "Controls support benchmark credibility, not VUS reclassification.",
        },
        {
            "application_arm": "RAD51C_conflicting_or_VUS_review",
            "gene": "RAD51C",
            "current_n": int(len(rad_candidates)),
            "selection_rule": "Top depleted current ClinVar conflicting/VUS/unobserved direct SNVs from the RAD51C GRCh38-mapped SGE table.",
            "endpoint": "Review or assay should enrich for pathogenic/LOF evidence relative to B/LB and synonymous controls.",
            "claim_if_positive": "RAD51C provides an HR-pathway application beyond BRCA1/BRCA2.",
            "current_boundary": (
                "RAD51C checkpoint is now positive; use review/assay language until locked outcomes exist."
                if rad_both_high
                else "ClinVar/consequence anchors are strong, but model generalization is still pending."
            ),
        },
        {
            "application_arm": "RAD51C_benign_and_synonymous_controls",
            "gene": "RAD51C",
            "current_n": int(rad_clin.get("second_n_binary", 0)) + int(rad_nonsense.get("second_n_binary", 0)),
            "selection_rule": "ClinVar B/LB plus synonymous controls from the same mapped direct-SNV space.",
            "endpoint": "Negative controls should remain mostly unchanged; current B/LB depleted rate is " + rate(rad_clin, "second") + ".",
            "claim_if_positive": "RAD51C follow-up panel has a clean negative-control arm.",
            "current_boundary": "Useful as assay design control, not a standalone model result.",
        },
        {
            "application_arm": "Third_gene_DNA_protein_mechanism_strata",
            "gene": "RAD51C/BAP1",
            "current_n": int(rad_both_high.get("n", 0)) + int(bap1_both_high.get("n", 0)),
            "selection_rule": "Split variants into DNA-high/protein-high, DNA-only, protein-only, and both-low strata by pre-specified top/bottom 20% quantiles.",
            "endpoint": (
                "RAD51C both-high missense LOF rate "
                + fmt(rad_both_high.get("lof_rate"))
                + " vs both-low "
                + fmt(rad_both_low.get("lof_rate"))
                + "; BAP1 both-high LOF rate "
                + fmt(bap1_both_high.get("lof_rate"))
                + " vs both-low "
                + fmt(bap1_both_low.get("lof_rate"))
                + "."
                if rad_both_high and rad_both_low and bap1_both_high and bap1_both_low
                else "Both-high or discordant strata should enrich for depleted labels and align with domains/consequences after matched controls."
            ),
            "claim_if_positive": "Interpretability identifies third-gene mechanism classes rather than only scalar pathogenicity.",
            "current_boundary": "RAD51C is clean positive; BAP1 is finite-SNV positive with an imputed-LLR/indel caveat; neither is native-SAE causal replication.",
        },
        {
            "application_arm": "RAD51C_latent_SAE_sparse_mechanism_panel",
            "gene": "RAD51C",
            "current_n": int(rad_latent_app.get("selected_n", pd.Series(dtype=float)).sum()) if not rad_latent_app.empty else 0,
            "selection_rule": "Rank RAD51C missense variants by sparse-feature necessity drop, selected-feature-only lift, SAE-reconstructed LOF probability, and SGE LOF score; include both-high, protein-high/DNA-low, model-conflict, both-low, and non-missense control arms.",
            "endpoint": (
                "Current latent-SAE screen: recon-vs-top-feature-ablation delta AUROC "
                + fmt(rad_latent_row.get("delta_auroc_recon_minus_top_ablate"))
                + ", random p="
                + fmt(rad_latent_row.get("empirical_p_random_delta_ge_top_mean"))
                + ", label-permutation p="
                + fmt(rad_latent_row.get("empirical_p_label_permuted_delta_ge_top_mean"))
                + "; missense delta "
                + fmt(rad_latent_missense.get("delta_auroc_recon_minus_top_ablate"))
                + "."
                if rad_latent_positive
                else "Await latent-SAE intervention output."
            ),
            "claim_if_positive": "Sparse protein-side features prioritize RAD51C review/assay candidates beyond scalar DNA/protein checkpoint scores.",
            "current_boundary": "This is a checkpoint-level ESM latent-SAE application panel, not the stricter native gate-SAE causal replication.",
        },
        {
            "application_arm": "Third_gene_native_SAE_intervention",
            "gene": "BAP1/RAD51C",
            "current_n": 0,
            "selection_rule": "Train fold-native SAE on checkpoint-positive representation and select features using training labels only.",
            "endpoint": "Top-feature ablation reduces held-out AUROC more than matched random features; addback rescue restores performance; label permutation fails.",
            "claim_if_positive": "Third-gene causal interpretability replication.",
            "current_boundary": "This is the top-journal upgrade, but it starts only after checkpoint metrics identify a meaningful target.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(repo: Path, gates: pd.DataFrame, blueprint: pd.DataFrame) -> Path:
    out = repo / OUT_DIR / "third_gene_interpretability_application_plan.md"
    lines = [
        "# Third-Gene Interpretability Application Plan",
        "",
        "## Scope",
        "",
        "This report turns the current BAP1/RAD51C work into an explicit application and review plan. It separates three levels of evidence: benchmark-anchor readiness, checkpoint model generalization, and causal native-SAE mechanism replication.",
        "",
        "## Top-Journal Proof Pattern",
        "",
        "Bioinformatics and functional-genomics papers usually make interpretability convincing by combining functional maps such as SGE/MAVE/DMS, clinical/population anchors such as ClinVar and gnomAD, held-out or grouped evaluation, confound controls, causal feature intervention, and a concrete downstream use case. For this repo, the downstream use case should be variant review and assay prioritization, not clinical reclassification.",
        "",
        "## Decision Gates",
        "",
        md_table(gates),
        "",
        "## Application Blueprint",
        "",
        md_table(blueprint),
        "",
        "## How To Use The Result",
        "",
        "- If BAP1 or RAD51C checkpoint metrics are strong: promote the gene to a third-gene validation/checkpoint figure, then run native-SAE intervention only on the positive representation or stratum.",
        "- Current update: RAD51C checkpoint metrics are strong enough to promote as a third-gene checkpoint/mechanism-stratification result; BAP1 checkpoint is also positive within the finite-SNV/imputed-LLR design and should be framed as a boundary-confirming secondary checkpoint.",
        "- New RAD51C update: the checkpoint-level ESM latent-SAE screen is positive and has a 64-row sparse-mechanism follow-up panel; use it as application evidence while keeping native gate-SAE replication pending.",
        "- If checkpoint metrics are weak but anchors are strong: report BAP1/RAD51C as benchmark-ready boundary evidence and keep BRCA1 causal intervention plus BRCA2 mechanism/VUS application as the main paper.",
        "- If a mechanism stratum is strong: build a blinded 48-64 row review/assay panel with pathogenic-review, benign-control, discordant-mechanism, and model-conflict arms.",
        "- A pre-checkpoint 64-row BAP1/RAD51C review/assay panel can be regenerated with `python scripts/design_third_gene_review_assay_panel.py --repo-root .`; it provides immediate review/assay utility while remaining separate from CrossBioSAE validation.",
        "- The panel's binary endpoint thresholds can be regenerated with `python scripts/design_third_gene_assay_statistical_plan.py --repo-root .`; this fixes the success rule before any external readout is available.",
        "- If a native-SAE intervention passes: the paper can claim third-gene causal interpretability replication; otherwise do not upgrade C12 beyond benchmark/application readiness.",
        "- Current RAD51C SAE route: `slurm_rad51c_latent_sae.sh` screens ESM-pdelta sparse-feature necessity; `slurm_rad51c_evo2_full.sh -> slurm_rad51c_gate_analysis.sh -> slurm_rad51c_native_sae.sh` is the all-variant BRCA1-style native-SAE chain, and `slurm_rad51c_native_sae_missense.sh` is the primary missense-only z_prot replication chain if all-variant z_prot is diluted by non-missense masking.",
        "",
        "## Immediate Commands",
        "",
        "```bash",
        "squeue -j 9802557,9802559,9802605,9802785,9802786,9802811",
        "python scripts/summarize_third_gene_application_plan.py --repo-root .",
        "python scripts/design_third_gene_review_assay_panel.py --repo-root .",
        "python scripts/design_third_gene_assay_statistical_plan.py --repo-root .",
        "python scripts/summarize_rad51c_latent_sae_application.py --repo-root .",
        "sbatch scripts/slurm_rad51c_latent_sae.sh",
        "sbatch scripts/slurm_rad51c_evo2_full.sh",
        "sbatch --dependency=afterok:<rad51c_full_evo2_jobid> scripts/slurm_rad51c_gate_analysis.sh",
        "sbatch --dependency=afterok:<rad51c_gate_jobid> scripts/slurm_rad51c_native_sae.sh",
        "sbatch --dependency=afterok:<rad51c_gate_jobid> scripts/slurm_rad51c_native_sae_missense.sh",
        "python scripts/summarize_interpretability_submission_evidence_matrix.py --repo-root .",
        "python scripts/summarize_interpretability_manuscript_package.py --repo-root .",
        "python scripts/summarize_interpretability_reproducibility_package.py --repo-root .",
        "```",
        "",
        "## Claim Boundary",
        "",
        "The current supported C12 claim is RAD51C third-gene checkpoint/mechanism stratification plus checkpoint-level latent-SAE sparse-feature necessity, with BAP1 finite-SNV checkpoint/boundary support and BAP1/RAD51C benchmark readiness. Third-gene native-SAE causal replication, VUS reclassification, and locked clinical utility remain unsupported until the corresponding intervention and external endpoint artifacts exist.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = repo / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = build_decision_gates(repo)
    blueprint = build_application_blueprint(repo)
    gates.to_csv(out_dir / "third_gene_application_decision_gates.csv", index=False)
    blueprint.to_csv(out_dir / "third_gene_application_blueprint.csv", index=False)
    report = write_report(repo, gates, blueprint)
    print(f"wrote {out_dir / 'third_gene_application_decision_gates.csv'}")
    print(f"wrote {out_dir / 'third_gene_application_blueprint.csv'}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
