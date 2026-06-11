#!/usr/bin/env python
"""Generate a manuscript-style draft from the interpretability evidence matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


@dataclass
class DraftSection:
    section_id: str
    title: str
    primary_claims: str
    status: str
    key_message: str
    evidence: str
    limitation: str


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_matrix(repo: Path) -> pd.DataFrame:
    path = repo / OUT_DIR / "interpretability_submission_evidence_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run summarize_interpretability_submission_evidence_matrix.py first."
        )
    return pd.read_csv(path).fillna("")


def claim_map(claims: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        str(row["claim_id"]): {k: str(v) for k, v in row.items()}
        for _, row in claims.iterrows()
    }


def claim(c: dict[str, dict[str, str]], claim_id: str, field: str) -> str:
    return c.get(claim_id, {}).get(field, "")


def sentence(text: str) -> str:
    return str(text).strip().rstrip(".")


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, item in df.iterrows():
        vals = [str(item.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item]


def build_sections(repo: Path, claims: pd.DataFrame) -> list[DraftSection]:
    c = claim_map(claims)
    brca2_full = repo / "results/variant/brca2_evo2.npz"
    brca2_native = repo / OUT_DIR / "brca2_native_finetuned_sae_intervention_summary.csv"
    c9_readiness = claim(c, "C9", "readiness")
    c9_mixed = "mixed" in c9_readiness or "negative" in c9_readiness
    c12_ready = "complete_checkpoint" in claim(c, "C12", "readiness")
    return [
        DraftSection(
            "R1",
            "A top-journal interpretability claim needs functional labels, intervention, controls, replication, and downstream use.",
            "C1-C9",
            "ready",
            "The manuscript should define interpretability as a testable mechanism layer, not as feature visualization.",
            "The evidence matrix separates causal BRCA1 intervention, BRCA2 checkpoint replication, VUS triage, and pending gates.",
            "This section is framing; it does not itself prove biological utility.",
        ),
        DraftSection(
            "R2",
            "Fold-native SAE features are causally necessary for BRCA1 functional prediction.",
            "C1",
            claim(c, "C1", "readiness"),
            claim(c, "C1", "manuscript_claim"),
            claim(c, "C1", "key_result") + " Controls: " + claim(c, "C1", "controls_or_comparators"),
            claim(c, "C1", "missing_for_top_journal"),
        ),
        DraftSection(
            "R3",
            "BRCA1 native-SAE effects localize to biological variant classes.",
            "C2",
            claim(c, "C2", "readiness"),
            claim(c, "C2", "manuscript_claim"),
            claim(c, "C2", "key_result") + " Controls: " + claim(c, "C2", "controls_or_comparators"),
            "Global-SAE evidence should be described as a bridge to fold-local effects, not as causal sufficiency.",
        ),
        DraftSection(
            "R4",
            "BRCA2 is an independent SGE checkpoint, but not a generic fusion-gain success.",
            "C3",
            claim(c, "C3", "readiness"),
            claim(c, "C3", "manuscript_claim"),
            claim(c, "C3", "key_result") + " Negative control: " + claim(c, "C3", "controls_or_comparators"),
            claim(c, "C3", "missing_for_top_journal"),
        ),
        DraftSection(
            "R5",
            "BRCA2 DNA/protein concordance identifies high-risk missense mechanism strata.",
            "C4",
            claim(c, "C4", "readiness"),
            claim(c, "C4", "manuscript_claim"),
            claim(c, "C4", "key_result"),
            claim(c, "C4", "missing_for_top_journal"),
        ),
        DraftSection(
            "R6",
            "BRCA2 interpretation yields an executable VUS review and assay-triage application.",
            "C5,C6",
            "ready_as_triage_not_reclassification",
            claim(c, "C5", "manuscript_claim") + " " + claim(c, "C6", "manuscript_claim"),
            claim(c, "C5", "key_result") + " " + claim(c, "C6", "key_result"),
            "No external blinded review or wet-lab readout is available yet, so the claim is application triage only.",
        ),
        DraftSection(
            "R7",
            "Collagen-glycine features are a supplementary structural VUS case study.",
            "C7",
            claim(c, "C7", "readiness"),
            claim(c, "C7", "manuscript_claim"),
            claim(c, "C7", "key_result"),
            claim(c, "C7", "missing_for_top_journal"),
        ),
        DraftSection(
            "R8",
            (
                "BRCA2 native-SAE is currently a mixed replication and error-analysis result."
                if c9_mixed
                else "BRCA2 native-SAE replication is the decisive remaining mechanistic gate."
            ),
            "C9",
            c9_readiness or ("pending" if not brca2_native.exists() else "ready"),
            claim(c, "C9", "manuscript_claim"),
            (
                f"Full BRCA2 Evo2 embeddings exist: {brca2_full.exists()}; "
                f"BRCA2 native-SAE summary exists: {brca2_native.exists()}."
            ),
            claim(c, "C9", "missing_for_top_journal"),
        ),
        DraftSection(
            "R9",
            (
                "RAD51C and finite-SNV BAP1 define positive third-gene checkpoints and the review/assay route."
                if c12_ready
                else "BAP1/RAD51C define a third-gene benchmark and review/assay application route."
            ),
            "C12",
            claim(c, "C12", "readiness"),
            claim(c, "C12", "manuscript_claim"),
            claim(c, "C12", "key_result"),
            claim(c, "C12", "missing_for_top_journal"),
        ),
    ]


def build_abstract(c: dict[str, dict[str, str]]) -> list[str]:
    c9_ready = claim(c, "C9", "readiness").startswith("complete")
    c9_mixed = "mixed" in claim(c, "C9", "readiness") or "negative" in claim(c, "C9", "readiness")
    c12_ready = "complete_checkpoint" in claim(c, "C12", "readiness")
    conclusion = (
        "The current supported claim is that sparse, fold-native features provide causal interpretability for BRCA1 and independent BRCA2 mechanistic replication, while also driving BRCA2 mechanism stratification and review/assay triage. External review or assay outcomes remain required before claiming clinical reclassification."
        if c9_ready
        else "The current supported claim is that sparse, fold-native features provide strong causal interpretability for BRCA1, while BRCA2 supports mechanism stratification and review/assay triage. BRCA2 native-SAE currently shows a positive ablation effect but does not pass the strict label-permutation replication gate."
        if c9_mixed
        else "The current supported claim is that sparse, fold-native features can provide causal interpretability for BRCA1 and can drive BRCA2 mechanism stratification and review/assay triage. BRCA2 native-SAE causal replication and external review or assay outcomes remain required before claiming full independent mechanistic replication or clinical reclassification."
    )
    result_text = " ".join(
        [
            sentence(claim(c, "C1", "key_result")) + ".",
            sentence(claim(c, "C2", "key_result")) + ".",
            "In BRCA2, " + sentence(claim(c, "C3", "key_result")) + ".",
            sentence(claim(c, "C4", "key_result")) + ".",
            "The application layer produced BRCA2 VUS tiers and an executable review/assay panel: "
            + sentence(claim(c, "C5", "key_result"))
            + ". "
            + sentence(claim(c, "C6", "key_result"))
            + ". "
            + sentence(claim(c, "C9", "key_result"))
            + ".",
            (
                "RAD51C adds a clean positive third-gene checkpoint, while BAP1 adds finite-SNV positive/boundary support: ESM+LLR improves key functional-prediction settings, and both-high missense variants show strong depleted-label enrichment in both genes."
                if c12_ready
                else ""
            ),
        ]
    )
    return [
        "## Structured Abstract",
        "",
        "**Background.** Biological foundation models increasingly support variant-effect prediction, but high-performing scores are difficult to use as mechanistic evidence unless explanations are tied to functional labels and perturbation tests.",
        "",
        "**Methods.** We evaluated CrossBioSAE as a mechanism-aware layer over DNA and protein variant representations. The evidence ladder combines saturation genome editing labels, sparse-feature reconstruction and ablation, matched random-feature and raw-dimension controls, BRCA2 independent checkpointing, ClinVar unresolved-variant triage, and a blinded review/assay handoff design.",
        "",
        "**Results.** " + result_text,
        "",
        "**Conclusions.** " + conclusion,
        "",
    ]


def build_figure_legends(figures: pd.DataFrame) -> list[str]:
    if figures.empty:
        return [
            "## Figure Legend Drafts",
            "",
            "- Figure legends are unavailable because interpretability_manuscript_figure_plan.csv was not found.",
            "",
        ]
    lines = ["## Figure Legend Drafts", ""]
    for _, row in figures.iterrows():
        figure = str(row.get("figure", ""))
        panel = str(row.get("panel", ""))
        title = str(row.get("title", ""))
        message = str(row.get("panel_message", ""))
        artifact = str(row.get("data_or_artifact", ""))
        status = str(row.get("status", ""))
        label = figure if not panel else f"{figure}, panel {panel}"
        lines.append(f"**{label}. {title}.** {message} Data: {artifact}. Status: {status}.")
        lines.append("")
    return lines


def write_report(repo: Path, claims: pd.DataFrame, sections: list[DraftSection]) -> Path:
    out = repo / OUT_DIR / "interpretability_manuscript_draft.md"
    c = claim_map(claims)
    figures = read_csv(repo / OUT_DIR / "interpretability_manuscript_figure_plan.csv").fillna("")
    risks = read_csv(repo / OUT_DIR / "interpretability_reviewer_risk_register.csv").fillna("")
    c9_ready = claim(c, "C9", "readiness").startswith("complete")
    c9_mixed = "mixed" in claim(c, "C9", "readiness") or "negative" in claim(c, "C9", "readiness")
    c12_ready = "complete_checkpoint" in claim(c, "C12", "readiness")
    brca2_chain = (
        "BRCA2 downstream and native-SAE intervention chain"
        if (repo / "results/variant/brca2_evo2.npz").exists()
        else "BRCA2 full Evo2, downstream, and native-SAE intervention chain"
    )
    discussion_limitation = (
        "The main mechanistic limitation has shifted from replication to clinical scope: unresolved BRCA2 and collagen candidates are review or assay priorities, not clinical reclassifications, until an external blinded review or assay readout is available."
        if c9_ready
        else "The main limitation is that only BRCA1 currently has completed native-SAE causal intervention evidence that passes the full control ladder. BRCA2 already supports independent checkpointing and VUS triage, and its native-SAE run shows a positive broad ablation effect, but it fails the strict label-permutation replication gate; the focused-strata screen does not provide a narrow biological replacement claim. A second limitation is clinical scope: unresolved BRCA2 and collagen candidates are review or assay priorities, not clinical reclassifications."
        if c9_mixed
        else "The main limitation is that only BRCA1 currently has completed native-SAE causal intervention evidence. BRCA2 already supports independent checkpointing and VUS triage, but full mechanistic replication requires the pending native-SAE intervention. A second limitation is clinical scope: unresolved BRCA2 and collagen candidates are review or assay priorities, not clinical reclassifications."
    )
    c9_gate_sentence = (
        "The BRCA2 native-SAE section should be promoted to a decisive replication result and paired with its controls, strata, and feature cards."
        if c9_ready
        else "The BRCA2 native-SAE result should be written as mixed evidence: targeted ablation has a positive effect and beats random-feature controls, but it does not survive label-permutation significance. It belongs in an error-analysis or focused-stratum section unless further validation improves it."
        if c9_mixed
        else f"The final paper should be upgraded only after the {brca2_chain} completes. The BRCA2 native-SAE section should mirror the BRCA1 intervention layout: reconstruction preservation, top-feature ablation, matched random features, label permutation, dose response, strata, and preferably a raw-dimension control."
    )

    section_df = pd.DataFrame([asdict(s) for s in sections])
    incomplete_mask = claims["readiness"].str.contains("pending|partial|mixed|negative", case=False, na=False)
    complete = claims.loc[~incomplete_mask]
    incomplete = claims.loc[incomplete_mask]

    lines: list[str] = [
        "# CrossBioSAE Interpretability Manuscript Draft",
        "",
        "## Working Title",
        "",
        "Causal sparse-feature interpretation of DNA and protein foundation-model variant mechanisms enables BRCA1 validation and BRCA2 VUS assay triage",
        "",
        "## Manuscript Thesis",
        "",
        "CrossBioSAE should be submitted as a mechanistic interpretability and variant-triage study. The paper should not claim generic multimodal prediction superiority. Its strongest contribution is a testable evidence ladder: sparse features preserve predictions when reconstructed, degrade them when targeted features are ablated, localize to biological variant classes, transfer into an independent BRCA2 SGE checkpoint, and produce a blinded review/assay panel for unresolved variants.",
        "",
    ]
    lines.extend(build_abstract(c))
    lines.extend(
        [
            "## Results Roadmap",
            "",
            md_table(
                section_df,
                ["section_id", "title", "primary_claims", "status", "key_message", "limitation"],
            ),
            "",
            "## Introduction Draft",
            "",
            "Foundation models trained on DNA or protein sequences can score variants at genome scale, but a score alone rarely explains why a variant is damaging or how the result should be used downstream. Top-tier bioinformatics interpretation papers usually connect model behavior to external functional labels, perturbation evidence, biological localization, negative controls, and an actionable use case. We therefore treat interpretability as an experimentally auditable object: an explanation is useful only if it identifies a model-internal mechanism, changes predictions under targeted intervention, recovers known biological structure, and produces a reviewable hypothesis for new variants.",
            "",
            "CrossBioSAE follows this standard by decomposing variant representations into sparse features and testing whether those features support functional variant prediction. The central application is not automatic clinical classification. Instead, the goal is to prioritize unresolved variants for expert review or assay selection, while preserving explicit controls that identify where the model is wrong or incomplete.",
            "",
            "## Results Draft",
            "",
            "### Evidence Standard And Claim Boundaries",
            "",
            "We first organized the study around a claim-to-evidence matrix. Each claim is tied to a dataset, a statistical or intervention result, controls, downstream use, and a remaining top-journal gap. This prevents the manuscript from overstating descriptive feature labels as mechanistic proof.",
            "",
            "Supported current claims:",
            "",
        ]
    )
    lines.extend(
        bullets(
            [
                f"{row['claim_id']}: {row['manuscript_claim']} Evidence: {row['key_result']}"
                for _, row in complete.iterrows()
            ]
        )
    )
    lines.extend(["", "Incomplete or context-only claims:", ""])
    lines.extend(
        bullets(
            [
                f"{row['claim_id']}: {row['manuscript_claim']} Gap: {row['missing_for_top_journal']}"
                for _, row in incomplete.iterrows()
            ]
        )
    )
    lines.extend(
        [
            "",
            "### BRCA1 Native-SAE Intervention Establishes Causal Interpretability",
            "",
            claim(c, "C1", "key_result"),
            "",
            "This is the anchor result because it uses the strongest validation standard available in the current project: reconstruction preserves the predictive signal, targeted sparse-feature ablation reduces it, random feature sets and label permutations do not reproduce the same effect, and a raw dense-dimension ablation is much weaker. The result supports feature necessity for BRCA1 functional prediction.",
            "",
            "### BRCA1 Sparse Effects Localize To Biological Variant Classes",
            "",
            claim(c, "C2", "key_result"),
            "",
            "High native-effect variants concentrate in LOF-like categories and disease-relevant BRCA1 regions, while global features recover part of the native-effect signal above random and permutation controls. This section should be written carefully: the global feature bridge helps name and inspect features, but the causal proof remains the native fold-local intervention.",
            "",
            "### BRCA2 Provides Independent Mechanism Stratification, Not Fusion SOTA",
            "",
            claim(c, "C3", "key_result"),
            "",
            claim(c, "C3", "controls_or_comparators"),
            "",
            "This negative control is useful. It rules out a shallow story in which the paper claims that concatenating DNA and protein scores is enough. The better story is that Evo2 and ESM capture partly different mechanisms, and their agreement or disagreement defines biologically useful strata.",
            "",
            "### BRCA2 DNA/Protein Concordance Identifies High-Risk Missense Strata",
            "",
            claim(c, "C4", "key_result"),
            "",
            "Both-high BRCA2 missense variants are enriched for SGE LOF and recurring residue hotspots across CTDB domains. Both-low variants are depleted for LOF, while discordant quadrants become stress tests for the model. These strata define positive controls, negative controls, split-mechanism cases, and model-conflict cases for downstream follow-up.",
            "",
            "### BRCA2 VUS Triage Converts Interpretation Into A Downstream Application",
            "",
            claim(c, "C5", "key_result"),
            "",
            claim(c, "C6", "key_result"),
            "",
            "The downstream output is a review and assay triage package rather than a clinical classifier. The blinded review sheet hides model arm, SGE label, DNA/protein percentiles, and expected class, enabling an external reviewer or assay collaborator to produce locked judgments before unblinding.",
            "",
            "### Collagen-Glycine Features Are A Supplementary Structural Case Study",
            "",
            claim(c, "C7", "key_result"),
            "",
            "This case study shows biological specificity on known pathogenic collagen-glycine controls, but current VUS set-level enrichment is not strong enough for a main clinical-utility claim. It should remain supplementary unless temporal or external disease-database validation improves.",
            "",
            (
                "### BRCA2 Native-SAE Is A Mixed Replication/Error-Analysis Result"
                if c9_mixed
                else "### BRCA2 Native-SAE Replication Remains The Decisive Pending Gate"
            ),
            "",
            claim(c, "C9", "key_result"),
            "",
            c9_gate_sentence,
            "",
            (
                "### RAD51C And BAP1 Extend The Checkpoint Mechanism Route While Native-SAE Replication Remains Pending"
                if c12_ready
                else "### BAP1/RAD51C Extend The Application Route But Remain Pending For CrossBioSAE Generalization"
            ),
            "",
            claim(c, "C12", "key_result"),
            "",
            (
                "RAD51C can now be written as a clean third-gene checkpoint and DNA/protein mechanism-stratification result. BAP1 can be written as a finite-SNV checkpoint/boundary result because the checkpoint is positive but many BAP1 indel or multibase rows lack zero-shot LLR. Neither result should be written as third-gene causal CrossBioSAE replication because native-SAE ablation/rescue evidence is not available."
                if c12_ready
                else "These third-gene datasets are now credible external benchmark anchors and can support a pre-specified review/assay panel. They should not yet be written as CrossBioSAE validation because Evo2 LLR, checkpoint metrics, and native-SAE intervention evidence are still pending."
            ),
            "",
            "## Discussion Draft",
            "",
            "The current evidence supports a pragmatic role for interpretability in variant biology: it identifies model-internal features that are necessary for prediction, links those effects to functional assay labels, and converts score disagreement into reviewable hypotheses. This is useful even when fusion does not improve scalar prediction, because the interpretation layer organizes variants into mechanistic and application-facing groups.",
            "",
            discussion_limitation,
            "",
            "## Claim Boundaries",
            "",
            "- Safe now: BRCA1 native-SAE causal necessity for functional prediction.",
            "- Safe now: BRCA1 biological localization and feature-card support, with global features framed as bridge evidence.",
            "- Safe now: BRCA2 independent SGE checkpoint and DNA/protein mechanism stratification.",
            "- Safe now: BRCA2 VUS review and assay-triage package.",
            (
                "- Safe now if C9 remains complete after audit: BRCA2 native-SAE causal replication."
                if c9_ready
                else "- Not safe yet: BRCA2 native-SAE causal replication."
            ),
            "- Not safe yet: clinical VUS reclassification.",
            (
                "- Safe now: RAD51C third-gene Evo2/ESM checkpoint and DNA/protein mechanism stratification; BAP1 finite-SNV Evo2/ESM checkpoint and boundary support; BAP1/RAD51C benchmark-anchor readiness and pre-specified review/assay panel design."
                if c12_ready
                else "- Safe now: BAP1/RAD51C benchmark-anchor readiness and pre-specified review/assay panel design."
            ),
            "- Not safe yet: BAP1/RAD51C third-gene causal native-SAE replication.",
            "- Not safe given current data: generic multimodal fusion SOTA.",
            "",
            "## Methods Outline",
            "",
            "- Datasets: BRCA1 and BRCA2 saturation genome editing labels; ClinVar unresolved/conflicting BRCA2 missense variants; supplementary collagen structural-gene variants.",
            "- Representations: DNA-side Evo2 LLR and embeddings; protein-side ESM deltas; fold-native sparse autoencoders for mechanism intervention.",
            "- Primary intervention: reconstruct activations through SAE, ablate top sparse features, compare AUROC/AUPRC against original or reconstruction predictions.",
            "- Controls: matched random features, bottom features, label permutation, dose response, raw dense-dimension ablation, benign/model-conflict review-panel arms.",
            "- Application: rank unresolved variants into pathogenic-review, benign-control, split-mechanism, and model-conflict groups; export blinded review and assay sheets.",
            "",
        ]
    )
    lines.extend(build_figure_legends(figures))
    lines.extend(
        [
            "## Reviewer Risk Summary",
            "",
        ]
    )
    if risks.empty:
        lines.append("- Risk register unavailable; regenerate interpretability_manuscript_package first.")
    else:
        for _, row in risks.iterrows():
            lines.append(
                f"- {row['risk_id']} ({row['severity']}): {row['risk']} Current answer: {row['current_answer']} Mitigation: {row['mitigation']}"
            )
    lines.extend(
        [
            "",
            "## Final Submission Gate",
            "",
            (
                "A strong methods/application manuscript can be drafted now around BRCA1 causal interpretability and BRCA2 triage. A top-journal BRCA2 native-SAE replication claim should wait for a successful original-style per-stratum control rerun, improved native-SAE intervention, or external assay/review outcome. A clinical-utility claim should wait until blinded review, temporal validation, or assay readout is available."
                if c9_mixed
                else "A strong methods/application manuscript can be drafted now around BRCA1 causal interpretability and BRCA2 triage. A top-journal mechanistic replication claim should wait until BRCA2 native-SAE causal outputs exist. A clinical-utility claim should wait until blinded review, temporal validation, or assay readout is available."
            ),
            (
                "RAD51C can be included now as a positive third-gene checkpoint/mechanism-stratification result, and BAP1 can be included as finite-SNV positive/boundary support; third-gene native-SAE causal replication remains a future gate."
                if c12_ready
                else "BAP1/RAD51C can be included now as benchmark-ready third-gene application scaffolding, but the third-gene CrossBioSAE claim should wait for the running LLR/checkpoint jobs and native-SAE follow-up."
            ),
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
    sections = build_sections(repo, claims)
    section_path = out / "interpretability_manuscript_draft_sections.csv"
    pd.DataFrame([asdict(s) for s in sections]).to_csv(section_path, index=False)
    report_path = write_report(repo, claims, sections)

    print(f"wrote {section_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
