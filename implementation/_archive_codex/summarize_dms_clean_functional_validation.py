#!/usr/bin/env python
"""Summarize avGFP DMS clean validation as functional-assay context."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/interpretability_applications")


def find_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def find_ints(pattern: str, text: str) -> tuple[int, ...] | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def value(results: pd.DataFrame, test: str, rep: str) -> float | None:
    hit = results.loc[results["test"].eq(test) & results["rep"].eq(rep)]
    if hit.empty:
        return None
    val = hit.iloc[0]["spearman"]
    return None if pd.isna(val) else float(val)


def fmt(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--clean-results", type=Path, default=Path("results/dms_fitness/clean_results.csv"))
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/interpretability_applications/hpc3_logs/dms_clean_321303.out"),
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(repo / args.clean_results)
    log_text = (repo / args.log).read_text(encoding="utf-8", errors="replace") if (repo / args.log).exists() else ""

    reliable = find_ints(r"Variants with >=2 barcodes: (\d+)", log_text)
    usable = find_ints(r"Usable: (\d+), distinct proteins: (\d+), synonymous: (\d+)", log_text)
    syn = find_ints(r"synonymous groups: (\d+), variants: (\d+)", log_text)
    brightness_std = find_float(r"within-group brightness std mean: ([0-9.]+)", log_text)
    noise_std = find_float(r"meas-noise mean: ([0-9.]+)", log_text)

    protein_a = value(results, "A_groupcv", "Protein-only (ESM2)")
    dna_a = value(results, "A_groupcv", "DNA-only (NT)")
    concat_a = value(results, "A_groupcv", "Concat")
    protein_b = value(results, "B_within_syn", "Protein-only (ESM2)")
    dna_b = value(results, "B_within_syn", "DNA-only (NT)")
    concat_b = value(results, "B_within_syn", "Concat")

    summary = pd.DataFrame(
        [
            {
                "summary_item": "reliable_variants_min2_barcodes",
                "value": reliable[0] if reliable else None,
                "interpretation": "Input variants after barcode reliability filter.",
            },
            {
                "summary_item": "usable_variants",
                "value": usable[0] if usable else None,
                "interpretation": "Usable no-stop variants after sequence construction.",
            },
            {
                "summary_item": "distinct_proteins",
                "value": usable[1] if usable else None,
                "interpretation": "Protein groups used for GroupKFold leakage control.",
            },
            {
                "summary_item": "synonymous_variants",
                "value": usable[2] if usable else None,
                "interpretation": "Variants with wild-type protein sequence.",
            },
            {
                "summary_item": "synonymous_groups_min3",
                "value": syn[0] if syn else None,
                "interpretation": "Protein-identical CDS groups used for within-group codon-level test.",
            },
            {
                "summary_item": "synonymous_group_variants",
                "value": syn[1] if syn else None,
                "interpretation": "Variants in protein-identical CDS groups.",
            },
            {
                "summary_item": "synonymous_brightness_std_mean",
                "value": brightness_std,
                "interpretation": "Mean within-group brightness standard deviation.",
            },
            {
                "summary_item": "synonymous_measurement_noise_mean",
                "value": noise_std,
                "interpretation": "Mean measurement-noise standard deviation from DMS table.",
            },
            {
                "summary_item": "groupcv_concat_minus_protein_spearman",
                "value": concat_a - protein_a if concat_a is not None and protein_a is not None else None,
                "interpretation": "Leakage-controlled cross-modal gain over protein-only.",
            },
            {
                "summary_item": "groupcv_concat_minus_dna_spearman",
                "value": concat_a - dna_a if concat_a is not None and dna_a is not None else None,
                "interpretation": "Leakage-controlled cross-modal gain over DNA-only.",
            },
            {
                "summary_item": "within_syn_dna_spearman",
                "value": dna_b,
                "interpretation": "Codon-level synonymous signal; near zero means weak codon-level generalization.",
            },
            {
                "summary_item": "within_syn_concat_spearman",
                "value": concat_b,
                "interpretation": "Cross-modal synonymous-group signal.",
            },
        ]
    )

    rows = [
        {
            "test": "A_groupcv",
            "claim_boundary": "Leakage-controlled protein-group split; tests broad functional prediction.",
            "protein_spearman": protein_a,
            "dna_spearman": dna_a,
            "concat_spearman": concat_a,
            "concat_minus_protein": concat_a - protein_a if concat_a is not None and protein_a is not None else None,
            "concat_minus_dna": concat_a - dna_a if concat_a is not None and dna_a is not None else None,
            "interpretation": "Weak positive cross-modal functional-assay context if concat exceeds both single modalities.",
        },
        {
            "test": "B_within_synonymous",
            "claim_boundary": "Protein sequence is constant within each group; tests codon-level signal.",
            "protein_spearman": protein_b,
            "dna_spearman": dna_b,
            "concat_spearman": concat_b,
            "concat_minus_protein": None,
            "concat_minus_dna": concat_b - dna_b if concat_b is not None and dna_b is not None else None,
            "interpretation": "Current synonymous/codon-level signal is weak; do not claim strong codon mechanism.",
        },
    ]
    tests = pd.DataFrame(rows)

    summary_path = out / "dms_clean_functional_validation_summary.csv"
    tests_path = out / "dms_clean_functional_validation_tests.csv"
    report_path = out / "dms_clean_functional_validation.md"
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)

    report = [
        "# avGFP DMS Clean Functional Validation",
        "",
        "## Purpose",
        "",
        "This summarizes the HPC3 clean avGFP DMS run as functional-assay context for the interpretability track.",
        "The clean test removes two known confounds: unreliable single-barcode measurements and random-fold leakage across synonymous variants of the same protein.",
        "",
        "## Dataset And Run",
        "",
        f"- Reliable variants with at least two barcodes: {fmt(reliable[0] if reliable else None, 0)}.",
        f"- Usable variants: {fmt(usable[0] if usable else None, 0)}.",
        f"- Distinct protein groups: {fmt(usable[1] if usable else None, 0)}.",
        f"- Synonymous variants: {fmt(usable[2] if usable else None, 0)}.",
        f"- Protein-identical synonymous groups: {fmt(syn[0] if syn else None, 0)} groups / {fmt(syn[1] if syn else None, 0)} variants.",
        f"- Within-group brightness std mean: {fmt(brightness_std)}; measurement-noise mean: {fmt(noise_std)}.",
        "",
        "## Results",
        "",
        tests.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- GroupKFold by protein gives weak but positive cross-modal context: concat Spearman "
        + fmt(concat_a)
        + " versus protein-only "
        + fmt(protein_a)
        + " and DNA-only "
        + fmt(dna_a)
        + ".",
        "- The within-synonymous-group test does not support a strong codon-level mechanism: DNA-only Spearman "
        + fmt(dna_b)
        + " and concat Spearman "
        + fmt(concat_b)
        + ".",
        "- Therefore avGFP DMS is useful as functional-assay method context, but it should not replace BRCA1/BRCA2 as the main interpretable mechanism/application evidence.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"wrote {summary_path}")
    print(f"wrote {tests_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
