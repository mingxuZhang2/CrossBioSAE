#!/usr/bin/env python3
"""
Summarize BRCA2 SGE replication outputs and compare them with BRCA1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument("--output-dir", type=Path, default=root / "results" / "interpretability_applications")
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def one_metric(metrics: pd.DataFrame, name: str, col: str = "auroc_for_sge_lof") -> float:
    if metrics.empty or "metric" not in metrics.columns or col not in metrics.columns:
        return float("nan")
    hit = metrics.loc[metrics["metric"].eq(name), col]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def one_summary(summary: pd.DataFrame, col: str) -> float:
    if summary.empty or col not in summary.columns:
        return float("nan")
    return float(summary[col].iloc[0])


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], s[ok]))


def representation_status(root: Path) -> pd.DataFrame:
    variants = root / "data" / "variant" / "brca2" / "brca2_variants.csv"
    esm = root / "results" / "variant" / "brca2_esm_delta.npz"
    evo_llr = root / "results" / "variant" / "brca2_evo2_llr.npz"
    evo = root / "results" / "variant" / "brca2_evo2.npz"
    rows = []
    n = len(pd.read_csv(variants)) if variants.exists() else 0
    rows.append({"artifact": "brca2_variants.csv", "exists": variants.exists(), "detail": f"rows={n}" if n else ""})
    if esm.exists():
        z = np.load(esm)
        rows.append({
            "artifact": "brca2_esm_delta.npz",
            "exists": True,
            "detail": f"pdelta={tuple(z['pdelta'].shape)} pmask_true={int(z['pmask'].sum())}",
        })
    else:
        rows.append({"artifact": "brca2_esm_delta.npz", "exists": False, "detail": ""})
    if evo_llr.exists():
        z = np.load(evo_llr)
        rows.append({
            "artifact": "brca2_evo2_llr.npz",
            "exists": True,
            "detail": f"llr={tuple(z['llr'].shape)} finite_llr={int(np.isfinite(z['llr']).sum())}",
        })
    else:
        rows.append({"artifact": "brca2_evo2_llr.npz", "exists": False, "detail": ""})
    if evo.exists():
        z = np.load(evo)
        rows.append({
            "artifact": "brca2_evo2.npz",
            "exists": True,
            "detail": f"edelta={tuple(z['edelta'].shape)} finite_llr={int(np.isfinite(z['llr']).sum())}",
        })
    else:
        rows.append({"artifact": "brca2_evo2.npz", "exists": False, "detail": ""})
    return pd.DataFrame(rows)


def evo2_llr_checkpoint(root: Path) -> pd.DataFrame:
    variants = root / "data" / "variant" / "brca2" / "brca2_variants.csv"
    evo_llr = root / "results" / "variant" / "brca2_evo2_llr.npz"
    if not variants.exists() or not evo_llr.exists():
        return pd.DataFrame()

    df = pd.read_csv(variants).reset_index(drop=True)
    llr = np.load(evo_llr)["llr"].astype(float)
    if len(df) != len(llr):
        return pd.DataFrame([{"scope": "error", "n": len(df), "n_finite": int(np.isfinite(llr).sum())}])

    y = df["label"].astype(int).to_numpy()
    score = -llr
    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(df), dtype=bool))]
    for col in ["consequence", "brca2_domain"]:
        if col not in df.columns:
            continue
        for val, sub in df.groupby(col, dropna=False):
            mask = np.zeros(len(df), dtype=bool)
            mask[sub.index.to_numpy()] = True
            if mask.sum() >= 40:
                groups.append((f"{col}:{val}", mask))
    if "is_missense" in df.columns:
        missense = df["is_missense"].astype(bool).to_numpy()
        groups.append(("is_missense:true", missense))
        groups.append(("is_missense:false", ~missense))

    rows = []
    for scope, mask in groups:
        yy = y[mask]
        ss = score[mask]
        ok = np.isfinite(ss)
        if ok.sum() < 20 or len(np.unique(yy[ok])) < 2:
            continue
        rows.append({
            "scope": scope,
            "n": int(mask.sum()),
            "n_finite": int(ok.sum()),
            "n_lof": int(yy[ok].sum()),
            "auroc_for_sge_lof": safe_auc(yy, ss),
            "auprc_for_sge_lof": safe_auprc(yy, ss),
        })
    return pd.DataFrame(rows)


def comparison(out: Path) -> pd.DataFrame:
    brca1_metrics = read_csv(out / "brca1_sge_metric_summary.csv")
    brca2_metrics = read_csv(out / "brca2_sge_metric_summary.csv")
    brca1_native = read_csv(out / "brca1_native_finetuned_sae_intervention_summary.csv")
    brca2_native = read_csv(out / "brca2_native_finetuned_sae_intervention_summary.csv")
    rows = [
        {
            "gene": "BRCA1",
            "fusion_auroc": one_metric(brca1_metrics, "fusion prediction"),
            "discordance_auroc": one_metric(brca1_metrics, "cross-modal discordance (1-cos)"),
            "native_recon_auroc": one_summary(brca1_native, "native_sae_recon_auroc"),
            "native_top_ablate_auroc": one_summary(brca1_native, "top_feature_ablate_auroc"),
            "native_delta_auroc": one_summary(brca1_native, "delta_auroc_recon_minus_top_ablate"),
            "native_random_p": one_summary(brca1_native, "empirical_p_random_delta_ge_top_mean"),
            "native_label_perm_p": one_summary(brca1_native, "empirical_p_label_permuted_delta_ge_top_mean"),
        },
        {
            "gene": "BRCA2",
            "fusion_auroc": one_metric(brca2_metrics, "fusion prediction"),
            "discordance_auroc": one_metric(brca2_metrics, "cross-modal discordance (1-cos)"),
            "native_recon_auroc": one_summary(brca2_native, "native_sae_recon_auroc"),
            "native_top_ablate_auroc": one_summary(brca2_native, "top_feature_ablate_auroc"),
            "native_delta_auroc": one_summary(brca2_native, "delta_auroc_recon_minus_top_ablate"),
            "native_random_p": one_summary(brca2_native, "empirical_p_random_delta_ge_top_mean"),
            "native_label_perm_p": one_summary(brca2_native, "empirical_p_label_permuted_delta_ge_top_mean"),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    status = representation_status(root)
    brca2_metrics = read_csv(out / "brca2_sge_metric_summary.csv")
    brca2_evo2_llr = evo2_llr_checkpoint(root)
    brca2_llr_esm = read_csv(out / "brca2_llr_esm_checkpoint_metric_summary.csv")
    brca2_llr_esm_boot = read_csv(out / "brca2_llr_esm_checkpoint_bootstrap_delta.csv")
    brca2_llr_esm_discordance = read_csv(out / "brca2_llr_esm_discordance_category_summary.csv")
    brca2_llr_esm_discordance_domains = read_csv(out / "brca2_llr_esm_discordance_domain_summary.csv")
    brca2_llr_esm_discordance_hotspots = read_csv(out / "brca2_llr_esm_discordance_residue_hotspots.csv")
    brca2_clinvar_candidates = read_csv(out / "brca2_clinvar_interpretability_candidates_tier_summary.csv")
    brca2_clinvar_known = read_csv(out / "brca2_clinvar_interpretability_candidates_known_validation.csv")
    brca2_clinvar_top = read_csv(out / "brca2_clinvar_interpretability_candidates_top_candidates.csv")
    brca2_review_panel = read_csv(out / "brca2_interpretability_review_panel_summary.csv")
    brca2_review_panel_variants = read_csv(out / "brca2_interpretability_review_panel.csv")
    brca2_temporal_candidate = read_csv(out / "brca2_clinvar_temporal_context_candidate_tier_summary.csv")
    brca2_temporal_old = read_csv(out / "brca2_clinvar_temporal_context_candidate_old_status_summary.csv")
    brca2_temporal_panel = read_csv(out / "brca2_clinvar_temporal_context_panel_summary.csv")
    brca2_temporal_panel_old = read_csv(out / "brca2_clinvar_temporal_context_panel_old_status_summary.csv")
    brca2_prospective_panel = read_csv(out / "brca2_prospective_followup_panel_summary.csv")
    brca2_prospective_domains = read_csv(out / "brca2_prospective_followup_panel_domain_summary.csv")
    brca2_prospective_variants = read_csv(out / "brca2_prospective_followup_panel.csv")
    brca2_prospective_sanity = read_csv(out / "brca2_prospective_followup_panel_old_pathogenic_sanity_checks.csv")
    brca2_protocol_hypotheses = read_csv(out / "brca2_review_assay_protocol_arm_hypotheses.csv")
    brca2_protocol_manifest = read_csv(out / "brca2_review_assay_protocol_manifest.csv")
    brca2_stat_endpoints = read_csv(out / "brca2_assay_statistical_plan_endpoints.csv")
    brca2_stat_thresholds = read_csv(out / "brca2_assay_statistical_plan_fisher_thresholds.csv")
    brca2_stat_power = read_csv(out / "brca2_assay_statistical_plan_power_grid.csv")
    brca2_stat_split = read_csv(out / "brca2_assay_statistical_plan_split_thresholds.csv")
    brca2_esm_only = read_csv(out / "brca2_esm_only_replication_metric_summary.csv")
    brca2_esm_null = read_csv(out / "brca2_esm_only_replication_null_summary.csv")
    brca2_esm_strata = read_csv(out / "brca2_esm_only_replication_strata.csv")
    brca2_tests = read_csv(out / "brca2_sge_stat_tests.csv")
    brca2_quartiles = read_csv(out / "brca2_sge_discordance_quartiles.csv")
    brca2_shared = read_csv(out / "brca2_shared_embedding_bootstrap.csv")
    brca2_fusion = read_csv(root / "results" / "variant" / "brca2_pretrained_fusion_bootstrap.csv")
    brca2_native = read_csv(out / "brca2_native_finetuned_sae_intervention_summary.csv")
    brca2_native_strata = read_csv(out / "brca2_native_finetuned_sae_intervention_stratum_summary.csv")
    comp = comparison(out)

    comp.to_csv(out / "brca1_brca2_replication_comparison.csv", index=False)
    brca2_evo2_llr.to_csv(out / "brca2_evo2_llr_checkpoint.csv", index=False)

    ready = status["exists"].all() and not brca2_metrics.empty and not brca2_native.empty
    interpretation = (
        "BRCA2 replication is complete enough to compare against BRCA1."
        if ready
        else "BRCA2 replication is not complete yet; missing downstream or native-SAE outputs should be treated as pending rather than negative evidence."
    )

    lines = [
        "# BRCA2 SGE Replication Summary",
        "",
        "## Artifact Status",
        "",
        table(status),
        "",
        "## BRCA1 vs BRCA2 Key Evidence",
        "",
        table(comp),
        "",
        "## BRCA2 SGE Metric Summary",
        "",
        table(brca2_metrics),
        "",
        "## BRCA2 Evo2 LLR Checkpoint",
        "",
        table(brca2_evo2_llr, max_rows=60),
        "",
        "## BRCA2 ESM + Evo2 LLR Checkpoint",
        "",
        table(brca2_llr_esm, max_rows=80),
        "",
        "## BRCA2 ESM + Evo2 LLR Bootstrap Delta",
        "",
        table(brca2_llr_esm_boot, max_rows=80),
        "",
        "## BRCA2 ESM-vs-Evo2 LLR Missense Discordance",
        "",
        table(brca2_llr_esm_discordance, max_rows=20),
        "",
        "## BRCA2 ESM-vs-Evo2 LLR Domain Discordance",
        "",
        table(brca2_llr_esm_discordance_domains, max_rows=40),
        "",
        "## BRCA2 ESM-vs-Evo2 LLR Residue Hotspots",
        "",
        table(brca2_llr_esm_discordance_hotspots, max_rows=30),
        "",
        "## BRCA2 ClinVar Interpretability Candidate Tiers",
        "",
        table(brca2_clinvar_candidates, max_rows=20),
        "",
        "## BRCA2 ClinVar Known-Label Sanity Check",
        "",
        table(brca2_clinvar_known, max_rows=20),
        "",
        "## BRCA2 ClinVar Top Review Candidates",
        "",
        table(brca2_clinvar_top, max_rows=40),
        "",
        "## BRCA2 Interpretability Review Panel",
        "",
        table(brca2_review_panel, max_rows=20),
        "",
        "## BRCA2 Interpretability Review Panel Variants",
        "",
        table(brca2_review_panel_variants, max_rows=64),
        "",
        "## BRCA2 ClinVar Temporal Context: Candidate Tiers",
        "",
        table(brca2_temporal_candidate, max_rows=20),
        "",
        "## BRCA2 ClinVar Temporal Context: Candidate Old Status",
        "",
        table(brca2_temporal_old, max_rows=40),
        "",
        "## BRCA2 ClinVar Temporal Context: Panel Arms",
        "",
        table(brca2_temporal_panel, max_rows=20),
        "",
        "## BRCA2 ClinVar Temporal Context: Panel Old Status",
        "",
        table(brca2_temporal_panel_old, max_rows=40),
        "",
        "## BRCA2 Prospective Follow-up Panel",
        "",
        table(brca2_prospective_panel, max_rows=20),
        "",
        "## BRCA2 Prospective Follow-up Panel Domain Coverage",
        "",
        table(brca2_prospective_domains, max_rows=40),
        "",
        "## BRCA2 Prospective Follow-up Panel Variants",
        "",
        table(brca2_prospective_variants, max_rows=64),
        "",
        "## BRCA2 Old P/LP-like Sanity Checks Excluded from Prospective Panel",
        "",
        table(brca2_prospective_sanity, max_rows=20),
        "",
        "## BRCA2 Review/Assay Protocol Hypotheses",
        "",
        table(brca2_protocol_hypotheses, max_rows=20),
        "",
        "## BRCA2 Review/Assay Protocol Manifest",
        "",
        table(brca2_protocol_manifest, max_rows=80),
        "",
        "## BRCA2 Assay Statistical Plan Endpoints",
        "",
        table(brca2_stat_endpoints, max_rows=20),
        "",
        "## BRCA2 Assay Fisher Thresholds",
        "",
        table(brca2_stat_thresholds, max_rows=20),
        "",
        "## BRCA2 Assay Power Grid",
        "",
        table(brca2_stat_power, max_rows=40),
        "",
        "## BRCA2 Split-Mechanism Statistical Thresholds",
        "",
        table(brca2_stat_split, max_rows=20),
        "",
        "## BRCA2 ESM-Only Protein-Side Baseline",
        "",
        table(brca2_esm_only),
        "",
        "## BRCA2 ESM-Only Label-Permutation Nulls",
        "",
        table(brca2_esm_null),
        "",
        "## BRCA2 ESM-Only Strata",
        "",
        table(brca2_esm_strata, max_rows=60),
        "",
        "## BRCA2 Shared-Embedding Bootstrap",
        "",
        table(brca2_shared, max_rows=40),
        "",
        "## BRCA2 Fine-Tuned Fusion Bootstrap",
        "",
        table(brca2_fusion, max_rows=40),
        "",
        "## BRCA2 Mechanism Tests",
        "",
        table(brca2_tests, max_rows=60),
        "",
        "## BRCA2 Discordance Quartiles",
        "",
        table(brca2_quartiles),
        "",
        "## BRCA2 Native-SAE Summary",
        "",
        table(brca2_native),
        "",
        "## BRCA2 Native-SAE Strata",
        "",
        table(brca2_native_strata, max_rows=60),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
    ]
    (out / "brca2_replication_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'brca2_replication_summary.md'}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
