#!/usr/bin/env python3
"""Evaluate RAD51C public missense baselines against SGE labels.

The primary public baseline is AlphaMissense from the GRR hg38 tabix mirror.
The script queries only the RAD51C genomic interval, joins by exact
chromosome/position/ref/alt alleles, and compares it to current Evo2/ESM and
latent-SAE checkpoint scores.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("results/interpretability_applications")
DEFAULT_ALPHA_MISSENSE_URL = (
    "https://grr.iossifovlab.com/hg38/scores/AlphaMissense/"
    "AlphaMissense_hg38_modified.tsv.gz"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--variants",
        type=Path,
        default=repo_root / "data/variant/rad51c/rad51c_grch38_variants.csv",
    )
    parser.add_argument(
        "--checkpoint-scores",
        type=Path,
        default=repo_root / OUT_DIR / "rad51c_llr_esm_checkpoint_variant_scores.csv",
    )
    parser.add_argument(
        "--latent-scores",
        type=Path,
        default=repo_root / OUT_DIR / "rad51c_latent_sae_intervention_variant_scores.csv",
    )
    parser.add_argument(
        "--latent-priorities",
        type=Path,
        default=repo_root / OUT_DIR / "rad51c_latent_sae_application_variant_priorities.csv",
    )
    parser.add_argument("--alphamissense-tsv", type=Path, default=None)
    parser.add_argument("--alphamissense-url", default=DEFAULT_ALPHA_MISSENSE_URL)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--output-prefix", default="rad51c_public_baseline_predictors")
    return parser.parse_args()


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def configure_htslib_ca() -> None:
    if os.environ.get("CURL_CA_BUNDLE") and os.environ.get("SSL_CERT_FILE"):
        return
    for candidate in (
        Path("/etc/ssl/certs/ca-certificates.crt"),
        Path("/etc/pki/tls/certs/ca-bundle.crt"),
    ):
        if candidate.exists():
            os.environ.setdefault("CURL_CA_BUNDLE", str(candidate))
            os.environ.setdefault("SSL_CERT_FILE", str(candidate))
            break


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def alpha_columns(header: list[str]) -> list[str]:
    for line in header:
        if line.startswith("#chrom") or line.startswith("#CHROM"):
            cols = line.lstrip("#").split("\t")
            return ["chrom" if col.lower() == "chrom" else col for col in cols]
    return [
        "chrom",
        "pos",
        "ref",
        "alt",
        "genome",
        "uniprot_id",
        "transcript_id",
        "protein_variant",
        "am_pathogenicity",
        "am_class",
    ]


def fetch_alpha_missense(
    source: str | Path,
    chrom: str,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, str]:
    """Return AlphaMissense rows for a 1-based inclusive interval."""
    configure_htslib_ca()
    try:
        import pysam  # type: ignore
    except ImportError as exc:
        return pd.DataFrame(), f"unavailable:pysam_import_error:{exc}"

    source_text = str(source)
    try:
        tabix = pysam.TabixFile(source_text)
        cols = alpha_columns(list(tabix.header))
        contig = chrom if chrom.startswith("chr") else f"chr{chrom}"
        rows = [line.split("\t") for line in tabix.fetch(contig, max(0, start - 1), end)]
        tabix.close()
    except Exception as exc:  # pragma: no cover - depends on remote htslib state
        return pd.DataFrame(), f"unavailable:tabix_error:{type(exc).__name__}:{exc}"
    if not rows:
        return pd.DataFrame(columns=cols), "available:no_rows_in_interval"
    data = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    data["pos"] = pd.to_numeric(data["pos"], errors="coerce").astype("Int64")
    data["am_pathogenicity"] = pd.to_numeric(data["am_pathogenicity"], errors="coerce")
    return data, "available"


def variant_key(df: pd.DataFrame, chrom_col: str = "chrom", pos_col: str = "pos_hg38") -> pd.Series:
    chrom = df[chrom_col].astype(str).str.replace("^chr", "", regex=True)
    return chrom + "_" + df[pos_col].astype("Int64").astype(str) + "_" + df["ref"].astype(str) + "_" + df["alt"].astype(str)


def prepare_base_variants(args: argparse.Namespace) -> pd.DataFrame:
    variants = read_table(args.checkpoint_scores)
    if variants.empty:
        variants = read_table(args.variants)
    if variants.empty:
        raise FileNotFoundError("RAD51C variant table is missing")

    variants = variants.copy()
    variants["variant_key"] = variant_key(variants)
    variants["binary_lof"] = pd.to_numeric(variants["label"], errors="coerce")
    variants["is_binary_sge_label"] = normalize_bool(variants["is_binary_sge_label"])
    variants["is_missense"] = normalize_bool(variants["is_missense"])
    variants["external_lof_score"] = -pd.to_numeric(variants["function_score"], errors="coerce")

    latent = read_table(args.latent_scores)
    if not latent.empty:
        latent_cols = [
            "id",
            "original_pred",
            "sae_recon_pred",
            "top_feature_ablate_pred",
            "selected_feature_only_pred",
            "bias_only_pred",
        ]
        latent_cols = [col for col in latent_cols if col in latent.columns]
        variants = variants.merge(
            latent[latent_cols].drop_duplicates("id"),
            on="id",
            how="left",
            suffixes=("", "_latent"),
        )

    priorities = read_table(args.latent_priorities)
    if not priorities.empty and "variant_key" in priorities.columns:
        parsed = priorities["variant_key"].astype(str).str.extract(r"^(?P<chrom>[^:]+):(?P<pos>\d+):(?P<ref>[ACGT]+)>(?P<alt>[ACGT]+)$")
        priorities = priorities.copy()
        priorities["chrom"] = parsed["chrom"]
        priorities["pos_hg38"] = pd.to_numeric(parsed["pos"], errors="coerce").astype("Int64")
        priorities["ref"] = parsed["ref"]
        priorities["alt"] = parsed["alt"]
        priorities["variant_key_join"] = variant_key(priorities)
        priority_cols = [
            "variant_key_join",
            "sparse_necessity_drop",
            "sparse_sufficiency_lift",
            "latent_application_score",
            "discordance_category",
            "dna_percentile",
            "protein_percentile",
            "clinvar_simple",
        ]
        priority_cols = [col for col in priority_cols if col in priorities.columns]
        variants = variants.merge(
            priorities[priority_cols].drop_duplicates("variant_key_join"),
            left_on="variant_key",
            right_on="variant_key_join",
            how="left",
        )
        variants = variants.drop(columns=[col for col in ["variant_key_join"] if col in variants.columns])

    return variants


def join_alpha_missense(variants: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    out = variants.copy()
    if alpha.empty:
        out["am_pathogenicity"] = np.nan
        out["am_class"] = np.nan
        out["am_protein_variant"] = np.nan
        out["am_transcript_id"] = np.nan
        return out
    am = alpha.copy()
    am["chrom_join"] = am["chrom"].astype(str).str.replace("^chr", "", regex=True)
    am["variant_key"] = am["chrom_join"] + "_" + am["pos"].astype("Int64").astype(str) + "_" + am["ref"].astype(str) + "_" + am["alt"].astype(str)
    keep = [
        "variant_key",
        "am_pathogenicity",
        "am_class",
        "protein_variant",
        "transcript_id",
        "uniprot_id",
    ]
    keep = [col for col in keep if col in am.columns]
    am = am[keep].drop_duplicates("variant_key")
    am = am.rename(columns={"protein_variant": "am_protein_variant", "transcript_id": "am_transcript_id"})
    return out.merge(am, on="variant_key", how="left")


def metric_row(data: pd.DataFrame, score_col: str, label: str, score_type: str) -> dict[str, Any]:
    y = pd.to_numeric(data["binary_lof"], errors="coerce")
    score = pd.to_numeric(data[score_col], errors="coerce") if score_col in data.columns else pd.Series(np.nan, index=data.index)
    ok = y.notna() & score.notna()
    yy = y[ok].astype(int).to_numpy()
    ss = score[ok].astype(float).to_numpy()
    if len(ss) < 20 or len(np.unique(yy)) < 2 or len(np.unique(ss)) < 2:
        auroc = auprc = pval = float("nan")
    else:
        auroc = float(roc_auc_score(yy, ss))
        auprc = float(average_precision_score(yy, ss))
        pval = float(mannwhitneyu(ss[yy == 1], ss[yy == 0], alternative="greater").pvalue)
    return {
        "baseline": label,
        "score_col": score_col,
        "score_type": score_type,
        "n": int(ok.sum()),
        "n_lof": int(yy.sum()) if len(yy) else 0,
        "coverage_fraction": float(ok.mean()) if len(ok) else float("nan"),
        "auroc_for_sge_lof": auroc,
        "auprc_for_sge_lof": auprc,
        "mannwhitney_p_greater": pval,
    }


def bootstrap_delta(
    data: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    y = pd.to_numeric(data["binary_lof"], errors="coerce")
    left = pd.to_numeric(data[left_col], errors="coerce") if left_col in data.columns else pd.Series(np.nan, index=data.index)
    right = pd.to_numeric(data[right_col], errors="coerce") if right_col in data.columns else pd.Series(np.nan, index=data.index)
    ok = y.notna() & left.notna() & right.notna()
    yy = y[ok].astype(int).to_numpy()
    aa = left[ok].astype(float).to_numpy()
    bb = right[ok].astype(float).to_numpy()
    if len(yy) < 20 or len(np.unique(yy)) < 2:
        return {
            "comparison": label,
            "left_col": left_col,
            "right_col": right_col,
            "n": int(ok.sum()),
            "median_delta_auroc": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_delta_le_zero": float("nan"),
            "bootstrap_sets": 0,
        }
    observed = float(roc_auc_score(yy, aa) - roc_auc_score(yy, bb))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    idx = np.arange(len(yy))
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(yy[sample])) < 2:
            continue
        deltas.append(float(roc_auc_score(yy[sample], aa[sample]) - roc_auc_score(yy[sample], bb[sample])))
    if not deltas:
        ci_lo = ci_hi = p_delta = float("nan")
    else:
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        p_delta = (sum(delta <= 0 for delta in deltas) + 1) / (len(deltas) + 1)
    return {
        "comparison": label,
        "left_col": left_col,
        "right_col": right_col,
        "n": int(ok.sum()),
        "observed_delta_auroc": observed,
        "median_delta_auroc": float(np.median(deltas)) if deltas else float("nan"),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_delta_le_zero": float(p_delta),
        "bootstrap_sets": len(deltas),
    }


def bootstrap_prediction_delta(
    pred_table: pd.DataFrame,
    left_col: str,
    right_col: str,
    label: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    y = pd.to_numeric(pred_table["binary_lof"], errors="coerce")
    left = pd.to_numeric(pred_table[left_col], errors="coerce") if left_col in pred_table.columns else pd.Series(np.nan, index=pred_table.index)
    right = pd.to_numeric(pred_table[right_col], errors="coerce") if right_col in pred_table.columns else pd.Series(np.nan, index=pred_table.index)
    ok = y.notna() & left.notna() & right.notna()
    yy = y[ok].astype(int).to_numpy()
    aa = left[ok].astype(float).to_numpy()
    bb = right[ok].astype(float).to_numpy()
    if len(yy) < 20 or len(np.unique(yy)) < 2:
        return {
            "comparison": label,
            "left_col": left_col,
            "right_col": right_col,
            "n": int(ok.sum()),
            "median_delta_auroc": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_delta_le_zero": float("nan"),
            "bootstrap_sets": 0,
        }
    observed = float(roc_auc_score(yy, aa) - roc_auc_score(yy, bb))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    idx = np.arange(len(yy))
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(yy[sample])) < 2:
            continue
        deltas.append(float(roc_auc_score(yy[sample], aa[sample]) - roc_auc_score(yy[sample], bb[sample])))
    if not deltas:
        ci_lo = ci_hi = p_delta = float("nan")
    else:
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        p_delta = (sum(delta <= 0 for delta in deltas) + 1) / (len(deltas) + 1)
    return {
        "comparison": label,
        "left_col": left_col,
        "right_col": right_col,
        "n": int(ok.sum()),
        "observed_delta_auroc": observed,
        "median_delta_auroc": float(np.median(deltas)) if deltas else float("nan"),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_delta_le_zero": float(p_delta),
        "bootstrap_sets": len(deltas),
    }


def grouped_cv_predictions(
    data: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[pd.Series, str, int]:
    y = pd.to_numeric(data["binary_lof"], errors="coerce")
    available = [col for col in feature_cols if col in data.columns]
    if len(available) != len(feature_cols):
        return pd.Series(np.nan, index=data.index), "missing_feature", 0
    ok = y.notna() & data[feature_cols].notna().all(axis=1)
    pred = pd.Series(np.nan, index=data.index, dtype=float)
    if ok.sum() < 20 or y[ok].nunique() < 2:
        return pred, "insufficient_data", 0
    groups = pd.to_numeric(data.loc[ok, "aa_pos"], errors="coerce")
    groups = groups.fillna(pd.to_numeric(data.loc[ok, "pos_hg38"], errors="coerce")).astype(int)
    yy = y[ok].astype(int).to_numpy()
    xx = data.loc[ok, feature_cols].astype(float).to_numpy()
    unique_groups = int(groups.nunique())
    if unique_groups >= 5:
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(xx, yy, groups=groups.to_numpy())
        scheme = "GroupKFold5_by_aa_pos"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        split_iter = splitter.split(xx, yy)
        scheme = "StratifiedKFold5"
    local_pred = np.full(len(yy), np.nan)
    for train_idx, test_idx in split_iter:
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        pipe.fit(xx[train_idx], yy[train_idx])
        local_pred[test_idx] = pipe.predict_proba(xx[test_idx])[:, 1]
    pred.loc[ok] = local_pred
    return pred, scheme, unique_groups


def fusion_cv_outputs(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        ("am_cv", "AlphaMissense GroupKFold logistic", ["am_pathogenicity"], "public_baseline_cv"),
        ("esm_llr_cv_refit", "ESM+Evo2 LLR GroupKFold logistic", ["esm_llr_cv_pred"], "internal_refit_cv"),
        ("latent_recon_cv", "Latent-SAE recon GroupKFold logistic", ["sae_recon_pred"], "internal_refit_cv"),
        ("latent_app_cv", "Latent-SAE app-score GroupKFold logistic", ["latent_application_score"], "internal_refit_cv"),
        ("am_plus_esm_llr_cv", "AlphaMissense + ESM+LLR", ["am_pathogenicity", "esm_llr_cv_pred"], "public_plus_internal_cv"),
        ("am_plus_latent_recon_cv", "AlphaMissense + latent-SAE recon", ["am_pathogenicity", "sae_recon_pred"], "public_plus_internal_cv"),
        ("am_plus_latent_app_cv", "AlphaMissense + latent-SAE app score", ["am_pathogenicity", "latent_application_score"], "public_plus_internal_cv"),
        (
            "am_plus_all_internal_cv",
            "AlphaMissense + ESM+LLR + latent sparse scores",
            ["am_pathogenicity", "esm_llr_cv_pred", "sae_recon_pred", "sparse_necessity_drop", "latent_application_score"],
            "public_plus_internal_cv",
        ),
    ]
    pred_table = data[["id", "variant_key", "binary_lof", "aa_pos", "pos_hg38"]].copy()
    metric_rows: list[dict[str, Any]] = []
    for i, (pred_col, label, features, kind) in enumerate(specs):
        pred, scheme, n_groups = grouped_cv_predictions(data, features, args.seed + 100 + i)
        pred_table[pred_col] = pred
        ok = pred.notna() & data["binary_lof"].notna()
        yy = data.loc[ok, "binary_lof"].astype(int).to_numpy()
        pp = pred[ok].astype(float).to_numpy()
        if len(pp) < 20 or len(np.unique(yy)) < 2 or len(np.unique(pp)) < 2:
            auroc = auprc = float("nan")
        else:
            auroc = float(roc_auc_score(yy, pp))
            auprc = float(average_precision_score(yy, pp))
        metric_rows.append(
            {
                "baseline": label,
                "prediction_col": pred_col,
                "score_type": kind,
                "feature_cols": ";".join(features),
                "cv_scheme": scheme,
                "n": int(ok.sum()),
                "n_groups": n_groups,
                "n_lof": int(yy.sum()) if len(yy) else 0,
                "auroc_for_sge_lof": auroc,
                "auprc_for_sge_lof": auprc,
            }
        )

    delta_specs = [
        ("am_plus_esm_llr_cv", "am_cv", "AM+ESM+LLR CV minus AM CV"),
        ("am_plus_latent_recon_cv", "am_cv", "AM+latent recon CV minus AM CV"),
        ("am_plus_latent_app_cv", "am_cv", "AM+latent app CV minus AM CV"),
        ("am_plus_all_internal_cv", "am_cv", "AM+all internal CV minus AM CV"),
    ]
    delta_rows = [
        bootstrap_prediction_delta(pred_table, left, right, label, args.bootstrap, args.seed + 500 + i)
        for i, (left, right, label) in enumerate(delta_specs)
    ]
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows), pred_table


def binary_rule_row(data: pd.DataFrame, pred: pd.Series, label: str) -> dict[str, Any]:
    y = pd.to_numeric(data["binary_lof"], errors="coerce")
    ok = y.notna() & pred.notna()
    yy = y[ok].astype(int)
    pp = pred[ok].astype(int)
    if len(yy) == 0:
        odds = pval = float("nan")
    else:
        table = [
            [int(((pp == 1) & (yy == 1)).sum()), int(((pp == 1) & (yy == 0)).sum())],
            [int(((pp == 0) & (yy == 1)).sum()), int(((pp == 0) & (yy == 0)).sum())],
        ]
        odds, pval = fisher_exact(table, alternative="greater")
    return {
        "baseline": label,
        "n": int(ok.sum()),
        "n_pred_positive": int(pp.eq(1).sum()) if len(yy) else 0,
        "n_pred_negative": int(pp.eq(0).sum()) if len(yy) else 0,
        "n_lof_among_pred_positive": int(((pp == 1) & (yy == 1)).sum()) if len(yy) else 0,
        "n_functional_among_pred_positive": int(((pp == 1) & (yy == 0)).sum()) if len(yy) else 0,
        "n_lof_among_pred_negative": int(((pp == 0) & (yy == 1)).sum()) if len(yy) else 0,
        "n_functional_among_pred_negative": int(((pp == 0) & (yy == 0)).sum()) if len(yy) else 0,
        "fisher_oddsratio": float(odds),
        "fisher_p_greater": float(pval),
    }


def build_outputs(joined: pd.DataFrame, source_status: str, n_alpha_interval: int, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    eval_data = joined[joined["is_binary_sge_label"] & joined["is_missense"]].copy()
    metric_specs = [
        ("am_pathogenicity", "AlphaMissense pathogenicity", "public_baseline"),
        ("evo2_llr_zero_shot", "Evo2 LLR zero-shot", "internal_dna_model"),
        ("evo2_llr_cv_pred", "Evo2 LLR CV logistic", "internal_dna_model"),
        ("esm_cv_pred", "ESM-only CV logistic", "internal_protein_model"),
        ("esm_llr_cv_pred", "ESM+Evo2 LLR CV logistic", "internal_fusion_model"),
        ("original_pred", "Latent-SAE source predictor", "internal_sparse_model"),
        ("sae_recon_pred", "Latent-SAE reconstruction", "internal_sparse_model"),
        ("selected_feature_only_pred", "Latent-SAE selected-feature only", "internal_sparse_model"),
        ("sparse_necessity_drop", "Latent sparse necessity drop", "internal_sparse_mechanism"),
        ("latent_application_score", "Latent-SAE application score", "internal_application_ranker"),
    ]
    metrics = pd.DataFrame([metric_row(eval_data, col, label, kind) for col, label, kind in metric_specs])

    class_text = joined.get("am_class", pd.Series(np.nan, index=joined.index)).astype(str).str.lower()
    call_lp_vs_lb = pd.Series(np.nan, index=joined.index)
    call_lp_vs_lb[class_text.eq("likely_pathogenic")] = 1
    call_lp_vs_lb[class_text.eq("likely_benign")] = 0
    call_lp_vs_rest = pd.Series(np.nan, index=joined.index)
    call_lp_vs_rest[class_text.ne("nan") & joined["am_class"].notna()] = class_text.eq("likely_pathogenic").astype(int)
    calls = pd.DataFrame(
        [
            binary_rule_row(eval_data, call_lp_vs_lb.loc[eval_data.index], "AlphaMissense likely_pathogenic vs likely_benign"),
            binary_rule_row(eval_data, call_lp_vs_rest.loc[eval_data.index], "AlphaMissense likely_pathogenic vs all other AM classes"),
        ]
    )

    delta_specs = [
        ("esm_llr_cv_pred", "am_pathogenicity", "ESM+LLR minus AlphaMissense"),
        ("esm_cv_pred", "am_pathogenicity", "ESM-only minus AlphaMissense"),
        ("evo2_llr_zero_shot", "am_pathogenicity", "Evo2 LLR minus AlphaMissense"),
        ("sae_recon_pred", "am_pathogenicity", "Latent-SAE recon minus AlphaMissense"),
        ("selected_feature_only_pred", "am_pathogenicity", "Selected sparse features only minus AlphaMissense"),
        ("esm_llr_cv_pred", "evo2_llr_zero_shot", "ESM+LLR minus Evo2 LLR"),
    ]
    deltas = pd.DataFrame(
        [
            bootstrap_delta(eval_data, left, right, label, args.bootstrap, args.seed + i)
            for i, (left, right, label) in enumerate(delta_specs)
        ]
    )
    fusion_metrics, fusion_deltas, fusion_predictions = fusion_cv_outputs(eval_data, args)

    coverage = pd.DataFrame(
        [
            {
                "item": "alphamissense_source_status",
                "value": source_status,
            },
            {
                "item": "alphamissense_interval_rows",
                "value": str(n_alpha_interval),
            },
            {
                "item": "rad51c_total_rows",
                "value": str(len(joined)),
            },
            {
                "item": "rad51c_binary_missense_rows",
                "value": str(len(eval_data)),
            },
            {
                "item": "rad51c_binary_missense_alphamissense_matched",
                "value": str(int(eval_data["am_pathogenicity"].notna().sum())),
            },
            {
                "item": "rad51c_binary_missense_alphamissense_coverage",
                "value": fmt(eval_data["am_pathogenicity"].notna().mean(), 6),
            },
        ]
    )
    if "am_class" in eval_data.columns:
        class_summary = (
            eval_data.groupby("am_class", dropna=False)
            .agg(
                n=("id", "count"),
                n_lof=("binary_lof", "sum"),
                lof_rate=("binary_lof", "mean"),
                mean_am_pathogenicity=("am_pathogenicity", "mean"),
            )
            .reset_index()
            .sort_values(["lof_rate", "n"], ascending=[False, False])
        )
    else:
        class_summary = pd.DataFrame()

    return {
        "coverage": coverage,
        "class_summary": class_summary,
        "metrics": metrics,
        "deltas": deltas,
        "fusion_metrics": fusion_metrics,
        "fusion_deltas": fusion_deltas,
        "fusion_predictions": fusion_predictions,
        "calls": calls,
        "joined": joined,
    }


def write_report(out: Path, prefix: str, tables: dict[str, pd.DataFrame], alpha_source: str) -> None:
    metrics = tables["metrics"]
    public = metrics[metrics["score_type"].eq("public_baseline")]
    internal = metrics[metrics["score_type"].ne("public_baseline")].sort_values("auroc_for_sge_lof", ascending=False)
    best_internal = internal.iloc[0].to_dict() if not internal.empty else {}
    am = public.iloc[0].to_dict() if not public.empty else {}
    deltas = tables["deltas"]
    am_delta = deltas[deltas["comparison"].eq("ESM+LLR minus AlphaMissense")]
    am_delta_row = am_delta.iloc[0].to_dict() if not am_delta.empty else {}
    fusion = tables["fusion_metrics"]
    fusion_best = fusion.sort_values("auroc_for_sge_lof", ascending=False).iloc[0].to_dict() if not fusion.empty else {}
    fusion_delta = tables["fusion_deltas"]
    fusion_best_delta = fusion_delta[fusion_delta["comparison"].eq("AM+all internal CV minus AM CV")]
    fusion_best_delta_row = fusion_best_delta.iloc[0].to_dict() if not fusion_best_delta.empty else {}
    lines = [
        "# RAD51C Public Baseline Predictors",
        "",
        "## Purpose",
        "",
        "This evaluates public AlphaMissense missense predictions on the RAD51C SGE benchmark and compares them with the current Evo2/ESM and latent-SAE checkpoint scores.",
        "",
        "## AlphaMissense Source",
        "",
        f"- Source: {alpha_source}",
        "- Join key: exact GRCh38 chrom/pos/ref/alt within the RAD51C interval.",
        "",
        "## Main Result",
        "",
        f"- AlphaMissense AUROC/AUPRC on RAD51C binary missense SGE labels: {fmt(am.get('auroc_for_sge_lof'), 4)} / {fmt(am.get('auprc_for_sge_lof'), 4)} over n={fmt(am.get('n'), 0)} matched rows.",
        f"- Best internal score in this table: {best_internal.get('baseline', 'NA')} with AUROC/AUPRC {fmt(best_internal.get('auroc_for_sge_lof'), 4)} / {fmt(best_internal.get('auprc_for_sge_lof'), 4)}.",
        f"- Paired bootstrap ESM+LLR minus AlphaMissense AUROC delta: median {fmt(am_delta_row.get('median_delta_auroc'), 4)} CI [{fmt(am_delta_row.get('ci_lo'), 4)}, {fmt(am_delta_row.get('ci_hi'), 4)}], p(delta<=0)={fmt(am_delta_row.get('p_delta_le_zero'), 4)}.",
        f"- Best RAD51C-specific grouped-CV fusion: {fusion_best.get('baseline', 'NA')} with AUROC/AUPRC {fmt(fusion_best.get('auroc_for_sge_lof'), 4)} / {fmt(fusion_best.get('auprc_for_sge_lof'), 4)}.",
        f"- AM+all-internal grouped-CV minus AM grouped-CV AUROC delta: median {fmt(fusion_best_delta_row.get('median_delta_auroc'), 4)} CI [{fmt(fusion_best_delta_row.get('ci_lo'), 4)}, {fmt(fusion_best_delta_row.get('ci_hi'), 4)}], p(delta<=0)={fmt(fusion_best_delta_row.get('p_delta_le_zero'), 4)}.",
        "",
        "## Interpretation",
        "",
        "RAD51C now has a public missense-predictor comparator, which closes a major review gap for the third-gene application track. AlphaMissense is a very strong standalone baseline here, so the defensible claim is not that current CrossBioSAE scores beat AlphaMissense alone. The more useful application result is that AlphaMissense plus current DNA/protein/latent sparse scores improves grouped-CV performance, suggesting complementary assay-relevant information. This does not replace the stricter pending native-SAE causal replication.",
        "",
        "## Coverage",
        "",
        tables["coverage"].to_markdown(index=False),
        "",
        "## Continuous Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Paired Bootstrap Deltas",
        "",
        deltas.to_markdown(index=False),
        "",
        "## RAD51C-Specific Public+Internal Fusion",
        "",
        "These are 5-fold grouped-CV logistic fusions grouped by amino-acid position. They test whether current internal scores add RAD51C SGE information beyond AlphaMissense, but they are supervised RAD51C-specific benchmarks rather than zero-shot public baselines.",
        "",
        fusion.to_markdown(index=False),
        "",
        "## Fusion Bootstrap Deltas",
        "",
        fusion_delta.to_markdown(index=False),
        "",
        "## AlphaMissense Binary Calls",
        "",
        tables["calls"].to_markdown(index=False),
        "",
        "## AlphaMissense Class Summary",
        "",
        tables["class_summary"].to_markdown(index=False) if not tables["class_summary"].empty else "No AlphaMissense class rows were available.",
        "",
        "## Outputs",
        "",
        f"- {prefix}_coverage.csv",
        f"- {prefix}_alphamissense_class_summary.csv",
        f"- {prefix}_metric_summary.csv",
        f"- {prefix}_bootstrap_delta.csv",
        f"- {prefix}_fusion_cv.csv",
        f"- {prefix}_fusion_bootstrap_delta.csv",
        f"- {prefix}_fusion_predictions.csv",
        f"- {prefix}_binary_calls.csv",
        f"- {prefix}_joined_scores.csv",
    ]
    (out / f"{prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    variants = prepare_base_variants(args)
    chrom = str(variants["chrom"].dropna().astype(str).iloc[0])
    start = int(pd.to_numeric(variants["pos_hg38"], errors="coerce").min())
    end = int(pd.to_numeric(variants["pos_hg38"], errors="coerce").max())
    alpha_source: str | Path = args.alphamissense_tsv if args.alphamissense_tsv else args.alphamissense_url
    alpha, source_status = fetch_alpha_missense(alpha_source, chrom, start, end)
    joined = join_alpha_missense(variants, alpha)
    tables = build_outputs(joined, source_status, len(alpha), args)

    tables["coverage"].to_csv(out / f"{args.output_prefix}_coverage.csv", index=False)
    tables["class_summary"].to_csv(out / f"{args.output_prefix}_alphamissense_class_summary.csv", index=False)
    tables["metrics"].to_csv(out / f"{args.output_prefix}_metric_summary.csv", index=False)
    tables["deltas"].to_csv(out / f"{args.output_prefix}_bootstrap_delta.csv", index=False)
    tables["fusion_metrics"].to_csv(out / f"{args.output_prefix}_fusion_cv.csv", index=False)
    tables["fusion_deltas"].to_csv(out / f"{args.output_prefix}_fusion_bootstrap_delta.csv", index=False)
    tables["fusion_predictions"].to_csv(out / f"{args.output_prefix}_fusion_predictions.csv", index=False)
    tables["calls"].to_csv(out / f"{args.output_prefix}_binary_calls.csv", index=False)
    tables["joined"].to_csv(out / f"{args.output_prefix}_joined_scores.csv", index=False)
    write_report(out, args.output_prefix, tables, str(alpha_source))
    print(f"wrote {out / (args.output_prefix + '.md')}")
    print(f"wrote {out / (args.output_prefix + '_metric_summary.csv')}")


if __name__ == "__main__":
    main()
