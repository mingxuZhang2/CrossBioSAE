#!/usr/bin/env python3
"""
Controls and ablations for the collagen-glycine interpretability mechanism.

This is a CPU-only downstream analysis. It asks whether the structural
collagen-gly SAE feature set is specifically activated by collagen Gly-X-Y
substitution variants, rather than merely tracking generic pathogenicity or
feature activation frequency.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


COLLAGEN_GENES = {
    "COL1A1",
    "COL1A2",
    "COL2A1",
    "COL3A1",
    "COL4A3",
    "COL4A4",
    "COL4A5",
    "COL5A1",
    "COL5A2",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "COL7A1",
    "COL9A1",
    "COL11A1",
    "COL11A2",
    "COL17A1",
}
STRUCTURAL_GENES = COLLAGEN_GENES | {"FBN1", "FBN2", "COMP", "FLNA", "FLNB"}
BASELINE_SCORE_COLS = ["CADD", "ESM-1b", "GPN-MSA"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--activation-threshold", type=float, default=1e-6)
    parser.add_argument("--min-path-rate", type=float, default=0.85)
    parser.add_argument("--min-struct-enrichment", type=float, default=1.5)
    parser.add_argument("--min-structural-support", type=int, default=3)
    parser.add_argument("--min-collagen-support", type=int, default=2)
    parser.add_argument("--min-gly-collagen-support", type=int, default=1)
    parser.add_argument("--n-random-sets", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=630)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def read_inputs(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    out_dir = repo_root / "results" / "interpretability_applications"
    known = pd.read_csv(require_file(out_dir / "known_variant_novel_scores.csv"))
    known_acts = np.load(
        require_file(repo_root / "results" / "sae_genomewide" / "sae_genomewide_acts.npz")
    )["acts"]
    vus = pd.read_csv(require_file(repo_root / "results" / "vus_pilot" / "vus_mechanism_profiles.csv"))
    vus_acts = np.load(require_file(repo_root / "results" / "vus_pilot" / "vus_sae_acts.npz"))["acts"]
    novel = pd.read_csv(require_file(repo_root / "results" / "novel_features" / "novel_structural_enrichment.csv"))
    cards = pd.read_csv(require_file(repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv"))

    if len(known) != known_acts.shape[0]:
        raise ValueError(f"Known variant rows and activation rows differ: {len(known)} vs {known_acts.shape[0]}")
    if len(vus) != vus_acts.shape[0]:
        raise ValueError(f"VUS rows and activation rows differ: {len(vus)} vs {vus_acts.shape[0]}")

    return known, known_acts, vus, vus_acts, novel, cards


def strict_missense_pchange(value: object) -> bool:
    text = "" if pd.isna(value) else str(value)
    return bool(re.match(r"^[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", text))


def strict_gly_substitution(value: object) -> bool:
    text = "" if pd.isna(value) else str(value)
    return bool(re.match(r"^Gly\d+[A-Z][a-z]{2}$", text))


def safe_auc(case_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if len(case_scores) < 2 or len(control_scores) < 2:
        return float("nan")
    y = np.r_[np.ones(len(case_scores), dtype=int), np.zeros(len(control_scores), dtype=int)]
    s = np.r_[case_scores, control_scores]
    if not np.isfinite(s).all() or len(np.unique(s)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_mannwhitney(case_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if len(case_scores) < 2 or len(control_scores) < 2:
        return float("nan")
    return float(mannwhitneyu(case_scores, control_scores, alternative="greater").pvalue)


def score_feature_set(acts: np.ndarray, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(acts.shape[0], dtype=float)
    return np.abs(acts[:, features]).sum(axis=1)


def active_feature_count(acts: np.ndarray, features: np.ndarray, threshold: float) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(acts.shape[0], dtype=int)
    return (np.abs(acts[:, features]) > threshold).sum(axis=1)


def select_feature_sets(novel: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = novel[
        (novel["path_rate"] >= args.min_path_rate)
        & (novel["struct_enrichment"] >= args.min_struct_enrichment)
        & (novel["n_structural"] >= args.min_structural_support)
    ].copy()
    collagen_gly = structural[
        (structural["n_collagen"] >= args.min_collagen_support)
        & (structural["n_gly_collagen"] >= args.min_gly_collagen_support)
    ].copy()
    if collagen_gly.empty:
        raise ValueError("No collagen-gly features passed the configured filters.")
    return structural, collagen_gly


def annotate_known(known: pd.DataFrame, known_acts: np.ndarray, structural_ids: np.ndarray, collagen_ids: np.ndarray, threshold: float) -> pd.DataFrame:
    out = known.copy()
    out["gene"] = out["gene"].fillna("")
    out["label"] = out["label"].astype(int)
    out["is_collagen_gene"] = out["gene"].isin(COLLAGEN_GENES)
    out["is_structural_gene"] = out["gene"].isin(STRUCTURAL_GENES)
    out["is_missense_like"] = out["pchange"].apply(strict_missense_pchange)
    out["is_gly_sub_strict"] = out["pchange"].apply(strict_gly_substitution)
    out["collagen_gly_feature_score"] = score_feature_set(known_acts, collagen_ids)
    out["structural_feature_score"] = score_feature_set(known_acts, structural_ids)
    out["collagen_gly_features_active"] = active_feature_count(known_acts, collagen_ids, threshold)
    out["structural_features_active"] = active_feature_count(known_acts, structural_ids, threshold)
    out["structural_score_without_collagen_gly"] = (
        out["structural_feature_score"] - out["collagen_gly_feature_score"]
    ).clip(lower=0)
    return out


def annotate_vus(vus: pd.DataFrame, vus_acts: np.ndarray, structural_ids: np.ndarray, collagen_ids: np.ndarray, threshold: float) -> pd.DataFrame:
    out = vus.copy()
    out["is_collagen_gene"] = out["gene"].isin(COLLAGEN_GENES)
    out["is_structural_gene"] = out["gene"].isin(STRUCTURAL_GENES)
    out["is_gly_sub_strict"] = out["pchange"].apply(strict_gly_substitution)
    out["collagen_gly_feature_score"] = score_feature_set(vus_acts, collagen_ids)
    out["structural_feature_score"] = score_feature_set(vus_acts, structural_ids)
    out["collagen_gly_features_active"] = active_feature_count(vus_acts, collagen_ids, threshold)
    out["structural_features_active"] = active_feature_count(vus_acts, structural_ids, threshold)
    out["structural_score_without_collagen_gly"] = (
        out["structural_feature_score"] - out["collagen_gly_feature_score"]
    ).clip(lower=0)
    out["structural_features_without_collagen_gly_active"] = (
        out["structural_features_active"] - out["collagen_gly_features_active"]
    ).clip(lower=0)
    out["is_tier1_collagen_gly"] = (
        out["is_collagen_gene"] & out["is_gly_sub_strict"] & (out["collagen_gly_features_active"] > 0)
    )
    return out


def class_masks_known(df: pd.DataFrame) -> dict[str, pd.Series]:
    case = df["is_collagen_gene"] & df["is_gly_sub_strict"] & df["label"].eq(1)
    return {
        "case_pathogenic_collagen_gly": case,
        "pathogenic_collagen_non_gly_missense": df["is_collagen_gene"] & ~df["is_gly_sub_strict"] & df["is_missense_like"] & df["label"].eq(1),
        "pathogenic_noncollagen_gly_missense": ~df["is_collagen_gene"] & df["is_gly_sub_strict"] & df["is_missense_like"] & df["label"].eq(1),
        "pathogenic_nonstructural_missense": ~df["is_structural_gene"] & df["is_missense_like"] & df["label"].eq(1),
        "benign_structural_missense": df["is_structural_gene"] & df["is_missense_like"] & df["label"].eq(0),
        "benign_nonstructural_missense": ~df["is_structural_gene"] & df["is_missense_like"] & df["label"].eq(0),
        "benign_all_mapped": df["gene"].ne("") & df["label"].eq(0),
    }


def summarize_classes(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in masks.items():
        sub = df[mask].copy()
        score = sub["collagen_gly_feature_score"].to_numpy(dtype=float)
        structural = sub["structural_feature_score"].to_numpy(dtype=float)
        rows.append(
            {
                "class": name,
                "n": len(sub),
                "n_active_collagen_gly_features": int((sub["collagen_gly_features_active"] > 0).sum()),
                "active_frac": float((sub["collagen_gly_features_active"] > 0).mean()) if len(sub) else float("nan"),
                "mean_collagen_gly_score": float(np.mean(score)) if len(score) else float("nan"),
                "median_collagen_gly_score": float(np.median(score)) if len(score) else float("nan"),
                "p90_collagen_gly_score": float(np.quantile(score, 0.90)) if len(score) else float("nan"),
                "mean_structural_score": float(np.mean(structural)) if len(structural) else float("nan"),
                "median_structural_score": float(np.median(structural)) if len(structural) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def pairwise_tests(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    case_scores = df.loc[masks["case_pathogenic_collagen_gly"], "collagen_gly_feature_score"].to_numpy(dtype=float)
    rows = []
    for control_name, control_mask in masks.items():
        if control_name == "case_pathogenic_collagen_gly":
            continue
        control_scores = df.loc[control_mask, "collagen_gly_feature_score"].to_numpy(dtype=float)
        rows.append(
            {
                "case": "case_pathogenic_collagen_gly",
                "control": control_name,
                "n_case": len(case_scores),
                "n_control": len(control_scores),
                "case_mean": float(np.mean(case_scores)) if len(case_scores) else float("nan"),
                "control_mean": float(np.mean(control_scores)) if len(control_scores) else float("nan"),
                "delta_mean": float(np.mean(case_scores) - np.mean(control_scores)) if len(case_scores) and len(control_scores) else float("nan"),
                "case_active_frac": float(np.mean(case_scores > 0)) if len(case_scores) else float("nan"),
                "control_active_frac": float(np.mean(control_scores > 0)) if len(control_scores) else float("nan"),
                "auroc_case_vs_control": safe_auc(case_scores, control_scores),
                "mannwhitney_greater_p": safe_mannwhitney(case_scores, control_scores),
            }
        )
    return pd.DataFrame(rows)


def matched_pathogenic_controls(df: pd.DataFrame, control_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_mask = (
        df["is_collagen_gene"]
        & df["is_gly_sub_strict"]
        & df["is_missense_like"]
        & df["label"].eq(1)
    )
    controls = df[control_mask].copy()
    cases = df[case_mask].copy()
    controls = controls[np.isfinite(controls[BASELINE_SCORE_COLS]).all(axis=1)]
    cases = cases[np.isfinite(cases[BASELINE_SCORE_COLS]).all(axis=1)]

    if len(cases) < 5 or len(controls) < 5:
        return pd.DataFrame(), pd.DataFrame()

    scaler = StandardScaler()
    x_control = scaler.fit_transform(controls[BASELINE_SCORE_COLS].to_numpy(dtype=float))
    x_case = scaler.transform(cases[BASELINE_SCORE_COLS].to_numpy(dtype=float))
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(x_control)
    dist, idx = nn.kneighbors(x_case)
    matched_controls = controls.iloc[idx[:, 0]].reset_index(drop=True)
    matched_cases = cases.reset_index(drop=True)

    pairs = pd.DataFrame(
        {
            "case_gene": matched_cases["gene"],
            "case_pchange": matched_cases["pchange"],
            "case_score": matched_cases["collagen_gly_feature_score"],
            "control_gene": matched_controls["gene"],
            "control_pchange": matched_controls["pchange"],
            "control_score": matched_controls["collagen_gly_feature_score"],
            "case_CADD": matched_cases["CADD"],
            "control_CADD": matched_controls["CADD"],
            "case_ESM1b": matched_cases["ESM-1b"],
            "control_ESM1b": matched_controls["ESM-1b"],
            "case_GPN_MSA": matched_cases["GPN-MSA"],
            "control_GPN_MSA": matched_controls["GPN-MSA"],
            "match_distance": dist[:, 0],
        }
    )
    pairs["delta_score"] = pairs["case_score"] - pairs["control_score"]

    try:
        pval = float(wilcoxon(pairs["delta_score"], alternative="greater").pvalue)
    except ValueError:
        pval = float("nan")
    summary = pd.DataFrame(
        [
            {
                "n_pairs": len(pairs),
                "unique_control_variants": int(
                    matched_controls[["chrom", "pos", "ref", "alt", "gene"]].drop_duplicates().shape[0]
                ),
                "case_mean_score": float(pairs["case_score"].mean()),
                "control_mean_score": float(pairs["control_score"].mean()),
                "mean_delta": float(pairs["delta_score"].mean()),
                "median_delta": float(pairs["delta_score"].median()),
                "frac_case_greater": float((pairs["delta_score"] > 0).mean()),
                "wilcoxon_greater_p": pval,
                "mean_match_distance": float(pairs["match_distance"].mean()),
            }
        ]
    )
    return summary, pairs


def nearest_candidate_lists(cards: pd.DataFrame, observed_ids: np.ndarray, n_candidates: int = 50) -> dict[int, np.ndarray]:
    meta_cols = ["feature", "n_active", "path_rate", "modality", "corr_ESM1b", "corr_GPN", "corr_CADD"]
    meta = cards[meta_cols].copy()
    meta = meta[(meta["modality"].eq("novel")) & (meta["path_rate"] >= 0.80)]
    meta = meta[~meta["feature"].astype(int).isin(set(map(int, observed_ids)))]
    meta = meta.replace([np.inf, -np.inf], np.nan).fillna(0)
    if meta.empty:
        raise ValueError("No candidate novel features available for random controls.")

    obs_meta = cards[meta_cols].copy()
    obs_meta = obs_meta[obs_meta["feature"].astype(int).isin(set(map(int, observed_ids)))]
    obs_meta = obs_meta.replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(obs_meta) != len(observed_ids):
        missing = set(map(int, observed_ids)) - set(obs_meta["feature"].astype(int))
        raise ValueError(f"Observed features missing from cards: {sorted(missing)}")

    feature_cols = ["n_active", "path_rate", "corr_ESM1b", "corr_GPN", "corr_CADD"]
    cand_x = meta[feature_cols].to_numpy(dtype=float)
    obs_x = obs_meta.set_index("feature").loc[observed_ids, feature_cols].to_numpy(dtype=float)
    cand_x[:, 0] = np.log1p(cand_x[:, 0])
    obs_x[:, 0] = np.log1p(obs_x[:, 0])

    scaler = StandardScaler()
    cand_z = scaler.fit_transform(cand_x)
    obs_z = scaler.transform(obs_x)
    nn = NearestNeighbors(n_neighbors=min(n_candidates, len(meta)), metric="euclidean")
    nn.fit(cand_z)
    _, idx = nn.kneighbors(obs_z)
    cand_features = meta["feature"].astype(int).to_numpy()
    return {int(fid): cand_features[row] for fid, row in zip(observed_ids, idx)}


def draw_random_feature_sets(
    cards: pd.DataFrame,
    observed_ids: np.ndarray,
    n_sets: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    candidate_lists = nearest_candidate_lists(cards, observed_ids)
    random_sets: list[np.ndarray] = []
    for _ in range(n_sets):
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for obs_id in observed_ids:
            candidates = candidate_lists[int(obs_id)]
            unused = [int(x) for x in candidates if int(x) not in chosen_set]
            pool = unused if unused else [int(x) for x in candidates]
            pick = int(rng.choice(pool))
            chosen.append(pick)
            chosen_set.add(pick)
        random_sets.append(np.array(chosen, dtype=int))
    return random_sets


def random_null_analysis(
    acts: np.ndarray,
    case_mask: np.ndarray,
    control_mask: np.ndarray,
    observed_ids: np.ndarray,
    random_sets: list[np.ndarray],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_score = score_feature_set(acts, observed_ids)
    observed_active = active_feature_count(acts, observed_ids, threshold) > 0
    obs_delta = float(observed_score[case_mask].mean() - observed_score[control_mask].mean())
    obs_active_delta = float(observed_active[case_mask].mean() - observed_active[control_mask].mean())

    rows = []
    for i, feature_ids in enumerate(random_sets):
        score = score_feature_set(acts, feature_ids)
        active = active_feature_count(acts, feature_ids, threshold) > 0
        rows.append(
            {
                "random_set": i,
                "n_features": len(feature_ids),
                "delta_mean_score": float(score[case_mask].mean() - score[control_mask].mean()),
                "delta_active_frac": float(active[case_mask].mean() - active[control_mask].mean()),
                "case_mean_score": float(score[case_mask].mean()),
                "control_mean_score": float(score[control_mask].mean()),
                "case_active_frac": float(active[case_mask].mean()),
                "control_active_frac": float(active[control_mask].mean()),
            }
        )
    null = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "observed_delta_mean_score": obs_delta,
                "null_mean_delta_mean_score": float(null["delta_mean_score"].mean()),
                "null_sd_delta_mean_score": float(null["delta_mean_score"].std()),
                "empirical_p_delta_mean_score": float(((null["delta_mean_score"] >= obs_delta).sum() + 1) / (len(null) + 1)),
                "observed_delta_active_frac": obs_active_delta,
                "null_mean_delta_active_frac": float(null["delta_active_frac"].mean()),
                "null_sd_delta_active_frac": float(null["delta_active_frac"].std()),
                "empirical_p_delta_active_frac": float(((null["delta_active_frac"] >= obs_active_delta).sum() + 1) / (len(null) + 1)),
            }
        ]
    )
    return summary, null


def vus_ablation_summary(vus: pd.DataFrame, vus_acts: np.ndarray, observed_ids: np.ndarray, random_sets: list[np.ndarray], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = (vus["is_collagen_gene"] & vus["is_gly_sub_strict"]).to_numpy()
    background = (vus["is_structural_gene"] & ~(vus["is_collagen_gene"] & vus["is_gly_sub_strict"])).to_numpy()
    observed_score = score_feature_set(vus_acts, observed_ids)
    observed_active = active_feature_count(vus_acts, observed_ids, threshold) > 0
    observed_rate_strict = float(observed_active[strict].mean())
    observed_rate_background = float(observed_active[background].mean())
    observed_enrichment = observed_rate_strict / max(observed_rate_background, 1e-12)
    observed_score_strict = float(observed_score[strict].mean())
    observed_score_background = float(observed_score[background].mean())
    observed_score_enrichment = observed_score_strict / max(observed_score_background, 1e-12)

    rows = []
    for i, feature_ids in enumerate(random_sets):
        score = score_feature_set(vus_acts, feature_ids)
        active = active_feature_count(vus_acts, feature_ids, threshold) > 0
        strict_rate = float(active[strict].mean())
        background_rate = float(active[background].mean())
        strict_score = float(score[strict].mean())
        background_score = float(score[background].mean())
        rows.append(
            {
                "random_set": i,
                "strict_collagen_gly_active": int(active[strict].sum()),
                "background_structural_active": int(active[background].sum()),
                "strict_collagen_gly_active_rate": strict_rate,
                "background_structural_active_rate": background_rate,
                "strict_vs_background_enrichment": strict_rate / max(background_rate, 1e-12),
                "strict_collagen_gly_mean_score": strict_score,
                "background_structural_mean_score": background_score,
                "strict_vs_background_score_enrichment": strict_score / max(background_score, 1e-12),
            }
        )
    null = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "strict_collagen_gly_vus": int(strict.sum()),
                "background_structural_vus": int(background.sum()),
                "observed_strict_active": int(observed_active[strict].sum()),
                "observed_background_active": int(observed_active[background].sum()),
                "observed_strict_active_rate": observed_rate_strict,
                "observed_background_active_rate": observed_rate_background,
                "observed_strict_vs_background_enrichment": observed_enrichment,
                "observed_strict_mean_score": observed_score_strict,
                "observed_background_mean_score": observed_score_background,
                "observed_strict_vs_background_score_enrichment": observed_score_enrichment,
                "null_mean_strict_active": float(null["strict_collagen_gly_active"].mean()),
                "null_sd_strict_active": float(null["strict_collagen_gly_active"].std()),
                "empirical_p_strict_active": float(((null["strict_collagen_gly_active"] >= observed_active[strict].sum()).sum() + 1) / (len(null) + 1)),
                "null_mean_enrichment": float(null["strict_vs_background_enrichment"].mean()),
                "null_sd_enrichment": float(null["strict_vs_background_enrichment"].std()),
                "empirical_p_enrichment": float(((null["strict_vs_background_enrichment"] >= observed_enrichment).sum() + 1) / (len(null) + 1)),
                "null_mean_score_enrichment": float(null["strict_vs_background_score_enrichment"].mean()),
                "null_sd_score_enrichment": float(null["strict_vs_background_score_enrichment"].std()),
                "empirical_p_score_enrichment": float(((null["strict_vs_background_score_enrichment"] >= observed_score_enrichment).sum() + 1) / (len(null) + 1)),
            }
        ]
    )
    return summary, null


def ablation_effects(known: pd.DataFrame, vus: pd.DataFrame) -> pd.DataFrame:
    known_case = known[known["is_collagen_gene"] & known["is_gly_sub_strict"] & known["label"].eq(1)]
    vus_tier1 = vus[vus["is_tier1_collagen_gly"]]
    rows = []
    for name, df, score_col, without_col in [
        ("known_pathogenic_collagen_gly", known_case, "structural_feature_score", "structural_score_without_collagen_gly"),
        ("vus_tier1_collagen_gly", vus_tier1, "structural_feature_score", "structural_score_without_collagen_gly"),
    ]:
        full = df[score_col].to_numpy(dtype=float)
        ablated = df[without_col].to_numpy(dtype=float)
        drop = full - ablated
        rows.append(
            {
                "set": name,
                "n": len(df),
                "mean_full_structural_score": float(np.mean(full)) if len(full) else float("nan"),
                "mean_after_removing_collagen_gly_features": float(np.mean(ablated)) if len(ablated) else float("nan"),
                "mean_score_drop": float(np.mean(drop)) if len(drop) else float("nan"),
                "median_score_drop": float(np.median(drop)) if len(drop) else float("nan"),
                "mean_fraction_removed": float(np.mean(drop / np.maximum(full, 1e-12))) if len(drop) else float("nan"),
                "n_drop_to_zero": int((ablated <= 0).sum()) if len(ablated) else 0,
                "frac_drop_to_zero": float((ablated <= 0).mean()) if len(ablated) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    structural_features: pd.DataFrame,
    collagen_features: pd.DataFrame,
    class_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    matched_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    vus_summary: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    def fmt(value: object, digits: int = 4) -> str:
        if pd.isna(value):
            return "nan"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    top_pairwise = pairwise.sort_values("mannwhitney_greater_p").copy()
    for col in [
        "case_mean",
        "control_mean",
        "delta_mean",
        "case_active_frac",
        "control_active_frac",
        "auroc_case_vs_control",
        "mannwhitney_greater_p",
    ]:
        top_pairwise[col] = top_pairwise[col].apply(fmt)

    md = [
        "# Collagen-Gly Feature Controls and Ablations",
        "",
        "## Purpose",
        "",
        "Test whether the collagen-gly SAE feature set is a mechanism-specific signal for collagen Gly-X-Y disruption, rather than a generic pathogenicity score.",
        "",
        "## Feature Sets",
        "",
        f"- Novel structural SAE features retained: {len(structural_features):,}.",
        f"- Collagen-gly subset used for the mechanism test: {len(collagen_features):,}.",
        "- Random controls sample matched novel pathogenic features by activation frequency, path rate, and correlations with baseline scores.",
        "",
        "## Known ClinVar Specificity",
        "",
        class_summary.to_markdown(index=False),
        "",
        "## Pairwise Controls",
        "",
        top_pairwise.to_markdown(index=False),
        "",
        "## Baseline-Matched Pathogenic Missense Control",
        "",
        matched_summary.to_markdown(index=False) if not matched_summary.empty else "_No matched control rows._",
        "",
        "## Random Feature-Set Null",
        "",
        null_summary.to_markdown(index=False),
        "",
        "## VUS Random Feature-Set Control",
        "",
        vus_summary.to_markdown(index=False),
        "",
        "## Feature Ablation Within Structural Score",
        "",
        ablation.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This strengthens the mechanistic claim if the observed collagen-gly feature set separates collagen Gly substitutions from matched pathogenic controls and exceeds matched random feature sets. The ablation here is a triage-score necessity test, not yet a causal intervention on the fusion predictor.",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    known, known_acts, vus, vus_acts, novel, cards = read_inputs(repo_root)
    structural_features, collagen_features = select_feature_sets(novel, args)
    structural_ids = structural_features["feature"].astype(int).to_numpy()
    collagen_ids = collagen_features["feature"].astype(int).to_numpy()

    known_scored = annotate_known(known, known_acts, structural_ids, collagen_ids, args.activation_threshold)
    vus_scored = annotate_vus(vus, vus_acts, structural_ids, collagen_ids, args.activation_threshold)

    masks = class_masks_known(known_scored)
    class_summary = summarize_classes(known_scored, masks)
    pairwise = pairwise_tests(known_scored, masks)
    matched_control_mask = (
        ~known_scored["is_structural_gene"]
        & known_scored["is_missense_like"]
        & known_scored["label"].eq(1)
    )
    matched_summary, matched_pairs = matched_pathogenic_controls(known_scored, matched_control_mask)

    random_sets = draw_random_feature_sets(cards, collagen_ids, args.n_random_sets, rng)
    case_mask = masks["case_pathogenic_collagen_gly"].to_numpy()
    control_mask = masks["pathogenic_nonstructural_missense"].to_numpy()
    null_summary, null_rows = random_null_analysis(
        known_acts, case_mask, control_mask, collagen_ids, random_sets, args.activation_threshold
    )
    vus_summary, vus_null_rows = vus_ablation_summary(
        vus_scored, vus_acts, collagen_ids, random_sets, args.activation_threshold
    )
    ablation = ablation_effects(known_scored, vus_scored)

    known_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "pchange",
        "label",
        "is_collagen_gene",
        "is_structural_gene",
        "is_gly_sub_strict",
        "is_missense_like",
        "collagen_gly_feature_score",
        "collagen_gly_features_active",
        "structural_feature_score",
        "structural_features_active",
        "structural_score_without_collagen_gly",
        *BASELINE_SCORE_COLS,
    ]
    vus_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "pchange",
        "is_collagen_gene",
        "is_gly_sub_strict",
        "collagen_gly_feature_score",
        "collagen_gly_features_active",
        "structural_feature_score",
        "structural_features_active",
        "structural_score_without_collagen_gly",
        "is_tier1_collagen_gly",
    ]

    known_scored[known_cols].to_csv(output_dir / "collagen_gly_known_variant_scores.csv", index=False)
    vus_scored[vus_cols].to_csv(output_dir / "collagen_gly_vus_ablation_scores.csv", index=False)
    class_summary.to_csv(output_dir / "collagen_gly_known_class_summary.csv", index=False)
    pairwise.to_csv(output_dir / "collagen_gly_pairwise_controls.csv", index=False)
    matched_summary.to_csv(output_dir / "collagen_gly_matched_control_summary.csv", index=False)
    matched_pairs.to_csv(output_dir / "collagen_gly_matched_control_pairs.csv", index=False)
    null_summary.to_csv(output_dir / "collagen_gly_random_feature_null_summary.csv", index=False)
    null_rows.to_csv(output_dir / "collagen_gly_random_feature_null.csv", index=False)
    vus_summary.to_csv(output_dir / "collagen_gly_vus_random_feature_summary.csv", index=False)
    vus_null_rows.to_csv(output_dir / "collagen_gly_vus_random_feature_null.csv", index=False)
    ablation.to_csv(output_dir / "collagen_gly_ablation_summary.csv", index=False)

    write_report(
        output_dir / "collagen_gly_feature_controls.md",
        structural_features,
        collagen_features,
        class_summary,
        pairwise,
        matched_summary,
        null_summary,
        vus_summary,
        ablation,
    )

    print(f"Wrote collagen-gly feature control outputs to {output_dir}")
    print(class_summary.to_string(index=False))
    print(pairwise.to_string(index=False))
    print(matched_summary.to_string(index=False))
    print(null_summary.to_string(index=False))
    print(vus_summary.to_string(index=False))
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
