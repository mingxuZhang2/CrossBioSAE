#!/usr/bin/env python3
"""
Domain-level localization for BRCA1 SGE and genome-wide SAE features.

This analysis asks whether the BRCA1 functional-assay signal found in the
shared representation and SAE feature layer localizes to interpretable BRCA1
regions such as RING and BRCT domains. It uses only cached outputs from the
BRCA1 SGE mechanism and SAE-feature scripts, so it is CPU-only and fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


BRCA1_DOMAIN_RANGES = [
    ("RING_24_65", 24, 65),
    ("coiled_coil_1364_1437", 1364, 1437),
    ("BRCT1_1642_1736", 1642, 1736),
    ("BRCT2_1756_1855", 1756, 1855),
]

DOMAIN_ORDER = [
    "RING_24_65",
    "coiled_coil_1364_1437",
    "BRCT1_1642_1736",
    "BRCT2_1756_1855",
    "coding_other",
    "splice_or_noncoding",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--mechanism-scores",
        type=Path,
        default=repo_root / "results" / "interpretability_applications" / "brca1_sge_mechanism_scores.csv",
    )
    parser.add_argument(
        "--sae-acts",
        type=Path,
        default=repo_root / "results" / "interpretability_applications" / "brca1_sge_sae_acts.npz",
    )
    parser.add_argument(
        "--feature-summary",
        type=Path,
        default=repo_root / "results" / "interpretability_applications" / "brca1_sge_sae_feature_summary.csv",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--min-domain-n", type=int, default=40)
    parser.add_argument("--min-feature-active", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=630)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(q, index=pvals.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return pd.Series(q, index=pvals.index)


def safe_auc(y: np.ndarray, score: np.ndarray, larger_is_lof: bool = True) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    s = score[ok] if larger_is_lof else -score[ok]
    return float(roc_auc_score(y[ok], s))


def safe_auprc(y: np.ndarray, score: np.ndarray, larger_is_lof: bool = True) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    s = score[ok] if larger_is_lof else -score[ok]
    return float(average_precision_score(y[ok], s))


def safe_mannwhitney(x_lof: np.ndarray, x_func: np.ndarray, alternative: str) -> float:
    x_lof = np.asarray(x_lof, dtype=float)
    x_func = np.asarray(x_func, dtype=float)
    x_lof = x_lof[np.isfinite(x_lof)]
    x_func = x_func[np.isfinite(x_func)]
    if len(x_lof) < 5 or len(x_func) < 5:
        return float("nan")
    return float(mannwhitneyu(x_lof, x_func, alternative=alternative).pvalue)


def annotate_domain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def assign(row: pd.Series) -> str:
        if str(row.get("vtype", "")).lower() != "coding":
            return "splice_or_noncoding"
        pos = row.get("aa_pos")
        if pd.isna(pos):
            return "coding_other"
        pos = int(pos)
        for name, start, end in BRCA1_DOMAIN_RANGES:
            if start <= pos <= end:
                return name
        return "coding_other"

    def assign_variant_class(row: pd.Series) -> str:
        consequence = str(row.get("consequence", "")).lower()
        if "missense" in consequence:
            return "missense"
        if "synonymous" in consequence:
            return "synonymous"
        if "nonsense" in consequence:
            return "nonsense"
        if "canonical splice" in consequence:
            return "canonical_splice"
        if "splice" in consequence:
            return "splice_region"
        if "intronic" in consequence:
            return "intronic"
        if "utr" in consequence:
            return "utr"
        return consequence.replace(" ", "_") or "unknown"

    df["brca1_region"] = df.apply(assign, axis=1)
    df["variant_class"] = df.apply(assign_variant_class, axis=1)
    df["region_order"] = df["brca1_region"].map({v: i for i, v in enumerate(DOMAIN_ORDER)}).fillna(99).astype(int)
    return df


def crossval_probe_predictions(df: pd.DataFrame, acts: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["sge_lof"].to_numpy(dtype=int)
    groups = df["pos_hg19"].to_numpy()
    pred = np.zeros(len(df), dtype=float)
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(acts, y, groups)):
        scaler = StandardScaler(with_mean=False).fit(acts[tr])
        xtr = scaler.transform(acts[tr])
        xte = scaler.transform(acts[te])
        clf = LogisticRegression(max_iter=3000, C=0.2, solver="lbfgs", class_weight="balanced")
        clf.fit(xtr, y[tr])
        pred[te] = clf.predict_proba(xte)[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_test": len(te),
                "auroc": safe_auc(y[te], pred[te], True),
                "auprc": safe_auprc(y[te], pred[te], True),
            }
        )
    return pred, pd.DataFrame(rows)


def add_scores(df: pd.DataFrame, acts: np.ndarray, feature_summary: pd.DataFrame, top_features: int) -> tuple[pd.DataFrame, list[int]]:
    df = df.copy()
    top = feature_summary.sort_values(["mannwhitney_lof_gt_func_p", "auroc_for_sge_lof"], ascending=[True, False]).head(top_features)
    top_ids = [int(x) for x in top["feature"].tolist() if 0 <= int(x) < acts.shape[1]]
    abs_acts = np.abs(acts)
    df["sae_all_abs_score"] = abs_acts.sum(axis=1)
    df["sae_top_sge_feature_score"] = abs_acts[:, top_ids].sum(axis=1) if top_ids else 0.0
    rho = spearmanr(df["sae_top_sge_feature_score"], df["function_score"], nan_policy="omit").statistic
    df.attrs["top_feature_spearman_vs_function_score"] = float(rho) if np.isfinite(rho) else float("nan")
    return df, top_ids


def summarize_group(df: pd.DataFrame, group_col: str, min_n: int) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(group_col, sort=False):
        y = sub["sge_lof"].to_numpy(dtype=int)
        if len(sub) < min_n:
            continue
        lof = sub[y == 1]
        func = sub[y == 0]
        row = {
            group_col: name,
            "n": len(sub),
            "n_lof": int(y.sum()),
            "n_func": int((1 - y).sum()),
            "lof_rate": float(y.mean()),
            "mean_function_score": float(sub["function_score"].mean()),
            "mean_shared_cosine_lof": float(lof["shared_cosine"].mean()) if len(lof) else float("nan"),
            "mean_shared_cosine_func": float(func["shared_cosine"].mean()) if len(func) else float("nan"),
            "delta_shared_cosine_lof_minus_func": (
                float(lof["shared_cosine"].mean() - func["shared_cosine"].mean())
                if len(lof) and len(func)
                else float("nan")
            ),
            "p_shared_cosine_lof_lt_func": safe_mannwhitney(
                lof["shared_cosine"].to_numpy(), func["shared_cosine"].to_numpy(), "less"
            )
            if len(lof) and len(func)
            else float("nan"),
            "auroc_function_score": safe_auc(y, sub["function_score"].to_numpy(), larger_is_lof=False),
            "auroc_fusion_pred": safe_auc(y, sub["fusion_pred"].to_numpy(), larger_is_lof=True),
            "auroc_discordance": safe_auc(y, sub["discordance"].to_numpy(), larger_is_lof=True),
            "auroc_shared_cosine": safe_auc(y, sub["shared_cosine"].to_numpy(), larger_is_lof=False),
            "auroc_sae_probe": safe_auc(y, sub["sae_probe_pred"].to_numpy(), larger_is_lof=True),
            "auroc_sae_top_features": safe_auc(y, sub["sae_top_sge_feature_score"].to_numpy(), larger_is_lof=True),
            "auroc_sae_all_features": safe_auc(y, sub["sae_all_abs_score"].to_numpy(), larger_is_lof=True),
            "mean_sae_top_score_lof": float(lof["sae_top_sge_feature_score"].mean()) if len(lof) else float("nan"),
            "mean_sae_top_score_func": float(func["sae_top_sge_feature_score"].mean()) if len(func) else float("nan"),
            "p_sae_top_score_lof_gt_func": safe_mannwhitney(
                lof["sae_top_sge_feature_score"].to_numpy(),
                func["sae_top_sge_feature_score"].to_numpy(),
                "greater",
            )
            if len(lof) and len(func)
            else float("nan"),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if group_col == "brca1_region" and not out.empty:
        out["region_order"] = out[group_col].map({v: i for i, v in enumerate(DOMAIN_ORDER)}).fillna(99).astype(int)
        out = out.sort_values(["region_order", group_col]).drop(columns=["region_order"])
    else:
        out = out.sort_values("n", ascending=False) if not out.empty else out
    return out


def feature_localization(
    df: pd.DataFrame,
    acts: np.ndarray,
    cards: pd.DataFrame,
    min_feature_active: int,
) -> pd.DataFrame:
    rows = []
    y_all = df["sge_lof"].to_numpy(dtype=int)
    abs_acts = np.abs(acts)
    active_all = abs_acts > 1e-6
    for region in DOMAIN_ORDER:
        in_region = df["brca1_region"].eq(region).to_numpy()
        if in_region.sum() == 0 or (~in_region).sum() == 0:
            continue
        y = y_all[in_region]
        for feature in range(acts.shape[1]):
            active = active_all[:, feature]
            a = int((active & in_region).sum())
            b = int((active & ~in_region).sum())
            c = int((~active & in_region).sum())
            d = int((~active & ~in_region).sum())
            if a < min_feature_active:
                continue
            odds, p_domain = fisher_exact([[a, b], [c, d]], alternative="greater")
            score = abs_acts[in_region, feature]
            active_region = active[in_region]
            lof = score[y == 1]
            func = score[y == 0]
            p_lof = safe_mannwhitney(lof, func, "greater") if len(np.unique(y)) == 2 else float("nan")
            rows.append(
                {
                    "brca1_region": region,
                    "feature": feature,
                    "n_region": int(in_region.sum()),
                    "n_active_region": a,
                    "n_active_rest": b,
                    "active_rate_region": float(a / in_region.sum()),
                    "active_rate_rest": float(b / (~in_region).sum()),
                    "domain_active_odds_ratio": float(odds) if np.isfinite(odds) else float("inf"),
                    "p_domain_enrichment": float(p_domain),
                    "lof_rate_active_in_region": float(y[active_region].mean()) if active_region.sum() else float("nan"),
                    "lof_rate_inactive_in_region": float(y[~active_region].mean()) if (~active_region).sum() else float("nan"),
                    "mean_abs_act_lof_in_region": float(lof.mean()) if len(lof) else float("nan"),
                    "mean_abs_act_func_in_region": float(func.mean()) if len(func) else float("nan"),
                    "delta_abs_act_lof_minus_func_in_region": (
                        float(lof.mean() - func.mean()) if len(lof) and len(func) else float("nan")
                    ),
                    "auroc_for_sge_lof_in_region": safe_auc(y, score, larger_is_lof=True),
                    "p_lof_gt_func_in_region": p_lof,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_domain_enrichment"] = bh_qvalues(out["p_domain_enrichment"])
    out["q_lof_gt_func_in_region"] = bh_qvalues(out["p_lof_gt_func_in_region"])
    meta_cols = [
        "feature",
        "modality",
        "concept_v2",
        "path_rate",
        "path_enrich",
        "n_active",
        "corr_ESM1b",
        "corr_GPN",
        "corr_CADD",
        "corr_phyloP",
    ]
    out = out.merge(cards[meta_cols], on="feature", how="left")
    out["region_order"] = out["brca1_region"].map({v: i for i, v in enumerate(DOMAIN_ORDER)}).fillna(99).astype(int)
    return out.sort_values(
        ["region_order", "q_domain_enrichment", "q_lof_gt_func_in_region", "auroc_for_sge_lof_in_region"],
        ascending=[True, True, True, False],
    ).drop(columns=["region_order"])


def format_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def write_report(
    path: Path,
    domain_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    feature_loc: pd.DataFrame,
    functional_feature_loc: pd.DataFrame,
    probe_folds: pd.DataFrame,
    top_feature_ids: list[int],
) -> None:
    covered = set(domain_summary["brca1_region"].tolist()) if not domain_summary.empty else set()
    missing_regions = [r for r in DOMAIN_ORDER if r not in covered]

    top_domain_enriched = []
    for region in DOMAIN_ORDER:
        sub = feature_loc[feature_loc["brca1_region"].eq(region)].head(5)
        if not sub.empty:
            top_domain_enriched.append(sub)
    top_domain_enriched_df = pd.concat(top_domain_enriched, ignore_index=True) if top_domain_enriched else pd.DataFrame()

    top_functional = []
    for region in DOMAIN_ORDER:
        sub = functional_feature_loc[functional_feature_loc["brca1_region"].eq(region)].head(8)
        if not sub.empty:
            top_functional.append(sub)
    top_functional_df = pd.concat(top_functional, ignore_index=True) if top_functional else pd.DataFrame()

    lines = [
        "# BRCA1 Domain-Level SAE Interpretability",
        "",
        "## Purpose",
        "",
        "Test whether BRCA1 SGE mechanism signals localize to known BRCA1 regions rather than only appearing as a whole-gene average.",
        "",
        "Domain definitions use canonical BRCA1/P38398 coordinates: RING 24-65, coiled-coil/PALB2-binding region 1364-1437, BRCT1 1642-1736, and BRCT2 1756-1855.",
        "",
        "Coordinate sources: UniProtKB P38398 reports BRCT1 1642-1736 and BRCT2 1756-1855; SMART reports BRCA1 RING 24-64; the PALB2-binding coiled-coil interval 1364-1437 follows Xia et al. 2009.",
        "",
        "Regions absent from the summary either have no assayed rows in the local BRCA1 SGE table or do not meet the minimum row threshold: "
        + (", ".join(missing_regions) if missing_regions else "none")
        + ".",
        "",
        "## Top SGE-Associated SAE Features Used",
        "",
        ", ".join(map(str, top_feature_ids)),
        "",
        "## Cross-Validated SAE Probe Folds",
        "",
        format_table(probe_folds),
        "",
        "## BRCA1 Region Summary",
        "",
        format_table(domain_summary),
        "",
        "## Variant-Class Summary",
        "",
        format_table(class_summary),
        "",
        "## Top Domain-Enriched Feature Localizations",
        "",
        format_table(top_domain_enriched_df, max_rows=40)
        if not top_domain_enriched_df.empty
        else "No localized features passed the minimum active-count filter.",
        "",
        "## Top LOF-Associated Features Within Region",
        "",
        format_table(top_functional_df, max_rows=48)
        if not top_functional_df.empty
        else "No region-specific LOF-associated features passed the filter.",
        "",
        "## Interpretation",
        "",
        "This analysis is a localization layer. It asks whether the representation and sparse features align with known BRCA1 functional domains and variant classes. It is still not a clinical reclassification experiment and does not replace prospective ClinVar or disease-database validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(require_file(args.mechanism_scores)).reset_index(drop=True)
    acts = np.load(require_file(args.sae_acts))["acts"].astype(np.float32)
    feature_summary = pd.read_csv(require_file(args.feature_summary))
    cards = pd.read_csv(require_file(args.cards))

    if len(df) != acts.shape[0]:
        raise ValueError(f"Row mismatch: mechanism scores has {len(df)} rows but SAE acts has {acts.shape[0]}")

    df = annotate_domain(df)
    df, top_feature_ids = add_scores(df, acts, feature_summary, args.top_features)
    probe_pred, probe_folds = crossval_probe_predictions(df, acts)
    df["sae_probe_pred"] = probe_pred

    domain_summary = summarize_group(df.sort_values("region_order"), "brca1_region", args.min_domain_n)
    class_summary = summarize_group(df, "variant_class", args.min_domain_n)
    feature_loc = feature_localization(df, acts, cards, args.min_feature_active)
    functional_feature_loc = feature_loc[
        (feature_loc["delta_abs_act_lof_minus_func_in_region"] > 0)
        & (feature_loc["p_lof_gt_func_in_region"] < 0.05)
    ].sort_values(
        ["brca1_region", "p_lof_gt_func_in_region", "p_domain_enrichment", "auroc_for_sge_lof_in_region"],
        ascending=[True, True, True, False],
    )

    variant_out = df.drop(columns=["region_order"]).copy()
    variant_out.to_csv(out / "brca1_domain_sae_variant_scores.csv", index=False)
    domain_summary.to_csv(out / "brca1_domain_sae_domain_summary.csv", index=False)
    class_summary.to_csv(out / "brca1_domain_sae_variant_class_summary.csv", index=False)
    feature_loc.to_csv(out / "brca1_domain_sae_feature_localization.csv", index=False)
    functional_feature_loc.to_csv(out / "brca1_domain_sae_functional_feature_localization.csv", index=False)
    probe_folds.to_csv(out / "brca1_domain_sae_probe_folds.csv", index=False)
    write_report(
        out / "brca1_domain_sae_interpretability.md",
        domain_summary,
        class_summary,
        feature_loc,
        functional_feature_loc,
        probe_folds,
        top_feature_ids,
    )

    print(f"Wrote BRCA1 domain SAE interpretability outputs to {out}")
    print(domain_summary.to_string(index=False))
    print()
    print(class_summary.to_string(index=False))
    print()
    print(feature_loc.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
