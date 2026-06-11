#!/usr/bin/env python3
"""BRCA2 checkpoint: does ESM add signal beyond the Evo2 LLR scalar?"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", type=Path, default=root / "data" / "variant" / "brca2" / "brca2_variants.csv")
    p.add_argument("--esm", type=Path, default=root / "results" / "variant" / "brca2_esm_delta.npz")
    p.add_argument("--evo2-llr", type=Path, default=root / "results" / "variant" / "brca2_evo2_llr.npz")
    p.add_argument("--output-dir", type=Path, default=root / "results" / "interpretability_applications")
    p.add_argument("--output-prefix", default="brca2_llr_esm_checkpoint")
    p.add_argument("--gene-name", default="")
    p.add_argument("--group-col", default="pos")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--bootstrap-sets", type=int, default=2000)
    p.add_argument("--seed", type=int, default=630)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Rewrite the markdown report from existing metric/bootstrap CSVs without recomputing predictions.",
    )
    return p.parse_args()


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


def cv_pred(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> pd.Series:
    pred = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (train, test) in enumerate(gkf.split(x, y, groups), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed + fold),
        )
        model.fit(x[train], y[train])
        pred[test] = model.predict_proba(x[test])[:, 1]
    return pd.Series(pred)


def metric_rows(df: pd.DataFrame, y: np.ndarray, scopes: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    metrics = [
        ("Evo2 LLR zero-shot", "evo2_llr_zero_shot"),
        ("Evo2 LLR CV logistic", "evo2_llr_cv_pred"),
        ("ESM-only CV logistic", "esm_cv_pred"),
        ("ESM+Evo2 LLR CV logistic", "esm_llr_cv_pred"),
    ]
    for scope, mask in scopes.items():
        yy = y[mask]
        for metric, col in metrics:
            vals = df.loc[mask, col].to_numpy(dtype=float)
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n": int(np.isfinite(vals).sum()),
                    "n_lof": int(yy[np.isfinite(vals)].sum()),
                    "auroc_for_sge_lof": safe_auc(yy, vals),
                    "auprc_for_sge_lof": safe_auprc(yy, vals),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    scopes: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    comparisons = [
        ("ESM+LLR minus LLR-CV", "esm_llr_cv_pred", "evo2_llr_cv_pred"),
        ("ESM+LLR minus LLR-zero-shot", "esm_llr_cv_pred", "evo2_llr_zero_shot"),
        ("ESM-only minus LLR-zero-shot", "esm_cv_pred", "evo2_llr_zero_shot"),
    ]
    for label, a_col, b_col in comparisons:
        deltas_by_scope = {scope: [] for scope in scopes}
        for _ in range(repeats):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
            for scope, mask in scopes.items():
                local = mask[idx]
                if local.sum() < 20:
                    deltas_by_scope[scope].append(np.nan)
                    continue
                yy = y[idx][local]
                a = df[a_col].to_numpy(dtype=float)[idx][local]
                b = df[b_col].to_numpy(dtype=float)[idx][local]
                if len(np.unique(yy[np.isfinite(a) & np.isfinite(b)])) < 2:
                    deltas_by_scope[scope].append(np.nan)
                    continue
                deltas_by_scope[scope].append(safe_auc(yy, a) - safe_auc(yy, b))
        for scope, vals in deltas_by_scope.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            rows.append(
                {
                    "comparison": label,
                    "scope": scope,
                    "median_delta_auroc": float(np.median(arr)),
                    "ci_lo": float(np.percentile(arr, 2.5)),
                    "ci_hi": float(np.percentile(arr, 97.5)),
                    "bootstrap_sets": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    try:
        return show.to_markdown(index=False)
    except ImportError:
        return show.to_csv(index=False)


def write_report(out: Path, prefix: str, metrics: pd.DataFrame, boot: pd.DataFrame, gene_name: str) -> None:
    gene = gene_name.strip() or prefix.split("_")[0].upper()
    report = [
        f"# {gene} ESM + Evo2 LLR Checkpoint",
        "",
        "## Purpose",
        "",
        "Use the completed Evo2 LLR checkpoint to ask whether protein-side ESM deltas add held-out signal beyond a strong DNA-side scalar baseline.",
        "",
        "## Metric Summary",
        "",
        table(metrics),
        "",
        "## Paired Group Bootstrap Delta AUROC",
        "",
        table(boot),
        "",
        "## Interpretation",
        "",
        "This checkpoint sets the baseline for the cross-modal claim. If ESM+LLR does not beat LLR alone, the publishable claim should emphasize mechanistic explanation, domain behavior, and native-SAE necessity rather than raw predictive gain over Evo2 LLR.",
        "",
    ]
    (out / f"{prefix}.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    if args.report_only:
        metrics_path = out / f"{prefix}_metric_summary.csv"
        boot_path = out / f"{prefix}_bootstrap_delta.csv"
        if not metrics_path.exists() or not boot_path.exists():
            raise FileNotFoundError(f"Missing existing metric/bootstrap CSVs for prefix {prefix!r}")
        write_report(out, prefix, pd.read_csv(metrics_path), pd.read_csv(boot_path), args.gene_name)
        print(f"wrote {out / f'{prefix}.md'}")
        return

    df = pd.read_csv(args.variants).reset_index(drop=True)
    esm = np.load(args.esm)
    pdelta = esm["pdelta"].astype(np.float32)
    llr = np.load(args.evo2_llr)["llr"].astype(np.float32)
    if len(df) != len(llr) or len(df) != pdelta.shape[0]:
        raise RuntimeError("BRCA2 variant, ESM, and Evo2 LLR row counts differ")
    labels = pd.to_numeric(df["label"], errors="coerce")
    binary_mask = labels.notna().to_numpy()
    if not binary_mask.all():
        excluded = int((~binary_mask).sum())
        print(f"Excluding {excluded} rows without binary label from checkpoint evaluation")
        df = df.loc[binary_mask].reset_index(drop=True)
        pdelta = pdelta[binary_mask]
        llr = llr[binary_mask]
        labels = labels.loc[binary_mask].reset_index(drop=True)
    if args.group_col not in df.columns:
        raise KeyError(f"group column {args.group_col!r} not found")

    y = labels.astype(int).to_numpy()
    groups = df[args.group_col].to_numpy()
    llr_imp = np.where(np.isfinite(llr), llr, np.nanmedian(llr)).astype(np.float32)
    zero_score = -llr
    llr_x = llr_imp[:, None]
    esm_x = pdelta
    esm_llr_x = np.hstack([pdelta, llr_imp[:, None]])

    df = df.copy()
    df["evo2_llr_zero_shot"] = zero_score
    df["evo2_llr_cv_pred"] = cv_pred(llr_x, y, groups, args.n_splits, args.seed)
    df["esm_cv_pred"] = cv_pred(esm_x, y, groups, args.n_splits, args.seed + 100)
    df["esm_llr_cv_pred"] = cv_pred(esm_llr_x, y, groups, args.n_splits, args.seed + 200)

    scopes = {"all": np.ones(len(df), dtype=bool)}
    if "is_missense" in df.columns:
        scopes["missense"] = df["is_missense"].astype(bool).to_numpy()
        scopes["non_missense"] = ~scopes["missense"]
    for col in ["brca2_domain", "consequence"]:
        if col not in df.columns:
            continue
        for val, sub in df.groupby(col, dropna=False):
            mask = np.zeros(len(df), dtype=bool)
            mask[sub.index.to_numpy()] = True
            if mask.sum() >= 200:
                scopes[f"{col}:{val}"] = mask

    metrics = metric_rows(df, y, scopes)
    boot = bootstrap_delta(df, y, groups, scopes, args.bootstrap_sets, args.seed)

    metrics.to_csv(out / f"{prefix}_metric_summary.csv", index=False)
    boot.to_csv(out / f"{prefix}_bootstrap_delta.csv", index=False)
    df.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(out, prefix, metrics, boot, args.gene_name)
    print(metrics.to_string(index=False))
    print()
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
