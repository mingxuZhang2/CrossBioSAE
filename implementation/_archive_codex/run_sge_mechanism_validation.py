#!/usr/bin/env python3
"""
Generic saturation genome editing mechanism validation.

Uses a variant table plus saved gate-analysis artifacts to test whether
cross-modal signals correspond to orthogonal SGE function labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", required=True)
    p.add_argument("--gate_npz", required=True)
    p.add_argument("--output_dir", default="results/interpretability_applications")
    p.add_argument("--output_prefix", default="sge")
    p.add_argument("--run_name", default="SGE")
    p.add_argument("--domain_col", default=None)
    return p.parse_args()


def safe_auc(y: np.ndarray, s: np.ndarray, larger_is_damaging: bool = True) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_damaging else -s[ok]
    return float(roc_auc_score(y[ok], score))


def safe_auprc(y: np.ndarray, s: np.ndarray, larger_is_damaging: bool = True) -> float:
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    score = s[ok] if larger_is_damaging else -s[ok]
    return float(average_precision_score(y[ok], score))


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe without pandas' optional tabulate dependency."""
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    rows = [
        "| " + " | ".join(show.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in show.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def metric_table(df: pd.DataFrame) -> pd.DataFrame:
    y = df["sge_lof"].to_numpy(dtype=int)
    metrics: list[tuple[str, str, bool]] = [
        ("fusion prediction", "fusion_pred", True),
        ("cross-modal discordance (1-cos)", "discordance", True),
        ("shared cosine", "shared_cosine", False),
        ("protein gate", "gate_protein", True),
        ("dna gate", "gate_dna", False),
    ]
    if "function_score" in df.columns:
        metrics.insert(0, ("SGE damage score (-function_score)", "sge_damage_score", True))
    for col in ["CADD", "phyloP", "esm1b_score", "gpn_msa_score"]:
        if col in df.columns:
            metrics.append((col, col, True))

    rows = []
    function_score = df["function_score"].to_numpy(dtype=float) if "function_score" in df.columns else None
    for name, col, larger in metrics:
        vals = df[col].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        rows.append({
            "metric": name,
            "n": int(ok.sum()),
            "auroc_for_sge_lof": safe_auc(y, vals, larger),
            "auprc_for_sge_lof": safe_auprc(y, vals, larger),
            "spearman_vs_function_score": safe_spearman(vals, function_score) if function_score is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def group_masks(df: pd.DataFrame, domain_col: str | None) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = [("all", np.ones(len(df), dtype=bool))]
    if "vtype" in df.columns:
        for vt in ["coding", "noncoding"]:
            masks.append((vt, df["vtype"].eq(vt).to_numpy()))
    if "is_missense" in df.columns:
        masks.append(("missense", df["is_missense"].astype(bool).to_numpy()))
    if domain_col and domain_col in df.columns:
        for domain, sub in df.groupby(domain_col, dropna=False):
            name = str(domain) if pd.notna(domain) else "domain_missing"
            if len(sub) >= 50:
                masks.append((f"{domain_col}:{name}", df[domain_col].eq(domain).to_numpy()))
    return masks


def group_summary(df: pd.DataFrame, domain_col: str | None) -> pd.DataFrame:
    rows = []
    for group_name, mask in group_masks(df, domain_col):
        sub = df[mask]
        if len(sub) == 0:
            continue
        for label, label_name in [(1, "LOF"), (0, "FUNC")]:
            ss = sub[sub["sge_lof"].eq(label)]
            if len(ss) == 0:
                continue
            row = {
                "group": group_name,
                "sge_class": label_name,
                "n": len(ss),
                "lof_rate_in_group": float(sub["sge_lof"].mean()),
                "mean_fusion_pred": float(ss["fusion_pred"].mean()),
                "mean_shared_cosine": float(ss["shared_cosine"].mean()),
                "mean_discordance": float(ss["discordance"].mean()),
                "mean_gate_protein": float(ss["gate_protein"].mean()),
                "mean_gate_dna": float(ss["gate_dna"].mean()),
            }
            if "function_score" in ss.columns:
                row["mean_function_score"] = float(ss["function_score"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def test_table(df: pd.DataFrame, domain_col: str | None) -> pd.DataFrame:
    rows = []
    for group_name, mask in group_masks(df, domain_col):
        sub = df[mask]
        lof = sub[sub["sge_lof"].eq(1)]
        func = sub[sub["sge_lof"].eq(0)]
        if len(lof) < 10 or len(func) < 10:
            continue
        for col in ["shared_cosine", "discordance", "gate_protein", "gate_dna", "fusion_pred"]:
            stat = mannwhitneyu(lof[col], func[col], alternative="two-sided")
            rows.append({
                "group": group_name,
                "feature": col,
                "lof_mean": float(lof[col].mean()),
                "func_mean": float(func[col].mean()),
                "delta_lof_minus_func": float(lof[col].mean() - func[col].mean()),
                "mannwhitney_p": float(stat.pvalue),
            })
    return pd.DataFrame(rows)


def quartile_table(df: pd.DataFrame) -> pd.DataFrame:
    qcodes = pd.qcut(df["discordance"], 4, labels=False, duplicates="drop")
    max_code = int(qcodes.max()) if qcodes.notna().any() else -1
    qlabels = {i: f"Q{i + 1}" for i in range(max_code + 1)}
    if max_code >= 0:
        qlabels[0] = "Q1_low_discordance"
        qlabels[max_code] = f"Q{max_code + 1}_high_discordance"
    return (
        df.assign(discordance_quartile=qcodes.map(qlabels))
        .groupby("discordance_quartile", observed=False)
        .agg(
            n=("sge_lof", "size"),
            lof_rate=("sge_lof", "mean"),
            mean_fusion_pred=("fusion_pred", "mean"),
            mean_shared_cosine=("shared_cosine", "mean"),
            mean_discordance=("discordance", "mean"),
            mean_function_score=("function_score", "mean") if "function_score" in df.columns else ("sge_lof", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    gate = np.load(args.gate_npz)
    if len(df) != len(gate["pred"]):
        raise RuntimeError(f"variant table has {len(df)} rows but gate predictions have {len(gate['pred'])}")

    zp = gate["z_prot"]
    zd = gate["z_dna"]
    cos = (zp * zd).sum(axis=1) / (np.linalg.norm(zp, axis=1) * np.linalg.norm(zd, axis=1) + 1e-8)

    df = df.copy()
    df["sge_lof"] = df["label"].astype(int)
    if "function_score" in df.columns:
        df["sge_damage_score"] = -df["function_score"].astype(float)
    df["fusion_pred"] = gate["pred"]
    df["gate_protein"] = gate["gate"][:, 0]
    df["gate_dna"] = gate["gate"][:, 1]
    df["shared_cosine"] = cos
    df["discordance"] = 1.0 - cos

    metrics = metric_table(df)
    groups = group_summary(df, args.domain_col)
    tests = test_table(df, args.domain_col)
    quartiles = quartile_table(df)

    score_cols = [
        "id", "chrom", "pos", "ref", "alt", "consequence", "aa_pos", "aa_ref", "aa_alt",
        "function_score", "func_class", "sge_lof", "vtype", "is_missense",
        "fusion_pred", "shared_cosine", "discordance", "gate_protein", "gate_dna",
    ]
    if args.domain_col:
        score_cols.append(args.domain_col)
    score_cols = [c for c in score_cols if c in df.columns]

    prefix = args.output_prefix
    df[score_cols].to_csv(out / f"{prefix}_sge_mechanism_scores.csv", index=False)
    metrics.to_csv(out / f"{prefix}_sge_metric_summary.csv", index=False)
    groups.to_csv(out / f"{prefix}_sge_group_summary.csv", index=False)
    tests.to_csv(out / f"{prefix}_sge_stat_tests.csv", index=False)
    quartiles.to_csv(out / f"{prefix}_sge_discordance_quartiles.csv", index=False)

    lines = [
        f"# {args.run_name} SGE Mechanism Validation",
        "",
        "## Purpose",
        "",
        "Use saturation genome editing function labels as orthogonal evidence for cross-modal mechanism signals.",
        "",
        "## Metric Summary",
        "",
        markdown_table(metrics),
        "",
        "## LOF vs FUNC Summary",
        "",
        markdown_table(groups),
        "",
        "## Discordance Quartiles",
        "",
        markdown_table(quartiles),
        "",
        "## Interpretation",
        "",
        "Fusion prediction, shared-space cosine/discordance, and gate weights are compared against SGE labels. Strong evidence requires held-out fusion prediction plus interpretable signals that separate LOF from FUNC variants in biologically meaningful strata.",
        "",
    ]
    (out / f"{prefix}_sge_mechanism_validation.md").write_text("\n".join(lines))

    print(f"Wrote {args.run_name} SGE mechanism validation outputs to {out}")
    print(metrics.to_string(index=False))
    print(tests.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
