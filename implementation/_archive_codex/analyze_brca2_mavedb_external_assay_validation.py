#!/usr/bin/env python3
"""Validate BRCA2 mechanism strata and panel arms against public MaveDB assays."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("results/interpretability_applications")

DEFAULT_SCORE_SETS = (
    "urn:mavedb:00001225-a-1",  # BRCA2 exons 15-26 sGE in Hap1 cells
    "urn:mavedb:00001224-a-1",  # BRCA2 arrayed HDR function in VC-8 cells
    "urn:mavedb:00001223-a-1",  # BRCA2 prime-editing essentiality
    "urn:mavedb:00001263-a-1",  # BRCA2 SGE, mostly earlier exons
    "urn:mavedb:00001268-0-1",  # BRCA2 exon 13 saturation mutagenesis global scores
)

LOCAL_CIRCULAR_SCORE_SETS = {
    "urn:mavedb:00001242-a-1",
}


@dataclass
class ScoreSet:
    urn: str
    title: str
    num_variants: int
    published_date: str
    relation_to_local: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=repo_root / OUT_DIR)
    parser.add_argument("--score-set", dest="score_sets", action="append", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def fetch_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch JSON after retries: {url}") from last_error


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/csv"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch CSV after retries: {url}") from last_error


def api_url(urn: str, suffix: str = "") -> str:
    encoded = urllib.parse.quote(urn, safe="")
    return f"https://api.mavedb.org/api/v1/score-sets/{encoded}{suffix}"


def c_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"c\.([A-Za-z0-9_+\-]+[ACGT]>[ACGT])", text)
    return "c." + match.group(1) if match else ""


def genomic_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"g\.(\d+)([ACGT])>([ACGT])", text)
    if not match:
        return ""
    return f"chr13:{match.group(1)}:{match.group(2)}>{match.group(3)}"


def row_genomic_key(row: pd.Series) -> str:
    chrom = str(row.get("chrom", "")).replace("chr", "")
    pos = str(row.get("pos", ""))
    ref = str(row.get("ref", ""))
    alt = str(row.get("alt", ""))
    if not chrom or chrom.lower() == "nan" or not pos or pos.lower() == "nan":
        return ""
    try:
        pos = str(int(float(pos)))
    except ValueError:
        return ""
    return f"chr{chrom}:{pos}:{ref}>{alt}"


def safe_auc(pos_scores: pd.Series, neg_scores: pd.Series) -> float:
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return float("nan")
    scores = np.r_[pos_scores.astype(float).to_numpy(), neg_scores.astype(float).to_numpy()]
    labels = np.r_[np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)]
    if not np.isfinite(scores).all() or len(np.unique(scores)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_mannwhitney(pos_scores: pd.Series, neg_scores: pd.Series) -> float:
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return float("nan")
    return float(mannwhitneyu(pos_scores.astype(float), neg_scores.astype(float), alternative="greater").pvalue)


def fmt(value: object, digits: int = 3, sci: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.2e}" if sci else f"{val:.{digits}f}"


def read_score_set(urn: str, timeout: int) -> tuple[ScoreSet, pd.DataFrame]:
    meta = fetch_json(api_url(urn), timeout)
    title = str(meta.get("title") or "")
    relation = "local_circular_current_sge" if urn in LOCAL_CIRCULAR_SCORE_SETS else "public_external_functional_assay"
    info = ScoreSet(
        urn=urn,
        title=title,
        num_variants=int(meta.get("numVariants") or 0),
        published_date=str(meta.get("publishedDate") or ""),
        relation_to_local=relation,
    )
    text = fetch_text(api_url(urn, "/scores"), timeout)
    scores = pd.read_csv(io.StringIO(text))
    scores["mavedb_urn"] = urn
    scores["mavedb_title"] = title
    scores["relation_to_local"] = relation
    if "score" not in scores.columns:
        raise ValueError(f"MaveDB score set has no score column: {urn}")
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    if "hgvs_nt" not in scores.columns:
        scores["hgvs_nt"] = ""
    if "hgvs_pro" not in scores.columns:
        scores["hgvs_pro"] = ""
    scores["c_key"] = scores["hgvs_nt"].apply(c_key)
    scores["g_key"] = scores["hgvs_nt"].apply(genomic_key)
    if {"chrom", "pos", "ref", "alt"}.issubset(scores.columns):
        row_g = scores.apply(row_genomic_key, axis=1)
        scores.loc[scores["g_key"].eq(""), "g_key"] = row_g[scores["g_key"].eq("")]
    return info, scores


def load_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = pd.read_csv(require_file(repo / "data/variant/brca2/brca2_variants.csv"))
    panel = pd.read_csv(require_file(repo / OUT_DIR / "brca2_review_assay_protocol_manifest.csv"))
    discordance_path = repo / OUT_DIR / "brca2_llr_esm_discordance_missense_scores.csv"
    if discordance_path.exists():
        discordance = pd.read_csv(discordance_path)[
            ["id", "dna_percentile", "protein_percentile", "discordance_category"]
        ].copy()
        variants = variants.merge(discordance, on="id", how="left", validate="one_to_one")
    if "discordance_category" not in variants.columns:
        variants["discordance_category"] = "unavailable"
    variants["c_key"] = variants["c.nom"].apply(c_key)
    variants["g_key"] = variants["g.nom"].apply(genomic_key)
    variants.loc[variants["g_key"].eq(""), "g_key"] = variants.apply(row_genomic_key, axis=1)
    panel["c_key"] = panel["c_nom"].apply(c_key)
    panel["g_key"] = panel["g_nom"].apply(genomic_key)
    panel.loc[panel["g_key"].eq(""), "g_key"] = panel.apply(row_genomic_key, axis=1)
    variants["label"] = variants["label"].astype(int)
    return variants, panel


def match_scores(left: pd.DataFrame, scores: pd.DataFrame, left_name: str) -> pd.DataFrame:
    parts = []
    c_matches = left[left["c_key"].ne("")].merge(
        scores[scores["c_key"].ne("")],
        on="c_key",
        how="inner",
        suffixes=("", "_mavedb"),
    )
    if not c_matches.empty:
        c_matches["match_type"] = "cdna"
        parts.append(c_matches)

    g_matches = left[left["g_key"].ne("")].merge(
        scores[scores["g_key"].ne("")],
        on="g_key",
        how="inner",
        suffixes=("", "_mavedb"),
    )
    if not g_matches.empty:
        g_matches["match_type"] = "genomic"
        parts.append(g_matches)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    key_cols = [left_name, "mavedb_urn"] if left_name in out.columns else ["mavedb_urn"]
    if "accession" in out.columns:
        key_cols.append("accession")
    key_cols.extend(["c_key", "g_key"])
    out = out.drop_duplicates(key_cols, keep="first")
    return out


def orient_scores(local_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for urn, sub in local_matches.groupby("mavedb_urn"):
        sub = sub[np.isfinite(sub["score"].astype(float))].copy()
        if sub.empty or sub["label"].nunique() < 2:
            rows.append(
                {
                    "mavedb_urn": urn,
                    "n_local_matches": len(sub),
                    "n_pathogenic": int(sub["label"].sum()) if "label" in sub.columns else 0,
                    "raw_score_auroc_vs_local_sge_label": float("nan"),
                    "negative_score_auroc_vs_local_sge_label": float("nan"),
                    "orientation": "undetermined",
                    "orientation_multiplier": np.nan,
                    "external_lof_auroc_vs_local_sge_label": float("nan"),
                }
            )
            continue
        raw_auc = float(roc_auc_score(sub["label"], sub["score"]))
        neg_auc = float(roc_auc_score(sub["label"], -sub["score"]))
        multiplier = 1.0 if raw_auc >= neg_auc else -1.0
        rows.append(
            {
                "mavedb_urn": urn,
                "n_local_matches": len(sub),
                "n_pathogenic": int(sub["label"].sum()),
                "raw_score_auroc_vs_local_sge_label": raw_auc,
                "negative_score_auroc_vs_local_sge_label": neg_auc,
                "orientation": "higher_score_more_LOF" if multiplier > 0 else "lower_score_more_LOF",
                "orientation_multiplier": multiplier,
                "external_lof_auroc_vs_local_sge_label": max(raw_auc, neg_auc),
            }
        )
    return pd.DataFrame(rows)


def add_oriented_score(matches: pd.DataFrame, orientation: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches
    out = matches.merge(orientation[["mavedb_urn", "orientation", "orientation_multiplier"]], on="mavedb_urn", how="left")
    out["orientation_multiplier"] = pd.to_numeric(out["orientation_multiplier"], errors="coerce")
    out["external_lof_score"] = out["score"].astype(float) * out["orientation_multiplier"]
    return out


def summarize_category_tests(local_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if local_matches.empty:
        return pd.DataFrame()
    for urn, sub in local_matches.groupby("mavedb_urn"):
        for comparison, pos_mask, neg_mask in [
            ("both_high_vs_not_both_high", sub["discordance_category"].eq("both_high"), ~sub["discordance_category"].eq("both_high")),
            ("both_high_vs_both_low", sub["discordance_category"].eq("both_high"), sub["discordance_category"].eq("both_low")),
        ]:
            pos = sub.loc[pos_mask, "external_lof_score"]
            neg = sub.loc[neg_mask, "external_lof_score"]
            rows.append(
                {
                    "mavedb_urn": urn,
                    "comparison": comparison,
                    "n_positive_group": len(pos),
                    "n_negative_group": len(neg),
                    "positive_mean_external_lof_score": float(pos.mean()) if len(pos) else float("nan"),
                    "negative_mean_external_lof_score": float(neg.mean()) if len(neg) else float("nan"),
                    "delta_mean_external_lof_score": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan"),
                    "external_lof_auroc": safe_auc(pos, neg),
                    "mannwhitney_p_greater": safe_mannwhitney(pos, neg),
                }
            )
    return pd.DataFrame(rows)


def summarize_panel_tests(panel_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    arm_rows = []
    if panel_matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    for urn, sub in panel_matches.groupby("mavedb_urn"):
        for arm, arm_sub in sub.groupby("panel_arm"):
            arm_rows.append(
                {
                    "mavedb_urn": urn,
                    "panel_arm": arm,
                    "n_panel_matches": len(arm_sub),
                    "mean_external_lof_score": float(arm_sub["external_lof_score"].mean()),
                    "median_external_lof_score": float(arm_sub["external_lof_score"].median()),
                    "mean_raw_score": float(arm_sub["score"].astype(float).mean()),
                }
            )
        pos = sub.loc[sub["panel_arm"].eq("prospective_pathogenic_review"), "external_lof_score"]
        benign = sub.loc[sub["panel_arm"].eq("prospective_benign_controls"), "external_lof_score"]
        conflict = sub.loc[sub["panel_arm"].eq("prospective_model_conflict_controls"), "external_lof_score"]
        split = sub.loc[sub["panel_arm"].eq("prospective_split_mechanism_tests"), "external_lof_score"]
        for comparison, neg in [
            ("pathogenic_review_vs_benign_controls", benign),
            ("pathogenic_review_vs_model_conflict_controls", conflict),
            ("pathogenic_review_vs_split_mechanism_tests", split),
        ]:
            rows.append(
                {
                    "mavedb_urn": urn,
                    "comparison": comparison,
                    "n_pathogenic_review_matches": len(pos),
                    "n_control_matches": len(neg),
                    "pathogenic_review_mean_external_lof_score": float(pos.mean()) if len(pos) else float("nan"),
                    "control_mean_external_lof_score": float(neg.mean()) if len(neg) else float("nan"),
                    "delta_mean_external_lof_score": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan"),
                    "external_lof_auroc": safe_auc(pos, neg),
                    "mannwhitney_p_greater": safe_mannwhitney(pos, neg),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(arm_rows)


def write_report(
    path: Path,
    score_sets: pd.DataFrame,
    orientation: pd.DataFrame,
    category_tests: pd.DataFrame,
    panel_tests: pd.DataFrame,
    panel_arm_summary: pd.DataFrame,
) -> None:
    hap1 = panel_tests[
        panel_tests["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
        & panel_tests["comparison"].eq("pathogenic_review_vs_benign_controls")
    ]
    hap1_row = hap1.iloc[0].to_dict() if not hap1.empty else {}
    both_high = category_tests[
        category_tests["mavedb_urn"].eq("urn:mavedb:00001225-a-1")
        & category_tests["comparison"].eq("both_high_vs_both_low")
    ]
    both_high_row = both_high.iloc[0].to_dict() if not both_high.empty else {}

    lines = [
        "# BRCA2 MaveDB External Assay Validation",
        "",
        "## Purpose",
        "",
        "This analysis checks whether BRCA2 mechanism strata and the prospective review/assay panel are supported by public MaveDB functional assays that were not used as the local BRCA2 SGE label source. The current local BRCA2 label source matches the mESC sGE score set; therefore that local-circular score set is excluded from the default external tests.",
        "",
        "## Score Sets",
        "",
        score_sets.to_markdown(index=False) if not score_sets.empty else "_No score sets loaded._",
        "",
        "## Main Readout",
        "",
    ]
    if hap1_row:
        lines.extend(
            [
                "- HAP1 sGE external panel endpoint: pathogenic-review arm matched "
                + fmt(hap1_row.get("n_pathogenic_review_matches"), 0)
                + " variants and benign-control arm matched "
                + fmt(hap1_row.get("n_control_matches"), 0)
                + " variants; external-lof AUROC "
                + fmt(hap1_row.get("external_lof_auroc"))
                + ", Mann-Whitney p="
                + fmt(hap1_row.get("mannwhitney_p_greater"), 4),
            ]
        )
    if both_high_row:
        lines.extend(
            [
                "- HAP1 sGE mechanism endpoint: both-high vs both-low external-lof AUROC "
                + fmt(both_high_row.get("external_lof_auroc"))
                + ", delta mean "
                + fmt(both_high_row.get("delta_mean_external_lof_score"))
                + ", p="
                + fmt(both_high_row.get("mannwhitney_p_greater"), 4),
            ]
        )
    lines.extend(
        [
            "",
            "## Orientation Calibration",
            "",
            orientation.to_markdown(index=False) if not orientation.empty else "_No orientation rows._",
            "",
            "## Panel Tests",
            "",
            panel_tests.to_markdown(index=False) if not panel_tests.empty else "_No panel tests._",
            "",
            "## Panel Arm Summary",
            "",
            panel_arm_summary.to_markdown(index=False) if not panel_arm_summary.empty else "_No panel arm matches._",
            "",
            "## Mechanism Stratum Tests",
            "",
            category_tests.to_markdown(index=False) if not category_tests.empty else "_No category tests._",
            "",
            "## Interpretation",
            "",
            "- Positive public MaveDB rows strengthen the BRCA2 assay-panel application beyond an internal SGE-only design.",
            "- This remains a public external functional-assay proxy, not a locked blinded clinical review endpoint.",
            "- The HAP1 sGE score set overlaps heavily with the local exons 15-26 SGE design but is a separate public assay/cell context; claims should describe it as orthogonal/public functional support rather than prospective clinical validation.",
            "",
            "## Outputs",
            "",
            "- `brca2_mavedb_external_assay_score_sets.csv`",
            "- `brca2_mavedb_external_assay_orientation.csv`",
            "- `brca2_mavedb_external_assay_local_matches.csv`",
            "- `brca2_mavedb_external_assay_panel_matches.csv`",
            "- `brca2_mavedb_external_assay_panel_tests.csv`",
            "- `brca2_mavedb_external_assay_category_tests.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    score_set_urns = args.score_sets or list(DEFAULT_SCORE_SETS)

    variants, panel = load_inputs(repo)
    info_rows = []
    score_frames = []
    for urn in score_set_urns:
        try:
            info, scores = read_score_set(urn, args.timeout)
        except Exception as exc:
            info_rows.append(
                {
                    "urn": urn,
                    "title": "",
                    "num_variants": 0,
                    "published_date": "",
                    "relation_to_local": "fetch_failed",
                    "fetch_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        info_rows.append(info.__dict__)
        score_frames.append(scores)

    score_sets = pd.DataFrame(info_rows)
    all_scores = pd.concat(score_frames, ignore_index=True, sort=False) if score_frames else pd.DataFrame()
    all_scores = all_scores[all_scores["relation_to_local"].ne("local_circular_current_sge")].copy()

    local_matches = match_scores(variants, all_scores, "id")
    panel_matches = match_scores(panel, all_scores, "id")
    orientation = orient_scores(local_matches)
    local_matches = add_oriented_score(local_matches, orientation)
    panel_matches = add_oriented_score(panel_matches, orientation)
    category_tests = summarize_category_tests(local_matches)
    panel_tests, panel_arm_summary = summarize_panel_tests(panel_matches)

    score_sets.to_csv(out / "brca2_mavedb_external_assay_score_sets.csv", index=False)
    orientation.to_csv(out / "brca2_mavedb_external_assay_orientation.csv", index=False)
    local_matches.to_csv(out / "brca2_mavedb_external_assay_local_matches.csv", index=False)
    panel_matches.to_csv(out / "brca2_mavedb_external_assay_panel_matches.csv", index=False)
    panel_tests.to_csv(out / "brca2_mavedb_external_assay_panel_tests.csv", index=False)
    panel_arm_summary.to_csv(out / "brca2_mavedb_external_assay_panel_arm_summary.csv", index=False)
    category_tests.to_csv(out / "brca2_mavedb_external_assay_category_tests.csv", index=False)
    write_report(
        out / "brca2_mavedb_external_assay_validation.md",
        score_sets,
        orientation,
        category_tests,
        panel_tests,
        panel_arm_summary,
    )

    print(f"wrote {out / 'brca2_mavedb_external_assay_validation.md'}")
    print(f"wrote {out / 'brca2_mavedb_external_assay_panel_tests.csv'}")
    print(f"wrote {out / 'brca2_mavedb_external_assay_category_tests.csv'}")


if __name__ == "__main__":
    main()
