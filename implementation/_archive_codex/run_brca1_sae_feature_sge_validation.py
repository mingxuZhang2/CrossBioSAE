#!/usr/bin/env python3
"""
BRCA1 SGE validation for genome-wide SAE features.

This CPU-friendly analysis projects BRCA1 Evo2 embedding deltas through the same
pretrained DNA encoder used by the genome-wide ClinVar SAE, applies the saved
TopK-SAE, and tests whether sparse features correspond to saturation genome
editing (SGE) function labels.

It also runs a probe-level feature ablation: a logistic probe is trained on SAE
activations in GroupKFold splits, then the fold-specific top coefficient
features are zeroed at test time and compared against matched random feature
sets. This is a probe-level necessity test, not a causal ablation of the
end-to-end fusion model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP


class TopKSAE(nn.Module):
    def __init__(self, d_in: int = 256, d_hidden: int = 2048, k: int = 32):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        _, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        h_sparse = h * mask
        return self.decoder(h_sparse), h_sparse


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--variants", type=Path, default=repo_root / "data" / "variant" / "brca1" / "brca1_variants.csv")
    parser.add_argument("--brca1-evo2", type=Path, default=repo_root / "results" / "variant" / "brca1_evo2.npz")
    parser.add_argument("--clinvar-emb-dir", type=Path, default=repo_root / "results" / "variant")
    parser.add_argument("--clip-checkpoint", type=Path, default=repo_root / "results" / "pretrain" / "crossmodal_clip_40k.pt")
    parser.add_argument("--sae-checkpoint", type=Path, default=repo_root / "results" / "sae_genomewide" / "sae_model.pt")
    parser.add_argument("--cards", type=Path, default=repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "interpretability_applications")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-feature-active", type=int, default=20)
    parser.add_argument("--top-ablate", type=int, default=32)
    parser.add_argument("--random-sets", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=630)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def load_clip(checkpoint: Path, device: str) -> CrossModalCLIP:
    ckpt = torch.load(require_file(checkpoint), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CrossModalCLIP(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def load_sae(checkpoint: Path, device: str) -> TopKSAE:
    state = torch.load(require_file(checkpoint), map_location="cpu", weights_only=True)
    d_hidden, d_in = state["encoder.weight"].shape
    model = TopKSAE(d_in=d_in, d_hidden=d_hidden, k=32)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def project_dna(clip: CrossModalCLIP, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    zs = []
    for start in range(0, len(x), batch_size):
        batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        zs.append(clip.enc_dna(batch).cpu().numpy())
    return np.concatenate(zs, axis=0).astype(np.float32)


def fit_genomewide_z_scaler(
    clip: CrossModalCLIP,
    emb_dir: Path,
    device: str,
    batch_size: int,
    n_shards: int = 8,
) -> StandardScaler:
    scaler = StandardScaler()
    n_seen = 0
    for shard in range(n_shards):
        path = emb_dir / f"clinvar_evo2_emb_shard{shard}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        edelta = data["edelta"].astype(np.float32)
        has_emb = np.any(edelta != 0, axis=1)
        edelta = edelta[has_emb]
        for start in range(0, len(edelta), batch_size):
            z = project_dna(clip, edelta[start : start + batch_size], device, batch_size)
            scaler.partial_fit(z)
            n_seen += len(z)
    if n_seen == 0:
        raise RuntimeError(f"No ClinVar Evo2 embedding shards found in {emb_dir}")
    return scaler


@torch.no_grad()
def sae_acts(sae: TopKSAE, z_scaled: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(z_scaled), batch_size):
        x = torch.tensor(z_scaled[start : start + batch_size], dtype=torch.float32, device=device)
        _, h = sae(x)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def safe_auc(y: np.ndarray, score: np.ndarray, larger_is_lof: bool = True) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    s = score[ok] if larger_is_lof else -score[ok]
    return float(roc_auc_score(y[ok], s))


def safe_auprc(y: np.ndarray, score: np.ndarray, larger_is_lof: bool = True) -> float:
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    s = score[ok] if larger_is_lof else -score[ok]
    return float(average_precision_score(y[ok], s))


def feature_summary(df: pd.DataFrame, acts: np.ndarray, cards: pd.DataFrame, min_active: int) -> pd.DataFrame:
    y = df["sge_lof"].to_numpy(dtype=int)
    function_score = df["function_score"].to_numpy(dtype=float)
    rows = []
    for feature in range(acts.shape[1]):
        a = acts[:, feature].astype(float)
        active = np.abs(a) > 1e-6
        n_active = int(active.sum())
        if n_active < min_active:
            continue
        score_abs = np.abs(a)
        lof = score_abs[y == 1]
        func = score_abs[y == 0]
        try:
            pval = float(mannwhitneyu(lof, func, alternative="greater").pvalue)
        except ValueError:
            pval = float("nan")
        rho = spearmanr(score_abs, function_score, nan_policy="omit").statistic
        rows.append(
            {
                "feature": feature,
                "n_active_brca1": n_active,
                "active_frac_brca1": n_active / len(df),
                "lof_rate_active": float(y[active].mean()) if n_active else float("nan"),
                "lof_rate_inactive": float(y[~active].mean()) if (~active).sum() else float("nan"),
                "mean_abs_act_lof": float(lof.mean()),
                "mean_abs_act_func": float(func.mean()),
                "delta_abs_act_lof_minus_func": float(lof.mean() - func.mean()),
                "auroc_for_sge_lof": safe_auc(y, score_abs, True),
                "auprc_for_sge_lof": safe_auprc(y, score_abs, True),
                "spearman_vs_function_score": float(rho) if np.isfinite(rho) else float("nan"),
                "mannwhitney_lof_gt_func_p": pval,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
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
    return out.sort_values(["mannwhitney_lof_gt_func_p", "auroc_for_sge_lof"], ascending=[True, False])


def category_summary(df: pd.DataFrame, acts: np.ndarray, cards: pd.DataFrame) -> pd.DataFrame:
    categories: dict[str, np.ndarray] = {}
    cards = cards.copy()
    categories["all_features"] = cards["feature"].astype(int).to_numpy()
    categories["novel_pathogenic"] = cards[(cards["modality"].eq("novel")) & (cards["path_rate"] > 0.85)]["feature"].astype(int).to_numpy()
    categories["crossmodal_pathogenic"] = cards[(cards["modality"].eq("cross-modal")) & (cards["path_rate"] > 0.80)]["feature"].astype(int).to_numpy()
    categories["protein_driven_pathogenic"] = cards[(cards["modality"].eq("protein-driven")) & (cards["path_rate"] > 0.80)]["feature"].astype(int).to_numpy()
    categories["dna_driven_pathogenic"] = cards[(cards["modality"].eq("dna-driven")) & (cards["path_rate"] > 0.80)]["feature"].astype(int).to_numpy()

    y = df["sge_lof"].to_numpy(dtype=int)
    function_score = df["function_score"].to_numpy(dtype=float)
    rows = []
    for name, features in categories.items():
        features = features[(features >= 0) & (features < acts.shape[1])]
        if len(features) == 0:
            continue
        score = np.abs(acts[:, features]).sum(axis=1)
        active = score > 1e-6
        rho = spearmanr(score, function_score, nan_policy="omit").statistic
        rows.append(
            {
                "category": name,
                "n_features": len(features),
                "n_active_variants": int(active.sum()),
                "mean_score_lof": float(score[y == 1].mean()),
                "mean_score_func": float(score[y == 0].mean()),
                "delta_lof_minus_func": float(score[y == 1].mean() - score[y == 0].mean()),
                "auroc_for_sge_lof": safe_auc(y, score, True),
                "auprc_for_sge_lof": safe_auprc(y, score, True),
                "spearman_vs_function_score": float(rho) if np.isfinite(rho) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("auroc_for_sge_lof", ascending=False)


def probe_ablation(
    df: pd.DataFrame,
    acts: np.ndarray,
    top_k: int,
    random_sets: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["sge_lof"].to_numpy(dtype=int)
    groups = df["pos_hg19"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    rng = np.random.default_rng(seed)

    pred_full = np.zeros(len(df), dtype=float)
    pred_top_ablate = np.zeros(len(df), dtype=float)
    random_auc_rows = []
    fold_rows = []

    for fold, (tr, te) in enumerate(gkf.split(acts, y, groups)):
        scaler = StandardScaler(with_mean=False).fit(acts[tr])
        xtr = scaler.transform(acts[tr])
        xte = scaler.transform(acts[te])
        clf = LogisticRegression(max_iter=3000, C=0.2, penalty="l2", solver="lbfgs", class_weight="balanced")
        clf.fit(xtr, y[tr])
        pred_full[te] = clf.predict_proba(xte)[:, 1]

        coefs = np.abs(clf.coef_[0])
        eligible = np.where((np.abs(acts[tr]) > 1e-6).sum(axis=0) >= 10)[0]
        top_features = eligible[np.argsort(coefs[eligible])[-top_k:]]

        xte_ablate = xte.copy()
        xte_ablate[:, top_features] = 0.0
        pred_top_ablate[te] = clf.predict_proba(xte_ablate)[:, 1]

        full_auc = roc_auc_score(y[te], pred_full[te])
        top_auc = roc_auc_score(y[te], pred_top_ablate[te])
        fold_rows.append(
            {
                "fold": fold,
                "n_test": len(te),
                "full_auc": full_auc,
                "top_ablate_auc": top_auc,
                "delta_auc_top_ablate": full_auc - top_auc,
                "top_features": ";".join(map(str, top_features.tolist())),
            }
        )

        pool = eligible[~np.isin(eligible, top_features)]
        for i in range(random_sets):
            rand_features = rng.choice(pool, size=min(top_k, len(pool)), replace=False)
            xr = xte.copy()
            xr[:, rand_features] = 0.0
            pr = clf.predict_proba(xr)[:, 1]
            random_auc_rows.append(
                {
                    "fold": fold,
                    "random_set": i,
                    "random_ablate_auc": roc_auc_score(y[te], pr),
                    "delta_auc_random_ablate": full_auc - roc_auc_score(y[te], pr),
                }
            )

    summary = pd.DataFrame(
        [
            {
                "n": len(df),
                "full_probe_auroc": roc_auc_score(y, pred_full),
                "full_probe_auprc": average_precision_score(y, pred_full),
                "top_feature_ablate_auroc": roc_auc_score(y, pred_top_ablate),
                "top_feature_ablate_auprc": average_precision_score(y, pred_top_ablate),
                "delta_auroc_top_ablate": roc_auc_score(y, pred_full) - roc_auc_score(y, pred_top_ablate),
                "delta_auprc_top_ablate": average_precision_score(y, pred_full) - average_precision_score(y, pred_top_ablate),
                "mean_fold_delta_auc_top_ablate": float(pd.DataFrame(fold_rows)["delta_auc_top_ablate"].mean()),
                "mean_random_delta_auc": float(pd.DataFrame(random_auc_rows)["delta_auc_random_ablate"].mean()),
                "sd_random_delta_auc": float(pd.DataFrame(random_auc_rows)["delta_auc_random_ablate"].std()),
                "empirical_p_random_delta_ge_top_mean": float(
                    (
                        (pd.DataFrame(random_auc_rows)["delta_auc_random_ablate"] >= pd.DataFrame(fold_rows)["delta_auc_top_ablate"].mean()).sum()
                        + 1
                    )
                    / (len(random_auc_rows) + 1)
                ),
            }
        ]
    )
    return summary, pd.DataFrame(fold_rows), pd.DataFrame(random_auc_rows)


def write_report(
    path: Path,
    feat: pd.DataFrame,
    cats: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    fold_ablation: pd.DataFrame,
) -> None:
    top_feat = feat.head(25).copy()
    for col in [
        "active_frac_brca1",
        "lof_rate_active",
        "lof_rate_inactive",
        "mean_abs_act_lof",
        "mean_abs_act_func",
        "delta_abs_act_lof_minus_func",
        "auroc_for_sge_lof",
        "auprc_for_sge_lof",
        "spearman_vs_function_score",
        "mannwhitney_lof_gt_func_p",
        "path_rate",
        "path_enrich",
    ]:
        if col in top_feat:
            top_feat[col] = top_feat[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")

    cat_table = cats.copy()
    for col in [
        "mean_score_lof",
        "mean_score_func",
        "delta_lof_minus_func",
        "auroc_for_sge_lof",
        "auprc_for_sge_lof",
        "spearman_vs_function_score",
    ]:
        if col in cat_table:
            cat_table[col] = cat_table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")

    lines = [
        "# BRCA1 SGE Validation of Genome-Wide SAE Features",
        "",
        "## Purpose",
        "",
        "Project BRCA1 Evo2 deltas through the same pretrained DNA encoder and genome-wide TopK-SAE used for ClinVar/VUS analyses, then test whether sparse feature activations correspond to BRCA1 SGE function labels.",
        "",
        "## Feature Category Scores",
        "",
        cat_table.to_markdown(index=False),
        "",
        "## Top Individual SGE-Associated Features",
        "",
        top_feat.to_markdown(index=False),
        "",
        "## Probe-Level Feature Ablation",
        "",
        ablation_summary.to_markdown(index=False),
        "",
        "## Fold Ablation Details",
        "",
        fold_ablation.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Individual SAE features and feature categories provide an orthogonal functional-assay calibration layer for BRCA1 SGE. The ablation is limited to a logistic probe over SAE activations; it does not yet prove causal necessity inside the original fusion model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    df = pd.read_csv(require_file(args.variants)).reset_index(drop=True)
    df["sge_lof"] = df["label"].astype(int)
    edelta = np.load(require_file(args.brca1_evo2))["edelta"].astype(np.float32)
    cards = pd.read_csv(require_file(args.cards))

    clip = load_clip(args.clip_checkpoint, device)
    scaler = fit_genomewide_z_scaler(clip, args.clinvar_emb_dir, device, args.batch_size)
    z_brca1 = project_dna(clip, edelta, device, args.batch_size)
    z_brca1_scaled = scaler.transform(z_brca1).astype(np.float32)
    sae = load_sae(args.sae_checkpoint, device)
    acts = sae_acts(sae, z_brca1_scaled, device, args.batch_size)

    feature_df = feature_summary(df, acts, cards, args.min_feature_active)
    category_df = category_summary(df, acts, cards)
    ablation_summary, ablation_folds, ablation_random = probe_ablation(
        df,
        acts,
        top_k=args.top_ablate,
        random_sets=args.random_sets,
        seed=args.random_seed,
    )

    np.savez_compressed(out / "brca1_sge_sae_acts.npz", acts=acts, z_dna=z_brca1, z_dna_scaled=z_brca1_scaled)
    feature_df.to_csv(out / "brca1_sge_sae_feature_summary.csv", index=False)
    category_df.to_csv(out / "brca1_sge_sae_category_summary.csv", index=False)
    ablation_summary.to_csv(out / "brca1_sge_sae_probe_ablation_summary.csv", index=False)
    ablation_folds.to_csv(out / "brca1_sge_sae_probe_ablation_folds.csv", index=False)
    ablation_random.to_csv(out / "brca1_sge_sae_probe_ablation_random.csv", index=False)
    write_report(out / "brca1_sge_sae_feature_validation.md", feature_df, category_df, ablation_summary, ablation_folds)

    print(f"Wrote BRCA1 SGE SAE feature validation outputs to {out}")
    print(category_df.to_string(index=False))
    print(feature_df.head(20).to_string(index=False))
    print(ablation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
