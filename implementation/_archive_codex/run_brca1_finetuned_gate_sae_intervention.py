#!/usr/bin/env python3
"""
BRCA1 fully fine-tuned gate/head SAE feature intervention.

This script uses per-fold checkpoints and scalers from finetune_gate_analysis.py.
For each held-out fold, it:

1. runs the original fine-tuned VariantFusionHead to recover z_prot, z_dna, gate,
   and prediction,
2. selects BRCA1 SGE-associated SAE features using only that fold's training
   labels,
3. removes selected features from pretrained genome-wide SAE activations,
4. decodes the perturbed SAE activations back to the SAE input space,
5. maps the decoded SAE representation into the fine-tuned fold-specific z_dna
   space using a train-only ridge alignment, and
6. re-evaluates the learned gate/head with z_dna replaced before the gate.

This is the closest current approximation to a model-internal feature
intervention while still using the pretrained genome-wide SAE feature space.
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
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP, VariantFusionHead


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
    parser.add_argument("--esm", type=Path, default=repo_root / "results" / "variant" / "brca1_esm_delta.npz")
    parser.add_argument("--evo2", type=Path, default=repo_root / "results" / "variant" / "brca1_evo2.npz")
    parser.add_argument(
        "--sae-acts",
        type=Path,
        default=repo_root / "results" / "interpretability_applications" / "brca1_sge_sae_acts.npz",
    )
    parser.add_argument("--sae-checkpoint", type=Path, default=repo_root / "results" / "sae_genomewide" / "sae_model.pt")
    parser.add_argument("--clip-checkpoint", type=Path, default=repo_root / "results" / "pretrain" / "crossmodal_clip_40k.pt")
    parser.add_argument("--cards", type=Path, default=repo_root / "results" / "sae_genomewide" / "sae_genomewide_cards_deep.csv")
    parser.add_argument(
        "--fold-artifacts",
        type=Path,
        default=repo_root / "results" / "gate_analysis_artifacts" / "fold_artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=630)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], score[ok]))


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((x - mean) / scale).astype(np.float32)


def load_sae(checkpoint: Path, device: str) -> TopKSAE:
    state = torch.load(require_file(checkpoint), map_location="cpu", weights_only=True)
    d_hidden, d_in = state["encoder.weight"].shape
    model = TopKSAE(d_in=d_in, d_hidden=d_hidden, k=32)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def decode_acts(sae: TopKSAE, acts: np.ndarray, device: str) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(acts), 1024):
            h = torch.tensor(acts[start : start + 1024], dtype=torch.float32, device=device)
            out.append(sae.decoder(h).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def make_model(ckpt: dict, device: str) -> VariantFusionHead:
    cfg = ckpt["config"]
    clip = CrossModalCLIP(
        d_prot=ckpt["d_prot_actual"],
        d_dna=ckpt["d_dna_actual"],
        d_hidden=cfg["d_hidden"],
        d_shared=cfg["d_shared"],
        n_layers=cfg["n_layers"],
    )
    model = VariantFusionHead(clip, d_shared=cfg["d_shared"], freeze_encoders=False, scalar_dim=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


@torch.no_grad()
def model_forward(
    model: VariantFusionHead,
    xp: np.ndarray,
    xd: np.ndarray,
    mask: np.ndarray,
    scalar: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_pred = []
    out_zp = []
    out_zd = []
    out_gate = []
    for start in range(0, len(xp), 1024):
        tp = torch.tensor(xp[start : start + 1024], dtype=torch.float32, device=device)
        td = torch.tensor(xd[start : start + 1024], dtype=torch.float32, device=device)
        tm = torch.tensor(mask[start : start + 1024], dtype=torch.float32, device=device)
        ts = torch.tensor(scalar[start : start + 1024], dtype=torch.float32, device=device)
        logit, zp, zd, gate = model(tp, td, tm, ts, return_gate=True)
        out_pred.append(torch.sigmoid(logit).cpu().numpy())
        out_zp.append(zp.cpu().numpy())
        out_zd.append(zd.cpu().numpy())
        out_gate.append(gate.cpu().numpy())
    return (
        np.concatenate(out_pred).astype(np.float32),
        np.concatenate(out_zp).astype(np.float32),
        np.concatenate(out_zd).astype(np.float32),
        np.concatenate(out_gate).astype(np.float32),
    )


@torch.no_grad()
def predict_with_zd_override(
    model: VariantFusionHead,
    zp: np.ndarray,
    zd_override: np.ndarray,
    scalar: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    gates = []
    for start in range(0, len(zp), 1024):
        tzp = torch.tensor(zp[start : start + 1024], dtype=torch.float32, device=device)
        tzd = torch.tensor(zd_override[start : start + 1024], dtype=torch.float32, device=device)
        ts = torch.tensor(scalar[start : start + 1024], dtype=torch.float32, device=device)
        gate_in = torch.cat([tzp, tzd, ts], dim=-1)
        gate = F.softmax(model.gate(gate_in), dim=-1)
        fused = gate[:, 0:1] * tzp + gate[:, 1:2] * tzd
        fused = torch.cat([fused, ts], dim=-1)
        logit = model.head(fused).squeeze(-1)
        preds.append(torch.sigmoid(logit).cpu().numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(gates).astype(np.float32)


def select_features(acts: np.ndarray, y: np.ndarray, train_idx: np.ndarray, top_k: int, min_active: int) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    yy = y[train_idx].astype(int)
    abs_train = np.abs(acts[train_idx])
    active_counts = (abs_train > 1e-6).sum(axis=0)
    for feature in np.where(active_counts >= min_active)[0]:
        score = abs_train[:, feature]
        lof = score[yy == 1]
        func = score[yy == 0]
        if len(lof) < 5 or len(func) < 5:
            continue
        delta = float(lof.mean() - func.mean())
        if delta <= 0:
            continue
        pval = float(mannwhitneyu(lof, func, alternative="greater").pvalue)
        rows.append(
            {
                "feature": int(feature),
                "train_active": int(active_counts[feature]),
                "train_delta_abs_lof_minus_func": delta,
                "train_auroc_feature": safe_auc(yy, score),
                "train_p_lof_gt_func": pval,
            }
        )
    scored = pd.DataFrame(rows)
    if scored.empty:
        return np.array([], dtype=int), scored
    scored = scored.sort_values(["train_p_lof_gt_func", "train_auroc_feature"], ascending=[True, False])
    return scored["feature"].head(top_k).to_numpy(dtype=int), scored


def matched_random_sets(selected: np.ndarray, scored: pd.DataFrame, top_k: int, random_sets: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(selected) == 0 or scored.empty:
        return []
    selected_set = set(map(int, selected))
    counts = scored.set_index("feature")["train_active"].to_dict()
    pool_df = scored[~scored["feature"].isin(selected_set)].copy()
    pool = pool_df["feature"].to_numpy(dtype=int)
    sets = []
    for _ in range(random_sets):
        chosen: list[int] = []
        used: set[int] = set()
        for feat in selected:
            target = counts[int(feat)]
            tmp = pool_df.copy()
            tmp["dist"] = (tmp["train_active"] - target).abs()
            candidates = tmp.sort_values("dist")["feature"].head(max(50, top_k * 4)).to_numpy(dtype=int)
            candidates = np.array([c for c in candidates if c not in used], dtype=int)
            if len(candidates) == 0:
                candidates = np.array([c for c in pool if c not in used], dtype=int)
            if len(candidates) == 0:
                break
            pick = int(rng.choice(candidates))
            chosen.append(pick)
            used.add(pick)
        sets.append(np.array(chosen[:top_k], dtype=int))
    return sets


def annotate_region(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def assign(row: pd.Series) -> str:
        if str(row.get("vtype", "")).lower() != "coding":
            return "splice_or_noncoding"
        pos = row.get("aa_pos")
        if pd.isna(pos):
            return "coding_other"
        pos = int(pos)
        if 24 <= pos <= 65:
            return "RING_24_65"
        if 1364 <= pos <= 1437:
            return "coiled_coil_1364_1437"
        if 1642 <= pos <= 1736:
            return "BRCT1_1642_1736"
        if 1756 <= pos <= 1855:
            return "BRCT2_1756_1855"
        return "coding_other"

    df["brca1_region"] = df.apply(assign, axis=1)
    return df


def stratum_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = scores["label"].to_numpy(dtype=int)
    strata = {
        "all": np.ones(len(scores), dtype=bool),
        "missense": scores["is_missense"].astype(bool).to_numpy(),
        "coding": scores["vtype"].eq("coding").to_numpy(),
        "noncoding": scores["vtype"].eq("noncoding").to_numpy(),
    }
    for region in scores["brca1_region"].dropna().unique():
        strata[f"region:{region}"] = scores["brca1_region"].eq(region).to_numpy()
    for name, mask in strata.items():
        if mask.sum() < 40 or len(np.unique(y[mask])) < 2:
            continue
        row = {"stratum": name, "n": int(mask.sum()), "n_lof": int(y[mask].sum())}
        for col in ["original_pred", "sae_recon_pred", "top_feature_ablate_pred"]:
            row[f"auroc_{col}"] = safe_auc(y[mask], scores.loc[mask, col].to_numpy())
            row[f"auprc_{col}"] = safe_auprc(y[mask], scores.loc[mask, col].to_numpy())
        row["delta_auroc_recon_minus_top_ablate"] = row["auroc_sae_recon_pred"] - row["auroc_top_feature_ablate_pred"]
        row["delta_auroc_original_minus_top_ablate"] = row["auroc_original_pred"] - row["auroc_top_feature_ablate_pred"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    strata: pd.DataFrame,
    selected: pd.DataFrame,
    cards: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if df.empty:
            return "No rows."
        show = df.copy() if max_rows is None else df.head(max_rows).copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        return show.to_markdown(index=False)

    selected_show = selected.copy()
    if not selected_show.empty:
        selected_show = (
            selected_show.groupby("feature", as_index=False)
            .agg(
                n_folds_selected=("fold", "nunique"),
                mean_train_active=("train_active", "mean"),
                mean_train_delta=("train_delta_abs_lof_minus_func", "mean"),
                mean_train_auc=("train_auroc_feature", "mean"),
                median_train_p=("train_p_lof_gt_func", "median"),
            )
            .sort_values(["n_folds_selected", "median_train_p"], ascending=[False, True])
        )
        meta_cols = ["feature", "modality", "concept_v2", "path_rate", "path_enrich", "n_active"]
        selected_show = selected_show.merge(cards[meta_cols], on="feature", how="left")

    lines = [
        "# BRCA1 Fully Fine-Tuned Gate SAE Feature Intervention",
        "",
        "## Purpose",
        "",
        "Use saved per-fold fine-tuned VariantFusionHead checkpoints and scalers, replace fold-specific z_dna before the learned gate/head with SAE-decoded z_dna after removing SGE-associated features, and compare against active-count-matched random feature removal.",
        "",
        "The SAE feature space is from the pretrained genome-wide DNA shared representation. To intervene in each fine-tuned fold, decoded SAE representations are aligned to the fold-specific fine-tuned z_dna space by a train-only ridge map.",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Fold-Level Intervention",
        "",
        table(folds),
        "",
        "## Stratum Summary",
        "",
        table(strata, max_rows=40),
        "",
        "## Frequently Selected Features",
        "",
        table(selected_show, max_rows=30),
        "",
        "## Interpretation",
        "",
        "This is the closest current model-internal necessity test. A strong result would require the selected-feature intervention to reduce held-out performance more than matched random feature interventions. A small or non-significant drop means the evidence remains associative/localizing rather than causal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    df = annotate_region(pd.read_csv(require_file(args.variants)).reset_index(drop=True))
    y = df["label"].to_numpy(dtype=int)
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)[:, None]
    evo = np.load(require_file(args.evo2))
    ddelta = evo["edelta"].astype(np.float32)
    llr = evo["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)[:, None]
    sae_npz = np.load(require_file(args.sae_acts))
    acts = sae_npz["acts"].astype(np.float32)
    z_sae_scaled = sae_npz["z_dna_scaled"].astype(np.float32)
    if not (len(df) == len(pdelta) == len(ddelta) == len(acts)):
        raise ValueError("Input row counts do not match.")
    sae = load_sae(args.sae_checkpoint, device)
    cards = pd.read_csv(require_file(args.cards))

    original_pred = np.zeros(len(df), dtype=np.float32)
    recon_pred = np.zeros(len(df), dtype=np.float32)
    top_pred = np.zeros(len(df), dtype=np.float32)
    original_gate = np.zeros((len(df), 2), dtype=np.float32)
    recon_gate = np.zeros((len(df), 2), dtype=np.float32)
    top_gate = np.zeros((len(df), 2), dtype=np.float32)
    fold_rows = []
    random_rows = []
    selected_rows = []

    z_sae_recon_full = decode_acts(sae, acts, device)

    for fold in range(5):
        ckpt_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}.pt")
        scaler_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}_scalers.npz")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scalers = np.load(scaler_path)
        tr = scalers["train_idx"].astype(int)
        te = scalers["test_idx"].astype(int)
        model = make_model(ckpt, device)

        xp_all = standardize(pdelta, scalers["prot_mean"], scalers["prot_scale"])
        xd_all = standardize(ddelta, scalers["dna_mean"], scalers["dna_scale"])
        xs_all = standardize(llr, scalers["scalar_mean"], scalers["scalar_scale"])

        pred_full, zp_all, zd_all, gate_full = model_forward(model, xp_all, xd_all, pmask, xs_all, device)
        original_pred[te] = pred_full[te]
        original_gate[te] = gate_full[te]

        selected, scored = select_features(acts, y, tr, args.top_k, args.min_active)
        if len(selected) == 0:
            continue
        selected_df = scored[scored["feature"].isin(selected)].copy()
        selected_df["fold"] = fold
        selected_rows.append(selected_df)

        # Train-only alignment from pretrained SAE decoded space to the fold-specific
        # fine-tuned z_dna space.
        align = Ridge(alpha=args.ridge_alpha)
        align.fit(z_sae_recon_full[tr], zd_all[tr])

        z_recon_fold = align.predict(z_sae_recon_full).astype(np.float32)
        pred_recon_te, gate_recon_te = predict_with_zd_override(model, zp_all[te], z_recon_fold[te], xs_all[te], device)
        recon_pred[te] = pred_recon_te
        recon_gate[te] = gate_recon_te

        acts_top_te = acts[te].copy()
        acts_top_te[:, selected] = 0.0
        z_top_te = align.predict(decode_acts(sae, acts_top_te, device)).astype(np.float32)
        pred_top_te, gate_top_te = predict_with_zd_override(model, zp_all[te], z_top_te, xs_all[te], device)
        top_pred[te] = pred_top_te
        top_gate[te] = gate_top_te

        full_auc = safe_auc(y[te], pred_recon_te)
        top_auc = safe_auc(y[te], pred_top_te)
        full_ap = safe_auprc(y[te], pred_recon_te)
        top_ap = safe_auprc(y[te], pred_top_te)
        orig_auc = safe_auc(y[te], original_pred[te])
        fold_rows.append(
            {
                "fold": fold,
                "n_test": len(te),
                "n_selected_features": len(selected),
                "original_auc": orig_auc,
                "sae_recon_auc": full_auc,
                "top_feature_ablate_auc": top_auc,
                "delta_auc_recon_minus_top_ablate": full_auc - top_auc,
                "delta_auc_original_minus_top_ablate": orig_auc - top_auc,
                "sae_recon_auprc": full_ap,
                "top_feature_ablate_auprc": top_ap,
                "delta_auprc_recon_minus_top_ablate": full_ap - top_ap,
                "mean_original_gate_protein": float(original_gate[te, 0].mean()),
                "mean_recon_gate_protein": float(gate_recon_te[:, 0].mean()),
                "mean_top_ablate_gate_protein": float(gate_top_te[:, 0].mean()),
                "selected_features": ";".join(map(str, selected.tolist())),
            }
        )

        random_sets = matched_random_sets(selected, scored, args.top_k, args.random_sets, args.random_seed + fold)
        for i, feats in enumerate(random_sets):
            acts_rand_te = acts[te].copy()
            acts_rand_te[:, feats] = 0.0
            z_rand_te = align.predict(decode_acts(sae, acts_rand_te, device)).astype(np.float32)
            pred_rand_te, _ = predict_with_zd_override(model, zp_all[te], z_rand_te, xs_all[te], device)
            rand_auc = safe_auc(y[te], pred_rand_te)
            rand_ap = safe_auprc(y[te], pred_rand_te)
            random_rows.append(
                {
                    "fold": fold,
                    "random_set": i,
                    "random_ablate_auc": rand_auc,
                    "delta_auc_random_ablate": full_auc - rand_auc,
                    "random_ablate_auprc": rand_ap,
                    "delta_auprc_random_ablate": full_ap - rand_ap,
                }
            )

    folds = pd.DataFrame(fold_rows)
    randoms = pd.DataFrame(random_rows)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    rand = randoms["delta_auc_random_ablate"].dropna() if not randoms.empty else pd.Series(dtype=float)
    top_mean = float(folds["delta_auc_recon_minus_top_ablate"].mean()) if not folds.empty else float("nan")
    summary = pd.DataFrame(
        [
            {
                "n": len(df),
                "original_auroc": safe_auc(y, original_pred),
                "sae_recon_auroc": safe_auc(y, recon_pred),
                "top_feature_ablate_auroc": safe_auc(y, top_pred),
                "delta_auroc_recon_minus_top_ablate": safe_auc(y, recon_pred) - safe_auc(y, top_pred),
                "delta_auroc_original_minus_top_ablate": safe_auc(y, original_pred) - safe_auc(y, top_pred),
                "original_auprc": safe_auprc(y, original_pred),
                "sae_recon_auprc": safe_auprc(y, recon_pred),
                "top_feature_ablate_auprc": safe_auprc(y, top_pred),
                "delta_auprc_recon_minus_top_ablate": safe_auprc(y, recon_pred) - safe_auprc(y, top_pred),
                "mean_fold_delta_auc_recon_minus_top_ablate": top_mean,
                "mean_random_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
                "sd_random_delta_auc": float(rand.std()) if len(rand) else float("nan"),
                "empirical_p_random_delta_ge_top_mean": float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan"),
            }
        ]
    )
    scores = df.copy()
    scores["original_pred"] = original_pred
    scores["sae_recon_pred"] = recon_pred
    scores["top_feature_ablate_pred"] = top_pred
    scores["original_gate_protein"] = original_gate[:, 0]
    scores["sae_recon_gate_protein"] = recon_gate[:, 0]
    scores["top_feature_ablate_gate_protein"] = top_gate[:, 0]
    strata = stratum_summary(scores)

    prefix = "brca1_finetuned_gate_sae_intervention"
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    folds.to_csv(out / f"{prefix}_folds.csv", index=False)
    randoms.to_csv(out / f"{prefix}_random_ablation.csv", index=False)
    selected.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    strata.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    scores.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(out / f"{prefix}.md", summary, folds, strata, selected, cards)

    print(f"Wrote BRCA1 fine-tuned gate SAE intervention outputs to {out}")
    print(summary.to_string(index=False))
    print()
    print(folds.to_string(index=False))
    print()
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
