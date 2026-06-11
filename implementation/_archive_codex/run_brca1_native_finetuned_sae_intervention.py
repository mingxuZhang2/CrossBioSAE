#!/usr/bin/env python3
"""
Fold-native fine-tuned shared-representation SAE intervention.

This is a stricter follow-up to run_brca1_finetuned_gate_sae_intervention.py.
Instead of using the pretrained genome-wide SAE and mapping it into each
fine-tuned fold with ridge regression, this script trains a small TopK SAE
directly on each fold's fine-tuned z_dna representation. It then replaces z_dna
inside the same learned gate/head with:

1. native SAE reconstruction of z_dna, and
2. native SAE reconstruction after removing training-fold-selected
   SGE-associated SAE features.

The fold-native features are local to each fold and are not meant to be reused
as named biological feature cards. The purpose is a cleaner model-internal
necessity test with no cross-space ridge alignment.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP, VariantFusionHead

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_brca1_finetuned_gate_sae_intervention import (  # noqa: E402
    annotate_region,
    matched_random_sets,
    model_forward,
    require_file,
    safe_auc,
    safe_auprc,
    standardize,
)


class NativeTopKSAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        self.k = min(k, d_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))
        if self.k >= h.shape[-1]:
            return h
        _, idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, idx, 1.0)
        return h * mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decoder(acts), acts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--variants", type=Path, default=repo_root / "data" / "variant" / "brca1" / "brca1_variants.csv")
    parser.add_argument("--esm", type=Path, default=repo_root / "results" / "variant" / "brca1_esm_delta.npz")
    parser.add_argument("--evo2", type=Path, default=repo_root / "results" / "variant" / "brca1_evo2.npz")
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
    parser.add_argument("--top-k-features", type=int, default=32)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--permutation-sets", type=int, default=20)
    parser.add_argument("--dose-k-values", default="8,16,32,64")
    parser.add_argument("--sae-hidden", type=int, default=1024)
    parser.add_argument("--sae-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1", type=float, default=1e-5)
    parser.add_argument("--random-seed", type=int, default=630)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default="BRCA1")
    parser.add_argument("--output-prefix", default="brca1_native_finetuned_sae_intervention")
    parser.add_argument(
        "--target-representation",
        choices=["z_dna", "z_prot"],
        default="z_dna",
        help="Shared-space branch to reconstruct and ablate with a fold-native SAE.",
    )
    parser.add_argument(
        "--domain-col",
        default="brca1_region",
        help="optional variant-table column used for domain/region strata",
    )
    parser.add_argument(
        "--analysis-filter-col",
        default=None,
        help="optional variant-table column used to restrict SAE training, feature selection, and primary evaluation",
    )
    parser.add_argument(
        "--analysis-filter-value",
        default=None,
        help="required value for --analysis-filter-col; booleans accept true/false",
    )
    parser.add_argument(
        "--no-normalize-recon",
        action="store_true",
        help="Do not L2-normalize SAE decoded vectors before sending them to the gate/head.",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return sorted(set(v for v in vals if v > 0))


def analysis_mask_from_args(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if not args.analysis_filter_col:
        return np.ones(len(df), dtype=bool)
    if args.analysis_filter_col not in df.columns:
        raise KeyError(f"--analysis-filter-col {args.analysis_filter_col!r} is not in variants table")
    if args.analysis_filter_value is None:
        raise ValueError("--analysis-filter-value is required when --analysis-filter-col is set")
    raw_value = str(args.analysis_filter_value).strip()
    col = df[args.analysis_filter_col]
    if raw_value.lower() in {"true", "false"}:
        target = raw_value.lower() == "true"
        mask = col.astype(str).str.lower().isin({str(target).lower(), "1" if target else "0"})
        if col.dtype == bool:
            mask = col.eq(target)
    else:
        numeric_col = pd.to_numeric(col, errors="coerce")
        try:
            numeric_value = float(raw_value)
        except ValueError:
            numeric_value = None
        if numeric_value is not None and numeric_col.notna().any():
            mask = numeric_col.eq(numeric_value)
        else:
            mask = col.astype(str).eq(raw_value)
    if int(mask.sum()) == 0:
        raise ValueError(
            f"analysis filter {args.analysis_filter_col}={args.analysis_filter_value} selected zero rows"
        )
    return mask.to_numpy(dtype=bool)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def train_native_sae(
    z: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> tuple[NativeTopKSAE, dict[str, float]]:
    set_seed(seed)
    model = NativeTopKSAE(d_in=z.shape[1], d_hidden=args.sae_hidden, k=args.sae_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = torch.tensor(z[train_idx], dtype=torch.float32, device=device)
    val = torch.tensor(z[val_idx], dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), args.batch_size):
            batch = train[torch.tensor(order[start : start + args.batch_size], device=device)]
            recon, acts = model(batch)
            loss = F.mse_loss(recon, batch) + args.l1 * acts.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            recon_val, acts_val = model(val)
            val_loss = F.mse_loss(recon_val, val).item()
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        recon_train, acts_train = model(train)
        recon_val, acts_val = model(val)
        metrics = {
            "sae_best_epoch": float(best_epoch),
            "sae_train_mse": float(F.mse_loss(recon_train, train).item()),
            "sae_val_mse": float(F.mse_loss(recon_val, val).item()),
            "sae_train_mean_l0": float((acts_train > 1e-8).sum(dim=1).float().mean().item()),
            "sae_val_mean_l0": float((acts_val > 1e-8).sum(dim=1).float().mean().item()),
            "sae_train_dead_frac": float(((acts_train > 1e-8).sum(dim=0) == 0).float().mean().item()),
            "sae_val_dead_frac": float(((acts_val > 1e-8).sum(dim=0) == 0).float().mean().item()),
        }
    return model, metrics


@torch.no_grad()
def encode_all(model: NativeTopKSAE, z: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    rows = []
    for start in range(0, len(z), batch_size):
        x = torch.tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
        rows.append(model.encode(x).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


@torch.no_grad()
def decode_acts(model: NativeTopKSAE, acts: np.ndarray, device: str, normalize: bool) -> np.ndarray:
    rows = []
    for start in range(0, len(acts), 1024):
        h = torch.tensor(acts[start : start + 1024], dtype=torch.float32, device=device)
        rows.append(model.decoder(h).cpu().numpy())
    out = np.concatenate(rows, axis=0).astype(np.float32)
    return normalize_rows(out) if normalize else out


def score_native_features(acts: np.ndarray, y: np.ndarray, train_idx: np.ndarray, min_active: int) -> pd.DataFrame:
    rows = []
    yy = y[train_idx].astype(int)
    abs_train = np.abs(acts[train_idx])
    active_counts = (abs_train > 1e-8).sum(axis=0)
    for feature in np.where(active_counts >= min_active)[0]:
        score = abs_train[:, feature]
        lof = score[yy == 1]
        func = score[yy == 0]
        if len(lof) < 5 or len(func) < 5:
            continue
        delta = float(lof.mean() - func.mean())
        rows.append(
            {
                "feature": int(feature),
                "train_active": int(active_counts[feature]),
                "train_delta_abs_lof_minus_func": delta,
                "train_abs_delta": abs(delta),
                "train_auroc_feature": safe_auc(yy, score),
                "train_p_lof_gt_func": float(mannwhitneyu(lof, func, alternative="greater").pvalue),
            }
        )
    return pd.DataFrame(rows)


def top_features(scored: pd.DataFrame, top_k: int) -> np.ndarray:
    if scored.empty:
        return np.array([], dtype=int)
    use = scored[scored["train_delta_abs_lof_minus_func"] > 0].copy()
    if use.empty:
        return np.array([], dtype=int)
    use = use.sort_values(["train_p_lof_gt_func", "train_auroc_feature"], ascending=[True, False])
    return use["feature"].head(top_k).to_numpy(dtype=int)


def matched_bottom_features(selected: np.ndarray, scored: pd.DataFrame, seed: int) -> np.ndarray:
    if len(selected) == 0 or scored.empty:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    selected_set = set(map(int, selected))
    counts = scored.set_index("feature")["train_active"].to_dict()
    pool = scored[~scored["feature"].isin(selected_set)].copy()
    if pool.empty:
        return np.array([], dtype=int)
    chosen = []
    used = set()
    for feat in selected:
        target = counts[int(feat)]
        tmp = pool[~pool["feature"].isin(used)].copy()
        if tmp.empty:
            break
        tmp["count_dist"] = (tmp["train_active"] - target).abs()
        candidates = tmp.sort_values(["train_abs_delta", "count_dist", "train_p_lof_gt_func"], ascending=[True, True, False])
        candidates = candidates.head(max(50, len(selected) * 4))["feature"].to_numpy(dtype=int)
        pick = int(rng.choice(candidates))
        chosen.append(pick)
        used.add(pick)
    return np.array(chosen, dtype=int)


def permuted_label_features(
    acts: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    min_active: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    y_perm[train_idx] = rng.permutation(y_perm[train_idx])
    scored = score_native_features(acts, y_perm, train_idx, min_active)
    return top_features(scored, top_k)


@torch.no_grad()
def predict_with_target_override(
    model: VariantFusionHead,
    target: str,
    fixed_other: np.ndarray,
    z_override: np.ndarray,
    scalar: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    gates = []
    for start in range(0, len(z_override), 1024):
        zt = torch.tensor(z_override[start : start + 1024], dtype=torch.float32, device=device)
        zo = torch.tensor(fixed_other[start : start + 1024], dtype=torch.float32, device=device)
        ts = torch.tensor(scalar[start : start + 1024], dtype=torch.float32, device=device)
        if target == "z_dna":
            tzp, tzd = zo, zt
        elif target == "z_prot":
            tzp, tzd = zt, zo
        else:  # pragma: no cover - argparse constrains this.
            raise ValueError(f"unknown target representation: {target}")
        gate_in = torch.cat([tzp, tzd, ts], dim=-1)
        gate = F.softmax(model.gate(gate_in), dim=-1)
        fused = gate[:, 0:1] * tzp + gate[:, 1:2] * tzd
        fused = torch.cat([fused, ts], dim=-1)
        logit = model.head(fused).squeeze(-1)
        preds.append(torch.sigmoid(logit).cpu().numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(gates).astype(np.float32)


def evaluate_ablation(
    sae: NativeTopKSAE,
    acts_all: np.ndarray,
    features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    model: VariantFusionHead,
    target: str,
    fixed_other_te: np.ndarray,
    xs_te: np.ndarray,
    recon_auc: float,
    recon_auprc: float,
    device: str,
    normalize_recon: bool,
) -> tuple[float, float, float, float]:
    if len(features) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    acts_tmp = acts_all[te].copy()
    acts_tmp[:, features] = 0.0
    z_tmp = decode_acts(sae, acts_tmp, device, normalize_recon)
    pred_tmp, _ = predict_with_target_override(model, target, fixed_other_te, z_tmp, xs_te, device)
    auc = safe_auc(y[te], pred_tmp)
    ap = safe_auprc(y[te], pred_tmp)
    return auc, ap, recon_auc - auc, recon_auprc - ap


def evaluate_feature_only(
    sae: NativeTopKSAE,
    acts_all: np.ndarray,
    features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    model: VariantFusionHead,
    target: str,
    fixed_other_te: np.ndarray,
    xs_te: np.ndarray,
    device: str,
    normalize_recon: bool,
) -> tuple[np.ndarray, float, float]:
    acts_tmp = np.zeros_like(acts_all[te])
    if len(features):
        acts_tmp[:, features] = acts_all[te][:, features]
    z_tmp = decode_acts(sae, acts_tmp, device, normalize_recon)
    pred_tmp, _ = predict_with_target_override(model, target, fixed_other_te, z_tmp, xs_te, device)
    return pred_tmp, safe_auc(y[te], pred_tmp), safe_auprc(y[te], pred_tmp)


def evaluate_common_background_rescue(
    sae: NativeTopKSAE,
    acts_all: np.ndarray,
    selected: np.ndarray,
    random_features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    model: VariantFusionHead,
    target: str,
    fixed_other_te: np.ndarray,
    xs_te: np.ndarray,
    device: str,
    normalize_recon: bool,
) -> dict[str, float | int | str]:
    selected = np.asarray(selected, dtype=int)
    random_features = np.asarray(random_features, dtype=int)
    pool = np.array(sorted(set(selected.tolist()) | set(random_features.tolist())), dtype=int)
    if len(pool) == 0:
        return {}

    acts_base = acts_all[te].copy()
    acts_base[:, pool] = 0.0
    z_base = decode_acts(sae, acts_base, device, normalize_recon)
    pred_base, _ = predict_with_target_override(model, target, fixed_other_te, z_base, xs_te, device)

    acts_selected_rescue = acts_base.copy()
    if len(selected):
        acts_selected_rescue[:, selected] = acts_all[te][:, selected]
    z_selected = decode_acts(sae, acts_selected_rescue, device, normalize_recon)
    pred_selected, _ = predict_with_target_override(model, target, fixed_other_te, z_selected, xs_te, device)

    acts_random_rescue = acts_base.copy()
    if len(random_features):
        acts_random_rescue[:, random_features] = acts_all[te][:, random_features]
    z_random = decode_acts(sae, acts_random_rescue, device, normalize_recon)
    pred_random, _ = predict_with_target_override(model, target, fixed_other_te, z_random, xs_te, device)

    base_auc = safe_auc(y[te], pred_base)
    selected_auc = safe_auc(y[te], pred_selected)
    random_auc = safe_auc(y[te], pred_random)
    base_ap = safe_auprc(y[te], pred_base)
    selected_ap = safe_auprc(y[te], pred_selected)
    random_ap = safe_auprc(y[te], pred_random)

    return {
        "n_pool_features": int(len(pool)),
        "n_selected_features": int(len(selected)),
        "n_random_features": int(len(random_features)),
        "base_auc": base_auc,
        "selected_rescue_auc": selected_auc,
        "random_rescue_auc": random_auc,
        "selected_rescue_gain_auc": selected_auc - base_auc,
        "random_rescue_gain_auc": random_auc - base_auc,
        "delta_auc_selected_rescue_minus_random_rescue": selected_auc - random_auc,
        "base_auprc": base_ap,
        "selected_rescue_auprc": selected_ap,
        "random_rescue_auprc": random_ap,
        "selected_rescue_gain_auprc": selected_ap - base_ap,
        "random_rescue_gain_auprc": random_ap - base_ap,
        "delta_auprc_selected_rescue_minus_random_rescue": selected_ap - random_ap,
        "pool_features": ";".join(map(str, pool.tolist())),
    }


def stratum_summary_generic(scores: pd.DataFrame, domain_col: str | None) -> pd.DataFrame:
    rows = []
    y = scores["label"].to_numpy(dtype=int)
    strata: dict[str, np.ndarray] = {"all": np.ones(len(scores), dtype=bool)}
    if "is_missense" in scores.columns:
        strata["missense"] = scores["is_missense"].astype(bool).to_numpy()
    if "vtype" in scores.columns:
        strata["coding"] = scores["vtype"].eq("coding").to_numpy()
        strata["noncoding"] = scores["vtype"].eq("noncoding").to_numpy()
    if domain_col and domain_col in scores.columns:
        for region in scores[domain_col].dropna().unique():
            strata[f"{domain_col}:{region}"] = scores[domain_col].eq(region).to_numpy()

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
    run_name: str,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    strata: pd.DataFrame,
    selected: pd.DataFrame,
    controls: pd.DataFrame,
    dose_response: pd.DataFrame,
    sufficiency: pd.DataFrame | None = None,
    rescue: pd.DataFrame | None = None,
    target_representation: str = "z_dna",
) -> None:
    def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if df.empty:
            return "No rows."
        show = df.copy() if max_rows is None else df.head(max_rows).copy()
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

    lines = [
        f"# {run_name} Fold-Native Fine-Tuned SAE Intervention",
        "",
        "## Purpose",
        "",
        f"Train one TopK SAE per fine-tuned {run_name} fold directly on that fold's selected shared representation, then replace that representation inside the same learned gate/head with native SAE reconstructions before and after removing training-fold-selected SGE-associated SAE features.",
        "",
        "This removes the ridge-alignment approximation required by the pretrained genome-wide SAE intervention. The fold-native feature IDs are local to each fold and should not be interpreted as stable global biological feature cards.",
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
        "## Selected Fold-Native Features",
        "",
        table(selected, max_rows=80),
        "",
        "## Control Ablations",
        "",
        table(controls, max_rows=80),
        "",
        "## Dose Response",
        "",
        table(dose_response, max_rows=80),
        "",
        "## Feature-Only Sufficiency Controls",
        "",
        table(sufficiency if sufficiency is not None else pd.DataFrame(), max_rows=80),
        "",
        "## Common-Background Addback Rescue Controls",
        "",
        table(rescue if rescue is not None else pd.DataFrame(), max_rows=80),
        "",
        "## Interpretation",
        "",
        f"This is a cleaner model-internal necessity test than the ridge-aligned pretrained-SAE intervention because the SAE is trained directly in the fold-specific fine-tuned `{target_representation}` space. A strong result requires native SAE reconstruction to preserve the original fine-tuned predictor, while removing training-fold-selected SGE-associated native SAE features causes a larger drop than active-count-matched random feature ablations.",
        "",
        "The additional controls make the necessity claim more specific: bottom/non-associated feature ablations and label-permuted feature selection should be near the null, while top-k dose-response should increase as more SGE-associated sparse directions are removed. Those checks test whether the effect is tied to the SGE-selected features rather than generic sparse-feature deletion.",
        "",
        "Feature-only controls are a sufficiency sanity check. They decode only the selected native features, or matched control features, before the same gate/head. This does not prove full mechanistic rescue because fold-native feature IDs are local and the decoder bias/protein/scalar inputs remain present. Selected-only performance above matched controls would support that the selected sparse directions carry positive task signal rather than merely acting as deletion-sensitive coordinates; failure to beat matched controls bounds the claim to necessity within the full representation.",
        "",
        "The common-background addback rescue is a paired specificity check. For each matched random set, it first removes both the selected features and the matched random features, then compares adding back the selected set versus adding back the random set into the same low-background reconstruction. A positive selected-minus-random rescue gap is the reverse-direction counterpart to the deletion test; it supports that the selected directions carry more recoverable task signal than activity-matched alternatives, but it remains a local rescue within a fold-specific SAE basis.",
        "",
        f"If the intervention is positive, it supports a real local necessity signal inside the fine-tuned {run_name} gate/head: selected native sparse directions are not just associated with SGE LOF labels, they are used by the learned predictor. The claim should be stated by stratum rather than as a universal gene-wide mechanism.",
        "",
        "The result should still be interpreted as a local necessity test, not a reusable mechanism atlas: every fold has its own SAE basis, and selected feature IDs are not comparable across folds. Biological interpretation still needs the pretrained genome-wide SAE cards, domain localization, and external functional or clinical validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    normalize_recon = not args.no_normalize_recon
    dose_k_values = parse_int_list(args.dose_k_values)

    df = pd.read_csv(require_file(args.variants)).reset_index(drop=True)
    if args.domain_col == "brca1_region" and args.domain_col not in df.columns:
        df = annotate_region(df)
    y = df["label"].to_numpy(dtype=int)
    analysis_mask = analysis_mask_from_args(df, args)
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)[:, None]
    evo = np.load(require_file(args.evo2))
    ddelta = evo["edelta"].astype(np.float32)
    llr = evo["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)[:, None]

    original_pred = np.full(len(df), np.nan, dtype=np.float32)
    recon_pred = np.full(len(df), np.nan, dtype=np.float32)
    top_pred = np.full(len(df), np.nan, dtype=np.float32)
    selected_only_pred = np.full(len(df), np.nan, dtype=np.float32)
    bias_only_pred = np.full(len(df), np.nan, dtype=np.float32)
    original_gate = np.full((len(df), 2), np.nan, dtype=np.float32)
    recon_gate = np.full((len(df), 2), np.nan, dtype=np.float32)
    top_gate = np.full((len(df), 2), np.nan, dtype=np.float32)
    fold_rows = []
    random_rows = []
    control_rows = []
    dose_rows = []
    sufficiency_rows = []
    rescue_rows = []
    selected_rows = []

    for fold in range(5):
        ckpt_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}.pt")
        scaler_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}_scalers.npz")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scalers = np.load(scaler_path)
        tr = scalers["train_idx"].astype(int)
        va = scalers["val_idx"].astype(int)
        te = scalers["test_idx"].astype(int)
        tr_sae = tr[analysis_mask[tr]]
        va_sae = va[analysis_mask[va]]
        te_eval = te[analysis_mask[te]]
        skip_reasons = []
        if len(tr_sae) < 40:
            skip_reasons.append("analysis_train_lt_40")
        if len(va_sae) < 20:
            skip_reasons.append("analysis_val_lt_20")
        if len(te_eval) < 20:
            skip_reasons.append("analysis_test_lt_20")
        if len(tr_sae) and len(np.unique(y[tr_sae])) < 2:
            skip_reasons.append("analysis_train_single_label")
        if len(te_eval) and len(np.unique(y[te_eval])) < 2:
            skip_reasons.append("analysis_test_single_label")
        if skip_reasons:
            fold_rows.append(
                {
                    "fold": fold,
                    "n_train": len(tr_sae),
                    "n_val": len(va_sae),
                    "n_test": len(te_eval),
                    "skip_reason": ";".join(skip_reasons),
                }
            )
            continue
        model = make_model(ckpt, device)

        xp_all = standardize(pdelta, scalers["prot_mean"], scalers["prot_scale"])
        xd_all = standardize(ddelta, scalers["dna_mean"], scalers["dna_scale"])
        xs_all = standardize(llr, scalers["scalar_mean"], scalers["scalar_scale"])

        pred_full, zp_all, zd_all, gate_full = model_forward(model, xp_all, xd_all, pmask, xs_all, device)
        original_pred[te_eval] = pred_full[te_eval]
        original_gate[te_eval] = gate_full[te_eval]
        if args.target_representation == "z_dna":
            z_target_all = zd_all
            z_fixed_other_all = zp_all
        else:
            z_target_all = zp_all
            z_fixed_other_all = zd_all

        sae, sae_metrics = train_native_sae(z_target_all, tr_sae, va_sae, args, device, args.random_seed + fold)
        acts_all = encode_all(sae, z_target_all, device)
        scored = score_native_features(acts_all, y, tr_sae, args.min_active)
        selected = top_features(scored, args.top_k_features)
        if len(selected) == 0:
            fold_rows.append(
                {
                    "fold": fold,
                    "n_train": len(tr_sae),
                    "n_val": len(va_sae),
                    "n_test": len(te_eval),
                    "skip_reason": "no_positive_sge_associated_features",
                    **sae_metrics,
                }
            )
            continue
        selected_df = scored[scored["feature"].isin(selected)].copy()
        selected_df["fold"] = fold
        selected_df["note"] = "fold_native_feature_id_not_global"
        selected_rows.append(selected_df)

        z_recon_te = decode_acts(sae, acts_all[te_eval], device, normalize_recon)
        pred_recon_te, gate_recon_te = predict_with_target_override(
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            z_recon_te,
            xs_all[te_eval],
            device,
        )
        recon_pred[te_eval] = pred_recon_te
        recon_gate[te_eval] = gate_recon_te

        acts_top_te = acts_all[te_eval].copy()
        acts_top_te[:, selected] = 0.0
        z_top_te = decode_acts(sae, acts_top_te, device, normalize_recon)
        pred_top_te, gate_top_te = predict_with_target_override(
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            z_top_te,
            xs_all[te_eval],
            device,
        )
        top_pred[te_eval] = pred_top_te
        top_gate[te_eval] = gate_top_te

        pred_selected_only_te, selected_only_auc, selected_only_ap = evaluate_feature_only(
            sae,
            acts_all,
            selected,
            te_eval,
            y,
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            xs_all[te_eval],
            device,
            normalize_recon,
        )
        pred_bias_only_te, bias_only_auc, bias_only_ap = evaluate_feature_only(
            sae,
            acts_all,
            np.array([], dtype=int),
            te_eval,
            y,
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            xs_all[te_eval],
            device,
            normalize_recon,
        )
        selected_only_pred[te_eval] = pred_selected_only_te
        bias_only_pred[te_eval] = pred_bias_only_te

        full_auc = safe_auc(y[te_eval], pred_recon_te)
        top_auc = safe_auc(y[te_eval], pred_top_te)
        full_ap = safe_auprc(y[te_eval], pred_recon_te)
        top_ap = safe_auprc(y[te_eval], pred_top_te)
        orig_auc = safe_auc(y[te_eval], original_pred[te_eval])
        orig_ap = safe_auprc(y[te_eval], original_pred[te_eval])
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(tr_sae),
                "n_val": len(va_sae),
                "n_test": len(te_eval),
                "skip_reason": "",
                "n_selected_features": len(selected),
                "original_auc": orig_auc,
                "native_sae_recon_auc": full_auc,
                "top_feature_ablate_auc": top_auc,
                "delta_auc_recon_minus_top_ablate": full_auc - top_auc,
                "delta_auc_original_minus_top_ablate": orig_auc - top_auc,
                "original_auprc": orig_ap,
                "native_sae_recon_auprc": full_ap,
                "top_feature_ablate_auprc": top_ap,
                "delta_auprc_recon_minus_top_ablate": full_ap - top_ap,
                "selected_feature_only_auc": selected_only_auc,
                "bias_only_auc": bias_only_auc,
                "delta_auc_selected_only_minus_bias_only": selected_only_auc - bias_only_auc,
                "selected_feature_only_auprc": selected_only_ap,
                "bias_only_auprc": bias_only_ap,
                "delta_auprc_selected_only_minus_bias_only": selected_only_ap - bias_only_ap,
                "mean_original_gate_protein": float(original_gate[te_eval, 0].mean()),
                "mean_recon_gate_protein": float(gate_recon_te[:, 0].mean()),
                "mean_top_ablate_gate_protein": float(gate_top_te[:, 0].mean()),
                "selected_features": ";".join(map(str, selected.tolist())),
                **sae_metrics,
            }
        )

        for dose_k in dose_k_values:
            feats = top_features(scored, dose_k)
            if len(feats) == 0:
                continue
            auc, ap, delta_auc, delta_ap = evaluate_ablation(
                sae,
                acts_all,
                feats,
                te_eval,
                y,
                model,
                args.target_representation,
                z_fixed_other_all[te_eval],
                xs_all[te_eval],
                full_auc,
                full_ap,
                device,
                normalize_recon,
            )
            dose_rows.append(
                {
                    "fold": fold,
                    "dose_k": int(len(feats)),
                    "requested_k": int(dose_k),
                    "ablate_auc": auc,
                    "delta_auc_recon_minus_ablate": delta_auc,
                    "ablate_auprc": ap,
                    "delta_auprc_recon_minus_ablate": delta_ap,
                    "features": ";".join(map(str, feats.tolist())),
                }
            )

        bottom = matched_bottom_features(selected, scored, args.random_seed + 10_000 + fold)
        _, bottom_only_auc, bottom_only_ap = evaluate_feature_only(
            sae,
            acts_all,
            bottom,
            te_eval,
            y,
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            xs_all[te_eval],
            device,
            normalize_recon,
        )
        sufficiency_rows.append(
            {
                "fold": fold,
                "control": "bottom_abs_delta_active_matched_only",
                "set_id": 0,
                "n_features": len(bottom),
                "feature_only_auc": bottom_only_auc,
                "feature_only_auprc": bottom_only_ap,
                "delta_auc_selected_only_minus_control": selected_only_auc - bottom_only_auc,
                "delta_auprc_selected_only_minus_control": selected_only_ap - bottom_only_ap,
                "features": ";".join(map(str, bottom.tolist())),
            }
        )
        auc, ap, delta_auc, delta_ap = evaluate_ablation(
            sae,
            acts_all,
            bottom,
            te_eval,
            y,
            model,
            args.target_representation,
            z_fixed_other_all[te_eval],
            xs_all[te_eval],
            full_auc,
            full_ap,
            device,
            normalize_recon,
        )
        control_rows.append(
            {
                "fold": fold,
                "control": "bottom_abs_delta_active_matched",
                "set_id": 0,
                "n_features": len(bottom),
                "ablate_auc": auc,
                "delta_auc_recon_minus_ablate": delta_auc,
                "ablate_auprc": ap,
                "delta_auprc_recon_minus_ablate": delta_ap,
                "features": ";".join(map(str, bottom.tolist())),
            }
        )

        for perm_i in range(args.permutation_sets):
            feats = permuted_label_features(
                acts_all,
                y,
                tr_sae,
                args.top_k_features,
                args.min_active,
                args.random_seed + 20_000 + 100 * fold + perm_i,
            )
            auc, ap, delta_auc, delta_ap = evaluate_ablation(
                sae,
                acts_all,
                feats,
                te_eval,
                y,
                model,
                args.target_representation,
                z_fixed_other_all[te_eval],
                xs_all[te_eval],
                full_auc,
                full_ap,
                device,
                normalize_recon,
            )
            control_rows.append(
                {
                    "fold": fold,
                    "control": "label_permuted_selection",
                    "set_id": perm_i,
                    "n_features": len(feats),
                    "ablate_auc": auc,
                    "delta_auc_recon_minus_ablate": delta_auc,
                    "ablate_auprc": ap,
                    "delta_auprc_recon_minus_ablate": delta_ap,
                    "features": ";".join(map(str, feats.tolist())),
                }
            )

        random_sets = matched_random_sets(selected, scored, args.top_k_features, args.random_sets, args.random_seed + fold)
        for i, feats in enumerate(random_sets):
            _, rand_only_auc, rand_only_ap = evaluate_feature_only(
                sae,
                acts_all,
                feats,
                te_eval,
                y,
                model,
                args.target_representation,
                z_fixed_other_all[te_eval],
                xs_all[te_eval],
                device,
                normalize_recon,
            )
            sufficiency_rows.append(
                {
                    "fold": fold,
                    "control": "random_active_matched_only",
                    "set_id": i,
                    "n_features": len(feats),
                    "feature_only_auc": rand_only_auc,
                    "feature_only_auprc": rand_only_ap,
                    "delta_auc_selected_only_minus_control": selected_only_auc - rand_only_auc,
                    "delta_auprc_selected_only_minus_control": selected_only_ap - rand_only_ap,
                    "features": ";".join(map(str, feats.tolist())),
                }
            )
            acts_rand_te = acts_all[te_eval].copy()
            acts_rand_te[:, feats] = 0.0
            z_rand_te = decode_acts(sae, acts_rand_te, device, normalize_recon)
            pred_rand_te, _ = predict_with_target_override(
                model,
                args.target_representation,
                z_fixed_other_all[te_eval],
                z_rand_te,
                xs_all[te_eval],
                device,
            )
            rand_auc = safe_auc(y[te_eval], pred_rand_te)
            rand_ap = safe_auprc(y[te_eval], pred_rand_te)
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
            rescue = evaluate_common_background_rescue(
                sae,
                acts_all,
                selected,
                feats,
                te_eval,
                y,
                model,
                args.target_representation,
                z_fixed_other_all[te_eval],
                xs_all[te_eval],
                device,
                normalize_recon,
            )
            if rescue:
                rescue_rows.append(
                    {
                        "fold": fold,
                        "random_set": i,
                        **rescue,
                        "selected_features": ";".join(map(str, selected.tolist())),
                        "random_features": ";".join(map(str, feats.tolist())),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    randoms = pd.DataFrame(random_rows)
    controls = pd.DataFrame(control_rows)
    dose_response = pd.DataFrame(dose_rows)
    sufficiency = pd.DataFrame(sufficiency_rows)
    rescue = pd.DataFrame(rescue_rows)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    def numeric_col(frame: pd.DataFrame, col: str) -> pd.Series:
        if frame.empty or col not in frame.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[col], errors="coerce").dropna()

    rand = randoms["delta_auc_random_ablate"].dropna() if not randoms.empty else pd.Series(dtype=float)
    perm = (
        controls.loc[controls["control"].eq("label_permuted_selection"), "delta_auc_recon_minus_ablate"].dropna()
        if not controls.empty
        else pd.Series(dtype=float)
    )
    bottom = (
        controls.loc[controls["control"].eq("bottom_abs_delta_active_matched"), "delta_auc_recon_minus_ablate"].dropna()
        if not controls.empty
        else pd.Series(dtype=float)
    )
    fold_top_delta = numeric_col(folds, "delta_auc_recon_minus_top_ablate")
    fold_selected_only_auc = numeric_col(folds, "selected_feature_only_auc")
    top_mean = float(fold_top_delta.mean()) if len(fold_top_delta) else float("nan")
    selected_only_mean = float(fold_selected_only_auc.mean()) if len(fold_selected_only_auc) else float("nan")
    random_only = (
        sufficiency.loc[sufficiency["control"].eq("random_active_matched_only"), "feature_only_auc"].dropna()
        if not sufficiency.empty
        else pd.Series(dtype=float)
    )
    bottom_only = (
        sufficiency.loc[sufficiency["control"].eq("bottom_abs_delta_active_matched_only"), "feature_only_auc"].dropna()
        if not sufficiency.empty
        else pd.Series(dtype=float)
    )
    rescue_delta_auc = (
        rescue["delta_auc_selected_rescue_minus_random_rescue"].dropna()
        if not rescue.empty
        else pd.Series(dtype=float)
    )
    rescue_delta_ap = (
        rescue["delta_auprc_selected_rescue_minus_random_rescue"].dropna()
        if not rescue.empty
        else pd.Series(dtype=float)
    )
    score_mask = analysis_mask & np.isfinite(original_pred)
    metric_y = y[score_mask]
    def auc_on_analysis(pred: np.ndarray) -> float:
        return safe_auc(metric_y, pred[score_mask])

    def auprc_on_analysis(pred: np.ndarray) -> float:
        return safe_auprc(metric_y, pred[score_mask])

    summary_row = {
        "n": int(score_mask.sum()),
        "n_analysis": int(analysis_mask.sum()),
        "n_total": len(df),
        "analysis_filter_col": args.analysis_filter_col or "",
        "analysis_filter_value": args.analysis_filter_value or "",
        "original_auroc": auc_on_analysis(original_pred),
        "native_sae_recon_auroc": auc_on_analysis(recon_pred),
        "top_feature_ablate_auroc": auc_on_analysis(top_pred),
        "delta_auroc_recon_minus_top_ablate": auc_on_analysis(recon_pred) - auc_on_analysis(top_pred),
        "delta_auroc_original_minus_top_ablate": auc_on_analysis(original_pred) - auc_on_analysis(top_pred),
        "original_auprc": auprc_on_analysis(original_pred),
        "native_sae_recon_auprc": auprc_on_analysis(recon_pred),
        "top_feature_ablate_auprc": auprc_on_analysis(top_pred),
        "delta_auprc_recon_minus_top_ablate": auprc_on_analysis(recon_pred) - auprc_on_analysis(top_pred),
        "mean_fold_delta_auc_recon_minus_top_ablate": top_mean,
        "mean_random_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
        "sd_random_delta_auc": float(rand.std()) if len(rand) else float("nan"),
        "empirical_p_random_delta_ge_top_mean": float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan"),
        "mean_bottom_delta_auc": float(bottom.mean()) if len(bottom) else float("nan"),
        "empirical_p_bottom_delta_ge_top_mean": float(((bottom >= top_mean).sum() + 1) / (len(bottom) + 1)) if len(bottom) else float("nan"),
        "mean_label_permuted_delta_auc": float(perm.mean()) if len(perm) else float("nan"),
        "sd_label_permuted_delta_auc": float(perm.std()) if len(perm) else float("nan"),
        "empirical_p_label_permuted_delta_ge_top_mean": float(((perm >= top_mean).sum() + 1) / (len(perm) + 1)) if len(perm) else float("nan"),
        "selected_feature_only_auroc": auc_on_analysis(selected_only_pred),
        "bias_only_auroc": auc_on_analysis(bias_only_pred),
        "delta_auroc_selected_only_minus_bias_only": auc_on_analysis(selected_only_pred) - auc_on_analysis(bias_only_pred),
        "mean_fold_selected_feature_only_auc": selected_only_mean,
        "mean_random_feature_only_auc": float(random_only.mean()) if len(random_only) else float("nan"),
        "sd_random_feature_only_auc": float(random_only.std()) if len(random_only) else float("nan"),
        "empirical_p_random_feature_only_auc_ge_selected_mean": float(((random_only >= selected_only_mean).sum() + 1) / (len(random_only) + 1)) if len(random_only) else float("nan"),
        "mean_bottom_feature_only_auc": float(bottom_only.mean()) if len(bottom_only) else float("nan"),
        "empirical_p_bottom_feature_only_auc_ge_selected_mean": float(((bottom_only >= selected_only_mean).sum() + 1) / (len(bottom_only) + 1)) if len(bottom_only) else float("nan"),
        "mean_rescue_delta_auc_selected_minus_random": float(rescue_delta_auc.mean()) if len(rescue_delta_auc) else float("nan"),
        "sd_rescue_delta_auc_selected_minus_random": float(rescue_delta_auc.std()) if len(rescue_delta_auc) else float("nan"),
        "empirical_p_rescue_delta_auc_le_zero": float(((rescue_delta_auc <= 0).sum() + 1) / (len(rescue_delta_auc) + 1)) if len(rescue_delta_auc) else float("nan"),
        "mean_rescue_delta_auprc_selected_minus_random": float(rescue_delta_ap.mean()) if len(rescue_delta_ap) else float("nan"),
        "sd_rescue_delta_auprc_selected_minus_random": float(rescue_delta_ap.std()) if len(rescue_delta_ap) else float("nan"),
        "empirical_p_rescue_delta_auprc_le_zero": float(((rescue_delta_ap <= 0).sum() + 1) / (len(rescue_delta_ap) + 1)) if len(rescue_delta_ap) else float("nan"),
        "mean_sae_val_mse": float(numeric_col(folds, "sae_val_mse").mean()) if len(numeric_col(folds, "sae_val_mse")) else float("nan"),
        "mean_sae_val_dead_frac": float(numeric_col(folds, "sae_val_dead_frac").mean()) if len(numeric_col(folds, "sae_val_dead_frac")) else float("nan"),
        "target_representation": args.target_representation,
        "normalize_recon": normalize_recon,
    }
    if not dose_response.empty:
        for dose_k, group in dose_response.groupby("dose_k"):
            summary_row[f"mean_dose_{int(dose_k)}_delta_auc"] = float(group["delta_auc_recon_minus_ablate"].mean())
    summary = pd.DataFrame([summary_row])

    scores = df.loc[score_mask].copy()
    scores["original_pred"] = original_pred[score_mask]
    scores["sae_recon_pred"] = recon_pred[score_mask]
    scores["top_feature_ablate_pred"] = top_pred[score_mask]
    scores["selected_feature_only_pred"] = selected_only_pred[score_mask]
    scores["bias_only_pred"] = bias_only_pred[score_mask]
    scores["original_gate_protein"] = original_gate[score_mask, 0]
    scores["sae_recon_gate_protein"] = recon_gate[score_mask, 0]
    scores["top_feature_ablate_gate_protein"] = top_gate[score_mask, 0]
    strata = stratum_summary_generic(scores, args.domain_col)

    prefix = args.output_prefix
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    folds.to_csv(out / f"{prefix}_folds.csv", index=False)
    randoms.to_csv(out / f"{prefix}_random_ablation.csv", index=False)
    controls.to_csv(out / f"{prefix}_controls.csv", index=False)
    dose_response.to_csv(out / f"{prefix}_dose_response.csv", index=False)
    sufficiency.to_csv(out / f"{prefix}_sufficiency_controls.csv", index=False)
    rescue.to_csv(out / f"{prefix}_rescue_controls.csv", index=False)
    selected.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    strata.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    scores.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(
        out / f"{prefix}.md",
        args.run_name,
        summary,
        folds,
        strata,
        selected,
        controls,
        dose_response,
        sufficiency,
        rescue,
        target_representation=args.target_representation,
    )

    print(f"Wrote {args.run_name} fold-native fine-tuned {args.target_representation} SAE intervention outputs to {out}")
    print(summary.to_string(index=False))
    print()
    print(folds.to_string(index=False))
    print()
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
