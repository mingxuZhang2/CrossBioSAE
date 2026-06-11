#!/usr/bin/env python
"""Third-gene latent-SAE intervention on ESM delta features.

This is a checkpoint-level causal test for BAP1/RAD51C. It trains one TopK SAE
per grouped CV fold on the fold's ESM pdelta representation, trains a logistic
variant-effect head on the original ESM pdelta plus Evo2 LLR scalar, then tests
whether removing train-selected sparse ESM features changes held-out SGE LOF
prediction more than matched random features or label-permuted feature
selection.

The result is intentionally weaker than the BRCA1 fold-native fine-tuned z_dna
intervention because the head is a checkpoint logistic probe, not the original
CrossBioSAE gate/head. It is still useful as the next third-gene causal screen:
positive results identify whether checkpoint-positive genes have a sparse
protein-latent mechanism worth promoting to a full native-SAE replication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


OUT_DIR = Path("results/interpretability_applications")


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int) -> None:
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


@dataclass
class FoldData:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
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


def fmt(x: Any, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}g}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x[train_idx], axis=0, keepdims=True)
    scale = np.nanstd(x[train_idx], axis=0, keepdims=True)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-6), 1.0, scale)
    out = (x - mean) / scale
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[FoldData]:
    folds: list[FoldData] = []
    group_counts = pd.Series(groups).nunique()
    if group_counts >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(np.zeros(len(y)), y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(y)), y)
    rng = np.random.default_rng(seed)
    for fold, (train_val, test_idx) in enumerate(split_iter):
        train_val = np.asarray(train_val, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        yy = y[train_val]
        val_parts = []
        for label in [0, 1]:
            label_idx = train_val[yy == label]
            n_val = max(1, int(round(0.15 * len(label_idx))))
            val_parts.append(rng.choice(label_idx, size=n_val, replace=False))
        val_idx = np.sort(np.concatenate(val_parts).astype(int))
        val_set = set(val_idx.tolist())
        train_idx = np.array([i for i in train_val if int(i) not in val_set], dtype=int)
        folds.append(FoldData(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    return folds


def train_sae(
    z: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> tuple[TopKSAE, dict[str, float]]:
    set_seed(seed)
    model = TopKSAE(d_in=z.shape[1], d_hidden=args.sae_hidden, k=args.sae_k).to(device)
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
    return model, {
        "sae_best_epoch": float(best_epoch),
        "sae_train_mse": float(F.mse_loss(recon_train, train).item()),
        "sae_val_mse": float(F.mse_loss(recon_val, val).item()),
        "sae_train_mean_l0": float((acts_train > 1e-8).sum(dim=1).float().mean().item()),
        "sae_val_mean_l0": float((acts_val > 1e-8).sum(dim=1).float().mean().item()),
        "sae_train_dead_frac": float(((acts_train > 1e-8).sum(dim=0) == 0).float().mean().item()),
        "sae_val_dead_frac": float(((acts_val > 1e-8).sum(dim=0) == 0).float().mean().item()),
    }


@torch.no_grad()
def encode_all(model: TopKSAE, z: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    rows = []
    for start in range(0, len(z), batch_size):
        x = torch.tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
        rows.append(model.encode(x).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


@torch.no_grad()
def decode_acts(model: TopKSAE, acts: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    rows = []
    for start in range(0, len(acts), batch_size):
        h = torch.tensor(acts[start : start + batch_size], dtype=torch.float32, device=device)
        rows.append(model.decoder(h).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def score_features(acts: np.ndarray, y: np.ndarray, train_idx: np.ndarray, min_active: int) -> pd.DataFrame:
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
        rows.append({
            "feature": int(feature),
            "train_active": int(active_counts[feature]),
            "train_delta_abs_lof_minus_func": float(lof.mean() - func.mean()),
            "train_abs_delta": float(abs(lof.mean() - func.mean())),
            "train_auroc_feature": safe_auc(yy, score),
            "train_p_lof_gt_func": float(mannwhitneyu(lof, func, alternative="greater").pvalue),
        })
    return pd.DataFrame(rows)


def top_features(scored: pd.DataFrame, top_k: int) -> np.ndarray:
    if scored.empty:
        return np.array([], dtype=int)
    use = scored[scored["train_delta_abs_lof_minus_func"] > 0].copy()
    if use.empty:
        return np.array([], dtype=int)
    use = use.sort_values(["train_p_lof_gt_func", "train_auroc_feature"], ascending=[True, False])
    return use["feature"].head(top_k).to_numpy(dtype=int)


def matched_random_sets(
    selected: np.ndarray,
    scored: pd.DataFrame,
    top_k: int,
    random_sets: int,
    seed: int,
) -> list[np.ndarray]:
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
            tmp = pool_df[~pool_df["feature"].isin(used)].copy()
            if tmp.empty:
                break
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


def matched_bottom_features(selected: np.ndarray, scored: pd.DataFrame, seed: int) -> np.ndarray:
    if len(selected) == 0 or scored.empty:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    selected_set = set(map(int, selected))
    counts = scored.set_index("feature")["train_active"].to_dict()
    pool = scored[~scored["feature"].isin(selected_set)].copy()
    chosen = []
    used = set()
    for feat in selected:
        target = counts[int(feat)]
        tmp = pool[~pool["feature"].isin(used)].copy()
        if tmp.empty:
            break
        tmp["count_dist"] = (tmp["train_active"] - target).abs()
        candidates = tmp.sort_values(["train_abs_delta", "count_dist"], ascending=[True, True])
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
    return top_features(score_features(acts, y_perm, train_idx, min_active), top_k)


def make_x(z_esm: np.ndarray, llr: np.ndarray | None) -> np.ndarray:
    if llr is None:
        return z_esm
    return np.concatenate([z_esm, llr], axis=1).astype(np.float32)


def predict(head: LogisticRegression, z_esm: np.ndarray, llr: np.ndarray | None) -> np.ndarray:
    return head.predict_proba(make_x(z_esm, llr))[:, 1].astype(np.float32)


def evaluate_ablation(
    sae: TopKSAE,
    head: LogisticRegression,
    acts_all: np.ndarray,
    features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    llr: np.ndarray | None,
    recon_auc: float,
    recon_ap: float,
    device: str,
) -> tuple[np.ndarray, float, float, float, float]:
    if len(features) == 0:
        nan = float("nan")
        return np.full(len(te), nan, dtype=np.float32), nan, nan, nan, nan
    acts_tmp = acts_all[te].copy()
    acts_tmp[:, features] = 0.0
    z_tmp = decode_acts(sae, acts_tmp, device)
    pred = predict(head, z_tmp, llr[te] if llr is not None else None)
    auc = safe_auc(y[te], pred)
    ap = safe_auprc(y[te], pred)
    return pred, auc, ap, recon_auc - auc, recon_ap - ap


def evaluate_feature_only(
    sae: TopKSAE,
    head: LogisticRegression,
    acts_all: np.ndarray,
    features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    llr: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, float, float]:
    acts_tmp = np.zeros_like(acts_all[te])
    if len(features):
        acts_tmp[:, features] = acts_all[te][:, features]
    z_tmp = decode_acts(sae, acts_tmp, device)
    pred = predict(head, z_tmp, llr[te] if llr is not None else None)
    return pred, safe_auc(y[te], pred), safe_auprc(y[te], pred)


def evaluate_rescue(
    sae: TopKSAE,
    head: LogisticRegression,
    acts_all: np.ndarray,
    selected: np.ndarray,
    random_features: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    llr: np.ndarray | None,
    device: str,
) -> dict[str, float | int | str]:
    pool = np.array(sorted(set(selected.tolist()) | set(random_features.tolist())), dtype=int)
    if len(pool) == 0:
        return {}
    acts_base = acts_all[te].copy()
    acts_base[:, pool] = 0.0
    pred_base = predict(head, decode_acts(sae, acts_base, device), llr[te] if llr is not None else None)
    acts_selected = acts_base.copy()
    acts_selected[:, selected] = acts_all[te][:, selected]
    pred_selected = predict(head, decode_acts(sae, acts_selected, device), llr[te] if llr is not None else None)
    acts_random = acts_base.copy()
    acts_random[:, random_features] = acts_all[te][:, random_features]
    pred_random = predict(head, decode_acts(sae, acts_random, device), llr[te] if llr is not None else None)
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


def stratum_summary(scores: pd.DataFrame, y: np.ndarray, domain_col: str | None) -> pd.DataFrame:
    strata: dict[str, np.ndarray] = {"all": np.ones(len(scores), dtype=bool)}
    for col, name in [("is_missense", "missense"), ("is_synonymous", "synonymous"), ("is_nonsense", "nonsense")]:
        if col in scores:
            strata[name] = scores[col].astype(str).str.lower().eq("true").to_numpy()
    if "consequence" in scores:
        for value in scores["consequence"].dropna().astype(str).unique():
            strata[f"consequence:{value}"] = scores["consequence"].astype(str).eq(value).to_numpy()
    if domain_col and domain_col in scores:
        for value in scores[domain_col].dropna().astype(str).unique():
            if value and value != "-":
                strata[f"{domain_col}:{value}"] = scores[domain_col].astype(str).eq(value).to_numpy()
    rows = []
    for name, mask in strata.items():
        if int(mask.sum()) < 40 or len(np.unique(y[mask])) < 2:
            continue
        row = {"stratum": name, "n": int(mask.sum()), "n_lof": int(y[mask].sum())}
        for col in ["original_pred", "sae_recon_pred", "top_feature_ablate_pred"]:
            row[f"auroc_{col}"] = safe_auc(y[mask], scores.loc[mask, col].to_numpy())
            row[f"auprc_{col}"] = safe_auprc(y[mask], scores.loc[mask, col].to_numpy())
        row["delta_auroc_recon_minus_top_ablate"] = row["auroc_sae_recon_pred"] - row["auroc_top_feature_ablate_pred"]
        row["delta_auprc_recon_minus_top_ablate"] = row["auprc_sae_recon_pred"] - row["auprc_top_feature_ablate_pred"]
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    lines = [
        "| " + " | ".join(show.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in show.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    strata: pd.DataFrame,
    selected: pd.DataFrame,
    controls: pd.DataFrame,
    dose: pd.DataFrame,
    sufficiency: pd.DataFrame,
    rescue: pd.DataFrame,
) -> None:
    s = summary.iloc[0].to_dict() if not summary.empty else {}
    positive = (
        float(s.get("delta_auroc_recon_minus_top_ablate", np.nan)) > 0
        and float(s.get("empirical_p_random_delta_ge_top_mean", np.nan)) <= 0.05
        and float(s.get("empirical_p_label_permuted_delta_ge_top_mean", np.nan)) <= 0.05
    )
    lines = [
        f"# {args.gene_name} Latent-SAE Checkpoint Intervention",
        "",
        "## Purpose",
        "",
        "This experiment is a third-gene causal screen on ESM pdelta latent features. It trains a TopK SAE inside each grouped CV training fold and asks whether removing train-selected sparse ESM features degrades held-out SGE LOF prediction through the same fold-local logistic checkpoint head.",
        "",
        "It is not a full BRCA1-style native CrossBioSAE gate intervention. A positive result supports sparse latent-feature necessity for the checkpoint representation and prioritizes the gene/stratum for a full native-SAE replication.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Fold-Level Intervention",
        "",
        md_table(folds),
        "",
        "## Strata",
        "",
        md_table(strata, max_rows=60),
        "",
        "## Selected Sparse Features",
        "",
        md_table(selected, max_rows=80),
        "",
        "## Control Ablations",
        "",
        md_table(controls, max_rows=100),
        "",
        "## Dose Response",
        "",
        md_table(dose, max_rows=80),
        "",
        "## Feature-Only Controls",
        "",
        md_table(sufficiency, max_rows=80),
        "",
        "## Common-Background Addback Rescue",
        "",
        md_table(rescue, max_rows=80),
        "",
        "## Interpretation",
        "",
        (
            "The preset screen is positive: selected sparse features reduce held-out prediction more than matched random features and label-permuted feature selection. This can be written as checkpoint-level sparse-feature necessity, not as full CrossBioSAE causal replication."
            if positive
            else "The preset screen is not positive. Treat this as error analysis or a boundary result unless a stricter rerun with better representation/head design passes random-feature and label-permutation controls."
        ),
        "",
        "A strong top-journal claim would still require a full native-SAE intervention in the trained CrossBioSAE representation, addback rescue, label-permutation controls, and preferably locked review or assay outcomes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=repo)
    p.add_argument("--gene-name", default="RAD51C")
    p.add_argument("--variant-scores", type=Path, default=repo / OUT_DIR / "rad51c_llr_esm_checkpoint_variant_scores.csv")
    p.add_argument("--esm", type=Path, default=repo / "results" / "variant" / "rad51c_esm_delta.npz")
    p.add_argument("--llr", type=Path, default=repo / "results" / "variant" / "rad51c_evo2_llr.npz")
    p.add_argument("--output-dir", type=Path, default=repo / OUT_DIR)
    p.add_argument("--output-prefix", default="rad51c_latent_sae_intervention")
    p.add_argument("--group-col", default="pos_hg38")
    p.add_argument("--domain-col", default="domains")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--top-k-features", type=int, default=32)
    p.add_argument("--min-active", type=int, default=10)
    p.add_argument("--random-sets", type=int, default=100)
    p.add_argument("--permutation-sets", type=int, default=20)
    p.add_argument("--dose-k-values", default="8,16,32,64")
    p.add_argument("--sae-hidden", type=int, default=1024)
    p.add_argument("--sae-k", type=int, default=32)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--l1", type=float, default=1e-5)
    p.add_argument("--logistic-c", type=float, default=0.5)
    p.add_argument("--no-llr-scalar", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--random-seed", type=int, default=630)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dose_values = sorted({int(x) for x in str(args.dose_k_values).split(",") if str(x).strip()})

    raw = pd.read_csv(require_file(args.variant_scores)).reset_index(drop=True)
    keep = pd.to_numeric(raw["label"], errors="coerce").notna()
    df = raw.loc[keep].reset_index(drop=True)
    raw_idx = np.flatnonzero(keep.to_numpy())
    y = df["label"].astype(int).to_numpy()

    esm = np.load(require_file(args.esm))
    pdelta_all = esm["pdelta"].astype(np.float32)
    z_raw = pdelta_all[raw_idx]
    llr_raw: np.ndarray | None = None
    if not args.no_llr_scalar:
        llr_all = np.load(require_file(args.llr))["llr"].astype(np.float32)
        llr_raw = llr_all[raw_idx][:, None]
        median = np.nanmedian(llr_raw)
        llr_raw = np.where(np.isfinite(llr_raw), llr_raw, median).astype(np.float32)

    groups = df[args.group_col].to_numpy() if args.group_col in df else np.arange(len(df))
    folds = make_folds(y, groups, args.n_splits, args.random_seed)

    original_pred = np.full(len(df), np.nan, dtype=np.float32)
    recon_pred = np.full(len(df), np.nan, dtype=np.float32)
    top_pred = np.full(len(df), np.nan, dtype=np.float32)
    selected_only_pred = np.full(len(df), np.nan, dtype=np.float32)
    bias_only_pred = np.full(len(df), np.nan, dtype=np.float32)
    fold_rows = []
    selected_rows = []
    random_rows = []
    control_rows = []
    dose_rows = []
    suff_rows = []
    rescue_rows = []

    for fd in folds:
        z, _, _ = standardize_train(z_raw, fd.train_idx)
        llr = None
        if llr_raw is not None:
            llr, _, _ = standardize_train(llr_raw, fd.train_idx)

        head = LogisticRegression(
            C=args.logistic_c,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=args.random_seed + fd.fold,
        )
        head.fit(make_x(z[fd.train_idx], llr[fd.train_idx] if llr is not None else None), y[fd.train_idx])
        original_pred[fd.test_idx] = predict(head, z[fd.test_idx], llr[fd.test_idx] if llr is not None else None)

        sae, sae_metrics = train_sae(z, fd.train_idx, fd.val_idx, args, device, args.random_seed + fd.fold)
        acts_all = encode_all(sae, z, device)
        scored = score_features(acts_all, y, fd.train_idx, args.min_active)
        selected = top_features(scored, args.top_k_features)
        if len(selected) == 0:
            fold_rows.append({"fold": fd.fold, "status": "no_selected_features", **sae_metrics})
            continue

        chosen = scored[scored["feature"].isin(selected)].copy()
        chosen["fold"] = fd.fold
        chosen["note"] = "fold_local_feature_id_not_global"
        selected_rows.append(chosen)

        z_recon_te = decode_acts(sae, acts_all[fd.test_idx], device)
        pred_recon = predict(head, z_recon_te, llr[fd.test_idx] if llr is not None else None)
        recon_pred[fd.test_idx] = pred_recon
        recon_auc = safe_auc(y[fd.test_idx], pred_recon)
        recon_ap = safe_auprc(y[fd.test_idx], pred_recon)
        pred_top, top_auc, top_ap, delta_auc, delta_ap = evaluate_ablation(
            sae, head, acts_all, selected, fd.test_idx, y, llr, recon_auc, recon_ap, device
        )
        top_pred[fd.test_idx] = pred_top
        pred_selected_only, selected_only_auc, selected_only_ap = evaluate_feature_only(
            sae, head, acts_all, selected, fd.test_idx, y, llr, device
        )
        pred_bias_only, bias_only_auc, bias_only_ap = evaluate_feature_only(
            sae, head, acts_all, np.array([], dtype=int), fd.test_idx, y, llr, device
        )
        selected_only_pred[fd.test_idx] = pred_selected_only
        bias_only_pred[fd.test_idx] = pred_bias_only

        fold_rows.append({
            "fold": fd.fold,
            "status": "tested",
            "n_train": int(len(fd.train_idx)),
            "n_test": int(len(fd.test_idx)),
            "n_test_lof": int(y[fd.test_idx].sum()),
            "n_selected_features": int(len(selected)),
            "original_auc": safe_auc(y[fd.test_idx], original_pred[fd.test_idx]),
            "sae_recon_auc": recon_auc,
            "top_feature_ablate_auc": top_auc,
            "delta_auc_recon_minus_top_ablate": delta_auc,
            "original_auprc": safe_auprc(y[fd.test_idx], original_pred[fd.test_idx]),
            "sae_recon_auprc": recon_ap,
            "top_feature_ablate_auprc": top_ap,
            "delta_auprc_recon_minus_top_ablate": delta_ap,
            "selected_feature_only_auc": selected_only_auc,
            "bias_only_auc": bias_only_auc,
            "delta_auc_selected_only_minus_bias_only": selected_only_auc - bias_only_auc,
            "selected_features": ";".join(map(str, selected.tolist())),
            **sae_metrics,
        })

        for dose_k in dose_values:
            feats = top_features(scored, dose_k)
            _, auc, ap, d_auc, d_ap = evaluate_ablation(
                sae, head, acts_all, feats, fd.test_idx, y, llr, recon_auc, recon_ap, device
            )
            dose_rows.append({
                "fold": fd.fold,
                "dose_k": int(len(feats)),
                "requested_k": int(dose_k),
                "ablate_auc": auc,
                "delta_auc_recon_minus_ablate": d_auc,
                "ablate_auprc": ap,
                "delta_auprc_recon_minus_ablate": d_ap,
                "features": ";".join(map(str, feats.tolist())),
            })

        bottom = matched_bottom_features(selected, scored, args.random_seed + 10_000 + fd.fold)
        _, auc, ap, d_auc, d_ap = evaluate_ablation(
            sae, head, acts_all, bottom, fd.test_idx, y, llr, recon_auc, recon_ap, device
        )
        control_rows.append({
            "fold": fd.fold,
            "control": "bottom_abs_delta_active_matched",
            "set_id": 0,
            "n_features": int(len(bottom)),
            "ablate_auc": auc,
            "delta_auc_recon_minus_ablate": d_auc,
            "ablate_auprc": ap,
            "delta_auprc_recon_minus_ablate": d_ap,
            "features": ";".join(map(str, bottom.tolist())),
        })
        _, bottom_only_auc, bottom_only_ap = evaluate_feature_only(sae, head, acts_all, bottom, fd.test_idx, y, llr, device)
        suff_rows.append({
            "fold": fd.fold,
            "control": "bottom_abs_delta_active_matched_only",
            "set_id": 0,
            "n_features": int(len(bottom)),
            "feature_only_auc": bottom_only_auc,
            "feature_only_auprc": bottom_only_ap,
            "features": ";".join(map(str, bottom.tolist())),
        })

        for perm_i in range(args.permutation_sets):
            feats = permuted_label_features(
                acts_all, y, fd.train_idx, args.top_k_features, args.min_active, args.random_seed + 20_000 + 100 * fd.fold + perm_i
            )
            _, auc, ap, d_auc, d_ap = evaluate_ablation(
                sae, head, acts_all, feats, fd.test_idx, y, llr, recon_auc, recon_ap, device
            )
            control_rows.append({
                "fold": fd.fold,
                "control": "label_permuted_selection",
                "set_id": perm_i,
                "n_features": int(len(feats)),
                "ablate_auc": auc,
                "delta_auc_recon_minus_ablate": d_auc,
                "ablate_auprc": ap,
                "delta_auprc_recon_minus_ablate": d_ap,
                "features": ";".join(map(str, feats.tolist())),
            })

        random_sets = matched_random_sets(selected, scored, args.top_k_features, args.random_sets, args.random_seed + fd.fold)
        for set_id, feats in enumerate(random_sets):
            pred_rand, rand_auc, rand_ap, d_auc, d_ap = evaluate_ablation(
                sae, head, acts_all, feats, fd.test_idx, y, llr, recon_auc, recon_ap, device
            )
            random_rows.append({
                "fold": fd.fold,
                "random_set": set_id,
                "random_ablate_auc": rand_auc,
                "delta_auc_random_ablate": d_auc,
                "random_ablate_auprc": rand_ap,
                "delta_auprc_random_ablate": d_ap,
            })
            _, random_only_auc, random_only_ap = evaluate_feature_only(sae, head, acts_all, feats, fd.test_idx, y, llr, device)
            suff_rows.append({
                "fold": fd.fold,
                "control": "random_active_matched_only",
                "set_id": set_id,
                "n_features": int(len(feats)),
                "feature_only_auc": random_only_auc,
                "feature_only_auprc": random_only_ap,
                "features": ";".join(map(str, feats.tolist())),
            })
            rescue = evaluate_rescue(sae, head, acts_all, selected, feats, fd.test_idx, y, llr, device)
            if rescue:
                rescue_rows.append({
                    "fold": fd.fold,
                    "random_set": set_id,
                    **rescue,
                    "selected_features": ";".join(map(str, selected.tolist())),
                    "random_features": ";".join(map(str, feats.tolist())),
                })

    folds_df = pd.DataFrame(fold_rows)
    random_df = pd.DataFrame(random_rows)
    controls_df = pd.DataFrame(control_rows)
    dose_df = pd.DataFrame(dose_rows)
    suff_df = pd.DataFrame(suff_rows)
    rescue_df = pd.DataFrame(rescue_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    top_mean = float(folds_df.loc[folds_df["status"].eq("tested"), "delta_auc_recon_minus_top_ablate"].mean())
    rand = random_df["delta_auc_random_ablate"].dropna() if not random_df.empty else pd.Series(dtype=float)
    perm = controls_df.loc[controls_df["control"].eq("label_permuted_selection"), "delta_auc_recon_minus_ablate"].dropna() if not controls_df.empty else pd.Series(dtype=float)
    bottom = controls_df.loc[controls_df["control"].eq("bottom_abs_delta_active_matched"), "delta_auc_recon_minus_ablate"].dropna() if not controls_df.empty else pd.Series(dtype=float)
    selected_only_mean = float(folds_df.loc[folds_df["status"].eq("tested"), "selected_feature_only_auc"].mean())
    random_only = suff_df.loc[suff_df["control"].eq("random_active_matched_only"), "feature_only_auc"].dropna() if not suff_df.empty else pd.Series(dtype=float)
    rescue_delta = rescue_df["delta_auc_selected_rescue_minus_random_rescue"].dropna() if not rescue_df.empty else pd.Series(dtype=float)
    y_all = y
    summary_row = {
        "gene": args.gene_name,
        "n": int(len(df)),
        "n_lof": int(y.sum()),
        "representation": "ESM pdelta SAE + Evo2 LLR scalar" if llr_raw is not None else "ESM pdelta SAE only",
        "original_auroc": safe_auc(y_all, original_pred),
        "sae_recon_auroc": safe_auc(y_all, recon_pred),
        "top_feature_ablate_auroc": safe_auc(y_all, top_pred),
        "delta_auroc_recon_minus_top_ablate": safe_auc(y_all, recon_pred) - safe_auc(y_all, top_pred),
        "original_auprc": safe_auprc(y_all, original_pred),
        "sae_recon_auprc": safe_auprc(y_all, recon_pred),
        "top_feature_ablate_auprc": safe_auprc(y_all, top_pred),
        "delta_auprc_recon_minus_top_ablate": safe_auprc(y_all, recon_pred) - safe_auprc(y_all, top_pred),
        "mean_fold_delta_auc_recon_minus_top_ablate": top_mean,
        "mean_random_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
        "empirical_p_random_delta_ge_top_mean": float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan"),
        "mean_bottom_delta_auc": float(bottom.mean()) if len(bottom) else float("nan"),
        "empirical_p_bottom_delta_ge_top_mean": float(((bottom >= top_mean).sum() + 1) / (len(bottom) + 1)) if len(bottom) else float("nan"),
        "mean_label_permuted_delta_auc": float(perm.mean()) if len(perm) else float("nan"),
        "empirical_p_label_permuted_delta_ge_top_mean": float(((perm >= top_mean).sum() + 1) / (len(perm) + 1)) if len(perm) else float("nan"),
        "selected_feature_only_auroc": safe_auc(y_all, selected_only_pred),
        "bias_only_auroc": safe_auc(y_all, bias_only_pred),
        "mean_fold_selected_feature_only_auc": selected_only_mean,
        "mean_random_feature_only_auc": float(random_only.mean()) if len(random_only) else float("nan"),
        "empirical_p_random_feature_only_auc_ge_selected_mean": float(((random_only >= selected_only_mean).sum() + 1) / (len(random_only) + 1)) if len(random_only) else float("nan"),
        "mean_rescue_delta_auc_selected_minus_random": float(rescue_delta.mean()) if len(rescue_delta) else float("nan"),
        "empirical_p_rescue_delta_auc_le_zero": float(((rescue_delta <= 0).sum() + 1) / (len(rescue_delta) + 1)) if len(rescue_delta) else float("nan"),
        "mean_sae_val_mse": float(folds_df["sae_val_mse"].mean()) if "sae_val_mse" in folds_df else float("nan"),
        "mean_sae_val_dead_frac": float(folds_df["sae_val_dead_frac"].mean()) if "sae_val_dead_frac" in folds_df else float("nan"),
    }
    for dose_k, group in dose_df.groupby("dose_k") if not dose_df.empty else []:
        summary_row[f"mean_dose_{int(dose_k)}_delta_auc"] = float(group["delta_auc_recon_minus_ablate"].mean())
    summary_df = pd.DataFrame([summary_row])

    scores = df.copy()
    scores["original_pred"] = original_pred
    scores["sae_recon_pred"] = recon_pred
    scores["top_feature_ablate_pred"] = top_pred
    scores["selected_feature_only_pred"] = selected_only_pred
    scores["bias_only_pred"] = bias_only_pred
    strata_df = stratum_summary(scores, y, args.domain_col)

    prefix = args.output_prefix
    summary_df.to_csv(out / f"{prefix}_summary.csv", index=False)
    folds_df.to_csv(out / f"{prefix}_folds.csv", index=False)
    random_df.to_csv(out / f"{prefix}_random_ablation.csv", index=False)
    controls_df.to_csv(out / f"{prefix}_controls.csv", index=False)
    dose_df.to_csv(out / f"{prefix}_dose_response.csv", index=False)
    suff_df.to_csv(out / f"{prefix}_sufficiency_controls.csv", index=False)
    rescue_df.to_csv(out / f"{prefix}_rescue_controls.csv", index=False)
    selected_df.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    strata_df.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    scores.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(out / f"{prefix}.md", args, summary_df, folds_df, strata_df, selected_df, controls_df, dose_df, suff_df, rescue_df)

    print(f"wrote {out / f'{prefix}_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
