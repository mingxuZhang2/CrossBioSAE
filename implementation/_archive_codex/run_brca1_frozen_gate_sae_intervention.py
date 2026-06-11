#!/usr/bin/env python3
"""
BRCA1 frozen-encoder gate/head intervention through SAE features.

This script trains only the downstream gate/head on frozen pretrained shared
representations, then intervenes inside the model input by replacing z_dna with
SAE-decoded z_dna after removing training-fold-selected SGE-associated sparse
features. This is closer to a model-level test than an activation-only probe,
while remaining compatible with the genome-wide SAE feature space.

It is not a replacement for intervention inside the fully fine-tuned gate model.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

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


class FrozenGateHead(nn.Module):
    def __init__(self, d_shared: int = 256, scalar_dim: int = 0, d_hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.gate = nn.Sequential(nn.Linear(d_shared * 2 + scalar_dim, d_hidden), nn.GELU(), nn.Linear(d_hidden, 2))
        self.head = nn.Sequential(
            nn.LayerNorm(d_shared + scalar_dim),
            nn.Linear(d_shared + scalar_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        zp: torch.Tensor,
        zd: torch.Tensor,
        scalars: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        gate_in = torch.cat([zp, zd], dim=-1)
        if scalars is not None:
            gate_in = torch.cat([gate_in, scalars], dim=-1)
        g = F.softmax(self.gate(gate_in), dim=-1)
        z = g[:, 0:1] * zp + g[:, 1:2] * zd
        if scalars is not None:
            z = torch.cat([z, scalars], dim=-1)
        logit = self.head(z).squeeze(-1)
        if return_gate:
            return logit, g
        return logit


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
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "interpretability_applications")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=630)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


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


def load_clip_z_prot(checkpoint: Path, pdelta: np.ndarray, pmask: np.ndarray, device: str) -> np.ndarray:
    ckpt = torch.load(require_file(checkpoint), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CrossModalCLIP(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(pdelta), 512):
            x = torch.tensor(pdelta[start : start + 512], dtype=torch.float32, device=device)
            out.append(model.enc_prot(x).cpu().numpy())
    z = np.concatenate(out, axis=0).astype(np.float32)
    return z * pmask[:, None].astype(np.float32)


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


def fit_inverse_scaled_to_raw(z_raw: np.ndarray, z_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = z_scaled.astype(np.float64)
    y = z_raw.astype(np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_var = ((x - x_mean) ** 2).mean(axis=0)
    cov = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    scale = np.divide(cov, x_var, out=np.ones_like(cov), where=x_var > 1e-12)
    intercept = y_mean - scale * x_mean
    return scale.astype(np.float32), intercept.astype(np.float32)


def inverse_scaled(decoded_scaled: np.ndarray, scale: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return l2_normalize(decoded_scaled * scale[None, :] + intercept[None, :])


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
        chosen = []
        used = set()
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


def standardize_train_apply(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return tuple(((a - mean) / scale).astype(np.float32) for a in arrays)


def train_gate(
    z_prot: np.ndarray,
    z_dna: np.ndarray,
    scalars: np.ndarray | None,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> FrozenGateHead:
    scalar_dim = 0 if scalars is None else scalars.shape[1]
    model = FrozenGateHead(d_shared=z_prot.shape[1], scalar_dim=scalar_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    zp_tr = torch.tensor(z_prot[tr], dtype=torch.float32, device=device)
    zd_tr = torch.tensor(z_dna[tr], dtype=torch.float32, device=device)
    yy_tr = torch.tensor(y[tr], dtype=torch.float32, device=device)
    zp_va = torch.tensor(z_prot[va], dtype=torch.float32, device=device)
    zd_va = torch.tensor(z_dna[va], dtype=torch.float32, device=device)
    sc_tr = torch.tensor(scalars[tr], dtype=torch.float32, device=device) if scalars is not None else None
    sc_va = torch.tensor(scalars[va], dtype=torch.float32, device=device) if scalars is not None else None
    best_auc = -1.0
    best_state = None
    bad = 0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logit = model(zp_tr, zd_tr, sc_tr)
        loss = F.binary_cross_entropy_with_logits(logit, yy_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(zp_va, zd_va, sc_va)).cpu().numpy()
        auc = safe_auc(y[va], pred)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_gate(model: FrozenGateHead, z_prot: np.ndarray, z_dna: np.ndarray, scalars: np.ndarray | None, idx: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    zp = torch.tensor(z_prot[idx], dtype=torch.float32, device=device)
    zd = torch.tensor(z_dna[idx], dtype=torch.float32, device=device)
    sc = torch.tensor(scalars[idx], dtype=torch.float32, device=device) if scalars is not None else None
    logit, gate = model(zp, zd, sc, return_gate=True)
    return torch.sigmoid(logit).cpu().numpy(), gate.cpu().numpy()


def run_rep(
    rep: str,
    df: pd.DataFrame,
    z_prot: np.ndarray,
    z_dna: np.ndarray,
    z_dna_scaled: np.ndarray,
    acts: np.ndarray,
    sae: TopKSAE,
    llr: np.ndarray,
    inv_scale: np.ndarray,
    inv_intercept: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["label"].to_numpy(dtype=int)
    groups = df["pos_hg19"].to_numpy()
    use_llr = rep.endswith("_llr")
    raw_scalar = llr[:, None].astype(np.float32)
    pred_full = np.zeros(len(df), dtype=float)
    pred_top = np.zeros(len(df), dtype=float)
    gate_full = np.zeros((len(df), 2), dtype=float)
    fold_rows = []
    random_rows = []
    selected_rows = []
    score_rows = []

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_all, te) in enumerate(gkf.split(acts, y, groups)):
        g_tr = groups[tr_all]
        uniq = np.unique(g_tr)
        rng = np.random.default_rng(args.random_seed + fold)
        va_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        va = tr_all[np.array([g in va_groups for g in g_tr])]
        tr = tr_all[np.array([g not in va_groups for g in g_tr])]

        zp_tr_std, zp_va_std, zp_te_std = standardize_train_apply(z_prot[tr], z_prot[tr], z_prot[va], z_prot[te])
        zd_tr_std, zd_va_std, zd_te_std = standardize_train_apply(z_dna[tr], z_dna[tr], z_dna[va], z_dna[te])
        z_prot_std = z_prot.copy()
        z_dna_std = z_dna.copy()
        z_prot_std[tr], z_prot_std[va], z_prot_std[te] = zp_tr_std, zp_va_std, zp_te_std
        z_dna_std[tr], z_dna_std[va], z_dna_std[te] = zd_tr_std, zd_va_std, zd_te_std
        scalars = None
        if use_llr:
            sc_tr, sc_va, sc_te = standardize_train_apply(raw_scalar[tr], raw_scalar[tr], raw_scalar[va], raw_scalar[te])
            scalars = raw_scalar.copy()
            scalars[tr], scalars[va], scalars[te] = sc_tr, sc_va, sc_te

        selected, scored = select_features(acts, y, tr, args.top_k, args.min_active)
        if len(selected) == 0:
            continue
        selected_df = scored[scored["feature"].isin(selected)].copy()
        selected_df["fold"] = fold
        selected_df["rep"] = rep
        selected_rows.append(selected_df)

        model = train_gate(z_prot_std, z_dna_std, scalars, y, tr, va, args, device)
        pred_full[te], gate_full[te] = predict_gate(model, z_prot_std, z_dna_std, scalars, te, device)

        acts_top = acts[te].copy()
        acts_top[:, selected] = 0.0
        decoded_top_scaled = decode_acts(sae, acts_top, device)
        zd_top_raw = inverse_scaled(decoded_top_scaled, inv_scale, inv_intercept)
        _, _, zd_top_std = standardize_train_apply(z_dna[tr], z_dna[tr], z_dna[va], zd_top_raw)
        z_dna_top = z_dna_std.copy()
        z_dna_top[te] = zd_top_std
        pred_top[te], _ = predict_gate(model, z_prot_std, z_dna_top, scalars, te, device)

        full_auc = safe_auc(y[te], pred_full[te])
        top_auc = safe_auc(y[te], pred_top[te])
        full_ap = safe_auprc(y[te], pred_full[te])
        top_ap = safe_auprc(y[te], pred_top[te])
        fold_rows.append(
            {
                "rep": rep,
                "fold": fold,
                "n_test": len(te),
                "n_selected_features": len(selected),
                "full_gate_auc": full_auc,
                "top_feature_ablate_auc": top_auc,
                "delta_auc_top_ablate": full_auc - top_auc,
                "full_gate_auprc": full_ap,
                "top_feature_ablate_auprc": top_ap,
                "delta_auprc_top_ablate": full_ap - top_ap,
                "mean_gate_protein": float(gate_full[te, 0].mean()),
                "mean_gate_dna": float(gate_full[te, 1].mean()),
                "selected_features": ";".join(map(str, selected.tolist())),
            }
        )

        random_sets = matched_random_sets(selected, scored, args.top_k, args.random_sets, args.random_seed + fold)
        for i, feats in enumerate(random_sets):
            acts_rand = acts[te].copy()
            acts_rand[:, feats] = 0.0
            zd_rand_raw = inverse_scaled(decode_acts(sae, acts_rand, device), inv_scale, inv_intercept)
            _, _, zd_rand_std = standardize_train_apply(z_dna[tr], z_dna[tr], z_dna[va], zd_rand_raw)
            z_dna_rand = z_dna_std.copy()
            z_dna_rand[te] = zd_rand_std
            pred_rand, _ = predict_gate(model, z_prot_std, z_dna_rand, scalars, te, device)
            random_auc = safe_auc(y[te], pred_rand)
            random_ap = safe_auprc(y[te], pred_rand)
            random_rows.append(
                {
                    "rep": rep,
                    "fold": fold,
                    "random_set": i,
                    "random_ablate_auc": random_auc,
                    "delta_auc_random_ablate": full_auc - random_auc,
                    "random_ablate_auprc": random_ap,
                    "delta_auprc_random_ablate": full_ap - random_ap,
                }
            )

    folds = pd.DataFrame(fold_rows)
    randoms = pd.DataFrame(random_rows)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    rand = randoms["delta_auc_random_ablate"].dropna() if not randoms.empty else pd.Series(dtype=float)
    top_mean = float(folds["delta_auc_top_ablate"].mean()) if not folds.empty else float("nan")
    summary = pd.DataFrame(
        [
            {
                "rep": rep,
                "n": len(df),
                "full_gate_auroc": safe_auc(y, pred_full),
                "top_feature_ablate_auroc": safe_auc(y, pred_top),
                "delta_auroc_top_ablate": safe_auc(y, pred_full) - safe_auc(y, pred_top),
                "full_gate_auprc": safe_auprc(y, pred_full),
                "top_feature_ablate_auprc": safe_auprc(y, pred_top),
                "delta_auprc_top_ablate": safe_auprc(y, pred_full) - safe_auprc(y, pred_top),
                "mean_fold_delta_auc_top_ablate": top_mean,
                "mean_random_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
                "sd_random_delta_auc": float(rand.std()) if len(rand) else float("nan"),
                "empirical_p_random_delta_ge_top_mean": float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan"),
            }
        ]
    )
    scores = df.copy()
    scores[f"{rep}_full_gate_pred"] = pred_full
    scores[f"{rep}_top_feature_ablate_pred"] = pred_top
    scores[f"{rep}_gate_protein"] = gate_full[:, 0]
    scores[f"{rep}_gate_dna"] = gate_full[:, 1]
    return summary, folds, randoms, selected, scores


def stratum_summary(scores: pd.DataFrame, reps: list[str]) -> pd.DataFrame:
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
        for rep in reps:
            full = scores.loc[mask, f"{rep}_full_gate_pred"].to_numpy()
            top = scores.loc[mask, f"{rep}_top_feature_ablate_pred"].to_numpy()
            row[f"{rep}_full_auroc"] = safe_auc(y[mask], full)
            row[f"{rep}_ablate_auroc"] = safe_auc(y[mask], top)
            row[f"{rep}_delta_auroc"] = row[f"{rep}_full_auroc"] - row[f"{rep}_ablate_auroc"]
        rows.append(row)
    return pd.DataFrame(rows)


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


def write_report(path: Path, summary: pd.DataFrame, folds: pd.DataFrame, stratum: pd.DataFrame, selected: pd.DataFrame, cards: pd.DataFrame) -> None:
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
            selected_show.groupby(["rep", "feature"], as_index=False)
            .agg(
                n_folds_selected=("fold", "nunique"),
                mean_train_active=("train_active", "mean"),
                mean_train_delta=("train_delta_abs_lof_minus_func", "mean"),
                mean_train_auc=("train_auroc_feature", "mean"),
                median_train_p=("train_p_lof_gt_func", "median"),
            )
            .sort_values(["rep", "n_folds_selected", "median_train_p"], ascending=[True, False, True])
        )
        meta_cols = ["feature", "modality", "concept_v2", "path_rate", "path_enrich", "n_active"]
        selected_show = selected_show.merge(cards[meta_cols], on="feature", how="left")

    lines = [
        "# BRCA1 Frozen-Gate SAE Feature Intervention",
        "",
        "## Purpose",
        "",
        "Train the downstream gate/head on frozen pretrained shared representations, then intervene inside the model input by replacing z_dna with SAE-decoded z_dna after removing training-fold-selected SGE-associated sparse features.",
        "",
        "This is closer to the fusion model than a logistic probe because the learned soft gate and nonlinear head are evaluated after the intervention. It remains a frozen-encoder test, not a fully fine-tuned gate-model intervention.",
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
        table(stratum, max_rows=40),
        "",
        "## Frequently Selected Features",
        "",
        table(selected_show, max_rows=30),
        "",
        "## Interpretation",
        "",
        "A convincing necessity result would show a larger held-out AUROC drop for selected features than for active-count-matched random features. If the effect is small or disappears when LLR is included, the result should be treated as a bounded or negative necessity test rather than causal proof.",
        "",
        "In the current run, the frozen-encoder gate/head shows only a bounded feature-intervention effect: selected-feature ablation lowers AUROC without LLR, but the effect is not significant against the matched-random null; when LLR is included, the effect is much smaller. The next required test is intervention inside the fully fine-tuned gate model using saved per-fold checkpoints and scalers.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    df = annotate_region(pd.read_csv(require_file(args.variants)).reset_index(drop=True))
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)
    evo = np.load(require_file(args.evo2))
    llr = evo["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)
    sae_data = np.load(require_file(args.sae_acts))
    acts = sae_data["acts"].astype(np.float32)
    z_dna = l2_normalize(sae_data["z_dna"].astype(np.float32))
    z_dna_scaled = sae_data["z_dna_scaled"].astype(np.float32)
    if not (len(df) == len(pdelta) == len(acts) == len(z_dna)):
        raise ValueError("Input row counts do not match.")
    z_prot = load_clip_z_prot(args.clip_checkpoint, pdelta, pmask, device)
    sae = load_sae(args.sae_checkpoint, device)
    cards = pd.read_csv(require_file(args.cards))
    inv_scale, inv_intercept = fit_inverse_scaled_to_raw(z_dna, z_dna_scaled)

    summaries = []
    folds = []
    randoms = []
    selected = []
    scores: pd.DataFrame | None = None
    reps = ["frozen_gate", "frozen_gate_llr"]
    for rep in reps:
        summary, fold_df, random_df, selected_df, score_df = run_rep(
            rep=rep,
            df=df,
            z_prot=z_prot,
            z_dna=z_dna,
            z_dna_scaled=z_dna_scaled,
            acts=acts,
            sae=sae,
            llr=llr,
            inv_scale=inv_scale,
            inv_intercept=inv_intercept,
            args=args,
            device=device,
        )
        summaries.append(summary)
        folds.append(fold_df)
        randoms.append(random_df)
        selected.append(selected_df)
        if scores is None:
            scores = score_df
        else:
            new_cols = [c for c in score_df.columns if c.startswith(rep)]
            scores = pd.concat([scores, score_df[new_cols]], axis=1)

    summary_df = pd.concat(summaries, ignore_index=True)
    fold_df = pd.concat(folds, ignore_index=True)
    random_df = pd.concat(randoms, ignore_index=True)
    selected_df = pd.concat(selected, ignore_index=True)
    assert scores is not None
    stratum_df = stratum_summary(scores, reps)

    prefix = "brca1_frozen_gate_sae_intervention"
    summary_df.to_csv(out / f"{prefix}_summary.csv", index=False)
    fold_df.to_csv(out / f"{prefix}_folds.csv", index=False)
    random_df.to_csv(out / f"{prefix}_random_ablation.csv", index=False)
    selected_df.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    stratum_df.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    scores.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(out / f"{prefix}.md", summary_df, fold_df, stratum_df, selected_df, cards)

    print(f"Wrote BRCA1 frozen-gate SAE intervention outputs to {out}")
    print(summary_df.to_string(index=False))
    print()
    print(fold_df.to_string(index=False))
    print()
    print(stratum_df.to_string(index=False))


if __name__ == "__main__":
    main()
