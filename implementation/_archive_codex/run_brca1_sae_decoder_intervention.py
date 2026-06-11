#!/usr/bin/env python3
"""
BRCA1 SAE decoder-level feature intervention.

This analysis tests whether BRCA1 SGE-associated genome-wide SAE features are
useful after being decoded back into the shared DNA representation. In each
GroupKFold split, features are selected using only the training fold, decoded
representations with those features removed are evaluated on the held-out fold,
and the AUROC drop is compared with active-count-matched random feature sets.

This is stronger than an activation-only probe ablation because the downstream
classifier sees a shared-space representation reconstructed by the SAE decoder.
It is still not a causal ablation inside the original fine-tuned gate model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "interpretability_applications",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--random-sets", type=int, default=100)
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


def load_clip_protein_encoder(checkpoint: Path, pdelta: np.ndarray, device: str) -> np.ndarray:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from contrastive_pretrain import CrossModalCLIP

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
    return np.concatenate(out, axis=0).astype(np.float32)


def load_sae(checkpoint: Path, device: str) -> TopKSAE:
    state = torch.load(require_file(checkpoint), map_location="cpu", weights_only=True)
    d_hidden, d_in = state["encoder.weight"].shape
    model = TopKSAE(d_in=d_in, d_hidden=d_hidden, k=32)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def decode_acts(sae: TopKSAE, acts: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(acts), batch_size):
            h = torch.tensor(acts[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(sae.decoder(h).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def select_train_features(
    acts: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    min_active: int,
) -> tuple[np.ndarray, pd.DataFrame]:
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
        try:
            pval = float(mannwhitneyu(lof, func, alternative="greater").pvalue)
        except ValueError:
            pval = float("nan")
        rows.append(
            {
                "feature": int(feature),
                "train_active": int(active_counts[feature]),
                "train_mean_abs_lof": float(lof.mean()),
                "train_mean_abs_func": float(func.mean()),
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


def matched_random_sets(
    selected: np.ndarray,
    scored: pd.DataFrame,
    top_k: int,
    random_sets: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if scored.empty or len(selected) == 0:
        return []
    counts = scored.set_index("feature")["train_active"].to_dict()
    eligible = scored[~scored["feature"].isin(selected)].copy()
    if eligible.empty:
        return []
    pool = eligible["feature"].to_numpy(dtype=int)
    sets = []
    for _ in range(random_sets):
        chosen: list[int] = []
        used: set[int] = set()
        for feat in selected:
            target = counts[int(feat)]
            eligible_pool = eligible.copy()
            eligible_pool["dist"] = (eligible_pool["train_active"] - target).abs()
            candidates = eligible_pool.sort_values("dist")["feature"].head(max(50, top_k * 4)).to_numpy(dtype=int)
            candidates = np.array([c for c in candidates if c not in used], dtype=int)
            if len(candidates) == 0:
                candidates = np.array([c for c in pool if c not in used], dtype=int)
            if len(candidates) == 0:
                break
            pick = int(rng.choice(candidates))
            used.add(pick)
            chosen.append(pick)
        if len(chosen) < min(top_k, len(pool)):
            rest = np.array([c for c in pool if c not in used], dtype=int)
            need = min(top_k, len(pool)) - len(chosen)
            if len(rest) >= need:
                chosen.extend(rng.choice(rest, size=need, replace=False).astype(int).tolist())
        sets.append(np.array(chosen[:top_k], dtype=int))
    return sets


def make_features(
    z_prot: np.ndarray,
    z_dna_recon: np.ndarray,
    pmask: np.ndarray,
    llr: np.ndarray,
    include_llr: bool,
) -> np.ndarray:
    pieces = [z_prot * pmask[:, None].astype(np.float32), z_dna_recon, pmask[:, None].astype(np.float32)]
    if include_llr:
        pieces.append(llr[:, None].astype(np.float32))
    return np.hstack(pieces).astype(np.float32)


def eval_intervention(
    df: pd.DataFrame,
    z_prot: np.ndarray,
    acts: np.ndarray,
    sae: TopKSAE,
    pmask: np.ndarray,
    llr: np.ndarray,
    include_llr: bool,
    top_k: int,
    min_active: int,
    random_sets: int,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    y = df["label"].to_numpy(dtype=int)
    groups = df["pos_hg19"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    pred_full = np.zeros(len(df), dtype=float)
    pred_top = np.zeros(len(df), dtype=float)
    fold_rows = []
    random_rows = []
    selected_rows = []

    z_full = decode_acts(sae, acts, device)
    rep_name = "protein_dna_llr" if include_llr else "protein_dna"

    for fold, (tr, te) in enumerate(gkf.split(acts, y, groups)):
        selected, scored = select_train_features(acts, y, tr, top_k, min_active)
        if len(selected) == 0:
            continue
        selected_meta = scored[scored["feature"].isin(selected)].copy()
        selected_meta["fold"] = fold
        selected_meta["rep"] = rep_name
        selected_rows.append(selected_meta)

        acts_top_te = acts[te].copy()
        acts_top_te[:, selected] = 0.0
        z_top = decode_acts(sae, acts_top_te, device)

        x_train = make_features(z_prot[tr], z_full[tr], pmask[tr], llr[tr], include_llr)
        x_test_full = make_features(z_prot[te], z_full[te], pmask[te], llr[te], include_llr)
        x_test_top = make_features(z_prot[te], z_top, pmask[te], llr[te], include_llr)

        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(max_iter=3000, C=0.2, solver="lbfgs", class_weight="balanced")
        clf.fit(scaler.transform(x_train), y[tr])
        pred_full[te] = clf.predict_proba(scaler.transform(x_test_full))[:, 1]
        pred_top[te] = clf.predict_proba(scaler.transform(x_test_top))[:, 1]

        full_auc = safe_auc(y[te], pred_full[te])
        top_auc = safe_auc(y[te], pred_top[te])
        full_ap = safe_auprc(y[te], pred_full[te])
        top_ap = safe_auprc(y[te], pred_top[te])
        fold_rows.append(
            {
                "rep": rep_name,
                "fold": fold,
                "n_test": len(te),
                "n_selected_features": len(selected),
                "full_recon_auc": full_auc,
                "top_feature_ablate_auc": top_auc,
                "delta_auc_top_ablate": full_auc - top_auc,
                "full_recon_auprc": full_ap,
                "top_feature_ablate_auprc": top_ap,
                "delta_auprc_top_ablate": full_ap - top_ap,
                "selected_features": ";".join(map(str, selected.tolist())),
            }
        )

        random_feature_sets = matched_random_sets(selected, scored, top_k, random_sets, seed + fold)
        for i, random_features in enumerate(random_feature_sets):
            acts_random_te = acts[te].copy()
            acts_random_te[:, random_features] = 0.0
            z_random = decode_acts(sae, acts_random_te, device)
            x_test_random = make_features(z_prot[te], z_random, pmask[te], llr[te], include_llr)
            pred_random = clf.predict_proba(scaler.transform(x_test_random))[:, 1]
            random_auc = safe_auc(y[te], pred_random)
            random_ap = safe_auprc(y[te], pred_random)
            random_rows.append(
                {
                    "rep": rep_name,
                    "fold": fold,
                    "random_set": i,
                    "random_feature_count": len(random_features),
                    "random_ablate_auc": random_auc,
                    "delta_auc_random_ablate": full_auc - random_auc,
                    "random_ablate_auprc": random_ap,
                    "delta_auprc_random_ablate": full_ap - random_ap,
                    "random_features": ";".join(map(str, random_features.tolist())),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    random_df = pd.DataFrame(random_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    summary = summarize(rep_name, y, pred_full, pred_top, fold_df, random_df)
    return summary, fold_df, random_df, selected_df, pred_full, pred_top


def summarize(
    rep_name: str,
    y: np.ndarray,
    pred_full: np.ndarray,
    pred_top: np.ndarray,
    folds: pd.DataFrame,
    randoms: pd.DataFrame,
) -> pd.DataFrame:
    top_mean = float(folds["delta_auc_top_ablate"].mean()) if not folds.empty else float("nan")
    rand = randoms["delta_auc_random_ablate"].dropna() if not randoms.empty else pd.Series(dtype=float)
    empirical_p = float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan")
    return pd.DataFrame(
        [
            {
                "rep": rep_name,
                "n": len(y),
                "full_recon_auroc": safe_auc(y, pred_full),
                "top_feature_ablate_auroc": safe_auc(y, pred_top),
                "delta_auroc_top_ablate": safe_auc(y, pred_full) - safe_auc(y, pred_top),
                "full_recon_auprc": safe_auprc(y, pred_full),
                "top_feature_ablate_auprc": safe_auprc(y, pred_top),
                "delta_auprc_top_ablate": safe_auprc(y, pred_full) - safe_auprc(y, pred_top),
                "mean_fold_delta_auc_top_ablate": top_mean,
                "mean_random_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
                "sd_random_delta_auc": float(rand.std()) if len(rand) else float("nan"),
                "empirical_p_random_delta_ge_top_mean": empirical_p,
            }
        ]
    )


def stratum_summary(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    rows = []
    strata = {
        "all": np.ones(len(df), dtype=bool),
        "missense": df["is_missense"].astype(bool).to_numpy(),
        "coding": df["vtype"].eq("coding").to_numpy(),
        "noncoding": df["vtype"].eq("noncoding").to_numpy(),
    }
    if "brca1_region" in df.columns:
        for region in df["brca1_region"].dropna().unique():
            strata[f"region:{region}"] = df["brca1_region"].eq(region).to_numpy()
    y = df["label"].to_numpy(dtype=int)
    for name, mask in strata.items():
        if mask.sum() < 40 or len(np.unique(y[mask])) < 2:
            continue
        row = {"stratum": name, "n": int(mask.sum()), "n_lof": int(y[mask].sum())}
        for col in pred_cols:
            row[f"auroc_{col}"] = safe_auc(y[mask], df.loc[mask, col].to_numpy())
            row[f"auprc_{col}"] = safe_auprc(y[mask], df.loc[mask, col].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_brca1_region(df: pd.DataFrame) -> pd.DataFrame:
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


def write_report(
    path: Path,
    summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    stratum: pd.DataFrame,
    selected: pd.DataFrame,
    cards: pd.DataFrame,
) -> None:
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
        selected_show = selected_show.merge(cards[meta_cols], on="feature", how="left").head(30)

    def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if df.empty:
            return "No rows."
        show = df.copy() if max_rows is None else df.head(max_rows).copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        return show.to_markdown(index=False)

    lines = [
        "# BRCA1 SAE Decoder-Level Feature Intervention",
        "",
        "## Purpose",
        "",
        "Select BRCA1 SGE-associated SAE features using only each training fold, remove those features from sparse activations, decode back into the shared DNA representation, and test whether a held-out downstream shared-space classifier loses SGE LOF discrimination more than under matched random feature removal.",
        "",
        "This is a representation-level intervention through the genome-wide SAE decoder. It is stronger than an activation-only probe ablation, but it is not yet a causal ablation inside the original fine-tuned gate model.",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Fold-Level Intervention",
        "",
        table(fold_summary),
        "",
        "## Stratum Summary",
        "",
        table(stratum, max_rows=40),
        "",
        "## Most Frequently Selected Features",
        "",
        table(selected_show, max_rows=30),
        "",
        "## Interpretation",
        "",
        "Interpret this table as a necessity test. If top-feature ablation causes a larger AUROC drop than active-count-matched random feature ablations, the selected sparse features carry necessary information for this decoded shared-representation classifier. If the drop is small or comparable to random features, the evidence remains limited to feature association rather than necessity.",
        "",
        "In the current run, the top-feature AUROC drop is small and does not convincingly exceed the matched random-feature null, especially when the Evo2 LLR scalar is included. This is therefore a weak/negative representation-level necessity result: it supports caution and motivates original gate-model intervention rather than a causal claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    df = pd.read_csv(require_file(args.variants)).reset_index(drop=True)
    df = annotate_brca1_region(df)
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)
    evo2 = np.load(require_file(args.evo2))
    llr = evo2["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)
    sae_npz = np.load(require_file(args.sae_acts))
    acts = sae_npz["acts"].astype(np.float32)
    if len(df) != len(acts) or len(df) != len(pdelta):
        raise ValueError("Input row counts do not match.")
    cards = pd.read_csv(require_file(args.cards))

    z_prot = load_clip_protein_encoder(args.clip_checkpoint, pdelta, device)
    sae = load_sae(args.sae_checkpoint, device)

    all_summaries = []
    all_folds = []
    all_random = []
    all_selected = []
    score_df = df.copy()
    pred_cols = []

    for include_llr in [False, True]:
        summary, folds, randoms, selected, pred_full, pred_top = eval_intervention(
            df=df,
            z_prot=z_prot,
            acts=acts,
            sae=sae,
            pmask=pmask,
            llr=llr,
            include_llr=include_llr,
            top_k=args.top_k,
            min_active=args.min_active,
            random_sets=args.random_sets,
            seed=args.random_seed,
            device=device,
        )
        rep = summary.loc[0, "rep"]
        score_df[f"{rep}_full_recon_pred"] = pred_full
        score_df[f"{rep}_top_feature_ablate_pred"] = pred_top
        pred_cols.extend([f"{rep}_full_recon_pred", f"{rep}_top_feature_ablate_pred"])
        all_summaries.append(summary)
        all_folds.append(folds)
        all_random.append(randoms)
        all_selected.append(selected)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    random_df = pd.concat(all_random, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)
    stratum_df = stratum_summary(score_df, pred_cols)

    prefix = "brca1_sae_decoder_intervention"
    summary_df.to_csv(out / f"{prefix}_summary.csv", index=False)
    fold_df.to_csv(out / f"{prefix}_folds.csv", index=False)
    random_df.to_csv(out / f"{prefix}_random_ablation.csv", index=False)
    selected_df.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    stratum_df.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    score_df.to_csv(out / f"{prefix}_variant_scores.csv", index=False)
    write_report(out / f"{prefix}.md", summary_df, fold_df, stratum_df, selected_df, cards)

    print(f"Wrote BRCA1 SAE decoder intervention outputs to {out}")
    print(summary_df.to_string(index=False))
    print()
    print(fold_df.to_string(index=False))
    print()
    print(stratum_df.to_string(index=False))


if __name__ == "__main__":
    main()
