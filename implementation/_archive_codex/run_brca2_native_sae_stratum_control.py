#!/usr/bin/env python3
"""Run original-style BRCA2 native-SAE controls inside candidate strata.

The focused-strata screen uses fixed predictions and is only descriptive. This
script performs the stricter check: within each candidate stratum and fold, it
selects SAE features on the training subset, evaluates targeted ablation on the
test subset, and compares against matched random features plus label-permuted
feature selection.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_brca1_native_finetuned_sae_intervention import (  # noqa: E402
    decode_acts,
    encode_all,
    evaluate_ablation,
    make_model,
    matched_bottom_features,
    matched_random_sets,
    model_forward,
    permuted_label_features,
    predict_with_zd_override,
    require_file,
    safe_auc,
    safe_auprc,
    score_native_features,
    standardize,
    top_features,
    train_native_sae,
)


OUT_DIR = Path("results/interpretability_applications")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--variants", type=Path, default=repo / "data/variant/brca2/brca2_variants.csv")
    parser.add_argument("--esm", type=Path, default=repo / "results/variant/brca2_esm_delta.npz")
    parser.add_argument("--evo2", type=Path, default=repo / "results/variant/brca2_evo2.npz")
    parser.add_argument("--fold-artifacts", type=Path, default=repo / "results/brca2_gate_analysis/fold_artifacts")
    parser.add_argument("--output-dir", type=Path, default=repo / OUT_DIR)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "candidate name to run. Defaults to protein_high_dna_low and ob1_both_high. "
            "Available: coding, non_missense, protein_high_dna_low, ob1_both_high, "
            "tier3_vus_conflicting, actionable_review_candidate"
        ),
    )
    parser.add_argument("--output-prefix", default="brca2_native_sae_stratum_control")
    parser.add_argument("--top-k-features", type=int, default=32)
    parser.add_argument("--min-active", type=int, default=5)
    parser.add_argument("--random-sets", type=int, default=50)
    parser.add_argument("--permutation-sets", type=int, default=20)
    parser.add_argument("--sae-hidden", type=int, default=512)
    parser.add_argument("--sae-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1", type=float, default=1e-5)
    parser.add_argument("--random-seed", type=int, default=20260601)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-train-per-label", type=int, default=5)
    parser.add_argument("--min-test-per-label", type=int, default=2)
    parser.add_argument("--no-normalize-recon", action="store_true")
    return parser.parse_args()


def load_context(repo: Path, variants: Path) -> pd.DataFrame:
    df = pd.read_csv(require_file(variants)).reset_index(drop=True)
    out = repo / OUT_DIR
    disc = out / "brca2_llr_esm_discordance_missense_scores.csv"
    if disc.exists():
        d = pd.read_csv(disc)
        keep = ["id", "discordance_category", "dna_percentile", "protein_percentile"]
        df = df.merge(d[keep].drop_duplicates("id"), how="left", on="id")
    df["discordance_category"] = df.get("discordance_category", pd.Series(index=df.index, dtype=object)).fillna(
        "not_missense_or_unscored"
    )
    cand = out / "brca2_clinvar_interpretability_candidates.csv"
    if cand.exists():
        c = pd.read_csv(cand)
        keep = ["id", "review_tier", "review_direction"]
        df = df.merge(c[keep].drop_duplicates("id"), how="left", on="id")
    df["review_tier"] = df.get("review_tier", pd.Series(index=df.index, dtype=object)).fillna(
        "not_clinvar_review_candidate"
    )
    df["review_direction"] = df.get("review_direction", pd.Series(index=df.index, dtype=object)).fillna(
        "not_clinvar_review_candidate"
    )
    return df


def candidate_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "coding": df["vtype"].eq("coding").to_numpy(),
        "non_missense": (~df["is_missense"].astype(bool)).to_numpy(),
        "protein_high_dna_low": df["discordance_category"].eq("protein_high_dna_low").to_numpy(),
        "ob1_both_high": (
            df["brca2_domain"].eq("CTDB_OB1_2682_2794") & df["discordance_category"].eq("both_high")
        ).to_numpy(),
        "tier3_vus_conflicting": df["review_tier"].eq("tier3_other_vus_conflicting").to_numpy(),
        "actionable_review_candidate": df["review_direction"].isin(["pathogenic_review", "benign_review"]).to_numpy(),
    }


def label_ok(y: np.ndarray, idx: np.ndarray, min_per_label: int) -> bool:
    if len(idx) == 0:
        return False
    yy = y[idx].astype(int)
    return int((yy == 0).sum()) >= min_per_label and int((yy == 1).sum()) >= min_per_label


def run_candidate(
    name: str,
    mask: np.ndarray,
    args: argparse.Namespace,
    df: pd.DataFrame,
    y: np.ndarray,
    pdelta: np.ndarray,
    pmask: np.ndarray,
    ddelta: np.ndarray,
    llr: np.ndarray,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    normalize_recon = not args.no_normalize_recon
    fold_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for fold in range(5):
        ckpt_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}.pt")
        scaler_path = require_file(args.fold_artifacts / f"variant_fusion_fold{fold}_scalers.npz")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scalers = np.load(scaler_path)
        tr0 = scalers["train_idx"].astype(int)
        va0 = scalers["val_idx"].astype(int)
        te0 = scalers["test_idx"].astype(int)
        tr = tr0[mask[tr0]]
        va = va0[mask[va0]]
        te = te0[mask[te0]]
        row_base = {
            "candidate": name,
            "fold": fold,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "n_train_lof": int(y[tr].sum()) if len(tr) else 0,
            "n_val_lof": int(y[va].sum()) if len(va) else 0,
            "n_test_lof": int(y[te].sum()) if len(te) else 0,
        }
        if not label_ok(y, tr, args.min_train_per_label) or not label_ok(y, te, args.min_test_per_label):
            fold_rows.append({**row_base, "status": "skipped_label_or_size"})
            continue
        if len(va) < max(4, args.min_train_per_label) or len(np.unique(y[va])) < 2:
            # Keep validation in-stratum when possible; otherwise borrow a small
            # deterministic split from the training subset for SAE early stopping.
            rng = np.random.default_rng(args.random_seed + 1000 + fold)
            order = rng.permutation(tr)
            va = order[: max(4, min(len(order) // 5, 32))]
            tr = order[max(4, min(len(order) // 5, 32)) :]
            if not label_ok(y, tr, args.min_train_per_label):
                fold_rows.append({**row_base, "status": "skipped_after_val_split"})
                continue

        model = make_model(ckpt, device)
        xp_all = standardize(pdelta, scalers["prot_mean"], scalers["prot_scale"])
        xd_all = standardize(ddelta, scalers["dna_mean"], scalers["dna_scale"])
        xs_all = standardize(llr, scalers["scalar_mean"], scalers["scalar_scale"])
        pred_full, zp_all, zd_all, _ = model_forward(model, xp_all, xd_all, pmask, xs_all, device)

        sae, sae_metrics = train_native_sae(zd_all, tr, va, args, device, args.random_seed + fold)
        acts_all = encode_all(sae, zd_all, device)
        scored = score_native_features(acts_all, y, tr, args.min_active)
        selected = top_features(scored, args.top_k_features)
        if len(selected) == 0:
            fold_rows.append({**row_base, "status": "skipped_no_selected_features"})
            continue

        selected_df = scored[scored["feature"].isin(selected)].copy()
        for item in selected_df.to_dict("records"):
            selected_rows.append({**item, "candidate": name, "fold": fold})

        z_recon_te = decode_acts(sae, acts_all[te], device, normalize_recon)
        pred_recon_te, _ = predict_with_zd_override(model, zp_all[te], z_recon_te, xs_all[te], device)
        acts_top_te = acts_all[te].copy()
        acts_top_te[:, selected] = 0.0
        z_top_te = decode_acts(sae, acts_top_te, device, normalize_recon)
        pred_top_te, _ = predict_with_zd_override(model, zp_all[te], z_top_te, xs_all[te], device)

        recon_auc = safe_auc(y[te], pred_recon_te)
        recon_ap = safe_auprc(y[te], pred_recon_te)
        top_auc = safe_auc(y[te], pred_top_te)
        top_ap = safe_auprc(y[te], pred_top_te)
        fold_rows.append(
            {
                **row_base,
                "status": "tested",
                "n_selected_features": int(len(selected)),
                "original_auc": safe_auc(y[te], pred_full[te]),
                "native_sae_recon_auc": recon_auc,
                "top_feature_ablate_auc": top_auc,
                "delta_auc_recon_minus_top_ablate": recon_auc - top_auc,
                "original_auprc": safe_auprc(y[te], pred_full[te]),
                "native_sae_recon_auprc": recon_ap,
                "top_feature_ablate_auprc": top_ap,
                "delta_auprc_recon_minus_top_ablate": recon_ap - top_ap,
                "selected_features": ";".join(map(str, selected.tolist())),
                **sae_metrics,
            }
        )

        bottom = matched_bottom_features(selected, scored, args.random_seed + 10_000 + fold)
        if len(bottom):
            auc, ap, delta_auc, delta_ap = evaluate_ablation(
                sae, acts_all, bottom, te, y, model, zp_all[te], xs_all[te], recon_auc, recon_ap, device, normalize_recon
            )
            control_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "control": "bottom_abs_delta_active_matched",
                    "set_id": 0,
                    "n_features": len(bottom),
                    "ablate_auc": auc,
                    "delta_auc_recon_minus_ablate": delta_auc,
                    "ablate_auprc": ap,
                    "delta_auprc_recon_minus_ablate": delta_ap,
                }
            )

        random_sets = matched_random_sets(
            selected, scored, args.top_k_features, args.random_sets, args.random_seed + 20_000 + fold
        )
        for set_id, feats in enumerate(random_sets):
            auc, ap, delta_auc, delta_ap = evaluate_ablation(
                sae, acts_all, feats, te, y, model, zp_all[te], xs_all[te], recon_auc, recon_ap, device, normalize_recon
            )
            control_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "control": "random_active_matched",
                    "set_id": set_id,
                    "n_features": len(feats),
                    "ablate_auc": auc,
                    "delta_auc_recon_minus_ablate": delta_auc,
                    "ablate_auprc": ap,
                    "delta_auprc_recon_minus_ablate": delta_ap,
                }
            )

        for set_id in range(args.permutation_sets):
            feats = permuted_label_features(
                acts_all, y, tr, args.top_k_features, args.min_active, args.random_seed + 30_000 + 100 * fold + set_id
            )
            auc, ap, delta_auc, delta_ap = evaluate_ablation(
                sae, acts_all, feats, te, y, model, zp_all[te], xs_all[te], recon_auc, recon_ap, device, normalize_recon
            )
            control_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "control": "label_permuted_selection",
                    "set_id": set_id,
                    "n_features": len(feats),
                    "ablate_auc": auc,
                    "delta_auc_recon_minus_ablate": delta_auc,
                    "ablate_auprc": ap,
                    "delta_auprc_recon_minus_ablate": delta_ap,
                }
            )

    return fold_rows, control_rows, selected_rows


def summarize(folds: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, f in folds.groupby("candidate"):
        tested = f[f["status"].eq("tested")].copy()
        finite_tested = tested[np.isfinite(tested["delta_auc_recon_minus_top_ablate"])].copy()
        c = controls[controls["candidate"].eq(candidate)].copy()
        top = finite_tested["delta_auc_recon_minus_top_ablate"].dropna()
        top_mean = float(top.mean()) if len(top) else float("nan")
        random = c.loc[c["control"].eq("random_active_matched"), "delta_auc_recon_minus_ablate"].dropna()
        perm = c.loc[c["control"].eq("label_permuted_selection"), "delta_auc_recon_minus_ablate"].dropna()
        bottom = c.loc[c["control"].eq("bottom_abs_delta_active_matched"), "delta_auc_recon_minus_ablate"].dropna()
        rows.append(
            {
                "candidate": candidate,
                "n_total": int(f["n_test"].sum()),
                "n_tested_folds": int(len(finite_tested)),
                "mean_fold_delta_auc_recon_minus_top_ablate": top_mean,
                "mean_fold_delta_auprc_recon_minus_top_ablate": float(
                    finite_tested["delta_auprc_recon_minus_top_ablate"].mean()
                )
                if len(finite_tested)
                else float("nan"),
                "mean_random_delta_auc": float(random.mean()) if len(random) else float("nan"),
                "empirical_p_random_delta_ge_top_mean": float(((random >= top_mean).sum() + 1) / (len(random) + 1))
                if len(random)
                else float("nan"),
                "mean_bottom_delta_auc": float(bottom.mean()) if len(bottom) else float("nan"),
                "empirical_p_bottom_delta_ge_top_mean": float(((bottom >= top_mean).sum() + 1) / (len(bottom) + 1))
                if len(bottom)
                else float("nan"),
                "mean_label_permuted_delta_auc": float(perm.mean()) if len(perm) else float("nan"),
                "empirical_p_label_permuted_delta_ge_top_mean": float(((perm >= top_mean).sum() + 1) / (len(perm) + 1))
                if len(perm)
                else float("nan"),
                "passes_original_style_gate": bool(
                    np.isfinite(top_mean)
                    and top_mean > 0
                    and len(random)
                    and len(perm)
                    and ((random >= top_mean).sum() + 1) / (len(random) + 1) <= 0.05
                    and ((perm >= top_mean).sum() + 1) / (len(perm) + 1) <= 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    df = load_context(repo, args.variants)
    y = df["label"].to_numpy(dtype=int)
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)[:, None]
    evo = np.load(require_file(args.evo2))
    ddelta = evo["edelta"].astype(np.float32)
    llr = evo["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)[:, None]

    masks = candidate_masks(df)
    candidates = args.candidate or ["protein_high_dna_low", "ob1_both_high"]
    unknown = [c for c in candidates if c not in masks]
    if unknown:
        raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")

    fold_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for name in candidates:
        f, c, s = run_candidate(name, masks[name], args, df, y, pdelta, pmask, ddelta, llr, device)
        fold_rows.extend(f)
        control_rows.extend(c)
        selected_rows.extend(s)

    folds = pd.DataFrame(fold_rows)
    controls = pd.DataFrame(control_rows)
    selected = pd.DataFrame(selected_rows)
    summary = summarize(folds, controls)

    prefix = args.output_prefix
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    folds.to_csv(out / f"{prefix}_folds.csv", index=False)
    controls.to_csv(out / f"{prefix}_controls.csv", index=False)
    selected.to_csv(out / f"{prefix}_selected_features.csv", index=False)
    print(f"wrote {out / f'{prefix}_summary.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
