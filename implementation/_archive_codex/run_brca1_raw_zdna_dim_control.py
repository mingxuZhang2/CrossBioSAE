#!/usr/bin/env python3
"""
BRCA1 raw fine-tuned z_dna dimension intervention control.

This control asks whether the fold-native SAE necessity result is merely a
generic consequence of deleting any supervised directions in the fine-tuned
DNA representation. For each saved BRCA1 fine-tuned gate/head fold, it:

1. extracts fold-specific fine-tuned z_dna,
2. selects raw z_dna dimensions associated with SGE LOF labels using train
   labels only,
3. zeros those raw dimensions in held-out test z_dna before the learned
   gate/head,
4. compares against random raw dimensions and label-permuted raw-dimension
   selection.

This is a control, not an interpretability layer: raw dimensions are dense and
not expected to be biologically nameable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP, VariantFusionHead

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_brca1_finetuned_gate_sae_intervention import (  # noqa: E402
    annotate_region,
    model_forward,
    predict_with_zd_override,
    require_file,
    safe_auc,
    safe_auprc,
    standardize,
    stratum_summary,
)
from run_brca1_native_finetuned_sae_intervention import normalize_rows, parse_int_list  # noqa: E402


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
    parser.add_argument("--top-k-dims", type=int, default=32)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--permutation-sets", type=int, default=20)
    parser.add_argument("--dose-k-values", default="8,16,32,64")
    parser.add_argument("--random-seed", type=int, default=630)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize z_dna after zeroing raw dimensions.")
    return parser.parse_args()


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


def score_dims(zd: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> pd.DataFrame:
    rows = []
    yy = y[train_idx].astype(int)
    abs_train = np.abs(zd[train_idx])
    for dim in range(abs_train.shape[1]):
        score = abs_train[:, dim]
        lof = score[yy == 1]
        func = score[yy == 0]
        delta = float(lof.mean() - func.mean())
        rows.append(
            {
                "dim": int(dim),
                "train_delta_abs_lof_minus_func": delta,
                "train_abs_delta": abs(delta),
                "train_mean_abs": float(score.mean()),
                "train_auroc_dim": safe_auc(yy, score),
                "train_p_lof_gt_func": float(mannwhitneyu(lof, func, alternative="greater").pvalue),
            }
        )
    return pd.DataFrame(rows)


def top_dims(scored: pd.DataFrame, top_k: int) -> np.ndarray:
    use = scored[scored["train_delta_abs_lof_minus_func"] > 0].copy()
    if use.empty:
        return np.array([], dtype=int)
    use = use.sort_values(["train_p_lof_gt_func", "train_auroc_dim"], ascending=[True, False])
    return use["dim"].head(top_k).to_numpy(dtype=int)


def permuted_dims(zd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, top_k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    y_perm[train_idx] = rng.permutation(y_perm[train_idx])
    return top_dims(score_dims(zd, y_perm, train_idx), top_k)


def random_dim_sets(n_dim: int, top_k: int, n_sets: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(np.arange(n_dim), size=min(top_k, n_dim), replace=False).astype(int) for _ in range(n_sets)]


def predict_zeroed_dims(
    model: VariantFusionHead,
    zp_te: np.ndarray,
    zd_te: np.ndarray,
    xs_te: np.ndarray,
    dims: np.ndarray,
    device: str,
    normalize: bool,
) -> np.ndarray:
    zd = zd_te.copy()
    zd[:, dims] = 0.0
    if normalize:
        zd = normalize_rows(zd)
    pred, _ = predict_with_zd_override(model, zp_te, zd.astype(np.float32), xs_te, device)
    return pred


def write_report(path: Path, summary: pd.DataFrame, folds: pd.DataFrame, randoms: pd.DataFrame, perms: pd.DataFrame, dose: pd.DataFrame, strata: pd.DataFrame) -> None:
    def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if df.empty:
            return "No rows."
        show = df.copy() if max_rows is None else df.head(max_rows).copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        return show.to_markdown(index=False)

    lines = [
        "# BRCA1 Raw z_dna Dimension Intervention Control",
        "",
        "## Purpose",
        "",
        "Test whether supervised deletion of dense raw `z_dna` dimensions produces the same kind of held-out gate/head drop as native SAE feature ablation. Raw dimensions are not biologically nameable; this is a control for generic supervised representation perturbation.",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Fold-Level Raw-Dim Intervention",
        "",
        table(folds),
        "",
        "## Random Raw-Dim Controls",
        "",
        table(randoms, max_rows=40),
        "",
        "## Label-Permuted Raw-Dim Controls",
        "",
        table(perms, max_rows=40),
        "",
        "## Dose Response",
        "",
        table(dose, max_rows=40),
        "",
        "## Stratum Summary",
        "",
        table(strata, max_rows=40),
        "",
        "## Interpretation",
        "",
        "If raw-dimension ablation matches native SAE ablation, the native result may partly reflect generic supervised directions in `z_dna`. If raw-dimension ablation is weaker or less controlled, the SAE result is more specific. In either case, raw dimensions are dense coordinates and should not replace sparse-feature biological interpretation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    normalize = not args.no_normalize
    dose_k_values = parse_int_list(args.dose_k_values)

    df = annotate_region(pd.read_csv(require_file(args.variants)).reset_index(drop=True))
    y = df["label"].to_numpy(dtype=int)
    esm = np.load(require_file(args.esm))
    pdelta = esm["pdelta"].astype(np.float32)
    pmask = esm["pmask"].astype(np.float32)[:, None]
    evo = np.load(require_file(args.evo2))
    ddelta = evo["edelta"].astype(np.float32)
    llr = evo["llr"].astype(np.float32)
    llr = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)[:, None]

    original_pred = np.zeros(len(df), dtype=np.float32)
    top_pred = np.zeros(len(df), dtype=np.float32)
    fold_rows = []
    random_rows = []
    perm_rows = []
    dose_rows = []

    for fold in range(5):
        ckpt = torch.load(require_file(args.fold_artifacts / f"variant_fusion_fold{fold}.pt"), map_location="cpu", weights_only=False)
        scalers = np.load(require_file(args.fold_artifacts / f"variant_fusion_fold{fold}_scalers.npz"))
        tr = scalers["train_idx"].astype(int)
        te = scalers["test_idx"].astype(int)
        model = make_model(ckpt, device)

        xp_all = standardize(pdelta, scalers["prot_mean"], scalers["prot_scale"])
        xd_all = standardize(ddelta, scalers["dna_mean"], scalers["dna_scale"])
        xs_all = standardize(llr, scalers["scalar_mean"], scalers["scalar_scale"])
        pred_full, zp_all, zd_all, _ = model_forward(model, xp_all, xd_all, pmask, xs_all, device)
        original_pred[te] = pred_full[te]

        scored = score_dims(zd_all, y, tr)
        dims = top_dims(scored, args.top_k_dims)
        pred_top = predict_zeroed_dims(model, zp_all[te], zd_all[te], xs_all[te], dims, device, normalize)
        top_pred[te] = pred_top
        orig_auc = safe_auc(y[te], pred_full[te])
        top_auc = safe_auc(y[te], pred_top)
        orig_ap = safe_auprc(y[te], pred_full[te])
        top_ap = safe_auprc(y[te], pred_top)
        fold_rows.append(
            {
                "fold": fold,
                "n_test": len(te),
                "n_selected_dims": len(dims),
                "original_auc": orig_auc,
                "top_raw_dim_ablate_auc": top_auc,
                "delta_auc_original_minus_top_raw_dim_ablate": orig_auc - top_auc,
                "original_auprc": orig_ap,
                "top_raw_dim_ablate_auprc": top_ap,
                "delta_auprc_original_minus_top_raw_dim_ablate": orig_ap - top_ap,
                "selected_dims": ";".join(map(str, dims.tolist())),
            }
        )

        for dose_k in dose_k_values:
            dose_dims = top_dims(scored, dose_k)
            pred = predict_zeroed_dims(model, zp_all[te], zd_all[te], xs_all[te], dose_dims, device, normalize)
            dose_rows.append(
                {
                    "fold": fold,
                    "dose_k": len(dose_dims),
                    "requested_k": dose_k,
                    "ablate_auc": safe_auc(y[te], pred),
                    "delta_auc_original_minus_ablate": orig_auc - safe_auc(y[te], pred),
                    "ablate_auprc": safe_auprc(y[te], pred),
                    "delta_auprc_original_minus_ablate": orig_ap - safe_auprc(y[te], pred),
                    "dims": ";".join(map(str, dose_dims.tolist())),
                }
            )

        for i, rand_dims in enumerate(random_dim_sets(zd_all.shape[1], args.top_k_dims, args.random_sets, args.random_seed + fold)):
            pred = predict_zeroed_dims(model, zp_all[te], zd_all[te], xs_all[te], rand_dims, device, normalize)
            random_rows.append(
                {
                    "fold": fold,
                    "random_set": i,
                    "ablate_auc": safe_auc(y[te], pred),
                    "delta_auc_original_minus_ablate": orig_auc - safe_auc(y[te], pred),
                    "ablate_auprc": safe_auprc(y[te], pred),
                    "delta_auprc_original_minus_ablate": orig_ap - safe_auprc(y[te], pred),
                }
            )

        for i in range(args.permutation_sets):
            dims_perm = permuted_dims(zd_all, y, tr, args.top_k_dims, args.random_seed + 20_000 + 100 * fold + i)
            pred = predict_zeroed_dims(model, zp_all[te], zd_all[te], xs_all[te], dims_perm, device, normalize)
            perm_rows.append(
                {
                    "fold": fold,
                    "permutation_set": i,
                    "ablate_auc": safe_auc(y[te], pred),
                    "delta_auc_original_minus_ablate": orig_auc - safe_auc(y[te], pred),
                    "ablate_auprc": safe_auprc(y[te], pred),
                    "delta_auprc_original_minus_ablate": orig_ap - safe_auprc(y[te], pred),
                    "dims": ";".join(map(str, dims_perm.tolist())),
                }
            )

    folds = pd.DataFrame(fold_rows)
    randoms = pd.DataFrame(random_rows)
    perms = pd.DataFrame(perm_rows)
    dose = pd.DataFrame(dose_rows)
    top_mean = float(folds["delta_auc_original_minus_top_raw_dim_ablate"].mean()) if not folds.empty else float("nan")
    rand = randoms["delta_auc_original_minus_ablate"].dropna()
    perm = perms["delta_auc_original_minus_ablate"].dropna()
    summary_row = {
        "n": len(df),
        "original_auroc": safe_auc(y, original_pred),
        "top_raw_dim_ablate_auroc": safe_auc(y, top_pred),
        "delta_auroc_original_minus_top_raw_dim_ablate": safe_auc(y, original_pred) - safe_auc(y, top_pred),
        "original_auprc": safe_auprc(y, original_pred),
        "top_raw_dim_ablate_auprc": safe_auprc(y, top_pred),
        "delta_auprc_original_minus_top_raw_dim_ablate": safe_auprc(y, original_pred) - safe_auprc(y, top_pred),
        "mean_fold_delta_auc_original_minus_top_raw_dim_ablate": top_mean,
        "mean_random_raw_dim_delta_auc": float(rand.mean()) if len(rand) else float("nan"),
        "sd_random_raw_dim_delta_auc": float(rand.std()) if len(rand) else float("nan"),
        "empirical_p_random_raw_dim_delta_ge_top_mean": float(((rand >= top_mean).sum() + 1) / (len(rand) + 1)) if len(rand) else float("nan"),
        "mean_label_permuted_raw_dim_delta_auc": float(perm.mean()) if len(perm) else float("nan"),
        "sd_label_permuted_raw_dim_delta_auc": float(perm.std()) if len(perm) else float("nan"),
        "empirical_p_label_permuted_raw_dim_delta_ge_top_mean": float(((perm >= top_mean).sum() + 1) / (len(perm) + 1)) if len(perm) else float("nan"),
        "normalize_after_zero": normalize,
    }
    if not dose.empty:
        for dose_k, group in dose.groupby("dose_k"):
            summary_row[f"mean_dose_{int(dose_k)}_delta_auc"] = float(group["delta_auc_original_minus_ablate"].mean())
    summary = pd.DataFrame([summary_row])

    scores = df.copy()
    scores["original_pred"] = original_pred
    scores["sae_recon_pred"] = original_pred
    scores["top_feature_ablate_pred"] = top_pred
    strata = stratum_summary(scores)

    prefix = "brca1_raw_zdna_dim_control"
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    folds.to_csv(out / f"{prefix}_folds.csv", index=False)
    randoms.to_csv(out / f"{prefix}_random.csv", index=False)
    perms.to_csv(out / f"{prefix}_label_permuted.csv", index=False)
    dose.to_csv(out / f"{prefix}_dose_response.csv", index=False)
    strata.to_csv(out / f"{prefix}_stratum_summary.csv", index=False)
    write_report(out / f"{prefix}.md", summary, folds, randoms, perms, dose, strata)
    print(f"Wrote BRCA1 raw z_dna dimension control outputs to {out}")
    print(summary.to_string(index=False))
    print()
    print(folds.to_string(index=False))
    print()
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
