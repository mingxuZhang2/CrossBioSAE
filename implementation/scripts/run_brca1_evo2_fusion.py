"""
BRCA1 cross-modal fusion with the REAL DNA model (Evo2-7b), replacing the weak
NT placeholder. The whole point: can fusing ESM into Evo2 beat Evo2 alone?

Reuses ESM-2 per-residue deltas from run_brca1_fusion (missense, 1280-d) and the
precomputed Evo2 features (results/variant/brca1_evo2.npz):
  - evo2 llr    : zero-shot delta log-likelihood (scalar) -> Evo2's headline score
  - evo2 edelta : window-token embedding delta (4096-d)

Evaluations (auROC, GroupKFold by genomic position; missense stratum included):
  1. Evo2 zero-shot          : roc_auc(-llr), NO training  (the number to beat)
  2. Evo2-LLR  (supervised)  : 1 feature, grouped CV
  3. Evo2-emb  (supervised)  : 4096-d, grouped CV
  4. ESM-only                : 1280-d protein delta
  5. ESM + Evo2-LLR          : protein delta + scalar  (lightweight fusion)
  6. ESM + Evo2-emb          : protein delta + 4096-d   (full fusion)

The cross-modal claim holds iff (5) or (6) > (1) and > (2)/(3).
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# reuse the exact ESM extraction + grouped CV from the NT fusion
from run_brca1_fusion import extract_protein_deltas, grouped_auroc, load_protein

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def strat_auroc(y, pred, df):
    per = {}
    for vt in ["coding", "noncoding"]:
        m = (df["vtype"] == vt).values
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            per[vt] = roc_auc_score(y[m], pred[m])
    mis = df["is_missense"].values
    if len(np.unique(y[mis])) == 2:
        per["missense"] = roc_auc_score(y[mis], pred[mis])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--protein", default="data/variant/brca1/brca1_P38398.fasta")
    ap.add_argument("--evo2", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    y = df["label"].values
    groups = df["pos_hg19"].values

    # --- Evo2 features ---
    ev = np.load(args.evo2)
    llr, edelta = ev["llr"], ev["edelta"]
    logger.info(f"evo2 llr={llr.shape} edelta={edelta.shape}; nan_llr={np.isnan(llr).sum()}")
    # impute any nan llr (windows that failed) with median so CV is well-defined
    llr_imp = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)

    # --- ESM protein deltas (cache to avoid recompute) ---
    pcache = os.path.join(args.output_dir, "brca1_esm_delta.npz")
    if os.path.exists(pcache) and np.load(pcache)["pdelta"].shape[0] == len(df):
        c = np.load(pcache); pdelta, pmask = c["pdelta"], c["pmask"]
        logger.info("loaded cached ESM deltas")
    else:
        import torch
        wt = load_protein(args.protein)
        pdelta, pmask = extract_protein_deltas(df, wt, args.device)
        np.savez_compressed(pcache, pdelta=pdelta, pmask=pmask)
    pmask_f = pmask[:, None].astype(np.float32)

    # --- configs ---
    configs = {
        "Evo2 zero-shot (no train)": ("zeroshot", -llr_imp),
        "Evo2-LLR  (supervised)":    ("sup", llr_imp[:, None]),
        "Evo2-emb  (supervised)":    ("sup", edelta),
        "ESM-only":                  ("sup", np.hstack([pdelta, pmask_f])),
        "ESM + Evo2-LLR":            ("sup", np.hstack([pdelta, llr_imp[:, None], pmask_f])),
        "ESM + Evo2-emb":            ("sup", np.hstack([pdelta, edelta, pmask_f])),
    }

    rows = []
    print("\n" + "=" * 78)
    print("BRCA1 cross-modal fusion vs Evo2 (auROC, GroupKFold by position)")
    print("=" * 78)
    for name, (kind, X) in configs.items():
        if kind == "zeroshot":
            pred = X  # already the score (-llr)
        else:
            pred = grouped_auroc(X, y, groups)
        au = roc_auc_score(y, pred)
        per = strat_auroc(y, pred, df)
        print(f"  {name:26s}: all={au:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in per.items()))
        rows.append({"rep": name, "auroc_all": au,
                     **{f"auroc_{k}": v for k, v in per.items()}})

    out = os.path.join(args.output_dir, "brca1_evo2_fusion_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")
    print("Cross-modal wins iff 'ESM + Evo2-*' > 'Evo2 zero-shot' and > 'Evo2-* supervised'.")


if __name__ == "__main__":
    main()
