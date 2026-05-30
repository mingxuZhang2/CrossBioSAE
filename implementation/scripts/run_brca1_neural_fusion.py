"""
BRCA1 variant effect from a LEARNED cross-modal shared embedding (CrossModalFusionNet).
Neural fusion of ESM-2 (1280-d) and Evo2 (4096-d) variant embeddings, trained on GPU,
GroupKFold by genomic position (no leakage), with internal val early-stopping.

Reports vs the numbers to beat:
  - Evo2 zero-shot LLR (0.889 baseline)   -> does the learned fusion beat Evo2?
  - Evo2-emb / ESM-emb single-modality MLP -> does fusing help over each modality?
Bootstrap CI on the margins.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from fusion_net import CrossModalFusionNet, fusion_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 0
N_BOOT = 2000
EPOCHS = 200
PATIENCE = 25
LR = 1e-3
WD = 1e-4


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_fold(Xp_tr, Xd_tr, m_tr, y_tr, Xp_va, Xd_va, m_va, y_va,
               Xp_te, Xd_te, m_te, single, dev):
    """Train one fold; return test-set probabilities. single in {None,'prot','dna'}."""
    net = CrossModalFusionNet(d_prot=Xp_tr.shape[1], d_dna=Xd_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    tp, td, tm, ty = (torch.tensor(a, dtype=torch.float32, device=dev)
                      for a in (Xp_tr, Xd_tr, m_tr, y_tr))
    vp, vd, vm = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (Xp_va, Xd_va, m_va))
    # single-modality ablation: blank the other input + mask
    if single == "prot":
        td = td * 0; vd = vd * 0
    elif single == "dna":
        tp = tp * 0; vp = vp * 0; tm = tm * 0; vm = vm * 0
    best_auc, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        net.train(); opt.zero_grad()
        logit, zp, zd, _ = net(tp, td, tm)
        loss, _, _ = fusion_loss(logit, ty, zp, zd, tm)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vlogit, _, _, _ = net(vp, vd, vm)
            va = roc_auc_score(y_va, torch.sigmoid(vlogit).cpu().numpy())
        if va > best_auc + 1e-4:
            best_auc, best_state, bad = va, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ep_, ed_, em_ = (torch.tensor(a, dtype=torch.float32, device=dev)
                         for a in (Xp_te, Xd_te, m_te))
        if single == "prot": ed_ = ed_ * 0
        elif single == "dna": ep_ = ep_ * 0; em_ = em_ * 0
        logit, _, _, _ = net(ep_, ed_, em_)
        return torch.sigmoid(logit).cpu().numpy()


def oof(pdelta, edelta, mask, y, groups, single, dev, n_splits=5):
    set_seed(SEED)
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for tr_all, te in gkf.split(pdelta, y, groups):
        # carve a val split out of train by group
        g_tr = groups[tr_all]; uniq = np.unique(g_tr)
        rng = np.random.default_rng(SEED)
        va_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        va = tr_all[np.array([g in va_groups for g in g_tr])]
        tr = tr_all[np.array([g not in va_groups for g in g_tr])]
        sp = StandardScaler().fit(pdelta[tr]); sd = StandardScaler().fit(edelta[tr])
        Pp, Pd = sp.transform, sd.transform
        pred[te] = train_fold(
            Pp(pdelta[tr]), Pd(edelta[tr]), mask[tr], y[tr],
            Pp(pdelta[va]), Pd(edelta[va]), mask[va], y[va],
            Pp(pdelta[te]), Pd(edelta[te]), mask[te], single, dev)
    return pred


def strat(y, p, df):
    out = {"all": roc_auc_score(y, p)}
    for vt in ["coding", "noncoding"]:
        m = (df["vtype"] == vt).values
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            out[vt] = roc_auc_score(y[m], p[m])
    mis = df["is_missense"].values
    if len(np.unique(y[mis])) == 2:
        out["missense"] = roc_auc_score(y[mis], p[mis])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--esm", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--evo2", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_llr", action="store_true",
                    help="append Evo2 LLR scalar to DNA branch (recovers noncoding)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device={dev}")

    df = pd.read_csv(args.variants).reset_index(drop=True)
    y = df["label"].values.astype(np.float32)
    groups = df["pos_hg19"].values
    pdz = np.load(args.esm)
    pdelta, pmask = pdz["pdelta"].astype(np.float32), pdz["pmask"].astype(np.float32)
    ev = np.load(args.evo2)
    edelta = ev["edelta"].astype(np.float32)
    llr = ev["llr"]; llr_imp = np.where(np.isnan(llr), np.nanmedian(llr), llr)
    mask = pmask[:, None]
    if args.use_llr:
        # give the DNA branch Evo2's headline signal (likelihood drop), which the
        # mid-layer embedding delta does NOT capture -- crucial for noncoding.
        edelta = np.hstack([edelta, llr_imp[:, None].astype(np.float32)])
        logger.info("appended Evo2 LLR to DNA features")
    logger.info(f"ESM {pdelta.shape} Evo2 {edelta.shape} n={len(df)}")

    preds = {}
    print("\n" + "=" * 76)
    print("BRCA1 variant effect: LEARNED cross-modal shared embedding (auROC, GroupKFold)")
    print("=" * 76)
    # Evo2 zero-shot reference (the number to beat)
    print(f"  {'Evo2 zero-shot':16s}: all={roc_auc_score(y, -llr_imp):.4f}  (no training, reference)")
    for name, single in [("ESM-emb MLP", "prot"), ("Evo2-emb MLP", "dna"),
                         ("Neural-Fused", None)]:
        p = oof(pdelta, edelta, mask, y, groups, single, dev)
        preds[name] = p
        s = strat(y, p, df)
        print(f"  {name:16s}: " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))

    # bootstrap CI
    preds["Evo2 zero-shot"] = -llr_imp
    uniq = np.unique(groups); gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    strata = {"all": np.ones(len(df), bool),
              "coding": (df["vtype"] == "coding").values,
              "noncoding": (df["vtype"] == "noncoding").values,
              "missense": df["is_missense"].values}
    comps = [("Neural-Fused", "Evo2 zero-shot"), ("Neural-Fused", "Evo2-emb MLP"),
             ("Neural-Fused", "ESM-emb MLP")]
    boot = {c: {s: [] for s in strata} for c in comps}
    for b in range(N_BOOT):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gs]); yb = y[idx]
        for (a, bn) in comps:
            pa, pb = preds[a][idx], preds[bn][idx]
            for s, mf in strata.items():
                mb = mf[idx]
                if mb.sum() < 20 or len(np.unique(yb[mb])) < 2:
                    boot[(a, bn)][s].append(np.nan); continue
                boot[(a, bn)][s].append(roc_auc_score(yb[mb], pa[mb]) - roc_auc_score(yb[mb], pb[mb]))

    print("\n" + "=" * 76)
    print(f"Delta auROC 95% CI (paired cluster bootstrap, B={N_BOOT}); '*'=excludes 0")
    print("=" * 76)
    rows = []
    for (a, bn) in comps:
        print(f"\n  {a}  -  {bn}")
        for s in strata:
            arr = np.array(boot[(a, bn)][s], float); arr = arr[~np.isnan(arr)]
            if len(arr) < 100:
                print(f"    {s:10s}: n/a"); continue
            lo, hi = np.percentile(arr, [2.5, 97.5]); med = np.median(arr)
            print(f"    {s:10s}: Δ={med:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {'*' if lo > 0 else ' '}")
            rows.append({"a": a, "b": bn, "stratum": s, "delta": med, "ci_lo": lo,
                         "ci_hi": hi, "sig": bool(lo > 0)})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir,
                              "brca1_neural_fusion_bootstrap.csv"), index=False)
    print("\nsaved brca1_neural_fusion_bootstrap.csv")


if __name__ == "__main__":
    main()
