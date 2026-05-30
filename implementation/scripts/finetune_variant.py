"""
Phase 2: Fine-tune the pretrained cross-modal shared embedding for variant effect.

Loads the CLIP-pretrained encoders, projects BRCA1 variant-level embedding deltas
(ESM protein + DNA) into the shared space, fuses there, and trains a classification
head. Encoders can be frozen or fine-tuned.

Reports auROC with GroupKFold by genomic position + bootstrap CI.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP, VariantFusionHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 0
N_BOOT = 2000
EPOCHS = 200
PATIENCE = 30
LR = 5e-4
WD = 1e-4


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_fold(model_fn, Xp_tr, Xd_tr, m_tr, y_tr, Xp_va, Xd_va, m_va, y_va,
               Xp_te, Xd_te, m_te, dev):
    model = model_fn().to(dev)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WD)
    tp = torch.tensor(Xp_tr, dtype=torch.float32, device=dev)
    td = torch.tensor(Xd_tr, dtype=torch.float32, device=dev)
    tm = torch.tensor(m_tr, dtype=torch.float32, device=dev)
    ty = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    vp = torch.tensor(Xp_va, dtype=torch.float32, device=dev)
    vd = torch.tensor(Xd_va, dtype=torch.float32, device=dev)
    vm = torch.tensor(m_va, dtype=torch.float32, device=dev)

    best_auc, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        logit, zp, zd = model(tp, td, tm)
        loss = F.binary_cross_entropy_with_logits(logit, ty)
        # cross-modal alignment reg on valid pairs
        valid = tm.squeeze(-1) > 0.5
        if valid.any():
            loss = loss + 0.1 * (1 - F.cosine_similarity(zp[valid], zd[valid])).mean()
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vlogit, _, _ = model(vp, vd, vm)
            va = roc_auc_score(y_va, torch.sigmoid(vlogit).cpu().numpy())
        if va > best_auc + 1e-4:
            best_auc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ep_ = torch.tensor(Xp_te, dtype=torch.float32, device=dev)
        ed_ = torch.tensor(Xd_te, dtype=torch.float32, device=dev)
        em_ = torch.tensor(m_te, dtype=torch.float32, device=dev)
        logit, _, _ = model(ep_, ed_, em_)
        return torch.sigmoid(logit).cpu().numpy()


def oof(model_fn, pdelta, ddelta, mask, y, groups, dev, n_splits=5):
    set_seed(SEED)
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for fold_i, (tr_all, te) in enumerate(gkf.split(pdelta, y, groups)):
        g_tr = groups[tr_all]; uniq = np.unique(g_tr)
        rng = np.random.default_rng(SEED + fold_i)
        va_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        va = tr_all[np.array([g in va_groups for g in g_tr])]
        tr = tr_all[np.array([g not in va_groups for g in g_tr])]
        # standardize using train stats — use the pretrained normalization for
        # the encoder, but standard-scale the raw deltas going IN to the encoder
        sp = StandardScaler().fit(pdelta[tr]); sd = StandardScaler().fit(ddelta[tr])
        pred[te] = train_fold(
            model_fn,
            sp.transform(pdelta[tr]), sd.transform(ddelta[tr]), mask[tr], y[tr],
            sp.transform(pdelta[va]), sd.transform(ddelta[va]), mask[va], y[va],
            sp.transform(pdelta[te]), sd.transform(ddelta[te]), mask[te], dev)
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
    ap.add_argument("--prot_delta", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--dna_delta", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--checkpoint", default="results/pretrain/crossmodal_clip.pt")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze_encoders", action="store_true",
                    help="freeze pretrained encoders (only train head+gate)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    # load pretrained checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    logger.info(f"pretrained config: {cfg}")

    df = pd.read_csv(args.variants).reset_index(drop=True)
    y = df["label"].values.astype(np.float32)
    groups = df["pos_hg19"].values

    pdz = np.load(args.prot_delta)
    pdelta = pdz["pdelta"].astype(np.float32)
    pmask = pdz["pmask"].astype(np.float32)[:, None]

    ev = np.load(args.dna_delta)
    ddelta = ev["edelta"].astype(np.float32)
    llr = ev["llr"]
    llr_imp = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)

    d_prot_actual = pdelta.shape[1]
    d_dna_actual = ddelta.shape[1]
    logger.info(f"variant deltas: prot {d_prot_actual}-d, dna {d_dna_actual}-d")
    logger.info(f"pretrained dims: prot {cfg['d_prot']}-d, dna {cfg['d_dna']}-d")

    # if DNA dim mismatch (pretrained on old 1024-d, variants have Evo2 4096-d),
    # re-initialize the DNA encoder for the new dim but keep protein encoder pretrained
    dim_match = (d_prot_actual == cfg["d_prot"] and d_dna_actual == cfg["d_dna"])

    def make_model():
        if dim_match:
            clip = CrossModalCLIP(**cfg)
            clip.load_state_dict(ckpt["model_state"])
            logger.info("loaded FULL pretrained weights (dims match)")
        else:
            clip = CrossModalCLIP(d_prot=d_prot_actual, d_dna=d_dna_actual,
                                  d_hidden=cfg["d_hidden"], d_shared=cfg["d_shared"],
                                  n_layers=cfg["n_layers"])
            # load protein encoder weights (dims match)
            prot_keys = {k: v for k, v in ckpt["model_state"].items() if "enc_prot" in k}
            clip.load_state_dict(prot_keys, strict=False)
            logger.info(f"dim mismatch: loaded protein encoder, DNA encoder random "
                        f"(pretrained {cfg['d_dna']}-d vs actual {d_dna_actual}-d)")
        return VariantFusionHead(clip, d_shared=cfg["d_shared"],
                                freeze_encoders=args.freeze_encoders)

    # --- Evo2 zero-shot reference ---
    print(f"\n{'='*76}")
    print("BRCA1 variant effect: PRETRAINED cross-modal shared embedding")
    print(f"  encoders={'frozen' if args.freeze_encoders else 'fine-tuned'}")
    print(f"  dim_match={dim_match}")
    print(f"{'='*76}")
    print(f"  {'Evo2 zero-shot':20s}: all={roc_auc_score(y, -llr_imp):.4f} (reference)")

    preds = {"Evo2 zero-shot": -llr_imp}
    p = oof(make_model, pdelta, ddelta, pmask, y, groups, dev)
    tag = "Pretrained-Fused"
    preds[tag] = p
    s = strat(y, p, df)
    print(f"  {tag:20s}: " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))

    # bootstrap CI
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    strata_m = {"all": np.ones(len(df), bool),
                "coding": (df["vtype"] == "coding").values,
                "noncoding": (df["vtype"] == "noncoding").values,
                "missense": df["is_missense"].values}
    comps = [(tag, "Evo2 zero-shot")]
    boot = {c: {s: [] for s in strata_m} for c in comps}
    for b in range(N_BOOT):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gs]); yb = y[idx]
        for (a, bn) in comps:
            pa, pb = preds[a][idx], preds[bn][idx]
            for s, mf in strata_m.items():
                mb = mf[idx]
                if mb.sum() < 20 or len(np.unique(yb[mb])) < 2:
                    boot[(a, bn)][s].append(np.nan); continue
                boot[(a, bn)][s].append(
                    roc_auc_score(yb[mb], pa[mb]) - roc_auc_score(yb[mb], pb[mb]))

    print(f"\n{'='*76}")
    print(f"Delta auROC 95% CI (bootstrap B={N_BOOT}); '*'=excludes 0")
    print(f"{'='*76}")
    rows = []
    for (a, bn) in comps:
        print(f"\n  {a}  -  {bn}")
        for s in strata_m:
            arr = np.array(boot[(a, bn)][s], float); arr = arr[~np.isnan(arr)]
            if len(arr) < 100: continue
            lo, hi = np.percentile(arr, [2.5, 97.5]); med = np.median(arr)
            print(f"    {s:10s}: Δ={med:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {'*' if lo>0 else ' '}")
            rows.append({"a": a, "b": bn, "stratum": s, "delta": med,
                         "ci_lo": lo, "ci_hi": hi, "sig": bool(lo > 0)})
    out = os.path.join(args.output_dir, "brca1_pretrained_fusion_bootstrap.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
