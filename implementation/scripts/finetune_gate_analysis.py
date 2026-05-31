"""
Fine-tune variant fusion + extract gate weights for every variant.

Same pipeline as finetune_variant.py but:
1. After training each fold, extracts gate weights [g_prot, g_dna] + z_prot, z_dna
2. Saves per-variant gate weights, representations, and predictions
3. Runs comprehensive gate analysis: how does fusion work per variant type?

Needs GPU. Submit via slurm_gate_analysis.sh.
"""

import argparse, logging, os, sys
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
EPOCHS = 200
PATIENCE = 30
LR = 5e-4
WD = 1e-4


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--prot_delta", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--dna_delta", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--checkpoint", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--output_dir", default="results/gate_analysis")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

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
    scalars_raw = llr_imp[:, None]

    d_prot_actual = pdelta.shape[1]
    d_dna_actual = ddelta.shape[1]
    dim_match = (d_prot_actual == cfg["d_prot"] and d_dna_actual == cfg["d_dna"])
    logger.info(f"prot={d_prot_actual}-d, dna={d_dna_actual}-d, dim_match={dim_match}")

    def make_model():
        clip = CrossModalCLIP(d_prot=d_prot_actual, d_dna=d_dna_actual,
                              d_hidden=cfg["d_hidden"], d_shared=cfg["d_shared"],
                              n_layers=cfg["n_layers"])
        if dim_match:
            clip.load_state_dict(ckpt["model_state"])
        else:
            prot_keys = {k: v for k, v in ckpt["model_state"].items() if "enc_prot" in k}
            clip.load_state_dict(prot_keys, strict=False)
        return VariantFusionHead(clip, d_shared=cfg["d_shared"],
                                freeze_encoders=False, scalar_dim=1)

    # OOF with gate extraction
    set_seed(SEED)
    gkf = GroupKFold(n_splits=5)
    n = len(y)
    all_gate = np.zeros((n, 2), dtype=np.float32)
    all_zp = np.zeros((n, cfg["d_shared"]), dtype=np.float32)
    all_zd = np.zeros((n, cfg["d_shared"]), dtype=np.float32)
    all_pred = np.zeros(n, dtype=np.float32)

    for fold_i, (tr_all, te) in enumerate(gkf.split(pdelta, y, groups)):
        logger.info(f"Fold {fold_i}: train={len(tr_all)}, test={len(te)}")

        g_tr = groups[tr_all]; uniq = np.unique(g_tr)
        rng = np.random.default_rng(SEED + fold_i)
        va_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        va = tr_all[np.array([g in va_groups for g in g_tr])]
        tr = tr_all[np.array([g not in va_groups for g in g_tr])]

        sp = StandardScaler().fit(pdelta[tr]); sd = StandardScaler().fit(ddelta[tr])
        ss = StandardScaler().fit(scalars_raw[tr])

        def prep(idx):
            return (torch.tensor(sp.transform(pdelta[idx]), dtype=torch.float32, device=dev),
                    torch.tensor(sd.transform(ddelta[idx]), dtype=torch.float32, device=dev),
                    torch.tensor(pmask[idx], dtype=torch.float32, device=dev),
                    torch.tensor(ss.transform(scalars_raw[idx]), dtype=torch.float32, device=dev))

        tp, td, tm, ts = prep(tr)
        vp, vd, vm, vs = prep(va)

        model = make_model().to(dev)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR, weight_decay=WD)

        best_auc, best_state, bad = -1, None, 0
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            logit, zp, zd = model(tp, td, tm, ts)
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[tr], device=dev))
            valid = tm.squeeze(-1) > 0.5
            if valid.any():
                loss = loss + 0.1 * (1 - F.cosine_similarity(zp[valid], zd[valid])).mean()
            loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                vl, _, _ = model(vp, vd, vm, vs)
                va_auc = roc_auc_score(y[va], torch.sigmoid(vl).cpu().numpy())
            if va_auc > best_auc + 1e-4:
                best_auc = va_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()

        # extract gate weights for test variants
        ep_, ed_, em_, es_ = prep(te)
        with torch.no_grad():
            logit, zp, zd, gate = model(ep_, ed_, em_, es_, return_gate=True)
            all_pred[te] = torch.sigmoid(logit).cpu().numpy()
            all_gate[te] = gate.cpu().numpy()
            all_zp[te] = zp.cpu().numpy()
            all_zd[te] = zd.cpu().numpy()

        logger.info(f"  fold {fold_i}: val_auc={best_auc:.4f}, "
                    f"gate_prot={all_gate[te, 0].mean():.3f}±{all_gate[te, 0].std():.3f}, "
                    f"gate_dna={all_gate[te, 1].mean():.3f}±{all_gate[te, 1].std():.3f}")

    # save checkpoint for fold 0 (for downstream SAE)
    torch.save({"model_state": best_state, "config": cfg,
                "d_prot_actual": d_prot_actual, "d_dna_actual": d_dna_actual},
               os.path.join(args.output_dir, "variant_fusion_fold4.pt"))

    # ═══ Gate analysis ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"GATE ANALYSIS: {len(df)} BRCA1 variants")
    print(f"{'='*70}")

    auc_all = roc_auc_score(y, all_pred)
    print(f"\nFusion AUC: {auc_all:.4f}  (Evo2 zero-shot: {roc_auc_score(y, -llr_imp):.4f})")

    g_prot = all_gate[:, 0]
    g_dna = all_gate[:, 1]

    # overall gate distribution
    print(f"\nGate weight distribution:")
    print(f"  g_prot: {g_prot.mean():.4f} ± {g_prot.std():.4f}  "
          f"[{g_prot.min():.4f}, {g_prot.max():.4f}]")
    print(f"  g_dna:  {g_dna.mean():.4f} ± {g_dna.std():.4f}  "
          f"[{g_dna.min():.4f}, {g_dna.max():.4f}]")

    # by variant type
    print(f"\nGate weights by variant type:")
    print(f"  {'type':20s} {'n':>6s} {'g_prot':>8s} {'g_dna':>8s} {'path%':>6s} {'AUC':>6s}")
    for vtype in ["coding", "noncoding"]:
        m = (df["vtype"] == vtype).values
        if m.sum() < 10 or len(np.unique(y[m])) < 2: continue
        print(f"  {vtype:20s} {m.sum():6d} {g_prot[m].mean():8.4f} {g_dna[m].mean():8.4f} "
              f"{y[m].mean():6.3f} {roc_auc_score(y[m], all_pred[m]):6.4f}")

    if "is_missense" in df.columns:
        m = df["is_missense"].values
        if m.sum() > 10 and len(np.unique(y[m])) == 2:
            print(f"  {'missense':20s} {m.sum():6d} {g_prot[m].mean():8.4f} {g_dna[m].mean():8.4f} "
                  f"{y[m].mean():6.3f} {roc_auc_score(y[m], all_pred[m]):6.4f}")

    # by pathogenicity
    print(f"\nGate weights by pathogenicity:")
    for label, name in [(1, "pathogenic"), (0, "benign")]:
        m = y == label
        print(f"  {name:20s} {m.sum():6d} {g_prot[m].mean():8.4f} {g_dna[m].mean():8.4f}")

    # gate vs prediction
    from scipy.stats import spearmanr
    rho_pred, _ = spearmanr(g_prot, all_pred)
    print(f"\nGate-prediction correlation:")
    print(f"  r(g_prot, pred) = {rho_pred:+.4f}")
    rho_llr, _ = spearmanr(g_dna, -llr_imp)
    print(f"  r(g_dna, -LLR) = {rho_llr:+.4f}")

    # cross-modal alignment in shared space
    cos = np.sum(all_zp * all_zd, axis=1) / (
        np.linalg.norm(all_zp, axis=1) * np.linalg.norm(all_zd, axis=1) + 1e-8)
    print(f"\nShared space alignment (cosine zp·zd):")
    print(f"  All: {cos.mean():.4f} ± {cos.std():.4f}")
    for vtype in ["coding", "noncoding"]:
        m = (df["vtype"] == vtype).values
        if m.sum() > 0:
            print(f"  {vtype}: {cos[m].mean():.4f} ± {cos[m].std():.4f}")
    print(f"  Pathogenic: {cos[y==1].mean():.4f} ± {cos[y==1].std():.4f}")
    print(f"  Benign: {cos[y==0].mean():.4f} ± {cos[y==0].std():.4f}")

    # save
    np.savez_compressed(os.path.join(args.output_dir, "brca1_gate_analysis.npz"),
                        gate=all_gate, z_prot=all_zp, z_dna=all_zd,
                        pred=all_pred, y=y)
    df[["pos_hg19", "label", "vtype", "is_missense"]].to_csv(
        os.path.join(args.output_dir, "brca1_gate_variants.csv"), index=False)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
