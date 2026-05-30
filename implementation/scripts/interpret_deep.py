"""
Deep interpretability: (1) map each SAE feature to a biological concept by
checking WHAT variants activate it, (2) redesign the gate with per-dimension
attention and retrain.

Part A — SAE feature → concept mapping:
  For each alive feature, find top-activating variants, compute enrichment
  over consequence type, genomic region, conservation, ClinVar, protein domain.
  Assign a concept label to each feature.

Part B — Improved gate:
  Current gate is static (w_prot≈0.475, w_dna≈0.525 everywhere).
  Fix: per-DIMENSION gate (256-d vector, not 2-scalar), so each shared-space
  dimension independently decides protein-vs-DNA. Plus a gate entropy loss
  to encourage decisive routing.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BRCA1_DOMAINS = {"RING": (1, 109), "Coiled-coil": (1364, 1437),
                 "BRCT_N": (1642, 1736), "BRCT_C": (1756, 1855)}
SEED = 0


def get_domain(aa_pos):
    if pd.isna(aa_pos):
        return "non-protein"
    aa_pos = int(aa_pos)
    for name, (s, e) in BRCA1_DOMAINS.items():
        if s <= aa_pos <= e:
            return name
    return "inter-domain"


def part_a_sae_concepts(df, sae_acts, reps, output_dir):
    """Map each SAE feature to biological concepts via enrichment analysis."""
    y = df["label"].values
    cons = df["consequence"].values
    domain = df["aa_pos"].apply(get_domain).values
    phyloP = df["phyloP"].values
    cadd = df["CADD"].values
    clinvar = df["clinvar_simple"].values

    cons_types = ["Missense", "Synonymous", "Intronic", "Splice region",
                  "Nonsense", "Canonical splice", "5' UTR"]
    base_rates = {c: (cons == c).mean() for c in cons_types}
    path_base = y.mean()

    n_features = sae_acts.shape[1]
    cards = []

    for fi in range(n_features):
        a = sae_acts[:, fi]
        active = np.abs(a) > 1e-6
        n_act = active.sum()
        if n_act < 5:
            continue

        card = {"feature": fi, "n_active": int(n_act)}

        # pathogenicity
        card["path_rate"] = y[active].mean()
        card["path_enrich"] = card["path_rate"] / max(path_base, 1e-8)

        # consequence enrichment (odds ratio vs background)
        top_cons, top_enrich = "none", 0
        for c in cons_types:
            frac = (cons[active] == c).mean()
            enrich = frac / max(base_rates[c], 1e-8)
            card[f"cons_{c}"] = frac
            card[f"enrich_{c}"] = enrich
            if enrich > top_enrich:
                top_enrich, top_cons = enrich, c
        card["top_consequence"] = top_cons
        card["top_cons_enrich"] = top_enrich

        # domain enrichment (for protein-affecting features)
        dom_counts = pd.Series(domain[active]).value_counts()
        card["top_domain"] = dom_counts.index[0] if len(dom_counts) > 0 else "none"
        card["top_domain_frac"] = dom_counts.iloc[0] / n_act if len(dom_counts) > 0 else 0

        # conservation of activated variants
        card["mean_phyloP"] = float(np.nanmean(phyloP[active]))
        card["mean_CADD"] = float(np.nanmean(cadd[active]))

        # positional spread (genomic)
        positions = df["pos_hg19"].values[active]
        card["pos_range"] = int(positions.max() - positions.min())
        card["pos_mean"] = float(positions.mean())

        # ClinVar enrichment
        clinvar_active = clinvar[active]
        card["clinvar_pathogenic_frac"] = ((clinvar_active == "Pathogenic") |
                                           (clinvar_active == "Likely pathogenic")).mean()

        # gate weights for active variants
        card["gate_prot"] = reps["gate_prot"][active].mean()
        card["gate_dna"] = reps["gate_dna"][active].mean()

        # concept label (heuristic)
        if card["path_rate"] > 0.8 and top_cons == "Nonsense":
            card["concept"] = "truncating_pathogenic"
        elif card["path_rate"] > 0.8 and top_cons == "Canonical splice":
            card["concept"] = "splice_disruption"
        elif card["path_rate"] > 0.7 and top_cons == "Missense" and card["top_domain"] in ["RING", "BRCT_N", "BRCT_C"]:
            card["concept"] = f"critical_missense_{card['top_domain']}"
        elif card["path_rate"] > 0.6 and card[f"cons_Intronic"] > 0.5:
            card["concept"] = "intronic_pathogenic"
        elif card["path_rate"] < 0.1 and top_cons == "Synonymous":
            card["concept"] = "silent_benign"
        elif card["path_rate"] < 0.1 and top_cons == "Missense":
            card["concept"] = "tolerated_missense"
        elif card["path_rate"] < 0.15 and card[f"cons_Intronic"] > 0.5:
            card["concept"] = "neutral_intronic"
        elif card["mean_CADD"] > 25 and card["path_rate"] > 0.5:
            card["concept"] = "high_CADD_damaging"
        elif card["mean_phyloP"] > 4 and card["path_rate"] > 0.5:
            card["concept"] = "highly_conserved_damaging"
        else:
            card["concept"] = "mixed"

        # top 5 activating variants (for manual inspection)
        top5_idx = np.argsort(-np.abs(a))[:5]
        top5_info = []
        for idx in top5_idx:
            r = df.iloc[idx]
            top5_info.append(f"{r['consequence']}|{r.get('aa_ref','')}{int(r['aa_pos']) if pd.notna(r['aa_pos']) else '?'}{r.get('aa_alt','')}|{'P' if r['label'] else 'B'}|phyloP={r['phyloP']:.1f}")
        card["top5_variants"] = "; ".join(top5_info)

        cards.append(card)

    fc = pd.DataFrame(cards)

    # print concept summary
    print("\n" + "=" * 70)
    print("Part A: SAE Feature → Biological Concept Mapping")
    print("=" * 70)
    concept_counts = fc["concept"].value_counts()
    print(f"\nConcept distribution ({len(fc)} alive features):")
    for c, n in concept_counts.items():
        sub = fc[fc["concept"] == c]
        print(f"  {c:30s}: {n:3d} features, mean path_rate={sub['path_rate'].mean():.2f}, "
              f"mean CADD={sub['mean_CADD'].mean():.1f}")

    # print most interesting features per concept
    for concept in ["truncating_pathogenic", "splice_disruption", "critical_missense_RING",
                     "critical_missense_BRCT_N", "critical_missense_BRCT_C",
                     "tolerated_missense", "silent_benign", "neutral_intronic",
                     "intronic_pathogenic"]:
        sub = fc[fc["concept"] == concept].sort_values("path_enrich", ascending=False)
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print(f"\n  [{concept}] Feature #{int(r['feature'])} (n={int(r['n_active'])}, "
              f"path={r['path_rate']:.0%}, top_cons={r['top_consequence']}, "
              f"domain={r['top_domain']}, CADD={r['mean_CADD']:.1f}, phyloP={r['mean_phyloP']:.1f})")
        print(f"    top variants: {r['top5_variants']}")

    fc.to_csv(os.path.join(output_dir, "sae_concept_cards.csv"), index=False)
    return fc


# ── Part B: Improved gate ──
class PerDimFusionHead(nn.Module):
    """Per-dimension gating: each of the 256 shared-space dimensions independently
    decides how much comes from protein vs DNA. Plus entropy loss for decisive routing."""
    def __init__(self, pretrained_clip, d_shared=256, d_hidden=128, p_drop=0.3,
                 freeze_encoders=True, scalar_dim=0):
        super().__init__()
        self.enc_prot = pretrained_clip.enc_prot
        self.enc_dna = pretrained_clip.enc_dna
        self.scalar_dim = scalar_dim
        if freeze_encoders:
            for p in self.enc_prot.parameters():
                p.requires_grad = False
            for p in self.enc_dna.parameters():
                p.requires_grad = False
        # per-dimension gate: (z_p, z_d, scalars) → 256-d sigmoid
        gate_in = d_shared * 2 + scalar_dim
        self.dim_gate = nn.Sequential(
            nn.Linear(gate_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_shared), nn.Sigmoid())
        head_in = d_shared + scalar_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, 1))

    def forward(self, xp, xd, mask, scalars=None):
        zp = self.enc_prot(xp) * mask
        zd = self.enc_dna(xd)
        gate_in = torch.cat([zp, zd], dim=-1)
        if scalars is not None:
            gate_in = torch.cat([gate_in, scalars], dim=-1)
        alpha = self.dim_gate(gate_in)  # (B, d_shared), 0=DNA, 1=protein
        z = alpha * zp + (1 - alpha) * zd
        if scalars is not None:
            z = torch.cat([z, scalars], dim=-1)
        return self.head(z).squeeze(-1), zp, zd, alpha

    def gate_entropy_loss(self, alpha):
        """Encourage decisive gates (push toward 0 or 1, not 0.5)."""
        ent = -(alpha * (alpha + 1e-8).log() + (1 - alpha) * (1 - alpha + 1e-8).log())
        return ent.mean()


def train_fold_v2(model_fn, Xp_tr, Xd_tr, m_tr, y_tr, s_tr,
                  Xp_va, Xd_va, m_va, y_va, s_va,
                  Xp_te, Xd_te, m_te, s_te, dev,
                  lambda_ent=0.05):
    model = model_fn().to(dev)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=5e-4, weight_decay=1e-4)
    tp, td, tm, ty = [torch.tensor(x, dtype=torch.float32, device=dev)
                      for x in (Xp_tr, Xd_tr, m_tr, y_tr)]
    ts = torch.tensor(s_tr, dtype=torch.float32, device=dev) if s_tr is not None else None
    vp, vd, vm = [torch.tensor(x, dtype=torch.float32, device=dev)
                  for x in (Xp_va, Xd_va, m_va)]
    vs = torch.tensor(s_va, dtype=torch.float32, device=dev) if s_va is not None else None

    best_auc, best_state, bad = -1, None, 0
    for ep in range(200):
        model.train(); opt.zero_grad()
        logit, zp, zd, alpha = model(tp, td, tm, ts)
        loss = F.binary_cross_entropy_with_logits(logit, ty)
        valid = tm.squeeze(-1) > 0.5
        if valid.any():
            loss = loss + 0.1 * (1 - F.cosine_similarity(zp[valid], zd[valid])).mean()
        loss = loss + lambda_ent * model.gate_entropy_loss(alpha)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl, _, _, _ = model(vp, vd, vm, vs)
            va = roc_auc_score(y_va, torch.sigmoid(vl).cpu().numpy())
        if va > best_auc + 1e-4:
            best_auc, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 30: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        ep_, ed_, em_ = [torch.tensor(x, dtype=torch.float32, device=dev)
                         for x in (Xp_te, Xd_te, m_te)]
        es_ = torch.tensor(s_te, dtype=torch.float32, device=dev) if s_te is not None else None
        logit, _, _, alpha = model(ep_, ed_, em_, es_)
        return torch.sigmoid(logit).cpu().numpy(), alpha.cpu().numpy()


def part_b_gate(df, pdelta, edelta, pmask, llr_imp, ckpt_path, device, output_dir):
    """Train with per-dimension gate and analyze."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    y = df["label"].values.astype(np.float32)
    groups = df["pos_hg19"].values
    mask = pmask[:, None]
    scalars = llr_imp[:, None]

    def make_model():
        clip = CrossModalCLIP(d_prot=pdelta.shape[1], d_dna=edelta.shape[1],
                              d_hidden=cfg["d_hidden"], d_shared=cfg["d_shared"],
                              n_layers=cfg["n_layers"])
        clip.load_state_dict(ckpt["model_state"])
        return PerDimFusionHead(clip, d_shared=cfg["d_shared"], scalar_dim=1)

    np.random.seed(SEED); torch.manual_seed(SEED)
    gkf = GroupKFold(n_splits=5)
    pred = np.zeros(len(y))
    all_alpha = np.zeros((len(y), cfg["d_shared"]))

    for fold_i, (tr_all, te) in enumerate(gkf.split(pdelta, y, groups)):
        g_tr = groups[tr_all]; uniq = np.unique(g_tr)
        rng = np.random.default_rng(SEED + fold_i)
        va_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        va = tr_all[np.array([g in va_groups for g in g_tr])]
        tr = tr_all[np.array([g not in va_groups for g in g_tr])]
        sp = StandardScaler().fit(pdelta[tr]); sd = StandardScaler().fit(edelta[tr])
        ss = StandardScaler().fit(scalars[tr])
        p_te, alpha_te = train_fold_v2(
            make_model,
            sp.transform(pdelta[tr]), sd.transform(edelta[tr]), mask[tr], y[tr], ss.transform(scalars[tr]),
            sp.transform(pdelta[va]), sd.transform(edelta[va]), mask[va], y[va], ss.transform(scalars[va]),
            sp.transform(pdelta[te]), sd.transform(edelta[te]), mask[te], ss.transform(scalars[te]),
            device)
        pred[te] = p_te
        all_alpha[te] = alpha_te

    auroc = roc_auc_score(y, pred)
    strata = {"all": np.ones(len(df), bool),
              "coding": (df["vtype"] == "coding").values,
              "noncoding": (df["vtype"] == "noncoding").values,
              "missense": df["is_missense"].values.astype(bool)}

    print("\n" + "=" * 70)
    print("Part B: Per-Dimension Gate (improved design)")
    print("=" * 70)
    for s, m in strata.items():
        if m.sum() < 20 and len(np.unique(y[m])) < 2: continue
        au = roc_auc_score(y[m], pred[m]) if len(np.unique(y[m])) == 2 else float("nan")
        alpha_mean = all_alpha[m].mean(axis=0)
        n_prot = (alpha_mean > 0.6).sum()
        n_dna = (alpha_mean < 0.4).sum()
        n_shared = ((alpha_mean >= 0.4) & (alpha_mean <= 0.6)).sum()
        print(f"  {s:12s}: auROC={au:.4f}  "
              f"prot-dims={n_prot}  dna-dims={n_dna}  shared-dims={n_shared}  "
              f"alpha_mean={alpha_mean.mean():.3f}±{alpha_mean.std():.3f}")

    # analyze which dimensions prefer protein vs DNA
    alpha_coding = all_alpha[strata["coding"]].mean(axis=0)
    alpha_nonc = all_alpha[strata["noncoding"]].mean(axis=0)
    diff = alpha_coding - alpha_nonc  # positive = dim prefers protein for coding

    print(f"\n  Per-dimension gate difference (coding - noncoding alpha):")
    print(f"    dims more protein for coding (diff>0.1): {(diff > 0.1).sum()}")
    print(f"    dims more DNA for noncoding (diff<-0.1): {(diff < -0.1).sum()}")
    print(f"    stable dims (|diff|<0.1):               {(np.abs(diff) < 0.1).sum()}")

    np.savez_compressed(os.path.join(output_dir, "perdim_gate.npz"),
                        alpha=all_alpha, pred=pred, diff=diff)
    return auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="results/variant/brca1_variants.csv")
    ap.add_argument("--esm", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--evo2", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--checkpoint", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--sae_acts", default="results/interpret/sae_activations.npz")
    ap.add_argument("--shared_reps", default="results/interpret/shared_reps.npz")
    ap.add_argument("--output_dir", default="results/interpret")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.variants).reset_index(drop=True)
    pdz = np.load(args.esm)
    pdelta, pmask = pdz["pdelta"].astype(np.float32), pdz["pmask"].astype(np.float32)
    ev = np.load(args.evo2)
    edelta = ev["edelta"].astype(np.float32)
    llr = ev["llr"]
    llr_imp = np.where(np.isnan(llr), np.nanmedian(llr), llr).astype(np.float32)

    sae_acts = np.load(args.sae_acts)["acts"]
    reps = dict(np.load(args.shared_reps))
    logger.info(f"loaded sae_acts {sae_acts.shape}, reps keys={list(reps.keys())}")

    # Part A: SAE concept mapping
    part_a_sae_concepts(df, sae_acts, reps, args.output_dir)

    # Part B: Improved gate
    part_b_gate(df, pdelta, edelta, pmask, llr_imp, args.checkpoint, dev, args.output_dir)


if __name__ == "__main__":
    main()
