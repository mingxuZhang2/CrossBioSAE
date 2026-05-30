"""
Interpretability analysis of the cross-modal shared embedding.

Part 1 — Linear probing: what biological concepts are encoded in z_fused?
  Train linear classifiers on frozen z_fused (256-d) to predict:
  - variant type (coding/noncoding)
  - missense flag
  - pathogenicity label
  - protein domain (BRCA1 RING / BRCT / other)
  - conservation bin (high/low based on positional conservation proxy)
  High probe accuracy = that concept is linearly encoded in the shared space.

Part 2 — SAE on shared space: decompose z_fused into sparse interpretable features.
  Train a TopK SAE (256 → 1024 sparse features) on the z_fused embeddings.
  For each SAE feature: find which variants activate it most, check if they
  cluster by biological annotation → interpretable cross-modal features.

Part 3 — Gate analysis: per-variant modality reliance.
  Save gate weights [w_prot, w_dna] and analyze vs variant annotations.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP, VariantFusionHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── BRCA1 domain annotations ──
BRCA1_DOMAINS = {
    "RING": (1, 109),
    "Coiled-coil": (1364, 1437),
    "BRCT_N": (1642, 1736),
    "BRCT_C": (1756, 1855),
}


def get_domain(aa_pos):
    for name, (s, e) in BRCA1_DOMAINS.items():
        if s <= aa_pos <= e:
            return name
    return "other"


# ── TopK SAE ──
class TopKSAE(nn.Module):
    def __init__(self, d_in=256, d_hidden=1024, k=32):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        self.k = k

    def forward(self, x):
        h = self.encoder(x)
        # TopK activation
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        h_sparse = h * mask
        x_hat = self.decoder(h_sparse)
        return x_hat, h_sparse, h

    def loss(self, x, x_hat, h_sparse):
        recon = F.mse_loss(x_hat, x)
        l1 = h_sparse.abs().mean() * 0.01
        return recon + l1, recon.item(), l1.item()


def extract_representations(df, pdelta, edelta, llr_imp, pmask, ckpt_path, device):
    """Run the pretrained model and extract z_p, z_d, z_fused, gate weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    clip = CrossModalCLIP(d_prot=pdelta.shape[1], d_dna=edelta.shape[1],
                          d_hidden=cfg["d_hidden"], d_shared=cfg["d_shared"],
                          n_layers=cfg["n_layers"])
    clip.load_state_dict(ckpt["model_state"])

    model = VariantFusionHead(clip, d_shared=cfg["d_shared"],
                              freeze_encoders=True, scalar_dim=1)
    model = model.eval().to(device)

    with torch.no_grad():
        xp = torch.tensor(pdelta, dtype=torch.float32, device=device)
        xd = torch.tensor(edelta, dtype=torch.float32, device=device)
        xm = torch.tensor(pmask[:, None], dtype=torch.float32, device=device)
        xs = torch.tensor(llr_imp[:, None], dtype=torch.float32, device=device)

        zp = model.enc_prot(xp) * xm
        zd = model.enc_dna(xd)

        gate_in = torch.cat([zp, zd, xs], dim=-1)
        g = F.softmax(model.gate(gate_in), dim=-1)
        z_fused = g[:, 0:1] * zp + g[:, 1:2] * zd

    return {
        "zp": zp.cpu().numpy(),
        "zd": zd.cpu().numpy(),
        "z_fused": z_fused.cpu().numpy(),
        "gate_prot": g[:, 0].cpu().numpy(),
        "gate_dna": g[:, 1].cpu().numpy(),
    }


def part1_probing(reps, df):
    """Linear probing: what's encoded in z_fused?"""
    z = reps["z_fused"]
    sc = StandardScaler().fit(z)
    z_s = sc.transform(z)

    # build probe targets
    targets = {}
    targets["is_coding"] = (df["vtype"] == "coding").astype(int).values
    targets["is_missense"] = df["is_missense"].astype(int).values
    targets["pathogenic"] = df["label"].astype(int).values

    # domain (missense only — need protein position)
    if "aa_pos" in df.columns:
        domains = df["aa_pos"].apply(lambda p: get_domain(int(p)) if pd.notna(p) else "other")
    else:
        # derive from hg19 position relative to BRCA1 CDS
        domains = pd.Series(["other"] * len(df))
    mis = df["is_missense"].values.astype(bool)
    if mis.sum() > 100:
        dom_labels = domains[mis].values
        uniq_doms = [d for d in np.unique(dom_labels) if (dom_labels == d).sum() > 10]
        if len(uniq_doms) > 1:
            dom_mask = np.isin(dom_labels, uniq_doms)
            targets["domain(missense)"] = (dom_labels[dom_mask], z_s[mis][dom_mask])

    print("\n" + "=" * 60)
    print("Part 1: Linear Probing — what is encoded in z_fused (256-d)?")
    print("=" * 60)
    rows = []
    for name, y in targets.items():
        if isinstance(y, tuple):
            y_arr, z_sub = y
            # multiclass
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(y_arr)
            y_enc = le.transform(y_arr)
            clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(clf, z_sub, y_enc, cv=5, scoring="accuracy")
            print(f"  {name:20s}: acc={scores.mean():.3f}±{scores.std():.3f}  "
                  f"(chance={1/len(le.classes_):.3f}, classes={list(le.classes_)})")
            rows.append({"target": name, "metric": "accuracy", "score": scores.mean(),
                         "chance": 1/len(le.classes_)})
        else:
            if len(np.unique(y)) < 2:
                continue
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            aucs = []
            for tr, te in skf.split(z_s, y):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(z_s[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(z_s[te])[:, 1]))
            print(f"  {name:20s}: auROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
            rows.append({"target": name, "metric": "auroc", "score": np.mean(aucs),
                         "chance": 0.5})

    # also probe zp and zd separately — does fusion add info?
    for rep_name, rep in [("z_prot", reps["zp"]), ("z_dna", reps["zd"])]:
        rep_s = StandardScaler().fit_transform(rep)
        y = targets["pathogenic"]
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(rep_s, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(rep_s[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(rep_s[te])[:, 1]))
        print(f"  pathogenic({rep_name:5s}): auROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
        rows.append({"target": f"pathogenic({rep_name})", "metric": "auroc",
                     "score": np.mean(aucs), "chance": 0.5})

    return pd.DataFrame(rows)


def part2_sae(reps, df, device, d_hidden=1024, k=32, epochs=500, lr=1e-3):
    """SAE on z_fused: find sparse interpretable features."""
    z = reps["z_fused"]
    sc = StandardScaler().fit(z)
    z_s = sc.transform(z).astype(np.float32)
    z_t = torch.tensor(z_s, dtype=torch.float32, device=device)

    sae = TopKSAE(d_in=z.shape[1], d_hidden=d_hidden, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        sae.train()
        x_hat, h_sparse, _ = sae(z_t)
        loss, recon, l1 = sae.loss(z_t, x_hat, h_sparse)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            logger.info(f"SAE ep {ep}: loss={loss.item():.4f} recon={recon:.4f} l1={l1:.4f}")

    sae.eval()
    with torch.no_grad():
        _, h_sparse, h_full = sae(z_t)
        acts = h_sparse.cpu().numpy()  # (N, d_hidden) sparse activations

    # analyze each SAE feature
    print("\n" + "=" * 60)
    print(f"Part 2: SAE on z_fused ({z.shape[1]}→{d_hidden}, TopK={k})")
    print("=" * 60)

    alive = (np.abs(acts) > 1e-6).sum(axis=0)  # how many variants activate each feature
    alive_pct = alive / len(z)
    dead = (alive == 0).sum()
    print(f"  alive features: {d_hidden - dead}/{d_hidden} ({100*(1-dead/d_hidden):.0f}%)")
    print(f"  mean activation density: {alive_pct[alive>0].mean():.3f}")

    y = df["label"].astype(int).values
    vtype = df["vtype"].values
    is_mis = df["is_missense"].values.astype(bool)

    feature_cards = []
    for fi in range(d_hidden):
        a = acts[:, fi]
        if (np.abs(a) > 1e-6).sum() < 10:
            continue
        active = np.abs(a) > 1e-6
        n_active = active.sum()

        # pathogenicity enrichment
        path_rate_active = y[active].mean() if n_active > 0 else 0
        path_rate_all = y.mean()

        # variant type enrichment
        coding_frac = (vtype[active] == "coding").mean() if n_active > 0 else 0
        missense_frac = is_mis[active].mean() if n_active > 0 else 0

        # mean gate weights for active variants
        gate_p = reps["gate_prot"][active].mean() if n_active > 0 else 0
        gate_d = reps["gate_dna"][active].mean() if n_active > 0 else 0

        card = {
            "feature": fi, "n_active": int(n_active),
            "path_rate": path_rate_active, "path_enrich": path_rate_active / max(path_rate_all, 1e-8),
            "coding_frac": coding_frac, "missense_frac": missense_frac,
            "gate_prot": gate_p, "gate_dna": gate_d,
        }
        feature_cards.append(card)

    fc = pd.DataFrame(feature_cards).sort_values("path_enrich", ascending=False)

    # print top pathogenic-enriched and benign-enriched features
    print("\n  Top 10 pathogenic-enriched SAE features:")
    print(f"  {'feat':>5s} {'n':>5s} {'path%':>6s} {'enrich':>6s} {'coding':>6s} {'miss':>6s} {'g_prot':>6s} {'g_dna':>6s}")
    for _, r in fc.head(10).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_active']):5d} {r['path_rate']:6.2f} "
              f"{r['path_enrich']:6.2f} {r['coding_frac']:6.2f} {r['missense_frac']:6.2f} "
              f"{r['gate_prot']:6.3f} {r['gate_dna']:6.3f}")

    print("\n  Top 10 benign-enriched SAE features:")
    for _, r in fc.tail(10).iterrows():
        print(f"  {int(r['feature']):5d} {int(r['n_active']):5d} {r['path_rate']:6.2f} "
              f"{r['path_enrich']:6.2f} {r['coding_frac']:6.2f} {r['missense_frac']:6.2f} "
              f"{r['gate_prot']:6.3f} {r['gate_dna']:6.3f}")

    return fc, acts


def part3_gate(reps, df):
    """Gate analysis: modality reliance by variant type."""
    print("\n" + "=" * 60)
    print("Part 3: Gate weights — modality reliance by variant annotation")
    print("=" * 60)

    gp, gd = reps["gate_prot"], reps["gate_dna"]
    cos = np.sum(reps["zp"] * reps["zd"], axis=1)  # cosine in shared space (both L2-normed)

    for label, mask in [("all", np.ones(len(df), bool)),
                        ("coding", (df["vtype"] == "coding").values),
                        ("noncoding", (df["vtype"] == "noncoding").values),
                        ("missense", df["is_missense"].values.astype(bool)),
                        ("pathogenic", df["label"].values.astype(bool)),
                        ("benign", ~df["label"].values.astype(bool))]:
        if mask.sum() < 10:
            continue
        print(f"  {label:12s} (n={mask.sum():5d}): "
              f"w_prot={gp[mask].mean():.3f}±{gp[mask].std():.3f}  "
              f"w_dna={gd[mask].mean():.3f}±{gd[mask].std():.3f}  "
              f"cos(zp,zd)={cos[mask].mean():.3f}±{cos[mask].std():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="results/variant/brca1_variants.csv")
    ap.add_argument("--esm", default="results/variant/brca1_esm_delta.npz")
    ap.add_argument("--evo2", default="results/variant/brca1_evo2.npz")
    ap.add_argument("--checkpoint", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--output_dir", default="results/interpret")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sae_hidden", type=int, default=1024)
    ap.add_argument("--sae_k", type=int, default=32)
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

    logger.info("extracting shared-space representations ...")
    reps = extract_representations(df, pdelta, edelta, llr_imp, pmask, args.checkpoint, dev)
    logger.info(f"z_fused shape: {reps['z_fused'].shape}")

    # Part 1: Probing
    probe_df = part1_probing(reps, df)
    probe_df.to_csv(os.path.join(args.output_dir, "probe_results.csv"), index=False)

    # Part 2: SAE
    sae_cards, sae_acts = part2_sae(reps, df, dev, d_hidden=args.sae_hidden, k=args.sae_k)
    sae_cards.to_csv(os.path.join(args.output_dir, "sae_feature_cards.csv"), index=False)
    np.savez_compressed(os.path.join(args.output_dir, "sae_activations.npz"), acts=sae_acts)

    # Part 3: Gate analysis
    part3_gate(reps, df)

    # save all representations for further analysis
    np.savez_compressed(os.path.join(args.output_dir, "shared_reps.npz"),
                        z_fused=reps["z_fused"], zp=reps["zp"], zd=reps["zd"],
                        gate_prot=reps["gate_prot"], gate_dna=reps["gate_dna"])
    print(f"\nsaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
