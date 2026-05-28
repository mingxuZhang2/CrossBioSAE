"""
eSOL cross-modal solubility pilot.

Question: does the CDS (DNA, codon usage) carry solubility signal that the protein
sequence alone does not? Compare protein-only / DNA-only / concat on eSOL solubility
regression. Existing baselines (SoluProt, GraphSol R2~=0.48) are ALL protein-only, so
a concat>protein gain is the cross-modal proof-of-concept.

Two evaluations:
  - 5-fold CV (random) on verified-match set: Spearman, Pearson, R2.
  - Official GraphSol train/test split: direct comparison to published numbers.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_esm2(seqs, device, batch_size=8):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = [(f"s{j}", s[:1022]) for j, s in enumerate(seqs[i:i + batch_size])]
        _, _, toks = bc(batch)
        toks = toks.to(device)
        reps = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        for j, (_, s) in enumerate(batch):
            out.append(reps[j, 1:len(s) + 1].mean(0).cpu().numpy())
        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM-2 {i + len(batch)}/{len(seqs)}")
    del model
    torch.cuda.empty_cache()
    return np.array(out)


@torch.no_grad()
def extract_nt(seqs, device, batch_size=8):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    mid = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    local = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         f"models--{mid.replace('/', '--')}/snapshots/main")
    src = local if os.path.isdir(local) else mid
    logger.info(f"  NT from {src}")
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(src, trust_remote_code=True).to(device).eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        t = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        t = {k: v.to(device) for k, v in t.items()}
        h = model(**t, output_hidden_states=True).hidden_states[-1]
        mask = t["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)
        out.append(pooled.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            logger.info(f"  NT {i + len(batch)}/{len(seqs)}")
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, 0)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_predict(Xtr, ytr, Xte, device, epochs=200):
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=device)
    m = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        lf(m(Xtr_t), ytr_t).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(Xte_t).cpu().numpy()


def _metrics(y, p):
    return (spearmanr(y, p)[0], pearsonr(y, p)[0], r2_score(y, p))


def eval_cv(reps, y, device, n_splits=5, seed=42):
    logger.info(f"\n=== 5-fold CV (random, seed={seed}) ===")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = {}
    for name, X in reps.items():
        sp, pe, r2 = [], [], []
        for tr, te in kf.split(X):
            p = _fit_predict(X[tr], y[tr], X[te], device)
            s, r, rr = _metrics(y[te], p)
            sp.append(s); pe.append(r); r2.append(rr)
        res[name] = (np.mean(sp), np.mean(pe), np.mean(r2))
        logger.info(f"  {name:20s} ({X.shape[1]:4d}d): Spearman={np.mean(sp):.4f} "
                    f"Pearson={np.mean(pe):.4f} R2={np.mean(r2):.4f}")
    return res


def eval_split(reps, y, is_train, device):
    logger.info("\n=== Official GraphSol train/test split ===")
    res = {}
    tr = np.where(is_train)[0]
    te = np.where(~is_train)[0]
    logger.info(f"  train={len(tr)} test={len(te)}")
    for name, X in reps.items():
        p = _fit_predict(X[tr], y[tr], X[te], device)
        s, r, rr = _metrics(y[te], p)
        res[name] = (s, r, rr)
        logger.info(f"  {name:20s} ({X.shape[1]:4d}d): Spearman={s:.4f} Pearson={r:.4f} R2={rr:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched_csv", default="data/solubility/esol_matched.csv")
    ap.add_argument("--output_dir", default="results/solubility")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.matched_csv)
    # keep only verified protein<->CDS correspondence
    df = df[df["match"].isin(["gene+verified", "protein-seq"])].reset_index(drop=True)
    logger.info(f"Verified-match proteins: {len(df)} "
                f"(train={ (df['split']=='train').sum() } test={ (df['split']=='test').sum() })")

    y = df["solubility"].values.astype(np.float32)
    is_train = (df["split"] == "train").values

    cache = os.path.join(args.output_dir, "esol_emb.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if c["prot"].shape[0] == len(df):
            logger.info(f"Loaded cache {cache}")
            prot, dna = c["prot"], c["dna"]
        else:
            prot = dna = None
    else:
        prot = dna = None

    if prot is None:
        logger.info("Extracting ESM-2...")
        prot = extract_esm2(df["protein"].tolist(), args.device)
        logger.info(f"  protein {prot.shape}")
        logger.info("Extracting NT...")
        dna = extract_nt(df["cds"].tolist(), args.device)
        logger.info(f"  dna {dna.shape}")
        np.savez_compressed(cache, prot=prot, dna=dna)

    reps = {
        "Protein-only (ESM2)": prot,
        "DNA-only (NT)": dna,
        "Concat": np.hstack([prot, dna]),
    }

    rcv = eval_cv(reps, y, args.device)
    rsp = eval_split(reps, y, is_train, args.device)

    print("\n" + "=" * 72)
    print("eSOL SOLUBILITY PILOT  (verified protein<->CDS, n=%d)" % len(df))
    print("=" * 72)
    print("\n5-fold CV (Spearman / Pearson / R2):")
    for k, (s, p, r) in rcv.items():
        print(f"   {k:20s}: {s:.4f} / {p:.4f} / {r:.4f}")
    print("\nOfficial split (Spearman / Pearson / R2):")
    for k, (s, p, r) in rsp.items():
        print(f"   {k:20s}: {s:.4f} / {p:.4f} / {r:.4f}")
    base = rcv["Protein-only (ESM2)"][2]
    fused = rcv["Concat"][2]
    print(f"\nCross-modal delta (CV R2): Concat - Protein = {fused - base:+.4f}")
    print("GraphSol (protein-only baseline) reports R2 ~= 0.48 on eSOL regression.")

    pd.DataFrame([
        {"eval": "cv5", "rep": k, "spearman": s, "pearson": p, "r2": r}
        for k, (s, p, r) in rcv.items()
    ] + [
        {"eval": "split", "rep": k, "spearman": s, "pearson": p, "r2": r}
        for k, (s, p, r) in rsp.items()
    ]).to_csv(os.path.join(args.output_dir, "esol_pilot_results.csv"), index=False)


if __name__ == "__main__":
    main()
