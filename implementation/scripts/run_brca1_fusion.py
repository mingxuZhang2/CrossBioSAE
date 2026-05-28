"""
BRCA1 cross-modal variant-effect fusion (lightweight design A).

Builds a per-variant fused representation from local ref->alt EMBEDDING DELTAS:
  - Protein (ESM-2, missense only): window around the mutated residue; delta =
    emb(alt)[center] - emb(ref)[center]  (1280-d). Non-missense -> zeros + mask.
  - DNA (NT now, Evo2 later): 8192bp genomic window centered on the variant;
    delta = emb(alt)[center-token] - emb(ref)[center-token]  (1024-d for NT).

Then evaluates Protein-only / DNA-only / Fused on LOF-vs-FUNC with GROUPED CV
(group by genomic position to block leakage), reporting auROC overall + per vtype.

Inputs:  brca1_variants.csv, brca1_P38398.fasta (WT protein), chr17_GRCh37.fna.gz
Output:  brca1_fusion_emb.npz (cached deltas) + brca1_fusion_results.csv
"""

import argparse
import gzip
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PWIN = 510   # protein half-window (residues) -> <=1021 tokens for ESM-2
DWIN = 4096  # DNA half-window (bp) -> 8192bp total (Evo2 recipe)


def load_protein(fasta):
    return "".join(l.strip() for l in open(fasta) if not l.startswith(">"))


def load_chr(fna_gz):
    op = gzip.open if fna_gz.endswith(".gz") else open
    seq = []
    with op(fna_gz, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()


@torch.no_grad()
def extract_protein_deltas(df, wt_prot, device, batch_size=8):
    """ESM-2 per-position delta for missense variants."""
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    D = 1280
    deltas = np.zeros((len(df), D), dtype=np.float32)
    mask = np.zeros(len(df), dtype=bool)

    mis = df[df["is_missense"]].copy()
    logger.info(f"protein: {len(mis)} missense variants")
    items = []  # (row_idx, ref_window, alt_window, center_in_window)
    for ridx, r in mis.iterrows():
        p = int(r["aa_pos"]) - 1
        s = max(0, p - PWIN)
        e = min(len(wt_prot), p + PWIN + 1)
        ref_w = wt_prot[s:e]
        c = p - s
        if ref_w[c] != str(r["aa_ref"]):
            continue
        alt_w = ref_w[:c] + str(r["aa_alt"]) + ref_w[c + 1:]
        items.append((df.index.get_loc(ridx), ref_w, alt_w, c))

    # batch: encode ref and alt windows, grab center-residue rep
    def embed_center(seqs, centers):
        out = []
        for i in range(0, len(seqs), batch_size):
            bseq = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch_size])]
            _, _, toks = bc(bseq)
            toks = toks.to(device)
            reps = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
            for j, (_, s) in enumerate(bseq):
                c = centers[i + j]
                out.append(reps[j, 1 + c].cpu().numpy())  # +1 for BOS
            if (i // batch_size) % 30 == 0:
                logger.info(f"  ESM {i+len(bseq)}/{len(seqs)}")
        return np.array(out)

    idxs = [it[0] for it in items]
    refs = [it[1] for it in items]
    alts = [it[2] for it in items]
    cens = [it[3] for it in items]
    ref_emb = embed_center(refs, cens)
    alt_emb = embed_center(alts, cens)
    for k, ix in enumerate(idxs):
        deltas[ix] = alt_emb[k] - ref_emb[k]
        mask[ix] = True
    del model
    torch.cuda.empty_cache()
    return deltas, mask


@torch.no_grad()
def extract_dna_deltas(df, chrom_seq, device, batch_size=4):
    """NT center-token delta for the 8192bp window around each variant."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    mid = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    local = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         f"models--{mid.replace('/', '--')}/snapshots/main")
    src = local if os.path.isdir(local) else mid
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(src, trust_remote_code=True).to(device).eval()

    D = model.config.hidden_size
    deltas = np.zeros((len(df), D), dtype=np.float32)
    n_bad = 0

    def build_window(pos, ref, alt):
        g = pos - 1  # 0-based
        s = g - DWIN
        e = g + DWIN
        if s < 0 or e > len(chrom_seq):
            return None, None, None
        win = chrom_seq[s:e]
        c = g - s
        if win[c] != ref:
            return None, None, None  # ref mismatch (strand/indel) -> skip
        alt_win = win[:c] + alt + win[c + 1:]
        return win, alt_win, c

    rows, refs, alts, cens = [], [], [], []
    for ridx, r in df.iterrows():
        win, alt_win, c = build_window(int(r["pos_hg19"]), str(r["ref"]), str(r["alt"]))
        if win is None:
            n_bad += 1
            continue
        rows.append(df.index.get_loc(ridx)); refs.append(win); alts.append(alt_win); cens.append(c)
    logger.info(f"dna: {len(rows)} windows ok, {n_bad} skipped (ref mismatch/edge)")

    def embed_center(seqs, centers):
        out = []
        for i in range(0, len(seqs), batch_size):
            bs = seqs[i:i + batch_size]
            bc_ = centers[i:i + batch_size]
            t = tok(bs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            t = {k: v.to(device) for k, v in t.items()}
            h = model(**t, output_hidden_states=True).hidden_states[-1]
            for j in range(len(bs)):
                # NT v2 = non-overlapping 6-mers; center base -> token (center//6)+1 (BOS offset)
                tok_idx = min(1 + bc_[j] // 6, h.shape[1] - 1)
                out.append(h[j, tok_idx].cpu().numpy())
            if (i // batch_size) % 50 == 0:
                logger.info(f"  NT {i+len(bs)}/{len(seqs)}")
        return np.array(out)

    ref_emb = embed_center(refs, cens)
    alt_emb = embed_center(alts, cens)
    for k, ix in enumerate(rows):
        deltas[ix] = alt_emb[k] - ref_emb[k]
    del model
    torch.cuda.empty_cache()
    return deltas


def grouped_auroc(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variant/brca1/brca1_variants.csv")
    ap.add_argument("--protein", default="data/variant/brca1/brca1_P38398.fasta")
    ap.add_argument("--chr17", default="data/variant/brca1/chr17_GRCh37.fna.gz")
    ap.add_argument("--output_dir", default="results/variant")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.variants).reset_index(drop=True)
    wt = load_protein(args.protein)
    chrom = load_chr(args.chr17)
    logger.info(f"variants={len(df)} protein={len(wt)} chr17={len(chrom)}bp")

    cache = os.path.join(args.output_dir, "brca1_fusion_emb.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if c["pdelta"].shape[0] == len(df):
            pdelta, pmask, ddelta = c["pdelta"], c["pmask"], c["ddelta"]
            logger.info("loaded cached deltas")
        else:
            pdelta = None
    else:
        pdelta = None
    if pdelta is None:
        pdelta, pmask = extract_protein_deltas(df, wt, args.device)
        ddelta = extract_dna_deltas(df, chrom, args.device)
        np.savez_compressed(cache, pdelta=pdelta, pmask=pmask, ddelta=ddelta)

    y = df["label"].values
    groups = df["pos_hg19"].values  # group by genomic position (block leakage)
    fused = np.hstack([pdelta, ddelta, pmask[:, None].astype(np.float32)])

    reps = {
        "Protein-only (ESM dpos)": pdelta,
        "DNA-only (NT dtok)": ddelta,
        "Fused": fused,
    }
    rows = []
    print("\n" + "=" * 64)
    print("BRCA1 cross-modal fusion (auROC, GroupKFold by position)")
    print("=" * 64)
    for name, X in reps.items():
        pred = grouped_auroc(X, y, groups)
        au = roc_auc_score(y, pred)
        # per variant-type
        per = {}
        for vt in ["coding", "noncoding"]:
            m = (df["vtype"] == vt).values
            if m.sum() > 20 and len(np.unique(y[m])) == 2:
                per[vt] = roc_auc_score(y[m], pred[m])
        mis = df["is_missense"].values
        per["missense"] = roc_auc_score(y[mis], pred[mis]) if len(np.unique(y[mis])) == 2 else np.nan
        print(f"  {name:24s}: all={au:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in per.items()))
        rows.append({"rep": name, "auroc_all": au, **{f"auroc_{k}": v for k, v in per.items()}})

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "brca1_fusion_results.csv"), index=False)
    print("\nNote: DNA side = NT (weak placeholder). Swap Evo2 for the real number.")


if __name__ == "__main__":
    main()
