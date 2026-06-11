"""
DMS assay benchmark: ESM-2 vs Evo2 vs CLIP-fusion.

For each ProteinGym DMS assay:
  1. Parse mutations (FROM_AA + POS + TO_AA)
  2. Extract ESM-2 edelta at mutation position (protein side)
  3. Extract Evo2 edelta at mutation position (DNA side, via CDS)
  4. Project through CLIP → z_prot(256), z_dna(256)
  5. Score: logistic regression on z → predict DMS_score_bin (or Spearman with DMS_score)

Metrics: Spearman ρ(score, DMS_score) per assay, and AUC on DMS_score_bin.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from contrastive_pretrain import CrossModalCLIP

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L',
    'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S',
    'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
    'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS = {}
for codon, aa in CODON_TABLE.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)


def parse_mutant(m):
    """Parse 'M1A' → (from_aa, pos_1based, to_aa)."""
    return m[0], int(m[1:-1]), m[-1]


def mutate_cds_single_nt(cds, pos0, to_aa):
    """Find single-nucleotide CDS mutation at codon pos0 that gives to_aa.
    Returns mutated CDS or None if >1 nt change needed."""
    codon_start = pos0 * 3
    if codon_start + 3 > len(cds):
        return None
    ref_codon = cds[codon_start:codon_start + 3]
    for alt_codon in AA_TO_CODONS.get(to_aa, []):
        diffs = sum(1 for a, b in zip(ref_codon, alt_codon) if a != b)
        if diffs == 1:
            return cds[:codon_start] + alt_codon + cds[codon_start + 3:]
    return None


def extract_esm2_deltas(prot_seq, mutations, model, batch_converter, device, bs=8):
    """Extract ESM-2 edelta for a list of (pos0, to_aa) on one protein."""
    ESM_LAYER = 33
    n = len(mutations)
    deltas = np.zeros((n, 1280), dtype=np.float32)

    for i in range(0, n, bs):
        batch = mutations[i:i + bs]
        ref_data = [(f"r{j}", prot_seq) for j in range(len(batch))]
        alt_data = []
        positions = []
        for j, (pos0, to_aa) in enumerate(batch):
            alt_seq = prot_seq[:pos0] + to_aa + prot_seq[pos0 + 1:]
            alt_data.append((f"a{j}", alt_seq))
            positions.append(pos0 + 1)  # +1 for BOS

        try:
            _, _, rt = batch_converter(ref_data)
            rt = rt[:, :1024].to(device)
            _, _, at = batch_converter(alt_data)
            at = at[:, :1024].to(device)
            with torch.no_grad():
                re = model(rt, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                ae = model(at, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
            for j, pos in enumerate(positions):
                if pos < re.shape[1] and pos < ae.shape[1]:
                    deltas[i + j] = (ae[j, pos] - re[j, pos]).float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for j, (pos0, to_aa) in enumerate(batch):
                try:
                    alt_seq = prot_seq[:pos0] + to_aa + prot_seq[pos0 + 1:]
                    _, _, rt = batch_converter([("r", prot_seq)])
                    rt = rt[:, :1024].to(device)
                    _, _, at = batch_converter([("a", alt_seq)])
                    at = at[:, :1024].to(device)
                    with torch.no_grad():
                        re = model(rt, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                        ae = model(at, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
                    p = pos0 + 1
                    if p < re.shape[1]:
                        deltas[i + j] = (ae[0, p] - re[0, p]).float().cpu().numpy()
                except:
                    pass
    return deltas


def extract_evo2_deltas(cds_seq, mutations, evo_model, device):
    """Extract Evo2 edelta for mutations that have single-nt CDS changes."""
    LAYER = "blocks.28.mlp.l3"
    WINDOW = 8192
    n = len(mutations)
    deltas = np.zeros((n, 4096), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, (pos0, to_aa) in enumerate(mutations):
        alt_cds = mutate_cds_single_nt(cds_seq, pos0, to_aa)
        if alt_cds is None:
            continue

        codon_start = pos0 * 3
        center = codon_start + 1
        s = max(0, center - WINDOW // 2)
        e = min(len(cds_seq), center + WINDOW // 2)
        ref_win = cds_seq[s:e]
        alt_win = alt_cds[s:e]
        c = center - s

        try:
            ref_ids = torch.tensor(evo_model.tokenizer.tokenize(ref_win), dtype=torch.int).unsqueeze(0).to(device)
            alt_ids = torch.tensor(evo_model.tokenizer.tokenize(alt_win), dtype=torch.int).unsqueeze(0).to(device)
            with torch.no_grad():
                _, ref_emb = evo_model(ref_ids, return_embeddings=True, layer_names=[LAYER])
                _, alt_emb = evo_model(alt_ids, return_embeddings=True, layer_names=[LAYER])
            deltas[i] = (alt_emb[LAYER][0, c] - ref_emb[LAYER][0, c]).float().cpu().numpy()
            valid[i] = True
        except:
            pass

    return deltas, valid


def cv_spearman(X, y, n_splits=5):
    """5-fold CV ridge regression, return Spearman ρ on held-out."""
    if X.shape[0] < 20:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    for tr, te in kf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    rho, _ = stats.spearmanr(y, preds)
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--clip_ckpt", default="results/pretrain/crossmodal_clip_40k.pt")
    ap.add_argument("--out_dir", default="results/dms_benchmark")
    ap.add_argument("--skip_evo2", action="store_true")
    ap.add_argument("--evo2_path", default="models/evo2_7b.pt")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load gene data
    with open(args.gene_pairs) as f:
        gp = json.load(f)
    gene_data = {g["gene_name"]: g for g in gp if "protein_seq" in g}

    # load reference
    ref = pd.read_csv(args.ref_csv)
    ref["gene"] = ref["DMS_id"].str.split("_").str[0]

    # find available DMS CSVs
    dms_files = glob.glob(os.path.join(args.dms_dir, "*.csv"))
    dms_map = {os.path.basename(f): f for f in dms_files}

    # load ESM-2
    print("loading ESM-2 ...", flush=True)
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # load Evo2 (optional)
    evo_model = None
    if not args.skip_evo2 and os.path.exists(args.evo2_path):
        print("loading Evo2 ...", flush=True)
        from evo2.models import Evo2
        evo_model = Evo2("evo2_7b", local_path=args.evo2_path)

    # load CLIP
    print("loading CLIP ...", flush=True)
    ck = torch.load(args.clip_ckpt, map_location="cpu", weights_only=False)
    clip = CrossModalCLIP(**ck["config"]).eval().to(device)
    clip.load_state_dict(ck["model_state"])
    del ck

    results = []
    for _, row in ref.iterrows():
        fname = row["DMS_filename"]
        gene = row["gene"]
        if fname not in dms_map or gene not in gene_data:
            continue

        dms = pd.read_csv(dms_map[fname])
        if "mutant" not in dms.columns or len(dms) < 10:
            continue
        dms = dms[~dms["mutant"].str.contains(":")].reset_index(drop=True)  # singles only
        if len(dms) < 50:
            continue

        prot_seq = gene_data[gene]["protein_seq"]
        cds_seq = gene_data[gene].get("cds_seq", "")

        # parse mutations
        mutations = []
        valid_rows = []
        for i, m in enumerate(dms["mutant"]):
            try:
                from_aa, pos, to_aa = parse_mutant(m)
                pos0 = pos - 1
                if 0 <= pos0 < len(prot_seq) and prot_seq[pos0] == from_aa:
                    mutations.append((pos0, to_aa))
                    valid_rows.append(i)
            except:
                pass

        if len(mutations) < 50:
            continue

        dms_valid = dms.iloc[valid_rows].reset_index(drop=True)
        y_score = dms_valid["DMS_score"].values
        y_bin = dms_valid["DMS_score_bin"].values if "DMS_score_bin" in dms_valid else None

        print(f"\n{fname}: {len(mutations)} mutations on {gene} (prot_len={len(prot_seq)})", flush=True)

        # ESM-2
        esm_deltas = extract_esm2_deltas(prot_seq, mutations, esm_model, batch_converter, device)
        esm_nz = np.any(esm_deltas != 0, axis=1).sum()
        print(f"  ESM-2: {esm_nz}/{len(mutations)} non-zero", flush=True)

        # Evo2
        evo_deltas = None
        evo_valid = None
        if evo_model and cds_seq:
            evo_deltas, evo_valid = extract_evo2_deltas(cds_seq, mutations, evo_model, device)
            print(f"  Evo2: {evo_valid.sum()}/{len(mutations)} valid (single-nt CDS change)", flush=True)

        # CLIP projections
        with torch.no_grad():
            z_prot = clip.enc_prot(torch.tensor(esm_deltas, dtype=torch.float32, device=device)).cpu().numpy()

        z_dna = None
        if evo_deltas is not None:
            with torch.no_grad():
                z_dna = clip.enc_dna(torch.tensor(evo_deltas, dtype=torch.float32, device=device)).cpu().numpy()

        # scores: norm-based (zero-shot) + ridge CV (supervised)
        res = {"assay": fname, "gene": gene, "n_mutations": len(mutations)}

        # ESM2-norm (zero-shot)
        esm_norm = np.linalg.norm(esm_deltas, axis=1)
        res["rho_esm2_norm"] = stats.spearmanr(y_score, esm_norm).statistic

        # CLIP-prot ridge
        res["rho_clip_prot"] = cv_spearman(z_prot, y_score)

        # Evo2 + fusion (if available)
        if evo_deltas is not None and evo_valid.sum() > 50:
            # only use variants with valid Evo2
            mask = evo_valid
            evo_norm = np.linalg.norm(evo_deltas[mask], axis=1)
            res["rho_evo2_norm"] = stats.spearmanr(y_score[mask], evo_norm).statistic
            res["rho_clip_dna"] = cv_spearman(z_dna[mask], y_score[mask])
            z_concat = np.concatenate([z_prot[mask], z_dna[mask]], axis=1)
            res["rho_clip_fusion"] = cv_spearman(z_concat, y_score[mask])
            res["n_dual"] = int(mask.sum())
        else:
            # protein-only fusion = just protein
            z_concat = z_prot
            res["rho_clip_fusion"] = res["rho_clip_prot"]
            res["rho_evo2_norm"] = np.nan
            res["rho_clip_dna"] = np.nan
            res["n_dual"] = 0

        # AUC on binary label (if available)
        if y_bin is not None and len(set(y_bin)) == 2:
            res["auc_clip_prot"] = roc_auc_score(y_bin, -esm_norm)  # lower norm = loss of function
            if evo_deltas is not None and evo_valid.sum() > 50:
                mask = evo_valid
                res["auc_clip_fusion"] = roc_auc_score(
                    y_bin[mask],
                    Ridge(alpha=1.0).fit(
                        StandardScaler().fit_transform(np.concatenate([z_prot[mask], z_dna[mask]], axis=1)),
                        y_bin[mask]
                    ).predict(StandardScaler().fit_transform(np.concatenate([z_prot[mask], z_dna[mask]], axis=1)))
                )

        results.append(res)
        for k, v in res.items():
            if k.startswith("rho") or k.startswith("auc"):
                print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: N/A", flush=True)

    # summary
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(args.out_dir, "dms_benchmark.csv"), index=False)

    print("\n=== Summary ===", flush=True)
    for col in ["rho_esm2_norm", "rho_clip_prot", "rho_evo2_norm", "rho_clip_dna", "rho_clip_fusion"]:
        if col in rdf:
            vals = rdf[col].dropna()
            if len(vals) > 0:
                print(f"  {col:20s}: mean={vals.mean():.3f}  median={vals.median():.3f}  n={len(vals)}", flush=True)

    # paired test: fusion vs protein-only
    both = rdf.dropna(subset=["rho_clip_prot", "rho_clip_fusion"])
    if len(both) > 3:
        w = stats.wilcoxon(both["rho_clip_fusion"], both["rho_clip_prot"], alternative="greater")
        print(f"\n  fusion vs prot-only: Wilcoxon p={w.pvalue:.4f} (n={len(both)} assays)", flush=True)

    print(f"\nsaved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
