"""
Step 2: Extract Evo2 edelta embeddings for ProteinGym DMS assays.
Requires CDS sequences — maps protein variants to single-nt codon changes.
Loads existing ESM-2 .npz files to get mutation lists.
"""

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_dms import parse_mutant, mutate_cds_single_nt, AA_TO_CODONS


def extract_evo2_deltas_batched(cds_seq, mutations, evo_model, device, window=8192):
    """Extract Evo2 edelta, one variant at a time (Evo2 doesn't batch well)."""
    LAYER = "blocks.28.mlp.l3"
    n = len(mutations)
    deltas = np.zeros((n, 4096), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, (pos0, to_aa) in enumerate(mutations):
        alt_cds = mutate_cds_single_nt(cds_seq, pos0, to_aa)
        if alt_cds is None:
            continue

        codon_start = pos0 * 3
        center = codon_start + 1
        s = max(0, center - window // 2)
        e = min(len(cds_seq), center + window // 2)
        ref_win = cds_seq[s:e]
        alt_win = alt_cds[s:e]
        c = center - s

        try:
            ref_ids = torch.tensor(
                evo_model.tokenizer.tokenize(ref_win),
                dtype=torch.int).unsqueeze(0).to(device)
            alt_ids = torch.tensor(
                evo_model.tokenizer.tokenize(alt_win),
                dtype=torch.int).unsqueeze(0).to(device)
            with torch.no_grad():
                _, ref_emb = evo_model(ref_ids, return_embeddings=True, layer_names=[LAYER])
                _, alt_emb = evo_model(alt_ids, return_embeddings=True, layer_names=[LAYER])
            deltas[i] = (alt_emb[LAYER][0, c] - ref_emb[LAYER][0, c]).float().cpu().numpy()
            valid[i] = True
        except Exception as ex:
            if i < 3:
                print(f"      Evo2 error at mut {i}: {ex}", flush=True)
    return deltas, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm2_dir", default="results/dms_embeddings",
                    help="Dir with ESM-2 .npz files (to get mutation lists)")
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--cds_json", default="data/proteingym/proteingym_cds.json")
    ap.add_argument("--evo2_path", default="models/evo2_7b.pt")
    ap.add_argument("--out_dir", default="results/dms_embeddings")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gene data for CDS sequences
    gene_cds = {}
    if os.path.exists(args.gene_pairs):
        with open(args.gene_pairs) as f:
            gp = json.load(f)
        for g in gp:
            name = g.get("gene_name", "").upper()
            if "cds_seq" in g:
                gene_cds[name] = g["cds_seq"]
    print(f"Gene CDS from gene_pairs: {len(gene_cds)}", flush=True)

    # Also load ProteinGym-specific CDS (keyed by DMS_id)
    pg_cds = {}
    if os.path.exists(args.cds_json):
        with open(args.cds_json) as f:
            pg_cds = json.load(f)
        print(f"ProteinGym CDS from {args.cds_json}: {len(pg_cds)}", flush=True)

    # Also try to get CDS from DMS reference file
    ref_csv = "data/proteingym/DMS_substitutions.csv"
    if os.path.exists(ref_csv):
        ref = pd.read_csv(ref_csv)

    # Load Evo2
    print("Loading Evo2 ...", flush=True)
    from evo2.models import Evo2
    evo_model = Evo2("evo2_7b", local_path=args.evo2_path)
    print("  Evo2 loaded", flush=True)

    # Process each DMS assay directly from CSV files (no ESM-2 dependency)
    dms_files = sorted(glob.glob(os.path.join(args.dms_dir, "*.csv")))
    print(f"DMS CSV files found: {len(dms_files)}", flush=True)

    done = 0
    skipped_no_cds = 0
    for dms_path in dms_files:
        dms_id = os.path.basename(dms_path).replace(".csv", "")
        evo2_path = os.path.join(args.out_dir, f"{dms_id}_evo2.npz")
        if os.path.exists(evo2_path):
            done += 1
            continue

        # Get CDS: try ProteinGym-specific first, then gene_pairs
        cds_seq = pg_cds.get(dms_id, "")
        if not cds_seq:
            gene = dms_id.split("_")[0].upper()
            cds_seq = gene_cds.get(gene, "")
        if not cds_seq:
            skipped_no_cds += 1
            continue

        # Read mutations from DMS CSV directly
        dms = pd.read_csv(dms_path)
        if "mutant" not in dms.columns:
            continue
        dms = dms[~dms["mutant"].str.contains(":", na=False)].reset_index(drop=True)

        prot_seq = dms["target_seq"].iloc[0] if "target_seq" in dms.columns else ""
        if not prot_seq:
            continue

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

        if len(mutations) < 30:
            continue

        print(f"[{done+1}] {dms_id}: {len(mutations)} muts, CDS_len={len(cds_seq)}", flush=True)

        evo_deltas, evo_valid = extract_evo2_deltas_batched(
            cds_seq, mutations, evo_model, device)

        np.savez_compressed(evo2_path,
            evo2_edelta=evo_deltas,
            evo2_valid=evo_valid,
            dms_id=dms_id,
        )
        done += 1
        print(f"    Saved: {evo_valid.sum()}/{len(mutations)} valid", flush=True)

    print(f"\nDone: {done} assays with Evo2 edelta (skipped {skipped_no_cds} without CDS)")


if __name__ == "__main__":
    main()
