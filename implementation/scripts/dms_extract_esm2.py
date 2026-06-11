"""
Step 1: Extract ESM-2 edelta embeddings for all ProteinGym DMS assays.
Saves per-assay .npz files with edelta, mutation info, and DMS scores.
"""

import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_dms import parse_mutant, extract_esm2_deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--out_dir", default="results/dms_embeddings")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = pd.read_csv(args.ref_csv)
    assay_ids = ref["DMS_id"].tolist()
    print(f"Total assays: {len(assay_ids)}", flush=True)

    print("Loading ESM-2 ...", flush=True)
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    done = 0
    for dms_id in assay_ids:
        out_path = os.path.join(args.out_dir, f"{dms_id}_esm2.npz")
        if os.path.exists(out_path):
            done += 1
            continue

        dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            continue

        dms = pd.read_csv(dms_path)
        if "mutant" not in dms.columns:
            continue
        dms = dms[~dms["mutant"].str.contains(":", na=False)].reset_index(drop=True)
        if len(dms) < 30:
            continue

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

        dms_valid = dms.iloc[valid_rows].reset_index(drop=True)
        print(f"[{done+1}/{len(assay_ids)}] {dms_id}: {len(mutations)} muts, "
              f"seq_len={len(prot_seq)}", flush=True)

        esm_deltas = extract_esm2_deltas(
            prot_seq, mutations, esm_model, batch_converter, device,
            bs=args.batch_size)

        np.savez_compressed(out_path,
            esm2_edelta=esm_deltas,
            y_score=dms_valid["DMS_score"].values,
            y_bin=dms_valid["DMS_score_bin"].values if "DMS_score_bin" in dms_valid else np.array([]),
            mutants=dms_valid["mutant"].values,
            dms_id=dms_id,
            prot_seq=prot_seq,
        )
        done += 1
        nz = np.any(esm_deltas != 0, axis=1).sum()
        print(f"    Saved: {nz}/{len(mutations)} non-zero", flush=True)

    print(f"\nDone: {done} assays extracted to {args.out_dir}/")


if __name__ == "__main__":
    main()
