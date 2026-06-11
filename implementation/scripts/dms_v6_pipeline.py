"""
DMS benchmark + mechanism decomposition using v6 model + SAE.

Pipeline:
  1. Extract ESM-2 edelta (1280-d) for all DMS variants
  2. Extract Evo2 edelta (4096-d) for variants with single-nt CDS changes
  3. Project through v6 frozen projection layers → 1536-d
  4. Run through frozen SAE → 12288-d activations
  5. Benchmark: Spearman(v6_score, DMS_score) vs baselines
  6. Mechanism analysis: SAE features vs DMS functional subtypes

Supports multi-GPU via DataParallel for ESM-2 inference.
"""

import argparse, glob, json, os, sys, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from train_variant_sae import ProjectionLayers, TopKSAE
from benchmark_dms import (
    parse_mutant, mutate_cds_single_nt, extract_esm2_deltas,
    extract_evo2_deltas, cv_spearman, CODON_TABLE, AA_TO_CODONS,
)


def load_v6_models(sae_dir, device):
    """Load frozen v6 projection layers and SAE from saved checkpoint."""
    sae_ckpt = torch.load(os.path.join(sae_dir, "sae_model.pt"),
                          map_location="cpu", weights_only=False)

    proj = None
    for key in ["proj_state_dict", "proj_state"]:
        if key in sae_ckpt:
            proj = ProjectionLayers().to(device)
            proj.load_state_dict(sae_ckpt[key])
            proj.eval()
            print(f"  Loaded projection layers from '{key}'")
            break

    sae = None
    for key in ["state_dict", "sae_state"]:
        if key in sae_ckpt and "encoder.weight" in sae_ckpt[key]:
            cfg = sae_ckpt.get("config", {})
            n_features = sae_ckpt[key]["encoder.weight"].shape[0]
            k = cfg.get("k", 32)
            d_input = cfg.get("d_input", 1536)
            sae = TopKSAE(d_input, n_features, k=k).to(device)
            sae.load_state_dict(sae_ckpt[key])
            sae.eval()
            print(f"  Loaded SAE from '{key}': {d_input}→{n_features}, k={k}")
            break

    return proj, sae


def get_cds_for_assay(dms_id, ref_csv, gene_pairs_json=None):
    """Try to find CDS sequence for a DMS assay protein."""
    if gene_pairs_json and os.path.exists(gene_pairs_json):
        with open(gene_pairs_json) as f:
            gp = json.load(f)
        gene = dms_id.split("_HUMAN")[0].split("_")[0]
        for g in gp:
            if g.get("gene_name", "").upper() == gene.upper():
                return g.get("cds_seq", "")
    return ""


def process_assay(dms_path, dms_id, prot_seq, cds_seq,
                  esm_model, batch_converter, evo_model,
                  proj, sae, device, args):
    """Process one DMS assay: extract embeddings, project, evaluate."""
    dms = pd.read_csv(dms_path)
    if "mutant" not in dms.columns:
        return None

    # Filter to single mutants only
    dms = dms[~dms["mutant"].str.contains(":", na=False)].reset_index(drop=True)
    if len(dms) < 50:
        return None

    # Use target_seq from DMS file if no prot_seq provided
    if not prot_seq and "target_seq" in dms.columns:
        prot_seq = dms["target_seq"].iloc[0]

    if not prot_seq:
        return None

    # Parse mutations
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
        return None

    dms_valid = dms.iloc[valid_rows].reset_index(drop=True)
    y_score = dms_valid["DMS_score"].values
    n = len(mutations)

    print(f"\n  {dms_id}: {n} mutations, prot_len={len(prot_seq)}", flush=True)

    # ── ESM-2 edelta ──
    esm_deltas = extract_esm2_deltas(
        prot_seq, mutations, esm_model, batch_converter, device,
        bs=args.esm_batch_size)
    esm_nz = np.any(esm_deltas != 0, axis=1).sum()
    print(f"    ESM-2: {esm_nz}/{n} non-zero", flush=True)

    # ── Evo2 edelta (optional) ──
    evo_deltas = None
    evo_valid = None
    if evo_model and cds_seq:
        evo_deltas, evo_valid = extract_evo2_deltas(
            cds_seq, mutations, evo_model, device)
        print(f"    Evo2: {evo_valid.sum()}/{n} valid", flush=True)

    # ── Zero-shot baselines ──
    res = {"assay": dms_id, "n_mutations": n}

    # ESM-2 norm (zero-shot)
    esm_norm = np.linalg.norm(esm_deltas, axis=1)
    rho_esm_norm = stats.spearmanr(y_score, -esm_norm).statistic
    res["rho_esm2_norm"] = rho_esm_norm
    print(f"    ESM-2 norm Spearman: {rho_esm_norm:.3f}", flush=True)

    # ESM-2 ridge CV
    res["rho_esm2_ridge"] = cv_spearman(esm_deltas, y_score)

    # ── v6 projection + SAE (needs both modalities) ──
    has_dual = evo_deltas is not None and evo_valid is not None and evo_valid.sum() > 30
    if has_dual and proj is not None:
        mask = evo_valid
        n_dual = int(mask.sum())
        res["n_dual"] = n_dual

        # Normalize (same as training)
        prot_sub = esm_deltas[mask]
        dna_sub = evo_deltas[mask]
        prot_z = (prot_sub - prot_sub.mean(0)) / (prot_sub.std(0) + 1e-8)
        dna_z = (dna_sub - dna_sub.mean(0)) / (dna_sub.std(0) + 1e-8)
        y_dual = y_score[mask]

        # Project
        with torch.no_grad():
            xp = torch.tensor(prot_z, dtype=torch.float32, device=device)
            xd = torch.tensor(dna_z, dtype=torch.float32, device=device)
            reps = proj(xp, xd).cpu().numpy()

        # v6 representation ridge
        res["rho_v6_rep_ridge"] = cv_spearman(reps, y_dual)

        # Evo2 norm
        evo_norm = np.linalg.norm(dna_sub, axis=1)
        res["rho_evo2_norm"] = stats.spearmanr(y_dual, -evo_norm).statistic

        # Evo2 ridge
        res["rho_evo2_ridge"] = cv_spearman(dna_sub, y_dual)

        # Fusion ridge (concat prot + dna raw)
        concat_raw = np.concatenate([prot_sub, dna_sub], axis=1)
        res["rho_fusion_raw_ridge"] = cv_spearman(concat_raw, y_dual)

        # ── SAE mechanism analysis ──
        if sae is not None:
            with torch.no_grad():
                sae_acts = sae.encode(
                    torch.tensor(reps, dtype=torch.float32, device=device)
                ).cpu().numpy()

            res["sae_alive_features"] = int((sae_acts > 0).any(0).sum())

            # SAE ridge
            active_cols = (sae_acts > 0).any(0)
            if active_cols.sum() > 10:
                res["rho_sae_ridge"] = cv_spearman(sae_acts[:, active_cols], y_dual)

            # Save per-variant SAE activations for mechanism analysis
            np.savez_compressed(
                os.path.join(args.out_dir, f"{dms_id}_sae.npz"),
                sae_acts=sae_acts, y_score=y_dual,
                mutations=[mutations[j] for j in np.where(mask)[0]],
                prot_reps=reps[:, :768], dna_reps=reps[:, 768:],
            )

    else:
        res["n_dual"] = 0
        res["rho_evo2_norm"] = np.nan
        res["rho_evo2_ridge"] = np.nan
        res["rho_v6_rep_ridge"] = np.nan
        res["rho_fusion_raw_ridge"] = np.nan
        res["rho_sae_ridge"] = np.nan

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_dir", default="data/proteingym/substitutions")
    ap.add_argument("--ref_csv", default="data/proteingym/DMS_substitutions.csv")
    ap.add_argument("--gene_pairs", default="data/full/gene_pairs_full.json")
    ap.add_argument("--sae_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/dms_v6")
    ap.add_argument("--esm_batch_size", type=int, default=16)
    ap.add_argument("--skip_evo2", action="store_true")
    ap.add_argument("--evo2_path", default="models/evo2_7b.pt")
    ap.add_argument("--assays", nargs="*", default=None,
                    help="Specific assay IDs to run (default: all human)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load reference ──
    ref = pd.read_csv(args.ref_csv)

    # ── Load ESM-2 ──
    print("Loading ESM-2 ...", flush=True)
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval()
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs for ESM-2", flush=True)
        esm_model = nn.DataParallel(esm_model)
    esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    # For DataParallel, use the underlying module for repr_layers
    esm_raw = esm_model.module if hasattr(esm_model, "module") else esm_model

    # ── Load Evo2 (optional) ──
    evo_model = None
    if not args.skip_evo2 and os.path.exists(args.evo2_path):
        print("Loading Evo2 ...", flush=True)
        from evo2.models import Evo2
        evo_model = Evo2("evo2_7b", local_path=args.evo2_path)

    # ── Load v6 projection + SAE ──
    print("Loading v6 projection + SAE ...", flush=True)
    proj, sae = load_v6_models(args.sae_dir, device)
    if proj is None:
        print("  WARNING: No projection layers in checkpoint. Will retrain.", flush=True)

    # ── Select assays ──
    if args.assays:
        assay_ids = args.assays
    else:
        assay_ids = ref["DMS_id"].tolist()
    print(f"\nProcessing {len(assay_ids)} assays (full ProteinGym benchmark) ...", flush=True)

    # ── Process each assay ──
    results = []
    for dms_id in assay_ids:
        dms_path = os.path.join(args.dms_dir, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            continue

        # Get protein sequence from DMS file
        dms_peek = pd.read_csv(dms_path, nrows=1)
        prot_seq = dms_peek["target_seq"].iloc[0] if "target_seq" in dms_peek.columns else ""

        # Get CDS for Evo2
        cds_seq = get_cds_for_assay(dms_id, args.ref_csv, args.gene_pairs)

        res = process_assay(
            dms_path, dms_id, prot_seq, cds_seq,
            esm_raw, batch_converter, evo_model,
            proj, sae, device, args)

        if res:
            results.append(res)
            for k, v in res.items():
                if k.startswith("rho"):
                    val = f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
                    print(f"    {k}: {val}", flush=True)

    # ── Summary ──
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(args.out_dir, "dms_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("DMS BENCHMARK SUMMARY")
    print("=" * 80)
    for col in rdf.columns:
        if col.startswith("rho"):
            vals = rdf[col].dropna()
            if len(vals) > 0:
                print(f"  {col:30s}: mean={vals.mean():.3f}  median={vals.median():.3f}  n={len(vals)}")

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
