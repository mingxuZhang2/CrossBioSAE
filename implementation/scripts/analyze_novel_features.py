"""
Deep characterization of "unresolved" high-pathogenic SAE features.
Goal: identify what biological concept each feature captures that our
existing templates (cys_loss, pro_intro, charge_rev, etc.) missed.

Analysis dimensions per feature:
1. Gene enrichment (Fisher exact vs background)
2. AA substitution enrichment
3. Position-in-protein distribution (N-term / middle / C-term)
4. Secondary structure context (from_aa patterns)
5. ClinVar condition enrichment
6. Cross-feature clustering (do groups of novel features share patterns?)
"""

import argparse, os, json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def load_data(args):
    print("Loading variant metadata ...", flush=True)
    df_full = pd.read_csv(args.annotated_csv, low_memory=False)
    dual_idx = np.load(args.dual_idx)
    df = df_full.iloc[dual_idx].reset_index(drop=True)

    print("Loading SAE activations ...", flush=True)
    sae_acts = np.load(os.path.join(args.sae_dir, "sae_acts.npz"))["acts"]
    concepts = pd.read_csv(os.path.join(args.sae_dir, "sae_concepts.csv"))

    is_path = df["clin_sig"].str.contains("Pathogenic", case=False, na=False) & \
              ~df["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    is_ben = df["clin_sig"].str.contains("Benign", case=False, na=False) & \
             ~df["clin_sig"].str.contains("Uncertain|Conflicting", case=False, na=False)
    df["label"] = -1
    df.loc[is_path, "label"] = 1
    df.loc[is_ben, "label"] = 0
    df["sub"] = df["from_aa"] + ">" + df["to_aa"]

    print(f"  {len(df)} variants, {sae_acts.shape[1]} features", flush=True)
    return df, sae_acts, concepts


def get_activating_variants(sae_acts, feature_idx, top_frac=0.1):
    col = sae_acts[:, feature_idx]
    threshold = np.percentile(col[col > 0], 100 - top_frac * 100) if (col > 0).sum() > 10 else 0
    mask = col > max(threshold, 1e-6)
    return mask


def gene_enrichment(df_feat, df_bg, top_n=15):
    """Fisher exact test for gene enrichment."""
    feat_genes = Counter(df_feat["gene"])
    bg_genes = Counter(df_bg["gene"])
    n_feat, n_bg = len(df_feat), len(df_bg)
    results = []
    for gene, count in feat_genes.most_common(50):
        bg_count = bg_genes.get(gene, 0)
        table = [[count, n_feat - count],
                 [bg_count, n_bg - bg_count]]
        odds, pval = stats.fisher_exact(table, alternative="greater")
        if pval < 0.05:
            results.append({
                "gene": gene,
                "count_feat": count,
                "frac_feat": count / n_feat,
                "count_bg": bg_count,
                "frac_bg": bg_count / n_bg,
                "odds_ratio": odds,
                "pval": pval
            })
    results.sort(key=lambda x: x["pval"])
    return results[:top_n]


def substitution_enrichment(df_feat, df_bg, top_n=10):
    """Which AA substitutions are enriched?"""
    feat_subs = Counter(df_feat["sub"])
    bg_subs = Counter(df_bg["sub"])
    n_feat, n_bg = len(df_feat), len(df_bg)
    results = []
    for sub, count in feat_subs.most_common(30):
        bg_count = bg_subs.get(sub, 0)
        table = [[count, n_feat - count],
                 [bg_count, n_bg - bg_count]]
        odds, pval = stats.fisher_exact(table, alternative="greater")
        if count >= 3:
            results.append({
                "sub": sub, "count": count,
                "frac_feat": count / n_feat,
                "frac_bg": bg_count / n_bg,
                "odds_ratio": odds, "pval": pval
            })
    results.sort(key=lambda x: x["pval"])
    return results[:top_n]


def position_analysis(df_feat):
    """Analyze position within protein."""
    positions = df_feat["prot_pos"].dropna().astype(int)
    if len(positions) < 5:
        return {}
    return {
        "median_pos": int(positions.median()),
        "mean_pos": float(positions.mean()),
        "std_pos": float(positions.std()),
        "min_pos": int(positions.min()),
        "max_pos": int(positions.max()),
    }


def condition_enrichment(df_feat, df_bg, top_n=10):
    """Enrichment in ClinVar conditions (if available)."""
    if "condition" not in df_feat.columns:
        return []
    feat_conds = Counter(df_feat["condition"].dropna())
    bg_conds = Counter(df_bg["condition"].dropna())
    n_feat = df_feat["condition"].notna().sum()
    n_bg = df_bg["condition"].notna().sum()
    if n_feat < 5:
        return []
    results = []
    for cond, count in feat_conds.most_common(30):
        if cond in ("not provided", "not specified", ""):
            continue
        bg_count = bg_conds.get(cond, 0)
        table = [[count, n_feat - count],
                 [bg_count, n_bg - bg_count]]
        odds, pval = stats.fisher_exact(table, alternative="greater")
        if count >= 3 and pval < 0.05:
            results.append({
                "condition": cond, "count": count,
                "frac": count / n_feat, "odds": odds, "pval": pval
            })
    results.sort(key=lambda x: x["pval"])
    return results[:top_n]


def aa_property_profile(df_feat):
    """Characterize amino acid property changes."""
    from_aa = Counter(df_feat["from_aa"])
    to_aa = Counter(df_feat["to_aa"])

    HYDRO = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
             "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
             "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    CHARGE = {"D":-1,"E":-1,"K":1,"R":1,"H":0.5}
    VOLUME = {"G":60,"A":89,"V":140,"L":166,"I":166,"P":115,"F":190,"W":228,
              "M":163,"S":89,"T":116,"C":108,"Y":194,"H":153,"D":111,"E":138,
              "N":114,"Q":143,"K":169,"R":173}

    hydro_changes, charge_changes, vol_changes = [], [], []
    for _, row in df_feat.iterrows():
        f, t = row.get("from_aa", ""), row.get("to_aa", "")
        if f in HYDRO and t in HYDRO:
            hydro_changes.append(HYDRO[t] - HYDRO[f])
        if f in VOLUME and t in VOLUME:
            vol_changes.append(VOLUME[t] - VOLUME[f])
        fc = CHARGE.get(f, 0)
        tc = CHARGE.get(t, 0)
        charge_changes.append(tc - fc)

    result = {}
    if hydro_changes:
        result["mean_hydro_change"] = np.mean(hydro_changes)
        result["std_hydro_change"] = np.std(hydro_changes)
    if vol_changes:
        result["mean_vol_change"] = np.mean(vol_changes)
    if charge_changes:
        result["mean_charge_change"] = np.mean(charge_changes)
        result["frac_charge_change"] = np.mean([abs(c) > 0.5 for c in charge_changes])

    result["top_from_aa"] = from_aa.most_common(3)
    result["top_to_aa"] = to_aa.most_common(3)
    return result


def characterize_feature(feat_idx, df, sae_acts, concepts_row):
    """Full characterization of one feature."""
    mask = get_activating_variants(sae_acts, feat_idx)
    df_feat = df[mask]
    df_bg = df[~mask]

    labeled_feat = df_feat[df_feat["label"] >= 0]
    path_rate = labeled_feat["label"].mean() if len(labeled_feat) > 0 else None

    result = {
        "feature": feat_idx,
        "n_active": int(mask.sum()),
        "n_labeled": len(labeled_feat),
        "path_rate": path_rate,
        "n_genes": df_feat["gene"].nunique(),
        "modality_ratio": concepts_row.get("modality_ratio", None),
    }

    result["gene_enrichment"] = gene_enrichment(df_feat, df_bg)
    result["sub_enrichment"] = substitution_enrichment(df_feat, df_bg)
    result["position"] = position_analysis(df_feat)
    result["aa_properties"] = aa_property_profile(df_feat)
    result["condition_enrichment"] = condition_enrichment(df_feat, df_bg)

    return result


def propose_concept(char):
    """Heuristic: propose a concept name based on enrichment patterns."""
    clues = []

    # Gene enrichment clue
    genes = char.get("gene_enrichment", [])
    if genes and genes[0]["odds_ratio"] > 5:
        top_genes = [g["gene"] for g in genes[:3] if g["odds_ratio"] > 3]
        if top_genes:
            clues.append(f"gene-enriched({','.join(top_genes)})")

    # Substitution clue
    subs = char.get("sub_enrichment", [])
    if subs and subs[0]["odds_ratio"] > 3:
        top_subs = [s["sub"] for s in subs[:3] if s["odds_ratio"] > 2]
        if top_subs:
            clues.append(f"sub({','.join(top_subs)})")

    # AA property clue
    props = char.get("aa_properties", {})
    if abs(props.get("mean_charge_change", 0)) > 0.5:
        clues.append("charge-change")
    if abs(props.get("mean_hydro_change", 0)) > 2.5:
        clues.append("hydrophobicity-change")
    if abs(props.get("mean_vol_change", 0)) > 40:
        clues.append("volume-change")
    if props.get("frac_charge_change", 0) > 0.6:
        clues.append("frequent-charge-change")

    # Modality clue
    ratio = char.get("modality_ratio")
    if ratio is not None:
        if ratio < 0.35:
            clues.append("strongly-DNA-driven")
        elif ratio > 0.65:
            clues.append("strongly-protein-driven")

    return " + ".join(clues) if clues else "truly-novel"


def cluster_features(all_chars):
    """Cluster features by their enrichment profiles."""
    from collections import defaultdict

    gene_groups = defaultdict(list)
    sub_groups = defaultdict(list)

    for char in all_chars:
        genes = char.get("gene_enrichment", [])
        top_gene = genes[0]["gene"] if genes and genes[0]["pval"] < 0.001 else "none"
        gene_groups[top_gene].append(char["feature"])

        subs = char.get("sub_enrichment", [])
        top_sub = subs[0]["sub"] if subs and subs[0]["pval"] < 0.001 else "none"
        sub_groups[top_sub].append(char["feature"])

    return {
        "gene_clusters": {k: v for k, v in gene_groups.items() if len(v) >= 2 and k != "none"},
        "sub_clusters": {k: v for k, v in sub_groups.items() if len(v) >= 2 and k != "none"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_csv", default="data/variant/sae_pretrain/missense_500k_annotated.csv")
    ap.add_argument("--dual_idx", default="data/variant/sae_pretrain/dual_idx.npy")
    ap.add_argument("--sae_dir", default="results/variant_sae_v6")
    ap.add_argument("--out_dir", default="results/novel_features")
    ap.add_argument("--min_path_rate", type=float, default=0.9)
    ap.add_argument("--min_active", type=int, default=30)
    ap.add_argument("--max_gene_frac", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df, sae_acts, concepts = load_data(args)

    # Select novel candidates
    novel = concepts[
        (concepts.concept == "unresolved") &
        (concepts.path_rate > args.min_path_rate) &
        (concepts.n_active >= args.min_active) &
        (concepts.top_gene_frac < args.max_gene_frac)
    ].copy()
    print(f"\n{len(novel)} novel high-pathogenic features to characterize\n")

    all_chars = []
    for i, (_, row) in enumerate(novel.iterrows()):
        fidx = int(row["feature"])
        print(f"[{i+1}/{len(novel)}] Feature {fidx} (n={row['n_active']}, "
              f"path={row['path_rate']:.2f}, ratio={row['modality_ratio']:.3f}) ...",
              flush=True)
        char = characterize_feature(fidx, df, sae_acts, row)
        char["proposed_concept"] = propose_concept(char)
        all_chars.append(char)

        # Print summary
        genes = char["gene_enrichment"]
        subs = char["sub_enrichment"]
        gene_strs = [g["gene"] + "(OR=" + f"{g['odds_ratio']:.1f}" + ")" for g in genes[:3]]
        sub_strs = [s["sub"] + "(OR=" + f"{s['odds_ratio']:.1f}" + ")" for s in subs[:3]]
        print(f"  Top genes: {', '.join(gene_strs)}")
        print(f"  Top subs:  {', '.join(sub_strs)}")
        print(f"  Proposed:  {char['proposed_concept']}")
        print()

    # Cluster analysis
    print("\n=== CROSS-FEATURE CLUSTERING ===")
    clusters = cluster_features(all_chars)
    for ctype, groups in clusters.items():
        if groups:
            print(f"\n{ctype}:")
            for k, feats in sorted(groups.items(), key=lambda x: -len(x[1])):
                print(f"  {k}: features {feats}")

    # Build summary table
    rows = []
    for char in all_chars:
        genes = char.get("gene_enrichment", [])
        subs = char.get("sub_enrichment", [])
        rows.append({
            "feature": char["feature"],
            "n_active": char["n_active"],
            "path_rate": char["path_rate"],
            "n_genes": char["n_genes"],
            "modality_ratio": char["modality_ratio"],
            "top_gene_1": genes[0]["gene"] if genes else "",
            "top_gene_1_OR": genes[0]["odds_ratio"] if genes else 0,
            "top_gene_1_p": genes[0]["pval"] if genes else 1,
            "top_sub_1": subs[0]["sub"] if subs else "",
            "top_sub_1_OR": subs[0]["odds_ratio"] if subs else 0,
            "mean_charge_chg": char["aa_properties"].get("mean_charge_change", 0),
            "mean_hydro_chg": char["aa_properties"].get("mean_hydro_change", 0),
            "mean_vol_chg": char["aa_properties"].get("mean_vol_change", 0),
            "proposed_concept": char["proposed_concept"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "novel_features_summary.csv"), index=False)

    # Save full characterizations
    with open(os.path.join(args.out_dir, "novel_features_full.json"), "w") as f:
        json.dump(all_chars, f, indent=2, default=str)

    # Print final summary
    print("\n" + "="*80)
    print("NOVEL FEATURE SUMMARY")
    print("="*80)
    concept_counts = Counter(char["proposed_concept"] for char in all_chars)
    print(f"\nTotal novel features: {len(all_chars)}")
    print(f"Truly novel (no matching pattern): {concept_counts.get('truly-novel', 0)}")
    print(f"\nProposed concept distribution:")
    for concept, count in concept_counts.most_common():
        print(f"  {concept}: {count}")

    # Highlight the most interesting ones
    print(f"\n--- TOP 10 MOST INTERESTING (highest gene OR, spread across genes) ---")
    interesting = sorted(all_chars, key=lambda x: (
        x["gene_enrichment"][0]["odds_ratio"] if x["gene_enrichment"] else 0
    ), reverse=True)
    for char in interesting[:10]:
        genes = char["gene_enrichment"]
        print(f"  F{char['feature']}: n={char['n_active']}, path={char['path_rate']:.2f}, "
              f"ratio={char['modality_ratio']:.3f}, "
              f"genes=[{', '.join(g['gene'] for g in genes[:3])}], "
              f"concept={char['proposed_concept']}")

    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
