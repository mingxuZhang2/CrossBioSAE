"""
Build clean BRCA1 SGE variant table (Findlay 2018) for the cross-modal
variant-effect fusion benchmark.

Outputs brca1_variants.csv with:
  id, chrom, pos_hg19, ref, alt, consequence, is_missense,
  aa_pos, aa_ref, aa_alt, label (1=LOF, 0=FUNC; INT dropped),
  function_score, CADD, phyloP, clinvar_simple

Label: func.class in {FUNC, LOF, INT}. Binary task = LOF vs FUNC (drop INT).
Coords are hg19/GRCh37 (match the chr17_GRCh37.fna.gz reference).
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/variant/brca1/findlay_brca1.xlsx")
    ap.add_argument("--out", default="data/variant/brca1/brca1_variants.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, header=2)
    cols = {
        "chromosome": "chrom", "position (hg19)": "pos_hg19",
        "reference": "ref", "alt": "alt", "consequence": "consequence",
        "aa_pos": "aa_pos", "aa_ref": "aa_ref", "aa_alt": "aa_alt",
        "function.score.mean": "function_score", "func.class": "func_class",
        "CADD.score": "CADD", "phyloP (mammalian)": "phyloP",
        "clinvar_simple": "clinvar_simple",
    }
    d = df[list(cols)].rename(columns=cols).copy()

    d["is_missense"] = d["consequence"] == "Missense"
    # binary label: LOF=1, FUNC=0, drop INT
    d = d[d["func_class"].isin(["LOF", "FUNC"])].copy()
    d["label"] = (d["func_class"] == "LOF").astype(int)
    d["id"] = [f"brca1_{i}" for i in range(len(d))]

    # variant-type grouping for stratified reporting
    noncoding = {"Intronic", "Splice region", "Canonical splice", "5' UTR"}
    coding = {"Missense", "Synonymous", "Nonsense"}
    d["vtype"] = d["consequence"].map(
        lambda c: "noncoding" if c in noncoding else ("coding" if c in coding else "other"))

    print(f"Variants (LOF/FUNC): {len(d)}  (LOF={d['label'].sum()}, FUNC={(d['label']==0).sum()})")
    print("\nby consequence:")
    print(d.groupby(["consequence"]).agg(n=("label", "size"), lof=("label", "sum")).to_string())
    print("\nby vtype:")
    print(d.groupby("vtype").agg(n=("label", "size"), lof=("label", "sum")).to_string())
    print(f"\nmissense: {d['is_missense'].sum()} (ESM-scorable)")
    print(f"noncoding/splice: {(d['vtype']=='noncoding').sum()} (protein-blind, DNA-only)")

    d.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(d)} rows)")


if __name__ == "__main__":
    main()
