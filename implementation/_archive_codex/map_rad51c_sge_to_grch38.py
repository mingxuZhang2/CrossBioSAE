#!/usr/bin/env python3
"""Map RAD51C MaveDB cDNA SNVs to GRCh38 coordinates for Evo2/ESM checks."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import requests


REST = "https://rest.ensembl.org"
TRANSCRIPT_ID = "ENST00000337432"
OUT_DIR = Path("results/interpretability_applications")
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "variant" / "rad51c"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument("--scores", type=Path, default=data_dir / "rad51c_mavedb_scores.csv")
    p.add_argument("--genome", type=Path, default=root / "data" / "variant" / "GRCh38.fa.gz")
    p.add_argument("--transcript-id", default=TRANSCRIPT_ID)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


def fetch_lookup(cache: Path, transcript_id: str, timeout: int, force_refresh: bool) -> dict:
    if cache.exists() and not force_refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{REST}/lookup/id/{transcript_id}?expand=1;content-type=application/json"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def load_chrom(genome: Path, chrom: str) -> str:
    op = gzip.open if str(genome).endswith(".gz") else open
    cur = None
    buf: list[str] = []
    with op(genome, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if cur == chrom:
                    break
                cur = line[1:].split()[0].removeprefix("chr")
                buf = []
            elif cur == chrom:
                buf.append(line.strip())
    if cur != chrom and not buf:
        raise RuntimeError(f"Chromosome {chrom} not found in {genome}")
    return "".join(buf).upper()


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1].upper()


def transcript_maps(lookup: dict) -> tuple[dict[int, int], dict[int, int], int, int, str, int]:
    strand = int(lookup["strand"])
    exons = sorted(lookup["Exon"], key=lambda x: int(x["start"]), reverse=strand < 0)
    cdna_to_genome: dict[int, int] = {}
    genome_to_cdna: dict[int, int] = {}
    cdna_pos = 1
    for exon in exons:
        start = int(exon["start"])
        end = int(exon["end"])
        genomic_positions = range(start, end + 1) if strand > 0 else range(end, start - 1, -1)
        for gpos in genomic_positions:
            cdna_to_genome[cdna_pos] = gpos
            genome_to_cdna[gpos] = cdna_pos
            cdna_pos += 1
    translation = lookup["Translation"]
    cds_start_cdna = genome_to_cdna[int(translation["start"])]
    cds_end_cdna = genome_to_cdna[int(translation["end"])]
    chrom = str(lookup["seq_region_name"]).removeprefix("chr")
    return cdna_to_genome, genome_to_cdna, cds_start_cdna, cds_end_cdna, chrom, strand


def build_cdna(chrom_seq: str, cdna_to_genome: dict[int, int], strand: int) -> str:
    bases = []
    for cdna_pos in range(1, max(cdna_to_genome) + 1):
        base = chrom_seq[cdna_to_genome[cdna_pos] - 1]
        bases.append(base if strand > 0 else base.translate(COMP).upper())
    return "".join(bases)


def translate(seq: str) -> str:
    aas = []
    for i in range(0, len(seq) - 2, 3):
        aas.append(CODON_TABLE.get(seq[i : i + 3].upper(), "X"))
    return "".join(aas)


def wrap_fasta(header: str, seq: str) -> str:
    return f">{header}\n" + "\n".join(textwrap.wrap(seq, 80)) + "\n"


def parse_hgvs_snv(hgvs: str) -> tuple[str, int, str, str] | None:
    text = str(hgvs).strip()
    m = re.search(r":c\.(-?\d+)([ACGT])>([ACGT])$", text)
    if m:
        return ("coding", int(m.group(1)), m.group(2), m.group(3))
    m = re.search(r":c\.\*(\d+)([ACGT])>([ACGT])$", text)
    if m:
        return ("star", int(m.group(1)), m.group(2), m.group(3))
    return None


def hgvs_to_cdna(kind: str, pos: int, cds_start_cdna: int, cds_end_cdna: int) -> int:
    if kind == "star":
        return cds_end_cdna + pos
    if pos >= 1:
        return cds_start_cdna + pos - 1
    return cds_start_cdna + pos


def classify_variant(hgvs_pos: int | None, ref: str, alt: str, cds_seq: str) -> dict[str, object]:
    out: dict[str, object] = {
        "consequence": "noncoding_or_unmapped",
        "vep_consequence": "noncoding_or_unmapped",
        "aa_pos": pd.NA,
        "aa_ref": "",
        "aa_alt": "",
        "is_missense": False,
        "is_synonymous": False,
        "is_nonsense": False,
        "HGVSp": "-",
    }
    if hgvs_pos is None or hgvs_pos < 1:
        out["consequence"] = "UTR"
        out["vep_consequence"] = "5_prime_UTR_variant"
        return out
    cds_index = hgvs_pos - 1
    if cds_index < 0 or cds_index >= len(cds_seq):
        return out
    codon_start = (cds_index // 3) * 3
    codon = cds_seq[codon_start : codon_start + 3]
    if len(codon) != 3:
        return out
    offset = cds_index % 3
    aa_pos = codon_start // 3 + 1
    aa_ref = CODON_TABLE.get(codon, "X")
    alt_codon = codon[:offset] + alt + codon[offset + 1 :]
    aa_alt = CODON_TABLE.get(alt_codon, "X")
    out.update({"aa_pos": aa_pos, "aa_ref": aa_ref, "aa_alt": aa_alt, "HGVSp": f"p.{aa_ref}{aa_pos}{aa_alt}"})
    if aa_ref == aa_alt:
        out.update({"consequence": "Synonymous", "vep_consequence": "synonymous_variant", "is_synonymous": True})
    elif aa_alt == "*":
        out.update({"consequence": "Nonsense", "vep_consequence": "stop_gained", "is_nonsense": True})
    elif aa_ref == "*":
        out.update({"consequence": "Stop codon", "vep_consequence": "stop_lost"})
    else:
        out.update({"consequence": "Missense", "vep_consequence": "missense_variant", "is_missense": True})
    return out


def table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return show.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    data_dir = repo / "data" / "variant" / "rad51c"
    out_dir = repo / OUT_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    cache = data_dir / f"rad51c_ensembl_{args.transcript_id}_GRCh38.json"
    lookup = fetch_lookup(cache, args.transcript_id, args.timeout, args.force_refresh)
    cdna_to_genome, _, cds_start_cdna, cds_end_cdna, chrom, strand = transcript_maps(lookup)
    chrom_seq = load_chrom(args.genome, chrom)
    cdna_seq = build_cdna(chrom_seq, cdna_to_genome, strand)
    cds_seq = cdna_seq[cds_start_cdna - 1 : cds_end_cdna]
    protein_with_stop = translate(cds_seq)
    protein = protein_with_stop[:-1] if protein_with_stop.endswith("*") else protein_with_stop

    rows = []
    all_rows = []
    for _, row in scores.iterrows():
        parsed = parse_hgvs_snv(row.get("hgvs_nt", ""))
        status = "not_simple_cdna_snv"
        mapped = {
            "hgvs_kind": "",
            "hgvs_c_pos": pd.NA,
            "transcript_cdna_pos": pd.NA,
            "chrom": chrom,
            "pos_hg38": pd.NA,
            "genomic_ref": "",
            "genomic_alt": "",
            "ref_matches_grch38": False,
        }
        extra = classify_variant(None, "", "", cds_seq)
        if parsed is not None:
            kind, hgvs_pos, ref, alt = parsed
            cdna_pos = hgvs_to_cdna(kind, hgvs_pos, cds_start_cdna, cds_end_cdna)
            mapped.update({"hgvs_kind": kind, "hgvs_c_pos": hgvs_pos, "transcript_cdna_pos": cdna_pos})
            if cdna_pos in cdna_to_genome:
                genomic_ref = ref if strand > 0 else revcomp(ref)
                genomic_alt = alt if strand > 0 else revcomp(alt)
                gpos = cdna_to_genome[cdna_pos]
                ref_match = chrom_seq[gpos - 1] == genomic_ref
                mapped.update(
                    {
                        "pos_hg38": gpos,
                        "genomic_ref": genomic_ref,
                        "genomic_alt": genomic_alt,
                        "ref_matches_grch38": ref_match,
                    }
                )
                extra = classify_variant(hgvs_pos if kind == "coding" else None, ref, alt, cds_seq)
                status = "mapped_ref_match" if ref_match else "mapped_ref_mismatch"
            else:
                status = "cdna_coordinate_out_of_transcript"

        all_entry = {**row.to_dict(), **mapped, **extra, "mapping_status": status}
        all_rows.append(all_entry)
        if status != "mapped_ref_match":
            continue
        if not bool(row.get("is_binary_sge_label", False)):
            continue
        rows.append(
            {
                "id": row["id"],
                "chrom": chrom,
                "pos_hg38": int(mapped["pos_hg38"]),
                "pos": int(mapped["pos_hg38"]),
                "ref": mapped["genomic_ref"],
                "alt": mapped["genomic_alt"],
                "label": row["label"],
                "is_binary_sge_label": row["is_binary_sge_label"],
                "functional_classification": row.get("functional_classification", ""),
                "function_score": row.get("score", pd.NA),
                "score": row.get("score", pd.NA),
                "consequence": extra["consequence"],
                "vep_consequence": extra["vep_consequence"],
                "is_missense": bool(extra["is_missense"]),
                "is_synonymous": bool(extra["is_synonymous"]),
                "is_nonsense": bool(extra["is_nonsense"]),
                "aa_pos": extra["aa_pos"],
                "aa_ref": extra["aa_ref"],
                "aa_alt": extra["aa_alt"],
                "c_nom": row.get("hgvs_nt", ""),
                "g_nom": f"NC_000017.11:g.{int(mapped['pos_hg38'])}{mapped['genomic_ref']}>{mapped['genomic_alt']}",
                "HGVSc": row.get("hgvs_nt", ""),
                "HGVSp": extra["HGVSp"],
                "chrom_pos_ref_alt": f"{chrom}_{int(mapped['pos_hg38'])}_{mapped['genomic_ref']}_{mapped['genomic_alt']}",
                "domains": "RAD51C_CDS" if extra["consequence"] != "UTR" else "RAD51C_5UTR",
                "source": row.get("source", "Stone_2024_Cell_RAD51C_SGE_MaveDB"),
                "mavedb_urn": row.get("mavedb_urn", "urn:mavedb:00000673-0-1"),
                "hgvs_c_pos": mapped["hgvs_c_pos"],
                "transcript_cdna_pos": mapped["transcript_cdna_pos"],
            }
        )

    mapped_all = pd.DataFrame(all_rows)
    out = pd.DataFrame(rows)
    out_path = data_dir / "rad51c_grch38_variants.csv"
    all_path = data_dir / "rad51c_grch38_mapping_all.csv"
    protein_path = data_dir / "rad51c_O43502.fasta"
    out.to_csv(out_path, index=False)
    mapped_all.to_csv(all_path, index=False)
    protein_path.write_text(wrap_fasta(f"RAD51C|{args.transcript_id}|GRCh38|O43502", protein), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "transcript_id": args.transcript_id,
                "assembly": lookup.get("assembly_name", "GRCh38"),
                "chrom": chrom,
                "strand": strand,
                "transcript_length": len(cdna_seq),
                "cds_start_cdna": cds_start_cdna,
                "cds_end_cdna": cds_end_cdna,
                "cds_length": len(cds_seq),
                "protein_length_no_stop": len(protein),
                "input_rows": len(scores),
                "simple_cdna_snv_rows": int(mapped_all["mapping_status"].ne("not_simple_cdna_snv").sum()),
                "mapped_ref_match_rows": int(mapped_all["mapping_status"].eq("mapped_ref_match").sum()),
                "output_binary_mapped_rows": len(out),
                "output_missense_rows": int(out["is_missense"].sum()) if not out.empty else 0,
                "output_synonymous_rows": int(out["is_synonymous"].sum()) if not out.empty else 0,
                "output_nonsense_rows": int(out["is_nonsense"].sum()) if not out.empty else 0,
                "output_lof_rows": int(pd.to_numeric(out["label"], errors="coerce").eq(1).sum()) if not out.empty else 0,
                "output_functional_rows": int(pd.to_numeric(out["label"], errors="coerce").eq(0).sum()) if not out.empty else 0,
            }
        ]
    )
    status_counts = mapped_all["mapping_status"].value_counts(dropna=False).rename_axis("mapping_status").reset_index(name="n")
    consequence_counts = (
        out.groupby("consequence", dropna=False)
        .agg(n=("id", "size"), n_lof=("label", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
        if not out.empty
        else pd.DataFrame()
    )
    summary.to_csv(out_dir / "rad51c_grch38_mapping_summary.csv", index=False)
    status_counts.to_csv(out_dir / "rad51c_grch38_mapping_status_counts.csv", index=False)
    consequence_counts.to_csv(out_dir / "rad51c_grch38_mapping_consequence_counts.csv", index=False)

    report = [
        "# RAD51C GRCh38 Mapping",
        "",
        "## Purpose",
        "",
        "Convert the RAD51C Cell/MaveDB SGE score set from transcript-level cDNA SNVs into a GRCh38 variant table that can run through the same Evo2 LLR, ESM, and checkpoint-discordance workflow used for BRCA2 and BAP1.",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Mapping Status",
        "",
        table(status_counts),
        "",
        "## Consequence Counts",
        "",
        table(consequence_counts),
        "",
        "## Outputs",
        "",
        "- `data/variant/rad51c/rad51c_grch38_variants.csv`",
        "- `data/variant/rad51c/rad51c_grch38_mapping_all.csv`",
        "- `data/variant/rad51c/rad51c_O43502.fasta`",
        "- `data/variant/rad51c/rad51c_ensembl_ENST00000337432_GRCh38.json`",
        "",
        "## Claim Boundary",
        "",
        "- This makes RAD51C executable as a third-gene checkpoint, but it is not yet a CrossBioSAE validation result.",
        "- The output currently covers direct cDNA SNVs with GRCh38 reference matches; codon-level delins and intronic splice edits remain excluded from this Evo2/ESM checkpoint table.",
        "- A publishable RAD51C claim still requires completed Evo2/ESM metrics and, if positive, native-SAE intervention or an external clinical/functional endpoint.",
        "",
    ]
    (out_dir / "rad51c_grch38_mapping.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out_path}")
    print(f"wrote {protein_path}")


if __name__ == "__main__":
    main()
