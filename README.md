# CrossBioSAE: Cross-modal Sparse Autoencoder for Biological Foundation Models

## Overview

CrossBioSAE uses **Sparse Autoencoders (SAEs)** to test whether independently trained **protein** and **DNA** foundation models learn the same underlying biology. We train independent SAEs on ESM-2 (protein) and Nucleotide Transformer (DNA) activations, then measure post-hoc alignment — no cross-modal training signal needed.

**Core finding**: Independently trained protein and DNA SAE features *are* alignable (CCA retrieval R@1 = 10.3%, 340× above chance), providing non-circular evidence that bio-FMs converge on shared biological representations.

**Target venues**: Nature Communications / ICLR / NeurIPS

## Two Experimental Approaches

### Approach 1: Independent SAEs + Post-hoc Alignment (addresses circular-evidence concern)

```
Protein sequences → ESM-2 650M → activations → Independent Protein-SAE → protein features ─┐
                                                                                             ├→ CCA / Procrustes / Linear Probe
DNA CDS sequences → NT v2 500M → activations → Independent DNA-SAE → DNA features ──────────┘
```

No shared parameters, no cross-modal loss. Alignment measured purely post-hoc.

### Approach 2: Joint CrossBioSAE (shared feature space)

```
Protein activations → Adapter → Shared SAE (TopK=128, 32k features) → reconstruct protein
DNA activations     → Adapter ↗                                      ↘ reconstruct DNA
```

Three losses: reconstruction + sparsity + cross-modal cosine consistency.

## Key Results

### Independent Alignment (the core scientific result)

| Method | Retrieval R@1 | R@5 | MRR | Notes |
|---|---|---|---|---|
| Random baseline | 0.03% | 0.15% | — | chance level |
| Independent SAEs + CCA | **10.3%** | **25.9%** | **0.188** | no cross-modal training |
| Independent SAEs + Procrustes | 8.4% | 18.6% | 0.141 | orthogonal rotation only |
| Joint CrossBioSAE + CCA | 14.3% | 32.4% | 0.239 | with cross-modal loss |

**Interpretation**: ESM-2 and NT independently learn partially aligned representations. Joint training improves alignment but independent features already carry substantial cross-modal structure.

### Cross-modal Alignment Metrics

| Metric | Independent SAEs | Joint CrossBioSAE |
|---|---|---|
| CCA mean correlation | 0.581 | 0.627 |
| Procrustes cosine sim | 0.083 | 0.101 |
| Linear probe R² | 0.073–0.079 | 0.088–0.097 |

### Downstream Tasks

**Task 1 — Variant Effect Prediction** (5,000 ClinVar variants):

| Method | AUROC |
|---|---|
| ESM-2 LLR (per-residue, protein only) | 0.851 |
| CrossBioSAE feature disruption (gene-level) | 0.556 |
| ESM-2 + NT ensemble (logistic regression) | 0.854 |

Gene-level SAE features are too coarse for single-residue variant scoring. Per-residue SAE is planned as future work.

**Task 2 — Gene Function Prediction**: Features labeled with GO terms via hypergeometric enrichment + Benjamini-Hochberg FDR correction. Predictions for 16,620 genes.

**Task 3 — Cross-modal Anomaly Detection**: Genes where protein and DNA models disagree. Top anomalous: ZFR2, RBMX, ZNF776. ZNF zinc-finger proteins systematically overrepresented (Fisher's exact test per gene family).

## Data

- **16,620 matched human genes** (protein from UniProt + CDS from NCBI RefSeq, translation-verified)
- **50,000 ClinVar variants** (26,841 pathogenic, 23,159 benign)
- **227,648 GO term assignments** for 16,039 genes
- **Models**: ESM-2 650M (protein), Nucleotide Transformer v2 500M (DNA)

## Project Structure

```
survey/                           # Stage 1: Literature survey
implementation/                   # Stage 2: Code
  src/
    model.py                      # CrossBioSAE + StandardSAE architectures
    data.py                       # Data pipeline + activation extraction
    trainer.py                    # CrossBioSAETrainer + StandardSAETrainer
    evaluation.py                 # Pilot eval, variant/function/anomaly tasks
    alignment.py                  # Post-hoc alignment: CCA, Procrustes, LinearProbe, Retrieval
  scripts/
    train_sae.py                  # Train joint CrossBioSAE
    train_independent_sae.py      # Train independent single-modality SAEs
    run_alignment_benchmark.py    # Compare alignment methods
    download_real_data.py         # Fetch verified gene pairs (UniProt + NCBI)
    run_baselines*.py             # Variant prediction baselines
    slurm_*.sh                    # HPC job scripts
  configs/
    pilot.yaml                    # 1k genes, 16k features
    full.yaml                     # 16k genes, 32k features
    independent.yaml              # Independent SAE + alignment benchmark
```

## Quick Start

```bash
# Independent SAE approach (recommended for scientific claims)
python scripts/train_independent_sae.py --config configs/independent.yaml --modality protein --device cuda
python scripts/train_independent_sae.py --config configs/independent.yaml --modality dna --device cuda
python scripts/run_alignment_benchmark.py --config configs/independent.yaml \
    --protein_sae_ckpt checkpoints/full/protein_sae/checkpoint_best.pt \
    --dna_sae_ckpt checkpoints/full/dna_sae/checkpoint_best.pt --device cuda

# Joint CrossBioSAE approach (for shared feature space)
python scripts/train_sae.py --config configs/full.yaml --device cuda
```

## GPT Pro Review Response

An external review identified 10 concerns. Status after addressing:

| # | Concern | Status |
|---|---|---|
| 1 | Variant pipeline issues | ⚠️ Documented as limitation |
| 2 | CDS translation consistency | ✅ translate(CDS)==protein verified |
| 3 | Silent synthetic fallback | ✅ Removed (fail-fast) |
| 4 | Alignment eval needs retrieval | ✅ Independent SAE + CCA/Procrustes/Retrieval |
| 5 | GO labeling needs FDR | ✅ Benjamini-Hochberg correction |
| 6 | Shared feature definition | ✅ Matched-gene co-activation (corr>0.1) |
| 7 | Per-residue SAE | Planned as future work |
| 8 | Ablation baselines | ✅ Shuffled pairs + multiple methods |
| 9 | Engineering fixes | ✅ Core issues fixed |
| 10 | Anomaly analysis | ✅ Gene-family enrichment (Fisher's exact) |

## Pending Work

- [ ] **Evo integration** (Science 2024): Character-level DNA LLR for variant ensemble
- [ ] **Per-residue CrossBioSAE**: Enable fine-grained variant scoring
- [ ] **Biological validation of anomalies**: Deep case study on ZNF discordance
- [ ] **Paper writing**: Target Nature Communications / ICLR

## Requirements

PyTorch >= 2.1, fair-esm >= 2.0, transformers >= 4.40, h5py, biopython, scikit-learn, scipy, pandas, statsmodels
