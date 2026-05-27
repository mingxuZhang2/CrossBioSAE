# CrossBioSAE: Cross-modal Sparse Autoencoder for Biological Foundation Models

## Overview

CrossBioSAE trains a shared Sparse Autoencoder (SAE) that bridges a **protein language model** (ESM-2 650M) and a **DNA language model** (Nucleotide Transformer v2 500M) through learned adapters. It tests whether biologically meaningful concepts live in a shared feature space across modalities — and uses that shared space for downstream biological tasks.

**Target venues**: Nature Communications / ICLR / NeurIPS

## Architecture

```
DNA sequence  → NT v2 500M  → mean-pool → Adapter_DNA (linear) ──┐
                                                                   ├→ Shared SAE → TopK sparse features → SAE decoder ─┬→ reconstruct DNA
Protein seq   → ESM-2 650M  → mean-pool → Adapter_Prot (linear) ─┘                                                    └→ reconstruct Protein
```

Three training losses:
1. **Reconstruction** (MSE): faithfully reconstruct both modalities
2. **Sparsity** (TopK k=128): enforce sparse feature activations
3. **Cross-modal consistency** (cosine): same gene should activate similar features from both modalities

## Key Results

### Cross-modal Alignment (16,604 human genes)

| Metric | Value |
|---|---|
| Shared features | 202 / 32,768 |
| Cross-modal similarity | 0.974 |
| Z-score vs permutation | **204.7** |
| P-value | 0.00 (underflow) |

ESM-2 and NT do converge on shared biological representations.

### Downstream Tasks

**Task 1 — Variant Effect Prediction** (5,000 ClinVar variants):

| Method | AUROC | AUPRC |
|---|---|---|
| ESM-2 LLR (protein only) | 0.851 | 0.990 |
| CrossBioSAE feature disruption | 0.556 | 0.951 |
| ESM-2 + NT ensemble (LR) | 0.854 | 0.990 |

Cross-modal ensemble marginally improves over protein-only. Evo (Science 2024) integration pending for stronger DNA signal.

**Task 2 — Gene Function Prediction**: 25,822 features labeled with GO terms via hypergeometric enrichment. Predictions for 16,620 genes.

**Task 3 — Cross-modal Anomaly Detection**: Genes where protein and DNA models disagree most. Top anomalous: ZFR2 (consistency=0), RBMX, ZNF776. ZNF zinc-finger proteins are overrepresented — potentially because zinc finger domains are encoded differently at DNA vs protein level.

## Data

- **Genes**: 16,620 matched human protein/CDS pairs from UniProt + NCBI RefSeq
- **Variants**: 50,000 ClinVar SNVs (26,841 pathogenic, 23,159 benign)
- **GO annotations**: 16,039 genes, 227,648 GO term assignments from UniProt
- **Models**: ESM-2 650M (protein, Facebook/Meta), NT v2 500M (DNA, InstaDeep)

## Project Structure

```
survey/                          # Stage 1: Literature survey (complete)
  sae_bio_ideas.md               # 5 research ideas
  sae_bio_gaps.md                # 12 identified gaps
  landscape.md                   # LLM-for-bio research landscape
  ...

implementation/                  # Stage 2: Code (active)
  src/
    model.py                     # CrossBioSAE architecture
    data.py                      # Data pipeline + activation extraction
    trainer.py                   # Training loop
    evaluation.py                # All evaluation tasks
  scripts/
    download_real_data.py        # Fetch gene pairs from UniProt + NCBI
    download_clinvar.py          # Fetch ClinVar variants
    download_go_annotations.py   # Fetch GO terms
    extract_activations.py       # Extract ESM-2 / NT activations
    train_sae.py                 # Train CrossBioSAE
    run_pilot_eval.py            # Pilot feasibility check
    run_downstream.py            # Downstream tasks
    run_baselines.py             # Baseline comparisons
    run_baselines_v3.py          # Cross-modal ensemble (ESM-2 + NT)
    run_baselines_evo.py         # ESM-2 + Evo ensemble (pending)
    prepare_variant_data.py      # Variant scoring pipeline
    slurm_*.sh                   # HPC job scripts
  configs/
    pilot.yaml                   # 1k genes, 16k features
    full.yaml                    # 16k genes, 32k features
```

## Training

```bash
# 1. Download data (requires internet)
python scripts/download_real_data.py --n_genes 20000
python scripts/download_clinvar.py
python scripts/download_go_annotations.py

# 2. Extract activations (requires GPU)
python scripts/extract_activations.py --config configs/full.yaml --modality both --device cuda

# 3. Train SAE
python scripts/train_sae.py --config configs/full.yaml --device cuda

# 4. Evaluate
python scripts/run_pilot_eval.py --config configs/full.yaml --checkpoint checkpoints/full/checkpoint_best.pt
python scripts/run_downstream.py --config configs/full.yaml --checkpoint checkpoints/full/checkpoint_best.pt --task all
```

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| crossmodal_weight | 1.0 | Critical — 0.1 leads to no alignment |
| crossmodal_warmup_steps | 200 | Ramp up alignment loss gradually |
| expansion_factor | 16 | 32k features for 2048-dim shared space |
| topk_k | 128 | Active features per sample |
| learning_rate | 1e-4 | With cosine decay |
| max_epochs | 200 | ~6400 steps for 16k genes |

## Pending Work

- [ ] **Evo integration** (Science 2024): Replace NT with Evo for character-level DNA LLR. Expected to significantly improve variant prediction ensemble.
- [ ] **Per-residue SAE**: Current SAE operates on gene-level mean-pooled activations. Per-residue variant would enable fine-grained variant scoring.
- [ ] **Real CDS for variant scoring**: Use actual genomic CDS with single-codon mutations instead of back-translation.
- [ ] **Paper writing**: Stage 3 — draft for Nature Communications / ICLR.

## Requirements

- PyTorch >= 2.1
- fair-esm >= 2.0 (ESM-2)
- transformers >= 4.40 (NT v2)
- h5py, biopython, scikit-learn, scipy, pandas

## Citation

If you use this work, please cite:
```
@article{crossbiosae2026,
  title={CrossBioSAE: Cross-modal Sparse Autoencoder for Protein-DNA Feature Alignment},
  author={Zhang, Mingxu},
  year={2026}
}
```
