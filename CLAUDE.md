# Project: LLM for Bioinformatics Research — CrossBioSAE

## Overview
Cross-modal Sparse Autoencoder aligning protein-LM (ESM-2/ESM-C) and DNA-LM (Evo-2) features on matched coding genes.
Research goal: test whether biologically meaningful concepts live in a shared feature space across modalities.
Focus: computational-only research, no wet lab access.

## Project Status
- Stage 1 (Survey): COMPLETE — general LLM-for-bio + deep SAE+bio sub-survey
- Stage 2 (Implementation): IN PROGRESS — Idea 3 (CrossBioSAE) selected by Dr. Zhang
  - Code: COMPLETE — model, data pipeline, training, evaluation, downstream tasks
  - Tests: ALL PASSING — verified on CPU with synthetic data
  - Experiments: NOT STARTED — awaiting HPC submission
- Stage 3 (Writing): Not started

## Selected Idea: CrossBioSAE
Train a single SAE that ingests activations from both a protein LM and a DNA LM on paired protein/coding-DNA inputs. Test whether biologically meaningful concepts (binding sites, secondary structure, signal peptides) live in a shared feature space across modalities.

Target venues: Nature Communications, ICLR, NeurIPS

## Directory Structure
```
survey/                              — Stage 1 survey (DO NOT MODIFY)
  sae_bio_ideas.md                   — 5 research ideas (CrossBioSAE = Idea 3)
  sae_bio_landscape.md               — SAE x bio-LLM landscape
  sae_bio_gaps.md                    — 12 gaps at SAE+bio intersection
  ...                                — Other survey files

implementation/                      — Stage 2 code
  src/
    __init__.py
    model.py                         — CrossBioSAE + StandardSAE architectures
    data.py                          — Gene pair download, activation extraction, dataset classes
    trainer.py                       — CrossBioSAETrainer + StandardSAETrainer
    evaluation.py                    — Pilot eval, variant prediction, function prediction, anomaly detection
    alignment.py                     — Post-hoc alignment: CCA, Procrustes, LinearProbe, Retrieval
  scripts/
    prepare_data.py                  — Step 1: Download/generate gene pairs
    extract_activations.py           — Step 2: Extract activations from ESM-2 and Evo-2
    train_sae.py                     — Step 3: Train CrossBioSAE (joint)
    train_independent_sae.py         — Step 3b: Train independent single-modality SAEs
    run_alignment_benchmark.py       — Step 3c: Post-hoc alignment benchmark (independent vs joint)
    run_pilot_eval.py                — Step 4: Pilot feasibility check
    run_downstream.py                — Step 5: Downstream tasks
    test_pipeline.py                 — Unit tests (CPU, no models needed, ALL PASSING)
    slurm_extract_protein.sh         — SLURM: protein activation extraction
    slurm_extract_dna.sh             — SLURM: DNA activation extraction
    slurm_train.sh                   — SLURM: SAE training
    slurm_pilot_full.sh              — SLURM: complete pilot pipeline
  configs/
    pilot.yaml                       — Pilot: 1k genes, ESM-2 650M, 8x expansion
    full.yaml                        — Full: 20k genes, ESM-C 600M, 32x expansion
    independent.yaml                 — Independent SAE training + alignment benchmark config
  environment.yml                    — Conda environment spec
  requirements.txt                   — Pip requirements
  README.md                          — Implementation documentation
```

## Architecture
```
DNA act (dim_dna) -> Adapter_DNA (linear) ────────┐
                                                    ├─> Shared SAE encoder -> TopK sparse -> SAE decoder ─┬─> Inv_Adapter_DNA -> recon DNA
Protein act (dim_prot) -> Adapter_Prot (linear) ──┘                                                       └─> Inv_Adapter_Prot -> recon Protein
```

Three losses: (1) reconstruction MSE, (2) TopK sparsity, (3) cross-modal cosine consistency

## Experiment Pipeline
1. `prepare_data.py` — download matched protein/CDS gene pairs from UniProt
2. `extract_activations.py` — extract mid-layer activations from ESM-2 + Evo-2 (GPU)
3. `train_sae.py` — train CrossBioSAE with 3 losses (GPU)
4. `run_pilot_eval.py` — feasibility check: shared features, z-score vs permutation
5. `run_downstream.py` — anomaly detection, function prediction, variant prediction

## Independent SAE + Post-hoc Alignment (Reviewer Response)
Addresses reviewer concern that joint cross-modal loss creates circular evidence.

**Pipeline:**
1. `train_independent_sae.py --modality protein` — train protein-only SAE (no cross-modal signal)
2. `train_independent_sae.py --modality dna` — train DNA-only SAE (no cross-modal signal)
3. `run_alignment_benchmark.py` — post-hoc alignment evaluation (CCA, Procrustes, linear probe, retrieval)

**Key classes:**
- `StandardSAE` / `StandardSAEConfig` (model.py) — single-modality SAE, no adapters
- `StandardSAELoss` (model.py) — reconstruction + sparsity only
- `StandardSAETrainer` (trainer.py) — simplified trainer for single modality
- `CCAAlignment`, `ProcrustesAlignment`, `LinearProbeAlignment`, `CrossModalRetrieval`, `AlignmentBenchmark` (alignment.py)

**Dimension mismatch handling** (protein 1280*16=20480 vs DNA 1024*16=16384):
- CCA: naturally handles different dims
- Procrustes: PCA to min(dim_a, dim_b) then orthogonal rotation
- LinearProbe: ridge regression handles any dim mapping
- Retrieval: PCA to common dim before distance computation

## Pilot Success Criteria
- Cross-modal alignment z-score > 2.0 (vs permutation baseline)
- At least 10 shared features (active in both modalities)
- Less than 90% dead features

## HPC Environment
- Cluster: HKUST-GZ HPC2 (A800 GPUs in i64m1tga800u partition)
- Conda: crossbiosae (or base with requirements.txt)
- SLURM scripts in `implementation/scripts/slurm_*.sh`

## Key Dependencies
- PyTorch >= 2.1, fair-esm >= 2.0 (ESM-2), transformers >= 4.40 (Evo-2)
- sae-lens >= 4.0 (reference), h5py, biopython, wandb
- Evo-2: install from source (github.com/arcinstitute/evo)

## Git Workflow
- Branch `stage2-crossbiosae` for all implementation work
- Merge to main after experiments show results
