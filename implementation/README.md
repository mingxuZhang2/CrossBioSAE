# CrossBioSAE: Cross-modal Sparse Autoencoder for Protein-DNA Feature Alignment

Train a single SAE that bridges a protein language model (ESM-2/ESM-C) and a DNA language model (Evo-2) via learned adapters, creating a shared interpretable feature space.

## Architecture

```
DNA activation (dim_dna)     -> Adapter_DNA (linear) ──────────┐
                                                                ├─> Shared SAE encoder -> sparse features -> SAE decoder ─┬─> Adapter_DNA_inv -> reconstruct DNA
Protein activation (dim_prot) -> Adapter_Prot (linear) ────────┘                                                          └─> Adapter_Prot_inv -> reconstruct Protein
```

Three training losses:
1. **Reconstruction**: MSE on reconstructed activations for both modalities
2. **Sparsity**: TopK (default), L1, or JumpReLU
3. **Cross-modal consistency**: Cosine similarity of feature patterns for matched gene pairs

## Directory Structure

```
implementation/
  src/
    __init__.py
    model.py         # CrossBioSAE architecture (adapters + shared SAE + losses)
    data.py          # Gene pair download, activation extraction, dataset classes
    trainer.py       # Training loop with wandb logging
    evaluation.py    # Downstream tasks (variant prediction, function prediction, anomaly detection)
  scripts/
    prepare_data.py          # Step 1: Download/generate gene pairs
    extract_activations.py   # Step 2: Extract activations from ESM-2 and Evo-2
    train_sae.py             # Step 3: Train CrossBioSAE
    run_pilot_eval.py        # Step 4: Pilot feasibility check
    run_downstream.py        # Step 5: Downstream tasks
    test_pipeline.py         # Unit tests (CPU, no models needed)
    slurm_extract_protein.sh # SLURM: protein activation extraction
    slurm_extract_dna.sh     # SLURM: DNA activation extraction
    slurm_train.sh           # SLURM: SAE training
    slurm_pilot_full.sh      # SLURM: complete pilot pipeline
  configs/
    pilot.yaml               # Pilot config (1k genes, ESM-2 650M, 8x expansion)
    full.yaml                # Full config (20k genes, ESM-C 600M, 32x expansion)
  environment.yml            # Conda environment
  requirements.txt           # Pip requirements
```

## Quick Start

### 1. Test pipeline locally (no GPU needed)

```bash
cd implementation
python scripts/test_pipeline.py
```

### 2. Set up environment on HPC

```bash
conda env create -f environment.yml
conda activate crossbiosae
pip install fair-esm
# Evo-2: pip install git+https://github.com/arcinstitute/evo.git
```

### 3. Run pilot experiment

Option A: All-in-one SLURM job:
```bash
sbatch scripts/slurm_pilot_full.sh
```

Option B: Step by step:
```bash
# Step 1: Prepare data (login node, no GPU)
python scripts/prepare_data.py --config configs/pilot.yaml --synthetic

# Step 2: Extract activations (GPU node)
sbatch scripts/slurm_extract_protein.sh pilot
sbatch scripts/slurm_extract_dna.sh pilot

# Step 3: Train SAE (GPU node, after Step 2 completes)
sbatch scripts/slurm_train.sh pilot

# Step 4: Evaluate pilot (GPU node, after Step 3 completes)
python scripts/run_pilot_eval.py --config configs/pilot.yaml \
    --checkpoint checkpoints/pilot/checkpoint_best.pt

# Step 5: Downstream tasks
python scripts/run_downstream.py --config configs/pilot.yaml \
    --checkpoint checkpoints/pilot/checkpoint_best.pt --task anomaly
```

### 4. Scale to full experiment (after pilot passes)

```bash
# Prepare real data from UniProt
python scripts/prepare_data.py --config configs/full.yaml

# Submit extraction and training jobs
sbatch scripts/slurm_extract_protein.sh full
sbatch scripts/slurm_extract_dna.sh full
# Wait for both to complete, then:
sbatch scripts/slurm_train.sh full
```

## Pilot Success Criteria

The pilot passes if:
- Cross-modal feature alignment z-score > 2.0 (vs permutation baseline)
- At least 10 shared features (active in both modalities)
- Less than 90% dead features

## Model Configuration

| Parameter | Pilot | Full |
|---|---|---|
| Protein model | ESM-2 650M | ESM-C 600M |
| DNA model | Evo-2 7B | Evo-2 7B |
| Genes | 1,000 | ~20,000 |
| Shared dim | 2,048 | 2,048 |
| Expansion | 8x (16k features) | 32x (65k features) |
| TopK | 64 | 64 |
| Cross-modal weight | 0.1 | 0.1 |

## Downstream Tasks

1. **Variant prediction**: ClinVar temporal holdout (2025-2026), AUROC/AUPRC vs AlphaMissense
2. **Function prediction**: GO term prediction via feature-label enrichment
3. **Anomaly detection**: Rank genes by cross-modal inconsistency, validate against alt-splicing databases
