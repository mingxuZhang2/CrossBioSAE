# CrossBioSAE — Project Overview

## 1. Research Question

Biological foundation models (bio-FMs) have been trained independently on different modalities: protein sequences (ESM-2, Science 2023), DNA sequences (Evo, Science 2024; Nucleotide Transformer, Nature Methods 2024), single-cell transcriptomics (scGPT, Geneformer), etc. Each achieves impressive performance on modality-specific tasks, but a fundamental question remains unanswered:

**Do these independently trained models learn the same underlying biology?**

If a protein model and a DNA model are both trained on sequences from the same organism, they should — in principle — learn shared biological concepts (e.g., "this is a transmembrane region," "this is a catalytic site"). But they might also learn entirely different, modality-specific statistical patterns with no biological overlap.

**CrossBioSAE** answers this question by using **Sparse Autoencoders (SAEs)** — a mechanistic interpretability tool — to extract interpretable features from both a protein model and a DNA model, align them into a shared feature space, and test whether the shared features correspond to real biology.

## 2. Why SAEs?

SAEs decompose a model's internal activations into a sparse set of interpretable features. Originally developed for understanding large language models (Anthropic, OpenAI), SAEs have recently been applied to biological models (InterPLM, Nature Methods 2025; ProtSAE, AAAI 2025). However, all prior work applies SAEs to **single-modality** models.

Our contribution: **cross-modal SAE** — one SAE that simultaneously processes activations from two different foundation models (protein + DNA), bridging them through a shared interpretable feature space.

## 3. Architecture Design

### 3.1 The Problem

ESM-2 (protein) has hidden dimension 1280. Nucleotide Transformer (DNA) has hidden dimension 1024. They live in completely different vector spaces with different coordinate systems. You cannot directly compare or align their activations.

### 3.2 Our Solution

We add **learned linear adapters** that project each modality into a common 2048-dimensional shared space, then train a single SAE on this shared space:

```
                              ┌─────────────────┐
DNA act (1024-d) → Adapter_DNA → │                 │ → Adapter_DNA_inv → reconstruct DNA
                              │   Shared SAE     │
Prot act (1280-d) → Adapter_Prot → │  (2048 → 32k → 2048)│ → Adapter_Prot_inv → reconstruct Protein
                              └─────────────────┘
```

### 3.3 Three Training Losses

1. **Reconstruction loss** (MSE): The SAE must faithfully reconstruct the original activations for both modalities. This ensures the adapter + SAE pipeline doesn't destroy information.

2. **Sparsity loss** (TopK, k=128): Out of 32,768 features, only 128 are active for any given input. This enforces interpretability — each feature should correspond to a specific biological concept.

3. **Cross-modal consistency loss** (cosine similarity): For the same gene, the feature activation pattern from the DNA side and the protein side should be similar. This is the key loss that forces the SAE to discover **shared** biological features.

The cross-modal consistency weight is **critical**: we found that weight=0.1 produces no alignment (z-score = -3.19), while weight=1.0 produces strong alignment (z-score = 204.7). The SAE needs strong pressure to align the two modalities.

## 4. Data

### 4.1 Training Data

- **16,620 human protein-coding genes** with matched protein sequences and coding DNA sequences (CDS)
- Protein sequences: from UniProt (reviewed Swiss-Prot entries)
- CDS sequences: from NCBI RefSeq (real genomic coding sequences, not back-translated)
- For each gene, we extract:
  - **Protein activation**: Run ESM-2 650M, take layer-16 hidden states, mean-pool over all residues → 1280-dim vector
  - **DNA activation**: Run Nucleotide Transformer v2 500M, take layer-14 hidden states, mean-pool over all tokens → 1024-dim vector

### 4.2 Evaluation Data

- **ClinVar variants**: 50,000 pathogenic/benign single nucleotide variants for variant effect prediction
- **GO annotations**: 227,648 Gene Ontology term assignments for 16,039 genes (for function prediction)

## 5. Downstream Tasks

### Task 1: Variant Effect Prediction

**Question**: Can the cross-modal SAE predict whether a DNA mutation is pathogenic?

**Method**: For each ClinVar variant, compute feature disruption = |SAE(WT protein) - SAE(MT protein)|. Higher disruption → more likely pathogenic.

**Current results**:

| Method | AUROC |
|---|---|
| ESM-2 log-likelihood ratio (baseline) | 0.851 |
| CrossBioSAE feature disruption | 0.556 |
| ESM-2 + Nucleotide Transformer ensemble | 0.854 |

**Analysis**: The gene-level mean-pooled SAE is too coarse for single-residue variant scoring. ESM-2 LLR operates at per-residue resolution and naturally captures evolutionary constraints. We are integrating **Evo** (Science 2024, character-level DNA model) for per-nucleotide scoring, which should provide genuinely complementary signal.

### Task 2: Gene Function Prediction

**Question**: Can SAE features predict gene function without supervised labels?

**Method**: Auto-label each SAE feature with enriched GO terms (hypergeometric test). For a gene, its predicted function = the GO terms associated with its active features.

**Current results**: 25,822 features labeled with GO terms. Predictions generated for all 16,620 genes. Quantitative evaluation (held-out GO terms) pending.

### Task 3: Cross-modal Anomaly Detection

**Question**: Which genes have the most inconsistent representations across protein and DNA models?

**Method**: For each gene, compute cosine similarity between its protein-side features and DNA-side features. Low similarity = the two models "disagree" about this gene.

**Current results**: Most anomalous genes:
- **ZFR2** (consistency = 0): Zinc finger RNA-binding protein 2 — complete disagreement between protein and DNA models
- **RBMX**: RNA-binding motif protein, X-linked
- **ZNF proteins**: Zinc finger proteins are overrepresented in anomalies (12 of top 20)

The ZNF enrichment is biologically interesting: zinc finger domains have a very distinctive amino acid pattern (Cys2His2 motif) that protein models recognize easily, but the DNA encoding of zinc fingers uses repetitive sequences that DNA models may represent differently.

## 6. Key Findings So Far

1. **Cross-modal alignment is real**: z-score = 204.7 (p ≈ 0). ESM-2 and NT independently learn representations that can be aligned into a shared space with 202 shared features.

2. **Gene-level SAE features are too coarse for variant prediction**: Mean-pooling over the entire protein sequence dilutes single-residue mutation signals by ~500x. The SAE needs per-residue training for this task.

3. **Cross-modal anomaly detection reveals real biology**: ZNF overrepresentation in anomalies is consistent with the known structural distinctiveness of zinc fingers at protein vs DNA level.

4. **Hyperparameter sensitivity**: The cross-modal consistency loss weight must be high (≥1.0). At 0.1, the SAE degenerates into two independent single-modality SAEs with no alignment.

## 7. Pending Work

| Priority | Task | Expected Impact |
|---|---|---|
| 1 | Integrate Evo (Science 2024) as DNA model | Character-level LLR should significantly boost variant prediction ensemble |
| 2 | Per-residue SAE training | Enable fine-grained variant scoring, potentially beat ESM-2 LLR |
| 3 | Biological validation of anomaly detection | Cross-reference top anomalous genes with alternative splicing databases, literature |
| 4 | Quantitative function prediction evaluation | Held-out GO term prediction accuracy |
| 5 | Paper writing | Target: Nature Communications or ICLR 2026 |

## 8. Relationship to Prior Work

| Paper | Venue | What they did | What we add |
|---|---|---|---|
| InterPLM (Simon & Zou) | Nature Methods 2025 | SAE on ESM-2, found binding site features | Cross-modal: align protein + DNA SAEs |
| ProtSAE | AAAI 2025 | Semantically-guided SAE on protein LMs | Cross-modal consistency loss |
| Universal SAE (Cunningham et al.) | arXiv 2025 | SAE across different vision models | Apply to biological models, not vision |
| AlphaMissense | Science 2023 | Protein LM for variant pathogenicity | We add DNA modality + interpretable features |
| Evo | Science 2024 | DNA foundation model, variant scoring | We combine with protein model for ensemble |

No prior work has trained a cross-modal SAE bridging protein and DNA foundation models.
