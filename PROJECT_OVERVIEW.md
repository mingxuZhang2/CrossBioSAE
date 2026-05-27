# CrossBioSAE — Project Overview

## 1. Research Question

Biological foundation models are trained independently on different modalities: protein sequences (ESM-2), DNA sequences (Nucleotide Transformer, Evo), single-cell data (scGPT, Geneformer). Each achieves impressive performance, but a fundamental question remains:

**Do independently trained protein and DNA models learn the same underlying biology?**

CrossBioSAE answers this using **Sparse Autoencoders (SAEs)** — a mechanistic interpretability tool — to extract interpretable features from both models and test whether those features are alignable *without any cross-modal training signal*.

## 2. Why This Matters

If protein and DNA models independently converge on shared biological representations, it implies:
- These representations capture real biology (not just statistical patterns)
- Cross-modal transfer is possible (annotate DNA using protein knowledge, and vice versa)
- Genes where models *disagree* are biologically interesting (alternative splicing, rapid evolution, unusual codon usage)

## 3. Two Experimental Approaches

### Approach 1: Independent SAEs + Post-hoc Alignment (primary scientific claim)

This addresses the key reviewer concern that joint training with a cross-modal loss creates circular evidence.

1. Train a **protein-SAE** on ESM-2 activations (no DNA information)
2. Train a **DNA-SAE** on NT activations (no protein information)
3. Measure alignment post-hoc using CCA, Procrustes rotation, linear probes, and cross-modal retrieval

**If alignment exists without cross-modal training**, it proves the models independently learned shared biology.

### Approach 2: Joint CrossBioSAE (shared feature space)

One SAE with learned adapters processes both modalities:

```
DNA act (1024-d)  → Adapter_DNA (linear) → Shared SAE (32k features) → reconstruct DNA
Prot act (1280-d) → Adapter_Prot (linear) →                         → reconstruct Protein
```

Three training losses: reconstruction (MSE) + sparsity (TopK) + cross-modal cosine consistency.

The joint model creates a *shared interpretable feature space* useful for downstream tasks, but its alignment metric alone cannot prove independent convergence.

## 4. Data Pipeline

### Training Data
- **16,620 human protein-coding genes** with matched protein sequences and coding DNA sequences
- Protein: from UniProt (reviewed Swiss-Prot entries)
- CDS: from NCBI RefSeq, **translation-verified** (translate(CDS) == UniProt protein)
- Activations: ESM-2 650M layer 16 (mean-pooled, 1280-d) + NT v2 500M layer 14 (mean-pooled, 1024-d)

### Quality Controls (addressing reviewer feedback)
- CDS-protein translation consistency check enforced at download time
- No silent synthetic fallback — API failures raise explicit errors
- All activation files include model, layer, pooling, and data-origin metadata

### Evaluation Data
- **50,000 ClinVar variants** (26,841 pathogenic, 23,159 benign)
- **227,648 GO term assignments** for 16,039 genes

## 5. Results

### 5.1 Independent Alignment (Core Finding)

Independently trained SAE features are alignable, providing non-circular evidence:

| Method | Retrieval R@1 | R@5 | MRR |
|---|---|---|---|
| Random baseline | ~0.03% | ~0.15% | — |
| **Independent SAEs + CCA** | **10.3%** | **25.9%** | **0.188** |
| Independent SAEs + Procrustes | 8.4% | 18.6% | 0.141 |
| Joint CrossBioSAE + CCA | 14.3% | 32.4% | 0.239 |

CCA mean correlation between independent protein and DNA features: **0.581** (out of 1.0).

**Interpretation**: ESM-2 and NT independently learn partially overlapping representations of coding gene biology. Joint training (CrossBioSAE) improves alignment ~40% over post-hoc methods, but independent features already carry strong cross-modal structure.

### 5.2 Cross-modal Anomaly Detection (Biological Discovery)

Genes ranked by protein-DNA representational disagreement (consistency score):

**Most anomalous** (protein and DNA models disagree):
- ZFR2 (consistency = 0.000) — zinc finger RNA-binding protein 2
- RBMX (consistency = 0.000) — RNA-binding motif, X-linked
- ZNF776, CMTM2, DMKN

**Gene family enrichment in top-200 anomalous genes**: ZNF (zinc-finger) proteins are systematically overrepresented (Fisher's exact test). This is biologically plausible — zinc finger domains have distinctive Cys2His2 amino acid motifs that protein models recognize easily, but the underlying DNA uses highly repetitive sequences that DNA models may represent differently.

### 5.3 Variant Effect Prediction (Limitation)

| Method | AUROC |
|---|---|
| ESM-2 LLR (per-residue) | 0.851 |
| CrossBioSAE feature disruption | 0.556 |
| ESM-2 + NT ensemble (LR) | 0.854 |

Gene-level mean-pooled SAE features are too coarse for single-residue variant scoring — a mutation affecting 1 residue out of 500 is diluted ~500× by mean pooling. Per-residue SAE is needed for this task (planned as future work).

### 5.4 Gene Function Prediction

25,822 SAE features labeled with GO terms via hypergeometric enrichment + Benjamini-Hochberg FDR correction (alpha=0.05). Function predictions generated for all 16,620 genes. Quantitative held-out evaluation pending.

## 6. Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Shared feature definition | Matched-gene co-activation correlation > 0.1 | Stricter than marginal overlap; reviewer-requested |
| GO enrichment | Hypergeometric + BH-FDR | Controls false positives across 32k features × 17k GO terms |
| Alignment evaluation | Train/test split, retrieval + CCA + Procrustes | Avoids overfitting; multiple complementary metrics |
| Cross-modal weight | 1.0 (not 0.1) | Weight=0.1 produces no alignment (z-score = -3.19) |
| Variant prediction | Reported as limitation | Gene-level SAE cannot compete with per-residue ESM-2 LLR |

## 7. Relationship to Prior Work

| Paper | Venue | What they did | What we add |
|---|---|---|---|
| InterPLM | Nature Methods 2025 | SAE on ESM-2 → interpretable protein features | Cross-modal: align protein + DNA SAEs |
| ProtSAE | AAAI 2025 | Semantically-guided SAE on protein LMs | Cross-modal consistency loss |
| Universal SAE | arXiv 2025 | Shared SAE across vision models | Apply to biological models; add independent-then-align validation |
| AlphaMissense | Science 2023 | Protein LM for variant pathogenicity | We add DNA modality + interpretable features |
| Evo | Science 2024 | DNA foundation model | We bridge with protein model for cross-modal analysis |

**No prior work** has trained independent SAEs on protein and DNA foundation models and measured post-hoc alignment to test whether independently trained bio-FMs learn shared biology.

## 8. Pending Work

| Priority | Task | Impact |
|---|---|---|
| 1 | Evo integration (7B, character-level) | Stronger DNA signal for variant ensemble |
| 2 | Per-residue/per-codon CrossBioSAE | Enable fine-grained variant scoring |
| 3 | ZNF case study with confound controls | Strengthen anomaly finding |
| 4 | Held-out GO prediction evaluation | Quantify function prediction accuracy |
| 5 | Species transfer (mouse/zebrafish) | Test cross-species generalization |
| 6 | Paper writing | Target: Nature Communications / ICLR |

## 9. Paper Narrative (Planned)

> **Title direction**: "Sparse alignment of protein and nucleotide language models reveals concordant and discordant gene biology"

**Figure 1**: Method — ESM-2, NT, independent SAEs, joint CrossBioSAE, alignment benchmark
**Figure 2**: Independent alignment — retrieval, CCA, Procrustes, baselines
**Figure 3**: Shared sparse biological concepts — feature cards with GO/Pfam/structure evidence
**Figure 4**: Modality-specific concepts — protein-only vs DNA-only features
**Figure 5**: Discordance atlas — ZNF enrichment, case studies, confound analysis
**Figure 6**: Downstream applications — function prediction, anomaly detection
