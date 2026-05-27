# CrossBioSAE — Full Experiment Results & Project Status

## Overview

Three rounds of experiments + three rounds of GPT Pro review. Results below reflect the complete picture.

---

## Round 1: Initial Results (16,604 genes)

| Metric | Joint CrossBioSAE |
|---|---|
| Cross-modal similarity | 0.974 |
| Z-score vs permutation | 204.7 |
| Shared features | 202 / 32,768 |

**GPT Pro concern**: This proves training objective, not independent convergence (cross-modal loss forces alignment).

## Round 2: Independent SAE + Post-hoc Alignment

| Method | Retrieval R@1 | R@5 | MRR |
|---|---|---|---|
| Independent SAEs + CCA | 10.3% | 25.9% | 0.188 |
| Independent SAEs + Procrustes | 8.4% | 18.6% | 0.141 |
| Joint CrossBioSAE + CCA | 14.3% | 32.4% | 0.239 |

**Finding**: Independent alignment exists (10.3% R@1, 340× chance). Not circular.

**GPT Pro concern**: But is this SAE's contribution, or do raw embeddings already align?

## Round 3: Fair Benchmark — Raw vs SAE (definitive)

### Exp A: Raw activation vs SAE vs PCA (best config per representation)

| Representation | Best R@1 | Best Config |
|---|---|---|
| **Raw embedding + CCA** | **30.3%** | pca=512, cca=32 |
| SAE reconstruction + CCA | 19.9% | pca=512, cca=32 |
| SAE features + CCA | 14.3% | pca=512, cca=32 |
| Random projection | ~0.2% | — |

**Raw beats SAE by 2×.** The SAE's sparse bottleneck (TopK=64 out of 20k features) throws away alignment-useful information. SAE reconstruction (dense output) recovers ~66% of raw alignment.

### Exp B: Hard-Negative Retrieval WITH CCA

| Method | Full Gallery R@1 | Length-Matched R@1 |
|---|---|---|
| Raw + CCA | 19.2% | 33.8% |
| SAE + CCA | 10.2% | 22.4% |

**Alignment survives length-matching** — not driven by protein/CDS length confound. Both Raw and SAE maintain strong retrieval in hard-negative setting.

### Exp C: Low-Label GO Prediction (mean AUROC across 30 GO terms)

| Features | 1% labels | 5% | 10% | 25% | 100% |
|---|---|---|---|---|---|
| **ESM-2 raw** | **0.660** | **0.735** | **0.777** | **0.828** | **0.873** |
| Protein SAE | 0.638 | 0.724 | 0.768 | 0.825 | 0.871 |
| ESM+NT raw concat | 0.660 | 0.737 | 0.777 | 0.828 | 0.873 |
| Prot+DNA SAE concat | 0.642 | 0.727 | 0.769 | 0.828 | 0.872 |
| NT raw | 0.583 | 0.651 | 0.677 | 0.726 | 0.784 |
| DNA SAE | 0.573 | 0.635 | 0.670 | 0.724 | 0.775 |

**SAE ~2% below raw at all label fractions.** No low-label advantage. Adding DNA modality doesn't help protein-side GO prediction.

### Exp D: Leakage-Free Cross-Modal Transfer

| Metric | Raw + CCA | SAE + CCA |
|---|---|---|
| Protein→Protein (same modality) | 0.840 | 0.836 |
| Protein→DNA (cross-modal) | 0.782 | 0.766 |
| DNA→DNA (baseline) | 0.789 | 0.779 |
| **Corrected transfer ratio** | **82.3%** | **78.5%** |

Strict protocol: CCA fit on train only, classifier fit on train only, evaluated on held-out test.

**Key insight**: Cross-modal AUROC (0.782) ≈ DNA baseline (0.789). The "transfer" may largely reflect DNA embedding's own predictive power, not protein knowledge transferred to DNA.

---

## Definitive Conclusions

### What we proved

1. **Bio-FM alignment is real**: ESM-2 and NT raw embeddings can be CCA-aligned with R@1=30.3% (1000× above chance). This is not confound — it survives length-matching.
2. **Alignment is not family memorization**: Family-split retrieval still gives R@1=10% (SAE+CCA).
3. **SAE preserves biological information**: GO AUROC 0.871 vs raw 0.873 — near-lossless compression.

### What we disproved

4. **SAE does NOT improve alignment**: Raw+CCA (30.3%) > SAE+CCA (14.3%). Sparse bottleneck hurts.
5. **SAE does NOT improve prediction**: GO AUROC identical to raw at all label fractions.
6. **Cross-modal transfer ≠ knowledge transfer**: Protein→DNA AUROC ≈ DNA→DNA baseline.
7. **Joint CrossBioSAE does NOT beat raw+CCA**: Joint (14.3%) < Raw+CCA (30.3%).

### SAE's actual value

8. **Interpretability**: SAE decomposes representations into sparse features mappable to GO/Pfam.
9. **Feature atlas**: Shared vs protein-specific vs DNA-specific feature classification.
10. **Anomaly detection**: ZNF/RBMX/ZFR2 discordance findings remain unique to SAE analysis.

---

## Project Direction

### Paper story (revised)

> **"Dense representations of protein and DNA foundation models contain strong cross-modal gene biology, unlockable by CCA. Sparse autoencoders provide interpretable decomposition into shared and modality-specific biological concepts."**

### Figure plan

1. **Raw+CCA alignment** — phenomenon exists (R@1=30.3%)
2. **Controls** — survives hard negatives, family split
3. **SAE as interpretable lens** — feature atlas, shared/specific/discordant
4. **Anomaly detection** — discordant genes and families
5. **Limitation** — SAE loses alignment signal, variant prediction too coarse

### What NOT to claim

- SAE improves prediction
- CrossBioSAE outperforms simpler methods
- Cross-modal transfer = protein knowledge transfer

### Remaining work

- [ ] Feature atlas with biological validation (feature cards)
- [ ] ZNF discordance case study with confound controls
- [ ] CCA-teacher sparse distillation (if SAE alignment improvement needed)
- [ ] Paper writing
