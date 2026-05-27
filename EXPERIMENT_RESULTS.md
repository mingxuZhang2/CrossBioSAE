# CrossBioSAE — Critical Experiment Results & Current Status

## Context

After two rounds of GPT Pro review, we ran 5 critical experiments to determine whether SAE features provide value beyond raw model embeddings. These experiments use independently trained protein-SAE and DNA-SAE on 16,604 human genes, with ESM-2 650M (protein) and Nucleotide Transformer v2 500M (DNA).

---

## Experiment 1: Raw Activation vs SAE Baseline

**Question**: Does the SAE improve cross-modal alignment, or do raw embeddings already align just as well?

| Method | Retrieval R@1 | R@5 | MRR |
|---|---|---|---|
| Random projection | 0.2% | 0.4% | 0.008 |
| Raw activation (direct cosine) | 0.1% | 0.6% | 0.008 |
| PCA (dim=256) | 0.0% | 0.4% | 0.007 |
| **Raw embeddings + CCA** | **19.9%** | **40.6%** | **0.304** |
| SAE features (direct cosine) | 0.3% | 0.7% | 0.009 |
| **SAE features + CCA** | **13.0%** | **34.1%** | **0.233** |

**Finding**: Raw + CCA (19.9%) beats SAE + CCA (13.0%). The SAE compresses away information useful for alignment. CCA is the key tool — it extracts shared structure regardless of whether input is raw or SAE features.

**Implication**: The cross-modal alignment signal lives primarily in the raw ESM-2/NT embeddings, not in the SAE features. The SAE loses ~35% of the alignment signal through its sparse bottleneck.

---

## Experiment 2: Family-Split Retrieval

**Question**: Is the alignment driven by gene family / paralog memorization, or by more abstract biological structure?

| Method | R@1 | R@5 | MRR |
|---|---|---|---|
| SAE + CCA (random split) | 12.8% | 32.3% | 0.227 |
| SAE + CCA (GO-family holdout) | **10.0%** | **23.6%** | **0.175** |

**Finding**: Family holdout drops R@1 from 12.8% to 10.0% (22% decrease), but 10.0% is still 330× above random chance (0.03%). The alignment is not purely gene family memorization — there is genuine biological structure beyond family.

---

## Experiment 3: Hard-Negative Retrieval (Length-Matched)

**Question**: Can SAE features distinguish same-gene pairs from same-length genes?

| Method | R@1 | R@5 | MRR | n |
|---|---|---|---|---|
| Full gallery (all genes) | 0.0% | 0.0% | 0.001 | 1000 |
| Length-matched gallery | 0.2% | 0.6% | 0.010 | 1000 |

**Finding**: Without CCA, SAE features cannot distinguish same-gene matches from length-matched controls. This confirms that raw SAE feature similarity (cosine distance) has no cross-modal alignment — CCA is essential.

**Note**: This experiment used PCA-reduced SAE features without CCA alignment. The near-zero retrieval is consistent with Experiment 1 where "SAE features (direct)" also had ~0% R@1.

---

## Experiment 4: Held-Out GO Function Prediction

**Question**: Do SAE features preserve biological information for function prediction?

| Feature Representation | Mean GO AUROC (50 terms, 5-fold CV) |
|---|---|
| **ESM-2 raw** | **0.856** |
| Protein SAE | 0.854 |
| ESM-2 + NT concat (raw) | 0.855 |
| Protein + DNA SAE concat | 0.855 |
| NT raw | 0.767 |
| DNA SAE | 0.761 |

**Finding**: SAE features perform equally to raw embeddings for GO prediction (0.854 vs 0.856). The SAE does not lose biological signal for function classification. However, it also does not improve it. DNA features (NT) are substantially weaker than protein features (ESM-2) regardless of representation.

**Implication**: The SAE faithfully preserves gene-level biological information, but provides no prediction advantage over raw embeddings.

---

## Experiment 5: Cross-Modal Zero-Shot Transfer

**Question**: Can a classifier trained on protein features transfer to DNA features?

| Metric | Value |
|---|---|
| Same-modality AUROC (train protein → test protein) | **0.839** |
| Cross-modal AUROC (train protein → test DNA) | **0.767** |
| **Transfer efficiency** | **91.4%** |

**Finding**: A GO classifier trained on protein CCA features retains 91.4% of its performance when applied to CCA-aligned DNA features. This is strong evidence that the two models encode functionally equivalent gene representations.

**Note**: This uses CCA alignment, not CrossBioSAE. The result demonstrates that ESM-2 and NT share transferable functional representations — but CCA alone is sufficient to unlock this.

---

## Summary Table

| Experiment | Key Number | What It Means |
|---|---|---|
| Raw vs SAE alignment | Raw+CCA 19.9% > SAE+CCA 13.0% | SAE hurts alignment (loses information) |
| Family-split | 10.0% R@1 after holdout | Real biological signal beyond family |
| Hard negatives | ~0% without CCA | CCA is essential, SAE features alone don't align |
| GO prediction | SAE 0.854 ≈ Raw 0.856 | SAE preserves biology but doesn't improve prediction |
| Cross-modal transfer | 91.4% transfer efficiency | Strong shared representations (but CCA alone suffices) |

---

## The Core Problem

**The scientific finding is real**: ESM-2 and NT independently learn partially aligned representations of gene biology. Cross-modal transfer works at 91.4% efficiency. This is a publishable result.

**But CrossBioSAE specifically doesn't add value over simpler methods**:

1. **For alignment**: Raw embeddings + CCA outperform SAE + CCA (19.9% vs 13.0%). The SAE's sparse bottleneck (TopK=128 out of 20k features) throws away alignment-useful information.

2. **For prediction**: SAE features match raw embeddings exactly on GO prediction (0.854 vs 0.856). No improvement.

3. **For transfer**: The 91.4% cross-modal transfer uses CCA, not the SAE. CCA alone is sufficient.

4. **Joint CrossBioSAE**: In our earlier alignment benchmark, joint CrossBioSAE + CCA (14.3% R@1) also underperforms raw + CCA (19.9%).

**The SAE's unique contribution is interpretability** — sparse features that can be mapped to GO terms and biological concepts. But the alignment, prediction, and transfer results don't require an SAE.

---

## Three Possible Directions

### Direction A: Paper about bio-FM alignment (de-emphasize SAE)

**Story**: "Independently trained protein and DNA foundation models learn functionally equivalent representations. CCA alignment enables 91.4% cross-modal transfer for gene function prediction."

- Core contribution: the scientific finding about bio-FM convergence
- SAE becomes one analysis tool among several (CCA, Procrustes, etc.)
- CrossBioSAE is a method variant, not the headline
- Strongest venue fit: ICLR / NeurIPS (representation learning angle)

### Direction B: Make SAE genuinely useful

**Story**: "SAE provides interpretable decomposition of shared vs modality-specific biology between protein and DNA models."

Requires finding tasks where SAE's interpretability matters:
- Feature-level discordance analysis (WHY does a gene have low cross-modal consistency?)
- Concept-level biological atlas (which biological concepts are shared vs modality-specific?)
- SAE feature cards with GO/Pfam/structure evidence
- This is harder to quantify but could work for Nature Methods (interpretability tool)

### Direction C: Different architecture

**Approach**: Instead of standard SAE → CCA, design an architecture where sparsity actually helps alignment.
- Contrastive sparse coding: sparsity as an inductive bias for cross-modal matching
- Per-residue SAE: local features where sparsity captures structural motifs
- Feature-gated alignment: use SAE features to weight CCA components

---

## Open Questions for Review

1. **Is "bio-FM alignment + CCA transfer" alone publishable?** The finding is strong (91.4% transfer), but CCA is a well-known method. What's the novelty?

2. **Can interpretability alone justify the SAE?** InterPLM (Nature Methods 2025) did single-modality SAE interpretability. Is cross-modal SAE interpretability a sufficient step up?

3. **Should we pursue Direction A (alignment paper) or Direction B (interpretability tool)?** Direction A has stronger numbers; Direction B has more methodological novelty.

4. **Is the variant prediction task salvageable?** Per-residue SAE would be a significant engineering effort. Is it worth it, or should we drop variants entirely?

5. **How to frame the "SAE doesn't beat raw+CCA" finding?** Honestly reporting this is important for scientific credibility, but it weakens the SAE motivation.
