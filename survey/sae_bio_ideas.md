# Research Ideas: SAE + Biological LLMs + Scientific Discovery

> Final Stage-1 deliverable. Each idea is executable in 3–6 months on HPC3 by a no-wet-lab researcher (Dr. Zhang).
> Each idea is designed so the SAE features themselves *carry the discovery claim*, validated by orthogonal databases or models, not by wet experiments.
> Updated: 2026-05-26

---

## Idea 1 — **PathoSAE**: Interpretable pathogenic-variant features in ESM-C via temporal ClinVar hold-out

**One-line pitch:** Train an SAE on ESM-Cambrian protein embeddings and discover the *interpretable feature subset* whose disruption predicts pathogenic missense variants on a strict temporal ClinVar hold-out — moving from "ESM predicts pathogenicity" (AlphaMissense, Brandes et al.) to "this *specific interpretable feature* predicts pathogenicity, and disrupting it predicts the phenotype."

**Hypothesis (biology):** Disease-causing missense variants disproportionately disrupt a small subset of SAE features that correspond to *structural* and *catalytic* core residues (binding pockets, catalytic triads, allosteric sites). Benign variants do not. The SAE feature attribution should beat the raw PLM logit on temporal-hold-out variants released after the SAE training cutoff.

**Concrete plan:**
- **Model:** ESM-Cambrian (recommended) or ESM-C 600M as fallback. Layer = mid (≈ layer 24/48). Compare to ESM-2 650M layer 30 as legacy baseline.
- **SAE variant:** JumpReLU SAE (best stability per Gao et al.); compare to TopK (k=64). Expansion factor 32x → ~12k features for ESM-C 600M (d=2048).
- **Training data:** UniRef50, exclude any sequences with date > 2024-12. ~50M sequences. ~3–5 days on 4×H100.
- **Auto-label:** UniProt features (sites, binding, active_site, modified_residue), Pfam domains, InterPro signatures. Auto-label every feature with the top-3 enriched annotation classes.
- **Discovery probe:**
  1. Freeze SAE before 2025-01.
  2. Take ClinVar variants released between 2025-01 and 2026-05 (post-SAE-freeze) that are not in ESM-Cambrian training.
  3. For each variant, compute Δfeature-activation = SAE(WT) − SAE(MT).
  4. Predict pathogenicity from feature-disruption attribution; compare to AlphaMissense and ESM-1b baselines.
  5. Identify the top-100 *causally implicated* features; show they correspond to structurally / catalytically conserved residues in the AlphaFold3 confidence map for the same proteins.

**Discovery validation strategy (no wet lab):**
- **Primary:** Temporal hold-out — ClinVar 2025–2026 entries with high-confidence pathogenicity calls. Sensitivity/specificity against AlphaMissense.
- **Orthogonal oracle 1:** AlphaFold3 confidence at the variant position; structurally critical positions should overlap with high-feature-disruption residues.
- **Orthogonal oracle 2:** Recovered features should be enriched in DMS assays from ProteinGym (217 assays).
- **Cross-species:** Repeat on mouse / zebrafish ClinVar-equivalents (HGMD-like sources).
- **Statistical test:** Pre-register feature rankings, then test; report effect sizes with multiple-testing correction.

**Compute budget estimate:**
- SAE training on ESM-C 600M with 50M sequences: ~96 GPU-hours (4×H100 × 1 day).
- Feature labelling + analyses: 1 H100-week.
- **Total: ~2 weeks on 4×H100.**

**Risk (failure mode):**
- ESM-C may already implicitly encode pathogenicity in a non-sparse way, making the SAE no better than the raw model. Mitigation: pre-register baseline comparison; report honestly if SAE = ESM-C performance.
- Temporal contamination: ESM-Cambrian release notes need careful checking; if its training cutoff is after some 2024 ClinVar entries, the hold-out is contaminated. Mitigation: use very recent ClinVar (2025-Q4 or 2026) only.

**Target venue:** **Nature Methods** (method + clinical discovery angle) or **Nature Genetics**. Strong fit because AlphaMissense → Science precedent exists; reviewers will accept SAE-interpretation as the value-add. Fallback: PNAS, Nature Communications.

**Closest related work and what they didn't do:**
- *AlphaMissense* (Cheng et al., Science 2023): pathogenicity prediction, no interpretability.
- *ESM-1b variant prediction* (Brandes et al., Nature Genetics 2023): raw model, no SAE.
- *InterPLM* (Simon & Zou, Nature Methods 2025): features ↔ binding sites, no disease angle.
- *Tsui et al., 2025 (Low-$N$ SAE)*: fitness from DMS, no clinical variants.

None of these combine (i) ESM-Cambrian, (ii) SAE interpretability, (iii) temporal ClinVar hold-out, (iv) cross-validation with AlphaFold3 confidence. The combination is the contribution.

---

## Idea 2 — **EnhancerSAE**: Discovering combinatorial enhancer grammar via SAE on a chromatin-accessibility foundation model

**One-line pitch:** Train an SAE on activations of CLM-access (the new scATAC-seq foundation model, 2.8M cells × 1M peaks) and demonstrate that SAE features recover known *combinatorial* TF binding motifs, while also surfacing candidate novel cell-type-specific cofactor combinations — testable against ENCODE TFBS data not used in CLM-access training.

**Hypothesis (biology):** Chromatin accessibility is governed by *combinations* of TFs (e.g., GATA + TAL1 in erythroid cells, OCT4 + SOX2 in pluripotent cells). A well-trained chromatin FM should encode these combinations as compositional latent variables. SAE features should map to TF *cocktails*, not single TFs, distinguishing them from JASPAR motif scanning.

**Concrete plan:**
- **Model:** CLM-access (just released, bioRxiv 2025). Fallback: Enformer or Borzoi at the layer that processes 100kb windows.
- **SAE variant:** TopK (k=128) with Matryoshka hierarchy (groups of 32, 128, 512 features) to capture multi-scale concepts (TF → TF-cocktail → cell-state).
- **Training data:** CLM-access activations on ENCODE chromatin atlas (or its training subset minus held-out cell types).
- **Discovery probe:**
  1. Train SAE on activations from ~70% of cell types.
  2. Held-out cell types (30%): predict their accessibility patterns using only SAE features. Compare to raw CLM-access.
  3. Auto-label every feature with: (a) top JASPAR/HOMER TF motif match, (b) top KEGG pathway, (c) cell-type specificity score.
  4. Identify "TF cocktail" features — features that correspond to a *combination* of motifs not explainable by any single motif. Test whether these cocktails recur in ENCODE ChIP-seq co-occurrence data not used in CLM-access training.
  5. Steer the model with a cocktail feature; predict differential accessibility; compare to held-out cell-type-specific ATAC peaks.

**Discovery validation strategy (no wet lab):**
- **Primary:** Recovery of canonical TF cocktails (GATA1+TAL1, OCT4+SOX2+NANOG, MYOD1+MYF5) without supervision.
- **Orthogonal oracle 1:** ENCODE ChIP-seq peak co-occurrence (~thousands of TFs across hundreds of cell types) — used as held-out validation.
- **Orthogonal oracle 2:** scATAC + scRNA cross-modal cohort (e.g., 10x Multiome) — does a feature predicted to be active in cell type X also correlate with the cell type's RNA profile?
- **Cross-species:** Mouse ENCODE for conservation of discovered cocktails.

**Compute budget estimate:**
- SAE training on CLM-access (assumed similar to Geneformer scale): ~120 GPU-hours.
- Feature labelling + ENCODE cross-reference: 2 H100-weeks (this is the heavy step due to ENCODE size).
- **Total: ~3–4 weeks on 4×H100.**

**Risk (failure mode):**
- CLM-access may not have stable enough activations for clean SAE training (scATAC data is extremely sparse). Mitigation: fallback to Enformer (proven activations) or Borzoi.
- TF cocktails may be too cell-type-specific to recover with a single SAE. Mitigation: train per-cell-type-cluster SAEs; or use Matryoshka hierarchy explicitly.
- The "Systematic Evaluation" 2026 paper warned that attention in scFMs captures co-expression rather than regulation. The same risk applies here — SAE may capture co-accessibility, not causality. Mitigation: include a causal-intervention experiment using held-out CRISPRi data.

**Target venue:** **Nature Methods** (chromatin FM + interpretability = clear fit), **Genome Research**, **Nature Communications**. Genome Biology if more biology-heavy framing. Workshop fallback: GenBio @ ICML/NeurIPS.

**Closest related work and what they didn't do:**
- *CLM-access* (bioRxiv 2025): a chromatin FM with no interpretability analysis.
- *Enformer / Borzoi* (Nature 2021 / Nature Methods 2024): chromatin accessibility prediction but no SAE.
- *EpiGePT* (Genome Biology 2024): epigenome predictor, no SAE.
- *Bio-SAE landscape* (sae_bio_landscape.md): explicitly lists "SAE on epigenomic/chromatin models" as Gap 5 — currently empty.

This is the cleanest open territory in the SAE+bio landscape today.

---

## Idea 3 — **CrossBioSAE**: Universal Sparse Autoencoder aligning protein-LM and DNA-LM features on matched coding genes

**One-line pitch:** Train a single SAE that ingests activations from *both* a PLM (ESM-C) and a DNA-LM (Evo-2) on paired protein/coding-DNA inputs, and test whether biologically meaningful concepts (binding sites, secondary structure, signal peptides) live in a *shared* feature space across modalities — the strongest possible test of whether bio-FMs converge on the same latent biology.

**Hypothesis (biology):** Some biological concepts (e.g., "this is a transmembrane region," "this is a signal peptide," "this region is highly conserved") should be encoded similarly by *both* the protein and the DNA model, because both models see the same underlying biology from different angles. Shared SAE features should preferentially activate on the same gene across modalities.

**Concrete plan:**
- **Models:** ESM-C 600M (protein) + Evo-2 7B (DNA). Layer choice: mid-depth in both.
- **SAE variant:** Universal SAE (Cunningham et al., USAE 2025) adapted for biology: one SAE decodes from a shared latent space to either modality's activations. Expansion factor 32x.
- **Training data:** Matched protein/coding-DNA pairs from RefSeq + UniProt. ~1M genes. ~5 H100-days.
- **Discovery probe:**
  1. For each gene, compute SAE features from protein activation and from DNA activation; measure feature overlap.
  2. Identify *shared features* (active in both modalities for the same gene) and *modality-specific features*.
  3. Auto-label shared features against UniProt + ENCODE annotations. Hypothesis: shared features should be enriched for *biological* annotations (transmembrane, signal peptide, conserved domain); modality-specific features should be enriched for *technical* artefacts (codon usage on the DNA side, amino acid composition on the protein side).
  4. Test on a held-out gene set: given only the DNA, predict the protein-side feature activation pattern via the shared latent; compare to a no-shared-latent baseline.

**Discovery validation strategy (no wet lab):**
- **Primary:** Cross-modal alignment score — how many SAE features fire on the same gene from both modalities, against a permutation baseline.
- **Orthogonal oracle 1:** AlphaFold3 confidence map for the same gene's protein product.
- **Orthogonal oracle 2:** GTEx eQTL data — do DNA variants that disrupt a *shared* feature have larger eQTL effects than variants that disrupt only a DNA-specific feature?
- **Negative-result fallback:** If no shared features exist, that itself is a striking result: it would mean Evo-2 and ESM-C, despite both being trained on biological sequences, do not converge on shared concepts. This is a defensible Nature Communications / ICLR paper either way.

**Compute budget estimate:**
- Activation extraction (both models on 1M pairs): ~50 GPU-hours.
- Universal SAE training: ~100 GPU-hours.
- Cross-modal probing + eQTL correlation: 1 H100-week.
- **Total: ~3 weeks on 4×H100.**

**Risk (failure mode):**
- Evo-2 7B activations are large (long context); storage and IO may be bottleneck. Mitigation: subsample to short coding sequences (<2kb).
- Shared latent space may collapse to trivial features (mostly codon-usage on DNA side). Mitigation: add an explicit decorrelation loss; weight modalities equally in reconstruction.
- The two models' representations may simply not be alignable, in which case the paper becomes a *negative* discovery (also publishable but with different framing).

**Target venue:** **Nature Communications**, **ICLR**, **NeurIPS**. The cross-modal angle has strong ML interest (USAE precedent) + strong biology interest (do FMs converge?).

**Closest related work and what they didn't do:**
- *Universal SAEs* (Cunningham et al., 2025): vision models only.
- *Generalised biological FM* (Nature MI 2025): joint protein+DNA training, no SAE.
- *GenBioAI* (Nature Biotechnology 2026): joint vision paper, no SAE.
- *Goodfire / Arc Institute Evo-2 SAE*: single modality (DNA only).

No paper has attempted cross-modal SAE alignment in biology. The convergence-or-not result is intrinsically interesting.

---

## Idea 4 — **RNA-MotifSAE**: Discovering RNA regulatory grammar from SAEs on Orthrus and AIDO.RNA, validated against m6A and IRES databases

**One-line pitch:** Apply SAEs to two recent RNA foundation models (Orthrus, Nature Methods 2026; AIDO.RNA 1.6B, 2024) and demonstrate that SAE features recover RNA regulatory primitives (m6A sites, IRES elements, poly(A) signals, secondary-structure stems) without supervision, with each feature class validated against a database not used in RNA-FM training.

**Hypothesis (biology):** RNA biology is built from a few hundred recurring functional primitives. A well-trained RNA-LM should learn these primitives as distinct latent directions. SAEs should surface them as monosemantic features — more cleanly than for proteins (because RNA primitives are sharper than protein domains).

**Concrete plan:**
- **Models:** Orthrus (Nature Methods 2026; primary), AIDO.RNA 1.6B (secondary). Pick a mid-layer.
- **SAE variant:** TopK (k=64), expansion factor 16x. Matryoshka if cross-scale features expected.
- **Training data:** RNAcentral (40M ncRNAs) + GENCODE mRNAs. ~3 H100-days.
- **Discovery probe (one per RNA primitive):**
  - **m6A sites:** Held-out validation = m6A-Atlas v3.0 (released 2024) sites; check SAE features that fire selectively on m6A sites.
  - **IRES elements:** IRESbase (2025 updates); check features that fire on viral and cellular IRES separately.
  - **Poly(A) signals:** PolyA_DB v3.2; features should recover AAUAAA hexamer plus flanking context.
  - **Secondary-structure stems:** bpRNA-1m, RNAcentral structures; features should align with hairpin loops and stems of various lengths.
  - **Stop-codon read-through / SECIS elements:** PMID-curated catalogs.
- **Steering experiment:** Use the IRES feature to steer Orthrus to generate IRES candidates; score with an external IRES predictor (e.g., IRESpy, or the GRAPE-LM RNA generator's stability score).

**Discovery validation strategy (no wet lab):**
- **Primary:** F1 score of feature ↔ database hit, across 6 RNA primitives.
- **Orthogonal oracle:** External RNA structure predictor (RNAfold, EternaFold, RhoFold+) for structural features; m6A predictors (DeepM6A, SRAMP) for modification features.
- **Cross-species:** Train on human RNA, test on yeast and plant RNAs. Are the same features recovered? Use PlantRNA-FM hold-out cohort.
- **Temporal hold-out:** Train SAE on m6A-Atlas v2 (2023); test on v3 (2024) additions.

**Compute budget estimate:**
- AIDO.RNA 1.6B activations + SAE training: ~80 GPU-hours.
- Database cross-reference + steering experiments: 1 H100-week.
- **Total: ~2 weeks on 4×H100.**

**Risk (failure mode):**
- Orthrus may have learned mostly evolutionary similarity, not functional primitives. Mitigation: compare to AIDO.RNA which is autoregressive and may have learned different things.
- m6A sites are notoriously hard to predict from sequence alone (they depend on RNA-binding-protein context). SAE features may not separate them cleanly. Mitigation: focus on the cleaner primitives (IRES, poly(A), stems) and treat m6A as the hardest test.

**Target venue:** **Nature Communications**, **Nature Methods**, **RNA Society conferences**, **NeurIPS Datasets & Benchmarks**. The RNA-FM community is small but high-quality.

**Closest related work and what they didn't do:**
- *SAE-RNA* (arXiv 2025): RiNALMo only, small-scale, no temporal or database hold-out.
- *Orthrus* (Nature Methods 2026): no interpretability analysis published.
- *AIDO.RNA*: no SAE in release.
- *PlantRNA-FM*: no SAE.

The RNA-FM space is one of the fewest-papers / highest-payoff corners of bio-LLM interpretability.

---

## Idea 5 (skeptical alternative) — **AuditSAE**: Pre-registered causal test of whether scFMs encode regulatory logic, or only co-expression

**One-line pitch:** Run a pre-registered, falsifiable test of whether SAE features in scGPT / Geneformer / scFoundation encode true *regulatory causality* (vs co-expression) by perturbing each feature in silico and predicting the effect of orthogonal CRISPRi Perturb-seq experiments — a deliberate replication and extension of the 2026 negative-result papers, designed to either solidify the negative finding or provide the first positive demonstration of causal scFM features.

**Hypothesis (biology):** Either (i) scFMs encode regulatory causality and feature-perturbation predicts gene-perturbation effects significantly above chance on held-out Perturb-seq (positive result), or (ii) they only encode co-expression and the directional accuracy stays near the 55% reported by Causal Circuit Tracing 2026 (negative result that consolidates the field's understanding).

**Why this is publishable either way:** "Interpretability without actionability" (2026) and "Systematic Evaluation of scFM Interpretability" (Kendiukhov 2026) both showed weak/negative results and were noticed. A *pre-registered* version, with cleaner methodology and three different scFMs, would be the definitive paper on this question.

**Concrete plan:**
- **Models:** scGPT, Geneformer V2, scFoundation. All three (the negative-result papers used only two).
- **SAE variant:** TopK (k=128), expansion 16x on residual stream of all layers.
- **Pre-registration:** Publish a pre-registration on OSF *before* any analysis — specify the metric (directional AUROC), the threshold for "positive" (>0.70 across all three models), and the Perturb-seq datasets used (Replogle K562 + RPE1, Norman combinatorial).
- **Experiment:**
  1. For each model: train SAE, auto-label features.
  2. For each feature in the top quartile by activation density: predict the gene-perturbation effect that *should* be observed if the feature were causal.
  3. Compare to held-out Perturb-seq.
  4. Report directional accuracy + cell-type-stratified breakdown (CSSI from Kendiukhov 2026).
  5. Cross-model consensus: does any feature behave causally in all three models?
- **Discovery either way:**
  - Positive: "We identify N features in scFMs that are causal regulators across models." → Cell Systems / Nature Methods.
  - Negative: "scFMs do not encode regulatory logic — they encode co-expression." Pre-registered, larger-scale confirmation of 2026 papers. → Nature Methods / Cell Patterns.

**Discovery validation strategy (no wet lab):**
- **Held-out Perturb-seq:** Replogle K562 + RPE1, Norman et al. combinatorial. None of these are in the standard scFM training data.
- **Cross-model consensus:** A feature is causal only if it behaves causally in all three scFMs.
- **Orthogonal oracle:** GO + STRING + TRRUST for the gene targets implicated by each feature.

**Compute budget estimate:**
- Three SAEs across three scFMs × all layers: ~250 GPU-hours.
- Perturbation simulation + causal tests: 2 H100-weeks.
- **Total: ~4 weeks on 4×H100.**

**Risk (failure mode):**
- This idea has the highest competition risk. The 2026 papers may already have made this point. Mitigation: be the first to do it with three models, pre-registered, and with proper cell-state stratification.
- If the result is strongly negative, framing matters — "current scFMs are not yet causally informative" is publishable; "scFMs are useless" is not.

**Target venue:** **Nature Methods** (with positive result), **Cell Patterns** or **Nature Computational Science** (with negative result framed as field-defining).

**Closest related work and what they didn't do:**
- *Causal Circuit Tracing* (arXiv 2026, Geneformer + scGPT): 56.4% directional accuracy. Not pre-registered.
- *Systematic Evaluation of scFM Interpretability* (Kendiukhov 2026): negative on attention; not specifically on SAE features.
- *Exhaustive Circuit Mapping of Geneformer* (Kendiukhov 2026): layer-specific results; no Perturb-seq held-out tied to feature predictions.

Pre-registration + three-model consensus is what's missing.

---

## Comparison table

| | Idea 1 PathoSAE | Idea 2 EnhancerSAE | Idea 3 CrossBioSAE | Idea 4 RNA-MotifSAE | Idea 5 AuditSAE |
|---|---|---|---|---|---|
| Discovery potential | High (clinical) | High (regulatory) | High (cross-modal) | Moderate-High (RNA primitives) | High either way |
| Feasibility | High | Moderate-High | Moderate | High | High |
| Compute | 2 weeks 4×H100 | 3–4 weeks 4×H100 | 3 weeks 4×H100 | 2 weeks 4×H100 | 4 weeks 4×H100 |
| Wet-lab risk | None | None | None | None | None |
| Competition risk | Low | Low | Low | Moderate | High (others active) |
| Target venue | Nature Methods | Nature Methods | Nature Comm / ICLR | Nature Comm | Nature Methods (+/−) |
| Code reuse | InterPLM + AlphaMissense | InterPLM + Enformer/Borzoi | USAE + Goodfire Evo SAE | SAE-RNA + Orthrus | Kendiukhov circuit codebase |
| 八股文 angle | Strong (3 contributions clear) | Strong | Strong | Moderate | Strong (negative-result framing) |

---

## Recommendation to Dr. Zhang

If only one idea is to be pursued, **Idea 1 (PathoSAE) is the safest high-impact choice**: it combines two well-trodden no-wet-lab pathways (AlphaMissense-style pathogenicity + InterPLM-style SAE), the validation is cleanly orthogonal (temporal ClinVar + AlphaFold3 confidence), competition risk is low, and Nature Methods has explicitly accepted both styles. Compute is modest.

**Idea 2 (EnhancerSAE) has the highest novelty** because the chromatin-FM-SAE space is entirely empty — but it depends on CLM-access being stable enough.

**Idea 5 (AuditSAE) is the highest-EV "defensive" choice**: pre-registration insulates against scoop risk, and the negative-result framing has just been validated by 2026 papers.

The choice depends on Dr. Zhang's risk tolerance:
- For maximum-impact / lowest-risk → **Idea 1**.
- For highest-novelty / open territory → **Idea 2**.
- For deepest ML+bio integration → **Idea 3**.
- For RNA focus → **Idea 4**.
- For pre-registered, scoop-proof, either-way-publishable → **Idea 5**.
