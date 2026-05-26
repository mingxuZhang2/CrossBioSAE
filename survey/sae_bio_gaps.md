# Research Gaps: SAE + Biological LLMs + Scientific Discovery (without wet lab)

> Companion to `sae_bio_landscape.md` and `no_wetlab_publishing_sae_addendum.md`.
> Updated: 2026-05-26
> Lens: gaps sharply defined for a computational researcher (no wet lab) who wants the SAE features themselves to function as scientific discovery — not a method paper.

---

## How this list differs from the previous `gaps.md` and `sae_bio_landscape.md` §8

`gaps.md` covers gaps in *LLM-for-bio* writ large (RNA FMs, multi-omics benchmarks, Mamba for single-cell, etc.). `sae_bio_landscape.md` §8 enumerates *missing-paper* gaps (no SAE for AlphaFold, no benchmark like SAEBench). This document is narrower and sharper: each gap below is something a no-wet-lab researcher can attack in 3–6 months and that could plausibly land at Nature Methods / Nature Communications / PNAS / ICML / NeurIPS, with **scientific discovery** as the central claim.

The gaps also incorporate eight 2025–2026 papers that the previous landscape did not cover:
- ProtSAE (AAAI 2025)
- MotifAE (bioRxiv 2025)
- Sparse Autoencoders for Low-$N$ Protein Function Prediction (Tsui et al., 2025)
- Mechanistic Interpretability of Antibody LMs Using SAEs (arXiv 2025)
- Discovering Interpretable Biological Concepts in scRNA-seq FMs (2025)
- Causal Circuit Tracing in scFMs (2026)
- Systematic Evaluation of scFM Interpretability (Kendiukhov, 2026)
- Discovery of a Hematopoietic Manifold in scGPT (2026)
- Interpretability without actionability (2026) — important critique
- Automated Neuron Labelling in PLMs (2025)

---

## Gap A. SAE features that **predict** novel biology have never been **temporally** validated

**Gap (one sentence):** Existing SAE+bio papers validate by retrieving features that match *already-known* annotations (UniProt, Pfam, GO, JASPAR); none has frozen the SAE before a release of new annotations and shown that an "unexplained" feature later matched the new release.

**Why it matters:** This is the cleanest way to claim discovery in a purely computational paper. AlphaMissense uses essentially this temporal-split trick. The SAE+bio field has not yet adopted it.

**Feasibility (no wet lab):** High. UniProt releases monthly with new annotations; Pfam, InterPro, MEME-suite databases get updates; ProteinGym adds new DMS assays. Train SAE on ESM-2 with a freeze-date, then test feature-annotation alignment against post-freeze releases.

**Most natural target venue:** Nature Methods (method + discovery framing), or PNAS.

**Closest existing work:** InterPLM (Nature Methods 2025) speculates about "filling missing annotations" but does not run the temporal experiment. Decode-gLM does an analogous audit but for *contamination* not discovery.

**Why this is publishable:** It directly tests the central interpretability claim (do SAE features represent biology?) against the strongest possible falsifier.

---

## Gap B. **No SAE on ESM-C / ESM-Cambrian** despite it being the current SOTA PLM

**Gap:** ESM-C (released Dec 2024) and ESM-Cambrian (2025) are the new SOTA protein language models that explicitly improve representation quality over ESM-2, yet every SAE+bio paper still uses ESM-2.

**Why it matters:** Better representations → more interpretable features → more discoverable biology. Reticular AI already showed SAE quality scales with PLM scale (ESM-2 3B beats 650M). Moving to ESM-C is the natural next step.

**Feasibility:** Very high. ESM-C checkpoints are public. The "Sparse Autoencoders for Low-$N$" recipe (Tsui et al. 2025) can be re-run on ESM-C with one config change. SAE training is cheap relative to PLM training.

**Most natural target venue:** ICLR / NeurIPS or Nature Methods (if discovery angle is strong).

**Closest existing work:** Adams et al. ICML 2025 (ESM-2), Gujral et al. PNAS 2025 (ESM-2), Tsui et al. 2025 (ESM-2 fine-tuned). All ESM-2.

**Watch:** This is a low-hanging fruit; expect at least one preprint in the next 2 months. Speed of execution matters.

---

## Gap C. SAEs have not been used to **bridge protein and DNA modalities**

**Gap:** No paper trains a single SAE that ingests activations from both a PLM (ESM-2 or ESM-C) and a DNA-LM (Evo-2 or Nucleotide Transformer) on matched protein-coding genes, and asks "do any features represent the *same* concept across modalities?"

**Why it matters:** This is the most direct test of whether different bio-FMs converge on the same latent biology. If they do, that is a strong claim about how biology is represented; if they don't, that is itself a discovery about why the field's "generalist" models (Evo-2, GenBioAI) are or aren't really unified.

**Feasibility:** Moderate. Universal Sparse Autoencoders (Cunningham et al. 2025) already exist for vision; the architecture extends naturally to bio. Need matched gene-protein pairs: NCBI Gene + UniProt provides millions. Compute: ~1 H100-week.

**Most natural target venue:** Nature Communications or ICLR.

**Closest existing work:** Universal SAEs (USAEs, 2025) for vision; Goodfire's Evo-2 SAE for DNA; multiple PLM SAEs. None combine the two.

**Why this is publishable as discovery:** Finding a "promoter-binding-site" feature in ESM-C that aligns with a "promoter" feature in Evo-2 would be a genuine cross-modal observation about how biology is encoded.

---

## Gap D. **No SAE has been applied to RNA-LMs at scale** beyond a single proof-of-concept

**Gap:** Only SAE-RNA (arXiv 2025) exists, applied to RiNALMo at small scale. No SAE on AIDO.RNA (1.6B params), Orthrus (Nature Methods 2026), GRAPE-LM, or RNA-FM-large.

**Why it matters:** RNA is rich in structured concepts (stems, loops, pseudoknots, IRES, riboswitches, m6A modification sites) that should map cleanly to SAE features. The interpretability payoff per dollar of compute is likely highest in RNA because there are well-defined structural primitives.

**Feasibility:** High. Orthrus and AIDO.RNA are open. SAE training on a 1.6B model is single-A100-day work.

**Most natural target venue:** Nature Communications (RNA Society audience), Nature Methods, or NeurIPS.

**Closest existing work:** SAE-RNA (RiNALMo only). Orthrus's release notes do not include interpretability analysis.

**Discovery angle:** Recover known RNA structural elements unsupervised. Then ask: do features fire on bpRNA secondary-structure elements that match the SAE-found cluster? Use external structure predictors (RNAfold, EternaFold, RhoFold+) as orthogonal oracle.

---

## Gap E. SAE features have never been linked to **disease/phenotype** through a database hold-out

**Gap:** No SAE+bio paper uses ClinVar / OpenTargets / GWAS catalog to ask: "do SAE feature activations at a residue/variant predict pathogenicity better than the raw PLM and better than chance, on variants the PLM has not seen?"

**Why it matters:** AlphaMissense rode pathogenicity prediction to a Science paper using exactly this strategy. Doing it with SAE features adds an *interpretability* angle: not just predicting pathogenicity, but identifying *which interpretable feature* the variant disrupts.

**Feasibility:** High. ProteinGym has 217 DMS assays; ClinVar has millions of variants. SAE-on-ESM-2 already exists; running a probe is a weekend project.

**Most natural target venue:** Nature Methods, Nature Genetics, or American Journal of Human Genetics.

**Closest existing work:** Brandes et al. (Nature Genetics 2023) for raw ESM-1b on ClinVar; no SAE-based analog. InterPLM and Gujral et al. probe for binding sites but not disease.

**Why this is "discovery":** An interpretable feature-attribution for missense pathogenicity is a clinically valuable artefact, not just a method paper.

---

## Gap F. SAEs on **single-cell ATAC-seq foundation models** are absent

**Gap:** scATAC-seq foundation models (CLM-access, bioRxiv 2025) and chromatin accessibility models (Enformer, Borzoi, EpiGePT) have no SAE analysis. All scFM-SAE work is on scRNA-seq (scGPT, Geneformer, scFoundation).

**Why it matters:** ATAC-seq is *closer to mechanism* than RNA-seq — it tells you which regulatory elements are accessible. SAE features on a chromatin FM should naturally map to *cis*-regulatory grammar (enhancers, silencers, TFBS combinations). This is where the interpretability payoff is biggest and the data is densest (ENCODE has thousands of cell types).

**Feasibility:** Moderate-High. CLM-access is the cleanest target (just released, 2.8M cells, 1M peaks). Could also use the published ENCODE atlas as an SAE-feature evaluation set.

**Most natural target venue:** Genome Biology, Nature Communications, Nature Methods.

**Closest existing work:** None on chromatin FMs. SAEs on scRNA-FMs exist; bridging is what's missing.

**Discovery angle:** Does the model learn cell-type-specific TF cofactor combinations as SAE features? If yes, recovering enhancer codes purely from chromatin-FM SAE features would be a Nature Genetics-style result.

---

## Gap G. **No causal SAE→phenotype intervention** that has been validated against held-out Perturb-seq

**Gap:** Kendiukhov 2026 (Geneformer circuits) and the Causal Circuit Tracing 2026 paper come close, but neither pre-registers an SAE feature → predicted phenotype → check on held-out Perturb-seq experiment in the strict sense. The 2026 negative-result papers in fact suggest current scFMs do not encode the regulatory logic needed for this.

**Why it matters:** This is the strongest possible interpretability claim — the feature has *causal* content, not just correlational. Even a partially-positive result here would be a major paper, and a clearly-negative result with a good methodology would also be publishable as a "honest limit of interpretability" paper (cf. "Interpretability without actionability" 2026, which got attention precisely as a negative result).

**Feasibility:** High. Replogle et al. K562 + RPE1 Perturb-seq is public (~2.5M cells, all genes). Pre-register predictions, then test.

**Most natural target venue:** Nature Methods, Cell Systems, or Nature Communications.

**Closest existing work:** "Causal Circuit Tracing" (2026, 56.4% directional accuracy on CRISPRi); Kendiukhov (2026, layer-dependent differentiation). Both are good — leaves room for a more refined causal protocol.

---

## Gap H. **No SAEBench analog for biology** — no shared evaluation suite

**Gap:** NLP has SAEBench (2025) for standardized SAE evaluation. Bio has no equivalent. Every paper invents its own GO/UniProt/STRING enrichment protocol, making cross-paper comparison impossible.

**Why it matters:** Without a benchmark, the field is full of incomparable claims. Reviewers increasingly request comparison to baselines (PCA, NMF, raw neurons), but there is no shared protocol. OmniGenBench (2025) is a model benchmark, not an SAE benchmark.

**Feasibility:** Very high. This is a curation + engineering effort plus a clear paper. Combine ProteinGym, ENCODE TFBS, GO, Reactome, STRING, BEND, ProteinShake.

**Most natural target venue:** NeurIPS Datasets & Benchmarks, Nature Methods (Resource), or Bioinformatics.

**Closest existing work:** SAEBench (NLP); CE-Bench (2025, contrastive SAE eval); OmniGenBench (GFM benchmark, not SAE).

**Discovery angle is weak here.** This is a high-impact infrastructure paper, not a discovery paper. Listed because reviewers in many other gaps will demand such a benchmark.

---

## Gap I. **SAE features have not been used as a primitive for de novo design** validated by orthogonal predictors

**Gap:** Steering papers (Garcia & Ansuini 2025 for zinc fingers; Reticular AI for ESMFold solvent accessibility; Decode-gLM for antibiotic resistance) demonstrate steering. But none uses SAE features to design *combinations* of biological properties and validates them via a *different* model than the one being interpreted.

**Why it matters:** If SAE features are biologically real and compositional, you should be able to combine them. "Generate a sequence with high feature_734 (= solvent-accessible) AND high feature_1812 (= alpha-helix) AND low feature_99 (= disorder)" — and have ESMFold (a different model than ESM-C) confirm the structure. This is closer to actual protein design than current single-feature steering.

**Feasibility:** Moderate. Need an SAE on a strong PLM, plus orthogonal structure predictors (ESMFold, AlphaFold3, RoseTTAFold). Compute: ~2 weeks on 8x A100.

**Most natural target venue:** Nature Communications, ICLR, NeurIPS.

**Closest existing work:** Reticular AI (single-feature ESMFold steering); MotifAE (single-feature stability); Tsui et al. 2025 (Low-$N$ steering on fitness). No compositional steering with orthogonal validation.

**Discovery angle:** If features compose, that is a statement about how biology is represented; if they don't, that is also informative.

---

## Gap J. **Cross-species transfer of SAE features** has been claimed but never measured at scale

**Gap:** Bio-FMs are trained on data spanning many species. Does a feature that fires on a human GPCR transmembrane region also fire on a yeast or bacterial homolog? No paper has measured this systematically.

**Why it matters:** Cross-species feature transfer is the cleanest evidence that the SAE has captured an evolutionarily conserved concept rather than a species-specific quirk. It also gates whether bio-LLM interpretability generalises beyond human biology.

**Feasibility:** Very high. UniRef and OrthoDB provide orthologue groups. PLM features can be probed in a held-out species.

**Most natural target venue:** Genome Biology, Bioinformatics, PNAS.

**Closest existing work:** Some discussion in Adams et al. 2025 (family-specific vs generic features). No paper does cross-species feature-conservation at scale.

---

## Gap K. **No SAE has been trained on a multimodal bio-FM** despite their proliferation

**Gap:** Multimodal bio-FMs are appearing (BioVERSE, ProtT3, Prot2Chat, GenBioAI 2026, Generalised biological FM 2025). None has been analysed with SAEs.

**Why it matters:** Multimodal FMs may store cross-modal concepts (e.g., "this protein binds this drug class," combining protein + chemistry). SAEs are the right tool to surface these.

**Feasibility:** Moderate. Requires picking a multimodal FM with released weights and matched data. GenBioAI may not have open weights yet, but ProtT3 and Prot2Chat do.

**Most natural target venue:** Nature Communications, NeurIPS, ICML.

**Closest existing work:** SAE on chemistry LM (SMI-TED, 2024) and SAE on PLMs (multiple). Nothing on joint models.

---

## Gap L. **No SAE-based audit** has been done on biological FMs for training-data leakage / contamination

**Gap:** Decode-gLM showed SAEs can detect contamination (CMV enhancer in Nucleotide Transformer training data). No analogous audit exists for ESM-2 (Did UniRef contain known DMS-validated variants? Did training leak ProteinGym sequences?), Evo-2 (Did training leak antibiotic-resistance mutations?), or scGPT (Did training leak Perturb-seq labels in any disguise?).

**Why it matters:** This is one of the few cases where the *act of interpretation* itself is the discovery. Reviewers love it. It also addresses an active concern in the bio-FM community about benchmark contamination.

**Feasibility:** High. The general recipe is in Decode-gLM. Re-applying to other models is mostly engineering.

**Most natural target venue:** Nature Methods (Brief Communications), Nature Biotechnology Comment, or a top-tier ML conference workshop.

**Closest existing work:** Decode-gLM only.

---

## Priority ranking for Dr. Zhang

Sorted by (Discovery potential × Feasibility / Competition risk):

| Rank | Gap | Discovery | Feasibility | Competition risk | Combined |
|---|---|---|---|---|---|
| 1 | E. SAE→pathogenicity via held-out ClinVar | High | High | Low (no one has tried) | **★★★★★** |
| 2 | F. SAE on chromatin (CLM-access or Enformer) | High | High | Low (CLM-access just released) | **★★★★★** |
| 3 | A. Temporal validation of SAE features | High | High | Low | **★★★★** |
| 4 | I. Compositional steering with orthogonal validator | High | Moderate | Moderate (Reticular AI is close) | **★★★★** |
| 5 | C. Cross-modal SAE (PLM ⊗ DNA-LM) | High | Moderate | Low | **★★★★** |
| 6 | D. SAE on AIDO.RNA / Orthrus | Moderate-High | High | Moderate | **★★★** |
| 7 | B. SAE on ESM-C | Moderate | Very high | High (someone will do this soon) | **★★★** |
| 8 | G. Causal SAE→Perturb-seq | High | High | High (2026 papers already in this space) | **★★★** |
| 9 | J. Cross-species SAE feature conservation | Moderate | Very high | Low | **★★★** |
| 10 | K. SAE on multimodal bio-FM | High | Moderate | Low | **★★★** |
| 11 | L. SAE audit for contamination | Moderate | High | Low | **★★** |
| 12 | H. SAEBench-for-biology | Low (infra, not discovery) | Very high | Moderate | **★★** |

The top-5 gaps inform the ideas in `sae_bio_ideas.md`.
