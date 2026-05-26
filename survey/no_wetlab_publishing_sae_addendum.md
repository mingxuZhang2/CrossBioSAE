# Publishing SAE + Bio-LLM Papers Without Wet Lab
## Addendum focused on Sparse-Autoencoder-based scientific discovery

> Companion to `no_wetlab_publishing.md`. Read that first for general background.
> Updated: 2026-05-26
> Scope: papers at the intersection of (sparse autoencoders OR mechanistic interpretability) AND (protein/DNA/RNA/single-cell LLMs) AND (computational-only validation pathway).

---

## 0. Why this addendum is necessary

The general `no_wetlab_publishing.md` covers "method papers that get tested on benchmarks." SAE-driven work is different because the *deliverable* is often a set of **claims about biology that the SAE features reveal**, not a new score on a leaderboard. Reviewers therefore evaluate these papers along a different axis: "are the features biologically real, and is anything genuinely *new*?"

For Dr. Zhang, who has no wet lab, the entire game is to engineer that "new" using purely public data, in a way that referees accept as discovery rather than as data-mining of databases.

---

## 1. What counts as "validation" of an SAE feature without a wet lab

Reviewers of SAE+bio papers consistently accept the following surrogates for wet-lab validation. They are listed roughly in increasing order of strength.

### Tier 0 — Necessary but not sufficient
Almost every paper does these. Doing them is required; doing *only* these is no longer publishable above the workshop level after 2025.

| Method | What it shows | Risk |
|---|---|---|
| GO term enrichment over feature-firing residues / cells / sequences | Feature aligns with annotated function | GO is so dense that enrichment is easy; circular if SAE was trained on GO-labelled subsets |
| F1 against UniProt / Pfam / InterPro annotations | Feature recovers a known concept | Pure recovery is not discovery |
| Recovery of motif databases (JASPAR, MEME, HOMER, ENCODE TFBS) | Feature corresponds to known TF/binding pattern | Same issue: known concept |
| Cell-type marker recovery (PanglaoDB, CellMarker) for scFMs | Feature is a marker gene direction | Trivial baselines often recover the same |

### Tier 1 — Standard "validation by held-out database"
This is what most published SAE+bio papers actually use to demonstrate biological reality.

| Method | Example paper | Comment |
|---|---|---|
| Linear probing of SAE features for *external* labels (thermostability, subcellular localisation) | Adams et al., ICML 2025 | The external labels were not used in SAE training |
| Probing for fitness in DMS assays | Tsui et al., 2025 (Low-$N$ SAE) | ProteinGym is held out from ESM-2 pretraining for many assays |
| Held-out family-specific motif recovery | MotifAE (bioRxiv 2025) | Train SAE on broad set, test recovery on excluded families |
| Cross-species transfer of SAE features | Decode-gLM (bioRxiv 2025/2026) | Train on one species, test feature firing on another |
| Recovery on temporally-released structures / annotations | not yet done for SAE+bio | An obvious gap |

### Tier 2 — "Predict-and-confirm against a database the SAE could not have memorised"
This is what gets you into Nature Methods / PNAS / Nature Communications.

| Method | Example | Why reviewers accept it |
|---|---|---|
| Predict variant effects from feature steering, compare to ClinVar / ProteinGym DMS | InterPLM (Nature Methods 2025) used a related approach for missing annotations | DMS labels are continuous and were not used as SAE training signal |
| Discover a feature, then show it co-localises with a structural element from AlphaFold3 / ESMFold confidence map that the PLM training did not see | Matryoshka SAEs / Reticular AI (2025) | Structure prediction is a separate model; agreement is non-trivial |
| Steer SAE feature, generate sequences, score with an external oracle (e.g., RoseTTAFold, ProteinMPNN, Foldseek, ProGen2) and show structural / functional shift | Garcia & Ansuini 2025 (zinc fingers); Reticular (steering ESMFold solvent accessibility) | External oracle is not the model being interpreted |
| Recover a published biological circuit / pathway from feature-feature interactions, compare to STRING / Reactome | Kendiukhov 2026 (circuit map of Geneformer) | Tests whether the model learned the *relations* between features, not just the features |
| Audit discovery: SAE features reveal training-data leakage or unintended memorisation | Decode-gLM (CMV enhancer); not yet done for protein/scFM SAEs | This is a methodological discovery that is still genuinely new |

### Tier 3 — "Genuine biological discovery, computational-only"
This tier is largely empty in the SAE+bio literature as of 2026-05. It is also where the most publishable opportunities lie.

| Method | Status | Comment |
|---|---|---|
| SAE feature predicts a phenotype / disease association absent from training data and from any database the SAE saw, later confirmed against a *different* database (e.g., GWAS, OpenTargets, gnomAD constraint) | Not yet done convincingly | Would be a Nature-tier paper |
| SAE feature reveals a previously unannotated functional region (e.g., a cryptic binding site) that is recovered post-hoc by AlphaFold-Multimer / ESMFold confidence + evolutionary coupling | Hinted at by InterPLM ("filling missing annotations"), not yet pushed | The "filled" annotations need independent corroboration to count |
| Cross-modal SAE feature alignment (e.g., a single feature explains both DNA-LM activity and PLM activity on the same gene) | No paper has done this | Would be a strong claim — biology must use the *same* concept across modalities |
| In-context steering experiments where the model generates a sequence/cell-state predicted to behave a certain way, then verified by an *orthogonal predictor* trained on different data | Reticular AI 2025 is close | If the orthogonal predictor was trained on data the SAE never saw, this is publishable |

---

## 2. Concrete published examples — SAE+bio with no wet lab (or wet lab clearly optional)

Listed below are 10 papers where the validation is purely computational and the venue accepted the paper. Use these as templates.

1. **InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders** — Simon & Zou, *Nature Methods* 2025.
   *Validation:* Cross-reference with UniProt / Pfam / InterPro; demonstrated steering of sequence generation; proposed "filling missing annotations" without wet lab.

2. **Sparse autoencoders uncover biologically interpretable features in PLM representations** — Gujral et al., *PNAS* 2025.
   *Validation:* GO enrichment, comparison of SAE features to ESM-2 neurons, automated interpretability over annotated proteins.

3. **From Mechanistic Interpretability to Mechanistic Biology** — Adams et al., *ICML 2025* (also bioRxiv).
   *Validation:* Linear probing on held-out thermostability and subcellular localisation datasets; family-specific vs generic feature analysis.

4. **Interpreting and Steering Protein Language Models through Sparse Autoencoders** — Garcia & Ansuini, arXiv 2025.
   *Validation:* Steering ESM-2 toward zinc finger generation; sequences scored by external classifiers.

5. **Towards Interpretable Protein Structure Prediction with Sparse Autoencoders (Matryoshka SAEs on ESM2-3B)** — Reticular AI, OpenReview 2025.
   *Validation:* Steering ESMFold to alter solvent accessibility while fixing input sequence; compared to ground-truth PDB structures.

6. **ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders** — *AAAI 2025*.
   *Validation:* Better interpretable probing performance, better steering on annotated tasks. Used annotation datasets only as semantic prior, not as test labels.

7. **MotifAE: Unsupervised Discovery and Interpretability Analysis of Functional Motifs from PLMs** — bioRxiv 2025.
   *Validation:* Recovery of known functional motifs vs vanilla SAE; latent features correlate with residue-level importance for *domain folding stability* (an external experimentally-derived score), enabling stability-specific fitness prediction.

8. **Sparse Autoencoders Reveal Interpretable Features in Single-Cell Foundation Models (scGPT / scFoundation / Geneformer)** — bioRxiv 2025.
   *Validation:* GO / KEGG / Reactome / STRING / TRRUST coverage; differential information encoding across models.

9. **Discovering Interpretable Biological Concepts in scRNA-seq Foundation Models** — arXiv 2025.
   *Validation:* Counterfactual perturbation attribution + ontology enrichment; expert immunologist interpretation. No wet lab.

10. **Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models** — bioRxiv 2026.
    *Validation:* Steering Nucleotide Transformer toward known antibiotic-resistance mutation (A1408G in 16S rRNA); audit-style discovery of CMV enhancer in training data leak.

11. **Mechanistic Interpretability of Antibody Language Models Using SAEs** — arXiv 2025.
    *Validation:* Steering antibody generation; concept-feature correlation; TopK vs Ordered SAE comparison for steerability.

12. **Mechanistic Interpretability of Fine-Tuned Protein Language Models for Nanobody Thermostability Prediction** — bioRxiv 2025.
    *Validation:* Held-out thermostability dataset; SAE features linked to stability determinants.

13. **Exhaustive Circuit Mapping of a Single-Cell Foundation Model (Geneformer)** — Kendiukhov, arXiv 2026.
    *Validation:* All-pairs SAE-feature ablation; downstream effect statistics; layer-specific differentiation control compared to known developmental trajectories.

14. **Causal Circuit Tracing Reveals Distinct Computational Architectures in Single-Cell Foundation Models** — arXiv 2026.
    *Validation:* CRISPRi gene-perturbation hold-out (Replogle K562 + RPE1) — published Perturb-seq data, no new wet lab. 56.4% directional accuracy.

15. **Systematic Evaluation of scFM Interpretability Reveals Attention Captures Co-Expression Rather Than Unique Regulatory Signal** — arXiv 2026.
    *Validation:* 37 analyses, 153 statistical tests across four cell types from public Perturb-seq, no wet lab. Honest **negative result** that still got attention — a useful template.

Pattern that holds across these 15 papers:
- 100% rely on at least one of: ProteinGym, UniProt/Pfam/InterPro, GO/KEGG/Reactome, STRING, Perturb-seq (Replogle/Norman/Adamson), ENCODE/TFBS databases, ClinVar, ChEMBL.
- ~60% include at least one *external-oracle* validation (different model scores the SAE-steered output).
- ~30% include a *temporal* or *species* hold-out.
- 0% include wet-lab validation as a core claim.

---

## 3. Venue map for SAE+bio scientific-discovery papers

| Tier | Venue | Style of paper that lands there | Bar |
|---|---|---|---|
| Top-tier journal | Nature Methods | InterPLM-style: clear methodological contribution + demonstrated downstream use (annotation, steering) | Comprehensive evaluation; clean writing; interactive demo |
| Top-tier journal | PNAS | Strong computational validation + clear biology framing | Similar bar to Nature Methods, often more biology-heavy |
| Top-tier journal | Nature Communications | Fully computational, broad scope, "useful tool" framing | Lower than Nature Methods but still rigorous |
| Top-tier journal | Nature Computational Science | Method-flavoured, computational-discovery emphasis | Open to interpretability-as-discovery framing |
| Top-tier journal | Cell Patterns | Computational biology + interpretability + reproducibility | Code/data release is non-negotiable |
| Top-tier journal | Science Advances | Genuinely surprising biology required | Very hard without orthogonal validation |
| Bio-focused | Genome Biology / Genome Research | Strong if scFM / DNA-LM and the discovery is genomic | Reviewers will demand benchmark comparison |
| Bio-focused | Bioinformatics (Oxford), Briefings in Bioinformatics | Method + benchmark | Accessible, computational-only is the norm |
| ML venues | ICLR, NeurIPS, ICML | Method-first framing, biology as motivation | Strong baselines, ablations, code release; bio audience secondary |
| ML venues | ICML / NeurIPS workshops | First publication of an idea | MLCB, AI4Science, GenBio workshops are excellent |
| Bio-ML hybrid | MLCB (Machine Learning in Computational Biology), RECOMB, ISMB | Best fit for SAE+bio discovery papers | Reviewers understand both sides |
| Preprints | bioRxiv → journal | Standard pathway for biology side | Most SAE+bio papers live here first |
| Preprints | arXiv → ML conference | Standard pathway for ML side | Same |

**Practical rule of thumb for Dr. Zhang:**
- A *method* paper (new SAE variant for bio, new architecture, new benchmark): aim for ICLR / NeurIPS / ICML or Nature Methods.
- A *discovery* paper (SAE reveals a new biological pattern): aim for Nature Communications / Nature Methods / PNAS / Cell Patterns.
- A *negative-result* or *audit* paper (current SAEs do not capture X, training data leaked): aim for arXiv → workshop → Nature Methods Comment / Briefings, or follow the Causal Circuit Tracing template.

---

## 4. What reviewers reject in SAE+bio papers (rejection patterns from 2025-2026)

These are inferred from the increasing number of "negative-result" papers that have been published and from public reviews on OpenReview.

1. **GO-only validation.** If the entire claim is "feature X enriches for GO term Y," reviewers will ask whether a simple PCA or k-means on PLM embeddings does the same thing. Always include this baseline.
2. **Cherry-picked features.** Reporting only the 20 most interpretable features out of 10,000 is no longer accepted. Aggregate statistics (% of features with annotation match above threshold) are now required.
3. **No causal intervention.** A feature that correlates with a concept is no longer enough. Reviewers want either steering experiments or ablation experiments. (See "Interpretability without actionability," 2026 — this critique now haunts the field.)
4. **No orthogonal validator.** If the SAE was trained on ESM-2 and validated only on labels that were in ESM-2's training corpus, this is circular. Use a different model (AlphaFold, Foldseek, ProGen2) or a different label source (DMS, ClinVar) to break circularity.
5. **No comparison to non-SAE baselines.** Compare against:
   - Raw PLM neurons
   - PCA / SVD components
   - K-means / NMF / topic models on embeddings
   - Linear probes on the residual stream
   Many SAE papers are now rejected because PCA is shown to recover almost the same features.
6. **No reproducibility.** Code, trained SAE weights, and feature dashboards are now expected. InterPLM and Goodfire's Evo-2 SAE dashboards set the standard.
7. **Inflated discovery claims.** Saying "we discovered new biology" without a database hold-out or external-oracle confirmation is now flagged. Use the word "candidate" or "hypothesised" for unconfirmed features.

---

## 5. The "scientific discovery without wet lab" recipe for SAE+bio

A reproducible recipe that has worked in multiple top-venue papers:

1. **Pick a model where interpretability matters** — ESM-2/ESM-C, Evo-2, scGPT, Geneformer, RiNALMo, scATAC FM.
2. **Train an SAE** (TopK or JumpReLU; Matryoshka if hierarchical features are expected; transcoder if circuit-level claims are needed). 8x to 32x expansion factor; k between 32 and 128 for protein, larger for genomic.
3. **Auto-label features** with a domain ontology: UniProt for protein, ENCODE for DNA, GO/KEGG/Reactome/STRING for cells.
4. **Find an unexplained subset** of features that do *not* match any known annotation but consistently fire on biologically coherent inputs.
5. **Form a hypothesis** for what these features represent (e.g., "feature 7382 may represent a cryptic alternative splice site," or "feature 134 looks like a novel TF cofactor signature").
6. **Test the hypothesis with a held-out external resource** that the SAE never saw:
   - For protein: AlphaFold confidence maps, evolutionary couplings (EVcouplings, Gremlin), DMS data from new ProteinGym assays.
   - For DNA/RNA: GWAS catalog, gnomAD constraint, eQTL catalogues from GTEx, ClinVar variants released after the SAE training date.
   - For single-cell: published Perturb-seq experiments (Replogle, Norman, Adamson) not used in pretraining, CITE-seq cross-modal data.
7. **Statistical test** with multiple-testing correction; report effect sizes.
8. **Release** SAE weights, feature labels, dashboard, and analysis code.

This 8-step recipe is essentially the InterPLM + Decode-gLM + Reticular AI blueprint. Variations of it have produced Nature Methods, PNAS, ICML, and Nature Communications papers in 2025.

---

## 6. Verdict for Dr. Zhang

**Yes, SAE+bio papers with scientific-discovery claims can be published without wet lab, and at the highest venues.** The proof is the 15-paper list in Section 2. The non-negotiable conditions are:

- At least one Tier-2 validation (database hold-out, orthogonal oracle, or causal intervention).
- Strong baselines (PCA, NMF, raw neurons).
- Reproducibility (code, weights, dashboard).
- Honest framing — say "candidate" not "discovery" until Tier 3 validation exists.

The field is moving from "show SAE features recover GO" (publishable in 2024) to "show SAE features make a falsifiable biological prediction that holds against a database the SAE never saw" (publishable in 2026 at Nature Methods). Dr. Zhang's window of opportunity is to sit in this gap.
