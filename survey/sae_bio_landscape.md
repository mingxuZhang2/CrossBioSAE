# Sparse Autoencoders (SAEs) x Biological Language Models: Landscape Analysis

**Date**: 2026-05-25
**Author**: Research Pipeline (for Dr. Zhang)

---

## Executive Summary

The intersection of sparse autoencoders and biological language models is a **rapidly emerging field** that has seen explosive growth since late 2024. Contrary to initial expectations that this would be a sparsely populated niche, we found **approximately 20+ dedicated papers** applying SAEs to bio-LLMs, with the earliest appearing in late 2024 and the majority published in 2025-2026. The field has already produced publications in Nature Methods, PNAS, and ICML. However, significant gaps remain, particularly in (1) cross-modal/multi-omics SAE analysis, (2) SAE-guided biological discovery beyond annotation recovery, (3) single-cell model circuit analysis at scale, and (4) theoretical frameworks for biological SAE evaluation.

---

## 1. SAE Applied to Protein Language Models

This is the **most mature** sub-area, with multiple concurrent papers and high-profile publications.

### 1.1 Core Papers

**InterPLM** (Simon & Zou, 2024/2025)
- Published in **Nature Methods** (2025)
- Trained SAEs on ESM-2 embeddings (8M and 650M parameter models)
- Identified up to 2,548 human-interpretable latent features per layer
- Features correlate with 143 known biological concepts: binding sites, structural motifs, functional domains
- Key finding: PLMs store concepts in **superposition** (individual neurons show poor conceptual alignment)
- Superposition persists across model scales; larger PLMs capture more interpretable concepts
- Practical applications: filling missing database annotations, targeted steering of sequence generation
- Released interactive dashboard at interPLM.ai and code at github.com/ElanaPearl/InterPLM
- **95 citations** (as of search date)

**From Mechanistic Interpretability to Mechanistic Biology** (Adams, Bai, Lee et al., 2025)
- Published at **ICML 2025** (poster), also on bioRxiv
- From Columbia University and Ginkgo Bioworks
- Trained SAEs on the residual stream of ESM-2
- Key finding: pLMs use a combination of **generic features** and **family-specific features**
- Demonstrated linear probing of SAE features to identify sequence determinants of thermostability and subcellular localization
- For features without known functional associations, hypothesized their role in unknown mechanisms
- Released InterProt visualizer tool
- **37 citations**

**Sparse Autoencoders Uncover Biologically Interpretable Features in PLM Representations** (Gujral, Bafna, Alm et al., 2025)
- Published in **PNAS** (2025)
- Leveraged both SAEs and **transcoders** (a related technique) on protein-level and amino acid-level representations
- Gene Ontology Analysis and automated interpretability protocols
- Sparse features are more interpretable than ESM-2 neurons
- **39 citations**

**Interpreting and Steering Protein Language Models through Sparse Autoencoders** (Garcia & Ansuini, 2025)
- arXiv, February 2025
- Applied SAEs to ESM-2 8M parameter model
- Statistical analysis linking latent components to protein annotations: transmembrane regions, binding sites, motifs
- Demonstrated **steering** of sequence generation toward zinc finger domains
- **16 citations**

### 1.2 Structure Prediction Interpretability

**Towards Interpretable Protein Structure Prediction with Sparse Autoencoders** (Reticular AI, 2025)
- Scaled SAE training to **ESM2-3B** (the base model for ESMFold) -- first to do this at this scale
- Introduced **Matryoshka SAEs** for protein models: hierarchical features via nested groups of latents
- Hierarchical organization aligns with multi-scale nature of protein structure
- SAEs trained on ESM2-3B significantly outperform those on smaller models for concept discovery and contact map prediction
- Demonstrated **targeted steering of ESMFold predictions** (e.g., increasing solvent accessibility while fixing input sequence)
- Published on OpenReview (appears to be under review at a top venue)

### 1.3 Related Protein Interpretability (Non-SAE)

**Knowledge Neurons in Protein Language Models** (2023)
- Identified "knowledge neurons" in ESM via activation-based and integrated gradient selection
- Found high density of knowledge neurons in key vector prediction networks of self-attention modules
- Predates SAE-based approaches; less scalable

**Paying Attention to Attention** (2024/2025)
- Published in PLOS Computational Biology
- Identified "High Attention (HA)" sites in ESM that drive family classification
- HA sites overlap with biologically important residues (active sites)

**Protein Language Model Fitness Is a Matter of Preference** (Gordon, Lu, Abbeel, 2024)
- Used influence functions to understand how training data affects protein likelihoods
- Found power-law tail due to sequence homology

---

## 2. SAE Applied to DNA/Genomic Language Models

### 2.1 Evo 2 (Arc Institute + Goodfire)

**Genome Modeling and Design Across All Domains of Life with Evo 2** (2025)
- Evo 2: 40B parameter genomic foundation model, 1 megabase context, trained on 9T+ nucleotides
- Arc Institute collaborated with **Goodfire** to train SAEs on Evo 2 representations
- Discovered features corresponding to:
  - Canonical gene structures (CDS, UTRs, exons)
  - Intron/exon boundaries
  - Transcription factor motifs
  - Structural motifs (alpha-helices, RNA stem-loops)
  - Open reading frames (ORFs), intergenic regions, tRNAs, rRNAs in E. coli genome
- Features span from nucleotide patterns to protein structure
- Released interactive mechanistic interpretability visualizer
- Applications: genome annotation, biological discovery, steering of sequence generation

### 2.2 Decode-gLM (Nucleotide Transformer)

**Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models** (2025, bioRxiv)
- Trained SAEs on **Nucleotide Transformer** activations
- Three capabilities demonstrated:
  1. **Interpret**: SAEs identified 100+ diverse functional annotations (F1 > 20% for many)
  2. **Audit**: Discovered training data contamination (CMV enhancer encoded despite viral genomes being excluded)
  3. **Steer**: Used antibiotic-resistance SAE feature to steer toward A1408G aminoglycoside-resistance mutation in 16S rRNA
- Important methodological contribution: SAE as audit tool for training data leakage

### 2.3 HyenaDNA

**Sparse Autoencoders Reveal Interpretable Structure in Small Gene Language Models** (Guan, He, Zhang, 2025)
- Applied SAEs to embeddings from **HyenaDNA-small-32k** (compact, single-nucleotide resolution, pretrained on human reference genome)
- Identified sparse features corresponding to:
  - Individual nucleotides
  - Transcription factor binding sites (TFBS)
- Demonstrates SAEs work even on small genomic models

### 2.4 DNABERT (Attention-Based Interpretability)

**Evaluating DNA Function Understanding in Genomic Language Models** (2025+)
- After fine-tuning on tasks, DNABERT attention concentrates on:
  - Transcription factor binding sites
  - TATA boxes
  - Intronic regions flanking splice sites
- Note: This uses attention analysis, not SAEs specifically
- Limitation noted: all genomic LMs show sharp accuracy drop as sequence diverges from evolutionary prior (pattern-matching vs. mechanistic understanding)

---

## 3. SAE Applied to Single-Cell Foundation Models

### 3.1 Core SAE Papers

**Sparse Autoencoders Reveal Interpretable Features in Single-Cell Foundation Models** (2025, bioRxiv)
- Trained SAEs on hidden representations of **scGPT, scFoundation, and Geneformer**
- Learned features reveal diverse and complex biological and technical signals
- Information encoding differs between models with distinct training protocols and architectures
- Released on bioRxiv (October 2025)

**Sparse Autoencoders Reveal Organized Biological Knowledge but Minimal Regulatory Logic in Single-Cell Foundation Models** (2026, arXiv)
- Trained TopK SAEs on residual stream activations from **all layers** of:
  - Geneformer V2-316M: 82,525 features
  - scGPT: 24,527 features
- **Key finding**: Massive superposition confirmed -- 99.8% of features invisible to SVD
- 29-59% of features annotate to Gene Ontology, KEGG, Reactome, STRING, or TRRUST
- **Critical limitation**: Rich biological *knowledge* organization but **minimal regulatory logic**
- SAE features are functionally related to model behavior and can be intervened upon

**Can Sparse Autoencoders Make Sense of Gene Expression Latent Variable Models?** (Schuster, 2024)
- Explored SAE decomposition of embeddings from pretrained single-cell models
- SAEs can find and steer key biological processes
- Introduced **scFeatureLens**: automated interpretability linking SAE features to biological concepts from gene sets
- Important methodological contribution for automated evaluation

### 3.2 Circuit Analysis

**Transcoder-based Circuit Analysis for Interpretable Single-Cell Foundation Models** (2025, arXiv)
- Trained a **transcoder** (not SAE, but closely related) on **cell2sentence (C2S)** model
- Extracted internal decision-making circuits
- Circuits correspond to real-world biological mechanisms
- Uses wide, sparsely activated intermediate layers replacing MLP layers

**Exhaustive Circuit Mapping of a Single-Cell Foundation Model** (Kendiukhov, 2026, arXiv)
- Exhaustive tracing of all 4,065 active SAE features at layer 5 of Geneformer
- Generated 1,393,850 significant downstream edges (27x expansion over selective sampling)
- Heavy-tailed hub distribution: 1.8% of features account for disproportionate connectivity
- 40% of top-20 hubs **lack biological annotation** (novel features?)
- Redundancy: fundamentally subadditive at all tested orders (zero synergy)
- **Layer-dependent differentiation control**: late-layer features (L17) push toward maturity; early/mid-layer features (L0, L11) push away from maturity
- Important for understanding how scFMs internally represent developmental processes

---

## 4. SAE Applied to RNA Language Models

**SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA Language Model Representations** (2025, arXiv)
- Applied SAEs to **RiNALMo** representations
- Maps hidden states to biological features of mRNA and ncRNA families
- Frames RNA interpretability as concept discovery in pretrained embeddings
- Provides tools to probe what RNA LMs encode about ncRNA families
- Extends the SAE interpretability paradigm to the RNA domain

---

## 5. SAE Applied to Scientific Models Beyond Biology

### 5.1 Chemistry

**Unveiling Latent Knowledge in Chemistry Language Models through Sparse Autoencoders** (2024, arXiv/OpenReview)
- Applied SAEs to **SMI-TED** chemistry foundation model
- Extracted features corresponding to:
  - Chemical substructures / structural motifs
  - Physicochemical properties
  - Pharmacological drug classes
- Models encode rich landscape of chemical concepts
- Motivation: distinguish models that learn physical laws vs. those that memorize statistical correlations

### 5.2 Neural Operators / PDEs

**Mechanistic Interpretability with Sparse Autoencoder Neural Operators** (2026, arXiv)
- SAE-NOs: operate directly in infinite-dimensional function spaces
- Generalizes linear representation hypothesis to **functional representation hypothesis**
- SAE Fourier neural operators (SAE-FNOs): concepts as integral operators in Fourier domain
- Improved stability, robustness to distribution shifts, generalization across discretizations
- Applied to PDE solving models

### 5.3 Vision Models

**Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models** (2025)
- SAEs on frozen ViT activations; supports patch-level causal edits
- saev PyTorch package released
- Features: object boundaries, textures, semantic concepts

### 5.4 Equivariant SAEs

**Group Equivariance Meets Mechanistic Interpretability** (Erdogan & Lucic, 2025)
- Incorporates group symmetries into SAEs
- Adaptively equivariant SAEs discover features with superior probing performance
- Directly relevant to scientific models with inherent symmetries (e.g., molecular symmetries in 3D protein/molecule representations)

---

## 6. SAE in NLP/General LLMs (Analogies for Bio Applications)

### 6.1 Foundational SAE Work

**Sparse Autoencoders Find Highly Interpretable Features in Language Models** (Cunningham et al., 2023)
- Seminal paper demonstrating SAEs for LLM interpretability
- Core idea: decompose polysemantic neurons into monosemantic features

**Scaling Monosemanticity** (Anthropic, 2024)
- Trained SAEs on Claude 3 Sonnet
- Identified millions of interpretable features (Golden Gate Bridge, code, deception, etc.)
- Demonstrated feature steering

**Gemma Scope, Llama Scope** (Google, Meta, 2024-2025)
- Pre-trained SAE suites released as community resources

### 6.2 SAE Variants and Improvements

- **TopK SAEs**: Fixed sparsity via top-k activation selection
- **JumpReLU SAEs**: Improved training stability
- **Matryoshka SAEs**: Hierarchical feature organization
- **Transcoders**: Replace MLP layers directly; find input-output circuits
- **Meta-SAEs**: SAEs on SAE decoder directions; find higher-order structure
- **Switch SAEs** (Efficient Dictionary Learning): MoE-style routing for efficiency
- **End-to-End SAEs** (Functionally Important Features): Optimize for downstream task performance

### 6.3 Domain-Specific SAEs

**Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific SAEs** (O'Neill et al., 2025)
- Training SAEs on domain-confined data (medical text) improves interpretability
- Domain SAEs explain up to 20% more variance than broad-domain SAEs
- Features align with clinically meaningful concepts
- **Key insight for bio-LLMs**: domain-specific SAE training may be essential

### 6.4 SAE Limitations and Critiques

**Sparse Autoencoders Do Not Find Canonical Features** (ICLR 2025)
- Challenges whether SAE features represent ground truth

**Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts** (2025)
- Argues SAEs are best for discovery, not for known-concept extraction

### 6.5 Survey Paper

**A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs** (EMNLP Findings 2025)
- Comprehensive survey covering architecture, training, evaluation, applications

---

## 7. Comprehensive Review Paper

**What Do Biological Foundation Models Compute? SAEs from Feature Recovery to Mechanistic Interpretability** (2026, bioRxiv)
- **The first comprehensive review** of SAE applications across biological foundation models
- Covers protein, genomic, and single-cell models
- Key finding: independent studies using different architectures consistently recover features spanning biological scales
- Distinguishes three levels:
  1. **Representational interpretability**: which directions encode which concepts
  2. **Computational interpretability**: how features are computed
  3. **Causal mechanistic understanding**: why features matter
- Most current work is at level 1; levels 2-3 are largely unexplored

---

## 8. Gap Analysis: What Does NOT Exist

### Gap 1: Cross-Modal / Multi-Omics SAE Analysis
- **No paper** applies SAEs to multi-modal biological models that integrate protein + DNA + expression data
- Models like the "Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language" (He et al., 2025) exist but have NOT been analyzed with SAEs
- Question: Do SAEs on multi-modal bio-models learn features that bridge molecular levels?

### Gap 2: SAE-Guided Genuine Biological Discovery
- Current papers primarily **recover known annotations** (GO terms, binding sites, etc.)
- The "discovering new biology" angle is aspirational but not yet delivered at scale
- Gap: No paper has used SAE features to **predict and experimentally validate** a previously unknown biological mechanism
- Exception: Decode-gLM's audit discovery of training data contamination is a form of discovery, but not biological per se

### Gap 3: SAE on 3D Structure-Aware Models
- Matryoshka SAEs on ESM2-3B/ESMFold is closest
- **No SAE analysis on AlphaFold2/3** or other dedicated structure prediction models
- No SAE on 3D-aware protein models (e.g., GearNet, ScanNet, ProteinMPNN)
- Equivariant SAEs (Erdogan & Lucic, 2025) are relevant but not yet applied to bio

### Gap 4: Theoretical Framework for Biological SAE Evaluation
- NLP SAEs have SAEBench (2025) for standardized evaluation
- **No equivalent benchmark exists for biological SAEs**
- Current bio-SAE papers use ad-hoc evaluation: GO enrichment, known annotation recovery, probing
- Need: standardized metrics for biological interpretability (beyond GO F1)

### Gap 5: SAE on Epigenomics / Chromatin Models
- No SAE work on epigenomic foundation models (chromatin accessibility, histone modification models)
- Models like Enformer, EpiGePT exist but are unanalyzed with SAEs
- Chromatin state prediction involves complex combinatorial logic that SAEs could decompose

### Gap 6: Temporal/Developmental SAE Analysis
- Kendiukhov (2026) touches on differentiation in Geneformer
- But no systematic SAE analysis of how bio-LLMs represent **temporal processes**, developmental trajectories, or dynamic cellular states
- scFMs process snapshots; do their SAE features encode trajectory information?

### Gap 7: Comparative SAE Architecture Study for Bio-LLMs
- Which SAE variant (TopK, JumpReLU, Matryoshka, transcoder, switch) works best for which biological modality?
- No systematic comparison exists
- The "Resurrecting the Salmon" insight about domain-specific SAEs has not been tested for biological domains

### Gap 8: SAE for Drug Discovery / Therapeutic Applications
- Chemistry SAEs exist (SMI-TED), protein SAEs exist
- **No SAE work on drug-target interaction models** or molecular docking models
- No connection between SAE features and actionable drug design decisions

### Gap 9: Scaling Laws for Biological SAEs
- How does SAE dictionary size, expansion factor, and sparsity level affect biological feature recovery?
- Reticular AI's work on ESM2-3B vs smaller models is a start, but no systematic scaling study

### Gap 10: SAE as Training Signal / Regularizer for Bio-LLMs
- All current work is post-hoc analysis
- No paper trains bio-LLMs **with** SAE-informed objectives or uses SAE features as auxiliary losses to improve biological fidelity

---

## 9. Key Technical Considerations for Applying SAE to Bio-LLMs

### 9.1 Tokenization Differences
- NLP: subword tokens (BPE, SentencePiece)
- Protein: individual amino acids (20 canonical + special tokens)
- DNA: individual nucleotides (4 + special) or k-mers (DNABERT uses 6-mers)
- Single-cell: gene tokens (variable vocabulary, ~20k-60k genes)
- **Implication**: Token-level SAE features have very different granularity across modalities. Protein/DNA tokens are biologically atomic; single-cell gene tokens are high-level.

### 9.2 Sequence Length and Context
- Protein sequences: typically 100-1000 residues
- DNA: can be megabases (Evo 2 handles 1Mb context)
- Single-cell: ~2000-5000 genes per cell (treated as "sentence")
- **Implication**: SAE computational cost scales with sequence length x model dimension. Long genomic contexts may need efficient SAE variants.

### 9.3 Superposition in Bio-LLMs
- InterPLM confirmed superposition in ESM-2
- Single-cell papers confirmed 99.8% of features invisible to SVD
- **Implication**: SAEs are well-motivated for bio-LLMs; biology IS stored in superposition, just like language concepts in NLP LLMs.

### 9.4 Evaluation Challenge
- NLP SAEs can be evaluated via human interpretability judgments
- Bio-SAEs require domain expertise or automated evaluation against biological databases
- **Key databases for evaluation**: Gene Ontology, UniProt, KEGG, Reactome, STRING, TRRUST, Pfam, InterPro
- **Risk**: Evaluation circularity -- if SAE features only recover what's in databases, how do we assess genuinely novel features?

### 9.5 Model Architecture Considerations
- ESM-2: Encoder-only transformer (BERT-style) -- most SAE work targets this
- Evo: Autoregressive (GPT-style) with StripedHyena / Mamba layers
- scGPT: GPT-style with gene tokens
- Geneformer: BERT-style with gene tokens
- **Implication**: SAE placement (residual stream, MLP output, attention output) and expected feature types may differ by architecture. Autoregressive models may have different feature structures than bidirectional ones.

### 9.6 Biological Specificity of SAE Features
- In NLP, features often correspond to human-interpretable concepts (languages, topics, sentiment)
- In biology, features may be:
  - **Structural**: secondary structure, solvent accessibility, domain boundaries
  - **Functional**: binding sites, catalytic residues, post-translational modification sites
  - **Evolutionary**: conservation patterns, family-specific motifs
  - **Regulatory**: promoters, enhancers, splice sites, TFBS
- **Open question**: Are biological SAE features more or less interpretable than NLP features? Early evidence suggests they are highly interpretable (perhaps more so, because biology provides rich ground-truth annotations).

### 9.7 Steering and Control
- Multiple papers show SAE features enable targeted steering of bio-LLM outputs
- Applications: generating proteins with specific properties, steering genomic sequences toward resistance mutations
- **Implication**: SAEs for bio-LLMs are not just interpretability tools but also **design tools** -- this is a major selling point over NLP applications where steering is more of a safety concern.

---

## 10. Timeline of Key Publications

| Date | Paper | Domain | Venue |
|------|-------|--------|-------|
| 2023-10 | Codebook Features (Tamkin et al.) | General | ICML |
| 2023-12 | Knowledge Neurons in PLMs | Protein | arXiv |
| 2024-10 | SAEs for gene expression models (Schuster) | Single-cell | arXiv |
| 2024-11 | InterPLM (Simon & Zou) | Protein | bioRxiv -> Nature Methods |
| 2024-12 | SAE on chemistry LMs (SMI-TED) | Chemistry | arXiv |
| 2025-02 | Mech. Interp. to Mech. Biology (Adams et al.) | Protein | bioRxiv -> ICML |
| 2025-02 | Garcia & Ansuini: Steering ESM-2 | Protein | arXiv |
| 2025-02 | Evo 2 + Goodfire SAE analysis | Genomic | bioRxiv |
| 2025-03 | Matryoshka SAEs on ESM2-3B (Reticular AI) | Protein/Structure | OpenReview |
| 2025-07 | SAEs on HyenaDNA (Guan et al.) | Genomic | arXiv |
| 2025-08 | Gujral et al.: SAEs+Transcoders on ESM-2 | Protein | PNAS |
| 2025-08 | Resurrecting the Salmon (domain-specific SAEs) | Medical/General | arXiv |
| 2025-09 | SAE Neural Operators | Scientific/PDEs | arXiv |
| 2025-09 | Transcoder circuits for scFMs | Single-cell | arXiv |
| 2025-10 | SAE-RNA on RiNALMo | RNA | arXiv |
| 2025-10 | SAEs on scGPT/scFoundation/Geneformer | Single-cell | bioRxiv |
| 2025-10 | Decode-gLM (Nucleotide Transformer) | Genomic | bioRxiv |
| 2025-11 | Equivariant SAEs | General/Scientific | arXiv |
| 2026-03 | SAE atlas of Geneformer + scGPT | Single-cell | arXiv |
| 2026-03 | Exhaustive circuit mapping of Geneformer | Single-cell | arXiv |
| 2026-03 | Review: What Do Bio FMs Compute? | Multi-domain | bioRxiv |

---

## 11. Key Research Groups

| Group | Affiliation | Focus |
|-------|-------------|-------|
| Elana Simon, James Zou | Stanford | InterPLM, protein SAEs |
| Etowah Adams, Ginkgo Bioworks/Columbia | Columbia / Ginkgo | Mechanistic biology, protein SAEs |
| Onkar Gujral, Eric Alm | MIT | SAEs + transcoders on proteins |
| Goodfire (company) | San Francisco | Evo 2 interpretability |
| Arc Institute | Palo Alto | Evo 2 genomic foundation model |
| Reticular AI (company) | -- | Matryoshka SAEs on ESMFold |
| Ihor Kendiukhov | U. Tubingen | Circuit mapping of scFMs |
| Viktoria Schuster | -- | Gene expression SAEs, scFeatureLens |
| Charles O'Neill | -- | Domain-specific SAEs |
| Ege Erdogan, Ana Lucic | U. Amsterdam | Equivariant SAEs |

---

## 12. Summary Assessment

### What exists:
- SAE on protein LMs: **mature, 5+ papers, Nature Methods + PNAS + ICML**
- SAE on genomic LMs: **active, 3-4 papers, strong results on Evo 2 and Nucleotide Transformer**
- SAE on single-cell FMs: **rapidly growing, 4-5 papers, extensive feature atlases**
- SAE on RNA LMs: **emerging, 1 paper**
- SAE on chemistry LMs: **emerging, 1 paper**
- Review/meta-analysis: **1 comprehensive review (2026)**

### What does NOT exist (clear gaps):
1. Cross-modal biological SAE analysis (protein + DNA + expression)
2. SAE-guided genuine biological discovery with experimental validation
3. SAE on 3D structure-aware models (AlphaFold, ProteinMPNN)
4. Standardized biological SAE benchmark (like SAEBench for NLP)
5. SAE on epigenomic/chromatin models
6. Systematic SAE architecture comparison for bio-LLMs
7. SAE as training signal for bio-LLMs (not just post-hoc analysis)
8. SAE on drug-target interaction or molecular docking models
9. Scaling laws for biological SAEs
10. Temporal/developmental trajectory analysis via SAE features

### The field's maturity level:
The field is at **"representational interpretability"** (level 1 of 3) -- we can identify which features encode which biological concepts. The next frontier is **computational interpretability** (how features are computed) and **causal mechanistic understanding** (using features to understand and control biology). This progression mirrors the NLP SAE field but is 1-2 years behind it.
