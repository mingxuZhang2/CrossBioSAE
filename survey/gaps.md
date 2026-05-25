# Research Gaps & Opportunities: LLM for Bioinformatics

> Focus: Opportunities for computational researchers without wet lab access
> Updated: 2026-05-25

---

## Gap 1: RNA Language Models Are Severely Under-Explored

**Current State:**
- Protein LMs: ESM-2 (15B), ESM3, ProtTrans, ProGen, SaProt, DPLM-2 — dozens of models
- DNA LMs: DNABERT-2, HyenaDNA, Caduceus, Evo, Evo 2, GROVER, Nucleotide Transformer — many well-established
- RNA LMs: Only RNA-FM (~100M), RiNALMo, Orthrus — **extremely few models, all relatively small**

**Why This Is a Gap:**
- RNA is a major drug target class (mRNA vaccines, antisense oligonucleotides, siRNA)
- RNA has unique features (secondary structure, pseudoknots, modifications) not well-captured by DNA/protein models
- Training data is now abundant: RNAcentral has 40M+ sequences
- No RNA LM approaches the scale or impact of ESM/Evo

**Opportunity:**
- Build a large-scale RNA foundation model (1B+ params) with structure-aware training
- Incorporate RNA modification data (m6A, pseudouridine)
- Validate on secondary structure, 3D structure, function prediction, RNA-protein interaction
- **Feasibility**: High — public data available, existing codebase can be adapted from protein/DNA LMs
- **No wet lab needed**: RNA benchmarks exist (bpRNA, ArchiveII, RNA-Puzzles)
- **Target venues**: Nature Methods, Nature Communications, NeurIPS/ICML

**Competitive Risk**: Moderate — Orthrus (Nature Methods 2026) is recent but leaves room for larger models and different architectures

---

## Gap 2: No Unified Cross-Modal Benchmark for Biological Foundation Models

**Current State:**
- Protein benchmarks: ProteinGym, TAPE, FLIP — well-established
- DNA benchmarks: GUE, BEND — adequate
- Single-cell benchmarks: emerging (BioLLM 2025)
- RNA benchmarks: fragmented
- **Cross-modal benchmarks: NON-EXISTENT**

**Why This Is a Gap:**
- Models like Evo and "Generalized biological foundation model" (Nature MI 2025) claim to handle DNA + RNA + protein
- But there is NO standardized way to evaluate cross-modal transfer or joint modeling
- Without such benchmarks, it is impossible to fairly compare generalist vs. specialist models

**Opportunity:**
- Create "BioGLUE" (Biological General Language Understanding Evaluation): unified benchmark spanning DNA, RNA, protein, and optionally single-cell tasks
- Include cross-modal tasks (e.g., predict protein function from DNA sequence, predict RNA structure from genomic context)
- **Feasibility**: Very high — all component datasets are public, this is a curation + engineering effort
- **No wet lab needed**: Pure benchmark paper
- **Target venues**: NeurIPS Datasets & Benchmarks track, Nature Methods
- **Impact potential**: Very high — benchmark papers get cited by every subsequent model paper

---

## Gap 3: Mamba/SSM Architecture Not Yet Applied to Single-Cell Data at Scale

**Current State:**
- DNA: Mamba has been very successful (HyenaDNA, Caduceus, Evo)
- Protein: Some Mamba exploration
- Single-cell: Almost all models use Transformer (scGPT, Geneformer, scFoundation)
- Only scMamba (arXiv 2025) has explored this, still early stage

**Why This Is a Gap:**
- Single-cell expression profiles can have 20,000+ genes per cell (long "sequence")
- Transformers have O(n^2) attention — wasteful for very long gene vectors
- Mamba's O(n) complexity is ideal for scaling to whole-transcriptome modeling
- The success of Mamba in DNA strongly suggests it will work for single-cell

**Opportunity:**
- Build a Mamba-based single-cell foundation model at scale (100M+ cells)
- Demonstrate efficiency gains over scGPT/Geneformer with comparable or better performance
- Include multi-omics (ATAC-seq, protein) capabilities
- **Feasibility**: High — scGPT codebase is open-source, CellxGene provides training data, Mamba implementations are well-documented
- **No wet lab needed**: Evaluate on standard single-cell benchmarks
- **Target venues**: NeurIPS/ICML, Nature Methods

**Competitive Risk**: Moderate — scMamba exists but is early/limited

---

## Gap 4: LLM Agents for Bioinformatics Lack Rigorous Evaluation

**Current State:**
- Several agents: BIA, BioMedAgent, BioAgents, BioMaster, CompBioAgent
- But evaluations are ad hoc — each paper uses its own tasks and metrics
- BixBench (53 scenarios) exists but is underutilized

**Why This Is a Gap:**
- Agent papers are trendy and getting published, but the field lacks standardization
- Most agents are tested on simple tasks (run BLAST, basic pipeline execution)
- Complex multi-step biological analyses remain untested
- No systematic comparison of agent architectures

**Opportunity:**
- Create a comprehensive agent benchmark for bioinformatics covering:
  - Simple tool use (BLAST, alignment)
  - Multi-step workflows (RNA-seq pipeline, variant calling pipeline)
  - Data interpretation (statistical analysis, visualization)
  - Hypothesis generation from data
- Evaluate existing agents + propose improved agent architecture
- **Feasibility**: High — bioinformatics tools are well-documented, test data is public
- **No wet lab needed**: All computational
- **Target venues**: NeurIPS, Nature Methods, Bioinformatics

---

## Gap 5: Spatial Transcriptomics Foundation Models Are Nascent

**Current State:**
- Only a few models: STORM, spaLLM, Nicheformer, TissueNarrator
- Most are very recent (2024-2025) and preliminary
- The visual-omics bridge (histology + spatial transcriptomics) is barely explored

**Why This Is a Gap:**
- Spatial transcriptomics is one of the fastest-growing experimental technologies
- Public data is rapidly accumulating (10x Visium, MERFISH, Slide-seq)
- Current models do not effectively integrate spatial context with gene expression
- Cross-platform generalization (Visium to MERFISH) remains unsolved

**Opportunity:**
- Build a spatial transcriptomics foundation model that captures:
  - Gene expression patterns
  - Spatial relationships (neighbor effects, gradients)
  - Tissue architecture
- Use graph neural networks or spatial-aware transformers
- **Feasibility**: Moderate — requires understanding spatial data formats, public datasets available
- **No wet lab needed**: Public spatial transcriptomics datasets from 10x Genomics, SpatialDB
- **Target venues**: Nature Methods, NeurIPS, Cell Systems

**Competitive Risk**: High — this space is heating up rapidly

---

## Gap 6: Perturbation Prediction Models Lack Combinatorial Generalization

**Current State:**
- scGPT, GEARS, GenePert can predict single-gene perturbation effects
- But real biological experiments often involve multi-gene perturbations (combinatorial)
- Current models struggle to generalize to unseen gene combinations

**Why This Is a Gap:**
- Combinatorial perturbation space is exponential — computational prediction is essential
- Existing models fail on novel combinations (shown by recent Genome Biology 2025 evaluation)
- This is exactly where ML should shine but current methods fall short

**Opportunity:**
- Develop models that explicitly capture gene-gene interactions for combinatorial perturbation prediction
- Use graph neural networks, attention mechanisms, or factorized representations
- Validate on the growing Perturb-seq combinatorial datasets (Norman et al. 2019)
- **Feasibility**: High — Perturb-seq data is public, GEARS codebase is open-source
- **No wet lab needed**: Perturb-seq ground truth available
- **Target venues**: NeurIPS/ICML, Nature Methods, Genome Biology

---

## Gap 7: Epigenomics Foundation Models Are Nearly Absent

**Current State:**
- Only EpiGePT (Genome Biology 2024) exists as a dedicated epigenomics model
- DNA LMs like DNABERT-2 and Evo predict some epigenomic features, but as secondary tasks
- No dedicated large-scale model for histone modifications, DNA methylation, chromatin conformation

**Why This Is a Gap:**
- Epigenomics is central to gene regulation, disease, and development
- Massive public data from ENCODE, Roadmap Epigenomics, BLUEPRINT
- Current DNA LMs do not effectively model the cell-type-specificity of epigenomic marks

**Opportunity:**
- Build a foundation model specifically for epigenomic data
- Model: DNA sequence + cell-type context → histone marks, methylation, chromatin accessibility
- Cross-cell-type transfer learning (train on well-characterized cell types, predict for rare cell types)
- **Feasibility**: High — ENCODE/Roadmap data is comprehensive and standardized
- **No wet lab needed**: All data is public
- **Target venues**: Nature Methods, Genome Biology, NeurIPS

---

## Gap 8: Microbiome / Metagenomics Foundation Models Are Immature

**Current State:**
- Only MGM (260k samples) and GenomeOcean exist
- Recent analysis shows AI methods barely outperform classical baselines for microbiome disease prediction
- The field lacks a "scGPT moment"

**Why This Is a Gap:**
- Human gut microbiome is linked to hundreds of diseases
- 16S/metagenomic data is abundant but underutilized for LLM-style modeling
- Compositional nature of microbiome data requires specialized modeling

**Opportunity:**
- Build a microbiome foundation model that captures species interactions, metabolic functions, and compositional patterns
- Use the HMP, MetaHIT, and other public metagenomic datasets
- **Feasibility**: Moderate — data is messy, but public datasets exist
- **No wet lab needed**: Public metagenomics data
- **Target venues**: Nature Methods, Genome Biology, Nature Microbiology

---

## Gap 9: Tokenization for Biological Sequences Remains Suboptimal

**Current State:**
- DNA: k-mer vs. BPE vs. single-nucleotide debate is unresolved
- Protein: single amino acid tokenization is standard but ignores structural motifs
- RNA: no consensus on tokenization
- No systematic study comparing tokenization strategies across modalities

**Why This Is a Gap:**
- Tokenization fundamentally affects what the model can learn
- BPE on DNA (DNABERT-2, GROVER) discovers interpretable genomic "words"
- But no one has systematically optimized tokenization for biological sequences
- Structure-aware tokenization (SaProt's Foldseek approach) is promising but under-explored for DNA/RNA

**Opportunity:**
- Systematic study of tokenization strategies for biological sequences
- Propose biologically-informed tokenization (e.g., codon-aware for coding regions, motif-aware for regulatory regions)
- **Feasibility**: Very high — requires training multiple models with different tokenizations on public data
- **No wet lab needed**: Entirely computational
- **Target venues**: ICLR/NeurIPS, Bioinformatics

---

## Gap 10: Bridging Protein LM Representations and Genomic Context

**Current State:**
- Protein LMs and DNA LMs are trained separately
- SATURN (Nature Methods 2024) used protein LM embeddings as gene representations for single-cell analysis, but this is a rare example
- No model jointly learns from DNA context (regulatory regions) + protein sequence + protein function

**Why This Is a Gap:**
- Gene expression depends on BOTH the protein-coding sequence AND the regulatory DNA context
- Current models ignore this multi-scale relationship
- A unified representation could enable cross-modal prediction (regulatory variant -> protein function change)

**Opportunity:**
- Build a model that connects genomic context (promoter, enhancers) to protein properties
- Predict: how do non-coding variants affect protein function indirectly through expression changes?
- Use existing eQTL data + protein function databases as supervision
- **Feasibility**: Moderate — requires integrating multiple data sources
- **No wet lab needed**: GTEx eQTL + UniProt + ClinVar data
- **Target venues**: Nature Methods, Nature Genetics, NeurIPS

---

## Summary: Top 5 Most Promising Gaps for Dr. Zhang

| Rank | Gap | Feasibility | Impact | Wet Lab Needed? | Competition |
|------|-----|-------------|--------|-----------------|-------------|
| 1 | Cross-modal benchmark (BioGLUE) | Very High | Very High | No | Low |
| 2 | RNA Foundation Model (large-scale) | High | Very High | No | Moderate |
| 3 | Mamba for Single-Cell | High | High | No | Moderate |
| 4 | Combinatorial Perturbation Prediction | High | High | No | Low |
| 5 | Biological Tokenization Study | Very High | Moderate-High | No | Low |
