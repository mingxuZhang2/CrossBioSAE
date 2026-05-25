# Publishing LLM-for-Bio Papers WITHOUT Wet Lab Experiments

> Critical analysis for computational researchers (no wet lab access)
> Updated: 2026-05-25

---

## Executive Summary

A significant number of high-impact papers in the LLM-for-bioinformatics space have been published WITHOUT any wet lab experiments. This is especially true in the following categories: (1) genomic/protein language model pretraining, (2) benchmark and evaluation papers, (3) method papers validated on existing databases, and (4) single-cell foundation models. The key is rigorous computational validation using established benchmarks and public databases.

---

## 1. Validation Strategies That Replace Wet Lab Experiments

### Strategy A: Existing Experimental Databases as Ground Truth

This is the most common and most accepted approach. Use publicly available experimental measurements as test sets.

**Key Databases:**

| Domain | Database | What It Contains | Usage Example |
|--------|----------|------------------|---------------|
| Protein variants | **ProteinGym** (217 DMS assays) | Deep mutational scanning measurements for 217 proteins | AlphaMissense (Science), ESM variant prediction |
| Protein variants | **ClinVar** | Clinically annotated genetic variants | AlphaMissense, ESM1b disease variant prediction |
| Protein structure | **PDB** (200k+ structures) | Experimentally solved protein structures | ESMFold, structure prediction models |
| Protein function | **Gene Ontology (GO)** | Curated functional annotations | Protein function prediction papers |
| Genomic regulation | **ENCODE** | Chromatin accessibility, TF binding, histone marks | DNABERT-2, HyenaDNA, Caduceus |
| Genomic variants | **gnomAD** | Population allele frequencies | Variant pathogenicity prediction |
| Genomic variants | **eQTL catalogues** (GTEx) | Expression quantitative trait loci | AlphaGenome |
| Single-cell | **CellxGene** | Curated single-cell atlases | scGPT, Geneformer |
| Single-cell perturbation | **Perturb-seq** datasets | CRISPR perturbation + scRNA-seq | scGPT perturbation prediction |
| Drug molecules | **CrossDocked2020** | Protein-ligand docking poses | DecompDiff, PocketFlow |
| Drug activity | **ChEMBL** | Bioactivity measurements | Molecular property prediction |
| RNA structure | **bpRNA, ArchiveII** | RNA secondary structures | RiNALMo, RNA-FM |

**Success Examples (Purely Computational, Top Venue):**

1. **AlphaMissense (Science 2023, 1557 citations)**: Classified pathogenicity of 71M missense variants. Validated ENTIRELY on ClinVar, gnomAD frequency data, and 217 DMS assays from ProteinGym. No new wet lab experiments. Published in Science.

2. **Genome-wide prediction of disease variant effects with ESM1b (Nature Genetics 2023, 326 citations)**: Used ESM1b to predict disease variant effects genome-wide. Validated on ClinVar, population frequency data, and existing DMS datasets. No wet lab. Published in Nature Genetics.

3. **DNABERT-2 (ICLR 2024, 405 citations)**: Validated on GUE benchmark (28 existing datasets). Purely computational.

4. **scGPT (Nature Methods 2024, 1011 citations)**: Validated on existing single-cell datasets from CellxGene, Perturb-seq databases, and GRN inference benchmarks. No new wet lab data generated.

---

### Strategy B: Held-Out Test Sets with Temporal Splits

Split data by time — train on older data, test on newer data. This mimics prospective validation without requiring new experiments.

**Examples:**
- **LiveProteinBench**: Uses protein structures solved AFTER training data cutoff
- **AlphaGenome**: Tests on eQTLs and GWAS variants published after model training
- **AlphaMissense**: Temporal split using ClinVar annotation dates

**Why reviewers accept this:** Temporal splits prevent data leakage and simulate real-world predictive utility.

---

### Strategy C: Cross-Species / Cross-Domain Transfer

Demonstrate that a model trained on one organism generalizes to another. Since the test organism's data was not in training, this proves genuine learning.

**Examples:**
- **Evo 2 (Nature 2026)**: Trained on prokaryotic + eukaryotic genomes, tested on held-out species
- **SATURN (Nature Methods 2024)**: Cross-species cell type mapping
- **DNABERT-2**: Multi-species genome understanding

---

### Strategy D: Established Computational Benchmarks

Many standardized benchmarks have become accepted as sufficient validation.

**Protein:**
- ProteinGym (variant effect prediction, 217 assays)
- TAPE (Tasks Assessing Protein Embeddings)
- CASP (structure prediction)
- FLIP (Fitness Landscape Inference for Proteins)
- ProteinBench

**DNA/Genomics:**
- GUE (Genome Understanding Evaluation, from DNABERT-2)
- BEND (Benchmarking DNA Language Models)
- Nucleotide Transformer benchmark

**Single-Cell:**
- Perturbation prediction on Perturb-seq (Norman et al., Adamson et al.)
- Cell type annotation accuracy
- Batch correction metrics (kBET, ASW, LISI)
- GRN inference (BEELINE benchmark)

**Drug Discovery:**
- CrossDocked2020 (structure-based drug design)
- MOSES, GuacaMol (molecular generation)
- MoleculeNet (property prediction)

**RNA:**
- bpRNA, ArchiveII (secondary structure)
- RNA-Puzzles (3D structure)

---

### Strategy E: In Silico Perturbation / Counterfactual Analysis

Computationally simulate biological experiments using the model itself.

**Examples:**
- **scGPT**: In silico gene perturbation — predict how gene expression changes after knocking out a gene, compare to Perturb-seq ground truth
- **Geneformer**: In silico perturbation of transcription factors, validated against known biology
- **Evo**: In silico CRISPR guide RNA design efficiency prediction

**Why this works:** If the model's in silico predictions match experimental results from existing databases, it demonstrates the model has captured meaningful biology.

---

### Strategy F: Ablation Studies and Representation Analysis

Prove the model learns meaningful biological features through:
1. **Attention map analysis**: Show attention heads correspond to known biological features (e.g., secondary structure contacts, binding sites)
2. **Embedding space visualization**: Demonstrate biologically meaningful clustering
3. **Probing tasks**: Train simple classifiers on frozen embeddings
4. **Ablation studies**: Systematically remove components to show necessity

**Examples:**
- **GROVER (Nature MI 2024)**: Extensive analysis showing BPE tokens correspond to known genomic elements
- **HyenaDNA**: Analysis of learned long-range dependencies in DNA

---

### Strategy G: Retrospective Analysis on Clinical Data

Use existing clinical cohorts and electronic health records.

**Examples:**
- **EHRSHOT (NeurIPS 2023)**: Foundation model evaluation on EHR data
- Clinical text summarization papers validated on de-identified medical records

---

## 2. Venue-Specific Patterns

### Nature Methods / Nature Biotechnology
- **Most common validation**: Existing databases + standardized benchmarks
- **Expectation**: Comprehensive evaluation on multiple downstream tasks, comparison with multiple baselines
- **Examples without wet lab**: scGPT, DNABERT-2 (originally arXiv), GROVER, scFoundation, CellRank 2, SATURN

### Science / Nature
- **Higher bar** but still possible without wet lab
- **Requirement**: Either massive scale (AlphaMissense: 71M variants) or fundamental insight
- **Examples without wet lab**: AlphaMissense (Science 2023), Evo (Science 2024), AlphaGenome (Nature 2026), GET (Nature 2025)
- Some may include minimal wet lab as "supporting evidence" rather than core validation

### NeurIPS / ICML / ICLR
- **Easiest venue for computational-only work** in this space
- **Requirement**: Novel method + computational benchmarks + ablation
- **Examples**: HyenaDNA (NeurIPS 2023), Caduceus (ICML 2024), DNABERT-2 (ICLR 2024), DecompDiff (ICML 2024)

### Nature Machine Intelligence
- **Accepts computational-only work readily**
- **Examples**: GROVER, PocketFlow, ChemCrow, molecular optimization papers

### Nature Communications
- **Lower impact factor but still Nature series**
- **Very receptive to computational-only work with thorough validation**
- **Examples**: RiNALMo, CellFM

### Bioinformatics / Briefings in Bioinformatics
- **Standard bioinformatics journals — computational-only is the norm**
- **Requirement**: Solid method + comprehensive benchmarks + available code/data

---

## 3. Paper Categories Best Suited for No-Wet-Lab Publishing

### Tier 1: Easiest (Highest Acceptance Rate for Computational-Only)

1. **Benchmark / Evaluation Papers**
   - Create new benchmark, evaluate existing models, identify gaps
   - Examples: ProteinGym, BEND, GUE, BixBench
   - Target: NeurIPS Datasets & Benchmarks, ICLR, Nature Methods

2. **Foundation Model Pretraining**
   - Pretrain a new model on public data, evaluate on standard benchmarks
   - Examples: DNABERT-2, HyenaDNA, Caduceus, RiNALMo
   - Target: NeurIPS/ICML/ICLR, Nature MI, Nature Methods

3. **Method papers with existing-data validation**
   - New algorithm evaluated on ProteinGym / GUE / Perturb-seq
   - Target: Any venue

### Tier 2: Moderate (Accepted but Needs Strong Validation)

4. **Variant Effect Prediction**
   - Rich existing ground truth (ClinVar, DMS, gnomAD)
   - Examples: AlphaMissense, ESM1b variant prediction
   - Target: Nature Genetics, Science, Nature Methods

5. **Single-Cell Foundation Models**
   - Large public datasets (CellxGene, HCA), established tasks
   - Examples: scGPT, Geneformer
   - Target: Nature Methods, Nature Comm

6. **Structure-Based Drug Design (Computational)**
   - CrossDocked2020 benchmark is standard
   - Target: ICML, NeurIPS, Nature MI

### Tier 3: Harder (May Need Minimal Wet Lab or Collaboration)

7. **Protein Design / Engineering**
   - Reviewers increasingly expect some experimental validation
   - But several pure computational papers have been published (ProteinMPNN validated computationally on held-out structures)
   - Potential workaround: Collaborate with wet lab for minimal validation

8. **Drug Discovery End-to-End**
   - Harder to publish high-impact without any wet lab
   - But molecular property prediction is fully computational

---

## 4. Practical Recommendations for Dr. Zhang

### Strongest Opportunities (No Wet Lab Required):

1. **Build a comprehensive benchmark** for an under-benchmarked area (e.g., RNA foundation models, multi-omics integration, spatial transcriptomics)
2. **Propose a new foundation model architecture** for biological sequences, validated on existing benchmarks
3. **Cross-modal transfer learning** — use protein LM embeddings for genomics tasks or vice versa, evaluated on public data
4. **In silico perturbation prediction** — new methods validated on Perturb-seq data
5. **Variant effect prediction** — rich ground truth from ClinVar, DMS, and gnomAD
6. **LLM agents for bioinformatics** — evaluated on BixBench or custom analysis tasks

### What Reviewers Look For in Computational-Only Papers:

1. **Multiple benchmarks** — not just one dataset, but diverse evaluation
2. **Ablation studies** — prove each component contributes
3. **Comparison with strong baselines** — not just older methods
4. **Biological interpretability** — show the model captures meaningful biology
5. **Released code and models** — reproducibility is non-negotiable
6. **Temporal or species holdout** — prevent data leakage concerns
7. **Error analysis** — honestly discuss where the model fails

### Red Flags That Lead to Rejection:

1. Single benchmark evaluation
2. Weak baselines only
3. No code/data availability
4. Claims of "design" without any validation (computational or experimental)
5. Ignoring well-known issues (data leakage, contamination)
