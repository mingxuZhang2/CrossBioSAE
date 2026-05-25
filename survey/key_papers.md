# Key Papers: LLM for Bioinformatics (2023-2026)

> Curated annotated bibliography of the most influential and representative papers
> Sources: Semantic Scholar, arXiv, Nature journals scraping
> Updated: 2026-05-25

---

## 1. Protein Language Models

### Foundational / High-Impact

1. **ESM3: Simulating 500 million years of evolution with a language model**
   - Authors: EvolutionaryScale team
   - Venue: bioRxiv 2024 (under review)
   - Key: First model to jointly model protein sequence, structure, and function. Demonstrated de novo functional protein generation (GFP with <60% identity to known proteins), suggesting true generalization beyond training distribution.
   - Validation: Purely computational + single wet lab validation of GFP fluorescence

2. **Efficient evolution of human antibodies from general protein language models**
   - Authors: Hie et al.
   - Venue: Nature Biotechnology 2023, 384 citations
   - Key: Used ESM-based protein language models to guide directed evolution of antibodies, achieving functional variants with only 20 or fewer mutations tested. Bridges PLM embeddings to experimental protein engineering.
   - Validation: Wet lab validation, but the computational screening component is the core methodological contribution

3. **Genome-wide prediction of disease variant effects with a deep protein language model**
   - Authors: Brandes et al.
   - Venue: Nature Genetics 2023, 326 citations
   - Key: ESM1b (650M params) applied to predict missense variant effects genome-wide. No need for multiple sequence alignments.
   - Validation: PURELY COMPUTATIONAL — validated against ClinVar, ProteinGym DMS assays, population frequency data

4. **DPLM-2: A Multimodal Diffusion Protein Language Model**
   - Authors: Wang et al.
   - Venue: ICLR 2025, 67 citations
   - Key: Combines protein sequence and structure in a diffusion language model framework.
   - Validation: ProteinGym benchmarks, inverse folding benchmarks

5. **Protein Large Language Models: A Comprehensive Survey**
   - Venue: EMNLP Findings 2025
   - Key: Comprehensive taxonomy of PLMs covering architectures, training strategies, downstream tasks

6. **SaProt: Protein Language Modeling with Structure-aware Vocabulary**
   - Venue: ICLR 2024
   - Key: Structure-aware tokenization using Foldseek 3Di structural alphabet combined with amino acid sequence

### Computational Validation Exemplars

7. **Retrieval-Enhanced Mutation Mastery: Augmenting Zero-Shot Prediction of Protein Language Model**
   - Venue: 2024, 21 citations
   - Key: Enhances PLM zero-shot predictions via retrieval of evolutionarily related sequences
   - Validation: PURELY COMPUTATIONAL — ProteinGym benchmark (217 DMS assays)

8. **A general temperature-guided language model to design proteins of enhanced stability and activity**
   - Venue: 2024, 41 citations
   - Key: Temperature-guided generation for stable protein design
   - Validation: Computational stability prediction + existing thermostability databases

---

## 2. DNA / Genomic Language Models

### Foundational

9. **DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome**
   - Authors: Zhou et al.
   - Venue: ICLR 2024, 405 citations
   - Key: BPE tokenization replacing k-mer, multi-species training. Established GUE (Genome Understanding Evaluation) benchmark.
   - Validation: PURELY COMPUTATIONAL — 28 tasks across 7 categories in the GUE benchmark

10. **HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution**
    - Authors: Nguyen et al.
    - Venue: NeurIPS 2023, 477 citations
    - Key: Hyena (SSM) architecture enabling 1M bp context at single-nucleotide resolution. Subquadratic complexity.
    - Validation: PURELY COMPUTATIONAL — GenomicBenchmarks, species classification, regulatory element prediction

11. **Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling**
    - Authors: Schiff et al.
    - Venue: ICML 2024, 202 citations
    - Key: BiMamba architecture with reverse-complement equivariance. Principled biological inductive bias.
    - Validation: PURELY COMPUTATIONAL — GUE benchmark, Nucleotide Transformer benchmark

12. **Evo: Sequence modeling and design from molecular to genome scale**
    - Authors: Nguyen et al. (Arc Institute)
    - Venue: Science 2024, 313 citations
    - Key: 7B parameter StripedHyena model trained on 2.7M genomes, 131k context. Models across DNA, RNA, and protein.
    - Validation: Mixed — primarily computational (gene essentiality prediction, regulatory element design) + limited wet lab (CRISPR guide RNA)

13. **Evo 2**
    - Venue: Nature 2026
    - Key: 40B params, 1M bp context, 9.3 trillion nucleotides training. Covers all domains of life.
    - Validation: Primarily computational — variant effect prediction, gene expression prediction

14. **DNA language model GROVER learns sequence context in the human genome**
    - Authors: Sanabria et al.
    - Venue: Nature Machine Intelligence 2024, 101 citations
    - Key: BPE tokenization, learns interpretable genome "words"
    - Validation: PURELY COMPUTATIONAL — chromatin accessibility, gene expression prediction

15. **AlphaGenome: advancing regulatory variant effect prediction**
    - Authors: DeepMind
    - Venue: Nature 2026, 79 citations
    - Key: Unified DNA sequence model for regulatory variant effect prediction across modalities
    - Validation: PURELY COMPUTATIONAL — eQTL datasets, GWAS fine-mapping, regulatory benchmarks

16. **BEND: Benchmarking DNA Language Models on biologically meaningful tasks**
    - Venue: ICLR 2024, 80 citations
    - Key: Standardized benchmark for evaluating DNA language models
    - Validation: PURELY COMPUTATIONAL — benchmark paper

17. **A DNA language model based on multispecies alignment predicts the effects of genome-wide variants**
    - Venue: Nature Biotechnology 2025, 76 citations
    - Key: MSA-based DNA language model for variant effect prediction

---

## 3. RNA Language Models

18. **RNA-FM: RNA Foundation Model**
    - Venue: Nature Methods (estimated)
    - Key: 100M parameters, trained on 23.7M non-coding RNAs. First large-scale RNA foundation model.
    - Validation: PURELY COMPUTATIONAL — secondary structure, function prediction benchmarks

19. **RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks**
    - Authors: Peniv et al.
    - Venue: Nature Communications 2024, 108 citations
    - Key: General-purpose RNA LM with strong structure prediction transfer
    - Validation: PURELY COMPUTATIONAL — RNA secondary/tertiary structure benchmarks

20. **Multiple sequence alignment-based RNA language model and its application to structural inference**
    - Venue: bioRxiv 2023, 96 citations
    - Key: MSA-based approach for RNA, analogous to ESM-MSA for proteins

21. **Orthrus: toward evolutionary and functional RNA foundation models**
    - Venue: Nature Methods 2026
    - Key: Latest RNA foundation model with evolutionary and functional understanding

22. **Generative AI powered by nucleic acid language model enables one-round evolution of RNA aptamers (GRAPE-LM)**
    - Venue: Nature Biotechnology 2026
    - Key: Nucleic acid language model for RNA aptamer design — one-round directed evolution

---

## 4. Single-Cell Foundation Models

23. **scGPT: toward building a foundation model for single-cell multi-omics using generative AI**
    - Authors: Cui et al.
    - Venue: Nature Methods 2024, 1011 citations
    - Key: GPT-style model trained on 33M cells. Tasks: cell type annotation, perturbation prediction, multi-omics integration, GRN inference.
    - Validation: PURELY COMPUTATIONAL — existing single-cell datasets, perturbation databases (Perturb-seq)

24. **Geneformer**
    - Venue: Nature 2024
    - Key: ~30M cells, rank-value encoding. Strong zero-shot performance on cell type annotation.
    - Validation: PURELY COMPUTATIONAL — CellxGene datasets, in silico perturbation against Perturb-seq

25. **scFoundation**
    - Venue: Nature Methods 2024
    - Key: 50M cells, 100M parameters. Largest training set at time of publication.
    - Validation: PURELY COMPUTATIONAL — standardized benchmarks

26. **CellFM**
    - Venue: Nature Communications 2025
    - Key: 100M cells for training. Scaled-up single-cell foundation model.

27. **Nicheformer: a foundation model for single-cell and spatial omics**
    - Venue: bioRxiv 2024, 111 citations
    - Key: Joint single-cell + spatial transcriptomics foundation model
    - Validation: PURELY COMPUTATIONAL — spatial + single-cell benchmarks

28. **Transformers in single-cell omics: a review and new perspectives**
    - Venue: Nature Methods 2024, 137 citations
    - Key: Comprehensive review of transformer architectures in single-cell analysis

29. **SATURN: Toward universal cell embeddings integrating single-cell RNA-seq datasets across species**
    - Venue: Nature Methods 2024, 98 citations
    - Key: Cross-species cell embedding using protein language model embeddings as gene representations

30. **A foundation model of transcription across human cell types (GET)**
    - Venue: Nature 2025, 97 citations
    - Key: Foundation model for transcriptional regulation across cell types

31. **scMamba: A Scalable Foundation Model for Single-Cell Multi-Omics Integration**
    - Venue: arXiv 2025
    - Key: Mamba architecture for single-cell, moving beyond Transformer

---

## 5. Drug Discovery & Molecular Generation

32. **Augmenting large language models with chemistry tools (ChemCrow)**
    - Authors: Bran et al.
    - Venue: Nature Machine Intelligence 2023, 910 citations
    - Key: LLM agent with 18 chemistry tools. Landmark paper for LLM-as-agent in science.
    - Validation: Computational + limited synthesis validation

33. **PocketFlow: a data-and-knowledge-driven structure-based molecular generative model**
    - Venue: Nature Machine Intelligence 2024, 85 citations
    - Key: Incorporates chemical knowledge into structure-based drug design

34. **DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design**
    - Venue: ICML 2024, 119 citations
    - Key: Decomposed diffusion for structure-based drug design
    - Validation: PURELY COMPUTATIONAL — CrossDocked2020 benchmark

35. **Generation of 3D molecules in pockets via a language model**
    - Venue: Nature Machine Intelligence 2023, 60 citations
    - Key: Language model approach for 3D pocket-conditioned molecular generation

36. **Unifying Molecular and Textual Representations via Multi-task Language Modelling**
    - Venue: ICML 2023, 127 citations
    - Key: Joint molecular-text model for property prediction and generation

37. **Large language models for scientific discovery in molecular property prediction**
    - Venue: Nature Machine Intelligence 2025, 93 citations

38. **Leveraging language model for advanced multiproperty molecular optimization via prompt engineering**
    - Venue: Nature Machine Intelligence 2024, 56 citations

39. **LaMGen: Large-scale 3D Molecular Generation**
    - Venue: Nature Communications 2026
    - Key: Multi-target drug design using LLM-based 3D molecular generation

---

## 6. Biomedical NLP & Clinical AI

40. **Large language models in medicine**
    - Venue: Nature Medicine 2023, 3101 citations
    - Key: Landmark review/perspective on LLMs in medicine

41. **Adapted large language models can outperform medical experts in clinical text summarization**
    - Venue: Nature Medicine 2023, 717 citations
    - Key: Demonstrated LLM superiority in clinical summarization

42. **A generalist vision-language foundation model for diverse biomedical tasks**
    - Venue: Nature Medicine 2023, 371 citations
    - Key: BiomedGPT — generalist model for biomedical vision-language tasks

---

## 7. LLM Agents for Bioinformatics

43. **BioInformatics Agent (BIA): Unleashing the Power of LLMs to Reshape Bioinformatics Workflow**
    - Venue: 2024, 31 citations
    - Key: LLM-powered bioinformatics workflow automation

44. **BioMedAgent: self-evolving multi-agent framework for bioinformatics**
    - Authors: ICT, CAS
    - Venue: 2025
    - Key: Multi-agent system that learns to use bioinformatics tools

45. **BioAgents: multi-agent system with RAG for bioinformatics**
    - Venue: Scientific Reports 2025
    - Key: Small language model + RAG for bioinformatics tasks

46. **SciToolAgent: knowledge-graph-driven scientific agent for multitool integration**
    - Venue: Nature Computational Science 2025
    - Key: KG-driven agent for scientific tool orchestration

47. **Agentic AI and the rise of in silico team science in biomedical research**
    - Venue: Nature Biotechnology 2026
    - Key: Perspective on AI agents replacing wet lab teams with computational teams

---

## 8. Multimodal Biological Foundation Models

48. **Towards multimodal foundation models in molecular cell biology**
    - Venue: Nature 2025, 82 citations
    - Key: Vision paper for multimodal bio foundation models

49. **A visual-omics foundation model to bridge histopathology with spatial transcriptomics**
    - Venue: Nature Methods 2025, 81 citations
    - Key: Bridges histology images and spatial transcriptomics

50. **STORM: spatial transcriptomics + histology multimodal foundation model**
    - Key: 1.2M spatial transcriptomics profiles + matched histology

51. **Generalist biological artificial intelligence (GenBioAI)**
    - Venue: Nature Biotechnology 2026
    - Key: Generalist model for the "language of life" across biological modalities

52. **Generalized biological foundation model with unified nucleic acid and protein language**
    - Venue: Nature Machine Intelligence 2025
    - Key: Unified model for DNA, RNA, and protein sequences

---

## 9. Variant Effect Prediction

53. **Accurate proteome-wide missense variant effect prediction with AlphaMissense**
    - Authors: Cheng et al. (DeepMind)
    - Venue: Science 2023, 1557 citations
    - Key: AlphaFold-derived model for pathogenicity prediction of all possible missense variants (71M variants classified)
    - Validation: PURELY COMPUTATIONAL — ClinVar, gnomAD, DMS data

54. **AlphaGenome**
    - Venue: Nature 2026, 79 citations
    - Key: Unified DNA sequence model from DeepMind for regulatory variant prediction

---

## 10. Benchmarks & Evaluation

55. **ProteinGym**: 217 DMS assays for protein variant effect prediction
56. **GUE (Genome Understanding Evaluation)**: from DNABERT-2, 28 genome tasks
57. **BEND**: DNA language model benchmark, ICLR 2024
58. **BixBench**: 53 bioinformatics analysis scenarios
59. **BioML-bench**: End-to-end biomedical ML evaluation
60. **LiveProteinBench**: Uncontaminated protein structure benchmark (post-2025 structures)

---

## 11. Key Reviews & Surveys

61. **Ruan et al.**: Comprehensive survey of LLMs in bioinformatics (Quantitative Biology 2026)
62. **ACL Findings 2025 Survey**: LLMs for biological sequence understanding
63. **Briefings in Bioinformatics 2025 Survey**: Foundation models in computational biology
64. **Nature Machine Intelligence 2026**: "Flow matching for generative modelling in bioinformatics"
65. **Nature Biotechnology 2026**: "Tracing the rise of biomedical foundation models"
