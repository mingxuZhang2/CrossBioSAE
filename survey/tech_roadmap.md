# Technology Roadmap: LLM for Bioinformatics

> Machine-readable synthesis of field evolution and trends
> Updated: 2026-05-25

---

## Timeline of Key Milestones

### Phase 1: Foundations (2019-2022)
- 2019: ESM-1b (Meta AI) — first large protein language model
- 2020: ProtTrans — BERT/GPT/XLNet applied to proteins
- 2021: AlphaFold2 — protein structure prediction revolution (not LM, but catalyzed the field)
- 2021: DNABERT — first BERT-based DNA model (k-mer tokenization)
- 2022: Geneformer (v1) — first transformer for single-cell gene expression
- 2022: ESM-2 — scaled to 15B parameters
- 2022: ProGen2 — protein generation at scale

### Phase 2: Rapid Expansion (2023)
- 2023: AlphaMissense (Science) — proteome-wide variant prediction
- 2023: HyenaDNA (NeurIPS) — SSM architecture enters genomics, 1M context
- 2023: DNABERT-2 — BPE tokenization for DNA, GUE benchmark
- 2023: ChemCrow — first LLM-agent for chemistry (Nature MI)
- 2023: Nature Medicine review: "Large language models in medicine" (3100+ cites)
- 2023: scGPT paper on bioRxiv (later Nature Methods 2024)

### Phase 3: Maturation & Scale (2024)
- 2024: scGPT published in Nature Methods (1000+ cites)
- 2024: Geneformer published in Nature
- 2024: Caduceus (ICML) — BiMamba for DNA with RC equivariance
- 2024: Evo (Science) — 7B model spanning DNA/RNA/protein
- 2024: GROVER (Nature MI) — BPE discovers interpretable genomic words
- 2024: RiNALMo (Nature Comms) — general-purpose RNA LM
- 2024: scFoundation (Nature Methods) — 50M cell training
- 2024: SATURN (Nature Methods) — protein LM embeddings for cross-species single-cell
- 2024: Nobel Prize in Chemistry for AI protein structure/design
- 2024: EpiGePT (Genome Biology) — first epigenomics foundation model

### Phase 4: Integration & Generalization (2025-2026)
- 2025: Nature editorial — "Towards multimodal foundation models in molecular cell biology"
- 2025: GET (Nature) — foundation model of transcription
- 2025: AlphaGenome (Nature 2026) — unified DNA regulatory model
- 2025: Nature MI — unified nucleic acid + protein language model
- 2025: CellFM (Nature Comms) — 100M cell single-cell model
- 2025: BioLLM benchmark framework for single-cell models
- 2026: Evo 2 (Nature) — 40B params, 1M context, all life domains
- 2026: Nature Biotech — "Generalist biological AI" and "Agentic AI in biomedical research"
- 2026: Orthrus (Nature Methods) — evolutionary RNA foundation model
- 2026: GRAPE-LM (Nature Biotech) — RNA aptamer design via nucleic acid LM

---

## Architecture Evolution

```
2019-2022: BERT/GPT Transformers (ESM, ProtTrans, DNABERT)
    |
    v
2023: State Space Models enter (HyenaDNA, StripedHyena)
    |
    v  
2024: Mamba + Biological Inductive Biases (Caduceus: BiMamba + RC equivariance)
    |
    v
2025-2026: Hybrid architectures + Unified cross-modal models
    (Evo 2: StripedHyena 40B, unified DNA/RNA/protein)
```

### Key Architecture Transitions:

| Year | From | To | Motivation | Example |
|------|------|----|------------|---------|
| 2023 | Transformer (O(n^2)) | Hyena/SSM (O(n)) | Long DNA sequences (>10kb) | HyenaDNA |
| 2024 | Unidirectional SSM | Bidirectional SSM | DNA has no natural direction | Caduceus (BiMamba) |
| 2024 | k-mer tokenization | BPE tokenization | Data-driven vocabulary discovery | DNABERT-2, GROVER |
| 2024 | Sequence-only | Sequence + Structure | Protein structure informs function | SaProt, ESM3 |
| 2025 | Single-modal | Multi-modal | Biological cross-talk | Evo, unified models |
| 2026 | Single-modality FM | Generalist Bio AI | One model for all biology | GenBioAI |

---

## Scaling Trends

### Model Size
```
2019: ESM-1b          ~650M params
2022: ESM-2           15B params
2023: HyenaDNA        ~1.5B params (long context)
2024: Evo             7B params
2026: Evo 2           40B params
```

### Training Data
```
2019: ESM-1b          ~250M protein sequences (UniRef)
2022: scGPT           33M cells
2024: Evo             2.7M whole genomes
2024: scFoundation    50M cells
2025: CellFM          100M cells
2026: Evo 2           9.3 trillion nucleotides
```

### Context Length (DNA)
```
2021: DNABERT          ~512bp (k-mer, effectively shorter)
2023: DNABERT-2        ~512bp  
2023: HyenaDNA         1,000,000bp
2024: Caduceus         >100,000bp
2024: Evo              131,072bp
2026: Evo 2            1,000,000bp
```

---

## Performance Landscape (Key Benchmarks)

### Protein Variant Effect Prediction (ProteinGym, Spearman ρ)
| Model | Year | ProteinGym (average Spearman) |
|-------|------|-------------------------------|
| EVE (evolutionary) | 2021 | ~0.44 |
| ESM-1v (zero-shot) | 2022 | ~0.43 |
| AlphaMissense | 2023 | ~0.48 (ClinVar AUROC 0.97) |
| ESM3 (zero-shot) | 2024 | ~0.46 |
| Retrieval-enhanced PLM | 2024 | ~0.49 |

### DNA Benchmark (GUE, average accuracy)
| Model | Year | GUE Average |
|-------|------|-------------|
| DNABERT | 2021 | baseline |
| Nucleotide Transformer | 2023 | improved |
| DNABERT-2 | 2023 | significant improvement |
| Caduceus | 2024 | competitive with DNABERT-2 |
| Evo 2 | 2026 | new SOTA on most tasks |

### Single-Cell (Cell Type Annotation, Accuracy)
| Model | Year | Accuracy (typical datasets) |
|-------|------|----------------------------|
| Traditional (scANVI) | pre-2023 | ~85-90% |
| scGPT | 2024 | ~90-95% |
| Geneformer | 2024 | ~88-93% |
| scFoundation | 2024 | ~90-95% |
| scMamba | 2025 | ~91-95% (with efficiency gains) |

---

## Mainstream Approach Families

### Family 1: Masked Language Modeling (BERT-style)
- Train: mask tokens, predict masked tokens
- Models: ESM-2, DNABERT-2, scBERT, RNA-FM
- Strengths: Strong representations for discriminative tasks
- Weaknesses: No generation capability

### Family 2: Autoregressive Generation (GPT-style)
- Train: next-token prediction
- Models: ProGen, scGPT, Evo, GROVER
- Strengths: Can generate new sequences, scalable training
- Weaknesses: Unidirectional context (problematic for DNA)

### Family 3: Sequence-to-Structure Prediction
- Train: predict 3D structure from sequence
- Models: ESM3, ESMFold, AlphaFold (not LM)
- Strengths: Captures physical constraints
- Weaknesses: Requires structural training data

### Family 4: Diffusion / Flow Matching
- Train: denoising / flow matching on sequences or structures
- Models: DPLM-2, DecompDiff, PocketFlow, flow matching for bio
- Strengths: High-quality generation, controllable
- Weaknesses: Slower inference than autoregressive

### Family 5: State Space Models
- Train: SSM-based sequence modeling
- Models: HyenaDNA, Caduceus (BiMamba), Evo/Evo 2 (StripedHyena)
- Strengths: O(n) complexity, very long context
- Weaknesses: Less established than Transformers, fewer available implementations

### Family 6: Contrastive / Multi-modal Alignment
- Train: align representations across modalities (sequence-text, sequence-structure)
- Models: SaProt, ProtLLM, MolBind, BioVERSE
- Strengths: Cross-modal transfer, zero-shot capability
- Weaknesses: Requires paired cross-modal data

---

## Emerging Trends (2025-2026)

1. **Unified biological language models** — single model for DNA, RNA, protein (Evo 2, GenBioAI)
2. **Agentic AI for biology** — LLM agents autonomously running bioinformatics workflows
3. **Flow matching** emerges as a dominant generative approach for biomolecular design
4. **Scaling hits diminishing returns** — Genome Biology 2025 shows single-cell FMs may not outperform simpler baselines for some tasks
5. **Benchmark-driven evaluation** becomes standard — ad hoc evaluation is no longer accepted
6. **Virtual cell** concept gaining traction — multi-scale models integrating genomics, proteomics, cell biology
7. **IRES/UTR design** — programmable translation regulation via LLMs (Nature MI 2026)
8. **One-round evolution** — LLMs reducing experimental cycles from many rounds to one (GRAPE-LM)
