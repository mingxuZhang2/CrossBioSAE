# Project: LLM for Bioinformatics Research

## Overview
Literature survey and research ideation project exploring LLM applications in bioinformatics.
Focus: Identify research opportunities for a computational researcher without wet lab access.
Latest deepening (2026-05-26): focused on **Sparse Autoencoders + Biological LLMs for scientific discovery**.

## Project Status
- Stage 1 (Survey): COMPLETE — general LLM-for-bio + deep SAE+bio sub-survey
- Stage 2 (Implementation): Not started — awaiting Dr. Zhang's idea selection from `survey/sae_bio_ideas.md`
- Stage 3 (Writing): Not started

## Directory Structure
```
survey/
  landscape.md                       — General LLM-for-bio landscape (8 sub-directions + 2026 updates)
  key_papers.md                      — Annotated bibliography of 65+ key papers
  no_wetlab_publishing.md            — General strategies for publishing without wet lab
  no_wetlab_publishing_sae_addendum.md — SAE+bio-specific publishing strategies & 15 example papers
  gaps.md                            — 10 general LLM-for-bio gaps
  tech_roadmap.md                    — Technology roadmap (timeline, architectures, benchmarks)
  sae_bio_landscape.md               — SAE × bio-LLM landscape (20+ papers across protein/DNA/RNA/scFM)
  sae_bio_gaps.md                    — 12 sharply defined gaps at SAE+bio+discovery intersection
  sae_bio_ideas.md                   — 5 executable research ideas (FINAL Stage-1 DELIVERABLE)
  papers_*.json                      — Raw Semantic Scholar search results (~460 papers)
  arxiv_*.json                       — Raw arXiv search results (~210+ papers)
  nature_*.json                      — Raw Nature journal scraping results (~193 papers)
  ss_sae_*.json / ss_mechinterp_bio.json / ss_superposition_bio.json / ss_sparse_dict_bio.json /
  ss_interp_foundation_bio.json      — SAE+bio targeted searches
```

## Data Sources
- Semantic Scholar API: ~14 topic-specific searches (10 general + 4 SAE-targeted), 50 papers each
- arXiv API: ~9 topic-specific searches, 30 papers each
- Nature journal scraping: 5 journals, 2023-2026

## Key Findings — General LLM-for-bio
1. RNA foundation models are severely under-explored compared to protein/DNA
2. No unified cross-modal benchmark exists for biological foundation models
3. Mamba/SSM architecture not yet applied to single-cell data at scale
4. LLM agents for bioinformatics lack rigorous evaluation standards
5. Purely computational papers regularly appear in Science, Nature, Nature Methods

## Key Findings — SAE + bio-LLM intersection
1. The SAE+bio field grew from ~0 papers in late 2023 to ~20+ in 2025–2026; Nature Methods (InterPLM), PNAS, ICML, AAAI accepted papers exist.
2. All published bio-SAE work is on ESM-2; **ESM-Cambrian / ESM-C is untouched** (clear low-hanging opportunity).
3. **No SAE on chromatin/ATAC FMs** (CLM-access, Enformer, Borzoi); the empty cell in the matrix.
4. RNA-FM SAE coverage is minimal (only SAE-RNA on RiNALMo).
5. **No cross-modal SAE** (protein-LM ⊗ DNA-LM) despite Universal SAE methodology existing.
6. 2026 negative-result papers (Causal Circuit Tracing, Systematic Evaluation, "Interpretability without actionability") show the field is now sceptical of mere correlation; causal validation is the new bar.
7. The "scientific discovery without wet lab" path requires Tier-2/Tier-3 validation: temporal hold-out, orthogonal oracle, or causal intervention against held-out Perturb-seq/ClinVar.

## Stage-1 Deliverables for Dr. Zhang's Review
- **Primary:** `survey/sae_bio_ideas.md` — 5 concrete, executable research ideas (PathoSAE, EnhancerSAE, CrossBioSAE, RNA-MotifSAE, AuditSAE).
- **Supporting:** `survey/sae_bio_gaps.md`, `survey/sae_bio_landscape.md`, `survey/no_wetlab_publishing_sae_addendum.md`.

Waiting for Dr. Zhang to pick one idea before launching Stage 2.
