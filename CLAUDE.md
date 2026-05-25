# Project: LLM for Bioinformatics Research

## Overview
Literature survey and research ideation project exploring LLM applications in bioinformatics.
Focus: Identify research opportunities for a computational researcher without wet lab access.

## Project Status
- Stage 1 (Survey): COMPLETE
- Stage 2 (Implementation): Not started — awaiting Dr. Zhang's idea selection
- Stage 3 (Writing): Not started

## Directory Structure
```
survey/
  landscape.md          — Current research landscape (8 sub-directions + 2026 updates)
  key_papers.md         — Annotated bibliography of 65+ key papers
  no_wetlab_publishing.md — Strategies for publishing without wet lab experiments
  gaps.md               — 10 identified research gaps with feasibility assessment
  tech_roadmap.md       — Technology roadmap (timeline, architectures, benchmarks)
  papers_*.json         — Raw Semantic Scholar search results (~460 papers)
  arxiv_*.json          — Raw arXiv search results (~210 papers)
  nature_*.json         — Raw Nature journal scraping results (~193 papers)
```

## Data Sources
- Semantic Scholar API: 10 topic-specific searches, 50 papers each
- arXiv API: 8 topic-specific searches, 30 papers each
- Nature journal scraping: 5 journals, 2023-2026

## Key Findings
1. RNA foundation models are severely under-explored compared to protein/DNA
2. No unified cross-modal benchmark exists for biological foundation models
3. Mamba/SSM architecture not yet applied to single-cell data at scale
4. LLM agents for bioinformatics lack rigorous evaluation standards
5. Purely computational papers regularly appear in Science, Nature, Nature Methods
