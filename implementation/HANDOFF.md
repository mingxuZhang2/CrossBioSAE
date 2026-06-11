# CrossCoder SAE — GPT Pro Handoff

Date: 2026-06-11 (updated)
Status: Analysis pipeline fixed per GPT Pro R3 feedback, ready to run on HPC

## 0. GPT Pro R3 Feedback Response (2026-06-11)

### What GPT Pro said → What we fixed

| GPT Pro Issue | Fix Applied |
|--------------|-------------|
| `classify_features()` called on different subsets → inconsistent categories | Canonical categories computed ONCE on ALL DMS data, saved to `feature_categories.npz`, all analyses read same file |
| ClinVar "OR" is actually risk ratio, not standard OR | Replaced with 2×2 Fisher exact test + Haldane-Anscombe pseudo-count + BH-FDR correction |
| Gene modality profile biased by DP having 3.8× more features | Now outputs 3 normalizations: raw mass, per-feature-mean, active-feature-normalized |
| Shuffled-pair control is inference-time only, need train-time control too | Split into Control A (inference-time shuffle) + Control B (retrain on shuffled pairs via `--shuffle_pairs` flag) |
| Results only printed, not saved as structured output | All analyses save JSON + CSV; `analysis_summary.json` aggregates everything |
| `sae_interpretability_controls.py` imports v6 modules, not CrossCoder | Noted — that script is NOT used for CrossCoder analyses. `analyze_crosscoder.py` is the correct pipeline |

### GPT Pro's agreed narrative

> **Variant-level paired-modal sparse decomposition: missense variant effects are mostly protein-private, but DNA/genomic features provide weak, distributed, gene-dependent modifiers.**

NOT: "shared biological concepts emerge across protein and DNA language models" (shared features too weak at 8%).

### GPT Pro's priority ordering (agreed)

```
1. Fix analyze_crosscoder.py stats ← DONE
2. Run feature atlas + gene modality + ClinVar FDR pathogenicity
3. Run Control A (inference-time shuffle)
4. Run Control B (train-on-shuffled-pairs retrain)
5. Run 3-seed stability
6. Run DNA whitening ablation + k=64
7. Then consider 8192 features + full ClinVar
8. Clean up README / ARCHITECTURE_REVIEW docs
```

### Files changed

- `scripts/analyze_crosscoder.py` — complete rewrite with all 5 fixes
- `scripts/crosscoder_sae.py` — added `--shuffle_pairs` and `--seed` flags for Control B

---

## 1. Where We Are Now

### Architecture Evolution (chronological)

| Version | Commit | Architecture | Result | Problem |
|---------|--------|-------------|--------|---------|
| Two-Tower SAE | `f44efc5` | Separate protein SAE + DNA SAE + residual cross SAE | DMS mean ρ=0.548 (MLP), cross features net-negative | Cross features useless; two towers don't share features |
| Raw CrossCoder v1 | `30a326e` | Single shared encoder on raw ESM2(1280)+Evo2(4096) | 97% dead DNA features | Dimension mismatch kills DNA tower |
| PCA CrossCoder | `d70d577` | PCA equalize → single shared encoder | Better alive ratio, but DMS dropped to 0.38 | PCA loses info vs raw projections |
| **GPT Pro v1**: Three-encoder | `a2aea03` | 3 independent encoders (pair/prot/dna) + activation typing | 75% dead, **0 shared features** | Independent encoders fragment feature space |
| **GPT Pro v2**: Factorized SharedIndex | `1559d2b` | `Wp·prot + Wd·dna + b`, two-axis typing | **Current best** | See results below |
| + ClinVar data | `8fd5866` | Same arch, +8.8K ClinVar variants | 250K pretrain total | |
| + Results | `4c75228` | Final training + full eval | 0.924 AUROC, 3864 alive | |
| + Interpretability pipeline | `ec812d0` | Feature atlas, modality profiles, ClinVar pathogenicity, shuffle control | Analysis ready | |

### Current Model: Factorized SharedIndex CrossCoder

```
Input: PCA(ESM2, 1280→768) + PCA(Evo2, 4096→512, whitened) = 1280-d
Encoder: z = TopK(Wp·prot + Wd·dna + b, k=32)   # 4096 features, shared index
Decoder: prot_hat = Dp·z,  dna_hat = Dd·z
Loss: MSE(prot_hat, prot) + MSE(dna_hat, dna)
Feature typing (post-hoc, two-axis):
  - Trigger-side: activation mass from prot-only encode (Wp·prot+b) vs dna-only (Wd·dna+b)
  - Decoder-side: ||Dp[j]|| vs ||Dd[j]|| norm ratio
  → Categories: protein-private (PP), DNA-private (DP), shared (SH), dead
```

Params: 10.5M | Training: 300 epochs, 1.7 min on single GPU

### Current Results

**Feature distribution:**
| Category | Count | % |
|----------|-------|---|
| Protein-private | 737 | 18% |
| DNA-private | 2781 | 68% |
| Shared | 346 | 8% |
| Dead | 232 | 6% |

**DMS fitness prediction (215 assays, per-assay Ridge, Spearman ρ):**
| Representation | Mean | Median |
|----------------|------|--------|
| Raw ESM-2 (1280) | 0.535 | 0.532 |
| CrossCoder z (all) | 0.524 | 0.528 |
| PCA concat (1280) | 0.522 | 0.506 |
| PP only | 0.493 | 0.500 |
| DP only | 0.358 | 0.367 |
| SH only | 0.307 | 0.295 |

**Modality importance (DMS, Δ = all − no_type):**
| Type | Mean Δ | Median Δ | #assays Δ>0 |
|------|--------|----------|-------------|
| PP | **+0.156** | +0.142 | 206/215 |
| DP | +0.029 | +0.040 | 148/215 |
| SH | +0.007 | +0.005 | 175/215 |

**ClinVar pathogenicity (logistic regression on sparse features, AUROC):**
| Features | Random 5-fold | Gene-held-out |
|----------|--------------|---------------|
| PCA protein | 0.852 | 0.854 |
| PCA DNA | 0.890 | 0.892 |
| PCA concat | 0.913 | 0.913 |
| **CC all** | **0.923** | **0.924** |
| CC prot-priv | 0.900 | 0.899 |
| CC dna-priv | 0.866 | 0.866 |
| CC shared | 0.587 | 0.578 |

**Two-Tower pretrain comparison (215 assays):**
- `two_tower_pretrain_comparison.csv`: 14 methods compared (tt_mlp, tt_ridge, prot_ridge, dna_ridge, concat_ridge, concat+hf_ridge, human_ridge, prot_mlp, concat_mlp, concat+hf_mlp, gate weights)
- `two_tower_pretrain_ablation.csv`: CrossCoder ablation (full, prot-only, dna-only, cross-only, no-prot, no-dna, no-cross, importance per type)
- Both CSVs have 215 rows (one per assay)

---

## 2. GPT Pro Previous Suggestions & Our Responses

### Round 1: Architecture Review (`dba1054`)
**GPT Pro said**: Use CrossCoder with shared dictionary — features classified by decoder norm ratio.
**We did**: Implemented single shared encoder (`9502844`). Worked well (0.919 AUROC, 0 dead).

### Round 1b: Feature typing concern
**GPT Pro said**: Decoder-norm-only classification is not rigorous enough. Need activation-based typing with separate encode paths.
**We did**: Built three-encoder version (`a2aea03`). **FAILED** — 75% dead features, 0 shared features.
**Root cause**: Independent encoders fragment the feature space (documented in `ISSUES.md`, `c498d36`).

### Round 2: Fix fragmentation (`1559d2b`)
**GPT Pro said** (via ISSUES.md options): Use factorized shared-index encoder (Option D hybrid approach).
**We did**: Implemented `Wp·prot + Wd·dna + b` with two-axis classification. **SUCCESS** — 3864 alive, meaningful PP/DP/SH split.

### Round 2b: Add ClinVar
**GPT Pro implied**: DMS-only pretrain is limited (100 proteins); ClinVar adds genome-wide diversity.
**We did**: Extracted ClinVar ESM-2 edelta (`fbf870d`), merged 8.8K ClinVar variants → 250K total pretrain. ClinVar AUROC = 0.924.

### Not yet addressed:
- No formal FDR/enrichment statistics on feature typing
- No seed stability (only 1 SAE seed)
- No PCA/NMF/ICA baseline comparison for decomposition
- No modality occlusion/causal intervention validation
- No structural mapping of identified features

---

## 3. Interpretability Pipeline (latest, `ec812d0`)

Four analysis modules built and ready to run:

### 3a. Feature Atlas
Per-feature: top activating variants, amino acid enrichment (which ref→alt substitutions), gene enrichment, DMS correlation per feature.

### 3b. Gene-Level Modality Profiles
Per-assay: what fraction of total activation mass comes from PP/DP/SH features. Shows which DMS assays (genes) are protein-driven vs DNA-driven.

### 3c. ClinVar Feature Pathogenicity
Per-feature: odds ratio for pathogenic variants among its top activators. Identifies features that are "pathogenic concept detectors."

### 3d. Shuffled-Pair Control
Randomly permute ESM-Evo pairings (break the biological correspondence), retrain CrossCoder on shuffled pairs. If shared features disappear and modality decomposition degrades → evidence that the structure is biologically meaningful, not a training artifact.

**Status**: Pipeline code written, not yet run on HPC. Needs GPU.

---

## 4. New/Modified Files Since Last GPT Pro Review

### Untracked scripts (need commit)

| File | Purpose |
|------|---------|
| `two_tower_sae.py` | Old Two-Tower architecture (Phase A pretrain + Phase B supervised). Superseded by CrossCoder but kept for comparison |
| `benchmark_dms.py` (modified) | DMS benchmark: ESM-2 vs Evo2 vs CLIP-fusion per-assay evaluation |
| `sae_interpretability_controls.py` (modified) | FDR enrichment, seed stability, PCA/NMF baselines, causal intervention, modality occlusion |
| `sae_cbm_benchmark.py` | Concept Bottleneck Model benchmark: SAE concepts → MLP head, compare with raw embedding baselines |
| `sae_e2e_finetune.py` | End-to-end joint SAE + prediction: `score = w·z + b`, fully interpretable linear scoring |
| `sae_interpretable_predict.py` | Per-variant mechanism decomposition: PROT/DNA/CROSS contribution + biochemical mechanism labels |
| `sae_interpretable_v2.py` | V2: modality-first decomposition (Level 1: modality %, Level 2: specific mechanism) |
| `sae_mechanism_analysis.py` | Deep mechanistic analysis: substitution matrices, biochemical property encoding, codon-level signals |
| `sae_modality_deepdive.py` | Per-gene modality profiles, discordant variants, position-level modality maps |
| `sae_retrain_predict.py` | SAE with joint reconstruction + prediction loss |
| `sae_selfexplain_hybrid.py` | Hybrid: SAE concepts + human concepts (BLOSUM62, Grantham) → linear prediction |
| `sae_selfexplain_model.py` | Pure self-explanatory: `score = Σ(c_i × w_i) + b` from SAE concepts only |
| `sae_topk_sweep.py` | TopK sweep: re-encode at k=16,32,64,128,256,512 and benchmark interpretability-performance tradeoff |
| `dms_extract_esm2.py` | Step 1: Extract ESM-2 edelta for all ProteinGym assays |
| `dms_extract_evo2.py` | Step 2: Extract Evo2 edelta (CDS-mapped codon changes) |
| `dms_evaluate.py` | Step 3: Run DMS benchmark on saved embeddings |
| `dms_finetune.py` | Fine-tune projection + MLP on pooled DMS data |
| `fetch_cds_proteingym.py` | Fetch CDS sequences from UniProt/EMBL for ProteinGym proteins |
| `fetch_cds_v2.py` / `fetch_cds_v3.py` | Improved CDS fetching (RefSeq, NCBI, Ensembl fallbacks) |

### Result files

| File | Content |
|------|---------|
| `results/crosscoder_sae/results_summary.json` | Full CrossCoder results: feature counts, DMS Spearman, ClinVar AUROC, ablation |
| `results/crosscoder_sae/ablation.csv` | Per-assay CrossCoder ablation (215 rows × 11 cols) |
| `results/crosscoder_sae/config.json` | CrossCoder hyperparams |
| `results/crosscoder_sae/run_log.txt` | Full training + eval log |
| `results/two_tower_pretrain_comparison.csv` | 14-method comparison across 215 assays |
| `results/two_tower_pretrain_ablation.csv` | Two-Tower modality ablation (215 rows) |
| `results/two_tower_pretrain_config.json` | Two-Tower hyperparams |

### Archive
`_archive_codex/` — ~85 scripts from the previous v6/ClinVar-focused narrative. Kept for reference but not part of current CrossCoder work.

### Survey data
`survey/arch_eval/`, `survey/interpretability/`, `survey/sae_scaling/` — Literature search JSONs (arxiv, semantic scholar) on SAE architectures, interpretability methods, and scaling.

---

## 5. Key Observations & Honest Assessment

### What works
1. **CrossCoder architecture is stable**: 94% alive features, clean PP/DP/SH decomposition
2. **ClinVar AUROC = 0.924**: CrossCoder sparse features beat PCA concat (0.913) by 1.1%
3. **Modality decomposition is real**: PP importance +0.156, DP +0.029 — protein dominates but DNA contributes independently in 69% of assays
4. **Gene-held-out = random**: No gene-level overfitting (0.924 vs 0.923)

### What doesn't work / concerns
1. **DMS performance doesn't exceed raw ESM-2**: CrossCoder z (0.524) < raw ESM (0.535). The SAE is not a better representation for prediction.
2. **Shared features are weak**: Only 346/4096 (8%), and SH importance = +0.007 (negligible). The "cross-modal" story is thin if shared features don't matter.
3. **DNA-private features dominate count (68%) but not importance**: 2781 DP features but only +0.029 importance. Most DP features may be noise.
4. **ClinVar sample is small and skewed**: Only 8.8K variants (8068 pathogenic, 763 benign). Extreme class imbalance. Need more ClinVar data.
5. **No controls run yet**: The interpretability pipeline is built but not executed. No FDR, no seed stability, no shuffled-pair control.

---

## 6. Open Questions for GPT Pro

### Q1: Paper narrative — what story can we tell?

The original plan was "cross-modal SAE reveals shared biological concepts between protein and DNA." But shared features are weak (8%, negligible importance). The actual finding is:
- **Protein features dominate** (PP: 18% of features but +0.156 importance)
- **DNA features are numerous but individually weak** (DP: 68% but +0.029)
- **The decomposition itself is the contribution** — we can say per-variant, per-gene "how much of this effect is protein-driven vs DNA-driven"

Is this enough for a Nature Communications paper? Or do we need a stronger cross-modal signal?

### Q2: Why so many DNA-private features?

2781 DP vs 737 PP is a 3.8:1 ratio. Possible explanations:
- Evo2 edelta has higher intrinsic dimensionality (needs more features to represent)
- DNA-level signal is more diffuse / less structured than protein-level
- PCA whitening of DNA inflates minor PCs

Which hypothesis should we test? Would equalizing feature counts (e.g., balanced initialization or per-modality TopK) help?

### Q3: Scaling the CrossCoder

Current: 4096 features, k=32, 250K variants. Should we:
- **Scale up features** (8192, 16384) to increase capacity?
- **Scale up data** (add full ClinVar ~300K, more DMS)?
- **Scale up k** (64, 128) for better reconstruction?
- **All of the above?**

What does the literature suggest for SAE scaling laws?

### Q4: The prediction gap

CrossCoder z (0.524) < raw ESM (0.535) on DMS. Two explanations:
- **Information loss**: TopK sparsity + PCA discards information → lower prediction
- **Representation mismatch**: CrossCoder is optimized for reconstruction, not prediction

Should we:
a) Accept the gap (interpretability paper, not prediction paper)
b) Try end-to-end fine-tuning (sae_e2e_finetune.py is ready)
c) Try larger k / expansion ratio
d) Something else?

### Q5: Priority next experiments

Given limited GPU time, what should we run first?
- [ ] Interpretability pipeline (feature atlas + modality profiles + ClinVar pathogenicity)
- [ ] Shuffled-pair control
- [ ] Seed stability (5 random seeds)
- [ ] PCA/NMF/ICA baseline comparison
- [ ] Larger CrossCoder (8192 features)
- [ ] More ClinVar data (genome-wide extraction)
- [ ] End-to-end fine-tuning experiment

### Q6: Relationship to v6 findings

The v6 narrative (previous work) had strong interpretability results:
- Ion channel pore-region features (F10750)
- RAS switch region feature (F11728)
- MSH2 domain-level modality decomposition
- 0.9629 AUROC on ClinVar

Those were on a **different architecture** (v6 projection + 12288-feature SAE on 768+768 concatenated representations). Are any of those findings transferable to the CrossCoder setting? Should we try to replicate them with the new architecture?

### Q7: Comparison with v6 as a baseline

v6 SAE had:
- 12288 features (vs our 4096)
- Trained on 768+768 projected space (vs PCA 768+512)
- ClinVar AUROC 0.962 (vs our 0.924)
- Rich interpretability (ion channels, RAS, etc.)

Should the CrossCoder paper include v6 as a **baseline** to show that the CrossCoder's modality decomposition provides additional interpretability value beyond what a standard SAE on concatenated features gives?

---

## 7. Suggested Next Steps (our opinion, pending GPT Pro feedback)

1. **Run interpretability pipeline** on current CrossCoder (ec812d0 code is ready)
2. **Run shuffled-pair control** to validate modality decomposition is non-trivial
3. **Compare CrossCoder features with v6 SAE features** — do we recover the same ion channel / RAS concepts?
4. **Scale to 8192 features** with more ClinVar data
5. **Write up modality decomposition as the core contribution**, not prediction improvement
