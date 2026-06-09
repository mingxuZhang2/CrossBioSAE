# Cross-Modal SAE Architecture Review

## 1. Research Goal

**Paper type**: Interpretability paper (NOT a prediction paper)
**Target venues**: Nature Communications, Genome Biology

**Core claim**: A cross-modal SAE trained on protein-LM (ESM-2) and DNA-LM (Evo-2) variant embeddings can decompose variant effects into interpretable protein-driven, DNA-driven, and cross-modal concepts. Different genes have different modality profiles (e.g., MSH2 = DNA-conservation-driven, RB1 = protein-damage-driven).

**What we are NOT trying to do**: Beat SOTA prediction (EVEE AUC=0.997 on ClinVar). We offer mechanistic decomposition that single-model approaches cannot provide.

---

## 2. Current Architecture

### Input Representations

For each missense variant, we compute **edelta** = embedding(mutant) - embedding(wildtype) at the mutation position:
- **ESM-2 edelta**: 1280-dimensional (protein language model)
- **Evo-2 edelta**: 4096-dimensional (DNA language model)
- **Human features**: 9-dimensional (charge change, volume change, hydrophobicity change, Cys loss, Pro gain/loss, Gly loss, Trp loss, charge reversal)

### Preprocessing

1. **Standardize**: Per-dimension zero mean + unit variance
2. **PCA reduce**: Both modalities → 512-d (equalizes asymmetric input dimensions)
   - ESM-2: 1280 → 512 (retains 92% variance)
   - Evo-2: 4096 → 512 (retains 100% variance — Evo-2 edelta lives in <512-d subspace)

### Model: Two-Tower SAE with Cross-Prediction

```
ESM-2 edelta (1280) → standardize → PCA(512) → SAE_prot (512→2048, TopK k=32) → prot_z (sparse)
Evo-2 edelta (4096) → standardize → PCA(512) → SAE_dna  (512→2048, TopK k=32) → dna_z  (sparse)

Cross-prediction scaffolding (Phase A only, discarded after):
  prot_z → Linear(2048→512) → predict DNA PCA embedding
  dna_z  → Linear(2048→512) → predict Protein PCA embedding

Projection to common space:
  prot_z → proj_prot (2048→256, LayerNorm) → prot_c (dense, 256-d)
  dna_z  → proj_dna  (2048→256, LayerNorm) → dna_c  (dense, 256-d)

Residual cross-modal features:
  residual = prot_c - dna_c  (captures where modalities DISAGREE)
  residual → SAE_cross (256→1024, TopK k=16) → cross_z (sparse)

Downstream evaluation features:
  concat(prot_c, dna_c, cross_z, human_features) → 256+256+1024+9 = 1545-d
```

**Total parameters**: 7.9M

### Training Procedure

**Phase A: Joint Cross-Predictive Pretraining (unsupervised)**
```
Loss = L_recon_prot + L_recon_dna
     + 0.3 × L_cross_predict    (each tower predicts other modality's PCA embedding)
     + 0.03 × L_InfoNCE          (contrastive alignment in projection space)
```
- Trains: SAE_prot, SAE_dna, proj_prot, proj_dna, xpred_p2d, xpred_d2p
- Epochs: 300, cosine LR with warmup, batch=8192

**Phase A2: Cross SAE on Residual (frozen towers)**
- Input: proj_prot(prot_z) - proj_dna(dna_z) 
- Loss: MSE reconstruction of residual
- Epochs: 100

**Phase B: Supervised Evaluation (freeze all SAEs)**
- Per-assay 5-fold CV with MLP or Ridge head
- Ablation: train on feature subsets to measure modality importance

### Design Rationale

1. **Cross-prediction loss**: Forces each tower to encode information about the OTHER modality. Without it, towers learn independently and cross-modal features are meaningless.
2. **InfoNCE (not cosine)**: Pure cosine alignment collapsed to cos=1.0 (trivial constant output). InfoNCE pushes matched pairs close AND unmatched pairs apart, preventing collapse.
3. **Residual cross SAE**: Cross features = prot_c - dna_c, so they literally capture where protein and DNA signals disagree. Positive residual = excess protein signal; negative = excess DNA signal.
4. **PCA equalization**: ESM-2 (1280-d) and Evo-2 (4096-d) are asymmetric. Without PCA, DNA SAE had 97% dead features because 4096-d is too hard. PCA to 512-d makes both towers symmetric.
5. **Projected features for evaluation**: Raw SAE codes are 2048+2048+1024=5120-d with ~70% dead. Projected features (256+256+1024+9=1545-d) are denser and more suitable for Ridge/MLP on small assays (median 544 variants).

---

## 3. Pretraining Data

### Current (Pilot)

| Source | Variants | Modalities | Status |
|--------|----------|------------|--------|
| ProteinGym DMS | 241,533 | ESM-2(1280) + Evo-2(4096) edelta | Available, used |

- 215 assays (DMS experiments), ~100 unique proteins
- Per-variant: alt_embedding - ref_embedding at mutation position
- Only variants with valid Evo-2 CDS mapping are included (696K ESM-2 total, 241K matched)

### Planned Expansion

| Source | Est. Variants | Priority | Rationale |
|--------|--------------|----------|-----------|
| ClinVar missense | ~300K | HIGH | Different distribution (clinical vs experimental), enriched pathogenics |
| ProteinGym full catalog | ~2.7M | MEDIUM | More assay diversity, diminishing returns |
| gnomAD common | ~5M+ | LOW | Overwhelmingly neutral, dilutes signal |

**Target**: ~1M diverse variants (DMS + ClinVar) before final training.

**Scaling rationale**: 2048+2048+1024 = 5120 total SAE features. Literature suggests 2K-30K samples per feature. 241K / 5120 = 47 samples/feature (marginal). 1M / 5120 = 195 samples/feature (adequate).

---

## 4. Downstream Benchmarks (5 tasks)

### Task 1: DMS Fitness Prediction
- **Data**: ProteinGym 215 assays, per-assay Spearman correlation
- **Metric**: Mean/median Spearman across assays, win rate vs baselines
- **Baselines**: Raw ESM-2 Ridge, raw Evo-2 Ridge, concat Ridge, MLP variants
- **Status**: Implemented, pilot results available

### Task 2: ClinVar Pathogenicity Classification
- **Data**: ClinVar pathogenic vs benign missense variants (~300K)
- **Metric**: AUROC, AUPRC
- **Baselines**: ESM-2 LLR, Evo-2 LLR, AlphaMissense, CADD
- **Status**: Need to extract embeddings

### Task 3: Protein Stability (ΔΔG) Prediction
- **Data**: Megascale/ProTherm thermodynamic stability data
- **Metric**: Spearman correlation with experimental ΔΔG
- **Baselines**: ESM-2 features, Rosetta, FoldX
- **Status**: Not started

### Task 4: VUS Mechanism Classification
- **Data**: ClinVar VUS with expert-curated mechanism labels
- **Metric**: Can SAE features distinguish protein-damage vs DNA-conservation driven VUS
- **Expected result**: Different genes activate different feature subsets → mechanism profiles
- **Status**: Not started

### Task 5: Drug Resistance Prediction
- **Data**: DMS assays for drug-target proteins (e.g., HIV protease, kinases)
- **Metric**: Spearman for resistance-associated functional scores
- **Status**: Subset of ProteinGym DMS

---

## 5. Pilot Results (on 241K DMS variants)

### Pretraining Metrics (Phase A, 300 epochs)

| Metric | Value | Meaning |
|--------|-------|---------|
| R² prot→dna | 0.718 | Protein features predict 72% of DNA signal |
| R² dna→prot | 0.526 | DNA features predict 53% of protein signal |
| R@1 (InfoNCE) | 14.5% | Cross-modal retrieval accuracy |
| Alive prot | 774/2048 (38%) | Protein SAE feature utilization |
| Alive dna | 249/2048 (12%) | DNA SAE feature utilization |
| Alive cross | 494/1024 (48%) | Cross SAE feature utilization |
| Training time | 2.3 min | Single GPU, 8192 batch |

### DMS Evaluation (per-assay 5-fold CV)

**Ablation (Ridge on projected features, 215 assays):**

| Mode | Mean Spearman |
|------|--------------|
| full (prot+dna+cross+human) | 0.379 |
| prot_only (prot+human) | 0.345 |
| dna_only (dna+human) | 0.304 |
| cross_only (cross+human) | ~0.30 |

**Modality importance (Δ = full - no_modality):**

| Modality | Mean Δ | >0 assays |
|----------|--------|-----------|
| PROT | +0.044 | majority |
| DNA | +0.005 | ~half |
| CROSS | -0.024 | minority |

**Previous v6 pipeline baseline (for reference):**
- v6 ensemble (ESM-2+Evo-2, 768-d projected, supervised): AUC=0.9629 on ClinVar
- Raw ESM-2 Ridge on DMS: ~0.51 mean Spearman
- Previous Two-Tower (768-d projected, no PCA): tt_mlp=0.548

---

## 6. Current Problems

### Problem 1: DNA SAE still has low alive features (12% vs prot 38%)
Even after PCA equalization to same 512-d, DNA SAE utilizes only 249/2048 features vs protein's 774/2048. Possible causes:
- DNA edelta signal is inherently lower-dimensional (PCA retains 100% at 512-d, meaning original 4096-d was mostly empty)
- Evo-2 may encode variant effects in fewer dimensions than ESM-2

### Problem 2: Overall performance dropped after PCA
- PCA version: full=0.379 (Ridge on 1545-d projected features)
- Non-PCA version: full=0.505 (Ridge on 15360-d sparse codes, but 97% dead)
- Previous Two-Tower (768-d projected): tt_mlp=0.548

The PCA + projection pipeline loses information. Possible causes:
- PCA 512-d may lose critical signal (especially for protein: only 92% variance)
- Projection from 2048 sparse to 256 dense is too aggressive a bottleneck
- The previous 768-d projections (from v6 pipeline) were specifically trained for variant prediction

### Problem 3: Cross-modal features are net-negative
imp_cross = -0.024 on average — removing cross features slightly IMPROVES performance. This is consistent across all architecture versions. Possible causes:
- Cross SAE on residual (prot_c - dna_c) captures noise, not signal
- With only 241K variants from ~100 proteins, within-assay cross-modal variation is limited
- The genuine cross-modal signal may only emerge with more diverse data (ClinVar)

### Problem 4: R@1 retrieval is low (14.5%)
InfoNCE retrieval accuracy dropped from 63% (non-PCA version) to 14.5% (PCA version). The projection space is not learning good alignment. May need larger proj_dim or stronger InfoNCE weight.

### Problem 5: Data limitation
241K variants from ~100 proteins. Each DMS assay measures one protein — cross-modal variation within an assay is structurally limited. The real cross-modal biology (MSH2 DNA-contact vs structural domains) requires comparing across many genes, which ClinVar would provide.

---

## 7. Questions for Review

1. **Is PCA equalization the right approach?** Or should we handle the 1280 vs 4096 asymmetry differently (e.g., separate adapters, different expansion ratios)?

2. **Is the cross-prediction training objective appropriate?** The R² values show it works (0.72 prot→dna), but the learned features don't help downstream prediction. Is there a better auxiliary loss?

3. **Should the cross SAE operate on residual (prot_c - dna_c)?** Or is there a better way to capture cross-modal interaction (e.g., element-wise product, gated attention, bilinear)?

4. **Is 241K DMS sufficient for pilot validation?** Or should we prioritize extracting ClinVar embeddings before further architecture iteration?

5. **For the paper narrative**: Given that protein consistently dominates, is "mechanistic decomposition" still viable? Or should we reframe as "protein-LM interpretability with DNA as auxiliary signal"?

6. **Architecture simplification**: With 7.9M params and 2.3 min training, the model may be too small. Should we increase PCA dim (512→768?), expansion (4x→8x?), or proj_dim (256→512?)?
