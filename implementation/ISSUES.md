# CrossCoder SAE — Current Issues (2026-06-09)

## Context

Following GPT Pro's review, we upgraded from single-encoder CrossCoder to three-encoder CrossCoder with activation-based feature typing. The architecture now has:

```
encoder_pair(concat(prot, dna)) → z_pair   (paired input)
encoder_prot(prot)              → z_prot   (protein-only input)
encoder_dna(dna)                → z_dna    (DNA-only input)

decoder_prot(z) → reconstruct protein PCA repr
decoder_dna(z)  → reconstruct DNA PCA repr
```

Training loss:
```
L = MSE(decoder_prot(z_pair), prot) + MSE(decoder_dna(z_pair), dna)
  + 0.5 * MSE(decoder_prot(z_prot), prot)
  + 0.5 * MSE(decoder_dna(z_dna), dna)
```

Feature classification (activation-based):
- protein-private: active in z_prot, not in z_dna
- DNA-private: active in z_dna, not in z_prot  
- shared: active in both z_prot and z_dna
- interaction: active in z_pair but not in z_prot or z_dna

## Problem: Three Independent Encoders Fragment the Feature Space

### Symptoms

| Metric | Single-encoder (v1) | Three-encoder (v2) |
|--------|--------------------|--------------------|
| Alive features | 4096/4096 (100%) | 1010/4096 (25%) |
| Dead features | 0 | 3086 (75%) |
| Prot-private | 638 | 145 |
| DNA-private | 2109 | 119 |
| Shared | 1349 | **0** |
| Interaction | 0 | 746 |
| DMS cc_z Spearman | 0.524 | 0.515 |
| ClinVar cc_all AUROC | 0.919 | 0.903 |
| imp_prot (DMS) | +0.124 | -0.000 |
| imp_dna (DMS) | +0.020 | +0.021 |

### Root Cause

The three encoders are completely independent `nn.Linear` layers. They learn separate feature subspaces:
- `encoder_pair` activates features 0-1000 (roughly)
- `encoder_prot` activates features 1000-1500
- `encoder_dna` activates features 1500-2000

Since there's no weight sharing, the same biological concept gets encoded as different features in different encoders. The activation-based classification then finds:
- **0 shared features** — because a feature active in z_prot is never active in z_dna (different encoder weights)
- **746 interaction features** — features active in z_pair but not in either single-modality encoder (because encoder_pair has its own weights)
- **75% dead** — each encoder only uses ~25% of features, and the union still leaves most features unused

### Why This Matters

1. The paper's core claim requires meaningful "shared" features — concepts that both protein and DNA representations agree on. With 0 shared features, this claim is unsupported.
2. The "interaction" category (746 features) is actually what the single-encoder version called "shared" — it's an artifact of independent encoders, not true cross-modal interaction.
3. Performance dropped (0.919→0.903 AUROC, 0.524→0.515 Spearman) because the model has 3x more encoder parameters but the same data, leading to under-training.

## What Needs to Be Fixed

### The Goal
We need feature typing that is:
1. **Scientifically valid** — based on what information triggers the feature, not just decoder norms
2. **Non-degenerate** — produces meaningful counts for all categories (PP, DP, shared, interaction)
3. **Compatible with good reconstruction** — doesn't sacrifice model quality for classification rigor

### Possible Approaches

**Option A: Shared-trunk encoder with modality-specific heads**
```
trunk = Linear(d_prot + d_dna, n_features)  # shared weights
head_prot = Linear(d_prot, n_features)       # additive residual
head_dna = Linear(d_dna, n_features)         # additive residual

z_pair = TopK(trunk(concat(prot, dna)))
z_prot = TopK(trunk(concat(prot, zeros)) + head_prot(prot))
z_dna  = TopK(trunk(concat(zeros, dna)) + head_dna(dna))
```
Pro: Features share the trunk, so the same feature CAN activate for both modalities.
Con: Zero-padding the missing modality may not be principled.

**Option B: Single encoder + zero-masking probe (post-hoc)**
Keep the single-encoder architecture (which works well). After training, probe feature types by:
```
z_pair = encode(prot, dna)           # normal
z_prot_only = encode(prot, zeros)    # mask DNA
z_dna_only = encode(zeros, dna)      # mask protein
```
Then classify features by comparing activation patterns across these three forward passes.
Pro: No architecture change, no retraining needed. Single encoder already works.
Con: Zero-input may not be the right "null" for PCA features; mean-input might be better.

**Option C: Gradient-based attribution (post-hoc)**
Keep single encoder. For each feature j, compute:
```
grad_prot_j = d(z_j) / d(prot_input)  → protein sensitivity
grad_dna_j  = d(z_j) / d(dna_input)   → DNA sensitivity
```
Features that respond to protein input = protein-driven. Features that respond to DNA = DNA-driven.
Pro: No architecture change. Gradient attribution is standard in interpretability.
Con: Gradients can be noisy; may need integrated gradients or smoothing.

**Option D: Decoder-norm + activation-frequency hybrid**
Keep single encoder. Use:
1. Decoder norms (as before) for reconstruction-side modality
2. Input-masking activation frequency for trigger-side modality
3. Classify by combining both signals
Pro: Uses information from both sides of the autoencoder.
Con: Still needs the zero/mean-masking question resolved.

## Questions for GPT Pro

1. Which approach (A/B/C/D) is most defensible for a Nature Communications paper?
2. For Option B, should we use zero-masking or mean-masking as the null input?
3. Is there a better way to share weights across encode paths that avoids the fragmentation problem?
4. Should we also add a consistency loss (e.g., z_pair should be close to z_prot + z_dna in some sense)?
5. The single-encoder version had 0 dead features and strong results (0.919 AUROC, all modalities positive). Is there a way to get rigorous feature typing WITHOUT changing the encoder architecture?

## Reference Results

### Single-Encoder CrossCoder (best results so far)
```
Pretrain: 250K variants (DMS+ClinVar), 300 epochs, 1.7 min
Features: PP=638, DP=2109, SH=1349, dead=0 (decoder-norm classification)
DMS: cc_z=0.524, imp_prot=+0.124, imp_dna=+0.020, imp_shared=+0.021
ClinVar random AUROC: cc_all=0.919, pca_concat=0.913
```

### Three-Encoder CrossCoder (current, broken)
```
Pretrain: 250K variants, 300 epochs, ~2 min
Features: PP=145, DP=119, SH=0, IX=746, dead=3086 (activation classification)
DMS: cc_z=0.515, imp_prot=-0.000, imp_dna=+0.021
ClinVar random AUROC: cc_all=0.903, gene-held-out=0.903
```
