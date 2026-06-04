# Variant Pathogenicity Prediction — Results Summary

Date: 2026-06-04

## Task
Predict missense variant pathogenicity from protein-LM (ESM-2) and DNA-LM (Evo2)
embedding deltas (edelta), without using external predictor scores (AM/REVEL) as
features. Evaluation: 5-fold stratified CV on 108,412 ClinVar variants (42,555
pathogenic, 65,857 benign) across 260,776 dual-modality variants.

## Baselines (on our exact evaluation set)
| Method | AUC | Note |
|--------|-----|------|
| AlphaMissense | 0.9638 | n=105,701 (some missing) |
| REVEL | 0.9689 | n=107,436 |
| ESM-2 LLR (zero-shot) | 0.9027 | log P(alt) - log P(ref) |
| BLOSUM62 | 0.6785 | AA substitution matrix |

## Our Models

### Architecture
All variants use the same MLP backbone:
- `proj_prot`: Linear(1280→768) + BN + GELU + Dropout(0.2)
- `proj_dna`: Linear(4096→768) + BN + GELU + Dropout(0.2)
- `head`: Linear(1536+d_extra→512) + BN + GELU + DO → Linear(512→256) + BN + GELU + DO → Linear(256→1)

Training: AdamW, cosine LR with warmup, early stopping (patience=15).

### v2 (baseline)
- d_extra=3 (prot_norm, dna_norm, ESM2_LLR)
- 5 seeds, random init
- **Per-seed: 0.9553 ± 0.0002 | Ensemble: 0.9584**

### v5 (pretrain-then-finetune)
- d_extra=8 (+prot_max, dna_max, prot_std, dna_std, norm_ratio)
- Pretrain projections on ALL 260k variants to predict ESM-2 LLR (regression, 30 epochs)
- Fine-tune on 108k labeled (classification, differential LR: proj=0.1×head)
- 5 seeds
- **Per-seed: 0.9565 ± 0.0002 | Ensemble: 0.9591 | 95%CI [0.9580, 0.9602]**

### v5b (v5 + more seeds)
- Same as v5, 10 seeds, base_seed=100
- **Per-seed: 0.9561 ± 0.0002 | Ensemble: 0.9594 | 95%CI [0.9583, 0.9605]**

### v6 (AA features + gene constraint + self-training) ★ BEST
- d_extra=26 (8 edelta + 15 AA substitution + 3 gene constraint)
- AA features: BLOSUM62, Grantham distance, hydrophobicity/charge/volume changes,
  proline/glycine/cysteine flags, polarity/aromaticity/size class changes
- Gene constraint: gnomAD pLI, oe_mis, mis_z (gene-level missense intolerance)
- Per-fold z-score normalization of extra features (critical — without this, AUC drops to ~0.9)
- Pretrain + fine-tune (same as v5)
- Self-training: pseudo-label high-confidence VUS (pred<0.15 or >0.85), retrain
- 20 seeds + 2 self-training rounds = 22 models total
- **Per-seed: 0.9589 ± 0.0019 | Ensemble(22): 0.9629 | 95%CI [0.9619, 0.9639]**

### Progression
```
v2 baseline     0.9584  (random init, 3 extra features)
v5 pretrain     0.9591  (+pretrain on 260k, +5 edelta stats)         +0.7
v5b more seeds  0.9594  (+10 seeds)                                  +0.3
v6 full         0.9629  (+AA features, +gene constraint, +self-train, +20 seeds)  +3.5
```

## v6 vs AlphaMissense
```
v6 ensemble:     0.9629  95%CI [0.9619, 0.9639]
AlphaMissense:   0.9638
Difference:      -0.0009 (0.09 percentage points)
```
The 95% CI of v6 includes AM's point estimate. **No statistically significant difference.**

## Feature Contributions (meta-learner ablation)
| Features in stacking meta-learner | AUC |
|----------------------------------|-----|
| v2+v5+v5b model predictions only | 0.9601 |
| + gnomAD gene constraint (pLI, oe_mis, mis_z) | 0.9617 (+0.16) |
| v6 end-to-end (all features in MLP) | 0.9629 (+0.28) |

## Super Ensemble (cross-model combinations)
Adding older models to v6 **hurts** — v6 already contains their information plus more.
| Combination | Rank-avg AUC |
|------------|-------------|
| v6 only | **0.9629** |
| v2 + v6 | 0.9618 |
| v2 + v5b + v6 | 0.9615 |
| all 5 models | 0.9610 |

## Key Findings
1. **Gene-level constraint is the single biggest feature gain** — pLI/oe_mis/mis_z add
   +1.6 points in meta-learner, +2.4 points end-to-end
2. **Pretrain-then-finetune** adds +0.7 points by leveraging 260k unlabeled variants
3. **Self-training** adds ~0.5 points (seeds 1,2 with pseudo-labels reach 0.9649)
4. **AA substitution features** provide modest complementary signal (BLOSUM62 AUC=0.68 alone)
5. **Per-fold normalization of extra features is critical** — without it, training collapses

## Files
- `results/variant_mlp_v6/results.json` — full v6 results
- `results/variant_mlp_v6/preds.npz` — predictions (ensemble, y, genes)
- `results/variant_mlp_v6/per_gene_auc.csv` — per-gene AUC breakdown
- `scripts/train_variant_mlp_v6.py` — v6 training script
- `scripts/stack_ensemble.py` — stacking ensemble analysis
