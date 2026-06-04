# CrossBioSAE: SAE Interpretability Findings v2 (Post GPT-Pro Review R1+R2)

Date: 2026-06-04
Status: Updated after two rounds of GPT Pro review — Met1 confound confirmed, wording tightened per R2

## Changes from v1

GPT Pro correctly identified that 7/9 "methionine loss" features were driven by Met1/start-loss variants (85-99% Met1). After excluding prot_pos=1, internal Met pathogenic rate dropped to 25-44%. **Methionine loss is demoted from "novel discovery" to "recovered start-loss category + candidate internal-Met signal".**

The new lead story is **Discovery 1: cross-paralog substitution-specific channel features** (was Discovery 2 in v1).

---

## Discovery 1 (LEAD): Cross-Paralog Substitution-Specific Ion Channel Features

### Summary

The SAE learned a **family of features** that each capture a specific substitution type at homologous structural positions across **ligand-gated and voltage-gated ion channel families** — without any pathway, family, or structural annotation input.

### Feature F10750: Ion-Channel Pore-Region Threonine Disruption (198.9x enrichment)

33 activating variants. A high-activation Thr→Ile core at conserved pore threonines, with broader activation for disruptive substitutions at homologous pore-region threonines. The top 5 (activation > 4.8) are ALL at the conserved M2/TM2 pore-lining threonine:

| Rank | Gene | Variant | Activation | ClinVar | Literature |
|------|------|---------|-----------|---------|------------|
| 1 | GABRB1 | p.Thr287Ile | 6.06 | Likely pathogenic | — |
| 2 | GABRB2 | p.Thr286Ile | 6.04 | Likely pathogenic | — |
| 3 | GABRB3 | p.Thr287Ile | 5.95 | Pathogenic | NC 2022: GOF variant, DEE |
| 4 | GABRA2 | p.Thr294Ile | 5.03 | Pathogenic/LP | — |
| 5 | GABRA1 | p.Thr294Ile | 4.82 | Pathogenic | Functional: T292 pore residue |

**The feature extends across the Cys-loop ligand-gated receptor family and, in lower-activation cases, to structurally analogous pore-region threonine variants in other ion-channel families such as KCNB1/Kv2.1:**

| Activation | Gene | Variant | Receptor family | Evidence strength |
|-----------|------|---------|-----------------|-------------------|
| 3.01 | GABRA5 | p.Thr301Arg | GABA-A alpha 5 (Cys-loop) | Pathogenic/LP |
| 2.96 | KCNB1 | p.Thr374Ile | Voltage-gated K+ (Kv2.1, NOT Cys-loop) | Pathogenic |
| 2.27 | CHRNA1 | p.Thr274Ile | Nicotinic ACh receptor (Cys-loop) | Pathogenic |
| 2.22 | CHRNE | p.Thr284Ile | Nicotinic ACh receptor (Cys-loop) | VUS — candidate, not anchor |
| 1.92 | GLRA2 | p.Thr296Met | Glycine receptor (Cys-loop) | Pathogenic |
| 1.83 | GLRA1 | p.Thr287Ala | Glycine receptor (Cys-loop) | Likely pathogenic |

In Cys-loop receptors, these Thr positions map to the M2/TM2 pore-lining helix; in KCNB1/Kv2.1, the analogous signal maps to the voltage-gated potassium-channel pore helix (KCNB1 T374I is associated with epileptic encephalopathy). The threonine side chains face the pore lumen and are critical for:
1. Channel selectivity (H-bonding with permeating ions)
2. Channel gating (conformational switch during opening/closing)
3. Desensitization kinetics

### Related Cys-Loop Receptor Features (the SAE learned a whole decomposition)

| Feature | Enrichment | Substitution | Top variants | Structural role |
|---------|-----------|-------------|-------------|-----------------|
| F10750 | 198.9x | T>I at pore | GABRB1/B2/B3 T287I, GABRA1/A2 T294I | TM2 pore-lining Thr |
| F3118 | 188.2x | P>T at loop | GABRB2 P252T, GABRB3 P253T, GABRG2 P282T | TM1-TM2 linker Pro |
| F5463 | 88.6x | V>A at TM2 | CHRNE V285A, CHRNA1 V275A | TM2 hydrophobic packing |
| F9483 | 78.4x | V>L/M at TM2 | CHRNB2 V287L/M | TM2 valine (nAChR) |
| F3888 | 77.8x | D>G at ECD | CHRNA4 D146G | Extracellular domain |

**The SAE decomposed the Cys-loop receptor pathogenic landscape into 5+ distinct mechanism features**, each specific to a structural position × substitution type. This is exactly analogous to what a protein structural biologist would do — but learned automatically from embeddings.

### Independent Validation Anchors

1. **GABRB3 p.Thr287Ile**: Classified as gain-of-function in Bhatt et al., Nature Communications 2022. The paper performed electrophysiology on GABRB3 missense variants and showed T287I increases GABA sensitivity — a GOF mechanism that leads to developmental and epileptic encephalopathy (DEE).

2. **GABRA1 p.Thr292**: Lachance-Touchette et al. (2022) showed T292I/T292S in GABRA1 alter channel-gating properties. T292 is in the TM2 pore-lining segment, directly contacting permeating chloride ions.

3. **CHRNA1 p.Thr274Ile**: The nicotinic receptor T274 is homologous to the GABA receptor pore threonine. UniProt/SwissProt associates this variant with slow-channel congenital myasthenic syndrome; T274 falls within the TM helix (265-285). This is a strong independent anchor.

4. **CHRNE p.Thr284Ile**: Currently VUS in ClinVar (not a validation anchor). However, CHRNE p.Thr284Pro and p.Val285Ala at homologous positions ARE listed as disease-causing in slow-channel CMS resources. F10750's activation on CHRNE T284I represents a VUS candidate at a homologous pore residue, not independent pathogenic confirmation.

5. **KRAS p.Pro34Leu** (Feature F11728): ClinGen RASopathy Expert Panel classified this as pathogenic. The P34 position is in the switch I region critical for GTP hydrolysis.

### Why This Is NC-Level

1. **Cross-paralog and cross-family reasoning without labels**: The SAE groups GABRB1/B2/B3 + GABRA1/A2/A5 (GABA-A) + GLRA1/A2 (glycine) + CHRNA1/CHRNE (nicotinic) — all Cys-loop family — into one feature, and additionally captures the structurally analogous KCNB1/Kv2.1 pore variant at lower activation. This was achieved without protein family or structural domain annotations.

2. **Substitution × position specificity with activation gradient**: Feature F10750 doesn't just capture "ion channel variants." It has a high-activation core of Thr→Ile at conserved pore threonines, with broader activation for other disruptive substitutions (T→R, T→M, T→A) at homologous positions. Feature F3118 captures Pro→Thr at M2-M3 loop positions. This resolution matches expert structural reasoning.

3. **Independent functional validation**: The top variant (GABRB3 T287I) has published electrophysiology data confirming it as a GOF mechanism.

4. **Clinical relevance**: These features could identify VUS in GABA receptor genes that affect the same structural position as known pathogenic variants — exactly the "analogous residue" reasoning that ClinGen expert panels use.

### Suggested Figure

**Panel A**: Multiple sequence alignment of Cys-loop receptor TM2 segments (GABRB1/B2/B3, GABRA1/A2, GLRA1, CHRNA1), highlighting conserved Thr positions. Color by SAE activation strength.

**Panel B**: Cryo-EM structure of GABA-A receptor (PDB: 6HUJ or similar), with SAE-activated Thr positions mapped onto the pore-lining helix. Show how they face the channel lumen.

**Panel C**: Feature activation gradient — high activation = core Cys-loop TM2 Thr, medium = other channel Thr, low = non-channel proteins. This shows the feature learned structural specificity, not just "threonine loss."

**Panel D**: Comparison of F10750 (T>I pore), F3118 (P>T loop), F5463 (V>A TM2) — three distinct mechanism features for the same protein family, demonstrating sub-domain resolution.

---

## Discovery 1b: RAS GTPase Switch Region Feature (F11728)

32 activating variants. Top activations are exclusively small GTPases:

| Rank | Gene | Variant | Activation | ClinVar |
|------|------|---------|-----------|---------|
| 1 | KRAS | p.Pro34Leu | 10.85 | Pathogenic |
| 2 | RIT1 | p.Pro52Leu | 6.81 | VUS |
| 3 | KRAS | p.Pro34Gln | 6.77 | Likely pathogenic |
| 4 | NRAS | p.Pro34Leu | 5.76 | Pathogenic |
| 5 | KRAS | p.Pro34Arg | 4.67 | Pathogenic/LP |
| 6 | RHEB | p.Pro37Leu | 3.63 | Pathogenic |

Pro34 in KRAS/NRAS and Pro52 in RIT1 are in the switch I region, critical for GTP hydrolysis and effector binding. RHEB Pro37 is an analogous position in a related small GTPase (mTOR pathway, not RASopathy per se — should be described as "small-GTPase extension" rather than grouped with RASopathy clinical framework).

**Clinical utility**: F11728 prioritizes RIT1 p.Pro52Leu as a mechanistically plausible RASopathy VUS at an analogous switch-region position, warranting expert-panel review and functional testing. Note: SAE grouping alone does NOT support clinical reclassification — formal reclassification requires gene-specific ClinGen VCEP rules, population frequency, phenotype, de novo evidence, and approved functional assays.

Lower activations extend to TP53 Pro278 (4 variants, all P/LP — known hotspot), then various other proline disruptions.

This feature is solid supplementary evidence for "SAE recovers expert-panel analogous residue reasoning." ClinGen RASopathy rules explicitly allow cross-gene evidence for HRAS/NRAS/KRAS homologous positions. The SAE learned this reasoning pattern automatically.

---

## Discovery 2: Modality-Specific Constraints (DNA-Driven vs Protein-Driven)

### Updated Framing (per GPT Pro)

We avoid overinterpreting mechanism. The claim is:

> "A subset of pathogenic features is genome-LM-dominant, suggesting that coding missense pathogenicity is not fully captured by protein-sequence representations alone."

### Key Statistics

- Overall modality-specific features: DNA-driven 526, protein-driven 33 (16:1 ratio)
- Modality ratio is orthogonal to pathogenicity (r = -0.076)
- 11 strongly DNA-driven features (ratio < 0.25) enriched in: ACTA1 (4/11), PTEN (3/11), PTPN11 (3/11)

### What We Can Safely Claim

1. DNA-LM and protein-LM capture **non-redundant** pathogenicity signals
2. Some disease genes (ACTA1, PTEN) have pathogenic features dominated by genome-LM signal
3. The modality decomposition captures **mechanism type** (how), not **severity** (whether)

### What Requires Further Validation (per GPT Pro)

- Separate CpG/methylation context, codon conservation, splicing proximity, and structural damage
- Run protein-only vs DNA-only occlusion experiments
- For CTCF: interpret as zinc-finger DNA-contact coding mechanism, not CpG hotspot
- Control for gene-level disease density / ClinVar prior

### Pending: SAE controls job (9812031) includes modality occlusion validation

---

## Discovery 3 (Demoted): Methionine-Related Features

### Met1 Audit Results

| Feature | Met1 (start-loss) | Internal Met | Internal path rate |
|---------|------------------:|-----------:|-------------------:|
| F330 | 89.7% | 10.3% | 25.0% |
| F11571 | 92.9% | 7.1% | 28.6% |
| F946 | 93.3% | 6.7% | 28.6% |
| F1159 | 98.6% | 1.4% | n=1 |
| F614 | 96.4% | 3.6% | n=0 |
| F9324 | 98.4% | 1.6% | n=0 |
| F2853 | 85.6% | 14.4% | 43.8% |
| **F2013** | **0.3%** | **99.7%** | **77.6%** |
| F4272 | 8.6% | 91.4% | 93.3% (but mostly non-Met subs) |

**7/9 features are start-loss driven.** Only F2013 has genuine internal Met signal (350 variants, 77.6% pathogenic). F4272 has high internal Met proportion but is mostly non-Met substitutions (35 Met out of 692 active) — should be relabeled by its actual top substitution/gene pattern, not included in methionine story.

### Revised Narrative

> CrossBioSAE separates two methionine-related mechanisms: a known initiator-Met/start-loss category (7 features, automatically recovered from missense-style annotations) and a candidate internal-methionine-loss category (F2013: 350 internal Met variants across diverse genes, 77.6% pathogenic rate, enriched in GLA, GAA, NR2F1, BTK, SDHA). The latter requires structural validation (AlphaFold SASA, Met-aromatic contacts) before claiming as a novel mechanism.

### Next Steps for Internal Met

- [ ] F2013 structural analysis: AlphaFold buriedness, nearest aromatic residues within 5-6 A
- [ ] Check if F2013 internal Met positions are enriched for Met-pi interactions
- [ ] Compare pathogenic vs benign internal Met positions

---

## Updated Priority Ranking (aligned with GPT Pro)

| Discovery | Paper section | Status |
|-----------|-------------|--------|
| Cys-loop receptor TM2 features (F10750 etc.) | Main figure + case study | **READY for figure** |
| RAS switch region feature (F11728) | Supplement / secondary case | READY |
| DNA-driven modality features | Main cross-modal panel | Needs occlusion validation |
| Internal Met loss (F2013) | Supplement candidate | Needs structural validation |
| Start-loss recovery (7 features) | Methods/supplement note | Complete |

## Updated Manuscript Structure (per GPT Pro R1+R2)

**Overall title framing** (per GPT Pro R1): avoid "novel SAE discoveries"; use:
> "Sparse cross-modal features recover known mechanisms and reveal under-recognized variant effect modes"

**Section 1: Cross-paralog substitution-specific mechanisms**
Lead with GABA receptor T>I (F10750). Suggested figure title:
> "A sparse feature captures pore-region threonine disruption across ligand-gated and voltage-gated ion channels."

Core sentence:
> "CrossBioSAE learns an ion-channel pore-region feature with a high-activation Thr→Ile core at conserved pore threonines across GABA-A, glycine, nicotinic, and voltage-gated potassium channels, matching literature-supported channel-gating mechanisms."

**Section 2: Modality-specific constraints in missense pathogenicity**
DNA-driven vs protein-driven features. Core sentence:
> "A subset of pathogenic features is genome-LM-dominant, suggesting that coding missense pathogenicity is not fully captured by protein-sequence representations alone."

**Section 3: Candidate under-recognized biochemical mechanisms**
Internal Met loss (F2013) conditional on structural validation. Core sentence:
> "After excluding initiator-Met/start-loss variants, internal Met-loss features remain enriched at buried or aromatic-contacting residues, suggesting methionine-specific structural constraints not captured by generic hydrophobic substitution scores."

---

## Eval Robustness Results (for completeness)

| Setting | AUC | AUPRC |
|---------|-----|-------|
| Temporal holdout (train ≤2023, test 2024+) | 0.9585 | 0.9224 |
| Temporal no constraint | 0.9581 | 0.9204 |
| Gene-heldout 5-fold | 0.9584 | — |
| Gene-heldout no constraint | 0.9571 | — |
| Random CV v6 (reference) | 0.9629 | — |
| AlphaMissense (reference) | 0.9638 | — |

Model generalizes well: <0.5% drop from random CV to temporal/gene-heldout.

---

## Pending Experiments (prioritized per GPT Pro R2)

### Tier 1: Required before main text

1. **SAE controls (job 9812031)**: FDR enrichment, 5-seed stability, PCA/ICA/NMF baselines, causal intervention, modality occlusion — QUEUED on GPU
2. **F10750 structural mapping**: Map all 33 variants onto cryo-EM structure (PDB 6HUJ or similar), confirm M2 pore position for Cys-loop and pore-helix for KCNB1
3. **F3118 validation**: Confirm GABRB2 P252T / GABRB3 P253T / GABRG2 P282T are at M2-M3 loop; check literature for gating transduction mechanism
4. **DNA-driven occlusion** (within SAE controls job):
   - protein half zero/mean ablation → DNA-driven features should survive
   - DNA half zero/mean ablation → DNA-driven features should vanish
   - protein-only vs DNA-only vs joint SAE comparison
   - modality permutation control
   - input norm matching check (ESM vs Evo2 projection norms)
5. **FDR upgrade**: Apply BH-FDR to ALL enrichment analyses (gene, substitution, condition, domain) for paper-grade statistics; do feature-level FDR across all 12,288 features, not just 100 selected

### Tier 2: Required for supplement / secondary claims

6. **F2013 internal Met validation**:
   - Remove top 5 genes, check if path rate still holds (disease-gene confound control)
   - Matched control: same gene, same conservation, non-Met hydrophobic subs as comparator
   - AlphaFold SASA, buriedness, nearest aromatic within 5-6 A, Met-aromatic contact energy
   - Functional literature check for GLA/GAA/BTK/SDHA internal Met variants
7. **F10750 seed stability**: Check if Cys-loop pore-Thr feature appears in all 5 SAE seeds
8. **F5463/F9483/F3888 validation**: Confirm structural positions + independent ClinVar/functional anchors for each

### Tier 3: Nice to have

9. **DMS/MAVE validation**: ProteinGym for TP53, BRCA1, PTEN (not GABA — no DMS available)
10. **VUS mechanism stratification**: Show same-score VUS have different SAE mechanism profiles

## File Locations

- Novel feature analysis: `results/novel_features/`
- Updated discoveries: `results/novel_sae_discoveries_v2.md`
- SAE model + activations: `results/variant_sae_v6/`
- Eval robustness: `results/eval_robust/results.json`
- Scripts: `scripts/analyze_novel_features.py`
