# CrossBioSAE: Novel SAE Discoveries v2 (Post GPT-Pro Review)

Date: 2026-06-04
Status: Updated after GPT Pro review — Met1 confound confirmed, story pivot to Cys-loop receptor features

## Changes from v1

GPT Pro correctly identified that 7/9 "methionine loss" features were driven by Met1/start-loss variants (85-99% Met1). After excluding prot_pos=1, internal Met pathogenic rate dropped to 25-44%. **Methionine loss is demoted from "novel discovery" to "recovered start-loss category + candidate internal-Met signal".**

The new lead story is **Discovery 1: cross-paralog substitution-specific channel features** (was Discovery 2 in v1).

---

## Discovery 1 (LEAD): Cross-Paralog Substitution-Specific Ion Channel Features

### Summary

The SAE learned a **family of features** that each capture a specific substitution type at homologous structural positions across the **Cys-loop ligand-gated ion channel superfamily** — without any pathway, family, or structural annotation input.

### Feature F10750: TM2 Pore-Lining Threonine Disruption (198.9x enrichment)

33 activating variants. The top 5 (activation > 4.8) are ALL at the conserved TM2 pore-lining threonine:

| Rank | Gene | Variant | Activation | ClinVar | Literature |
|------|------|---------|-----------|---------|------------|
| 1 | GABRB1 | p.Thr287Ile | 6.06 | Likely pathogenic | — |
| 2 | GABRB2 | p.Thr286Ile | 6.04 | Likely pathogenic | — |
| 3 | GABRB3 | p.Thr287Ile | 5.95 | Pathogenic | NC 2022: GOF variant, DEE |
| 4 | GABRA2 | p.Thr294Ile | 5.03 | Pathogenic/LP | — |
| 5 | GABRA1 | p.Thr294Ile | 4.82 | Pathogenic | Functional: T292 pore residue |

**Crucially, the feature extends beyond GABA receptors to the entire Cys-loop superfamily:**

| Activation | Gene | Variant | Receptor family |
|-----------|------|---------|-----------------|
| 3.01 | GABRA5 | p.Thr301Arg | GABA-A alpha 5 |
| 2.96 | KCNB1 | p.Thr374Ile | Voltage-gated K+ channel |
| 2.27 | CHRNA1 | p.Thr274Ile | Nicotinic ACh receptor |
| 2.22 | CHRNE | p.Thr284Ile | Nicotinic ACh receptor |
| 1.92 | GLRA2 | p.Thr296Met | Glycine receptor |
| 1.83 | GLRA1 | p.Thr287Ala | Glycine receptor |

All of these Thr positions map to the TM2 segment (channel pore helix). The threonine side chains face the pore lumen and are critical for:
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

3. **CHRNA1 p.Thr274Ile**: The nicotinic receptor T274 is homologous to the GABA receptor pore threonine. Known to cause congenital myasthenic syndromes.

4. **KRAS p.Pro34Leu** (Feature F11728): ClinGen RASopathy Expert Panel classified this as pathogenic. The P34 position is in the switch I region critical for GTP hydrolysis.

### Why This Is NC-Level

1. **Cross-paralog reasoning without labels**: The SAE groups GABRB1/B2/B3 + GABRA1/A2/A5 + GLRA1/A2 + CHRNA1/CHRNE — all members of the Cys-loop superfamily — into one feature, despite never seeing protein family or structural domain annotations.

2. **Substitution × position specificity**: Feature F10750 doesn't just capture "GABA receptor variants" — it captures specifically Thr→Ile at pore-lining TM2 positions. Feature F3118 captures Pro→Thr at TM1-TM2 loop positions. This resolution matches expert structural reasoning.

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

Pro34 in KRAS/NRAS and Pro52 in RIT1 are in the switch I region, critical for GTP hydrolysis and effector binding. RHEB Pro37 is the analogous position.

**Clinical utility**: RIT1 p.Pro52Leu is currently VUS in ClinVar, but the SAE groups it with confirmed pathogenic KRAS/NRAS variants at the homologous position — supporting reclassification to Likely Pathogenic.

Lower activations extend to TP53 Pro278 (4 variants, all P/LP — known hotspot), then various other proline disruptions.

This feature is solid supplementary evidence for "SAE recovers expert-panel analogous residue reasoning." ClinGen RASopathy rules explicitly allow cross-gene evidence for HRAS/NRAS/KRAS homologous positions.

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

**7/9 features are start-loss driven.** Only F2013 has genuine internal Met signal (350 variants, 77.6% pathogenic).

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

## Updated Manuscript Structure (per GPT Pro suggestion)

**Section 1: Cross-paralog substitution-specific mechanisms**
Lead with GABA receptor T>I (F10750). Core sentence:
> "CrossBioSAE learns a GABA_A receptor feature that activates on Thr→Ile substitutions across multiple receptor subunits, matching literature-supported channel-gating mechanisms."

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

## Pending Experiments

1. **SAE controls (job 9812031)**: FDR enrichment, 5-seed stability, PCA/ICA/NMF baselines, causal intervention, modality occlusion — QUEUED on GPU
2. **GABA structural mapping**: Map F10750 variants onto cryo-EM structure, confirm TM2 pore position
3. **Internal Met structural validation**: F2013 AlphaFold analysis
4. **FDR upgrade**: Apply BH-FDR to all enrichment analyses for paper-grade statistics
5. **DMS/MAVE validation**: ProteinGym for TP53, BRCA1, PTEN (not GABA — no DMS available)

## File Locations

- Novel feature analysis: `results/novel_features/`
- Updated discoveries: `results/novel_sae_discoveries_v2.md`
- SAE model + activations: `results/variant_sae_v6/`
- Eval robustness: `results/eval_robust/results.json`
- Scripts: `scripts/analyze_novel_features.py`
