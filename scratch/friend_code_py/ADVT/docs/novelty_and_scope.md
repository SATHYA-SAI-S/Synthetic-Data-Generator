# Novelty and Scope Lock — Privacy-Preserving Synthetic Healthcare Data Generation

**Document Status:** LOCKED (Phase 1 deliverable)
**Last Updated:** 2026-08-17
**Scope Authority:** This document is the single binding reference for what is in and out of scope for the entire project. Any deviation requires an explicit amendment with justification.

---

## 1. Problem Statement

Generating high-fidelity synthetic healthcare data that is both clinically useful and provably privacy-preserving remains an open problem. The core tension is between:

- **Utility**: Synthetic records must capture real statistical relationships between features (diagnoses, labs, demographics) to be useful for downstream ML training or policy analysis.
- **Privacy**: Any generative model trained on patient data risks memorizing individuals, enabling reconstruction attacks or membership inference.

Differential privacy (DP) provides the only mathematically rigorous privacy guarantee, but applying it naively to generative models degrades utility substantially—especially for minority cohorts and rare diagnoses that are disproportionately sensitive.

---

## 2. Limitations in Existing DP-GAN / DP-VAE Tabular Approaches

### 2.1 The Uniform Privacy Budget Problem

The dominant approach in existing work applies a **uniform, static differential privacy budget (epsilon)** across the entire training process and all features equally. This creates three compounding failure modes:

**Failure Mode 1: Temporal Budget Misallocation**
In DP-SGD-based generative training, noise is injected at every gradient update step with a fixed clip norm C and noise multiplier sigma. The privacy accountant (typically Renyi DP or zero-concentrated DP) accumulates budget linearly with the number of steps. Early training steps—where gradients carry the most structural information—consume the same per-step budget as late steps where gradients are small refinements, leading to inefficient budget usage.

> **[CITATION NEEDED]**: Abadi et al. (2016) introduced DP-SGD. The specific claim about early-vs-late gradient information content is consistent with training dynamics literature but requires an empirical citation from DP generative model training literature.

**Failure Mode 2: Feature-Agnostic Noise Injection**
Existing frameworks apply identical DP noise to every feature gradient regardless of the feature's sensitivity, re-identification risk, or rare-cohort prevalence. A ZIP code contributes far more re-identification risk than a cholesterol reading, yet both receive identical noise treatment. This is simultaneously wasteful (over-protecting low-risk features) and dangerous (under-protecting high-risk feature combinations).

> **[CITATION NEEDED]**: Sweeney (1997) on k-anonymity limitations; Narayanan & Shmatikoff (2008) on re-identification of anonymized data. FLAG: insert DP literature on per-feature budget allocation.

**Failure Mode 3: Small-N Cohort Collapse**
Rare disease subpopulations represent a small fraction of records but carry disproportionate clinical importance. Under uniform DP noise calibrated to the majority cohort, the noise-to-signal ratio for rare subgroups becomes catastrophic. The generative model effectively ignores them, producing majority-biased synthetic data.

> **[CITATION NEEDED]**: Minority group underrepresentation in DP ML is documented in the algorithmic fairness literature. FLAG: insert citation.

### 2.2 State of the Art

| Framework     | DP Mechanism       | Budget Strategy  | Tabular-Specific | Key Limitation              |
|---------------|--------------------|------------------|------------------|-----------------------------|
| CTGAN/TVAE    | None               | N/A              | Yes              | No privacy guarantee        |
| DP-CTGAN      | DP-SGD (uniform)   | Fixed eps, uniform | Yes            | Failure modes 1-3 above     |
| GReaT         | None               | N/A              | Yes (via LLM)    | No DP; huge compute cost    |
| PATE-GAN      | PATE mechanism     | Per-teacher vote | Partial          | Doesn't scale to high-dim   |
| DP-VAE        | DP-SGD (uniform)   | Fixed eps, uniform | No             | Requires domain adaptation  |

> **[FLAG]**: A systematic literature review of publications post-2022 on adaptive DP for tabular generation is required before final paper submission. Validate against ACM CCS, NeurIPS, ICLR 2022-2025.

---

## 3. Locked Novelty Mechanisms (Hard Scope Boundaries)

### Novelty Mechanism A: Adaptive Per-Timestep Noise Schedule

**Description:** Instead of a fixed noise multiplier sigma throughout training, the DP-SGD wrapper will implement a decreasing noise schedule sigma(t) where t is the training step. The schedule is constrained so that total composed privacy cost epsilon_total remains bounded, but allocates more noise in early training and less in later fine-tuning stages.

**Mechanism:** The adaptive schedule uses a composition theorem (Renyi DP moments accountant or PRV accountant) to compute a noise schedule that minimizes a utility proxy metric subject to the total epsilon budget. The schedule is pre-computed before training begins—not adapted online.

**Why novel:** No published DP-GAN or DP-VAE framework for tabular data has implemented non-uniform temporal budget allocation with formal privacy accounting. Directly addresses Failure Mode 1.

**Explicitly out-of-scope variants:** Online adaptive methods (adjusting sigma based on observed gradient norms) are out of scope—they introduce data-dependent noise that requires separate accounting treatment.

### Novelty Mechanism B: Non-Uniform Per-Feature / Per-Risk-Tier Privacy Budget

**Description:** Features will be partitioned into privacy risk tiers (e.g., Tier 1: direct HIPAA identifiers, Tier 2: quasi-identifiers, Tier 3: clinical measurements, Tier 4: administrative codes) by the auto-config engine (Phase 4/5). The DP training process will allocate differential noise budgets per tier using parallel composition: higher-risk tiers receive more aggressive noise (lower per-tier epsilon) while lower-risk tiers receive less noise (higher per-tier epsilon).

**Mechanism:** Parallel composition of DP guarantees applies when features are processed independently. For features processed jointly, sensitivity is bounded by the maximum-risk tier's sensitivity.

**Why novel:** Per-feature DP budget allocation has been explored in the query-release setting but not, to our knowledge, in generative model training for structured tabular healthcare data. Directly addresses Failure Modes 2 and 3.

**Explicitly out-of-scope variants:** Continuous risk scoring (vs. discrete tiers) and online tier re-assignment during training are out of scope.

---

## 4. Explicit Out-of-Scope Items

| Out-of-Scope Item                          | Reason                                               |
|--------------------------------------------|------------------------------------------------------|
| Unstructured data (notes, images, EHRs)   | Tabular only; NLP/vision DP has different threat model |
| Online / federated learning                | Single-node training is the target setting           |
| k-anonymity or l-diversity                 | DP subsumes these; mixing creates false confidence   |
| Synthetic data evaluation framework        | Phase 8/9 scope                                      |
| Model deployment / serving                 | Project ends at synthetic data artifact generation   |
| Clinical validation of synthetic quality   | Requires domain expert collaboration                 |
| Adversarial robustness beyond DP guarantee | Separate ML hardening concern                        |
| Hyperparameter search / AutoML             | Manual configuration with documented rationale       |
| Time series, graphs, non-tabular modalities| Requires separate architecture                      |

---

## 5. Threat Model

**In scope (defended against):**
- Membership inference attacks
- Attribute inference attacks
- Reconstruction attacks from model parameters

**Out of scope (not defended against):**
- Adversaries with unlimited side information
- Physical attacks on compute infrastructure
- Social engineering or insider threats

The formal guarantee is (epsilon, delta)-differential privacy on the training algorithm. The DP training guarantee implies protections against the three in-scope attack classes.
