# Architectural Design Document
## Privacy-Preserving Synthetic Healthcare Data Generation Framework
### Phases 1–3: Scope Lock · Environment · Preprocessing Pipeline

---

> **Document Type:** Architectural Reference  
> **Audience:** Engineers onboarding to Phases 4–10, technical reviewers, research collaborators  
> **Status:** Authoritative — reflects the implemented and tested codebase  
> **Last Updated:** 2026-08-17

---

## Table of Contents

1. [Project Mission and Motivation](#1-project-mission-and-motivation)
2. [Non-Negotiable Design Principles](#2-non-negotiable-design-principles)
3. [Phase 1 — Literature Review and Scope Lock](#3-phase-1--literature-review-and-scope-lock)
4. [Phase 2 — Environment and Infrastructure Setup](#4-phase-2--environment-and-infrastructure-setup)
5. [Phase 3 — The Preprocessing Pipeline Architecture](#5-phase-3--the-preprocessing-pipeline-architecture)
   - 5.1 [Configuration Layer](#51-configuration-layer)
   - 5.2 [Profiling Layer](#52-profiling-layer)
   - 5.3 [Preprocessing Layer](#53-preprocessing-layer)
   - 5.4 [Registry Layer](#54-registry-layer)
   - 5.5 [Test Suite](#55-test-suite)
6. [Module Dependency Graph](#6-module-dependency-graph)
7. [Interface Contracts for Downstream Phases](#7-interface-contracts-for-downstream-phases)
8. [Architectural Invariants (Must Never Be Broken)](#8-architectural-invariants-must-never-be-broken)
9. [Problems Encountered and Resolutions](#9-problems-encountered-and-resolutions)
10. [Phase Handoff Checklist](#10-phase-handoff-checklist)

---

## 1. Project Mission and Motivation

### 1.1 The Core Problem

Healthcare institutions generate enormous volumes of structured patient data — diagnoses, lab results, medication histories, demographics, outcomes. This data is invaluable for:

- Training ML models for clinical decision support
- Epidemiological research and drug discovery
- Policy simulation and resource planning

However, real patient data cannot be freely shared. HIPAA in the United States (and equivalent laws globally) places strict legal requirements on patient data access. The practical result: researchers cannot iterate quickly, models are trained on datasets that are too small, and collaboration across institutions is legally complex and slow.

**The solution this project pursues:** Generate *synthetic* patient data that is statistically faithful to the real data but carries a *mathematically provable privacy guarantee*, such that even a sophisticated adversary with background knowledge cannot reconstruct or identify any real patient from the synthetic dataset.

### 1.2 Why Differential Privacy?

Differential privacy (DP) is the gold standard for privacy guarantees. Its formal definition:

> A randomized algorithm $\mathcal{M}$ is $(\varepsilon, \delta)$-differentially private if, for any two adjacent datasets $D$ and $D'$ differing in one record, and for any output set $S$:
>
> $$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

In plain English: removing or adding any single patient's record from the training set changes the model's output distribution by at most a factor of $e^{\varepsilon}$. A small $\varepsilon$ (like 1.0 or 3.0) means the guarantee is strong.

### 1.3 The Gap This Project Fills

Existing DP generative models apply a *uniform, static* privacy budget across all features and all training steps. This project introduces two novel mechanisms:

| Mechanism | What It Does | What Problem It Solves |
|---|---|---|
| **A — Adaptive Noise Schedule** | Varies the DP noise multiplier $\sigma(t)$ across training steps | Early steps waste budget; late steps need fine-grained signal |
| **B — Per-Feature Risk-Tier Budget** | Allocates different $\varepsilon$ budgets to different feature risk tiers | Zip codes need more protection than cholesterol readings |

---

## 2. Non-Negotiable Design Principles

These principles were established before a single line of code was written and are binding on all phases.

### Principle 1 — High Cohesion, Low Coupling

Every module does *exactly one job*. If you cannot describe a module's purpose in one sentence without the word "and," it violates this principle.

| Module | Job (one sentence) |
|---|---|
| `config/schema.py` | Defines all numeric thresholds as typed, frozen Pydantic models |
| `profiling/dataset_profiler.py` | Computes statistical profiles and HIPAA flags for every column |
| `preprocessing/scalers.py` | Scales continuous columns and inverts that scaling |
| `preprocessing/encoders.py` | Encodes categorical columns and inverts that encoding |
| `preprocessing/missingness.py` | Injects missingness indicators and restores NaN positions |
| `preprocessing/pipeline.py` | Orchestrates the above in order, receiving all dependencies injected |
| `registry/schema_registry.py` | Persists and retrieves fitted pipeline state atomically |

### Principle 2 — Interface-First Design

Every module pair communicates through an Abstract Base Class, not a concrete class. This means Phase 4 can replace any component without touching the orchestrator.

```
AbstractProfiler      ← DatasetProfiler (Phase 3 concrete)
AbstractScaler        ← StandardScaler, MinMaxScaler, RobustScaler
AbstractEncoder       ← OneHotEncoder, FrequencyEncoder
AbstractMissingnessHandler ← MissingnessHandler
AbstractSchemaRegistry ← FileSchemaRegistry
```

### Principle 3 — Mandatory inverse_transform

**Every encoder and every scaler must implement `inverse_transform`.** This is non-negotiable because Phase 7's diffusion model generates data in the *encoded space* (normalized floats and integer indices), and that encoded output must be decoded back to a human-readable DataFrame. A missing or broken `inverse_transform` silently corrupts all downstream evaluation.

The test suite enforces this with exact round-trip tests: `inverse_transform(transform(x)) == x`.

### Principle 4 — I/O at the Boundary Only

File reading and writing happens in exactly one place: `PreprocessingPipeline.fit_transform_from_file()`. All internal logic receives DataFrames and returns arrays. This makes the entire pipeline unit-testable without touching the filesystem (except the registry tests, which use `tmp_path`).

### Principle 5 — No Magic Numbers

Every numeric threshold (cardinality cutoffs, missing rate limits, uniqueness ratios, correlation thresholds) is a named field in `PipelineConfig`. There are zero hardcoded numbers anywhere in the business logic.

### Principle 6 — Dependency Injection Throughout

The `PreprocessingPipeline` constructor accepts its dependencies — profiler, missingness handler, encoder factory, scaler factory, registry — as arguments. It does not import or instantiate any concrete class internally. This enables Phase 4 to inject alternative encoders (e.g., target encoding for high-cardinality features) without modifying the pipeline file.

---

## 3. Phase 1 — Literature Review and Scope Lock

### 3.1 Purpose

Phase 1 produced a binding scope document before any code was written. Its purpose is to prevent the most common failure mode in ML research: unbounded scope expansion that consumes time without advancing the core contribution.

### 3.2 What Was Reviewed

The literature review identified three critical failure modes in existing DP generative model work for tabular data:

#### Failure Mode 1 — Temporal Budget Misallocation

Standard DP-SGD training applies a fixed noise multiplier $\sigma$ at every gradient step. The privacy accountant (Rényi DP or PRV accountant) accumulates cost linearly:

$$\varepsilon_{\text{total}} = \sum_{t=1}^{T} \varepsilon_{\text{step}}(t)$$

But gradient information is not uniformly distributed over training time. Early steps (large gradients, high learning rate) carry the most structural information about the data distribution. Spending the same per-step budget on early and late steps is inefficient — it's like spending the same amount of money on the foundation of a building as on the paint.

**Our response:** Novelty Mechanism A — a pre-computed noise schedule $\sigma(t)$ that concentrates more noise (and thus less per-step cost) on early high-information steps, reserving tighter noise for late fine-tuning.

#### Failure Mode 2 — Feature-Agnostic Noise

A patient's ZIP code can identify them with a probability approaching 1 when combined with age and diagnosis. A patient's HDL cholesterol reading cannot. Yet existing DP frameworks apply identical noise to both features' gradient contributions.

**Our response:** Novelty Mechanism B — a risk-tier taxonomy that assigns each feature to one of four tiers, with separate $\varepsilon$ budgets per tier via parallel DP composition.

#### Failure Mode 3 — Small-N Cohort Collapse

For a dataset with 10,000 patients, the required DP noise (calibrated to $\varepsilon = 3$, $\delta = 10^{-5}$) renders minority subgroups (e.g., patients with rare diseases, n < 50) completely unlearnable. The synthetic data is majority-population-biased.

**Our response:** The profiler's `small_n_flag` and `SmallNConfig` provide a hook for Phase 7 to apply cohort-specific noise calibration.

### 3.3 The Scope Lock

The scope document in `docs/novelty_and_scope.md` locks:

- **In scope:** Two novelty mechanisms, tabular data only, single-node training, (ε,δ)-DP guarantee on the training algorithm
- **Out of scope:** Unstructured data, federated learning, k-anonymity, model deployment, clinical validation, time series, hyperparameter search

Any request to add an out-of-scope item requires an explicit amendment with written justification.

### 3.4 Threat Model

The framework defends against:

| Attack | Description | Defended By |
|---|---|---|
| Membership inference | Is patient X in the training set? | DP training guarantee |
| Attribute inference | What is patient X's HIV status given other attributes? | DP training guarantee |
| Reconstruction | Can we recover a training record from the model weights? | DP training guarantee |

Not defended: adversaries with unlimited side information, physical infrastructure attacks.

---

## 4. Phase 2 — Environment and Infrastructure Setup

### 4.1 Python and Dependency Decisions

**Python 3.10** was chosen as the runtime:
- Minimum version supported by Opacus 1.x (the chosen DP library)
- Maximum version before Python 3.12's deprecation of several scipy internals
- Long-term support status confirmed

**Key dependency choices with rationale:**

| Package | Version | Why Chosen | Why Not Alternative |
|---|---|---|---|
| `opacus` | 1.5.2 | PyTorch-native DP-SGD, BatchMemoryManager, built-in Rényi accountant | `tensorflow-privacy`: requires TF2, conflicts with PyTorch diffusion model |
| `pydantic` | 2.x | Frozen typed models, `model_dump_json()` for serialization, V2 performance | V1: slower, no `model_config` frozen support |
| `joblib` | 1.4+ | Thread-safe serialization of sklearn-style estimators with compression | `pickle`: no compression, unsafe for untrusted data |
| `scipy` | 1.13+ | `pointbiserialr`, `chi2_contingency` for missingness heuristics | `statsmodels`: heavier, not needed for these specific tests |
| `pyarrow` | 16+ | Parquet support for schema artifacts | `feather`: less stable across versions |

### 4.2 Kaggle GPU Constraint and Its Design Impact

This project runs on Kaggle free-tier notebooks with these hard limits:

```
GPU quota:        ~30 hours/week (rolling 7-day window)
Session limit:    12 hours (hard kill — no warning)
GPU VRAM:         ~16 GB (T4)
Working storage:  ~20 GB (/kaggle/working/)
```

These constraints directly shaped the architecture:

**12-hour session kill → Checkpointing contract (Phase 6)**
The session can die mid-epoch. The Phase 6 checkpointing system must:
1. Save every `checkpoint_every_n_steps` steps (default: 500)
2. Include the privacy accountant state (resetting it = privacy violation)
3. Use atomic writes (temp file → rename) to prevent partial checkpoint corruption
4. Rotate old checkpoints to stay within 20 GB

**16 GB VRAM → Gradient accumulation design (Phase 7)**
DP-SGD with Opacus requires per-sample gradient storage, which multiplies VRAM usage by the batch size. The Phase 7 training loop must use Opacus's `BatchMemoryManager` and virtual step accumulation.

### 4.3 setup_check.py

The `environment/setup_check.py` script is a standalone verifier with zero dependency on the rest of the codebase. It verifies:

1. Python ≥ 3.10
2. All required packages importable with correct versions
3. CUDA GPU available with ≥ 1 device
4. Pydantic V2 API (`model_config = {"frozen": True}`)
5. **Opacus DP-SGD smoke test:** creates a tiny model, wraps with `PrivacyEngine`, runs one training step, queries `get_epsilon()` and asserts `epsilon > 0`

The smoke test is the critical check — it validates the entire DP training stack is correctly installed, not just that `import opacus` succeeds.

---

## 5. Phase 3 — The Preprocessing Pipeline Architecture

The preprocessing pipeline is the heart of Phase 3. It transforms raw, messy healthcare CSV files into clean, encoded numpy arrays ready for the diffusion model in Phase 7, and provides a complete invertible path back from model outputs to readable DataFrames.

```
Raw CSV
  │
  ▼
[DatasetProfiler]          ← Produces DatasetProfile (typed, serializable)
  │                           Contains: per-column stats, HIPAA flags,
  │                           missingness patterns, structural dependencies
  ▼
[Column Drop Decision]     ← Drop: HIPAA-flagged, near-identifier, high-missing
  │
  ▼
[MissingnessHandler.fit]   ← Learn which columns need indicator injection
  │
[MissingnessHandler.transform]
  │                           Outputs: indicator columns + imputed values
  │                           (NaN positions preserved in indicators)
  ▼
[Per-Column Encoding/Scaling]
  │   CONTINUOUS/ORDINAL → StandardScaler (or MinMax, Robust)
  │   CATEGORICAL_LOW    → OneHotEncoder (rare → __other__)
  │   CATEGORICAL_HIGH   → FrequencyEncoder (vocab-sorted indices)
  │   BINARY             → OneHotEncoder
  ▼
[Concatenated Float32 Array]   shape: (n_rows, n_encoded_features)
  │
  ▼
[FileSchemaRegistry.save()]    ← Atomic write: profile.json + pipeline_state.joblib
                                 Versioned directory: /registry/<dataset>/v1/
```

### 5.1 Configuration Layer

**File:** `src/config/schema.py`

The configuration layer is a set of frozen Pydantic V2 models that serve as the **single source of truth** for every numeric threshold in the system.

#### Design Decision: Frozen Models

All config models use `model_config = {"frozen": True}`. This means:
- Config objects are immutable after construction
- They can be used as dictionary keys and in sets (hashable)
- Accidental mutation is a `TypeError`, not a silent bug

#### Structure

```
PipelineConfig (root)
├── CardinalityConfig
│   ├── low_card_max: int = 15          # ≤15 unique → OneHotEncoder
│   ├── near_identifier_ratio: float = 0.95  # >95% unique string → near-id
│   ├── rare_category_min_freq: int = 10     # <10 occurrences → __other__
│   └── rare_category_min_frac: Optional[float]  # alternative fraction form
│
├── MissingnessConfig
│   ├── drop_if_missing_above: float = 0.80   # >80% missing → drop column
│   ├── inject_indicator_above: float = 0.01  # >1% missing → add flag column
│   ├── structural_correlation_threshold: float = 0.40
│   └── structural_min_n: int = 30
│
├── SmallNConfig
│   ├── small_n_threshold: int = 500    # <500 rows → small-N flag
│   └── minimum_viable_n: int = 100    # <100 rows → refuse to proceed
│
└── DtypeInferenceConfig
    ├── numeric_confidence_threshold: float = 0.95
    └── ordinal_max_unique_int: int = 20
```

#### Extension Pattern for Phase 4/5

Phase 4 must add `PrivacyTierConfig` to the system **without modifying** `schema.py`:

```python
# Phase 4 creates this in src/config/privacy_schema.py
from src.config.schema import PipelineConfig
from pydantic import BaseModel

class PrivacyTierConfig(BaseModel):
    model_config = {"frozen": True}
    tier1_epsilon: float = 0.5   # HIPAA identifiers
    tier2_epsilon: float = 1.0   # quasi-identifiers
    tier3_epsilon: float = 3.0   # clinical measurements
    tier4_epsilon: float = 5.0   # administrative codes

class ExtendedPipelineConfig(PipelineConfig):
    privacy_tiers: PrivacyTierConfig = PrivacyTierConfig()
```

### 5.2 Profiling Layer

**Files:** `src/profiling/base.py`, `src/profiling/dataset_profiler.py`

The profiling layer answers: *What kind of data is in each column, and what does it look like?*

#### The DatasetProfile Contract

`DatasetProfile` is a Pydantic model, meaning it is fully serializable to JSON with `model_dump_json()` and deserializable with `model_validate_json()`. Phase 4's AutoConfigEngine calls `registry.load_profile()` to get this object without loading the heavy `pipeline_state.joblib`.

Key fields:

```python
class DatasetProfile:
    dataset_name: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]         # one per column
    hipaa_flagged_columns: list[str]     # HIPAA Safe Harbor matches
    near_identifier_columns: list[str]   # uniqueness_ratio > threshold (strings only)
    high_missing_columns: list[str]      # missing_rate > drop_if_missing_above
    small_n_flag: bool                   # n_rows < small_n_threshold
    profiler_config_snapshot: dict       # the PipelineConfig used (for audit trail)
```

#### ColumnProfile: The Per-Column Contract

```python
class ColumnProfile:
    name: str
    inferred_dtype: InferredDtype        # CONTINUOUS | CATEGORICAL_LOW | ... | NEAR_IDENTIFIER
    pandas_dtype: str                    # raw pandas dtype string ("float64", "object")

    # Universal stats
    n_total: int
    n_non_null: int
    n_null: int
    missing_rate: float
    n_unique: int
    uniqueness_ratio: float

    # Continuous-only (None for categorical)
    mean: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    median: Optional[float]
    skewness: Optional[float]

    # Categorical-only (None for continuous)
    top_categories: Optional[list[tuple[str, int]]]
    n_rare_categories: Optional[int]

    # Missingness
    missingness_pattern: MissingnessPattern   # NONE | MCAR_LIKE | STRUCTURAL | HIGH
    structural_dependency: Optional[StructuralDependency]

    # Privacy
    hipaa_flag: HipaaFlag
```

#### HIPAA Safe Harbor Matching

The function `check_hipaa_identifier(column_name)` matches column names against all 18 HIPAA Safe Harbor identifier categories using regex patterns. It is a **standalone function** — independently importable and testable without constructing a profiler.

**Technical challenge solved:** Python `\b` word boundaries treat `_` as a word character, so `\bzip\b` does not match `zip_code`. The solution: pad the column name with underscores and use `(?:^|_)...(?:_|$)` boundaries. The function internally runs matching on `_column_name_` so that both `zip` and `zip_code` correctly match the geographic pattern.

**Ordering matters:** `ip_address` contains the substring `address`, which would match the geographic category before reaching the IP address category. The list is therefore ordered with IP addresses before geographic subdivisions.

#### Dtype Inference Priority Order

```
1. NEAR_IDENTIFIER  ← string-only, uniqueness_ratio > 0.95
2. BINARY           ← exactly 2 unique non-null values (any type)
3. ORDINAL          ← integer dtype, n_unique ≤ ordinal_max_unique_int (20)
4. CONTINUOUS       ← numeric, passes confidence threshold
5. CATEGORICAL_LOW  ← string, n_unique ≤ low_card_max (15)
6. CATEGORICAL_HIGH ← string, n_unique > low_card_max
7. UNKNOWN          ← fallback
```

**Critical design decision:** The NEAR_IDENTIFIER check applies only to **non-numeric** columns. Continuous float data (ages, lab values, BMI) naturally produces near-100% unique values — this is expected behavior, not evidence of an identifier column. Numeric ID columns (sequential integers like `patient_id = [0, 1, 2, ...]`) are caught by the HIPAA name-matching check instead.

#### Structural Missingness Detection

For each column with missingness above the `inject_indicator_above` threshold, the profiler tests whether the missingness is structural (correlated with another column's values) or random (MCAR-like).

- **Numeric predictor:** Point-biserial correlation between the binary missingness indicator and the predictor values
- **Categorical predictor:** Cramér's V from the chi-square contingency table

If the maximum correlation across all other columns exceeds `structural_correlation_threshold` (0.40), the column is flagged `STRUCTURAL` and the predictor column is recorded. This informs Phase 5's tier assignment — a structurally missing column may require special handling in the generative model.

### 5.3 Preprocessing Layer

**Files:** `src/preprocessing/base.py`, `scalers.py`, `encoders.py`, `missingness.py`, `pipeline.py`

#### Scalers

Three concrete scalers, all implementing `AbstractScaler`:

| Scaler | Formula | Use Case | Edge Case |
|---|---|---|---|
| `StandardScaler` | $z = (x - \mu) / \sigma$ | Most continuous features | If $\sigma = 0$: set to 1.0 with warning |
| `MinMaxScaler` | $z = (x - \min) / (\max - \min)$ | Bounded features (e.g. 0–100 scores) | If $\min = \max$: range set to 1.0 |
| `RobustScaler` | $z = (x - Q_{50}) / (Q_{75} - Q_{25})$ | Skewed clinical measurements with outliers | If IQR = 0: set to 1.0 |

All scalers propagate NaN through both `transform` and `inverse_transform`. This is intentional — the missingness handler, applied before scalers, has already imputed NaN values. Any NaN remaining at this stage is unexpected and should be visible rather than silently dropped.

**Round-trip guarantee:** `inverse_transform(transform(x)) ≈ x` within `atol=1e-5` (float32 arithmetic tolerance).

#### Encoders

**OneHotEncoder** — for low-cardinality features (n_unique ≤ `low_card_max`):

```
Vocabulary construction (fit):
  1. Compute value_counts
  2. Categories below min_freq → grouped into "__other__" token
  3. Vocabulary: [__null__, __other__ (if needed), *frequent_cats_by_freq_desc]

Transform:
  value → vocabulary index → one-hot row (binary vector)
  NaN   → index 0 (__null__) → [1, 0, 0, ...]
  rare  → __other__ index   → [0, 1, 0, ...]

Inverse:
  argmax(one-hot row) → vocabulary index → category string
  __null__ index → NaN
```

**FrequencyEncoder** — for high-cardinality features:

```
Vocabulary construction (fit):
  1. Compute value_counts DESCENDING (most frequent first)
  2. Rare categories (below min_freq) → "__other__" appended at end
  3. Vocabulary: [__null__, *frequent_cats_by_freq_desc, __other__]
  4. Index 0 always reserved for __null__

Transform:
  value → integer index (0 = null, 1 = most common, ...)
  Output shape: (n_samples, 1) — directly usable as embedding lookup index

Inverse:
  integer index → vocabulary string
  index 0 → NaN
```

**Why frequency ordering?** The Phase 7 diffusion model uses embedding tables for categorical features. Embedding tables perform better when common tokens have low indices (better cache locality, better gradient flow in early training). Frequency-sorted vocabulary is a free performance optimization.

**The `vocab_size` property** is publicly accessible: `encoder.vocab_size` returns the number of distinct tokens. Phase 7 uses this to initialize the embedding table: `nn.Embedding(encoder.vocab_size, embedding_dim)`.

#### MissingnessHandler

The handler is the only component that fundamentally changes the schema — it adds new columns (`<col>__missing_flag`). Its design is driven by a single constraint: **it must be exactly reversible**.

```
fit():
  For each column with missing_rate >= inject_indicator_above:
    Record column name
    Compute imputation value:
      numeric columns → median of non-null values
      string columns  → mode of non-null values

transform():
  For each tracked column:
    1. Add binary indicator column: 1 = was_missing, 0 = was_observed
    2. Fill NaN in original column with imputation value
  Returns new DataFrame (original not mutated)

inverse_transform():
  For each tracked column:
    1. Read indicator column → boolean mask
    2. Set original column to NaN where mask == 1
    3. Drop indicator column
  Returns new DataFrame (input not mutated)
```

**Why impute with median/mode instead of zero?** Scalers cannot receive NaN — they would produce NaN in the encoded array, which the diffusion model cannot process. The imputed value is a placeholder; the actual signal about where values are missing is carried by the indicator column. The model learns the joint distribution of (imputed_value, indicator) together.

#### PreprocessingPipeline — The Orchestrator

`PreprocessingPipeline` is the only class with complete knowledge of the pipeline topology. It is also the only class that should be instantiated by external code.

**Constructor parameters (all injected):**

```python
PreprocessingPipeline(
    config:               PipelineConfig,
    profiler:             AbstractProfiler,
    missingness_handler:  AbstractMissingnessHandler,
    encoder_factory:      Callable[[str], AbstractEncoder],  # col_name → encoder
    scaler_factory:       Callable[[str], AbstractScaler],   # col_name → scaler
    registry:             AbstractSchemaRegistry,
)
```

The `encoder_factory` and `scaler_factory` are callables rather than instances. This enables Phase 4 to provide a factory that returns different encoder types based on the column's profile — for example, a target encoder for high-cardinality categoricals in the context of a specific outcome variable.

**The three-phase lifecycle:**

```
Phase A — fit_transform(df, dataset_name):
  1. Profile the DataFrame → DatasetProfile
  2. Drop HIPAA / near-identifier / high-missing columns
  3. MissingnessHandler.fit(working_df) + .transform(working_df)
  4. For each column: fit and apply scaler or encoder
  5. Concatenate all encoded arrays → float32 numpy array
  6. registry.save(all fitted state)
  Returns: (n_rows, n_encoded_features) float32 array

Phase B — transform(df):
  1. MissingnessHandler.transform(df)  [uses fitted state]
  2. Apply each fitted scaler/encoder in order
  3. Concatenate → float32 array
  Returns: same shape as fit_transform output

Phase C — inverse_transform(arr):
  1. Split array by column widths (scalers → 1 col, OHE → n_vocab cols)
  2. Apply each inverse_transform
  3. Build DataFrame from decoded columns
  4. MissingnessHandler.inverse_transform(df) → restore NaN positions
  Returns: DataFrame in original space
```

### 5.4 Registry Layer

**Files:** `src/registry/base.py`, `src/registry/schema_registry.py`

The registry is the persistence boundary between Phase 3 and all downstream phases. It stores the complete fitted state of the pipeline so Phase 7 can reconstruct the inverse transform without re-running Phase 3.

#### What Gets Saved

Every call to `registry.save()` writes two files:

```
/registry/<dataset_name>/v<N>/
├── profile.json              ← DatasetProfile serialized to JSON (Pydantic)
│                               Lightweight: Phase 4 reads this without loading estimators
└── pipeline_state.joblib     ← Python dict containing:
                                  - scalers: dict[str, AbstractScaler]
                                  - encoders: dict[str, AbstractEncoder]
                                  - missingness_handler: MissingnessHandler
                                  - training_columns: list[str]
                                  - column_types: dict[str, str]
                                  - encoded_col_names: list[str]
                                  - version: int
                                  - dataset_name: str
```

#### Atomic Write Pattern (Critical for Kaggle)

The Kaggle session kill can happen at any moment. A partial write of `pipeline_state.joblib` would produce a corrupt registry entry that silently loads damaged state. The solution:

```
1. Create a temporary directory in the registry root
2. Write profile.json to temp directory
3. Write pipeline_state.joblib to temp directory (with compression level 3)
4. shutil.move(temp_dir, final_version_dir)   ← atomic on same filesystem
5. Update latest.txt pointer
```

The `shutil.move` operation is atomic on same-filesystem moves (it's a rename at the OS level). If the session dies during the write to the temp directory, the final version directory never appears, and the registry is left in a valid state from the previous version.

#### Versioning

Each call to `save()` creates a new version directory and updates `latest.txt`. This means:

- Multiple training runs with different configs produce traceable versions
- Old versions are never overwritten (immutable versions)
- `load()` with no version argument always loads the latest
- `load(version=1)` loads a specific historical version

### 5.5 Test Suite

**Files:** `src/tests/`

The test suite has 112 tests covering four distinct concerns:

| File | Tests | Key Concerns |
|---|---|---|
| `test_profiler.py` | ~32 | All 18 HIPAA categories (positive + negative), dtype inference, missingness, serialization round-trip |
| `test_encoders_inverse.py` | ~32 | **Round-trip correctness for all scalers and encoders**, NaN propagation, constant columns, shape contracts |
| `test_registry_roundtrip.py` | ~12 | Atomic write, versioning, scaler correctness post-deserialization |
| `test_pipeline_integration.py` | ~12 | End-to-end pipeline, NaN-free output, HIPAA dropping, CSV I/O, continuous round-trip |

---

## 6. Module Dependency Graph

```
                    ┌─────────────────────────────┐
                    │      PipelineConfig          │
                    │     (config/schema.py)       │
                    └──────────────┬──────────────┘
                                   │ imported by all
            ┌──────────────────────┼───────────────────────┐
            │                      │                       │
            ▼                      ▼                       ▼
  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐
  │ profiling/base.py│  │ preprocessing/    │  │ registry/base.py │
  │ (ABCs + types)   │  │ base.py (ABCs)    │  │ (ABC)            │
  └────────┬─────────┘  └────────┬──────────┘  └───────┬──────────┘
           │                     │                      │
           ▼                     │                      ▼
  ┌────────────────────┐         │             ┌────────────────────┐
  │dataset_profiler.py │         │             │schema_registry.py  │
  │(concrete profiler) │         │             │(concrete registry) │
  └────────────────────┘         ├─────────────────────────────────┐
                                 │                                 │
                     ┌───────────┼──────────────┐                 │
                     ▼           ▼              ▼                  │
               ┌──────────┐ ┌─────────┐ ┌────────────┐           │
               │scalers.py│ │encoders │ │missingness │           │
               │(concrete)│ │.py      │ │.py         │           │
               └──────┬───┘ └────┬────┘ └─────┬──────┘           │
                      └──────────┴─────────────┘                  │
                                 │                                 │
                                 ▼                                 │
                      ┌─────────────────────┐                     │
                      │   pipeline.py       │◄────────────────────┘
                      │ (orchestrator)      │
                      │ (no concrete        │
                      │  imports inside)    │
                      └─────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   test suite        │
                      │ (tests all layers)  │
                      └─────────────────────┘
```

**Key property:** `pipeline.py` imports only the abstract base classes (`AbstractProfiler`, `AbstractEncoder`, etc.), never the concrete implementations. This keeps the dependency arrows pointing inward, not outward — a textbook application of the Dependency Inversion Principle.

---

## 7. Interface Contracts for Downstream Phases

### What Phase 4 (AutoConfigEngine) Receives

| Contract | Field/Method | File |
|---|---|---|
| Typed dataset profile | `DatasetProfile` | `profiling/base.py` |
| Per-column inferred dtype | `ColumnProfile.inferred_dtype: InferredDtype` | `profiling/base.py` |
| HIPAA identifier flag | `ColumnProfile.hipaa_flag.is_identifier` | `profiling/base.py` |
| Missingness pattern | `ColumnProfile.missingness_pattern: MissingnessPattern` | `profiling/base.py` |
| Lightweight profile load | `registry.load_profile(dataset_name)` | `registry/base.py` |
| Encoder injection point | `encoder_factory` parameter in pipeline | `preprocessing/pipeline.py` |

### What Phase 7 (DP-Diffusion Training) Receives

| Contract | Field/Method | File |
|---|---|---|
| Full pipeline state | `registry.load(dataset_name)` | `registry/base.py` |
| Vocabulary size for embeddings | `FrequencyEncoder.vocab_size` | `preprocessing/encoders.py` |
| Encoded training array | `pipeline.fit_transform(df, name)` | `preprocessing/pipeline.py` |
| Inverse decoding | `pipeline.inverse_transform(arr)` | `preprocessing/pipeline.py` |

### What Phase 6 (Checkpointing) Must Persist

Per `docs/environment_notes.md`, every checkpoint bundle must include:

```
model_weights          ← nn.Module state_dict
optimizer_state        ← includes momentum accumulators
privacy_accountant_state ← Rényi/PRV step count and accumulated moments
global_step            ← int
epoch                  ← int
rng_states             ← Python, NumPy, PyTorch CPU, PyTorch CUDA
noise_schedule_state   ← Novelty Mechanism A pre-computed schedule position
per_tier_budget_used   ← Novelty Mechanism B accumulated ε per tier
lr_scheduler_state     ← if used
validation_metric_history ← for early stopping
```

**The privacy accountant state is mandatory.** If it is not checkpointed, resuming training from a checkpoint will reset the accountant to zero, causing the pipeline to silently spend more than the budgeted ε — a privacy violation.

---

## 8. Architectural Invariants (Must Never Be Broken)

These are hard invariants that must hold across all phases. Any violation must be surfaced immediately as a bug.

> [!IMPORTANT]
> **Invariant 1 — No column names in source code**
> Column names appear only in user-supplied CSV files and test fixtures. No source file outside `tests/` contains a string like `"patient_id"` as a business logic condition.

> [!IMPORTANT]
> **Invariant 2 — inverse_transform is required on every encoder/scaler**
> No new encoder or scaler may be added without a corresponding `test_encoders_inverse.py` test that verifies the round-trip within tolerance.

> [!IMPORTANT]
> **Invariant 3 — Registry writes are atomic**
> Any new registry backend must use the temp-write + rename pattern. Writes that can produce partial files under session kill are not acceptable.

> [!IMPORTANT]
> **Invariant 4 — Privacy accountant state must be checkpointed**
> Failure to checkpoint accountant state makes the privacy guarantee meaningless. This is a correctness requirement, not a quality improvement.

> [!IMPORTANT]
> **Invariant 5 — PipelineConfig is frozen and immutable**
> Never mutate a `PipelineConfig` instance after construction. Always create a new instance or subclass for extensions.

> [!CAUTION]
> **Invariant 6 — The encoded feature order must be stable**
> The array produced by `fit_transform` has a column order defined by `pipeline._encoded_col_names`. Phase 7 depends on this order being identical between `fit_transform` (training) and `transform` (inference). Any change to column ordering is a silent correctness violation in the model.

---

## 9. Problems Encountered and Resolutions

See the dedicated **Problems and Resolutions Document** (`docs/problems_and_resolutions.md`) for full detail. Summary table:

| # | Problem | Impact | Resolution |
|---|---|---|---|
| P-01 | Python `\b` doesn't work with underscores in column names | All compound HIPAA names (zip_code, medical_record_number) missed | Pad column with `_` underscores; use `(?:^|_)...(?:_|$)` anchors |
| P-02 | `ip_address` matched geographic pattern before IP pattern | Wrong HIPAA category assigned | Move IP addresses check before geographic in the list |
| P-03 | Continuous float columns flagged as near-identifier | All continuous features dropped from pipeline | Restrict near-identifier check to non-numeric columns only |
| P-04 | `fax` matched Phone numbers (fax was inside phone pattern) | Wrong category assignment | Extract fax into separate pattern entry, order before phone |
| P-05 | Missingness indicator columns appeared in inverse_transform output check | Test failure | Update test to filter `__missing_flag` columns before checking |
| P-06 | `write_to_file` tried with ArtifactMetadata on project files | Files written to wrong path | Use `write_to_file` without ArtifactMetadata for project files |
| P-07 | PowerShell uses `;` not `&&` for command chaining | Syntax error on compound commands | Use separate PowerShell commands or semicolons |

---

## 10. Phase Handoff Checklist

Before Phase 4 begins, confirm:

- [ ] `pytest src/tests/ -v` runs 112 tests, 0 failures
- [ ] `python environment/setup_check.py` exits with code 0
- [ ] `docs/novelty_and_scope.md` has been reviewed and signed off
- [ ] `docs/environment_notes.md` Phase 6 checkpointing contract has been read by Phase 6 lead
- [ ] `FileSchemaRegistry` root directory is accessible at `/kaggle/working/registry/` on Kaggle
- [ ] Phase 4 has read the interface contracts in Section 7 of this document
- [ ] Phase 4 has confirmed they will NOT modify `ColumnProfile` or `DatasetProfile` field names without coordinating with all downstream phases
