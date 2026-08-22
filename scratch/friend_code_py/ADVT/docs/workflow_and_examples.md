# End-to-End Workflow & Concrete Execution Guide
## Privacy-Preserving Synthetic Healthcare Data Generation Framework
### Step-by-Step Data Flow, Practical Examples, and Developer Walkthrough

---

> **Document Type:** Execution & Workflow Reference  
> **Target Audience:** ML Engineers, Data Engineers, Core Framework Contributors (Phases 4–10)  
> **Status:** Authoritative (Validated against Python 3.10 test suite)  
> **Last Updated:** 2026-08-17

---

## Table of Contents

1. [Executive Workflow Overview](#1-executive-workflow-overview)
2. [End-to-End Concrete Example: The Patient Lifecycle](#2-end-to-end-concrete-example-the-patient-lifecycle)
   - 2.1 [Input: Raw Healthcare Dataset (`raw_patient_data.csv`)](#21-input-raw-healthcare-dataset-raw_patient_datacsv)
   - 2.2 [Step 1: Ingestion and Automated Profiling](#22-step-1-ingestion-and-automated-profiling)
   - 2.3 [Step 2: Safe Harbor HIPAA & Near-Identifier Column Filtering](#23-step-2-safe-harbor-hipaa--near-identifier-column-filtering)
   - 2.4 [Step 3: Missingness Handling & Indicator Injection](#24-step-3-missingness-handling--indicator-injection)
   - 2.5 [Step 4: Vectorized Feature Transformation & Encoding](#25-step-4-vectorized-feature-transformation--encoding)
   - 2.6 [Step 5: Atomic Schema Registry Persistence](#26-step-5-atomic-schema-registry-persistence)
   - 2.7 [Step 6: Synthetic Sampling & Exact Inverse Reconstruction](#27-step-6-synthetic-sampling--exact-inverse-reconstruction)
3. [Full Python Code Example (Copy-Paste Ready)](#3-full-python-code-example-copy-paste-ready)
4. [Intermediate State Representation Reference](#4-intermediate-state-representation-reference)
5. [Connecting to Downstream Phases (4–10)](#5-connecting-to-downstream-phases-410)

---

## 1. Executive Workflow Overview

The Phase 1–3 framework operates as a **deterministic, bidirectional transformation pipeline**. It converts raw tabular healthcare data with heterogeneous types, missing entries, and HIPAA identifiers into normalized numerical tensors suitable for Differential Privacy (DP) generative models (such as DP-Diffusion in Phase 7), while guaranteeing exact reconstruction back to domain-readable clinical records.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FORWARD TRANSFORMATION PIPELINE                                 │
├─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│   Raw Clinical  │───▶│ DatasetProfiler │───▶│ Safe Harbor     │───▶│ MissingnessHandler       │ │
│   CSV / DataFrame│   │ (base.py)       │    │ HIPAA & Near-ID │    │ (missingness.py)         │ │
│                 │    │                 │    │ Column Filter   │    │ Impute + Flag Injected   │ │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └────────────┬─────────────┘ │
                                                                                  │               │
                       ┌──────────────────────────────────────────────────────────┘               │
                       ▼                                                                          │
         ┌───────────────────────────┐    ┌───────────────────────────┐                           │
         │ Scalers & Encoders        │───▶│ Concatenated Float32      │                           │
         │ StandardScaler, OHE,      │    │ Tensor (Ready for Phase 7)│                           │
         │ FrequencyEncoder          │    └─────────────┬─────────────┘                           │
         └───────────────────────────┘                  │                                         │
                                                        ▼                                         │
                                          ┌───────────────────────────┐                           │
                                          │ FileSchemaRegistry (v1)   │                           │
                                          │ profile.json + joblib     │                           │
                                          └───────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INVERSE RECONSTRUCTION PIPELINE                                 │
├──────────────────────────┐    ┌──────────────────────────┐    ┌───────────────────────────────┐ │
│ Generative Model Sample  │───▶│ Invert Scalers & Encoders│───▶│ Invert MissingnessHandler     │ │
│ Array (Phase 7 DP Output)│    │ (Continuous & Categorical│    │ Restore Exact NaN Positions   │ │
│                          │    │  decoding)               │    │ Strip Flag Columns            │ │
└──────────────────────────┘    └──────────────────────────┘    └───────────────┬───────────────┘ │
                                                                                ▼                 │
                                                                ┌───────────────────────────────┐ │
                                                                │ Final Synthetic Clinical Data │ │
                                                                │ (Exact Schema & Clinical Type)│ │
                                                                └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Concrete Example: The Patient Lifecycle

To make this architecture concrete, we follow **5 sample patient records** containing realistic healthcare anomalies through every single step of the pipeline.

### 2.1 Input: Raw Healthcare Dataset (`raw_patient_data.csv`)

Consider this realistic input table with 6 columns of varying nature:

| row | `patient_id` (HIPAA) | `record_uuid` (Near-ID) | `age` (Continuous) | `systolic_bp` (Continuous) | `admission_type` (Low-Card) | `icd_code` (High-Card) |
|---|---|---|---|---|---|---|
| **0** | `PAT-001` | `550e8400-e29b-41d4` | `45.0` | `120.0` | `Emergency` | `E11.9` |
| **1** | `PAT-002` | `6ba7b810-9dad-11d1` | `62.0` | `NaN` *(Missing)* | `Elective` | `I10` |
| **2** | `PAT-003` | `6ba7b811-9dad-11d1` | `71.0` | `145.0` | `Emergency` | `RareCode_X` *(Rare)* |
| **3** | `PAT-004` | `7da7b812-9dad-11d1` | `33.0` | `115.0` | `Urgent` | `E11.9` |
| **4** | `PAT-005` | `8ea7b813-9dad-11d1` | `58.0` | `130.0` | `Emergency` | `I10` |

---

### 2.2 Step 1: Ingestion and Automated Profiling

When `pipeline.fit_transform()` is invoked, the `DatasetProfiler` inspects the data using vectorized operations across all rows.

```python
from src.config.schema import PipelineConfig
from src.profiling.dataset_profiler import DatasetProfiler

config = PipelineConfig.default()
profiler = DatasetProfiler(config=config)
profile = profiler.profile(df, dataset_name="cardio_cohort")
```

#### What the Profiler Identifies:
1. **HIPAA Safe Harbor Matching (`check_hipaa_identifier`):**
   - `patient_id` matches pattern `(^|_)patient_?id($|_)` $\rightarrow$ **FLAGGED** as *HIPAA Identifier* (`Any other unique identifying number or code`).
2. **Uniqueness Ratio & Dtype Inference:**
   - `record_uuid`: String column, `n_unique / n_non_null = 5/5 = 1.0 > 0.95` $\rightarrow$ Inferred as `NEAR_IDENTIFIER`.
   - `age`: Numeric float, non-integer $\rightarrow$ Inferred as `CONTINUOUS`.
   - `systolic_bp`: Numeric float with 20% missing $\rightarrow$ Inferred as `CONTINUOUS`, missingness classified as `MCAR_LIKE`.
   - `admission_type`: 3 unique strings $\le 15$ $\rightarrow$ Inferred as `CATEGORICAL_LOW`.
   - `icd_code`: High cardinality with rare category `RareCode_X` $\rightarrow$ Inferred as `CATEGORICAL_HIGH`.

---

### 2.3 Step 2: Safe Harbor HIPAA & Near-Identifier Column Filtering

Before any model training or encoding, the pipeline determines the active feature set:

```python
# pipeline.py Step 2:
drop_cols = set(
    profile.high_missing_columns +
    profile.near_identifier_columns +
    profile.hipaa_flagged_columns
)
# Result: drop_cols = {'patient_id', 'record_uuid'}
```

`working_df` after dropping non-generative columns:

| row | `age` (Continuous) | `systolic_bp` (Continuous) | `admission_type` (Low-Card) | `icd_code` (High-Card) |
|---|---|---|---|---|
| **0** | `45.0` | `120.0` | `Emergency` | `E11.9` |
| **1** | `62.0` | `NaN` | `Elective` | `I10` |
| **2** | `71.0` | `145.0` | `Emergency` | `RareCode_X` |
| **3** | `33.0` | `115.0` | `Urgent` | `E11.9` |
| **4** | `58.0` | `130.0` | `Emergency` | `I10` |

---

### 2.4 Step 3: Missingness Handling & Indicator Injection

`systolic_bp` has missingness ($\text{rate} = 0.20 \ge 0.01$). 
`MissingnessHandler.fit()` calculates the median of non-null values ($120.0, 145.0, 115.0, 130.0 \rightarrow \text{median} = 125.0$).

`MissingnessHandler.transform()` creates:
1. `systolic_bp__missing_flag`: Binary indicator (1 for row 1, 0 elsewhere).
2. Imputes `NaN` in `systolic_bp` with `125.0`.

| row | `age` | `systolic_bp` | `admission_type` | `icd_code` | `systolic_bp__missing_flag` |
|---|---|---|---|---|---|
| **0** | `45.0` | `120.0` | `Emergency` | `E11.9` | `0` |
| **1** | `62.0` | **`125.0`** *(Imputed)* | `Elective` | `I10` | **`1`** *(Flagged)* |
| **2** | `71.0` | `145.0` | `Emergency` | `RareCode_X` | `0` |
| **3** | `33.0` | `115.0` | `Urgent` | `E11.9` | `0` |
| **4** | `58.0` | `130.0` | `Emergency` | `I10` | `0` |

---

### 2.5 Step 4: Vectorized Feature Transformation & Encoding

Each column is scaled or encoded according to its inferred type:

#### A. Continuous Scaler (`StandardScaler` on `age` and `systolic_bp`):
- `age`: $\mu = 53.8$, $\sigma = 13.17 \rightarrow z = (x - \mu)/\sigma$
- `systolic_bp`: $\mu = 127.0$, $\sigma = 10.30 \rightarrow z = (x - \mu)/\sigma$
- `systolic_bp__missing_flag`: treated as continuous flag scaled to standard range.

#### B. Low-Cardinality Encoder (`OneHotEncoder` on `admission_type`):
- Vocabulary: `[__null__, Emergency, Elective, Urgent]` (4 dimensions)
- Row 0 (`Emergency`) $\rightarrow [0.0, 1.0, 0.0, 0.0]$
- Row 1 (`Elective`) $\rightarrow [0.0, 0.0, 1.0, 0.0]$

#### C. High-Cardinality Encoder (`FrequencyEncoder` on `icd_code`):
- Category counts: `E11.9` (2), `I10` (2), `RareCode_X` (1)
- If `rare_category_min_freq = 2`, `RareCode_X` $\rightarrow$ mapped to `__other__`.
- Vocabulary by frequency: `0: __null__, 1: E11.9, 2: I10, 3: __other__`
- Output shape: `(5, 1)` containing integer tokens: `[1], [2], [3], [1], [2]`.

#### Final Concatenated Float32 Tensor:
Output Shape: `(5 rows, 8 columns)`

```
[
  # age_z,   bp_z,   flag_z,  ohe_null, ohe_emerg, ohe_elect, ohe_urg,  freq_icd
  [ -0.668, -0.680,  -0.500,     0.0,      1.0,       0.0,      0.0,     1.0   ], # Row 0
  [  0.623, -0.194,   2.000,     0.0,      0.0,       1.0,      0.0,     2.0   ], # Row 1
  [  1.306,  1.748,  -0.500,     0.0,      1.0,       0.0,      0.0,     3.0   ], # Row 2
  [ -1.579, -1.165,  -0.500,     0.0,      0.0,       0.0,      1.0,     1.0   ], # Row 3
  [  0.319,  0.291,  -0.500,     0.0,      1.0,       0.0,      0.0,     2.0   ], # Row 4
]
```

---

### 2.6 Step 5: Atomic Schema Registry Persistence

The pipeline commits the entire transformation state into the filesystem:

```
e:\ADVT\registry\cardio_cohort\
├── latest.txt (contains "1")
└── v1\
    ├── profile.json            <-- Human-readable JSON profiling metadata
    └── pipeline_state.joblib   <-- Serialized scalers, vocabularies, missingness state
```

---

### 2.7 Step 6: Synthetic Sampling & Exact Inverse Reconstruction

When Phase 7's generative model generates a new synthetic tensor $\mathbf{X}_{\text{syn}}$:

```python
# Generated array from DP-Diffusion Model
X_syn = np.array([
    [0.623, -0.194, 2.000, 0.0, 0.0, 1.0, 0.0, 2.0]
], dtype=np.float32)

# pipeline.inverse_transform() decodes this in reverse:
synthetic_df = pipeline.inverse_transform(X_syn)
```

1. **Scaler Inversion:**
   - $x_{\text{age}} = 0.623 \times 13.17 + 53.8 = 62.0$
   - $x_{\text{bp}} = -0.194 \times 10.30 + 127.0 = 125.0$
2. **Encoder Inversion:**
   - $\text{argmax}([0.0, 0.0, 1.0, 0.0]) = 2 \rightarrow \text{"Elective"}$
   - Token index $2.0 \rightarrow \text{"I10"}$
3. **Missingness Inversion:**
   - `systolic_bp__missing_flag` evaluated: index 2.0 corresponds to `was_missing = True`.
   - `systolic_bp` at row 0 is replaced with `np.nan`.
   - `systolic_bp__missing_flag` column is dropped.

#### Reconstructed Synthetic Output:

| `age` | `systolic_bp` | `admission_type` | `icd_code` |
|---|---|---|---|
| `62.0` | `NaN` | `Elective` | `I10` |

> **Result:** Exact clinical structure, data types, and missingness characteristics are preserved without leaking raw identifiers!

---

## 3. Full Python Code Example (Copy-Paste Ready)

Here is the complete script demonstrating the full lifecycle from raw data to encoded tensors, registry persistence, and decoding:

```python
"""
demo_pipeline_workflow.py — Complete runnable walkthrough of Phases 1–3
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config.schema import PipelineConfig
from src.profiling.dataset_profiler import DatasetProfiler
from src.preprocessing.scalers import StandardScaler
from src.preprocessing.encoders import OneHotEncoder, FrequencyEncoder
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.registry.schema_registry import FileSchemaRegistry


def run_complete_demo():
    print("==================================================================")
    print("  PRIVACY-PRESERVING SYNTHETIC DATA PIPELINE DEMO (PHASES 1–3)   ")
    print("==================================================================")

    # 1. Instantiate Frozen Configuration
    config = PipelineConfig.default()
    registry_path = Path("./demo_registry")

    # 2. Define Encoder and Scaler Factories (Injected Dependencies)
    def encoder_factory(col_name: str):
        # High cardinality or specific feature custom routing can happen here
        if col_name == "icd_code":
            return FrequencyEncoder(min_freq=2)
        return OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq)

    def scaler_factory(col_name: str):
        return StandardScaler()

    # 3. Assemble Pipeline via Dependency Injection
    pipeline = PreprocessingPipeline(
        config=config,
        profiler=DatasetProfiler(config=config),
        missingness_handler=MissingnessHandler(config=config),
        encoder_factory=encoder_factory,
        scaler_factory=scaler_factory,
        registry=FileSchemaRegistry(root_dir=registry_path),
    )

    # 4. Create Sample Healthcare Dataset with Realistic Anomalies
    n = 200
    rng = np.random.default_rng(42)
    
    raw_df = pd.DataFrame({
        "patient_id": [f"PID_{i:04d}" for i in range(n)],              # HIPAA identifier
        "record_uuid": [f"uuid_{i}_{rng.integers(1000, 9999)}" for i in range(n)], # Near-identifier
        "age": rng.normal(55, 12, size=n),                            # Continuous
        "systolic_bp": rng.normal(125, 15, size=n),                   # Continuous w/ missing
        "admission_type": rng.choice(["Emergency", "Elective", "Urgent"], size=n), # Categorical Low
        "icd_code": rng.choice(["E11.9", "I10", "J45.0", "Rare_X"], p=[0.45, 0.45, 0.08, 0.02], size=n)
    })
    
    # Inject 15% missingness into systolic_bp
    missing_indices = rng.choice(n, size=30, replace=False)
    raw_df.loc[missing_indices, "systolic_bp"] = np.nan

    print(f"\n[1] Raw Input DataFrame Created: {raw_df.shape[0]} rows x {raw_df.shape[1]} cols")
    print("    Sample Input Row 0:")
    print(f"    {raw_df.iloc[0].to_dict()}")

    # 5. Execute fit_transform
    dataset_name = "cardiology_registry_demo"
    encoded_tensor = pipeline.fit_transform(raw_df, dataset_name=dataset_name)

    print(f"\n[2] Pipeline fit_transform Completed:")
    print(f"    Encoded Tensor Shape: {encoded_tensor.shape} (dtype: {encoded_tensor.dtype})")
    print(f"    Total Encoded Feature Dimensions: {encoded_tensor.shape[1]}")
    print(f"    Are there any NaNs in encoded tensor? {np.isnan(encoded_tensor).any()} (Must be False)")

    # 6. Verify Registry Output on Disk
    registry = FileSchemaRegistry(root_dir=registry_path)
    entry = registry.load(dataset_name)
    print(f"\n[3] Registry Verified:")
    print(f"    Dataset '{entry.dataset_name}' stored under Version: v{entry.version}")
    print(f"    Active Training Columns: {entry.training_columns}")
    print(f"    HIPAA Dropped Columns: {entry.profile.hipaa_flagged_columns}")
    print(f"    Near-Identifier Dropped: {entry.profile.near_identifier_columns}")

    # 7. Simulate Downstream Generation & Inverse Transform
    print("\n[4] Simulating Generative Model Output & Inverse Decoding:")
    # Take encoded row 1 (which had missing systolic_bp) as a synthetic sample
    synthetic_sample = encoded_tensor[missing_indices[0]: missing_indices[0] + 1]
    
    decoded_df = pipeline.inverse_transform(synthetic_sample)
    print("    Decoded Synthetic Record:")
    print(f"    {decoded_df.iloc[0].to_dict()}")
    
    # Check that NaN was correctly reconstructed
    assert pd.isna(decoded_df["systolic_bp"].iloc[0]), "Error: Missingness was not reconstructed as NaN!"
    print("    Exact NaN Position Reconstructed Successfully: TRUE")

    print("\n==================================================================")
    print("  ALL DEMO STEPS EXECUTED AND VERIFIED SUCCESSFULLY!             ")
    print("==================================================================")


if __name__ == "__main__":
    run_complete_demo()
```

---

## 4. Intermediate State Representation Reference

To assist debugging in Phases 4–10, the table below documents the exact mathematical form of data at each stage:

| Stage | Data Structure | Data Type | Null Representation | Description |
|---|---|---|---|---|
| **Raw Input** | `pd.DataFrame` | Mixed (`object`, `float64`, `int64`) | `np.nan`, `None` | Unsanitized input from clinical CSVs or databases. |
| **Filtered DataFrame** | `pd.DataFrame` | Mixed | `np.nan` | HIPAA and Near-ID columns removed. |
| **Imputed & Flagged** | `pd.DataFrame` | Mixed + `int8` flag columns | No nulls (temporarily imputed) | Missing values replaced with column median/mode; flag column injected. |
| **Encoded Tensor** | `np.ndarray` (2D) | `np.float32` | None (0.0 or normalized float) | Continuous values standardized to $\mathcal{N}(0, 1)$; categoricals one-hot or token indexed. |
| **Decoded DataFrame** | `pd.DataFrame` | Original domain dtypes | `np.nan` | Exact match to input schema with reconstructed missingness. |

---

## 5. Connecting to Downstream Phases (4–10)

```
                       ┌────────────────────────────────────────┐
                       │ Phase 1–3: Preprocessing & Registry   │
                       │ (Active & Tested Baseline)             │
                       └──────────────────┬─────────────────────┘
                                          │
                                          ▼
                       ┌────────────────────────────────────────┐
                       │ Phase 4 & 5: Auto-Config & Risk Tiers  │
                       │ - Reads profile.json from Registry     │
                       │ - Assigns Tier 1–4 Privacy Budgets     │
                       └──────────────────┬─────────────────────┘
                                          │
                                          ▼
                       ┌────────────────────────────────────────┐
                       │ Phase 6: Checkpointing & Resumption    │
                       │ - Preserves Privacy Accountant State   │
                       │ - Enforces Atomic Kaggle Step Checkpoints│
                       └──────────────────┬─────────────────────┘
                                          │
                                          ▼
                       ┌────────────────────────────────────────┐
                       │ Phase 7: DP-Diffusion Generative Loop  │
                       │ - Novelty A: Adaptive Noise Schedule   │
                       │ - Novelty B: Per-Tier DP SGD Training  │
                       │ - Samples new Float32 Tensors          │
                       └──────────────────┬─────────────────────┘
                                          │
                                          ▼
                       ┌────────────────────────────────────────┐
                       │ Phase 8 & 9: Evaluation & Verification │
                       │ - Calls pipeline.inverse_transform()   │
                       │ - Assesses Empirical Privacy & Utility │
                       └────────────────────────────────────────┘
```

- **Phase 4 (Auto-Config Engine):** Calls `registry.load_profile(name)` to inspect `ColumnProfile` definitions and assign feature columns into privacy tiers without re-profiling the dataset.
- **Phase 6 (Checkpointing System):** Uses the atomic file-write patterns demonstrated in `FileSchemaRegistry` to persist the model and privacy accountant states under Kaggle session limits.
- **Phase 7 (Generative Modeling):** Ingests the 2D `float32` array from `pipeline.fit_transform()` directly into PyTorch DataLoaders wrapped with Opacus's `PrivacyEngine`.
- **Phase 8 (Fidelity & Privacy Auditing):** Passes model-generated tensors directly to `pipeline.inverse_transform()` to evaluate statistical fidelity, Jensen-Shannon distance, and membership inference resistance on human-interpretable patient records.
