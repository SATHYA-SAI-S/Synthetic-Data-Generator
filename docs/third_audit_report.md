# Third Audit Report — Full-System Logical & Structural Audit
## Privacy-Preserving Synthetic Healthcare Data Generation Framework (Phases 1–7)

---

> **Document Type:** Read-Only Audit Report  
> **Scope:** Every file in the repository — `src/` (all modules), `scripts/`, `environment/`, `patch6_config.py`, `docs/`  
> **Constraint:** No source code was modified. This is an analysis-only deliverable.  
> **Date:** 2026-08-18

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Findings](#2-critical-findings)
3. [High-Severity Findings](#3-high-severity-findings)
4. [Medium-Severity Findings](#4-medium-severity-findings)
5. [Low-Severity & Hygiene Findings](#5-low-severity--hygiene-findings)
6. [Verified-Correct Components](#6-verified-correct-components)
7. [Severity-Ranked Issue Register](#7-severity-ranked-issue-register)
8. [Prioritized Remediation Roadmap](#8-prioritized-remediation-roadmap)
9. [Appendix: Files Audited](#9-appendix-files-audited)

---

## 1. Executive Summary

This audit examines **every file** in the repository after all phases were reported complete. The audit found:

- **3 Critical** issues — the system **cannot run end-to-end** in its current state.
- **5 High** issues — silent data corruption or guaranteed test failures.
- **7 Medium** issues — correctness/robustness gaps.
- **10 Low/hygiene** issues — maintainability and consistency.

**Key headline:** The only end-to-end entry point (`scripts/reproduce_end_to_end.py`) is incompatible with the actual implemented APIs at **4 separate call sites** — it cannot execute at all. Additionally, the DP-SGD trainer's tier-parameter mechanism is **silently broken**: no noise is ever added and no privacy accounting is ever recorded, meaning the DP guarantee is absent while the code appears to provide it.

**Positive findings:** The second audit's critical R-01 (B-03/B-14 conflict) and high R-02 (integer near-identifier reversal) are **both resolved** in the current code. The registry's atomic-write and width-validation issues (R-05, R-08) are also fixed.

---

## 2. Critical Findings

### C-1: `scripts/reproduce_end_to_end.py` is incompatible with the actual API — the system cannot run end-to-end

**File:** `scripts/reproduce_end_to_end.py`

The only end-to-end entry point calls APIs that do not match the implemented signatures. **Four separate call sites are broken:**

| Line | Code in script | Actual API | Result |
|------|---------------|-----------|--------|
| 46 | `profiler.profile(raw_df)` | `profile(df, dataset_name)` — 2 required args | `TypeError: profile() missing 1 required positional argument` |
| 49 | `FileSchemaRegistry(base_dir=...)` | `__init__(root_dir: Path)` | `TypeError: unexpected keyword argument 'base_dir'` |
| 50 | `PreprocessingPipeline(config, registry, dataset_name=...)` | `__init__(config, profiler, missingness_handler, encoder_factory, scaler_factory, registry)` | `TypeError: missing 4 required positional arguments` |
| 52 | `pipeline.fit_transform(raw_df, profile)` | `fit_transform(df, dataset_name: str)` | Passing `DatasetProfile` object where a string is required |

**Impact:** The script cannot be patched with one-line fixes — its architecture (registry as 2nd ctor arg, no profiler/handler/factories) doesn't match `PreprocessingPipeline`'s dependency-injection design. It requires a **rewrite** against the real pipeline API.

**Additional issues in the same script:**
- Line 119: default data path `data/diabetes_data.csv` does not exist — the actual file is `data/diabetes+130-us+hospitals+for+years+1999-2008.zip`.
- Line 72: `base_sigma=1.5 / target_eps` — for `target_eps=10.0`, sigma = 0.15, which is extremely low noise; the DP guarantee at that sigma over 1000 timesteps will be meaningless.
- No JSON report of epsilon/utility/privacy results — only CSVs are written, losing sweep metrics.

---

### C-2: `DPTrainer` tier-parameter split is silently broken — no DP noise is ever applied

**Files:** `src/privacy/dp_trainer.py`, `src/privacy/clip_and_noise.py`

**The failure chain:**

1. `DPTrainer.__init__` (line 34) wraps the denoiser: `self.denoiser = GradSampleModule(denoiser).to(device)`.
2. The caller extracts `tier_params` from the **original** denoiser: `list(denoiser.parameters())` (e.g., `scripts/reproduce_end_to_end.py:66`, `test_dp_trainer_fixed_sigma.py:30`).
3. Opacus's `GradSampleModule` **copies** the module and its parameters. The original parameter objects never receive `.grad_sample`.
4. In `clip_and_noise_tier` (line 30): `for p in params if hasattr(p, "grad_sample")` → **empty list** for every tier.
5. Line 33: `if not per_param_norms: return` — **silent early return**.
6. **No noise is added. No accountant step is recorded. The DP guarantee is absent.**

**Evidence this is real (tests will fail):**
- `test_dp_trainer_fixed_sigma.py:48` asserts `accountant.steps == 4` — but `clip_and_noise_tier` no-ops, so steps will be **0**.
- `test_dp_trainer_adaptive.py:51` asserts `p.grad is not None` on the **original** `denoiser.parameters()` — those never get gradients (the wrapped module's do).
- `test_per_tier_heads.py:71` asserts `accountant.steps == 4` — same failure.

**Impact:** The entire Phase 5 DP guarantee is silently absent. The framework produces synthetic data with **no differential privacy**, while the code and docs claim DP protection.

---

### C-3: `test_full_reproducibility.py` is a no-op test that masks C-1

**File:** `src/tests/test_full_reproducibility.py`

```python
try:
    run_sweep(data_path=str(data_path), output_dir=str(output_dir))
except Exception as e:
    # Opacus strict hooks might fail on CPU without proper requires_grad in integration
    # We acknowledge the architectural run completion
    pass

# Assert outputs were generated
# Since we passed a tiny dataframe, if it didn't crash before saving:
# assert os.path.exists(output_dir / "gpu_state.json")
pass
```

- Every exception is swallowed with `except Exception: pass`.
- The final "assertion" is a commented-out line followed by `pass`.
- The test provides **zero** verification value and actively **masks** the C-1 breakage.

**Impact:** CI would report this test as "passing" while the end-to-end script is completely broken.

---

## 3. High-Severity Findings

### H-1: DP trainer tests will fail (consequence of C-2)

**Files:** `src/tests/privacy/test_dp_trainer_fixed_sigma.py`, `src/tests/privacy/test_dp_trainer_adaptive.py`, `src/tests/privacy/test_per_tier_heads.py`

All three tests are built on the broken tier-param mechanism. Their expected accountant-step counts and gradient assertions cannot hold. Running the test suite will produce failures in these files.

---

### H-2: `forward_diffuse` device mismatch on GPU

**File:** `src/diffusion/forward_process.py:26`

```python
a_t = alphas_cumprod[t].view(-1, 1)
```

`schedule.get_alphas_cumprod()` returns tensors on `schedule._device` (default CPU). In `DPTrainer`/`DiffusionTrainer`, `x_0` is moved to `self.device`. If a GPU is used and the schedule was constructed without `device=device`, `alphas_cumprod[t]` (CPU) × `x_0` (CUDA) → **runtime device-mismatch crash**.

**Affected callers:**
- `src/diffusion/trainer.py:54` — `forward_diffuse(x_0, t, self.schedule)` where `x_0` is on `self.device`
- `src/privacy/dp_trainer.py:61` — same pattern

**Fix direction:** `forward_diffuse` should move `alphas_cumprod` to `x_0.device` internally, or callers must construct the schedule with the same device.

---

### H-3: `HeuristicRiskTierAssigner` uses string-min for tier comparison

**File:** `src/privacy/risk_tier_assigner.py:58`

```python
tighter = min(t1, t2)
```

This relies on lexicographic ordering of `"Tier1" < "Tier2" < "Tier3"`. It works today but breaks the moment tier names change (e.g., `"Tier10"`). Should use an explicit tier-rank mapping.

---

### H-4: `OneHotEncoder.transform` Python row loop still present

**File:** `src/preprocessing/encoders.py:107-121`

Despite `docs/verification_fixes_report.md` claiming both encoders were vectorized, `OneHotEncoder.transform` still iterates `for i, val in enumerate(series)` in Python. This is the documented O(n) performance bottleneck — 10–50× slower than vectorized on >100K rows.

---

### H-5: Inconsistent unseen-value handling between encoders

**Files:** `src/preprocessing/encoders.py`

| Encoder | Unseen value at inference | Decodes to |
|---------|--------------------------|------------|
| `OneHotEncoder` (line 117-121) | Mapped to `__null__` index | **NaN** — silently corrupts data |
| `FrequencyEncoder` (line 235) | Mapped to `__other__` index | `"__other__"` label |

Two different conventions for the same semantic (unknown category), leading to different round-trip behavior depending on which encoder was chosen. The `OneHotEncoder` behavior is particularly dangerous — an unseen category silently becomes a missing value.

---

## 4. Medium-Severity Findings

### M-1: `GPUBudgetGuard` counts wall-clock, not GPU-active time

**File:** `src/orchestration/gpu_budget_guard.py:16`

`session_start` is set in `__init__`; any idle time between checks counts against the budget. A 30-hr budget can be exhausted by an overnight idle process. The class name says "GPU" budget but it measures wall-clock.

---

### M-2: `MissingnessHandler.inverse_transform` no validation of indicator dtype

**File:** `src/preprocessing/missingness.py:184`

```python
was_missing_mask = result[indicator_col] >= 0.5
```

If a generative model produces soft values (e.g., `0.3`/`0.7`) instead of hard 0/1, NaN restoration happens at non-integer thresholds with no warning. The contract assumes hard 0/1 but nothing enforces it.

---

### M-3: `_infer_dtype` integer detection is fragile

**File:** `src/profiling/dataset_profiler.py:195-199`

```python
if pd.api.types.is_integer_dtype(non_null) or (
    non_null.dropna().apply(float.is_integer).all()
    if hasattr(non_null.iloc[0], "is_integer")
    else np.all(non_null.values == non_null.values.astype(int))
):
```

- `hasattr(non_null.iloc[0], "is_integer")` is brittle — depends on the scalar type.
- `np.all(... astype(int))` can overflow for large floats (e.g., `1e20` → `astype(int)` wraps).
- Should use a deterministic vectorized check on the already-coerced numeric series.

---

### M-4: `risk_tier_assigner` correlation guard is O(n²) memory

**File:** `src/privacy/risk_tier_assigner.py:44-48`

`pd.factorize` on all columns + `.corr().abs()` constructs an n_cols × n_cols matrix. For the diabetes dataset (~50 cols) this is fine, but for wide clinical datasets (hundreds of columns) this is 2.5M+ matrix entries.

---

### M-5: `PreprocessingPipeline.transform()` silently ignores extra columns

**File:** `src/preprocessing/pipeline.py:258-291`

Extra columns in `transform()` input that weren't in `_training_columns` are silently skipped. Should at least log a warning (B-13 from second audit partially unfixed).

---

### M-6: SHA-256 checksum verification silently skipped if `.sha256` absent

**File:** `src/registry/schema_registry.py:169`

```python
if hash_path.exists():
```

Deleting the `.sha256` file bypasses verification entirely. This protects against accidental corruption but not tampering. Should either require the file or log a warning when absent.

---

### M-7: `src/tests/evaluation/` contains only an empty `__init__.py` — no evaluation tests

**Directory:** `src/tests/evaluation/`

No tests exist for `PrivacyEvaluator` or `UtilityEvaluator`, despite those being Phase 7/8 critical components. The `evaluation/` module is completely untested.

---

## 5. Low-Severity & Hygiene Findings

### L-1: `schema.py:257` — blank line between `@classmethod` and `def default`

**File:** `src/config/schema.py:257-259`

```python
    @classmethod

    def default(cls) -> "PipelineConfig":
```

Valid Python but stylistically risky — easily misread as a decorator error. Also `DtypeInferenceConfig` has a double blank line at lines 165-166.

---

### L-2: `encoders.py:132` — stray character in docstring

**File:** `src/preprocessing/encoders.py:132`

```python
Uses argmax ?" ties broken by lowest index (i.e., __null__).
```

Stray `?"` in the docstring.

---

### L-3: `patch6_config.py` uses brittle string-replacement to mutate `schema.py`

**File:** `patch6_config.py`

```python
schema_code = schema_code.replace("class PipelineConfig(BaseModel):", ...)
schema_code = schema_code.replace("    @classmethod", ...)
```

Destructive on any future schema edits. If `schema.py` is reformatted, the replacements silently no-op or corrupt. Should be deprecated in favor of direct edits.

---

### L-4: Leftover scratch directories at repo root

**Directories:** `registry_test/`, `registry_test2/`, `registry_test3/`, `registry_test4/`

Unused test scratch directories with no documentation. Should be removed or moved under `tests/`.

---

### L-5: No `pytest.ini` / `pyproject.toml`

Package is not installable; tests rely on cwd being repo root. No linting/type-checking config (ruff/mypy absent).

---

### L-6: `test_risk_tier_assigner.py` uses unseeded `np.random`

**File:** `src/tests/privacy/test_risk_tier_assigner.py:10`

```python
'diagnosis': np.random.randint(0, 10, 100),
```

Unseeded random → flaky test.

---

### L-7: Inconsistent protocol pattern — `Protocol` vs `ABC`

**Files:** `src/privacy/base.py` uses `typing.Protocol` (structural, no runtime enforcement) while the rest of the codebase (`src/preprocessing/base.py`, `src/profiling/base.py`, `src/registry/base.py`, `src/diffusion/base.py`) uses `ABC`. Inconsistent pattern.

---

### L-8: `setup_check.py` CPU-only hard-fail

**File:** `environment/setup_check.py:114-115`

Missing GPU → `passed=False` → overall report `SOME CHECKS FAILED`, even on CPU-only machines where CPU training is "technically possible" per the code comment. Should be a warning, not a failure.

---

### L-9: `reproduce_end_to_end.py` default data path doesn't exist

**File:** `scripts/reproduce_end_to_end.py:119`

```python
run_sweep(data_path="data/diabetes_data.csv", output_dir="outputs/sweep_results")
```

The actual file is `data/diabetes+130-us+hospitals+for+years+1999-2008.zip` — the CSV path doesn't exist.

---

### L-10: `run_sweep` has no JSON report of sweep results

**File:** `scripts/reproduce_end_to_end.py`

Only CSVs are written. No JSON report of epsilon spent, utility metrics, or privacy metrics per sweep iteration.

---

## 6. Verified-Correct Components

The following were verified correct in the current code:

### ✅ Resolved from Second Audit

| ID | Issue | Resolution |
|----|-------|-----------|
| **R-01** | B-03 `fillna("__null__")` triggers B-14 `ValueError` | **Resolved** — `pipeline.py:200` now passes raw series to encoders (no more `fillna("__null__")`) |
| **R-02** | Integer near-identifier reversal drops legitimate features | **Resolved** — `dataset_profiler.py:176` now guards with `not is_numeric` |
| **R-05** | `latest.txt` write not atomic | **Resolved** — `schema_registry.py:149-151` uses temp+`os.replace` |
| **R-08** | No width validation in `inverse_transform` | **Resolved** — `pipeline.py:307-316` validates expected width |

### ✅ Correct by Inspection

- **`MissingnessHandler`** — round-trip contract correct (NaN restore via indicator, vectorized, no mutation of input).
- **`StandardScaler` / `MinMaxScaler` / `RobustScaler`** — exact inverse, NaN propagation, constant-column guards all correct.
- **Registry** — atomic write with temp dir + `os.replace`, SHA-256 integrity, versioning, path-traversal validation all correct.
- **`PreprocessingPipeline`** — dependency injection pattern clean; O(1) profile lookup; constant-column drop; HIPAA/near-identifier drop logic correct.
- **HIPAA regex** — 18 categories covered with boundary-anchored patterns; tested.
- **`DatasetProfiler`** — vectorized stats; structural missingness with early break; serializable Pydantic profile.
- **`LinearNoiseSchedule`** — correct DDPM linear beta schedule.
- **`MLPDenoiser`** — correct time-embedding injection; shape contract verified.
- **`generate_samples`** — correct DDPM reverse-process update rule.
- **`DiffusionTrainer`** — correct non-DP training loop.
- **`CentralPrivacyAccountant`** — correct Opacus RDP wrapper.
- **`AdaptiveNoiseSchedule`** — correct linear/constant strategies; monotonicity verified.
- **`PrivacyEvaluator` / `UtilityEvaluator`** — correct D-MIA and TVD/KS/correlation-RMSE implementations.
- **`setup_check.py`** — comprehensive environment verification with structured report.

---

## 7. Severity-Ranked Issue Register

| ID | Severity | File | Line(s) | Description |
|----|----------|------|---------|-------------|
| **C-1** | **Critical** | `scripts/reproduce_end_to_end.py` | 46, 49, 50, 52 | API incompatibility at 4 call sites — script cannot run |
| **C-2** | **Critical** | `src/privacy/dp_trainer.py`, `clip_and_noise.py` | 34, 30-33 | Tier params extracted pre-wrapping → no `.grad_sample` → no noise, no accounting — DP guarantee absent |
| **C-3** | **Critical** | `src/tests/test_full_reproducibility.py` | 35-49 | No-op test swallows all exceptions; masks C-1 |
| **H-1** | **High** | `src/tests/privacy/test_dp_trainer_*.py`, `test_per_tier_heads.py` | all | Tests will fail due to C-2 |
| **H-2** | **High** | `src/diffusion/forward_process.py` | 26 | Device mismatch on GPU — schedule tensors on CPU × data on CUDA |
| **H-3** | **High** | `src/privacy/risk_tier_assigner.py` | 58 | String-min tier comparison — fragile to tier-name changes |
| **H-4** | **High** | `src/preprocessing/encoders.py` | 107-121 | OneHotEncoder row loop not vectorized despite doc claim |
| **H-5** | **High** | `src/preprocessing/encoders.py` | 117-121, 235 | Inconsistent unseen-value handling — OneHot maps to NaN, Frequency maps to `__other__` |
| **M-1** | Medium | `src/orchestration/gpu_budget_guard.py` | 16 | Wall-clock budget, not GPU-active time |
| **M-2** | Medium | `src/preprocessing/missingness.py` | 184 | No validation of indicator dtype in inverse_transform |
| **M-3** | Medium | `src/profiling/dataset_profiler.py` | 195-199 | Fragile integer detection — `hasattr` + potential overflow |
| **M-4** | Medium | `src/privacy/risk_tier_assigner.py` | 44-48 | O(n²) correlation matrix memory |
| **M-5** | Medium | `src/preprocessing/pipeline.py` | 258-291 | Extra columns silently ignored in transform() |
| **M-6** | Medium | `src/registry/schema_registry.py` | 169 | Checksum verification silently skipped if `.sha256` absent |
| **M-7** | Medium | `src/tests/evaluation/` | — | No evaluation tests exist |
| **L-1** | Low | `src/config/schema.py` | 257 | Blank line between `@classmethod` and `def` |
| **L-2** | Low | `src/preprocessing/encoders.py` | 132 | Stray `?"` in docstring |
| **L-3** | Low | `patch6_config.py` | 38, 59 | Brittle string-replacement schema mutation |
| **L-4** | Low | `registry_test/` – `registry_test4/` | — | Leftover scratch directories |
| **L-5** | Low | repo root | — | No `pytest.ini` / `pyproject.toml` / linting config |
| **L-6** | Low | `src/tests/privacy/test_risk_tier_assigner.py` | 10 | Unseeded `np.random` → flaky test |
| **L-7** | Low | `src/privacy/base.py` | 11, 23 | `Protocol` vs `ABC` inconsistency |
| **L-8** | Low | `environment/setup_check.py` | 114-115 | CPU-only hard-fail |
| **L-9** | Low | `scripts/reproduce_end_to_end.py` | 119 | Default data path doesn't exist |
| **L-10** | Low | `scripts/reproduce_end_to_end.py` | — | No JSON sweep report |

---

## 8. Prioritized Remediation Roadmap

### P0 — Fix Immediately (Correctness / Security)

1. **C-2:** Fix `DPTrainer` tier-param mechanism. Extract `tier_params` from the **wrapped** `GradSampleModule` (`self.denoiser`), not the original denoiser. This restores the DP guarantee.
2. **C-1:** Rewrite `scripts/reproduce_end_to_end.py` against the real pipeline API:
   - `profiler.profile(raw_df, dataset_name)`
   - `FileSchemaRegistry(root_dir=Path(...))`
   - `PreprocessingPipeline(config, profiler, missingness_handler, encoder_factory, scaler_factory, registry)`
   - `pipeline.fit_transform(raw_df, dataset_name)`
3. **C-3:** Rewrite `test_full_reproducibility.py` to actually assert outputs exist and the pipeline completes.

### P1 — Fix Soon (Correctness / Performance)

4. **H-2:** Move `alphas_cumprod` to `x_0.device` inside `forward_diffuse`.
5. **H-4:** Vectorize `OneHotEncoder.transform` (replace the Python row loop with the same `np.arange` indexing used in `FrequencyEncoder`).
6. **H-5:** Unify unseen-value handling — map unseen values to `__other__` in both encoders (or document the difference explicitly).
7. **H-3:** Replace string-min with explicit tier-rank mapping.
8. **M-3:** Replace fragile integer detection with a deterministic vectorized check.
9. **M-5:** Add warning for extra columns in `transform()`.
10. **M-6:** Log a warning when `.sha256` file is absent.

### P2 — Hygiene

11. **M-1:** Track GPU-active time (e.g., via `torch.cuda.synchronize` timestamps) or rename to `WallClockBudgetGuard`.
12. **M-2:** Validate indicator column values are in {0, 1} before thresholding.
13. **M-4:** Use a sampled correlation matrix or skip the guard for very wide datasets.
14. **M-7:** Add tests for `PrivacyEvaluator` and `UtilityEvaluator`.
15. **L-1/L-2:** Fix formatting issues in `schema.py` and `encoders.py`.
16. **L-3:** Deprecate `patch6_config.py`.
17. **L-4:** Remove scratch directories.
18. **L-5:** Add `pytest.ini` + `pyproject.toml` + linting config.
19. **L-6:** Seed `np.random` in `test_risk_tier_assigner.py`.
20. **L-7:** Convert `Protocol` to `ABC` in `privacy/base.py` for consistency.
21. **L-8:** Make GPU check a warning, not a failure.
22. **L-9/L-10:** Fix default data path; add JSON sweep report.

---

## 9. Appendix: Files Audited

### Source (`src/`)
| Module | Files |
|--------|-------|
| `config/` | `__init__.py`, `schema.py` |
| `preprocessing/` | `__init__.py`, `base.py`, `encoders.py`, `missingness.py`, `pipeline.py`, `scalers.py` |
| `privacy/` | `__init__.py`, `accountant.py`, `adaptive_schedule.py`, `base.py`, `clip_and_noise.py`, `dp_trainer.py`, `risk_tier_assigner.py` |
| `diffusion/` | `__init__.py`, `base.py`, `denoiser.py`, `forward_process.py`, `sampler.py`, `schedule.py`, `trainer.py` |
| `evaluation/` | `__init__.py`, `privacy_metrics.py`, `utility_metrics.py` |
| `orchestration/` | `gpu_budget_guard.py` |
| `registry/` | `__init__.py`, `base.py`, `schema_registry.py` |
| `profiling/` | `__init__.py`, `base.py`, `dataset_profiler.py` |
| `tests/` | `__init__.py`, `conftest.py`, `test_encoders_inverse.py`, `test_full_reproducibility.py`, `test_pipeline_integration.py`, `test_profiler.py`, `test_registry_roundtrip.py` |
| `tests/diffusion/` | `test_denoiser_shape.py`, `test_forward_process.py`, `test_trainer_smoke.py` |
| `tests/privacy/` | `test_accountant_single_source.py`, `test_adaptive_schedule_shape.py`, `test_dp_trainer_adaptive.py`, `test_dp_trainer_fixed_sigma.py`, `test_per_tier_heads.py`, `test_risk_tier_assigner.py` |
| `tests/evaluation/` | `__init__.py` (empty — no tests) |

### Scripts & Environment
| File | Status |
|------|--------|
| `scripts/reproduce_end_to_end.py` | **Critical — incompatible with API** |
| `scripts/reproduce_end_to_end.sh` | OK (calls the broken script) |
| `environment/requirements.txt` | OK — pinned versions |
| `environment/setup_check.py` | OK — comprehensive; CPU-only hard-fail noted |
| `patch6_config.py` | Low — brittle string-replacement |

### Docs
| File | Status |
|------|--------|
| `docs/architecture.md` | Reference |
| `docs/code_review_and_architecture_analysis.md` | Reference |
| `docs/phase1-3_completion_report.md` | Reference |
| `docs/phase4_baseline_report.md` | Reference |
| `docs/phase5_dp_report.md` | Reference |
| `docs/phase6_sweep_report.md` | Reference |
| `docs/phase7_eval_report.md` | Reference |
| `docs/problems_and_resolutions.md` | Reference |
| `docs/second_audit_report.md` | Prior audit — findings re-verified |
| `docs/verification_fixes_report.md` | Contains claims contradicted by code (H-4) |
| `docs/workflow_and_examples.md` | Reference |
| `docs/novelty_and_scope.md` | Reference |
| `docs/environment_notes.md` | Reference |

---

*End of third audit. No source code was modified during this review.*