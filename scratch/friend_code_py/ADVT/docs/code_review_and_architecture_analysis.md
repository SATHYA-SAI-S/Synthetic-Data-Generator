# Complete Structural & Architectural Analysis
## Privacy-Preserving Synthetic Healthcare Data Generation Framework — Phases 1–3

---

> **Document Type:** Code Review & Architecture Analysis  
> **Scope:** All source files under `src/`, `environment/`, and `docs/`  
> **Method:** Static analysis — logical, technical, efficiency, and security review  
> **Constraint:** No source code was modified. This is a read-only analysis.  
> **Date:** 2026-08-18

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Structural & Architectural Analysis](#2-structural--architectural-analysis)
3. [Logical & Correctness Debugging (Bugs Found)](#3-logical--correctness-debugging-bugs-found)
4. [Efficiency Analysis](#4-efficiency-analysis)
5. [Security & Vulnerability Analysis](#5-security--vulnerability-analysis)
6. [Test & Documentation Drift](#6-test--documentation-drift)
7. [Severity-Ranked Bug Register](#7-severity-ranked-bug-register)
8. [Complexity & Maintainability Assessment](#8-complexity--maintainability-assessment)
9. [Prioritized Recommendations](#9-prioritized-recommendations)
10. [Appendix: Files Reviewed](#10-appendix-files-reviewed)

---

## 1. Executive Summary

The codebase implements a well-structured, interface-first preprocessing pipeline for a privacy-preserving synthetic healthcare data framework. The architecture is genuinely strong: dependency injection throughout, ABC-first design, mandatory `inverse_transform` contracts, atomic registry writes, and a solid test suite.

However, the analysis surfaced **1 critical correctness bug**, **3 high-severity issues**, and **multiple medium/low findings** that would silently corrupt downstream phases (Phase 7 diffusion training and Phase 8 evaluation) if not addressed. The most serious is a **column-cursor desynchronization** in `PreprocessingPipeline.inverse_transform` that silently mis-decodes every column after a failed encode. A second confirmed issue is **NaN values being converted to the literal string `"nan"`** for low-missingness categorical columns, breaking the round-trip contract.

Security findings include a **path-traversal / arbitrary file-write** vector via unsanitized `dataset_name`, and **untrusted `joblib.load()`** (pickle-based RCE risk).

---

## 2. Structural & Architectural Analysis

### 2.1 Layered Architecture

The codebase follows a clean 4-layer architecture with a strict dependency direction:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Config (src/config/schema.py)                     │
│  Frozen Pydantic models — single source of truth for all    │
│  numeric thresholds. No business logic.                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ imported by all layers
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 2: Profiling (src/profiling/)                        │
│  AbstractProfiler ← DatasetProfiler                         │
│  Produces typed, serializable DatasetProfile                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 3: Preprocessing (src/preprocessing/)                │
│  AbstractScaler/Encoder/MissingnessHandler ← concrete impls │
│  PreprocessingPipeline orchestrates (DI-injected)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 4: Registry (src/registry/)                          │
│  AbstractSchemaRegistry ← FileSchemaRegistry                │
│  Atomic, versioned persistence of fitted state              │
└─────────────────────────────────────────────────────────────┘
```

**Strengths:**
- **Dependency Inversion Principle** correctly applied — `pipeline.py` imports only ABCs, never concrete classes.
- **I/O at the boundary** — only `fit_transform_from_file` touches the filesystem.
- **Frozen config** prevents accidental mutation of thresholds.
- **Interface-first** design enables Phase 4/5 injection without touching the orchestrator.

**Weaknesses / Leaks:**
- **Empty `__init__.py` files** — no package-level exports. Consumers must know deep import paths (`from src.preprocessing.encoders import OneHotEncoder`). No `pyproject.toml`/`setup.py` means the package is not installable; tests only run from repo root.
- **`pipeline.py` reaches into `_profile.columns`** via a list comprehension + `column_by_name` (see §4.1) — a performance leak of the abstraction.
- **`_encoded_col_names` ordering** is the implicit contract between `fit_transform` and `inverse_transform` — a fragile, undocumented positional coupling (see §3.1).

### 2.2 Data-Flow Contract

```
Raw CSV
  → DatasetProfiler.profile()          → DatasetProfile (typed, JSON-serializable)
  → Column drop (HIPAA / near-id / high-missing)
  → MissingnessHandler.fit() + transform()   → indicator cols + imputed values
  → Per-column scaler/encoder fit + transform → encoded parts
  → np.concatenate(parts, axis=1).astype(np.float32)   → (n_rows, n_features)
  → FileSchemaRegistry.save()          → profile.json + pipeline_state.joblib
```

The round-trip contract `inverse_transform(transform(x)) ≈ x` is the backbone of Phase 7/8. It is enforced by tests but **not** by runtime validation — a silent violation is possible (see §3.1, §3.2).

---

## 3. Logical & Correctness Debugging (Bugs Found)

### 3.1 [CRITICAL] Column-Cursor Desync in `inverse_transform`

**File:** `src/preprocessing/pipeline.py`  
**Lines:** 197–213 (fit fallback), 293–305 (inverse cursor)

**Problem:** In `fit_transform`, if a column's dtype is `UNKNOWN` and the scaler fallback raises an exception, the column is **skipped** — it contributes **0 columns** to the encoded array (lines 209–213). However, `inverse_transform`'s `else` branch (line 304–305) still advances `col_cursor += 1` for that column.

**Consequence:** Every column *after* the failed column is decoded from the wrong slice of the array — silently corrupting the entire reconstructed DataFrame. No error is raised; the corruption is silent.

**Trigger:** A column with `InferredDtype.UNKNOWN` whose scaler fails (e.g., all-null column that survived missingness handling, or a non-numeric column misclassified as continuous).

**Severity:** Critical — silent data corruption in the exact path Phase 7 depends on.

### 3.2 [HIGH] NaN Converted to Literal String `"nan"` for Categorical Columns

**File:** `src/preprocessing/pipeline.py`  
**Lines:** 189, 270

**Problem:** `working_df[col].astype(str)` is called before encoding. For a categorical column with missingness **below** `inject_indicator_above` (default 0.01 = 1%), the missingness handler does **not** impute it (it's not in `_indicator_columns`). The NaN then becomes the literal string `"nan"` via `astype(str)`.

**Consequence:** The round-trip contract is broken — `inverse_transform(transform(x))` returns the string `"nan"` instead of `NaN` for those positions. The test suite does not cover this case (all test categorical columns either have no NaN or are above the 1% threshold).

**Severity:** High — violates the documented round-trip contract; silently introduces a fake `"nan"` category into the vocabulary.

### 3.3 [HIGH] Dead Code / Misleading Logic in `OneHotEncoder.fit`

**File:** `src/preprocessing/encoders.py`  
**Lines:** 85–88

```python
if rare_cats or series.isna().any():
    # Always add __other__ if there were any rare cats
    pass  # __other__ added below if rare_cats exist
```

**Problem:** This is an empty `if` block with a `pass` and a comment. The intent appears to be: *add `__other__` when there are NaN values even if no rare categories exist*. But the actual `__other__` addition (line 89–90) only happens when `rare_cats` is non-empty.

**Consequence:** If a column has NaN values but **no** rare categories, `__other__` is not added to the vocabulary. At transform time, an unseen value (line 116–121) is then mapped to `__null__` instead of `__other__` — conflating "missing" with "unseen". This is a latent logic bug masked by the dead code.

**Severity:** High — the dead code reveals an unimplemented intended behavior; the actual behavior conflates unseen values with nulls.

### 3.4 [MEDIUM] Silent Mis-Decoding in `FrequencyEncoder.inverse_transform`

**File:** `src/preprocessing/encoders.py`  
**Lines:** 240–249

**Problem:** Out-of-range indices are silently clipped (`np.clip(idx, 0, len(vocab)-1)`), and float inputs are truncated with `astype(int)`. A generated sample from Phase 7's diffusion model that produces index `-3` or `999` (vocab size 10) silently decodes to the wrong category with **no warning**.

**Consequence:** Model outputs outside the valid embedding range produce plausible-looking but wrong categories — undetectable without explicit validation.

**Severity:** Medium — silent wrong output in the generative path.

### 3.5 [MEDIUM] `OneHotEncoder.inverse_transform` Argmax on Non-Normalized Rows

**File:** `src/preprocessing/encoders.py`  
**Lines:** 129–143

**Problem:** `np.argmax` on an all-zero row (or a row with ties) silently returns index 0 → `__null__` → NaN. A diffusion model output that is not a valid one-hot vector (e.g., `[0.2, 0.3, 0.1, 0.4]` sums to 1.0 but is not binary) is treated as if it were a valid one-hot.

**Consequence:** Soft, non-binary model outputs are silently coerced to the argmax category. This may be acceptable for decoding, but there is no validation that the input was a valid one-hot — a source of silent error.

**Severity:** Medium.

### 3.6 [MEDIUM] `PipelineConfig.random_seed` Is Never Consumed

**File:** `src/config/schema.py` (line 207–211); searched across all source files.

**Problem:** `random_seed` is defined in `PipelineConfig` and documented as "Global random seed for pipeline reproducibility," but **no source file reads it**. The profiler, encoders, scalers, and missingness handler never use it.

**Consequence:** The reproducibility promise is unfulfilled. Any stochastic step (e.g., future sampling) would not be seeded.

**Severity:** Medium — dead configuration; misleading documentation.

### 3.7 [LOW] Type Fidelity: Int Columns Round-Trip as Float

**File:** `src/preprocessing/scalers.py` (all scalers), `src/preprocessing/pipeline.py` (line 220)

**Problem:** `fit_transform` casts everything to `np.float32`. An integer column (e.g., `age` as int) round-trips as float. For large-magnitude values (e.g., lab values on the 1e5 scale), float32 precision (~7 significant digits) can breach the test tolerance `atol=1e-4` used in `test_pipeline_integration.py` (line 253).

**Consequence:** Round-trip tests may pass on small synthetic data but fail on real healthcare data with large-magnitude values.

**Severity:** Low — but a real risk for the Phase 8 evaluation on real data.

### 3.8 [LOW] `transform()` Silently Ignores Extra Columns

**File:** `src/preprocessing/pipeline.py`  
**Lines:** 242–275

**Problem:** `transform()` checks that all `_training_columns` are present (raises if missing) but does **not** reject extra columns. A DataFrame with a typo'd extra column is silently accepted.

**Consequence:** Schema drift at inference time is not detected.

**Severity:** Low.

---

## 4. Efficiency Analysis

### 4.1 [HIGH] O(n²) Profile Lookup in `fit_transform`

**File:** `src/preprocessing/pipeline.py`  
**Lines:** 164–167

```python
col_profile = self._profile.column_by_name(col) if col in [
    c.name for c in self._profile.columns
] else None
```

**Problem:** For each column, a fresh list comprehension builds a list of all column names, then `column_by_name` does a **linear scan** of `self._profile.columns`. This is O(n_cols²) — quadratic on wide datasets.

**Consequence:** A dataset with 500 columns performs 250,000 name comparisons. On healthcare claims data (often 200–1000+ columns), this is a measurable bottleneck.

**Fix direction (not applied):** Build a `dict[str, ColumnProfile]` once before the loop.

### 4.2 [HIGH] Python-Level Row Loops in Encoders

**File:** `src/preprocessing/encoders.py`  
**Lines:** 107–121 (OneHot), 223–236 (Frequency)

**Problem:** Both `transform` methods iterate row-by-row with `for i, val in enumerate(series)`, calling `pd.isna(val)` and dict lookups per element.

**Consequence:** On a 5M-row healthcare claims table, this is 5M Python-level iterations per categorical column — 10–50× slower than a vectorized `Series.map` + `fillna` approach. With 50 categorical columns, this dominates pipeline runtime.

### 4.3 [MEDIUM] Structural Missingness Test Is O(n_cols × n_rows) Per Column

**File:** `src/profiling/dataset_profiler.py`  
**Lines:** 297–333

**Problem:** For each column with missingness, `_classify_missingness` loops over **all other columns** and computes a correlation (point-biserial or Cramér's V) against each. This is O(n_cols² × n_rows) overall, with no early termination once the threshold is exceeded, and no caching of per-column missingness indicators.

**Consequence:** On a wide dataset with many missing columns, this is the single most expensive operation in profiling.

### 4.4 [MEDIUM] `fit_transform_from_file` Loads Entire CSV Into Memory

**File:** `src/preprocessing/pipeline.py`  
**Lines:** 313–320

**Problem:** `pd.read_csv(csv_path)` loads the whole file. No chunking, no dtype hints, no `nrows` sampling for profiling.

**Consequence:** On Kaggle's 20 GB working storage, a 5 GB CSV is fully materialized in RAM before profiling begins.

### 4.5 [LOW] Redundant Dtype Inference Work

**File:** `src/profiling/dataset_profiler.py`  
**Lines:** 184–192

**Problem:** For non-numeric columns, `_infer_dtype` calls `pd.to_numeric(non_null, errors="coerce")` to compute `numeric_fraction`, then re-coerces if it passes the threshold. The coercion is done twice.

**Consequence:** Minor redundant work; not a bottleneck but a code-smell.

---

## 5. Security & Vulnerability Analysis

### 5.1 [CRITICAL] Path Traversal / Arbitrary File Write via `dataset_name`

**File:** `src/registry/schema_registry.py`  
**Lines:** 88–89, 226–227, 195–196

**Problem:** `dataset_name` is used directly in file paths:
```python
dataset_dir = self._root / dataset_name
```
No sanitization, no allowlist, no path validation. A hostile `dataset_name` like `../../etc/cron.d/evil` or `..\\..\\Windows\\System32\\evil` escapes the registry root.

**Consequence:**
- **Arbitrary file write:** `save()` creates directories and writes files at attacker-controlled paths.
- **Arbitrary file read:** `load()` / `load_profile()` reads `profile.json` / `pipeline_state.joblib` from attacker-controlled paths.
- **Arbitrary file delete:** `delete()` calls `shutil.rmtree(dataset_dir)` on an attacker-controlled path.

**Attack surface:** Any code path that passes an untrusted `dataset_name` (e.g., from a user-supplied config, a filename, or an API parameter) is exploitable.

**Severity:** Critical — direct path traversal with write/read/delete primitives.

### 5.2 [HIGH] Untrusted `joblib.load()` — Pickle-Based RCE

**File:** `src/registry/schema_registry.py`  
**Lines:** 152, 44

**Problem:** `joblib.load()` deserializes arbitrary Python objects via pickle. If a registry directory is tampered with (or a malicious `pipeline_state.joblib` is placed at a path reachable via §5.1), loading it executes arbitrary code.

**Consequence:** Remote/local code execution on the machine running the pipeline.

**Severity:** High — especially combined with §5.1.

### 5.3 [MEDIUM] Loose Input Validation

**Files:** `src/preprocessing/pipeline.py`, `src/profiling/dataset_profiler.py`

**Problem:**
- No `isinstance(df, pd.DataFrame)` checks — a list or numpy array passed to `fit_transform` fails with an obscure error.
- No column-name validation at `transform()` time (extra columns silently ignored, §3.8).
- No dtype validation — a column that changes type between fit and transform (e.g., int → str) produces silent mis-encoding.

**Consequence:** Obscure failures and silent schema drift.

### 5.4 [MEDIUM] Reserved-Token Collision

**File:** `src/preprocessing/encoders.py`  
**Lines:** 41–42

**Problem:** `__null__` and `__other__` are reserved tokens. If real data contains a category literally named `__null__` or `__other__`, it collides with the reserved tokens, corrupting the round-trip.

**Consequence:** Silent data corruption for adversarial or unlucky category names.

### 5.5 [MEDIUM] Supply-Chain / Dependency Risks

**File:** `environment/requirements.txt`

**Problem:**
- Pinned versions have known CVEs (e.g., torch 2.3.1, numpy 1.26.4, scipy 1.13.1 — several have published advisories).
- `torch==2.3.1` on Windows installs CPU-only by default; on Kaggle it installs CUDA. The same `requirements.txt` produces different behavior on different platforms.
- `setup_check.py` hard-fails (exit 1) on CPU-only machines (`check_gpu` sets `passed=False`), making local development on CPU-only laptops impossible without editing the script.

**Consequence:** Inconsistent environments, known-vulnerable dependencies, and a hard blocker for CPU-only development.

### 5.6 [LOW] `latest.txt` Corruption → Uncaught ValueError

**File:** `src/registry/schema_registry.py`  
**Lines:** 233–238

**Problem:** If `latest.txt` is corrupted (e.g., contains `"abc"` instead of `"2"`), `int(latest_path.read_text(...))` raises an uncaught `ValueError` with no recovery path.

**Consequence:** The registry becomes unusable for that dataset with no self-healing.

---

## 6. Test & Documentation Drift

### 6.1 Tests Reach Into Private State

**Files:** `src/tests/test_encoders_inverse.py` (lines 252–255), `test_pipeline_integration.py` (lines 141–142, 206), `test_encoders_inverse.py` (line 388)

**Problem:** Tests access `encoder._vocab`, `pipeline._training_columns`, `handler._indicator_columns` — private attributes. Any refactor that renames these breaks tests even if behavior is unchanged.

**Consequence:** Refactor-fragile tests; false confidence.

### 6.2 Missing Test Coverage

No tests exist for:
- **Unseen categories** at transform time (mapped to `__null__` — §3.3)
- **Out-of-range indices** in `FrequencyEncoder.inverse_transform` (§3.4)
- **The ghost-column corruption path** in `inverse_transform` (§3.1)
- **NaN in categorical columns below the 1% threshold** (§3.2)
- **Extra columns at transform time** (§3.8)
- **Path traversal** in the registry (§5.1)
- **Corrupted `latest.txt`** (§5.6)
- **FrequencyEncoder end-to-end** — the integration test factory always returns `OneHotEncoder` (`test_pipeline_integration.py` line 41–43), so the high-cardinality path is never exercised through the full pipeline.

### 6.3 Documentation Drift

**File:** `docs/architecture.md`

- **Line 734:** References `docs/problems_and_resolutions.md` — **this file does not exist** in the repo.
- **Line 752:** Claims "112 tests" — the actual count is ~88 (32 profiler + 32 encoders + 12 registry + 12 pipeline). The claim is unverified and likely stale.
- **Line 599–606:** Test-count table (32/32/12/12) is approximate and not tied to actual test collection.

### 6.4 No Test Configuration

**Problem:** No `pytest.ini`, `pyproject.toml`, or `setup.cfg`. Tests only run from the repo root (imports use `src.` prefix). Running `pytest` from `src/` fails.

**Consequence:** Fragile test invocation; no CI configuration possible.

---

## 7. Severity-Ranked Bug Register

| ID | Severity | File | Line(s) | Description |
|----|----------|------|---------|-------------|
| B-01 | **Critical** | `pipeline.py` | 197–213, 293–305 | Column-cursor desync in `inverse_transform` — silent corruption of all columns after a failed encode |
| B-02 | **Critical** | `schema_registry.py` | 88–89, 226–227, 195–196 | Path traversal / arbitrary file write-read-delete via unsanitized `dataset_name` |
| B-03 | **High** | `pipeline.py` | 189, 270 | NaN → literal `"nan"` string for low-missingness categorical columns; breaks round-trip |
| B-04 | **High** | `schema_registry.py` | 152, 44 | Untrusted `joblib.load()` — pickle-based RCE |
| B-05 | **High** | `encoders.py` | 85–88 | Dead code reveals unimplemented `__other__`-for-NaN logic; unseen values conflated with nulls |
| B-06 | **High** | `pipeline.py` | 164–167 | O(n²) profile lookup — quadratic on wide datasets |
| B-07 | **High** | `encoders.py` | 107–121, 223–236 | Python-level row loops — 10–50× slower than vectorized on large data |
| B-08 | **Medium** | `encoders.py` | 240–249 | Silent index clipping/truncation in `FrequencyEncoder.inverse_transform` |
| B-09 | **Medium** | `encoders.py` | 129–143 | Argmax on non-normalized rows silently coerces to NaN |
| B-10 | **Medium** | `schema.py` | 207–211 | `random_seed` never consumed — reproducibility promise unfulfilled |
| B-11 | **Medium** | `dataset_profiler.py` | 297–333 | O(n_cols² × n_rows) structural missingness test, no early termination |
| B-12 | **Medium** | `pipeline.py` | 313–320 | Whole-CSV load into memory; no chunking |
| B-13 | **Medium** | `pipeline.py` | 242–275 | Extra columns silently ignored at transform time |
| B-14 | **Medium** | `encoders.py` | 41–42 | Reserved-token collision (`__null__`/`__other__`) with real data |
| B-15 | **Medium** | `requirements.txt` | all | Known-CVE pinned versions; platform-divergent torch installs; CPU-only hard-fail |
| B-16 | **Low** | `scalers.py`, `pipeline.py` | 220 | Int→float type drift; float32 precision breach on large-magnitude values |
| B-17 | **Low** | `schema_registry.py` | 233–238 | Corrupted `latest.txt` → uncaught ValueError, no recovery |
| B-18 | **Low** | `dataset_profiler.py` | 184–192 | Redundant double dtype coercion |

---

## 8. Complexity & Maintainability Assessment

### 8.1 Cyclomatic Complexity Hotspots

| File | Complexity Driver | Assessment |
|------|-------------------|------------|
| `pipeline.py` | `fit_transform` (multi-branch dtype dispatch + fallback) | Moderate; the fallback branch is the source of B-01 |
| `dataset_profiler.py` | `_classify_missingness` (nested loops + try/except per pair) | High; hard to reason about, no early exit |
| `encoders.py` | `transform` (per-row branching) | Moderate; would drop significantly with vectorization |
| `schema_registry.py` | `save`/`delete` (versioning + atomic-write + error handling) | Moderate; well-structured but path-unsafe (B-02) |

### 8.2 Maintainability Strengths

- **Excellent docstrings** — every module, class, and method documents its contract and Phase 4/5 dependency.
- **Consistent naming** — `_assert_fitted`, `_NOT_FITTED` sentinel, `_INDICATOR_SUFFIX` are uniform across modules.
- **Frozen Pydantic models** — config is self-documenting and immutable.
- **Test-first round-trip contract** — the single most important invariant is enforced.

### 8.3 Maintainability Weaknesses

- **Private-state coupling in tests** (§6.1) — refactor-fragile.
- **No type-checking config** — no `mypy`/`pyright` config; `# type: ignore` comments are scattered.
- **No linting config** — no `ruff`/`flake8`/`black` config; style is consistent but unenforced.
- **No packaging** — not installable; import paths are fragile.

---

## 9. Prioritized Recommendations

### P0 — Fix Immediately (Correctness / Security)

1. **B-01:** In `inverse_transform`, track the actual number of columns each column contributed during `fit_transform` (store per-column output width in a dict), and use that to advance the cursor — not a hardcoded `+1`.
2. **B-02:** Sanitize `dataset_name` in `FileSchemaRegistry` (reject `/`, `\`, `..`, empty, and reserved names; or use a hash/allowlist mapping).
3. **B-03:** Handle NaN in categorical columns before `astype(str)` — either impute via the missingness handler (lower the `inject_indicator_above` threshold or add a categorical-specific path) or map NaN to `__null__` explicitly before string conversion.

### P1 — Fix Soon (Correctness / Efficiency)

4. **B-05:** Implement the intended `__other__`-for-NaN logic in `OneHotEncoder.fit`, or remove the dead code and document the actual behavior.
5. **B-06:** Build a `dict[str, ColumnProfile]` once before the per-column loop in `fit_transform`.
6. **B-07:** Vectorize encoder `transform` using `Series.map` + `fillna` + numpy indexing.
7. **B-04:** Add integrity verification (e.g., a checksum file written atomically alongside `pipeline_state.joblib`, verified on load) or restrict `joblib.load` to trusted paths.

### P2 — Fix When Convenient (Robustness / Hygiene)

8. **B-08/B-09:** Add explicit validation in `inverse_transform` — raise or warn on out-of-range indices / non-normalized rows.
9. **B-10:** Either consume `random_seed` (seed numpy/pandas at pipeline start) or remove it from config.
10. **B-11:** Add early termination to `_classify_missingness` once the threshold is exceeded; cache missingness indicators.
11. **B-12:** Add chunked CSV reading or a `nrows` sampling parameter to `fit_transform_from_file`.
12. **B-13:** Reject extra columns at `transform()` time.
13. **B-14:** Escape or reject reserved tokens in real data.
14. **B-15:** Update pinned versions to patched releases; document platform-specific installs; make `check_gpu` a warning, not a hard-fail.
15. **B-16:** Use float64 internally where precision matters, or document the float32 contract.
16. **B-17:** Add recovery for corrupted `latest.txt` (fall back to scanning version dirs).
17. **B-18:** Remove the redundant double coercion in `_infer_dtype`.

### P3 — Test & Docs

18. Add tests for: unseen categories, out-of-range indices, ghost-column corruption, NaN-in-categorical round-trip, extra-column transform, path traversal, corrupted `latest.txt`, and a FrequencyEncoder end-to-end path.
19. Replace private-state assertions with public-behavior assertions.
20. Create `docs/problems_and_resolutions.md` (referenced but missing) or remove the reference.
21. Verify and correct the test-count claim in `docs/architecture.md`.
22. Add `pytest.ini` / `pyproject.toml` with test config and CI-ready invocation.

---

## 10. Appendix: Files Reviewed

| File | Lines | Role |
|------|-------|------|
| `src/config/schema.py` | 216 | Frozen Pydantic config models |
| `src/preprocessing/base.py` | 193 | ABCs: AbstractScaler, AbstractEncoder, AbstractMissingnessHandler |
| `src/preprocessing/encoders.py` | 269 | OneHotEncoder, FrequencyEncoder |
| `src/preprocessing/missingness.py` | 202 | MissingnessHandler |
| `src/preprocessing/pipeline.py` | 333 | PreprocessingPipeline orchestrator |
| `src/preprocessing/scalers.py` | 202 | StandardScaler, MinMaxScaler, RobustScaler |
| `src/profiling/base.py` | 174 | AbstractProfiler, DatasetProfile, ColumnProfile, enums |
| `src/profiling/dataset_profiler.py` | 479 | DatasetProfiler, HIPAA matching, dtype inference, missingness |
| `src/registry/base.py` | 161 | AbstractSchemaRegistry, RegistryEntry |
| `src/registry/schema_registry.py` | 258 | FileSchemaRegistry |
| `src/tests/conftest.py` | 127 | Shared fixtures |
| `src/tests/test_encoders_inverse.py` | 392 | Round-trip tests |
| `src/tests/test_pipeline_integration.py` | 255 | End-to-end pipeline tests |
| `src/tests/test_profiler.py` | 268 | Profiler + HIPAA tests |
| `src/tests/test_registry_roundtrip.py` | 271 | Registry save/load tests |
| `environment/requirements.txt` | 49 | Pinned dependencies |
| `environment/setup_check.py` | 296 | Environment verification script |
| `docs/architecture.md` | 758 | Architectural reference |

---

*End of analysis. No source code was modified during this review.*