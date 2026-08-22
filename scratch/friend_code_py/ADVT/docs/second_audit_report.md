# Second Audit Report — Vulnerability, Logic & Architecture Re-Analysis
## Privacy-Preserving Synthetic Healthcare Data Generation Framework (Phases 1–3)

---

> **Document Type:** Read-Only Audit Report  
> **Scope:** Verification of previously reported fixes + new findings  
> **Constraint:** No source code was modified. This is an analysis-only deliverable.  
> **Date:** 2026-08-18

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Fix Verification Matrix](#2-fix-verification-matrix)
3. [Critical Regression — B-03 vs B-14 Conflict](#3-critical-regression--b-03-vs-b-14-conflict)
4. [High-Severity Design Reversal — Integer Near-Identifier](#4-high-severity-design-reversal--integer-near-identifier)
5. [Additional New & Remaining Findings](#5-additional-new--remaining-findings)
6. [Documentation Drift](#6-documentation-drift)
7. [Architectural Bottlenecks & Limitations](#7-architectural-bottlenecks--limitations)
8. [Initial Report Accuracy Corrections](#8-initial-report-accuracy-corrections)
9. [Severity-Ranked Issue Register](#9-severity-ranked-issue-register)
10. [Prioritized Action Items](#10-prioritized-action-items)

---

## 1. Executive Summary

This audit re-examines the codebase after the previously reported vulnerabilities were addressed. **12 of 18** original issues are confirmed fixed in code. However, the audit surfaced **1 critical regression** and **1 high-severity design reversal** introduced by the fixes themselves:

- **Critical:** The B-03 fix (`fillna("__null__")` in `pipeline.py`) directly triggers the new B-14 `ValueError` in `encoders.py`, causing the pipeline to **crash** on any categorical column with sparse (<1%) missingness — the exact case B-03 was meant to fix.
- **High:** The N-01 fix now flags **every integer column** with >95% uniqueness as a near-identifier and silently drops it — re-introducing the P-05 data-loss bug that the architecture documentation explicitly says was resolved.

Additionally, 5 original issues remain unfixed (B-09, B-12, B-13, B-15, B-16), 1 was partially fixed (B-07 — only `FrequencyEncoder` was vectorized despite the report claiming both), and 1 was retracted as a false positive (B-18).

---

## 2. Fix Verification Matrix

### 2.1 ✅ Confirmed Fixed (Verified in Code)

| ID | Fix | Verified In |
|----|-----|-------------|
| **B-01** | `valid_training_columns` strips failed columns from the training manifest → cursor desync eliminated | `pipeline.py:230` |
| **B-02** | `_validate_dataset_name()` rejects `/`, `\`, `..` in all save/load/delete paths | `schema_registry.py:59–61` |
| **B-04** | SHA-256 checksum written + verified on load | `schema_registry.py:134–136, 164–171` |
| **B-05** | Dead `if...pass` block removed | `encoders.py:85–91` |
| **B-06** | O(1) `profile_dict` lookup replaces O(n²) list scan | `pipeline.py:173` |
| **B-08** | Out-of-range index warning before clipping | `encoders.py:237–239` |
| **B-10** | `np.random.seed()` called in `fit_transform` | `pipeline.py:128–130` |
| **B-11** | Early `break` once structural-missingness threshold exceeded | `dataset_profiler.py:338–339` |
| **B-14** | Reserved-token collision raises `ValueError` | `encoders.py:73–74, 194–195` |
| **B-17** | Corrupted `latest.txt` → descriptive `RuntimeError` | `schema_registry.py:259–262` |
| **N-03** | `low_memory=False` on CSV read | `pipeline.py:337` |
| **N-04** | Constant columns (`nunique <= 1`) dropped | `pipeline.py:141–143` |
| **N-01a** | HIPAA regex extended (`encounter`, `nbr`) | `dataset_profiler.py:107` |

### 2.2 ⚠️ Partially Fixed

| ID | Status | Evidence |
|----|--------|----------|
| **B-03** | **BROKEN — see §3** | `pipeline.py:200` vs `encoders.py:73–74` |
| **B-07** | Only `FrequencyEncoder.transform` vectorized; **`OneHotEncoder.transform` still has the Python row loop** | `encoders.py:107–121` — the verification report (line 11) claims "both encoders were rewritten," but the actual code contradicts this |

### 2.3 ❌ Not Fixed (No Changes Found)

| ID | Issue | Still Present At |
|----|-------|------------------|
| **B-09** | Argmax on non-normalized/NaN rows silently coerces to NaN | `encoders.py:138` |
| **B-12** | Whole-CSV load, no chunking / `nrows` sampling | `pipeline.py:337` |
| **B-13** | Extra columns silently ignored in `transform()` | `pipeline.py:259–293` |
| **B-15** | Pinned CVEs / platform-divergent torch / CPU-only hard-fail in `setup_check.py` | `requirements.txt`, `setup_check.py:114–115` |
| **B-16** | float32 precision + int→float type drift | `pipeline.py:237` |

---

## 3. Critical Regression — B-03 vs B-14 Conflict

### 3.1 The Failure Chain

The B-03 fix and the B-14 fix are **mutually incompatible**:

1. A categorical column has NaN with missing rate **below** `inject_indicator_above` (default 1%) → `MissingnessHandler` does **not** impute it (it's not in `_indicator_columns`).
2. `pipeline.py:200`: `working_df[col].fillna("__null__").astype(str)` replaces those NaNs with the literal string `"__null__"`.
3. `encoders.py:73`: `if _NULL_TOKEN in counts.index: raise ValueError(...)` — the B-14 collision guard now sees `"__null__"` in the data and **raises**.

**Result:** The pipeline **hard-crashes** on any dataset containing a categorical column with sparse missingness (<1%). This is precisely the scenario B-03 was designed to fix.

### 3.2 Why the Test Suite Misses It

- `test_null_values_round_trip` calls `encoder.fit()` **directly** with NaN intact — the `fillna("__null__")` pre-processing in the pipeline is never exercised.
- `tiny_df`'s `bmi` column has 10% missingness (above threshold → imputed by the handler, so no NaN reaches the encoder).
- **No test exists** with a categorical column containing <1% missingness through the full pipeline path.

### 3.3 Fix Direction (Not Applied — Analysis Only)

- Let NaN pass through to the encoder unchanged (its `transform` already maps `pd.isna(val)` → index 0), **or**
- Make the encoder's `fit` treat `"__null__"` as the null token rather than rejecting it, **or**
- Pre-map NaN in the pipeline to the encoder's own null path without injecting the literal reserved string.

---

## 4. High-Severity Design Reversal — Integer Near-Identifier

### 4.1 The Change

**File:** `dataset_profiler.py:176–180`

```python
if uniqueness_ratio > config.cardinality.near_identifier_ratio:
    if not is_numeric:
        return InferredDtype.NEAR_IDENTIFIER
    elif pd.api.types.is_integer_dtype(non_null):
        return InferredDtype.NEAR_IDENTIFIER   # ← NEW (N-01)
```

### 4.2 The Problem

The original codebase — and `docs/architecture.md` §5.2, plus the P-05 resolution in `docs/problems_and_resolutions.md` — explicitly documented:

> "The NEAR_IDENTIFIER check applies only to **non-numeric** columns. Numeric ID columns are caught by the HIPAA name-matching check instead."

The N-01 fix **reversed** this decision. Now **every integer column** with uniqueness ratio > 0.95 is flagged as a near-identifier and **silently dropped** from training.

### 4.3 Collateral Damage on Legitimate Features

On small-N datasets (n < 500 — the exact cohort this framework claims to support), integer features routinely exceed 0.95 uniqueness:

- `visit_count`, `procedure_count`, `billing_units`
- `num_lab_procedures`, `num_medications`, `num_diagnoses`
- `day_of_week`-as-int, `month`-as-int, `year`-as-int

All of these are **legitimate clinical features** that would now be silently dropped.

### 4.4 The Fix Is Redundant

The columns N-01 was targeting (`encounter_id`, `patient_nbr`) **already match the expanded HIPAA regex** at `dataset_profiler.py:107`:

```
encounter_?(?:id)?|patient_?nbr|nbr
```

The integer uniqueness check is therefore **redundant** for those columns and **dangerous** for everything else.

### 4.5 Stale Test Comment

`test_profiler.py:229` still contains the comment:

> "Since our near-identifier check now skips numeric, row_index won't be flagged automatically."

This comment **contradicts the current code**, which now flags integer columns.

---

## 5. Additional New & Remaining Findings

### 5.1 Sparse-NaN Categorical Round-Trip Still Inconsistent (B-03 Incomplete)

Even if the §3 crash is resolved, the pipeline path (`fillna("__null__")` at `pipeline.py:200, 287`) and the encoder's direct path (NaN → index 0) now handle NaN **inconsistently** — two different NaN conventions depending on entry point. The round-trip contract is not uniform.

### 5.2 Checksum Is Integrity-Only, Not Authenticity

`schema_registry.py:166` — if the `.sha256` file is **absent** (e.g., pre-fix registry versions), verification is **silently skipped**. Deleting the `.sha256` file bypasses the check entirely. This protects against accidental corruption but **not** tampering. Acceptable for the Kaggle use case, but the limitation should be documented.

### 5.3 `latest.txt` Write Is Not Atomic

`schema_registry.py:147` — `latest_path.write_text()` has no temp+rename pattern. A session kill between `shutil.move` (new version dir) and the `latest.txt` write orphans the new version: it exists on disk but `load()` with no version returns the **old** version. This violates the atomic-write invariant for exactly the Kaggle session-kill scenario the architecture doc claims to protect against.

### 5.4 `np.random.seed()` Is a Partial B-10 Fix

`pipeline.py:129` seeds **only NumPy global state** — not `random`, not pandas sampling. It is also **global state**, which is not thread-safe under `pytest-xdist` parallel execution. Acceptable for single-process training; should be documented as such.

### 5.5 OneHotDecoder Accepts Any Row (B-09 Unfixed)

Still no validation that `inverse_transform` input is a valid one-hot vector. An all-NaN row → `argmax` → 0 → NaN silently. A softmax-soft row → argmax silently. Model-generated garbage decodes to plausible categories with no warning.

### 5.6 `inverse_transform` Has No Width Check

No validation that `arr.shape[1]` equals the expected total encoded dimension count. Wrong-width arrays silently decode garbage with no error.

---

## 6. Documentation Drift

| Doc | Claim | Reality |
|-----|-------|---------|
| `verification_fixes_report.md:11` | "Rewrote `transform` in **both** OneHotEncoder and FrequencyEncoder" | Only FrequencyEncoder was vectorized; OneHotEncoder still has the row loop |
| `verification_fixes_report.md:7` | B-03 "resolved" | **Crashes** the pipeline on sparse-NaN categoricals (conflicts with B-14) |
| `test_profiler.py:229` comment | "near-identifier check now skips numeric" | Code now flags **integer** columns (contradiction) |
| `problems_and_resolutions.md:280` | "112 passed" | Consistent with the suite, but the suite does **not** cover the B-03 crash path or the integer near-identifier drop |
| `workflow_and_examples.md:371` | Imputed stage has "No nulls" | True only for columns above 1% missingness; sparse-NaN categoricals still carry NaN |

---

## 7. Architectural Bottlenecks & Limitations

1. **OneHotEncoder row loop** (`encoders.py:107–121`) — still O(n) Python-level iteration; 10–50× slower than vectorized on >100K rows. The most impactful remaining performance fix.
2. **Structural missingness still O(n_cols² × n_rows)** — the early `break` helps, but there is no caching of per-column missingness indicators; wide datasets still pay heavily.
3. **Whole-file CSV load** — no chunking or `nrows` sampling for profiling; multi-GB CSVs fully materialize in RAM.
4. **No schema/size validation at `inverse_transform`** — wrong-width arrays silently corrupt.
5. **Empty `__init__.py` + no packaging** — package is not installable; tests only run from repo root.
6. **No CI / linting / type-checking config** — `pytest.ini`, `pyproject.toml`, `ruff`/`mypy` configs still absent.
7. **Tests reach into private state** (`encoder._vocab`, `pipeline._training_columns`, `handler._indicator_columns`) — unchanged; refactor-fragile.
8. **No regression tests for any of the new fixes** — B-02 path-traversal, B-04 checksum mismatch, B-14 collision, N-04 constant-drop, and the new integer near-identifier behavior all lack test coverage.

---

## 8. Initial Report Accuracy Corrections

- **B-18** (double dtype coercion) — **retracted**. On careful re-read, the coerced result is correctly reused in `_infer_dtype`; the original report was wrong on this point.
- **B-05** — the dead code was removed, but the underlying semantic intent (add `__other__` for NaN-only columns) remains unfulfilled; the behavior gap now manifests through the crash-prone `fillna` path instead.

---

## 9. Severity-Ranked Issue Register

| ID | Severity | File | Line(s) | Description |
|----|----------|------|---------|-------------|
| **R-01** | **Critical** | `pipeline.py` + `encoders.py` | 200, 73–74 | B-03 `fillna("__null__")` triggers B-14 `ValueError` → pipeline crashes on sparse-NaN categoricals |
| **R-02** | **High** | `dataset_profiler.py` | 176–180 | Integer near-identifier reversal silently drops legitimate integer features (P-05 regression) |
| **R-03** | **Medium** | `encoders.py` | 107–121 | OneHotEncoder row loop not vectorized despite report claim |
| **R-04** | **Medium** | `schema_registry.py` | 166 | Checksum silently skipped if `.sha256` file absent (integrity-only, not authenticity) |
| **R-05** | **Medium** | `schema_registry.py` | 147 | `latest.txt` write not atomic → session-kill orphans new version |
| **R-06** | **Medium** | `pipeline.py` | 129 | `np.random.seed()` seeds only NumPy global state; not thread-safe |
| **R-07** | **Medium** | `encoders.py` | 138 | B-09 unfixed — argmax on non-normalized rows silently coerces |
| **R-08** | **Medium** | `pipeline.py` | 295–329 | No width validation in `inverse_transform` |
| **R-09** | **Low** | `pipeline.py` | 337 | B-12 unfixed — whole-CSV load |
| **R-10** | **Low** | `pipeline.py` | 259–293 | B-13 unfixed — extra columns silently ignored |
| **R-11** | **Low** | `requirements.txt`, `setup_check.py` | all | B-15 unfixed — CVEs, platform-divergent torch, CPU-only hard-fail |
| **R-12** | **Low** | `pipeline.py` | 237 | B-16 unfixed — float32 precision / int→float drift |

---

## 10. Prioritized Action Items

### P0 — Fix Immediately (Correctness)

1. **R-01:** Resolve the B-03/B-14 conflict — the pipeline crashes on sparse-NaN categoricals.
2. **R-02:** Remove the integer-column near-identifier branch — it re-introduces the P-05 data-loss bug.

### P1 — Fix Soon (Correctness / Performance)

3. **R-03:** Vectorize `OneHotEncoder.transform` to match the documented claim.
4. Add regression tests for: R-01 crash path, R-02 integer drop, B-02 path-traversal, B-04 checksum mismatch, B-14 collision, N-04 constant-drop.
5. **R-08:** Add width validation to `inverse_transform`.
6. **R-05:** Make `latest.txt` write atomic (temp + rename).

### P2 — Hygiene

7. **R-04:** Document the integrity-only limitation of the checksum (or require the `.sha256` file).
8. **R-06:** Document the global-seed limitation (or seed `random`/pandas too).
9. Update `verification_fixes_report.md` and stale test comments to match actual code.
10. Add `pytest.ini` + `pyproject.toml` + linting/type-checking config.

---

*End of second audit. No source code was modified during this review.*