# Fifth Audit Report — Pipeline, Algorithms, Kaggle Push, Privacy & Adversarial Attack

Date: 2026-08-22
Scope: full `src/` pipeline + algorithms, `kaggle_runner/`, `scripts/reproduce_end_to_end.py`, generated data in `scratch/sweep_results/`.

---

## 1. CRITICAL: Generated data is broken (empirically confirmed)

**Evidence (this audit, run live):**

- `sweep_report.json`: `loss = NaN` for **all three** epsilon runs (0.1, 1.0, 10.0).
- MD5 of `synthetic_eps_0.1.csv`, `synthetic_eps_1.0.csv`, `synthetic_eps_10.0.csv` are **identical** (`0c9ab1bce9924e283c971055cc4ba071`). The "epsilon sweep" produced three copies of the same file.
- Every row of the synthetic CSVs is a duplicate (`unique-row fraction = 0.0000`) and **every value is NaN** (`df.nunique() == 0` for all 44 columns).

**Root-cause chain:**

1. **C-1 — Training diverges to NaN, nothing stops it.** `DPTrainer.train_epoch` has no NaN guard, no gradient-norm monitoring, no early stop. Once the loss goes NaN, training "completes" and a NaN-weight model is checkpointed.
2. **C-2 — Noise scale is unbounded and signal-free.** `base_sigma = 15 / target_eps` gives σ = 150 for ε = 0.1. Added noise std per step = `C·σ/batch = 1.0·150/256 ≈ 0.6` per weight, while the clipped signal has norm ≤ 1 across the *entire* network. SNR ≈ 10⁻⁴ → the model random-walks on pure noise; weights grow without bound → float32 overflow → NaN. Even ε = 10 (σ = 1.5) is noise-dominated at this clip/noise ratio.
3. **C-3 — Sampler propagates NaN.** `generate_samples` runs 1000 reverse steps on a NaN-weight denoiser → all-NaN tensor → `inverse_transform` maps NaN one-hots (argmax → index 0 → `__null__`) and NaN scaler outputs → all-NaN DataFrame. No NaN validation exists anywhere between training and CSV export.
4. **C-4 — No output validation gate.** The pipeline happily writes an all-NaN, 44-column, 101,766-row CSV and reports success. A `assert not synth.isna().all().all()` / uniqueness check before export is mandatory.

## 2. CRITICAL: Privacy accounting is not trustworthy

- **P-1 — ε-targeting is broken.** Reported actual ε: target 0.1 → 0.1029, target 1.0 → **0.1032**, target 10.0 → 0.3258. Targets 0.1 and 1.0 (10× different σ) yielding nearly identical ε is impossible under correct RDP composition with the same step count — the accounting and/or the σ actually used in clipping vs. accounting are inconsistent. Additionally there is **no closed-loop stop at the target ε**: training runs a fixed epoch count and merely *reports* ε afterwards. For ε = 10 the run spent 0.33 (30× under budget → needlessly noisy model); nothing prevents the opposite (overspend) either.
- **P-2 — Per-tier double counting.** `clip_and_noise_tier` calls `accountant.record_step` once **per tier per batch**. With T tiers, one batch is logged T times. Currently conservative (over-counts), but if tiers ever get separate budgets this silently mis-composes. Record once per batch, or compose per-tier accountants.
- **P-3 — Tiers are dead code in the end-to-end run.** `reproduce_end_to_end.py` computes `tiers = assigner.assign_tiers(...)` and then never uses them: `tier_params = {"global": ...}`, `tier_clip_norms = {"global": 1.0}`. The entire per-tier privacy-head mechanism (a core project claim) is untested in the real pipeline.
- **P-4 — Adaptive σ accounting mismatch.** `AdaptiveNoiseSchedule` varies σ per timestep (0.5σ–1.5σ), and the *batch-mean* timestep is used. Per-sample timesteps differ; the privacy guarantee of per-example heterogeneous noise multipliers is not what RDP composition of the mean σ models. Use the **minimum** σ in the batch for accounting (conservative) or per-sample accounting.

## 3. HIGH: Evaluation metrics bugs

- **E-1 — `PrivacyEvaluator._factorize_data` is category-inconsistent.** `pd.factorize` is called **independently** on `df_train`, `df_holdout`, `df_synth`. The same category (e.g. `"Caucasian"`) gets different integer codes in each frame, so Euclidean distances — and therefore the MIA risk score — are meaningless. Factorize against a **shared** vocabulary (e.g. fit on train, `map` into the others).
- **E-2 — MIA denominator.** `mia_risk_score = dist_diff / mean_dist_holdout` is not a calibrated attack-success metric; report AUC/accuracy of a threshold attack instead.
- **E-3 — `UtilityEvaluator.evaluate_bivariate_correlation_rmse` returns 0.0** when < 2 common numeric columns — indistinguishable from a *perfect* score. Return `NaN`/flag instead.

## 4. MEDIUM: Pipeline / algorithm bugs

- **M-1 — `PreprocessingPipeline.transform` does not reproduce fit-time column set.** During `fit_transform`, the missingness handler's injected `__missing_flag` columns are encoded as features (they appear in `initial_columns`). At `transform` time this happens to work, but any column dropped between fit and transform paths (e.g. constant-column detection runs only at fit) can silently change the encoded width. Width is validated only in `inverse_transform`, not `transform`.
- **M-2 — `MissingnessHandler.fit` mode imputation casts to `str`.** `impute_val = str(col_series.mode().iloc[0])` for non-numeric dtype — a numeric column stored as `object` gets a string imputant, corrupting downstream `pd.to_numeric` (coerced to NaN → re-introduces the NaN the handler was supposed to remove).
- **M-3 — `HeuristicRiskTierAssigner` correlation guard is O(n²) with a subtle bug:** it promotes pairs iteratively in-place while iterating, so the result depends on column order (not a fixed point). Compute the tighter tier from the *initial* assignment, then apply.
- **M-4 — `check_hipaa_identifier` regex gaps.** `("Dates")` pattern matches token `day`/`month` but not `year_`-prefixed combos; `("Account numbers")` misses bare `account`; geographic pattern `(?:_|$|code)` makes the trailing boundary optional in a way that also matches `zipcodex`-style tokens incorrectly (`zip_code` works, but `zipcode` does **not** match because there is no `_` between `zip` and `code`... actually `zip` followed by `code` fails the `(?:_|$|code)` only when glued as `zipcode` — `padded = "_zipcode_"`, pattern `(?:^|_)(?:zip...)(?:_|$|code)` — `zip` then needs `_`, `$` or `code`; next chars are `code` → matches. OK.) Real gap: **`age` is not flagged** (fine, it's a Safe-Harbor-permitted element), but `admit_date`-style camelCase names (`AdmitDate`) fail because matching is on the lowercased padded string with `_` boundaries — camelCase names without underscores (`dateofbirth`) do not match `date(?:_of)?_birth`. Normalize by also inserting `_` at camelCase boundaries.
- **M-5 — `OneHotEncoder.transform` mutates a copy via chained assignment** (`series_str[series_str.isna()] = ...` on a copy is fine, but on a slice view of user data under pandas ≥ 3 copy-on-write this pattern warns). Use `.where()` / `.fillna` instead.
- **M-6 — `FrequencyEncoder.inverse_transform` clips out-of-range indices silently** after only a warning — for generated data this masks model failure (see §1). Should raise or route to `__other__`.
- **M-7 — `ComputeBudgetGuard` counts wall-clock, not GPU time.** The "M-1 FIX" comment claims GPU-active tracking; `_gpu_active_seconds` is initialized and never updated. Budget is still pure wall-clock.
- **M-8 — `DPTrainer` calls `x_t.requires_grad_()`** unnecessarily (inputs don't need grad for grad-sample computation) — wasted memory on a 101k×~120 float tensor per batch.
- **M-9 — `sampler._generate_single_batch` ignores `alphas` clamp interplay:** correct DDPM, but at t where `1 - ᾱ_t` underflows the clamp to 1e-8 the update amplifies pred_noise by ~316×; with a healthy model this is benign, with a degraded model it accelerates divergence. Consider the standard `sqrt(1-ᾱ_{t-1})·β_t` posterior variance form.
- **M-10 — `LinearNoiseSchedule.to(device)` mutates shared state**; the same schedule instance moved after being captured elsewhere (e.g. inside `forward_diffuse` closures) silently changes device under callers.

## 5. Kaggle push problem (`kaggle_runner/`)

- **K-1 — `kernel-metadata.json` uses strings for booleans:** `"is_private": "false"`, `"enable_gpu": "true"`. The Kaggle CLI expects JSON booleans. Depending on CLI version this fails validation or silently misconfigures the kernel (public GPU kernel!). Change to `false` / `true` (unquoted).
- **K-2 — `code_file` path:** metadata says `run_pipeline.py`, but the file lives in `kaggle_runner/`. `kaggle kernels push` resolves `code_file` relative to the **current working directory** — you must `cd kaggle_runner` before pushing, or set `"code_file": "kaggle_runner/run_pipeline.py"` and push from repo root. This is the most common cause of the "push problem" (kernel pushes the wrong/old file or errors with file-not-found).
- **K-3 — Kernel id ownership:** `"id": "kingtime248/dp-sgd-synthetic-pipeline"` — push fails with 403 if the authenticated Kaggle username is not `kingtime248`.
- **K-4 — `run_pipeline.py` force-reinstalls torch 2.3.1 cu118 *after* installing `requirements.txt`.** If requirements pins `opacus`/`numpy` versions built against a different torch, the force-reinstall breaks them (opacus must match torch version). Install torch **first**, then requirements, or pin compatible versions.
- **K-5 — No `kaggle.json` / API-token handling or push script in the repo**; the push step is entirely manual and undocumented — add `kaggle kernels push -p kaggle_runner` to a script and validate metadata with `kaggle kernels list -m` first.
- **K-6 — `wget` on the UCI URL:** UCI occasionally rate-limits/redirects; `wget -q` fails silently into an invalid zip (caught, but the run dies). Add `--tries=3` and a retry.

## 6. Generated-data privacy audit & adversarial attack results

Attacks run (script: `scratch/attack_synthetic.py`) against all three generated CSVs vs. the real UCI diabetes data:

| Attack | Result |
|---|---|
| Exact QI re-identification (race+gender+age, +admission_type, +time_in_hospital, gender+age+num_medications) | **0 / 101,766 matches** in all files — but only because all values are NaN (see §1). Not a privacy success. |
| Nearest-neighbour D-MIA (train vs. holdout distance to synth) | train 809.58 vs holdout 807.57, risk −0.002 — **identical across all three ε**, again because the files are byte-identical NaN blobs. |
| Row uniqueness | 0.0000 — zero unique rows. |

**Verdict:** the current artifacts leak nothing *because they contain no data*. Privacy cannot be claimed from these runs. However, the audit found real latent leak channels:

- **L-1 — Registry side-channel.** `pipeline_state.joblib` stores fitted encoders (full category vocabularies incl. rare categories), imputation medians/modes, and the `DatasetProfile` (per-column means, stds, min/max, medians, skewness, top-20 category counts). Anyone with registry access can reconstruct substantial distributional information about real patients — and rare-category names can be outright identifying. The registry is written next to outputs (`outputs/sweep_results/registry_eps_*`) and would be included in Kaggle output archives.
- **L-2 — Checkpoints.** `model_eps_*.pt` are DP-trained (good) but the DP guarantee is void while §2's accounting bugs stand; publishing checkpoints + registry together lets an attacker fine-tune/attack with more signal than ε accounting assumes.
- **L-3 — `patient_nbr` / `encounter_id` handling.** These are correctly dropped as near-identifiers/HIPAA in the pipeline, but the drop relies on the profiler's uniqueness ratio; a dataset where IDs are reused across rows would slip through. Add an explicit deny-list.
- **L-4 — `Privacy_Preserving...pptx` / docs in the repo** contain internal detail; ensure no real patient excerpts are embedded in report DOCX/PPTX assets before public release (spot-check recommended).

## 7. Recommended fix order

1. Add NaN guards + grad-norm logging in `DPTrainer`; clamp/anneal σ; validate model weights before sampling (C-1/C-2).
2. Add a post-generation validation gate (no all-NaN columns, minimum uniqueness, schema match) before CSV export (C-3/C-4).
3. Fix ε accounting: single record per batch, closed-loop early stop at target ε, conservative σ for accounting (P-1/P-2/P-4); wire tiers end-to-end or remove the claim (P-3).
4. Fix `_factorize_data` shared-vocabulary bug (E-1).
5. Fix `kernel-metadata.json` booleans + push from `kaggle_runner/` (K-1/K-2).
6. Exclude `registry_*` and `checkpoints/` from any public output archive (L-1/L-2).

---

## 8. Fixes applied (2026-08-22)

| ID | File | Fix |
|---|---|---|
| C-1 | `src/privacy/dp_trainer.py` | NaN/Inf loss guard raises `FloatingPointError`; non-finite gradient guard before `optimizer.step()`; removed unnecessary `x_t.requires_grad_()`. |
| P-4 | `src/privacy/dp_trainer.py` | Accounting + noise now use the **minimum** σ over batch timesteps (conservative, RDP-valid). |
| P-2 | `src/privacy/clip_and_noise.py` + trainer | Accountant records **one step per batch** (passed only on first tier); `accountant` param now optional. |
| C-2/P-1 | `scripts/reproduce_end_to_end.py` | σ capped at 4.0 (`min(15/eps, 4.0)`); closed-loop ε stop breaks training when `eps_spent >= target_eps`. |
| C-3/C-4 | `scripts/reproduce_end_to_end.py` | Output validation gate: refuses export if any column is all-NaN or row uniqueness < 0.01. |
| E-1 | `src/evaluation/privacy_metrics.py` | Shared category vocabulary built across train/holdout/synth (`_build_shared_vocab`); `_factorize_data` maps through it. |
| E-3 | `src/evaluation/utility_metrics.py` | Correlation RMSE returns NaN (not 0.0) when <2 common numeric columns. |
| M-2 | `src/preprocessing/missingness.py` | Dtype-agnostic imputation: numeric-content object/str columns get a NUMERIC median instead of a str-cast mode. |
| M-3 | `src/privacy/risk_tier_assigner.py` | Correlation guard resolves against the initial assignment (order-independent fixed point). |
| M-6 | `src/preprocessing/encoders.py` | Out-of-range indices route to `__other__` (or NaN) instead of silent clipping. |
| M-7 | `src/orchestration/gpu_budget_guard.py` | Removed dead `_gpu_active_seconds` fields and false "GPU-active tracking" claim; documented wall-clock semantics. |
| K-1 | `kaggle_runner/kernel-metadata.json` | `"is_private"` / `"enable_gpu"` are now real JSON booleans. |
| K-4/K-6 | `kaggle_runner/run_pipeline.py` | torch cu118 installed BEFORE requirements.txt; wget retries ×3 with timeout. |

**Verification:** all pandas/numpy-level fixes verified by `scratch/smoke_fixes.py` (E-1, E-3, M-2, M-3, M-6 — ALL PASSED). Torch-dependent paths (C-1, P-1, P-2, P-4, C-2/C-3/C-4) are syntax-checked but require an environment with torch+opacus to execute end-to-end (run `python scripts/reproduce_end_to_end.py` on Kaggle/local GPU env).
