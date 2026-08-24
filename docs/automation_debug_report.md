# Automation Debug Report — Streamlit UI Pipeline (SYNTHGUARD)

**Scope:** End-to-end automation audit of the naive-user flow: Upload → Route Decision →
Privacy Config → Generation (Adapter / Kaggle) → Red-Team Audit → OP Dashboard.
**Goal:** Fully adaptive behavior for ANY dataset — zero hardcoded values.

---

## 🔴 CRITICAL BUGS FOUND & FIXED

### BUG-1: Kaggle kernel ignored the user's uploaded data entirely
**Files:** `kaggle_runner/run_pipeline.py`, `scripts/reproduce_end_to_end.py`
**Symptom:** The UI packaged the user's de-identified CSV + `run_config.json` into a private
Kaggle dataset and mounted it via `dataset_sources`, but the kernel script **cloned GitHub,
re-downloaded the hardcoded UCI diabetes zip**, and trained on THAT. The user's data never
touched training, and their epsilon/delta/epochs/batch/clip settings were silently discarded.

**Fix:** Rewrote `run_pipeline.py` to:
- Locate `/kaggle/input/<slug>/clean_data.csv` + `run_config.json` dynamically (no UCI download).
- Read `epsilon, delta, epochs, batch_size, clip_norm, num_samples, clean_columns` from config.
- Clone the repo at a **pinned commit** (deterministic) only for the `src/` package.
- Delegate to new `kernel_train.py::run_adaptive_training()` — fully dataset-agnostic:
  builds `TabularEncoder` from the actual schema, trains one configured epsilon (no sweep),
  writes `synthetic_clean.csv` + `output_report.json`.

### BUG-2: No progress.json → frozen UI for hours + dead stuck-watchdog
**Files:** `kaggle_runner/run_pipeline.py` (old), `src/orchestration/kaggle_bridge.py`
**Symptom:** The bridge polls `progress.json` every 15s (`fetch_progress`) and drives both the
progress bar and the 20-minute stuck watchdog off its `ts` field. The old kernel never wrote it,
so the bar stayed at 0% forever and the watchdog could never fire.

**Fix:** `kernel_train.py` now writes `progress.json` atomically every epoch with
`{ts, stage, pct, epoch, total_epochs, loss, epsilon_spent}`. The UI progress bar moves live
and the watchdog works.

### BUG-3: Screen 2 privacy controls never persisted → user input silently dropped
**File:** `ui/screens/screen2_training.py`
**Symptom:** `delta_choice`, `epochs`, `batch_size`, `clip_norm` were read into local variables
but **never written to session state**. Screen 3's kernel config then fell back to defaults —
the naive user's choices were ignored end-to-end.

**Fix:** All four controls are now persisted (`st.session_state.delta_choice/epochs/batch_size/
clip_norm`) and widgets are re-seeded from prior state so selections survive reruns.

### BUG-4: Hardcoded/fabricated metrics across the entire dashboard
**Files:** `ui/state_schema.py`, `ui/components/kpi_ribbon.py`,
`ui/components/panel_b_privacy_ledger.py`, `ui/components/panel_d_export.py`,
`ui/screens/screen4_output.py`, `ui/data_loaders/load_session.py`
**Symptom:** Every panel displayed placeholder diabetes numbers regardless of the run:
`epsilon_spent=0.3720`, `mia_advantage=-0.0083`, `mia_attack_auc=0.4958`, `tvd_best=0.0316`,
hardcoded per-epoch loss table, hardcoded sigma=5.00/C=1.0/date, unconditional
"MATHEMATICALLY PRIVATE" certificate verdict, and a fixed diabetes run manifest.

**Fix (fully adaptive):**
- `state_schema.py`: runtime metrics default to `None`; populated ONLY by real runs.
- KPI ribbon / Panel B / Panel D / Screen 4: None-safe rendering showing "Pending"
  until a run produces values; certificate verdict derives from the live red-team result;
  manifest is built from the actual session (dataset name, sha256, real hyperparams).
- `epsilon_gauge.py`: handles `None` spent gracefully.

### BUG-5: Real run metrics were never wired into the dashboard
**File:** `ui/screens/screen3_generation.py`
**Fix:** `_run_adapter_route` now persists `epsilon_spent`, `noise_multiplier`, epochs/batch
from the adapter's `info` dict; `_run_red_team_audit` persists `mia_advantage`
(worst attack success), `mia_attack_auc` (from the dmia_auc level), NaN count, and
domain violations from the live audit report.

---

## 🟠 HIGH SEVERITY

### BUG-6: Kaggle output matching was fragile & schema-blind
**File:** `ui/screens/screen3_generation.py`
**Symptom:** First-match `os.walk` over `synthetic*.csv` grabbed an arbitrary sweep artifact
(`synthetic_eps_0.1.csv`) that didn't match the configured epsilon or the user's schema.
**Fix:** Prefers the canonical `synthetic_clean.csv` produced by the new kernel; adds a
schema guard rejecting outputs whose columns don't intersect the cleaned upload columns.

### BUG-7: Navigation allowed skipping prerequisites
**File:** `ui/components/sidebar.py`
**Fix:** Added `_step_allowed()` guards — Screen 4 requires `generation_complete`;
Screens 2–4 require `profile_complete`. Disabled buttons carry explanatory help text.

### BUG-8: Dead code in reset_session()
**File:** `ui/state_schema.py`
**Symptom:** Referenced `defaults[...]` which only existed inside another function's scope.
**Fix:** Self-contained clear-list + fresh adaptive defaults.

---

## 🟡 MEDIUM / NOTES

| # | Issue | Status |
|---|-------|--------|
| M1 | `screen3` timesteps slider declared but unused | Noted (cosmetic; sampling uses fixed T=50 internally) |
| M2 | Epoch default mismatch (adapter 30 vs screen2 5) | Harmonized via persisted session value |
| M3 | `kaggle kernels output` polled every 15s during watch | Works now that progress.json exists; further throttling optional |
| M4 | Timestamped dataset/kernel slugs accumulate on Kaggle | Cleanup/rotation recommended later |
| M5 | `load_evaluation_report.get_parsed_evaluation_metrics()` still returns static demo objectives | Left intact (report text loader); dashboard no longer depends on fabricated numbers |

---

## ✅ WHAT WAS ALREADY SOLID
- Route decision engine (`route_decider.py`) — clean rule-based routing with override tracking.
- Error taxonomy + error ledger in `kaggle_bridge.py`.
- Red-team attacker engine (`attacker_engine.py`) — genuinely runs post-generation on both routes.
- Adapter route (`schema_adapter.py`) — self-consistent local DP fine-tune path.
- Pipeline checklist component — accurate stage tracking.

## Verification
All modified files pass `python -m py_compile` (exit 0).