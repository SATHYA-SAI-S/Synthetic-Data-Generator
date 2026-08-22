# Environment Notes — GPU Constraints and Checkpointing Strategy

**Phase:** 2 (Environment Setup)
**Status:** Documented; checkpointing not yet implemented (Phase 6 dependency)

---

## 1. Kaggle GPU-Hour Constraint

Kaggle Notebooks provide the following free-tier GPU resources (as of 2026):

- **GPU quota:** ~30 hours/week (T4 x2 or P100, depending on availability)
- **Session time limit:** 12 hours per session (hard cutoff — kernel is killed)
- **RAM:** ~13–16 GB system RAM, ~16 GB GPU VRAM (T4)
- **Persistent storage:** `/kaggle/working/` survives between sessions (up to ~20 GB); `/kaggle/input/` is read-only dataset mount

### Implications for This Project

| Constraint | Impact | Mitigation |
|---|---|---|
| 30 hr/week quota | A full DP-diffusion training run may exceed this | Checkpoint every N steps; resume across sessions |
| 12 hr session kill | Mid-epoch termination is expected, not exceptional | Checkpoint at epoch boundary AND mid-epoch |
| 16 GB VRAM | Limits batch size; DP-SGD has memory overhead (per-sample gradients) | Gradient accumulation; Opacus virtual steps |
| ~20 GB working storage | Model checkpoints + data + logs must fit | Prune old checkpoints; compress artifacts |

---

## 2. What the Phase 6 Checkpointing System Must Satisfy

The following requirements are **binding constraints** on Phase 6 implementation. Phase 6 developers must not deviate from these without updating this document.

### 2.1 Resumability Contract

A checkpoint must capture all state required to resume training as if it never stopped:

```
Checkpoint bundle (minimum contents):
  - Model weights (state_dict)
  - Optimizer state (state_dict) — includes momentum accumulators
  - Privacy accountant state (Renyi/PRV accountant step count and accumulated moments)
  - Training step counter (global_step)
  - Epoch counter
  - RNG states: Python random, NumPy, PyTorch CPU, PyTorch CUDA
  - Current noise schedule state (for Novelty Mechanism A)
  - Per-tier budget consumed (for Novelty Mechanism B)
  - LR scheduler state (if used)
  - Validation metric history (for early stopping logic)
```

**Critical**: The privacy accountant state MUST be checkpointed. If the accountant is reset on resume, the privacy guarantee is violated (double-spending the budget silently).

### 2.2 Checkpoint Frequency

The Phase 6 checkpointing strategy must support:
- **Epoch-level checkpoints** (mandatory): Save at end of every epoch.
- **Step-level checkpoints** (configurable): Save every `checkpoint_every_n_steps` steps (default: 500). This is the only defense against the 12-hour session kill.
- **Best-model checkpoint**: Separate checkpoint kept for the model with the best validation utility metric.

### 2.3 Atomic Write Requirement

Checkpoint writes must be atomic (write to temp file, then rename). A partial checkpoint caused by a session kill must be detectable and discarded, not silently loaded as corrupted state.

### 2.4 Storage Rotation

Keep at most `max_checkpoints_to_keep` (default: 3) step-level checkpoints. Epoch-level checkpoints may have a longer retention window. Phase 6 must implement a rotation policy.

### 2.5 Cross-Session Path Contract

All checkpoint paths must be relative to a configurable `checkpoint_root` directory (default: `/kaggle/working/checkpoints/`). Absolute paths from a previous session may point to a different ephemeral filesystem and must not be hardcoded.

---

## 3. Recommended Training Schedule for Kaggle Constraints

Given the 30 hr/week limit and ~12 hr session cap:

- **Pretraining phase** (warm-up, no DP): 2–4 hours. Run in a separate session first to confirm model architecture is correct before burning DP budget.
- **DP training phase**: Budget remaining hours. Expect 4–8 sessions of 3–6 hours each depending on dataset size and model complexity.
- **Evaluation**: Run as a separate lightweight session after training is complete.

**Recommendation**: Use Kaggle's "Save & Run All" with accelerator set to GPU. Monitor quota usage at `https://www.kaggle.com/me` — quota resets on a rolling 7-day window, not calendar week.

---

## 4. Phase Dependencies on This Document

- **Phase 5 (Auto-Config Engine)**: Must not schedule training runs that would exceed the GPU quota given a user-specified total budget. The auto-config engine should estimate compute requirements before training begins.
- **Phase 6 (Checkpointing)**: Must implement exactly the contract specified in Section 2 above.
- **Phase 7 (DP-Diffusion Training Loop)**: Must call the checkpointing system at the frequency specified in Section 2.2.
