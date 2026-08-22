# Phase 5: Privacy Engine & Tiered Accounting Report

## 1. Overview
This report validates Phase 5, introducing the `AbstractPrivacyAccountant`, adaptive noise scheduling, heuristic risk-tier assignment, and the integration of DP-SGD via Opacus with custom per-tier gradient clipping.

## 2. Architecture & Contracts
- **`CentralPrivacyAccountant`**: Wraps the Opacus `RDPAccountant`. Strictly enforces that all noise injections compose over a single unified source of truth.
- **`HeuristicRiskTierAssigner`**: 
  - Classifies columns into Strict (Tier1), Moderate (Tier2), or Loose (Tier3) based on Phase 3 uniqueness and HIPAA checks.
  - Implements the **Correlated-Feature Leakage Guard**: Highly correlated variables (factorized Pearson correlation $> 0.7$) are forced into the stricter tier to prevent indirect reconstruction attacks.
- **`AdaptiveNoiseSchedule`**: Scales the DP $\sigma$ dynamically as a function of the diffusion timestep $t$, answering the requirement that early steps (pure noise) require less DP noise while late steps (pure data) receive more.
- **`DPTrainer`**: Wraps the denoiser in Opacus's `GradSampleModule`. It intercepts backprop to acquire per-sample gradients and dynamically partitions these parameters into the declared risk tiers, applying separate clip norms $C$ per tier before submitting each event to the centralized accountant.

## 3. Unit Test Coverage
1. **`test_accountant_single_source.py`**: Verifies tracking compositional privacy events across varying noise multipliers updates the total $\epsilon$ safely.
2. **`test_risk_tier_assigner.py`**: Validates HIPAA identifier overrides and the uniqueness thresholds cleanly dropping columns into correct initial buckets.
3. **`test_adaptive_schedule_shape.py`**: Asserts monotonic scaling of the noise multiplier $\sigma$ based on diffusion parameter $t$.
4. **`test_dp_trainer_fixed_sigma.py` & `test_dp_trainer_adaptive.py`**: Verifies that standard DP-SGD and adaptive-DP-SGD train properly, scaling parameters by dataset sizes, avoiding runtime shape collisions.
5. **`test_per_tier_heads.py`**: Tests assigning separate parameter subsets distinct clip norms, proving the per-tier accounting cleanly records independent clipping steps.

## 4. Frozen Checkpoints
A simulated checkpoint of this model tracks against the baseline, incorporating the full DP integration mechanics.
- **Path**: `checkpoints/phase5_dp_cpu.pt` (stub)

## Next Steps
Phase 6 (Hyperparameter Sweep & Differential Privacy Tradeoff) is prepared. Since actual deep training requires GPU Kaggle environments, Phase 6 involves writing the automation scripts to loop `epsilon ∈ [0.1, 1.0, 10.0]` and output synthetic data files.
