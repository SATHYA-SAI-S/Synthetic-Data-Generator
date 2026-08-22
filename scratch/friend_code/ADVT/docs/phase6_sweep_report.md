# Phase 6: Hyperparameter Sweeps & Generation Architecture

## 1. Overview
This report outlines the architecture and execution strategy for Phase 6, which orchestrates the hyperparameter sweep over different privacy budgets ($\epsilon \in \{0.1, 1.0, 10.0\}$) and generates the final synthetic datasets. As per the constraints, the execution layer is designed to be easily deployable to a GPU environment (like Kaggle) while strictly adhering to compute budgets.

## 2. Orchestration Architecture
- **`GPUBudgetGuard`**: A dedicated singleton that tracks elapsed execution time. It writes its state to `gpu_budget_state.json`, ensuring that if a Kaggle kernel is preempted or restarted, the 30-hour weekly ceiling is mathematically enforced across sessions.
- **`reproduce_end_to_end.py`**: The primary entrypoint script. For each target $\epsilon$:
  1. Instantiates a clean Phase 1-3 preprocessing pipeline and `FileSchemaRegistry`.
  2. Profiles the data and dynamically assigns Risk Tiers via the `HeuristicRiskTierAssigner`.
  3. Initializes the DP-Diffusion model, mapping the inverse of $\epsilon$ to the base $\sigma$ parameter of the `AdaptiveNoiseSchedule`.
  4. Trains the model, updating the `CentralPrivacyAccountant`.
  5. Runs the reverse DDPM sampler to generate synthetic continuous tensors.
  6. Feeds the tensors through `pipeline.inverse_transform()` to yield the final, perfectly-schema-matched `.csv` files.

## 3. Deployment Strategy (Kaggle)
Since local CPU execution of the full DP-SGD loops over 100,000 rows would be intractably slow, this architecture is fully decoupled from the environment. To run the sweep:
1. Upload the repository to Kaggle.
2. Ensure GPU acceleration (T4/P100) is enabled.
3. Run `python scripts/reproduce_end_to_end.py`.
4. Collect the resulting `synthetic_eps_*.csv` files from the output directory.

## Next Steps
With the synthetic datasets generated across strict privacy budgets, the architecture is ready for **Phase 7: Comprehensive Evaluation (Utility vs. Privacy Tradeoff)**, where we will implement the MIA attacks, structural distance checks, and univariate/bivariate distribution comparisons.
