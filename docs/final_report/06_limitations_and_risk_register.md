# 6. Limitations & Risk Register

## Limitations
1. **Computational Overhead**: Partitioning parameter gradients into strict tiers increases DP-SGD memory utilization. Running the pipeline requires GPU isolation (enforced by `GPUBudgetGuard`).
2. **Missingness Imputation Vulnerability**: The model defaults to linear indicator propagation for missingness. In extremely sparse data, the DP noise severely impacts the discrete thresholds of these variables upon reverse decoding.
3. **Opacus Compatibility**: Strict reliance on `Opacus.GradSampleModule` implies that non-standard PyTorch architectures (e.g. advanced attention blocks with custom autograd functions) cannot be effortlessly injected without custom DP backward hooks.

## Risk Register Verification
- **R-01 (Sparse-NaN Crash)**: Handled seamlessly by the `pipeline.py` raw-string injection patches applied in early audits.
- **R-02 (Near-ID Reversal)**: The framework successfully categorizes Near-IDs into `Tier 1` strict subsets, mitigating raw deletion artifacts while preserving privacy.
