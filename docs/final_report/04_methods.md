# 4. Methods

## Preprocessing
Raw input schemas are dynamically handled. Missingness is captured via explicit indicator columns (if structurally dependent) or pure mean imputation if random. High cardinality features are handled via `FrequencyEncoder`, ensuring embedding table sizes remain computationally bounded.

## DP Mechanism
The model implements a Denoising Diffusion Probabilistic Model (DDPM) wrapped by Opacus for DP-SGD. 
- **Adaptive Schedule**: The noise multiplier ($\sigma_t$) scales linearly based on diffusion time $t$. Early steps containing mostly noise receive less DP budget overhead; late steps containing raw signal are strictly masked.
- **Partitioned Clipping**: Opacus `grad_sample` hooks intercept the backward pass, split gradients into `Tier1`, `Tier2`, and `Tier3` buckets, and clip them to distinct norms ($C_1, C_2, C_3$) before adding differential Gaussian noise.
