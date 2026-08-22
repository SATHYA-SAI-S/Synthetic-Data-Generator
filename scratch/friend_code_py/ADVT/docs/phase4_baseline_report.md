# Phase 4: Baseline Diffusion Model Report

## 1. Overview
This report summarizes the implementation and sanity checks for the baseline (non-DP) diffusion model, fulfilling the requirements for Phase 4. The diffusion architecture has been successfully layered on top of the Phase 1-3 preprocessing pipeline and schema registry.

## 2. Architecture & Contracts
- **Dynamic Sizing**: The `MLPDenoiser` automatically sizes its input and output dimensions strictly based on the registry's `output_dim` for encoders and scalers. There are no hardcoded dimensions.
- **Forward Process**: Implements a standard Gaussian noise injection schedule (linear $\beta$).
- **Sampler**: Standard reverse DDPM sampling loop from $T$ to $0$.
- **Trainer**: A non-DP PyTorch training loop predicting $\epsilon$.

## 3. Training & Convergence (Smoke Test)
A smoke test was conducted over a synthetic batch to ensure dimensions align and the model computes gradients without crashing.
- **Loss Convergence**: `DiffusionTrainer` successfully lowers MSE loss across epochs.
- **Dimensionality**: Verified against varying synthetic schema dimensions (e.g., width 12 and width 45). The denoiser correctly adapts its architecture.

## 4. Sampling Sanity Check
Reverse sampling successfully maps Gaussian noise $\mathcal{N}(0, I)$ back into the continuous proxy space defined by the preprocessing pipeline. 

The generated numpy arrays match the exact dimensional signature required by `pipeline.inverse_transform()`, which decodes the continuous outputs back into human-readable categorical strings and imputed numerics, fully honoring the Phase 3 schema.

## 5. Frozen Checkpoint
A simulated checkpoint of this baseline model acts as our utility ceiling before Phase 5 differential privacy constraints are added.
- **Path**: `checkpoints/phase4_baseline_cpu.pt` (stub)

## Next Steps
The codebase is now fully prepared to integrate DP-SGD and the Phase 5 adaptive noise schedule + risk tier assigner via `AbstractPrivacyAccountant`.
