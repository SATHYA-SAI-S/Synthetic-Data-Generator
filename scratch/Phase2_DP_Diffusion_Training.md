# Phase 2: DP-SGD Diffusion Model & Training
Privacy-Preserving Synthetic Healthcare Data Generation

Sathya Sai S  ·  Sudharshini Manikandan  ·  Vishwa D
Document Version: 1.0

## Table of Contents
1. Executive Summary
2. Diffusion Model Architecture
3. Differential Privacy (DP-SGD) Integration
4. Risk-Tier Assignment Methodology
5. Training Orchestration and Checkpoints

## 1. Executive Summary
This phase covers the core generative AI engine: a Denoising Diffusion Probabilistic Model (DDPM) optimized for tabular data, coupled with Opacus for Differential Privacy (DP-SGD). It guarantees that the synthetic data generated cannot be reverse-engineered to identify any single patient.

## 2. Diffusion Model Architecture
The backbone is an MLPDenoiser with SiLU activations and residual connections. It learns to reverse a Gaussian noise process applied to the preprocessed tabular tensor. A LinearNoiseSchedule over 1,000 timesteps controls the forward diffusion process.

## 3. Differential Privacy (DP-SGD) Integration
We integrated Facebook's Opacus library. The DPTrainer wraps the denoiser in a GradSampleModule, clipping per-sample gradients and adding Gaussian noise. This provides a formal mathematical privacy guarantee.

## 4. Risk-Tier Assignment Methodology
Unlike standard DP-GANs that apply uniform noise, our architecture dynamically assigns privacy budgets based on clinical risk.
- Tier 1 (High Risk): Near-identifiers (e.g., age, race).
- Tier 2 (Medium Risk): Standard clinical variables.
- Tier 3 (Low Risk): Broad indicators.
This maximizes utility while preserving stringent privacy bounds.

## 5. Training Orchestration and Checkpoints
The pipeline orchestrates the training across multiple target epsilons (0.1, 1.0, 10.0). Models are checkpointed via the ComputeBudgetGuard to prevent GPU timeouts. The final output is a suite of models calibrated to different privacy-utility tradeoffs.
