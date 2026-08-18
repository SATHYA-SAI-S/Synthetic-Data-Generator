# 1. Novelty & Related Work

## Introduction
The generation of synthetic healthcare data poses a fundamental challenge: maintaining deep clinical utility (complex multi-variate correlations) while strictly bounding the risk of re-identification. While traditional generative adversarial networks (GANs) and standard diffusion models provide high fidelity, they are notorious for memorizing outliers.

## Related Work
- **DP-GANs / DP-CTGAN**: Offer privacy guarantees via Differential Privacy (DP), but often suffer from mode collapse on complex tabular boundaries.
- **Standard DDPM (Denoising Diffusion Probabilistic Models)**: Show state-of-the-art utility but lack explicit reconstruction bounds.

## Novelty
This architecture introduces **Tiered Privacy Denoising**. Rather than applying a blanket noise multiplier ($\sigma$) and clip norm ($C$) to the entire gradient vector, we heuristically assign features to Risk Tiers based on their HIPAA status, empirical uniqueness, and cross-column correlation. Distinct parameter groups are clipped and noised proportionally to their risk tier, tracked centrally via a composition-safe RDP Accountant.
