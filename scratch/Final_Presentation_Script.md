# Final Presentation: Complete Architecture & Execution
Privacy-Preserving Synthetic Healthcare Data Generation

Sathya Sai S  ·  Sudharshini Manikandan  ·  Vishwa D

## Slide 1: Title Slide
- Title: Privacy-Preserving Synthetic Healthcare Data Generation
- Subtitle: A Generic, Auto-Configuring Diffusion + Differential Privacy Framework
- Presenters: Sathya Sai S, Sudharshini Manikandan, Vishwa D

## Slide 2: The Problem
- Healthcare data is locked behind strict privacy laws (HIPAA).
- Researchers need data to build AI, but hospitals cannot share it.
- Existing Solutions: Basic anonymization fails against linkage attacks. DP-GANs suffer from mode collapse.

## Slide 3: Our Novel Solution
- We built a generic, auto-configuring Denoising Diffusion Probabilistic Model (DDPM).
- Core Innovation: Per-feature risk tiering. Instead of adding uniform noise, we dynamically allocate privacy budgets based on column sensitivity.

## Slide 4: System Architecture (Layer 0 to Layer 4)
- Layer 0 (Auto-Config): Profiles any dataset, detects HIPAA identifiers.
- Layer 1 (Preprocessing): Missingness handling and robust One-Hot/Gaussian encoding.
- Layer 2 (Diffusion): MLPDenoiser with a 1,000-step linear noise schedule.
- Layer 3 (Privacy): Opacus DP-SGD integration for formal guarantees.

## Slide 5: Real-World Execution on Kaggle
- Successfully trained on the 100,000+ row UCI Diabetic Readmission dataset.
- Fully automated cloud deployment utilizing GPU acceleration (Tesla P100 / T4).
- Swept across multiple privacy budgets (Epsilon 0.1, 1.0, 10.0).

## Slide 6: Results & Evaluation
- Utility: Maintained high statistical fidelity (Bivariate Correlation RMSE).
- Privacy: Defeated simulated Membership Inference Attacks (MIA). Zero identical memorized records.
- Tradeoff Curve: Demonstrated the mathematical relationship between DP noise and model loss.

## Slide 7: Conclusion & Future Scope
- Conclusion: We have successfully built a pipeline that transforms locked raw data into safe, shareable synthetic data.
- Future Scope: Extend to multi-modal data (e.g., tabular + medical imaging) and deploy as a hospital-facing API.
