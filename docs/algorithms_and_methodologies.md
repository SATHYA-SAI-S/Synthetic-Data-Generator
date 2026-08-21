# Complete Algorithms & Methodologies Reference
### Privacy-Preserving Synthetic Healthcare Data Generation Framework

> **Total: 32 distinct algorithms, techniques, and methodologies across 6 categories**

---

## 🧠 Category 1: Generative AI — Diffusion Model

| # | Algorithm / Technique | Source File | Description |
|---|---|---|---|
| 1 | **Denoising Diffusion Probabilistic Model (DDPM)** | `denoiser.py`, `sampler.py` | Core generative engine (Ho et al., 2020). Learns to reverse a Gaussian noise process to generate synthetic tabular records. |
| 2 | **MLPDenoiser (Time-Conditioned MLP)** | `denoiser.py` | Neural backbone: `Linear → SiLU → LayerNorm` with learned timestep embedding injected via residual addition. Predicts noise $\epsilon_\theta(x_t, t)$. |
| 3 | **SiLU / Swish Activation** | `denoiser.py` | $\text{SiLU}(x) = x \cdot \sigma(x)$. Smooth, non-monotonic activation used throughout the denoiser. |
| 4 | **Layer Normalization** | `denoiser.py` | Stabilizes tabular gradient flow by normalizing across features per sample. |
| 5 | **Linear Noise Schedule** | `schedule.py` | $\beta_t = \beta_{\text{start}} + \frac{t}{T-1}(\beta_{\text{end}} - \beta_{\text{start}})$, with $\bar{\alpha}_t = \prod_{s=0}^t (1 - \beta_s)$. Default: $T=1000$, $\beta \in [10^{-4}, 0.02]$. |
| 6 | **Closed-Form Forward Diffusion** | `forward_process.py` | $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$. |
| 7 | **DDPM Ancestral Reverse Sampling** | `sampler.py` | Iterative denoising: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right) + \sqrt{\beta_t}\,z$. Chunked generation (batch 8192) for memory efficiency. |
| 8 | **Schema Adapter Network** | `adapter.py` | Parameter-efficient transfer learning: frozen pre-trained backbone + trainable input/output projection layers. Enables fine-tuning on small datasets with minimal privacy cost. |

---

## 🔒 Category 2: Differential Privacy

| # | Algorithm / Technique | Source File | Description |
|---|---|---|---|
| 9 | **DP-SGD (Differentially Private SGD)** | `dp_trainer.py` | Opacus `GradSampleModule` computes per-sample gradients. Each gradient is clipped to norm $C$ and Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ is added. |
| 10 | **Per-Sample Gradient Clipping** | `clip_and_noise.py` | $\text{clip}_i = \min(1, C / \|\|g_i\|\|_2)$. Bounds sensitivity of any single patient's contribution. |
| 11 | **Gaussian Noise Mechanism** | `clip_and_noise.py` | $\tilde{g} = \frac{1}{B}\left(\sum_i \text{clip}_i \cdot g_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$. Core DP mechanism. |
| 12 | **Rényi Differential Privacy (RDP) Accountant** | `accountant.py` | Tracks privacy via RDP composition, then converts: $\varepsilon(\delta) = \min_\alpha\left(\varepsilon_{\text{RDP}}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1}\right)$. |
| 13 | **Adaptive Per-Timestep Noise Schedule** | `adaptive_schedule.py` | $\sigma(t) = \sigma_{\text{base}} \times (0.5 + \text{ratio}(t))$. More noise at high-signal timesteps ($t \approx 0$), less noise at pure-noise timesteps ($t \approx T$). |
| 14 | **Per-Feature Risk-Tier Privacy Budget** | `risk_tier_assigner.py` | **Novel contribution.** Non-uniform privacy allocation: Tier 1 (near-identifiers, $>80\%$ unique) → strict clipping. Tier 2 ($>15\%$ unique) → moderate. Tier 3 → loose. |
| 15 | **Correlated-Feature Leakage Guard** | `risk_tier_assigner.py` | Computes $\|R\|$ (absolute Pearson correlation). If $\|r_{ij}\| > 0.7$, promotes the looser-tier feature to the stricter tier. Prevents indirect re-identification. |

---

## 📊 Category 3: Data Profiling & Preprocessing

| # | Algorithm / Technique | Source File | Description |
|---|---|---|---|
| 16 | **One-Hot Encoding** | `encoders.py` | Low-cardinality ($\le 15$ unique). Rare categories ($<10$ count) binned to `__other__`. Reserved `__null__` token at index 0. Inverted via argmax. |
| 17 | **Frequency-Descending Integer Encoding** | `encoders.py` | High-cardinality. Categories ranked by frequency → integer tokens. Compatible with embedding tables. |
| 18 | **Z-Score Standard Scaling** | `pipeline.py` | $z = (x - \mu) / \sigma$. Default for continuous features. |
| 19 | **Min-Max Scaling** | `pipeline.py` | $x_{\text{scaled}} = (x - \min) / (\max - \min)$. |
| 20 | **Robust (IQR) Scaling** | `pipeline.py` | $x_{\text{scaled}} = (x - \text{median}) / (Q_{75} - Q_{25})$. Outlier-resistant. |
| 21 | **Missingness Indicator Injection** | `pipeline.py` | Binary flag column ($1 = \text{missing}$) if missing rate $\ge 1\%$. Preserves clinical missing-data patterns. |
| 22 | **HIPAA Safe Harbor Regex Detection** | `dataset_profiler.py` | Boundary-anchored regex for all 18 HIPAA identifier categories (SSN, MRN, Names, DOB, ZIP, IP, etc.). |
| 23 | **Cramér's V Structural Missingness Test** | `dataset_profiler.py` | $V = \sqrt{\chi^2 / (N \cdot \min(R-1, C-1))}$. Flags missingness as `STRUCTURAL` if $V \ge 0.40$. |
| 24 | **Point-Biserial Correlation** | `dataset_profiler.py` | Continuous vs. binary missing indicator. Detects non-random missingness patterns. |
| 25 | **Dynamic Numeric Verification** | `pipeline.py` | `pd.to_numeric(col, errors="coerce")` — reroutes alphanumeric "continuous" columns to categorical encoding at runtime. |

---

## 📈 Category 4: Evaluation & Red-Teaming

| # | Algorithm / Technique | Source File | Description |
|---|---|---|---|
| 26 | **Distance-Based Membership Inference Attack (D-MIA)** | `privacy_metrics.py` | $\text{Risk} = \min(1, \max(0, \bar{d}_{\text{holdout}} - \bar{d}_{\text{train}}) / \bar{d}_{\text{holdout}})$. Simulates adversarial re-identification. |
| 27 | **Total Variation Distance (TVD)** | `utility_metrics.py` | $\text{TVD} = \frac{1}{2}\sum_c \|P_{\text{real}}(c) - P_{\text{synth}}(c)\|$. Categorical distribution fidelity. |
| 28 | **Kolmogorov-Smirnov (KS) Two-Sample Test** | `utility_metrics.py` | $D = \sup_x \|F_{\text{real}}(x) - F_{\text{synth}}(x)\|$. Continuous distribution fidelity. |
| 29 | **Bivariate Correlation Matrix RMSE** | `utility_metrics.py` | $\text{RMSE} = \sqrt{\frac{1}{M^2}\sum_{i,j}(R_{ij}^{\text{real}} - R_{ij}^{\text{synth}})^2}$. Measures inter-feature relationship preservation. |
| 30 | **TSTR (Train on Synthetic, Test on Real)** | `evaluate_tstr.py` | Trains `HistGradientBoostingClassifier` on synthetic data, evaluates on real holdout. Utility Retention $= \text{AUC}_{\text{TSTR}} / \text{AUC}_{\text{TRTR}} \times 100\%$. |

---

## 🏗️ Category 5: Optimization & Infrastructure

| # | Algorithm / Technique | Source File | Description |
|---|---|---|---|
| 31 | **Adam Optimizer** | `reproduce_end_to_end.py` | Adaptive moment estimation. $\text{lr} = 2 \times 10^{-4}$, default $\beta = (0.9, 0.999)$. |
| 32 | **Compute Budget Guard** | `gpu_budget_guard.py` | Wall-clock + GPU-active time tracking with persistent state. Aborts training if cumulative GPU time exceeds budget (default 30h). |

---

## 🔬 Category 6: Key Hyperparameters

| Parameter | Default Value | Defined In |
|---|---|---|
| Diffusion Timesteps $T$ | 1000 | `schema.py` |
| $\beta_{\text{start}}, \beta_{\text{end}}$ | $10^{-4}, 0.02$ | `schema.py` |
| Hidden Dims | $[256, 256, 256]$ | `schema.py` |
| Batch Size | 256 | `schema.py` |
| Learning Rate | $2 \times 10^{-4}$ | `schema.py` |
| Training Epochs | 50 | `schema.py` |
| Target $\varepsilon$ | 1.0 | `schema.py` |
| Target $\delta$ | $10^{-5}$ | `schema.py` |
| Max Grad Norm $C$ | 1.0 | `schema.py` |
| Random Seed | 42 | `schema.py` |
| Sweep Epsilons | $[0.1, 1.0, 10.0]$ | `reproduce_end_to_end.py` |
