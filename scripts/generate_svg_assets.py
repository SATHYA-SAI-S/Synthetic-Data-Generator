"""
Script to generate 4 publication-quality, high-contrast, pure white background (#FFFFFF)
SVG architecture and system modeling diagrams for the ADVT healthcare DP-diffusion project.
"""

import os
from pathlib import Path

docs_dir = Path("docs")
docs_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# 1. System Architecture / C4 Container Diagram SVG
# -------------------------------------------------------------------------------------------------
svg_architecture = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 1000" width="1450" height="1000" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="card-shadow" x="-5%" y="-5%" width="110%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <filter id="glow-shadow" x="-8%" y="-8%" width="116%" height="116%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#3b82f6" flood-opacity="0.12"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a"/>
    </marker>
  </defs>

  <!-- Background Canvas -->
  <rect width="1450" height="1000" fill="#FFFFFF"/>

  <!-- Title Header -->
  <g transform="translate(50, 45)">
    <rect x="0" y="0" width="1350" height="60" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="25" y="36" font-size="20" font-weight="700" fill="#0f172a">ADVT System Architecture &amp; Container Model</text>
    <text x="520" y="36" font-size="14" font-weight="500" fill="#64748b">C4-Level Container Architecture | Differential Privacy Healthcare Synthetic Data Framework</text>
  </g>

  <!-- External Actor / CLI -->
  <g transform="translate(50, 130)">
    <rect x="0" y="0" width="300" height="110" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="2" filter="url(#card-shadow)"/>
    <rect x="15" y="15" width="40" height="40" rx="20" fill="#dcfce7"/>
    <text x="35" y="39" font-size="18" text-anchor="middle" fill="#16a34a">👤</text>
    <text x="65" y="32" font-size="15" font-weight="700" fill="#166534">Data Scientist / Clinician</text>
    <text x="65" y="49" font-size="12" fill="#475569">Defines DP Budget (ε, δ) &amp; Configs</text>
    <path d="M 15 68 L 285 68" stroke="#bbf7d0" stroke-width="1"/>
    <text x="15" y="88" font-size="11" font-weight="600" fill="#15803d">ENTRYPOINT: <tspan font-family="monospace" font-size="11" fill="#0f172a">scripts/reproduce_end_to_end.py</tspan></text>
  </g>

  <!-- Persistent Storage Layer (Left Column) -->
  <g transform="translate(50, 270)">
    <rect x="0" y="0" width="300" height="680" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="15" width="270" height="32" rx="6" fill="#e2e8f0"/>
    <text x="150" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#334155">PERSISTENT STORAGE LAYER</text>

    <!-- Raw Data Store -->
    <g transform="translate(15, 65)">
      <rect x="0" y="0" width="270" height="120" rx="8" fill="#FFFFFF" stroke="#e2e8f0" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#0f172a">🗄️ Raw Clinical Data Store</text>
      <text x="15" y="48" font-size="12" fill="#64748b">Location: <tspan font-family="monospace" fill="#0f172a">data/*.zip, data/*.csv</tspan></text>
      <text x="15" y="68" font-size="11" fill="#475569">• Mixed clinical features (100k+ rows)</text>
      <text x="15" y="86" font-size="11" fill="#475569">• High-cardinality &amp; missingness</text>
      <text x="15" y="104" font-size="11" fill="#dc2626">• Contains raw PHI / HIPAA tokens</text>
    </g>

    <!-- Schema Registry Store -->
    <g transform="translate(15, 205)">
      <rect x="0" y="0" width="270" height="135" rx="8" fill="#FFFFFF" stroke="#e2e8f0" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#0f172a">📦 File Schema Registry</text>
      <text x="15" y="48" font-size="12" fill="#64748b">Location: <tspan font-family="monospace" fill="#0f172a">registry/v{N}/</tspan></text>
      <text x="15" y="68" font-size="11" fill="#475569">• <tspan font-family="monospace" fill="#2563eb">profile.json</tspan> (Pydantic metadata)</text>
      <text x="15" y="86" font-size="11" fill="#475569">• <tspan font-family="monospace" fill="#2563eb">pipeline_state.joblib</tspan> (Scalers)</text>
      <text x="15" y="104" font-size="11" fill="#475569">• <tspan font-family="monospace" fill="#2563eb">latest.txt</tspan> (Atomic version pointer)</text>
      <text x="15" y="122" font-size="11" fill="#16a34a">✓ SHA-256 Checksum Verified</text>
    </g>

    <!-- GPU State Store -->
    <g transform="translate(15, 360)">
      <rect x="0" y="0" width="270" height="110" rx="8" fill="#FFFFFF" stroke="#e2e8f0" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#0f172a">⏱️ Compute Budget Store</text>
      <text x="15" y="48" font-size="12" fill="#64748b">Location: <tspan font-family="monospace" fill="#0f172a">gpu_state.json</tspan></text>
      <text x="15" y="68" font-size="11" fill="#475569">• Wall-clock session tracking</text>
      <text x="15" y="86" font-size="11" fill="#475569">• Weekly hard cap (e.g. 30.0h)</text>
      <text x="15" y="102" font-size="11" fill="#16a34a">✓ Atomic .tmp -> replace save</text>
    </g>

    <!-- Synthetic Output & Reports -->
    <g transform="translate(15, 490)">
      <rect x="0" y="0" width="270" height="145" rx="8" fill="#FFFFFF" stroke="#e2e8f0" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#0f172a">📊 Synthetic Data &amp; Reports</text>
      <text x="15" y="48" font-size="12" fill="#64748b">Location: <tspan font-family="monospace" fill="#0f172a">outputs/sweep_results/</tspan></text>
      <text x="15" y="68" font-size="11" fill="#475569">• <tspan font-family="monospace" fill="#16a34a">synthetic_eps_*.csv</tspan></text>
      <text x="15" y="86" font-size="11" fill="#475569">• <tspan font-family="monospace" fill="#16a34a">sweep_report.json</tspan></text>
      <text x="15" y="104" font-size="11" fill="#475569">• Empirical Privacy (D-MIA)</text>
      <text x="15" y="122" font-size="11" fill="#475569">• Statistical Fidelity (KS/TVD)</text>
    </g>
  </g>

  <!-- Core Subsystems Area (Right/Center) -->

  <!-- Container 1: Ingestion, Profiling & Config -->
  <g transform="translate(390, 130)">
    <rect x="0" y="0" width="480" height="340" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="12" width="450" height="30" rx="6" fill="#e0f2fe"/>
    <text x="240" y="32" font-size="13" font-weight="700" text-anchor="middle" fill="#0369a1">CONTAINER 1: CONFIGURATION &amp; DATA PROFILING (src/profiling/, src/config/)</text>

    <!-- Config Engine Card -->
    <g transform="translate(15, 55)">
      <rect x="0" y="0" width="450" height="100" rx="8" fill="#FFFFFF" stroke="#bae6fd" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="25" font-size="14" font-weight="700" fill="#0369a1">⚙️ PipelineConfig Engine (Pydantic V2)</text>
      <text x="15" y="45" font-size="11" fill="#475569">• CardinalityConfig (low_card_max=15, rare_min_freq=10)</text>
      <text x="15" y="63" font-size="11" fill="#475569">• MissingnessConfig (drop_threshold=0.80, indicator_inj=0.01)</text>
      <text x="15" y="81" font-size="11" fill="#475569">• SmallNConfig (min_n=100) &amp; PrivacyConfig (target_eps, delta)</text>
    </g>

    <!-- Dataset Profiler Card -->
    <g transform="translate(15, 170)">
      <rect x="0" y="0" width="450" height="150" rx="8" fill="#FFFFFF" stroke="#bae6fd" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="25" font-size="14" font-weight="700" fill="#0369a1">🔍 DatasetProfiler &amp; HIPAA Scanner</text>
      <text x="15" y="45" font-size="11" fill="#475569">• 18 HIPAA Safe Harbor Regex Categories (case-insensitive token check)</text>
      <text x="15" y="63" font-size="11" fill="#475569">• Vectorized Dtype Inference (Continuous, Ordinal, Categorical, Binary)</text>
      <text x="15" y="81" font-size="11" fill="#475569">• Structural Missingness Correlation (Point-Biserial &amp; Cramér's V)</text>
      <text x="15" y="99" font-size="11" fill="#475569">• High-Missingness &amp; Near-Identifier Exclusion Flagging</text>
      <rect x="15" y="115" width="420" height="24" rx="4" fill="#f0f9ff"/>
      <text x="225" y="131" font-size="11" font-weight="600" text-anchor="middle" fill="#0284c7">Emits: Immutable DatasetProfile (JSON-serializable)</text>
    </g>
  </g>

  <!-- Container 2: Preprocessing & Invertible Transformations -->
  <g transform="translate(900, 130)">
    <rect x="0" y="0" width="500" height="340" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="12" width="470" height="30" rx="6" fill="#fef3c7"/>
    <text x="250" y="32" font-size="13" font-weight="700" text-anchor="middle" fill="#b45309">CONTAINER 2: PREPROCESSING &amp; TRANSFORMATION (src/preprocessing/)</text>

    <!-- Missingness Handler Card -->
    <g transform="translate(15, 55)">
      <rect x="0" y="0" width="470" height="100" rx="8" fill="#FFFFFF" stroke="#fde68a" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="25" font-size="14" font-weight="700" fill="#b45309">🩹 MissingnessHandler (Flag Injection)</text>
      <text x="15" y="45" font-size="11" fill="#475569">• Injects binary indicators: <tspan font-family="monospace">&lt;col&gt;__missing_flag</tspan> (1/0)</text>
      <text x="15" y="63" font-size="11" fill="#475569">• Imputes median (numeric) / mode (categorical) as placeholders</text>
      <text x="15" y="81" font-size="11" fill="#475569">• Invertible: Exact NaN restoration on synthesis via mask thresholding</text>
    </g>

    <!-- Invertible Encoders & Scalers Card -->
    <g transform="translate(15, 170)">
      <rect x="0" y="0" width="470" height="150" rx="8" fill="#FFFFFF" stroke="#fde68a" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="25" font-size="14" font-weight="700" fill="#b45309">🔄 Invertible Encoders &amp; Scalers</text>
      <text x="15" y="45" font-size="11" fill="#475569">• <tspan font-weight="600">OneHotEncoder</tspan>: Low-cardinality OHE with <tspan font-family="monospace">__null__</tspan> / <tspan font-family="monospace">__other__</tspan> bins</text>
      <text x="15" y="63" font-size="11" fill="#475569">• <tspan font-weight="600">FrequencyEncoder</tspan>: High-cardinality freq-ordered token indices</text>
      <text x="15" y="81" font-size="11" fill="#475569">• <tspan font-weight="600">StandardScaler / RobustScaler</tspan>: Z-score &amp; IQR normalization</text>
      <rect x="15" y="105" width="440" height="32" rx="4" fill="#fffbeb"/>
      <text x="235" y="125" font-size="11" font-weight="600" text-anchor="middle" fill="#d97706">Forward: fit_transform() -> Tensor(N, D) | Reverse: inverse_transform()</text>
    </g>
  </g>

  <!-- Container 3: Privacy & DP-SGD Diffusion Engine (The Protected Enclave) -->
  <g transform="translate(390, 500)">
    <rect x="0" y="0" width="1010" height="450" rx="14" fill="#fef2f2" stroke="#f87171" stroke-width="2" stroke-dasharray="6 4" filter="url(#glow-shadow)"/>
    <rect x="20" y="15" width="970" height="34" rx="6" fill="#fee2e2"/>
    <text x="505" y="38" font-size="14" font-weight="800" text-anchor="middle" fill="#b91c1c">CONTAINER 3: PROTECTED PRIVACY &amp; DP-DIFFUSION ENCLAVE (src/privacy/, src/diffusion/)</text>

    <!-- Sub-Box 1: Risk Tier Assigner -->
    <g transform="translate(25, 65)">
      <rect x="0" y="0" width="290" height="210" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#b91c1c">🎯 Heuristic Risk Tier Assigner</text>
      <text x="15" y="48" font-size="11" fill="#475569">• Uniqueness &amp; HIPAA Heuristics:</text>
      <text x="25" y="66" font-size="11" fill="#dc2626">  - Tier 1: Strict (HIPAA / unique &gt; 0.8)</text>
      <text x="25" y="84" font-size="11" fill="#d97706">  - Tier 2: Moderate (unique &gt; 0.15)</text>
      <text x="25" y="102" font-size="11" fill="#16a34a">  - Tier 3: Loose (low cardinality)</text>
      <path d="M 15 115 L 275 115" stroke="#fee2e2" stroke-width="1"/>
      <text x="15" y="135" font-size="12" font-weight="700" fill="#b91c1c">🛡️ Correlation Leakage Guard</text>
      <text x="15" y="153" font-size="11" fill="#475569">• Pairwise factorized correlation matrix</text>
      <text x="15" y="171" font-size="11" fill="#475569">• If corr(col1, col2) &gt; 0.70 threshold:</text>
      <text x="15" y="191" font-size="11" font-weight="600" fill="#b91c1c">  => Promotes both to stricter tier!</text>
    </g>

    <!-- Sub-Box 2: Diffusion Architecture -->
    <g transform="translate(330, 65)">
      <rect x="0" y="0" width="310" height="210" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#b91c1c">🌪️ Diffusion Denoiser &amp; Schedule</text>
      <text x="15" y="48" font-size="11" fill="#475569">• <tspan font-weight="600">MLPDenoiser</tspan>: Time-conditioned MLP</text>
      <text x="25" y="66" font-size="11" fill="#475569">  - Linear -> SiLU -> LayerNorm layers</text>
      <text x="25" y="84" font-size="11" fill="#475569">  - Embedding(T) time conditioning</text>
      <text x="15" y="104" font-size="11" fill="#475569">• <tspan font-weight="600">LinearNoiseSchedule</tspan>: Closed-form forward</text>
      <text x="25" y="122" font-size="11" font-family="monospace" fill="#0f172a">  q(x_t|x_0) = √(α_bar)*x_0 + √(1-α_bar)*ε</text>
      <text x="15" y="144" font-size="11" fill="#475569">• <tspan font-weight="600">DDPM Sampler</tspan>: Reverse diffusion</text>
      <text x="25" y="162" font-size="11" fill="#475569">  - Iterative reverse from pure noise x_T</text>
      <text x="25" y="180" font-size="11" fill="#475569">  - Numerical underflow &amp; variance guards</text>
      <text x="25" y="198" font-size="11" font-weight="600" fill="#16a34a">  => Emits synthetic tensor</text>
    </g>

    <!-- Sub-Box 3: DP-SGD Engine & Accounting -->
    <g transform="translate(655, 65)">
      <rect x="0" y="0" width="335" height="210" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#card-shadow)"/>
      <text x="15" y="28" font-size="14" font-weight="700" fill="#b91c1c">🔒 DPTrainer (DP-SGD Engine)</text>
      <text x="15" y="48" font-size="11" fill="#475569">• <tspan font-weight="600">Opacus GradSampleModule</tspan>: Per-sample grads</text>
      <text x="15" y="68" font-size="11" fill="#475569">• <tspan font-weight="600">Per-Sample L2 Clip</tspan>: factor = min(1, C / ||g_i||_2)</text>
      <text x="15" y="88" font-size="11" fill="#475569">• <tspan font-weight="600">Gaussian Noise</tspan>: Noise ~ N(0, σ^2 * C^2 * I)</text>
      <text x="15" y="108" font-size="11" fill="#475569">• <tspan font-weight="600">Adaptive Noise Schedule</tspan>: σ(t) = base*(0.5 + ratio)</text>
      <path d="M 15 125 L 320 125" stroke="#fee2e2" stroke-width="1"/>
      <text x="15" y="145" font-size="12" font-weight="700" fill="#b91c1c">🧮 CentralPrivacyAccountant</text>
      <text x="15" y="165" font-size="11" fill="#475569">• Rényi Differential Privacy (RDP) Tracker</text>
      <text x="15" y="183" font-size="11" fill="#475569">• Single source of truth for all noise steps</text>
      <text x="15" y="201" font-size="11" font-weight="600" fill="#15803d">✓ Guarantees strict (ε, δ)-DP bound</text>
    </g>

    <!-- Bottom Flow in Protected Enclave: Step Pipeline -->
    <g transform="translate(25, 290)">
      <rect x="0" y="0" width="965" height="140" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1" filter="url(#card-shadow)"/>
      <text x="20" y="25" font-size="13" font-weight="700" fill="#0f172a">DP-SGD Mini-Batch Execution Pipeline (Strict Isolation)</text>
      
      <!-- Mini Flow Blocks -->
      <rect x="20" y="40" width="160" height="75" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
      <text x="100" y="62" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">1. Forward Diffuse</text>
      <text x="100" y="80" font-size="10" text-anchor="middle" fill="#64748b">Batch x_0 -> x_t</text>
      <text x="100" y="96" font-size="10" text-anchor="middle" fill="#64748b">Sample t ~ U(0, T)</text>

      <path d="M 180 77 L 205 77" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)"/>

      <rect x="210" y="40" width="165" height="75" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
      <text x="292" y="62" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">2. Per-Sample Grad</text>
      <text x="292" y="80" font-size="10" text-anchor="middle" fill="#64748b">loss_per_sample.backward()</text>
      <text x="292" y="96" font-size="10" text-anchor="middle" fill="#64748b">Opacus .grad_sample</text>

      <path d="M 375 77 L 400 77" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)"/>

      <rect x="405" y="40" width="170" height="75" rx="6" fill="#fee2e2" stroke="#f87171"/>
      <text x="490" y="62" font-size="11" font-weight="700" text-anchor="middle" fill="#b91c1c">3. Per-Sample Clip</text>
      <text x="490" y="80" font-size="10" text-anchor="middle" fill="#b91c1c">Bound norm ||g_i|| ≤ C</text>
      <text x="490" y="96" font-size="10" text-anchor="middle" fill="#b91c1c">Sensitivity Δ = C</text>

      <path d="M 575 77 L 600 77" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow-red)"/>

      <rect x="605" y="40" width="170" height="75" rx="6" fill="#fee2e2" stroke="#f87171"/>
      <text x="690" y="62" font-size="11" font-weight="700" text-anchor="middle" fill="#b91c1c">4. Add Noise &amp; Sum</text>
      <text x="690" y="80" font-size="10" text-anchor="middle" fill="#b91c1c">Noise ~ N(0, σ^2 C^2 I)</text>
      <text x="690" y="96" font-size="10" text-anchor="middle" fill="#b91c1c">p.grad = (sum+noise)/B</text>

      <path d="M 775 77 L 800 77" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow-red)"/>

      <rect x="805" y="40" width="140" height="75" rx="6" fill="#dcfce7" stroke="#86efac"/>
      <text x="875" y="62" font-size="11" font-weight="700" text-anchor="middle" fill="#166534">5. Step &amp; Log</text>
      <text x="875" y="80" font-size="10" text-anchor="middle" fill="#15803d">optimizer.step()</text>
      <text x="875" y="96" font-size="10" text-anchor="middle" fill="#15803d">accountant.record()</text>
    </g>
  </g>

  <!-- Connective Arrows across Containers -->
  <!-- CLI to Ingestion -->
  <path d="M 200 240 L 200 330" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>
  
  <!-- Raw Data to Profiler -->
  <path d="M 320 380 L 390 380" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Profiler to Preprocessing -->
  <path d="M 870 300 L 900 300" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Preprocessing to Schema Registry -->
  <path d="M 900 400 L 330 400" stroke="#b45309" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Preprocessing to Privacy Enclave -->
  <path d="M 1150 470 L 1150 500" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
  
  <!-- Privacy Enclave to Outputs -->
  <path d="M 390 780 L 320 780" stroke="#16a34a" stroke-width="2.5" marker-end="url(#arrow-green)"/>
</svg>
"""

# -------------------------------------------------------------------------------------------------
# 2. End-to-End Workflow / Process Diagram SVG
# -------------------------------------------------------------------------------------------------
svg_workflow = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 880" width="1500" height="880" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="wf-shadow" x="-5%" y="-5%" width="110%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#3b82f6"/>
    </marker>
    <marker id="loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1500" height="880" fill="#FFFFFF"/>

  <!-- Title Header -->
  <g transform="translate(50, 35)">
    <rect x="0" y="0" width="1400" height="55" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="25" y="34" font-size="19" font-weight="700" fill="#0f172a">ADVT End-to-End Execution Workflow</text>
    <text x="430" y="34" font-size="13" font-weight="500" fill="#64748b">Chronological Execution Lifecycle: Ingestion ➔ Profiling ➔ Preprocessing ➔ DP-SGD ➔ Reverse DDPM ➔ Evaluation</text>
  </g>

  <!-- Step 1: Initialize & Budget Check -->
  <g transform="translate(50, 120)">
    <rect x="0" y="0" width="250" height="150" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#3b82f6"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">1</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#1e3a8a">Session &amp; Budget Init</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">ComputeBudgetGuard</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Reads <tspan font-family="monospace">gpu_state.json</tspan></text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Verifies elapsed &lt; 30.0h</text>
    <text x="15" y="121" font-size="11" fill="#64748b">• Loads <tspan font-family="monospace">PipelineConfig</tspan></text>
    <rect x="15" y="132" width="220" height="10" rx="2" fill="#bfdbfe"/>
  </g>

  <!-- Arrow 1 -> 2 -->
  <path d="M 300 195 L 340 195" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 2: Dataset Profiling & HIPAA Scan -->
  <g transform="translate(340, 120)">
    <rect x="0" y="0" width="260" height="150" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#3b82f6"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">2</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#1e3a8a">Data Profiling &amp; HIPAA</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">DatasetProfiler.profile()</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Scans 18 HIPAA regex rules</text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Infers dtypes (cont, cat, ord)</text>
    <text x="15" y="121" font-size="11" fill="#64748b">• Detects structural missingness</text>
    <rect x="15" y="132" width="230" height="10" rx="2" fill="#bfdbfe"/>
  </g>

  <!-- Arrow 2 -> 3 -->
  <path d="M 600 195 L 640 195" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 3: Preprocessing & Missingness -->
  <g transform="translate(640, 120)">
    <rect x="0" y="0" width="260" height="150" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#3b82f6"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">3</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#1e3a8a">Transform &amp; Encode</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">PreprocessingPipeline</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Drops HIPAA / high-missing</text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Injects missingness flags</text>
    <text x="15" y="121" font-size="11" fill="#64748b">• Fits Scalers &amp; Encoders</text>
    <rect x="15" y="132" width="230" height="10" rx="2" fill="#bfdbfe"/>
  </g>

  <!-- Arrow 3 -> 4 -->
  <path d="M 900 195 L 940 195" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 4: Schema Registry Persistence -->
  <g transform="translate(940, 120)">
    <rect x="0" y="0" width="240" height="150" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#3b82f6"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">4</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#1e3a8a">Schema Persistence</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">FileSchemaRegistry.save()</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Writes <tspan font-family="monospace">profile.json</tspan></text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Writes <tspan font-family="monospace">pipeline_state.joblib</tspan></text>
    <text x="15" y="121" font-size="11" fill="#64748b">• Computes SHA-256 hash</text>
    <rect x="15" y="132" width="210" height="10" rx="2" fill="#bfdbfe"/>
  </g>

  <!-- Arrow 4 -> 5 -->
  <path d="M 1180 195 L 1220 195" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 5: Risk Tier Assignment -->
  <g transform="translate(1220, 120)">
    <rect x="0" y="0" width="230" height="150" rx="10" fill="#fef2f2" stroke="#fca5a5" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#dc2626"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">5</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#991b1b">Risk Tiering</text>
    <text x="15" y="65" font-size="12" fill="#7f1d1d"><tspan font-weight="600">HeuristicRiskTierAssigner</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Uniqueness heuristics</text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Correlation guard matrix</text>
    <text x="15" y="121" font-size="11" fill="#dc2626">• Groups correlated fields</text>
    <rect x="15" y="132" width="200" height="10" rx="2" fill="#fecaca"/>
  </g>

  <!-- Turn Downward Arrow from Step 5 to Step 6 -->
  <path d="M 1335 270 L 1335 340 L 1400 340 L 1400 480 L 1380 480" stroke="#dc2626" stroke-width="2.5" marker-end="url(#loop-arrow)"/>

  <!-- Step 6: DP-SGD Diffusion Training Loop (Center Large Box) -->
  <g transform="translate(250, 330)">
    <rect x="0" y="0" width="1120" height="300" rx="12" fill="#fff1f2" stroke="#f43f5e" stroke-width="2.5" filter="url(#wf-shadow)"/>
    <circle cx="35" cy="35" r="18" fill="#e11d48"/>
    <text x="35" y="42" font-size="16" font-weight="800" text-anchor="middle" fill="#FFFFFF">6</text>
    <text x="70" y="40" font-size="16" font-weight="800" fill="#881337">STEP 6: DP-SGD DIFFUSION MODEL TRAINING LOOP (DPTrainer)</text>
    
    <!-- Iterative Sub-Steps Inside Step 6 -->
    <!-- Mini Step 6.1 -->
    <g transform="translate(30, 70)">
      <rect x="0" y="0" width="230" height="190" rx="8" fill="#FFFFFF" stroke="#fecdd3" stroke-width="1.5"/>
      <text x="15" y="28" font-size="13" font-weight="700" fill="#9f1239">6.1 Forward Diffusion</text>
      <text x="15" y="50" font-size="11" fill="#475569">• Draw batch <tspan font-family="monospace">x_0 ~ D_train</tspan></text>
      <text x="15" y="70" font-size="11" fill="#475569">• Sample <tspan font-family="monospace">t ~ U(0, T-1)</tspan></text>
      <text x="15" y="90" font-size="11" fill="#475569">• Add Gaussian schedule noise</text>
      <text x="15" y="110" font-size="11" font-family="monospace" fill="#0f172a">  x_t = √(α)*x_0 + √(1-α)*ε</text>
      <text x="15" y="130" font-size="11" fill="#475569">• MLPDenoiser predicts ε_pred</text>
      <rect x="15" y="150" width="200" height="24" rx="4" fill="#ffe4e6"/>
      <text x="115" y="166" font-size="10" font-weight="600" text-anchor="middle" fill="#be123c">MSELoss(ε_pred, ε)</text>
    </g>

    <!-- Arrow 6.1 -> 6.2 -->
    <path d="M 260 165 L 290 165" stroke="#f43f5e" stroke-width="2" marker-end="url(#loop-arrow)"/>

    <!-- Mini Step 6.2 -->
    <g transform="translate(290, 70)">
      <rect x="0" y="0" width="240" height="190" rx="8" fill="#FFFFFF" stroke="#fecdd3" stroke-width="1.5"/>
      <text x="15" y="28" font-size="13" font-weight="700" fill="#9f1239">6.2 Per-Sample Backprop</text>
      <text x="15" y="50" font-size="11" fill="#475569">• Wrapped in <tspan font-weight="600">GradSampleModule</tspan></text>
      <text x="15" y="70" font-size="11" fill="#475569">• Executes backward pass</text>
      <text x="15" y="90" font-size="11" fill="#475569">• Opacus hooks populate:</text>
      <text x="15" y="110" font-size="11" font-family="monospace" fill="#be123c">  p.grad_sample (B, ...)</text>
      <text x="15" y="130" font-size="11" fill="#475569">• Avoids global sum leakage</text>
      <rect x="15" y="150" width="210" height="24" rx="4" fill="#ffe4e6"/>
      <text x="120" y="166" font-size="10" font-weight="600" text-anchor="middle" fill="#be123c">Individual Gradient Capture</text>
    </g>

    <!-- Arrow 6.2 -> 6.3 -->
    <path d="M 530 165 L 560 165" stroke="#f43f5e" stroke-width="2" marker-end="url(#loop-arrow)"/>

    <!-- Mini Step 6.3 -->
    <g transform="translate(560, 70)">
      <rect x="0" y="0" width="250" height="190" rx="8" fill="#FFFFFF" stroke="#fecdd3" stroke-width="1.5"/>
      <text x="15" y="28" font-size="13" font-weight="700" fill="#9f1239">6.3 Clip &amp; Add Noise</text>
      <text x="15" y="50" font-size="11" fill="#475569">• Per-sample norm: <tspan font-family="monospace">||g_i||_2</tspan></text>
      <text x="15" y="70" font-size="11" fill="#475569">• Clip: <tspan font-family="monospace">min(1, C / ||g_i||_2)</tspan></text>
      <text x="15" y="90" font-size="11" fill="#475569">• Sum clipped grads over batch</text>
      <text x="15" y="110" font-size="11" fill="#475569">• Add Noise: <tspan font-family="monospace">N(0, σ^2 C^2 I)</tspan></text>
      <text x="15" y="130" font-size="11" fill="#475569">• Average by batch size: <tspan font-family="monospace">/ B</tspan></text>
      <rect x="15" y="150" width="220" height="24" rx="4" fill="#ffe4e6"/>
      <text x="125" y="166" font-size="10" font-weight="600" text-anchor="middle" fill="#be123c">Guaranteed DP Perturbation</text>
    </g>

    <!-- Arrow 6.3 -> 6.4 -->
    <path d="M 810 165 L 840 165" stroke="#f43f5e" stroke-width="2" marker-end="url(#loop-arrow)"/>

    <!-- Mini Step 6.4 -->
    <g transform="translate(840, 70)">
      <rect x="0" y="0" width="250" height="190" rx="8" fill="#FFFFFF" stroke="#fecdd3" stroke-width="1.5"/>
      <text x="15" y="28" font-size="13" font-weight="700" fill="#9f1239">6.4 Step &amp; Privacy Accounting</text>
      <text x="15" y="50" font-size="11" fill="#475569">• <tspan font-weight="600">optimizer.step()</tspan> updates</text>
      <text x="15" y="70" font-size="11" fill="#475569">• <tspan font-weight="600">accountant.record_step()</tspan></text>
      <text x="25" y="88" font-size="11" fill="#475569">  - noise_multiplier = σ(t)</text>
      <text x="25" y="106" font-size="11" fill="#475569">  - sample_rate = B / N</text>
      <text x="15" y="128" font-size="11" fill="#475569">• ComputeBudgetGuard check</text>
      <rect x="15" y="150" width="220" height="24" rx="4" fill="#dcfce7"/>
      <text x="125" y="166" font-size="10" font-weight="600" text-anchor="middle" fill="#166534">Epoch Iteration Complete</text>
    </g>
  </g>

  <!-- Left Arrow from Step 6 Down to Step 7 -->
  <path d="M 250 480 L 180 480 L 180 670 L 180 700" stroke="#16a34a" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 7: Reverse DDPM Sampling -->
  <g transform="translate(50, 700)">
    <rect x="0" y="0" width="300" height="150" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#16a34a"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">7</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#14532d">Reverse DDPM Sampling</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">generate_samples()</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Sample pure Gaussian noise <tspan font-family="monospace">x_T</tspan></text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Iterative denoising T-1 down to 0</text>
    <text x="15" y="121" font-size="11" fill="#64748b">• Yields synthetic tensor <tspan font-family="monospace">(N_gen, D)</tspan></text>
    <rect x="15" y="132" width="270" height="10" rx="2" fill="#bbf7d0"/>
  </g>

  <!-- Arrow 7 -> 8 -->
  <path d="M 350 775 L 430 775" stroke="#16a34a" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 8: Inverse Decoding -->
  <g transform="translate(430, 700)">
    <rect x="0" y="0" width="340" height="150" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#16a34a"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">8</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#14532d">Inverse Pipeline Decoding</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">pipeline.inverse_transform()</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Scaler inverse transforms continuous</text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Argmax decoding for One-Hot categorical</text>
    <text x="15" y="121" font-size="11" fill="#64748b">• MissingnessHandler restores NaNs</text>
    <rect x="15" y="132" width="310" height="10" rx="2" fill="#bbf7d0"/>
  </g>

  <!-- Arrow 8 -> 9 -->
  <path d="M 770 775 L 850 775" stroke="#16a34a" stroke-width="2.5" marker-end="url(#wf-arrow)"/>

  <!-- Step 9: Evaluation & Report Generation -->
  <g transform="translate(850, 700)">
    <rect x="0" y="0" width="600" height="150" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="2" filter="url(#wf-shadow)"/>
    <circle cx="30" cy="30" r="16" fill="#16a34a"/>
    <text x="30" y="36" font-size="14" font-weight="800" text-anchor="middle" fill="#FFFFFF">9</text>
    <text x="60" y="35" font-size="14" font-weight="700" fill="#14532d">Quality &amp; Privacy Evaluation (Reports)</text>
    <text x="15" y="65" font-size="12" fill="#334155"><tspan font-weight="600">UtilityEvaluator &amp; PrivacyEvaluator</tspan></text>
    <text x="15" y="85" font-size="11" fill="#64748b">• Distance-Based Membership Inference Attack (<tspan font-weight="600">D-MIA</tspan> score &lt; 0.5)</text>
    <text x="15" y="103" font-size="11" fill="#64748b">• Univariate Fidelity (<tspan font-weight="600">KS-test</tspan> &amp; <tspan font-weight="600">TVD</tspan>) &amp; Bivariate Correlation RMSE</text>
    <text x="15" y="121" font-size="11" fill="#15803d">• Saves <tspan font-family="monospace">synthetic_eps_{target}.csv</tspan> &amp; <tspan font-family="monospace">sweep_report.json</tspan></text>
    <rect x="15" y="132" width="570" height="10" rx="2" fill="#bbf7d0"/>
  </g>
</svg>
"""

# -------------------------------------------------------------------------------------------------
# 3. Data Flow Diagram (DFD) SVG
# -------------------------------------------------------------------------------------------------
svg_dfd = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 920" width="1450" height="920" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="dfd-shadow" x="-5%" y="-5%" width="110%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <marker id="dfd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
    </marker>
    <marker id="dfd-arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1450" height="920" fill="#FFFFFF"/>

  <!-- Title Header -->
  <g transform="translate(50, 35)">
    <rect x="0" y="0" width="1350" height="55" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="25" y="34" font-size="19" font-weight="700" fill="#0f172a">ADVT Sensitive Data Flow Diagram (DFD)</text>
    <text x="440" y="34" font-size="13" font-weight="500" fill="#64748b">Tracking PHI / Clinical Data Ingestion, DP Sensitivity Boundary Enclave, &amp; Synthetic Release</text>
  </g>

  <!-- Data Stores Row (Top) -->
  <g transform="translate(50, 115)">
    <!-- DS1: Raw Data Store -->
    <g transform="translate(0, 0)">
      <rect x="0" y="0" width="240" height="75" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
      <path d="M 15 0 L 15 75 M 225 0 L 225 75" stroke="#fca5a5" stroke-width="1.5"/>
      <text x="120" y="28" font-size="12" font-weight="700" text-anchor="middle" fill="#991b1b">DS1: RAW CLINICAL DB</text>
      <text x="120" y="48" font-size="11" text-anchor="middle" fill="#b91c1c">Contains Unsanitized PHI</text>
      <text x="120" y="64" font-size="10" text-anchor="middle" font-family="monospace" fill="#475569">data/*.zip, data/*.csv</text>
    </g>

    <!-- DS2: Schema Registry -->
    <g transform="translate(280, 0)">
      <rect x="0" y="0" width="240" height="75" rx="4" fill="#f0f9ff" stroke="#0284c7" stroke-width="1.5"/>
      <path d="M 15 0 L 15 75 M 225 0 L 225 75" stroke="#bae6fd" stroke-width="1.5"/>
      <text x="120" y="28" font-size="12" font-weight="700" text-anchor="middle" fill="#0369a1">DS2: SCHEMA REGISTRY</text>
      <text x="120" y="48" font-size="11" text-anchor="middle" fill="#0284c7">Fitted Encoders / Scalers</text>
      <text x="120" y="64" font-size="10" text-anchor="middle" font-family="monospace" fill="#475569">registry/v{N}/</text>
    </g>

    <!-- DS3: GPU State -->
    <g transform="translate(560, 0)">
      <rect x="0" y="0" width="240" height="75" rx="4" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
      <path d="M 15 0 L 15 75 M 225 0 L 225 75" stroke="#cbd5e1" stroke-width="1.5"/>
      <text x="120" y="28" font-size="12" font-weight="700" text-anchor="middle" fill="#334155">DS3: BUDGET STATE</text>
      <text x="120" y="48" font-size="11" text-anchor="middle" fill="#475569">Wall-Clock Elapsed Time</text>
      <text x="120" y="64" font-size="10" text-anchor="middle" font-family="monospace" fill="#64748b">gpu_state.json</text>
    </g>

    <!-- DS4: Model Checkpoints -->
    <g transform="translate(840, 0)">
      <rect x="0" y="0" width="240" height="75" rx="4" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.5"/>
      <path d="M 15 0 L 15 75 M 225 0 L 225 75" stroke="#e9d5ff" stroke-width="1.5"/>
      <text x="120" y="28" font-size="12" font-weight="700" text-anchor="middle" fill="#6b21a8">DS4: MODEL WEIGHTS</text>
      <text x="120" y="48" font-size="11" text-anchor="middle" fill="#9333ea">Denoising Net Parameters</text>
      <text x="120" y="64" font-size="10" text-anchor="middle" font-family="monospace" fill="#64748b">checkpoints/model.pt</text>
    </g>

    <!-- DS5: Synthetic Output Store -->
    <g transform="translate(1110, 0)">
      <rect x="0" y="0" width="240" height="75" rx="4" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
      <path d="M 15 0 L 15 75 M 225 0 L 225 75" stroke="#bbf7d0" stroke-width="1.5"/>
      <text x="120" y="28" font-size="12" font-weight="700" text-anchor="middle" fill="#15803d">DS5: SYNTHETIC DATA &amp; EVAL</text>
      <text x="120" y="48" font-size="11" text-anchor="middle" fill="#16a34a">Differentially Private CSVs</text>
      <text x="120" y="64" font-size="10" text-anchor="middle" font-family="monospace" fill="#15803d">outputs/sweep_results/</text>
    </g>
  </g>

  <!-- Process P1: Ingestion & Profiling -->
  <g transform="translate(50, 240)">
    <rect x="0" y="0" width="280" height="130" rx="10" fill="#FFFFFF" stroke="#0284c7" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#0284c7"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P1</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#0369a1">Ingestion &amp; HIPAA Profiling</text>
    <text x="15" y="55" font-size="11" fill="#475569">• Reads Raw CSV</text>
    <text x="15" y="73" font-size="11" fill="#dc2626">• Scans 18 HIPAA patterns (drops PHI)</text>
    <text x="15" y="91" font-size="11" fill="#475569">• Infers statistical dtypes</text>
    <text x="15" y="109" font-size="11" fill="#16a34a">Outputs: Sanitized DataFrame</text>
  </g>

  <!-- Process P2: Preprocessing & Missingness -->
  <g transform="translate(390, 240)">
    <rect x="0" y="0" width="280" height="130" rx="10" fill="#FFFFFF" stroke="#0284c7" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#0284c7"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P2</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#0369a1">Missingness &amp; Tensor Encoding</text>
    <text x="15" y="55" font-size="11" fill="#475569">• Injects binary missingness flags</text>
    <text x="15" y="73" font-size="11" fill="#475569">• One-Hot &amp; Frequency encoding</text>
    <text x="15" y="91" font-size="11" fill="#475569">• Standard / Robust scaling</text>
    <text x="15" y="109" font-size="11" font-weight="600" fill="#0284c7">Outputs: Normalized Tensor x_0</text>
  </g>

  <!-- Process P3: Risk Tier Assigner -->
  <g transform="translate(730, 240)">
    <rect x="0" y="0" width="280" height="130" rx="10" fill="#FFFFFF" stroke="#dc2626" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#dc2626"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P3</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#991b1b">Risk Tier Assignment</text>
    <text x="15" y="55" font-size="11" fill="#475569">• Evaluates uniqueness ratios</text>
    <text x="15" y="73" font-size="11" fill="#475569">• Builds correlation matrix</text>
    <text x="15" y="91" font-size="11" fill="#dc2626">• Enforces correlation leakage guard</text>
    <text x="15" y="109" font-size="11" font-weight="600" fill="#991b1b">Outputs: Feature Tier Norms C_tier</text>
  </g>

  <!-- The Protected Privacy Enclave Container -->
  <g transform="translate(50, 420)">
    <rect x="0" y="0" width="1350" height="230" rx="12" fill="#fff5f5" stroke="#ef4444" stroke-width="2" stroke-dasharray="6 4"/>
    <text x="25" y="28" font-size="13" font-weight="800" fill="#b91c1c">🔒 DIFFERENTIAL PRIVACY BOUNDARY / SENSITIVITY ENCLAVE (Opacus DP-SGD)</text>
    <text x="750" y="28" font-size="11" font-weight="600" fill="#dc2626">MATHEMATICAL GUARANTEE: Individual patient records cannot be inferred beyond budget (ε, δ)</text>

    <!-- Process P4: Forward Diffusion -->
    <g transform="translate(20, 50)">
      <rect x="0" y="0" width="290" height="155" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#dfd-shadow)"/>
      <circle cx="25" cy="25" r="14" fill="#dc2626"/>
      <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P4</text>
      <text x="48" y="29" font-size="13" font-weight="700" fill="#991b1b">Closed-Form Forward Diffuse</text>
      <text x="15" y="55" font-size="11" fill="#475569">• Uniformly draws timestep <tspan font-family="monospace">t</tspan></text>
      <text x="15" y="73" font-size="11" fill="#475569">• Adds schedule noise to <tspan font-family="monospace">x_0</tspan></text>
      <text x="15" y="91" font-size="11" font-family="monospace" fill="#0f172a">  x_t = √(α)*x_0 + √(1-α)*ε</text>
      <text x="15" y="111" font-size="11" fill="#475569">• MLPDenoiser predicts noise</text>
      <text x="15" y="131" font-size="11" font-weight="600" fill="#b91c1c">Outputs: Per-Sample Loss</text>
    </g>

    <!-- Arrow P4 -> P5a -->
    <path d="M 310 125 L 345 125" stroke="#dc2626" stroke-width="2" marker-end="url(#dfd-arrow-red)"/>

    <!-- Process P5a: Per-Sample Clipping -->
    <g transform="translate(350, 50)">
      <rect x="0" y="0" width="310" height="155" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#dfd-shadow)"/>
      <circle cx="25" cy="25" r="14" fill="#dc2626"/>
      <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P5.1</text>
      <text x="48" y="29" font-size="13" font-weight="700" fill="#991b1b">Per-Sample Gradient Clip</text>
      <text x="15" y="55" font-size="11" fill="#475569">• Opacus extracts individual <tspan font-family="monospace">g_i</tspan></text>
      <text x="15" y="73" font-size="11" fill="#475569">• Computes per-sample <tspan font-family="monospace">||g_i||_2</tspan></text>
      <text x="15" y="91" font-size="11" fill="#dc2626">• Clips: <tspan font-family="monospace">g_bar_i = g_i * min(1, C/||g_i||)</tspan></text>
      <text x="15" y="111" font-size="11" fill="#475569">• Sensitivity strictly bounded by <tspan font-weight="700">C</tspan></text>
      <text x="15" y="131" font-size="11" font-weight="600" fill="#b91c1c">Outputs: Clipped Gradients g_bar_i</text>
    </g>

    <!-- Arrow P5a -> P5b -->
    <path d="M 660 125 L 695 125" stroke="#dc2626" stroke-width="2" marker-end="url(#dfd-arrow-red)"/>

    <!-- Process P5b: Noise Injection & DP-SGD Step -->
    <g transform="translate(700, 50)">
      <rect x="0" y="0" width="310" height="155" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#dfd-shadow)"/>
      <circle cx="25" cy="25" r="14" fill="#dc2626"/>
      <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P5.2</text>
      <text x="48" y="29" font-size="13" font-weight="700" fill="#991b1b">Gaussian Noise &amp; Update</text>
      <text x="15" y="55" font-size="11" fill="#475569">• Sums clipped gradients over batch</text>
      <text x="15" y="73" font-size="11" fill="#dc2626">• Injects Noise: <tspan font-family="monospace">N(0, σ^2 C^2 I)</tspan></text>
      <text x="15" y="91" font-size="11" fill="#475569">• Assigns batch grad: <tspan font-family="monospace">(sum + noise)/B</tspan></text>
      <text x="15" y="111" font-size="11" fill="#475569">• Optimizer updates model weights</text>
      <text x="15" y="131" font-size="11" font-weight="600" fill="#16a34a">Outputs: Private Model Weights</text>
    </g>

    <!-- Arrow P5b -> P5c -->
    <path d="M 1010 125 L 1045 125" stroke="#dc2626" stroke-width="2" marker-end="url(#dfd-arrow-red)"/>

    <!-- Process P5c: Central RDP Accounting -->
    <g transform="translate(1050, 50)">
      <rect x="0" y="0" width="280" height="155" rx="8" fill="#FFFFFF" stroke="#fca5a5" stroke-width="1.5" filter="url(#dfd-shadow)"/>
      <circle cx="25" cy="25" r="14" fill="#dc2626"/>
      <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P5.3</text>
      <text x="48" y="29" font-size="13" font-weight="700" fill="#991b1b">RDP Privacy Accounting</text>
      <text x="15" y="55" font-size="11" fill="#475569">• Single centralized accountant</text>
      <text x="15" y="73" font-size="11" fill="#475569">• Logs event: <tspan font-family="monospace">(σ_t, q = B/N)</tspan></text>
      <text x="15" y="91" font-size="11" fill="#475569">• Accumulates Rényi divergences</text>
      <text x="15" y="111" font-size="11" fill="#475569">• Computes exact spent <tspan font-family="monospace">ε(δ)</tspan></text>
      <text x="15" y="131" font-size="11" font-weight="600" fill="#15803d">Outputs: Epsilon Spent Report</text>
    </g>
  </g>

  <!-- Post-Privacy Release Processes (Bottom Row) -->
  <!-- Process P6: DDPM Reverse Sampling -->
  <g transform="translate(150, 680)">
    <rect x="0" y="0" width="340" height="130" rx="10" fill="#FFFFFF" stroke="#16a34a" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#16a34a"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P6</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#15803d">Reverse Diffusion Sampling</text>
    <text x="15" y="55" font-size="11" fill="#475569">• Starts from Gaussian noise tensor <tspan font-family="monospace">x_T</tspan></text>
    <text x="15" y="73" font-size="11" fill="#475569">• Iteratively predicts noise with DP weights</text>
    <text x="15" y="91" font-size="11" fill="#475569">• Generates synthetic tensor <tspan font-family="monospace">(N_gen, D)</tspan></text>
    <text x="15" y="109" font-size="11" font-weight="600" fill="#15803d">Outputs: Synthetic Latent Tensor</text>
  </g>

  <!-- Process P7: Inverse Pipeline Decoding -->
  <g transform="translate(560, 680)">
    <rect x="0" y="0" width="340" height="130" rx="10" fill="#FFFFFF" stroke="#16a34a" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#16a34a"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P7</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#15803d">Inverse Pipeline Decoding</text>
    <text x="15" y="55" font-size="11" fill="#475569">• Fetches fitted scalers &amp; encoders from DS2</text>
    <text x="15" y="73" font-size="11" fill="#475569">• Inverts continuous features to original scale</text>
    <text x="15" y="91" font-size="11" fill="#475569">• Decodes categorical &amp; restores missing NaNs</text>
    <text x="15" y="109" font-size="11" font-weight="600" fill="#15803d">Outputs: Synthetic Clinical DataFrame</text>
  </g>

  <!-- Process P8: Evaluation & Quality Assurance -->
  <g transform="translate(970, 680)">
    <rect x="0" y="0" width="430" height="130" rx="10" fill="#FFFFFF" stroke="#16a34a" stroke-width="2" filter="url(#dfd-shadow)"/>
    <circle cx="25" cy="25" r="14" fill="#16a34a"/>
    <text x="25" y="30" font-size="12" font-weight="800" text-anchor="middle" fill="#FFFFFF">P8</text>
    <text x="50" y="29" font-size="13" font-weight="700" fill="#15803d">Statistical &amp; Empirical Privacy Evaluation</text>
    <text x="15" y="55" font-size="11" fill="#475569">• <tspan font-weight="600">PrivacyEvaluator</tspan>: Distance-Based MIA risk score</text>
    <text x="15" y="73" font-size="11" fill="#475569">• <tspan font-weight="600">UtilityEvaluator</tspan>: KS-test (cont), TVD (cat), Corr RMSE</text>
    <text x="15" y="91" font-size="11" fill="#475569">• Assesses epsilon vs utility Pareto frontier</text>
    <text x="15" y="109" font-size="11" font-weight="600" fill="#15803d">Outputs: sweep_report.json &amp; Metrics</text>
  </g>

  <!-- Inter-Process Flow Connectors -->
  <!-- DS1 -> P1 -->
  <path d="M 120 190 L 120 240" stroke="#475569" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  
  <!-- P1 -> P2 -->
  <path d="M 330 305 L 390 305" stroke="#475569" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  
  <!-- P2 -> DS2 -->
  <path d="M 460 240 L 460 190" stroke="#0284c7" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  
  <!-- P1 -> P3 -->
  <path d="M 330 350 L 730 350" stroke="#475569" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  
  <!-- P2 -> P4 -->
  <path d="M 530 370 L 530 400 L 165 400 L 165 470" stroke="#475569" stroke-width="2" marker-end="url(#dfd-arrow)"/>

  <!-- P3 -> P5.1 -->
  <path d="M 870 370 L 870 400 L 505 400 L 505 470" stroke="#dc2626" stroke-width="2" marker-end="url(#dfd-arrow-red)"/>

  <!-- P5.2 -> DS4 -->
  <path d="M 855 470 L 855 400 L 960 400 L 960 190" stroke="#a855f7" stroke-width="2" marker-end="url(#dfd-arrow)"/>

  <!-- DS4 -> P6 -->
  <path d="M 960 190 L 960 210 L 1420 210 L 1420 660 L 320 660 L 320 680" stroke="#a855f7" stroke-width="1.5" marker-end="url(#dfd-arrow)"/>

  <!-- P6 -> P7 -->
  <path d="M 490 745 L 560 745" stroke="#16a34a" stroke-width="2" marker-end="url(#dfd-arrow)"/>

  <!-- DS2 -> P7 -->
  <path d="M 400 190 L 400 210 L 370 210 L 370 670 L 730 670 L 730 680" stroke="#0284c7" stroke-width="1.5" marker-end="url(#dfd-arrow)"/>

  <!-- P7 -> P8 & DS5 -->
  <path d="M 900 745 L 970 745" stroke="#16a34a" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  <path d="M 730 810 L 730 850 L 1230 850 L 1230 190" stroke="#16a34a" stroke-width="2" marker-end="url(#dfd-arrow)"/>
  
  <!-- P8 -> DS5 -->
  <path d="M 1230 680 L 1230 190" stroke="#16a34a" stroke-width="2" marker-end="url(#dfd-arrow)"/>
</svg>
"""

# -------------------------------------------------------------------------------------------------
# 4. Core UML Class Diagram SVG
# -------------------------------------------------------------------------------------------------
svg_class_diagram = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1650 1150" width="1650" height="1150" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="uml-shadow" x="-5%" y="-5%" width="110%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <!-- UML Markers -->
    <marker id="inheritance" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <polygon points="0 0, 11 6, 0 12" fill="#FFFFFF" stroke="#0f172a" stroke-width="1.5"/>
    </marker>
    <marker id="composition" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <polygon points="0 6, 6 0, 12 6, 6 12" fill="#0f172a" stroke="#0f172a" stroke-width="1.5"/>
    </marker>
    <marker id="aggregation" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <polygon points="0 6, 6 0, 12 6, 6 12" fill="#FFFFFF" stroke="#0f172a" stroke-width="1.5"/>
    </marker>
    <marker id="dependency" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9" fill="none" stroke="#64748b" stroke-width="1.5"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1650" height="1150" fill="#FFFFFF"/>

  <!-- Title Header -->
  <g transform="translate(50, 30)">
    <rect x="0" y="0" width="1550" height="50" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="25" y="32" font-size="18" font-weight="700" fill="#0f172a">ADVT Object-Oriented Architecture &amp; Core Class Diagram (UML)</text>
    <text x="630" y="32" font-size="13" font-weight="500" fill="#64748b">Inheritance, Protocol Realization, Composition &amp; Dependency Mappings</text>
  </g>

  <!-- ==================== COLUMN 1: PREPROCESSING SUBSYSTEM ==================== -->
  
  <!-- Interface: AbstractScaler -->
  <g transform="translate(50, 105)">
    <rect x="0" y="0" width="240" height="100" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="120" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractScaler</text>
    <path d="M 0 45 L 240 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+fit(series: Series)*</text>
    <text x="10" y="77" font-size="10" font-family="monospace" fill="#334155">+transform(series: Series)*</text>
    <text x="10" y="92" font-size="10" font-family="monospace" fill="#334155">+inverse_transform(arr: ndarray)*</text>
  </g>

  <!-- Concrete Scalers -->
  <!-- StandardScaler -->
  <g transform="translate(50, 240)">
    <rect x="0" y="0" width="240" height="85" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">StandardScaler</text>
    <path d="M 0 30 L 240 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_mean: float, -_std: float</text>
    <path d="M 0 54 L 240 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+fit(), +transform(), +inverse()</text>
  </g>

  <!-- RobustScaler -->
  <g transform="translate(50, 345)">
    <rect x="0" y="0" width="240" height="85" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">RobustScaler</text>
    <path d="M 0 30 L 240 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_median: float, -_iqr: float</text>
    <path d="M 0 54 L 240 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+fit(), +transform(), +inverse()</text>
  </g>

  <!-- Inheritance arrows to AbstractScaler -->
  <path d="M 170 240 L 170 205" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>
  <path d="M 170 345 L 170 325 L 210 325 L 210 205" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- Interface: AbstractEncoder -->
  <g transform="translate(50, 460)">
    <rect x="0" y="0" width="240" height="110" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="120" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractEncoder</text>
    <path d="M 0 45 L 240 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+fit(series: Series)*</text>
    <text x="10" y="77" font-size="10" font-family="monospace" fill="#334155">+transform(series: Series)*</text>
    <text x="10" y="92" font-size="10" font-family="monospace" fill="#334155">+inverse_transform(arr)*</text>
    <text x="10" y="106" font-size="10" font-family="monospace" fill="#334155">+output_dim: int*</text>
  </g>

  <!-- OneHotEncoder -->
  <g transform="translate(50, 600)">
    <rect x="0" y="0" width="240" height="95" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">OneHotEncoder</text>
    <path d="M 0 30 L 240 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_min_freq: int, -_vocab: list</text>
    <path d="M 0 54 L 240 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+fit(), +transform()</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">+inverse_transform(), +output_dim</text>
  </g>

  <!-- FrequencyEncoder -->
  <g transform="translate(50, 715)">
    <rect x="0" y="0" width="240" height="95" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="120" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">FrequencyEncoder</text>
    <path d="M 0 30 L 240 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_min_freq: int, -_vocab: list</text>
    <path d="M 0 54 L 240 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+fit(), +transform()</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">+inverse_transform(), +vocab_size</text>
  </g>

  <!-- Inheritance arrows to AbstractEncoder -->
  <path d="M 170 600 L 170 570" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>
  <path d="M 170 715 L 170 695 L 210 695 L 210 570" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- ==================== COLUMN 2: PIPELINE & MISSINGNESS ==================== -->
  
  <!-- Interface: AbstractMissingnessHandler -->
  <g transform="translate(340, 105)">
    <rect x="0" y="0" width="260" height="100" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="130" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractMissingnessHandler</text>
    <path d="M 0 45 L 260 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+fit(df: DataFrame)*</text>
    <text x="10" y="77" font-size="10" font-family="monospace" fill="#334155">+transform(df: DataFrame)*</text>
    <text x="10" y="92" font-size="10" font-family="monospace" fill="#334155">+inverse_transform(df)*</text>
  </g>

  <!-- Concrete MissingnessHandler -->
  <g transform="translate(340, 240)">
    <rect x="0" y="0" width="260" height="120" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">MissingnessHandler</text>
    <path d="M 0 30 L 260 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_config: PipelineConfig</text>
    <text x="10" y="60" font-size="10" font-family="monospace" fill="#64748b">-_indicator_columns: list[str]</text>
    <text x="10" y="74" font-size="10" font-family="monospace" fill="#64748b">-_imputation_values: dict[str, Any]</text>
    <path d="M 0 82 L 260 82" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="98" font-size="10" font-family="monospace" fill="#334155">+fit(), +transform()</text>
    <text x="10" y="112" font-size="10" font-family="monospace" fill="#334155">+inverse_transform()</text>
  </g>

  <path d="M 470 240 L 470 205" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- PreprocessingPipeline Core Class -->
  <g transform="translate(340, 420)">
    <rect x="0" y="0" width="310" height="230" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="2" filter="url(#uml-shadow)"/>
    <text x="155" y="24" font-size="14" font-weight="700" text-anchor="middle" fill="#1e3a8a">PreprocessingPipeline</text>
    <path d="M 0 32 L 310 32" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="10" y="48" font-size="10" font-family="monospace" fill="#1e40af">-_config: PipelineConfig</text>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#1e40af">-_profiler: AbstractProfiler</text>
    <text x="10" y="76" font-size="10" font-family="monospace" fill="#1e40af">-_missingness_handler: AbstractHandler</text>
    <text x="10" y="90" font-size="10" font-family="monospace" fill="#1e40af">-_encoder_factory: Callable</text>
    <text x="10" y="104" font-size="10" font-family="monospace" fill="#1e40af">-_scaler_factory: Callable</text>
    <text x="10" y="118" font-size="10" font-family="monospace" fill="#1e40af">-_registry: AbstractSchemaRegistry</text>
    <text x="10" y="132" font-size="10" font-family="monospace" fill="#1e40af">-_scalers: dict[str, AbstractScaler]</text>
    <text x="10" y="146" font-size="10" font-family="monospace" fill="#1e40af">-_encoders: dict[str, AbstractEncoder]</text>
    <path d="M 0 154 L 310 154" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="10" y="170" font-size="10" font-weight="600" font-family="monospace" fill="#1e3a8a">+fit_transform(df, name): ndarray</text>
    <text x="10" y="185" font-size="10" font-weight="600" font-family="monospace" fill="#1e3a8a">+transform(df): ndarray</text>
    <text x="10" y="200" font-size="10" font-weight="600" font-family="monospace" fill="#1e3a8a">+inverse_transform(arr): DataFrame</text>
    <text x="10" y="215" font-size="10" font-family="monospace" fill="#1e3a8a">+get_profile(): DatasetProfile</text>
  </g>

  <!-- Composition & Aggregation connectors to Pipeline -->
  <path d="M 470 420 L 470 360" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>
  <path d="M 340 500 L 290 500" stroke="#0f172a" stroke-width="1.5" marker-start="url(#aggregation)"/>
  <path d="M 340 450 L 290 450 L 290 155 L 290 155" stroke="#0f172a" stroke-width="1.5" marker-start="url(#aggregation)"/>

  <!-- ==================== COLUMN 3: PROFILING & REGISTRY ==================== -->
  
  <!-- Interface: AbstractProfiler -->
  <g transform="translate(680, 105)">
    <rect x="0" y="0" width="260" height="85" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="130" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractProfiler</text>
    <path d="M 0 45 L 260 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="65" font-size="10" font-family="monospace" fill="#334155">+profile(df, name)*: DatasetProfile</text>
  </g>

  <!-- DatasetProfiler -->
  <g transform="translate(680, 220)">
    <rect x="0" y="0" width="260" height="110" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">DatasetProfiler</text>
    <path d="M 0 30 L 260 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_config: PipelineConfig</text>
    <path d="M 0 54 L 260 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+profile(df, name): DatasetProfile</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">-_profile_column(series, df)</text>
    <text x="10" y="100" font-size="10" font-family="monospace" fill="#334155">+check_hipaa_identifier(col)</text>
  </g>

  <path d="M 810 220 L 810 190" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>
  <path d="M 650 450 L 680 450 L 680 275" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>

  <!-- Interface: AbstractSchemaRegistry -->
  <g transform="translate(680, 360)">
    <rect x="0" y="0" width="260" height="120" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="130" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractSchemaRegistry</text>
    <path d="M 0 45 L 260 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+save(dataset_name, ...)*: int</text>
    <text x="10" y="77" font-size="10" font-family="monospace" fill="#334155">+load(dataset_name, ver)*: Entry</text>
    <text x="10" y="92" font-size="10" font-family="monospace" fill="#334155">+load_profile(name, ver)*: Profile</text>
    <text x="10" y="107" font-size="10" font-family="monospace" fill="#334155">+list_versions(name)*: list[int]</text>
  </g>

  <!-- FileSchemaRegistry -->
  <g transform="translate(680, 515)">
    <rect x="0" y="0" width="260" height="110" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="130" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">FileSchemaRegistry</text>
    <path d="M 0 30 L 260 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_root: Path</text>
    <path d="M 0 54 L 260 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+save(), +load(), +load_profile()</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">+list_datasets(), +list_versions()</text>
    <text x="10" y="100" font-size="10" font-family="monospace" fill="#334155">+delete(name, version)</text>
  </g>

  <path d="M 810 515 L 810 480" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>
  <path d="M 650 535 L 680 535" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>

  <!-- ==================== COLUMN 4: DIFFUSION & DP-TRAINER ==================== -->
  
  <!-- Interface: AbstractDenoiser -->
  <g transform="translate(980, 105)">
    <rect x="0" y="0" width="280" height="85" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="140" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="140" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractDenoiser</text>
    <path d="M 0 45 L 280 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+forward(x: Tensor, t: Tensor)*: Tensor</text>
    <text x="10" y="77" font-size="10" font-family="monospace" fill="#334155">+input_dim: int*</text>
  </g>

  <!-- MLPDenoiser -->
  <g transform="translate(980, 220)">
    <rect x="0" y="0" width="280" height="130" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="140" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">MLPDenoiser (nn.Module)</text>
    <path d="M 0 30 L 280 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_input_dim: int, -_num_timesteps: int</text>
    <text x="10" y="60" font-size="10" font-family="monospace" fill="#64748b">+time_embed: nn.Sequential</text>
    <text x="10" y="74" font-size="10" font-family="monospace" fill="#64748b">+net: nn.ModuleList, +out_layer: Linear</text>
    <path d="M 0 82 L 280 82" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="98" font-size="10" font-family="monospace" fill="#334155">+forward(x: Tensor, t: Tensor): Tensor</text>
    <text x="10" y="112" font-size="10" font-family="monospace" fill="#334155">+input_dim: int, +num_timesteps: int</text>
  </g>

  <path d="M 1120 220 L 1120 190" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- Interface: AbstractNoiseSchedule -->
  <g transform="translate(980, 375)">
    <rect x="0" y="0" width="280" height="95" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="140" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="140" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractNoiseSchedule</text>
    <path d="M 0 45 L 280 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#334155">+get_betas()*: Tensor</text>
    <text x="10" y="75" font-size="10" font-family="monospace" fill="#334155">+get_alphas()*: Tensor</text>
    <text x="10" y="88" font-size="10" font-family="monospace" fill="#334155">+get_alphas_cumprod()*: Tensor</text>
  </g>

  <!-- LinearNoiseSchedule -->
  <g transform="translate(980, 500)">
    <rect x="0" y="0" width="280" height="95" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="140" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">LinearNoiseSchedule</text>
    <path d="M 0 30 L 280 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_betas: Tensor, -_alphas: Tensor</text>
    <path d="M 0 54 L 280 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+get_betas(), +get_alphas()</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">+get_alphas_cumprod()</text>
  </g>

  <path d="M 1120 500 L 1120 470" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- DPTrainer (Core DP Engine) -->
  <g transform="translate(980, 630)">
    <rect x="0" y="0" width="310" height="230" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2" filter="url(#uml-shadow)"/>
    <text x="155" y="24" font-size="14" font-weight="700" text-anchor="middle" fill="#991b1b">DPTrainer (DP-SGD Engine)</text>
    <path d="M 0 32 L 310 32" stroke="#fecaca" stroke-width="1.5"/>
    <text x="10" y="48" font-size="10" font-family="monospace" fill="#991b1b">+denoiser: GradSampleModule</text>
    <text x="10" y="62" font-size="10" font-family="monospace" fill="#991b1b">+schedule: AbstractNoiseSchedule</text>
    <text x="10" y="76" font-size="10" font-family="monospace" fill="#991b1b">+optimizer: torch.optim.Optimizer</text>
    <text x="10" y="90" font-size="10" font-family="monospace" fill="#991b1b">+accountant: AbstractAccountant</text>
    <text x="10" y="104" font-size="10" font-family="monospace" fill="#991b1b">+privacy_schedule: AdaptiveSchedule</text>
    <text x="10" y="118" font-size="10" font-family="monospace" fill="#991b1b">+dataset_size: int</text>
    <text x="10" y="132" font-size="10" font-family="monospace" fill="#991b1b">+tier_params: dict[str, list]</text>
    <text x="10" y="146" font-size="10" font-family="monospace" fill="#991b1b">+tier_clip_norms: dict[str, float]</text>
    <path d="M 0 154 L 310 154" stroke="#fecaca" stroke-width="1.5"/>
    <text x="10" y="172" font-size="11" font-weight="700" font-family="monospace" fill="#7f1d1d">+train_epoch(loader): float</text>
    <text x="10" y="190" font-size="10" font-family="monospace" fill="#475569"># Invokes clip_and_noise_tier()</text>
    <text x="10" y="206" font-size="10" font-family="monospace" fill="#475569"># Enforces L2 clip C &amp; Noise σ*C</text>
  </g>

  <!-- Connectors into DPTrainer -->
  <path d="M 1120 630 L 1120 595" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>
  <path d="M 1030 630 L 1030 350" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>

  <!-- ==================== COLUMN 5: PRIVACY ACCOUNTING & ORCHESTRATION ==================== -->
  
  <!-- Interface: AbstractPrivacyAccountant -->
  <g transform="translate(1330, 105)">
    <rect x="0" y="0" width="270" height="95" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="135" y="20" font-size="11" font-style="italic" text-anchor="middle" fill="#64748b">&lt;&lt;interface&gt;&gt;</text>
    <text x="135" y="36" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">AbstractPrivacyAccountant</text>
    <path d="M 0 45 L 270 45" stroke="#cbd5e1" stroke-width="1"/>
    <text x="10" y="64" font-size="10" font-family="monospace" fill="#334155">+record_step(noise, q)*: None</text>
    <text x="10" y="80" font-size="10" font-family="monospace" fill="#334155">+get_epsilon(delta: float)*: float</text>
  </g>

  <!-- CentralPrivacyAccountant -->
  <g transform="translate(1330, 230)">
    <rect x="0" y="0" width="270" height="110" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="135" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">CentralPrivacyAccountant</text>
    <path d="M 0 30 L 270 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">-_accountant: RDPAccountant</text>
    <path d="M 0 54 L 270 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+record_step(noise, sample_rate)</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155">+get_epsilon(target_delta): float</text>
    <text x="10" y="100" font-size="10" font-family="monospace" fill="#334155">+steps: int</text>
  </g>

  <path d="M 1465 230 L 1465 200" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>

  <!-- AdaptiveNoiseSchedule -->
  <g transform="translate(1330, 370)">
    <rect x="0" y="0" width="270" height="110" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="135" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">AdaptiveNoiseSchedule</text>
    <path d="M 0 30 L 270 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">+base_sigma: float, +strategy: str</text>
    <text x="10" y="60" font-size="10" font-family="monospace" fill="#64748b">+num_timesteps: int, +sigmas: Tensor</text>
    <path d="M 0 68 L 270 68" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="84" font-size="10" font-family="monospace" fill="#334155">-_compute_schedule(): Tensor</text>
    <text x="10" y="99" font-size="10" font-family="monospace" fill="#334155">+get_sigma(t: int): float</text>
  </g>

  <!-- HeuristicRiskTierAssigner -->
  <g transform="translate(1330, 510)">
    <rect x="0" y="0" width="270" height="110" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="135" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">HeuristicRiskTierAssigner</text>
    <path d="M 0 30 L 270 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">+correlation_threshold: float</text>
    <path d="M 0 54 L 270 54" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="70" font-size="10" font-family="monospace" fill="#334155">+assign_tiers(df, profiles)</text>
    <text x="10" y="85" font-size="10" font-family="monospace" fill="#334155"># Heuristic tiering</text>
    <text x="10" y="100" font-size="10" font-family="monospace" fill="#334155"># Correlation Leakage Guard</text>
  </g>

  <!-- Connectors into DPTrainer from Privacy tools -->
  <path d="M 1290 680 L 1330 680 L 1330 285" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>
  <path d="M 1290 710 L 1330 710 L 1330 425" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>

  <!-- ComputeBudgetGuard -->
  <g transform="translate(1330, 650)">
    <rect x="0" y="0" width="270" height="120" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
    <text x="135" y="22" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">ComputeBudgetGuard</text>
    <path d="M 0 30 L 270 30" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="46" font-size="10" font-family="monospace" fill="#64748b">+state_file: Path, +max_seconds: float</text>
    <text x="10" y="60" font-size="10" font-family="monospace" fill="#64748b">+session_start: float</text>
    <path d="M 0 68 L 270 68" stroke="#e2e8f0" stroke-width="1"/>
    <text x="10" y="84" font-size="10" font-family="monospace" fill="#334155">+check_budget(): None</text>
    <text x="10" y="98" font-size="10" font-family="monospace" fill="#334155">+get_elapsed_seconds(): float</text>
    <text x="10" y="112" font-size="10" font-family="monospace" fill="#334155">-_save_state(total): None (atomic)</text>
  </g>

  <!-- ==================== BOTTOM ROW: EVALUATION ENGINE ==================== -->
  <g transform="translate(50, 890)">
    <rect x="0" y="0" width="1550" height="210" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="25" y="28" font-size="13" font-weight="700" fill="#334155">EVALUATION &amp; METRIC VALIDATION SUBSYSTEM (src/evaluation/)</text>

    <!-- UtilityEvaluator -->
    <g transform="translate(25, 45)">
      <rect x="0" y="0" width="450" height="140" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
      <text x="225" y="24" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">UtilityEvaluator</text>
      <path d="M 0 32 L 450 32" stroke="#e2e8f0" stroke-width="1"/>
      <text x="15" y="50" font-size="10" font-family="monospace" fill="#64748b">+df_real: DataFrame, +df_synth: DataFrame</text>
      <path d="M 0 58 L 450 58" stroke="#e2e8f0" stroke-width="1"/>
      <text x="15" y="76" font-size="10" font-family="monospace" fill="#334155">+evaluate_univariate(): dict[str, Any]  # KS-test &amp; TVD</text>
      <text x="15" y="94" font-size="10" font-family="monospace" fill="#334155">+evaluate_bivariate_correlation_rmse(): float</text>
      <text x="15" y="112" font-size="10" font-family="monospace" fill="#334155">-_compute_categorical_tvd(col: str): float</text>
      <text x="15" y="130" font-size="10" font-family="monospace" fill="#334155">-_compute_continuous_ks(col: str): float</text>
    </g>

    <!-- PrivacyEvaluator -->
    <g transform="translate(510, 45)">
      <rect x="0" y="0" width="450" height="140" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
      <text x="225" y="24" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">PrivacyEvaluator</text>
      <path d="M 0 32 L 450 32" stroke="#e2e8f0" stroke-width="1"/>
      <text x="15" y="50" font-size="10" font-family="monospace" fill="#64748b">+df_train: DataFrame, +df_holdout: DataFrame, +df_synth: DataFrame</text>
      <path d="M 0 58 L 450 58" stroke="#e2e8f0" stroke-width="1"/>
      <text x="15" y="76" font-size="10" font-family="monospace" fill="#334155">+evaluate_mia_risk(): dict[str, Any]  # D-MIA score</text>
      <text x="15" y="94" font-size="10" font-family="monospace" fill="#334155">-_factorize_data(df: DataFrame, reference_cols: list): ndarray</text>
      <text x="15" y="112" font-size="10" font-family="monospace" fill="#16a34a">✓ Fits NearestNeighbors on Synthetic data</text>
      <text x="15" y="130" font-size="10" font-family="monospace" fill="#16a34a">✓ Compares Train->Synth dist vs Holdout->Synth dist</text>
    </g>

    <!-- Legend -->
    <g transform="translate(1000, 45)">
      <rect x="0" y="0" width="520" height="140" rx="6" fill="#FFFFFF" stroke="#cbd5e1" stroke-width="1.5" filter="url(#uml-shadow)"/>
      <text x="260" y="24" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">UML Relationship Legend</text>
      <path d="M 0 32 L 520 32" stroke="#e2e8f0" stroke-width="1"/>
      
      <line x1="30" y1="55" x2="80" y2="55" stroke="#0f172a" stroke-width="1.5" marker-end="url(#inheritance)"/>
      <text x="95" y="59" font-size="11" fill="#334155">Inheritance / Realization (&lt;|--)</text>

      <line x1="30" y1="85" x2="80" y2="85" stroke="#0f172a" stroke-width="1.5" marker-start="url(#composition)"/>
      <text x="95" y="89" font-size="11" fill="#334155">Composition (*--) (Strict Lifecycle Ownership)</text>

      <line x1="30" y1="115" x2="80" y2="115" stroke="#0f172a" stroke-width="1.5" marker-start="url(#aggregation)"/>
      <text x="95" y="119" font-size="11" fill="#334155">Aggregation (o--) (Injected Reference)</text>

      <line x1="300" y1="55" x2="350" y2="55" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#dependency)"/>
      <text x="365" y="59" font-size="11" fill="#334155">Dependency (..&gt;)</text>
    </g>
  </g>
</svg>
"""

# Write all 4 SVGs to the docs/ directory
with open(docs_dir / "system_architecture_diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg_architecture)

with open(docs_dir / "end_to_end_workflow_diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg_workflow)

with open(docs_dir / "data_flow_diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg_dfd)

with open(docs_dir / "core_class_diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg_class_diagram)

print("Generated 4 SVG diagrams in docs/ successfully!")
