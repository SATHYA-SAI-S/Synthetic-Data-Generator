"""
SYNTHGUARD Component - Interactive Profiler Console
"""
import streamlit as st

def render_profiler_console(logs: list = None):
    """Render a themed terminal log window with color-coded tokens."""
    if not logs:
        logs = [
            "[01:21:43] [INFO] Loaded dataset into memory buffer (101,766 rows x 50 columns).",
            "[01:21:44] [INFO] Initializing DatasetProfiler with HIPAA Safe Harbor detection rules.",
            "[01:21:44] [HIPAA] Detected direct identifier: 'encounter_id' (Cardinality: 101,766) -> DROPPED",
            "[01:21:44] [HIPAA] Detected direct identifier: 'patient_nbr' (Cardinality: 71,518) -> DROPPED",
            "[01:21:44] [WARN] Column 'weight' exceeds 95% missingness -> DROPPED",
            "[01:21:44] [WARN] Column 'payer_code' non-clinical billing artifact -> DROPPED",
            "[01:21:44] [INFO] Detected 11 clinical features with sparsity -> Injected 28 __missing_flag indicators.",
            "[01:21:45] [INFO] Invertible StandardScaler fitted on 6 continuous clinical features.",
            "[01:21:45] [INFO] Invertible OneHotEncoder fitted on 23 categorical columns (D_encoded = 616).",
            "[01:21:45] [DONE] Schema Intelligence registry successfully compiled and saved to disk."
        ]
        
    formatted = []
    for line in logs:
        if "[HIPAA]" in line:
            formatted.append(f"<div class='log-hipaa'>{line}</div>")
        elif "[WARN]" in line:
            formatted.append(f"<div class='log-warn'>{line}</div>")
        elif "[DONE]" in line:
            formatted.append(f"<div class='log-success'>{line}</div>")
        else:
            formatted.append(f"<div class='log-info'>{line}</div>")
            
    html_content = "".join(formatted)
    st.markdown(f"""
    <div class="synth-terminal">
      {html_content}
    </div>
    """, unsafe_allow_html=True)
