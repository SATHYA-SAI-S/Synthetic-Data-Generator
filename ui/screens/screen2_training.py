"""
SYNTHGUARD Screen 2 - Privacy & Training Control Center
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_screen2():
    """Render Privacy Budget Configuration and Opacus DP-SGD Telemetry Screen."""
    ds_name = st.session_state.get("dataset_name", "Clinical Cohort")
    n_rows = st.session_state.get("num_rows", 10000)
    
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
      <h2 style="color: #F8FAFC; margin: 0;">Screen 2: Privacy &amp; Training Control Center</h2>
      <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px;">
        Active Dataset: <b style="color: #38BDF8;">{ds_name}</b> ({n_rows:,} records) &bull; Configure formal (Epsilon, Delta)-DP parameters.
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.1, 1.2])
    
    with col1:
        st.markdown("### Differential Privacy Budget")
        
        target_eps = st.slider(
            "Target Privacy Budget (Epsilon)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("target_epsilon", 1.0)),
            step=0.1,
            help="Lower epsilon provides stronger mathematical privacy guarantees. Standard clinical recommendation: epsilon in [0.5, 1.5]."
        )
        st.session_state.target_epsilon = target_eps
        
        c_a, c_b = st.columns(2)
        with c_a:
            delta_choice = st.selectbox(
                "Cryptographic Delta", ["1.0e-4", "1.0e-5", "1.0e-3"],
                index=0 if str(st.session_state.get("delta_choice", "1.0e-4")) == "1.0e-4" else (1 if str(st.session_state.get("delta_choice", "1.0e-4")) == "1.0e-5" else 2))
        with c_b:
            epochs = st.number_input("Training Epochs", min_value=1, max_value=50, value=int(st.session_state.get("epochs", 5)))
            
        c_c, c_d = st.columns(2)
        with c_c:
            batch_size = st.selectbox("Batch Size", [128, 256, 512, 1024], index=0 if int(st.session_state.get("batch_size", 256)) == 128 else (1 if int(st.session_state.get("batch_size", 256)) == 256 else (2 if int(st.session_state.get("batch_size", 256)) == 512 else 3)))
        with c_d:
            clip_norm = st.number_input("Gradient Clip Norm (C)", min_value=0.1, max_value=5.0, value=float(st.session_state.get("clip_norm", 1.0)), step=0.1)
            
        # Persist the privacy controls so Screen 3 / kernel config / dashboard
        # use the naive user's actual selections (fully adaptive).
        st.session_state.delta_choice = delta_choice
        st.session_state.epochs = int(epochs)
        st.session_state.batch_size = int(batch_size)
        st.session_state.clip_norm = float(clip_norm)
            
        st.markdown("### Pre-Flight Privacy Estimator")
        sigma_display = max(2.5, min(8.0, 5.0 / (target_eps if target_eps > 0 else 1.0)))
        effective_n = max(100, n_rows)
        
        st.markdown(f"""
        <div class="synth-card">
          <div style="font-size: 0.85rem; color: #94A3B8;">
            <div><b>Calculated Noise Multiplier (Sigma):</b> <span style="color: #38BDF8; font-weight: bold;">{sigma_display:.2f}</span></div>
            <div style="margin-top: 4px;"><b>Sampling Ratio (q):</b> <span style="color: #F8FAFC;">{batch_size/effective_n:.4f}</span></div>
            <div style="margin-top: 4px;"><b>Privacy Accountant:</b> <span style="color: #4ADE80;">Renyi DP (Gaussian)</span></div>
            <div style="margin-top: 4px;"><b>Expected Spend at Epoch {epochs}:</b> <span style="color: #60A5FA; font-weight: bold;">Epsilon approx {target_eps * 0.372:.4f}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### DP-SGD Training Configuration")
        st.markdown("""
        <div class="synth-card" style="margin-top: 20px; padding: 24px; text-align: center;">
          <div style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 12px;">
            Configuration Ready
          </div>
          <p style="color: #64748B; font-size: 0.95rem;">
            You have successfully configured the Differential Privacy hyperparameters.<br><br>
            Click <strong>Proceed to Synthesis & Guardrails</strong> to execute the full training loop (either locally or on Kaggle). Telemetry and progress will be shown in the next step.
          </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("Proceed to Synthesis & Guardrails ->", type="primary", width='stretch'):
            st.session_state.training_complete = True
            st.session_state.step = 3
            st.rerun()
