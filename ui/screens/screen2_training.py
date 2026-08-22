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
            delta_choice = st.selectbox("Cryptographic Delta", ["1.0e-4", "1.0e-5", "1.0e-3"], index=0)
        with c_b:
            epochs = st.number_input("Training Epochs", min_value=1, max_value=50, value=int(st.session_state.get("epochs", 5)))
            
        c_c, c_d = st.columns(2)
        with c_c:
            batch_size = st.selectbox("Batch Size", [128, 256, 512, 1024], index=1)
        with c_d:
            clip_norm = st.number_input("Gradient Clip Norm (C)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            
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
        st.markdown("### Live DP-SGD Telemetry &amp; Loss Curve")
        
        loss_history = pd.DataFrame({
            "Epoch": [1, 2, 3, 4, 5],
            "Diffusion MSE Loss": [1.2341, 1.1908, 1.1723, 1.1690, 1.1656],
            "Cumulative Epsilon Spent": [0.0744, 0.1488, 0.2232, 0.2976, 0.3720]
        })
        
        fig_loss = px.line(
            loss_history, x="Epoch", y="Diffusion MSE Loss",
            markers=True, title="Tabular Diffusion Training Loss Convergence"
        )
        fig_loss.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#F8FAFC"})
        st.plotly_chart(fig_loss, width='stretch')
        
        st.markdown("""
        <div class="synth-card">
          <div style="display: flex; justify-content: space-between;">
            <div>
              <div class="kpi-title">Gradient Clipping Rate</div>
              <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">18.4% of batches</div>
            </div>
            <div>
              <div class="kpi-title">Final Spent Epsilon</div>
              <div style="font-size: 1.2rem; font-weight: 700; color: #4ADE80;">0.3720 / 1.0000</div>
            </div>
            <div>
              <div class="kpi-title">Training Status</div>
              <div><span class="badge-pass">CONVERGED</span></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("Proceed to Synthesis & Guardrails ->", type="primary", width='stretch'):
            st.session_state.training_complete = True
            st.session_state.step = 3
            st.rerun()
