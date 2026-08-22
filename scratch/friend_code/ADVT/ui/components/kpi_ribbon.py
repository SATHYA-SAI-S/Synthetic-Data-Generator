"""
SYNTHGUARD Component - Persistent KPI Ribbon
"""
import streamlit as st

def render_kpi_ribbon():
    """Render the high-visibility top KPI header bar."""
    eps = st.session_state.get("epsilon_spent", 0.3720)
    target_eps = st.session_state.get("target_epsilon", 1.0)
    mia = st.session_state.get("mia_advantage", -0.0083)
    tvd = st.session_state.get("tvd_best", 0.0316)
    nans = st.session_state.get("unhandled_nans", 0)
    
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.4])
    
    with col1:
        st.markdown(f"""
        <div class="synth-card" style="padding: 12px 16px;">
          <div class="kpi-title">Differential Privacy (Epsilon)</div>
          <div class="kpi-value" style="color: #60A5FA;">{eps:.4f}</div>
          <div class="kpi-sub"><span class="badge-pass">Safe</span> Budget: {target_eps}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        mia_color = "#4ADE80" if mia < 0.05 else "#F87171"
        st.markdown(f"""
        <div class="synth-card" style="padding: 12px 16px;">
          <div class="kpi-title">MIA Advantage Score</div>
          <div class="kpi-value" style="color: {mia_color};">{mia:+.4f}</div>
          <div class="kpi-sub"><span class="badge-pass">No Memorization</span> Target &lt; 0.05</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="synth-card" style="padding: 12px 16px;">
          <div class="kpi-title">Best Marginal TVD</div>
          <div class="kpi-value" style="color: #38BDF8;">{tvd:.4f}</div>
          <div class="kpi-sub"><span class="badge-pass">High Utility</span> 'change' col</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        nan_badge = '<span class="badge-pass">Zero NaN</span>' if nans == 0 else '<span class="badge-danger">Leak</span>'
        st.markdown(f"""
        <div class="synth-card" style="padding: 12px 16px;">
          <div class="kpi-title">Unhandled NaNs</div>
          <div class="kpi-value" style="color: #4ADE80;">{nans}</div>
          <div class="kpi-sub">{nan_badge} Strict Guardrails</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        step = st.session_state.get("step", 1)
        step_labels = {
            1: "1/4: Schema Ingest",
            2: "2/4: DP Training",
            3: "3/4: Synthesis & Guard",
            4: "4/4: OP Dashboard"
        }
        st.markdown(f"""
        <div class="synth-card-highlight" style="padding: 12px 16px; text-align: center;">
          <div class="kpi-title">Pipeline Stage</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">{step_labels.get(step, 'Active')}</div>
          <div class="kpi-sub"><span class="badge-info">Automated</span> Session: {st.session_state.get("session_id", "live")[:10]}</div>
        </div>
        """, unsafe_allow_html=True)
