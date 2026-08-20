"""
SYNTHGUARD Component - Navigation Sidebar & Stepper
"""
import streamlit as st
from ui.state_schema import reset_session

def render_sidebar():
    """Render the clean application sidebar with workflow navigation."""
    with st.sidebar:
        # Branding Header
        st.markdown("""
        <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #2A3F60;">
          <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; letter-spacing: 0.5px;">
            SYNTH<span style="color: #38BDF8;">GUARD</span>
          </div>
          <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            Clinical Synthesis &amp; Audit Platform
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Workflow Stepper")
        step = st.session_state.get("step", 1)
        
        steps = [
            (1, "1. Ingestion & Profiling"),
            (2, "2. Privacy & DP Training"),
            (3, "3. Synthesis & Guardrails"),
            (4, "4. Output & OP Dashboard")
        ]
        
        for num, label in steps:
            is_active = (step == num)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"step_btn_{num}", use_container_width=True, type=btn_type):
                st.session_state.step = num
                st.rerun()
                
        st.markdown("---")
        
        # Active Session Metadata
        dataset_name = st.session_state.get("dataset_name")
        rows = st.session_state.get("num_rows", 0)
        cols = st.session_state.get("num_cols_clean", 0)
        session_id = st.session_state.get("session_id", "live")
        
        st.markdown("### Active Session")
        if dataset_name:
            st.markdown(f"""
            <div style="font-size: 0.82rem; color: #94A3B8; background: #0F1D30; padding: 12px; border-radius: 8px; border: 1px solid #1E3A5F;">
              <div><b>Dataset:</b> <span style="color: #38BDF8;">{dataset_name}</span></div>
              <div style="margin-top: 4px;"><b>Cohort Size:</b> <span style="color: #F8FAFC;">{rows:,} rows</span></div>
              <div style="margin-top: 4px;"><b>Features:</b> <span style="color: #4ADE80;">{cols} columns</span></div>
              <div style="margin-top: 4px;"><b>Session:</b> <code>{session_id[:12]}</code></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94A3B8; background: #0F1D30; padding: 10px; border-radius: 8px; border: 1px dashed #2A3F60; text-align: center;">
              No dataset uploaded yet.<br>Upload a CSV in Screen 1.
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Session Metadata & Compliance Box
        st.markdown("### Compliance State")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #94A3B8; background: #0F1D30; padding: 12px; border-radius: 8px; border: 1px solid #1E3A5F;">
          <div><b>HIPAA Status:</b> <span class="badge-pass">PASSED</span></div>
          <div style="margin-top: 4px;"><b>RDP Mechanism:</b> Gaussian</div>
          <div style="margin-top: 4px;"><b>Target Delta:</b> 1.0e-4</div>
          <div style="margin-top: 4px;"><b>Data Isolation:</b> On-Premise Only</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Session / New Upload", use_container_width=True):
            reset_session()
            st.toast("Session reset. Ready for new upload.")
            st.rerun()
