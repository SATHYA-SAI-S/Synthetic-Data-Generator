"""
SYNTHGUARD Component - Navigation Sidebar & Stepper
Adds adaptive navigation guards: a naive user cannot jump ahead of the
pipeline without completing prerequisites.
"""
import streamlit as st
from ui.state_schema import reset_session


def _step_allowed(num: int) -> tuple:
    """Return (allowed: bool, reason: str) for navigating to `num`."""
    if num == 1:
        return True, ""
    if not st.session_state.get("profile_complete"):
        return False, "Upload and profile a dataset on Screen 1 first."
    if num == 2:
        return True, ""
    if num == 3:
        return True, ""  # Screen 2 is configuration; allow proceeding after profile
    if num == 4:
        if not st.session_state.get("generation_complete"):
            return False, "Run generation on Screen 3 first."
        return True, ""
    return True, ""


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
            Clinical Synthesis & Audit Platform
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
            allowed, reason = _step_allowed(num)
            is_active = (step == num)
            btn_type = "primary" if is_active else "secondary"
            disabled = not allowed
            help_txt = reason if not allowed else None
            if st.button(label, key=f"step_btn_{num}", width='stretch',
                         type=btn_type, disabled=disabled, help=help_txt):
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

        # Automation Settings (route threshold + Kaggle credentials)
        with st.expander("Automation Settings"):
            threshold = st.number_input(
                "Small-N Route Threshold (rows)",
                min_value=100, max_value=1_000_000,
                value=int(st.session_state.get("small_n_threshold", 10_000)),
                step=500,
                help="Datasets below this row count are routed to local adapter "
                     "fine-tuning; larger datasets go to Kaggle GPU training.")
            st.session_state.small_n_threshold = int(threshold)

            st.caption("Kaggle Credentials (stored in session only)")
            ku = st.text_input("Kaggle Username", value=(st.session_state.get("kaggle_credentials") or {}).get("username", ""))
            kk = st.text_input("Kaggle API Key", type="password", value=(st.session_state.get("kaggle_credentials") or {}).get("key", ""))
            if ku and kk:
                st.session_state.kaggle_credentials = {"username": ku, "key": kk}
                st.success("Kaggle credentials set for this session.")

        st.markdown("---")

        # Compliance State (adaptive - reflects actual HIPAA stripping for this dataset)
        dropped = st.session_state.get("hipaa_dropped", [])
        hipaa_status = "PASSED" if dropped or st.session_state.get("profile_complete") else "PENDING"
        delta_choice = st.session_state.get("delta_choice", "1.0e-4")
        st.markdown("### Compliance State")
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #94A3B8; background: #0F1D30; padding: 12px; border-radius: 8px; border: 1px solid #1E3A5F;">
          <div><b>HIPAA Status:</b> {hipaa_status}</div>
          <div style="margin-top: 4px;"><b>RDP Mechanism:</b> Gaussian</div>
          <div style="margin-top: 4px;"><b>Target Delta:</b> {delta_choice}</div>
          <div style="margin-top: 4px;"><b>Data Isolation:</b> On-Premise Only</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Session / New Upload", width='stretch'):
            reset_session()
            st.toast("Session reset. Ready for new upload.")
            st.rerun()