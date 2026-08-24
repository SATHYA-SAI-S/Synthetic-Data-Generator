"""
SYNTHGUARD Screen 4 - The Master OP Output Dashboard
Fully adaptive: header badge reflects the ACTUAL red-team verdict and
the real epsilon spend (or "Pending" if no run has completed).
"""
import streamlit as st
from ui.components.panel_b_privacy_ledger import render_panel_b
from ui.components.panel_c_stats import render_panel_c
from ui.components.panel_d_export import render_panel_d
from ui.components.panel_e_redteam import render_panel_e


def render_screen4():
    """Render the Master Output Layer (The OP Dashboard) without data grid clutter."""
    ds_name = st.session_state.get("dataset_name", "Clinical Cohort")
    n_rows = st.session_state.get("num_rows", 0)
    cols = st.session_state.get("num_cols_clean", 0)
    eps = st.session_state.get("epsilon_spent")

    # Adaptive verdict from the live red-team audit.
    attack_report = st.session_state.get("attack_report") or {}
    verdict = attack_report.get("verdict", "PENDING_AUDIT")
    certified = verdict == "PRIVACY_CERTIFIED"
    badge_color = "#4ADE80" if certified else ("#F87171" if verdict != "PENDING_AUDIT" else "#FBBF24")
    badge_text = ("Mathematically Certified" if certified else
                  ("Privacy Flagged" if verdict != "PENDING_AUDIT" else "Audit Pending"))
    eps_disp = f"Epsilon={eps:.4f}" if eps is not None else "Epsilon=Pending"

    # Executive Synthesis Summary Header Card
    st.markdown(f"""
    <div class="synth-card-highlight" style="margin-bottom: 20px; padding: 20px 24px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h2 style="color: #F8FAFC; margin: 0; font-size: 1.6rem; font-weight: 800;">
            Master OP Output Dashboard
          </h2>
          <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
            Target Dataset: <b style="color: #38BDF8;">{ds_name}</b> ({n_rows:,} records &bull; {cols} protected clinical features)
          </div>
        </div>
        <div style="text-align: right;">
          <span class="badge-pass" style="font-size: 0.85rem; padding: 6px 14px; background: {badge_color};">{badge_text} ({eps_disp})</span>
          <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Red-Team Verified &bull; Domain Clamped</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Epsilon Selector Toolbar
    c_eps1, c_eps2 = st.columns([1, 3])
    with c_eps1:
        active_eps = st.selectbox(
            "Active Privacy Tier (Epsilon Sweep)",
            ["0.1", "1.0", "10.0"],
            index=1,
            help="Select which privacy sweep setting to inspect across the audit panels."
        )
        st.session_state.active_epsilon_view = active_eps

    # Master Audit Tabs (Without the Data Grid)
    tab_privacy, tab_stats, tab_redteam, tab_export = st.tabs([
        "Privacy Compliance Ledger",
        "Statistical Utility & Drift Suite",
        "Red-Team Privacy Audit",
        "Secure Artifact Export"
    ])

    with tab_privacy:
        render_panel_b()

    with tab_stats:
        render_panel_c()

    with tab_redteam:
        render_panel_e()

    with tab_export:
        render_panel_d()