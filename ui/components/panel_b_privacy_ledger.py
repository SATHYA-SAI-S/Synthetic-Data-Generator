"""
SYNTHGUARD Component - Panel B: Privacy Compliance Ledger
Fully adaptive: displays the actual run's privacy parameters and metrics,
or "Pending" placeholders until a real run completes.
"""
import streamlit as st
import pandas as pd
from ui.components.epsilon_gauge import render_epsilon_gauge


def render_panel_b():
    """Render the Differential Privacy compliance ledger and MIA attack proof (adaptive)."""
    # Pull session-derived values (None-safe; all adaptive).
    epsilon_spent = st.session_state.get("epsilon_spent")
    target_eps = st.session_state.get("target_epsilon", 1.0)
    delta_choice = st.session_state.get("delta_choice", "1.0e-4")
    noise_multiplier = st.session_state.get("noise_multiplier")
    clip_norm = st.session_state.get("clip_norm", 1.0)
    mia_score = st.session_state.get("mia_advantage")
    attack_auc = st.session_state.get("mia_attack_auc")
    dropped = st.session_state.get("hipaa_dropped", [])

    # Adaptive display strings.
    eps_disp = f"{epsilon_spent:.4f}" if epsilon_spent is not None else "Pending"
    sigma_disp = f"{noise_multiplier:.2f}" if noise_multiplier is not None else "Pending"
    clip_disp = f"{clip_norm:.2f}" if clip_norm is not None else "Pending"
    mia_disp = f"{mia_score:+.4f}" if mia_score is not None else "Pending"
    auc_disp = f"{attack_auc:.4f}" if attack_auc is not None else "Pending"
    mia_color = "#4ADE80" if (mia_score is not None and mia_score < 0.05) else ("#F87171" if mia_score is not None else "#94A3B8")

    # Adaptive identifier list (HIPAA Safe Harbor is dynamic per dataset).
    dropped_str = ", ".join(dropped[:3]) if dropped else "none detected"
    dropped_rendered = dropped_str

    st.markdown("""
    <div class="synth-card-highlight">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h3 style="margin: 0; color: #F8FAFC;">Panel B: Privacy Compliance Ledger & Audit</h3>
          <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
            Differential Privacy guarantee and Membership Inference Attack verification for this run.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Differential Privacy Accounting")
        render_epsilon_gauge(spent=epsilon_spent, target=target_eps)

        st.markdown(f"""
        <div class="synth-card" style="margin-top: -10px;">
          <table style="width: 100%; font-size: 0.85rem; color: #94A3B8;">
            <tr><td style="padding: 4px 0;"><b>Privacy Mechanism:</b></td><td style="color: #F8FAFC;">Gaussian Mechanism via Renyi DP</td></tr>
            <tr><td style="padding: 4px 0;"><b>Target Delta:</b></td><td style="color: #F8FAFC;">{delta_choice} (Cryptographic Bound)</td></tr>
            <tr><td style="padding: 4px 0;"><b>Noise Multiplier (Sigma):</b></td><td style="color: #F8FAFC;">{sigma_disp}</td></tr>
            <tr><td style="padding: 4px 0;"><b>Gradient Clip Norm (C):</b></td><td style="color: #F8FAFC;">{clip_disp}</td></tr>
            <tr><td style="padding: 4px 0;"><b>Accountant Engine:</b></td><td style="color: #4ADE80;">Renyi DP (Gaussian)</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Adversarial Attack Audit (D-MIA)")
        st.markdown(f"""
        <div class="synth-card">
          <div class="kpi-title">Membership Inference Advantage</div>
          <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {mia_color};">
            {mia_disp}
          </div>
          <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 8px;">
            <b>Shadow Attack AUC:</b> <span style="color: #F8FAFC;">{auc_disp}</span> (Baseline: 0.5000)
          </div>
          <div style="font-size: 0.78rem; color: #64748B; margin-top: 12px; line-height: 1.4;">
            The attacker's success relative to random baseline reflects the effective privacy of this run.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="synth-card" style="padding: 14px;">
          <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">HIPAA Safe Harbor Exclusion</div>
          <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
            Direct identifiers stripped for this dataset: {dropped_rendered}.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Per-Epoch Accounting History (adaptive: derive from real run when available,
    # otherwise show a single-row "pending" placeholder rather than fabricated data).
    st.markdown("### Per-Epoch Differential Privacy Budget Log")
    if epsilon_spent is None:
        # No real run yet - show placeholder only, no fabricated epochs.
        st.dataframe(pd.DataFrame([{
            "Status": "Awaiting training run",
            "Epsilon Spent": "Pending",
            "Loss": "Pending"
        }]), width='stretch')
    else:
        # If the run produced epoch telemetry, surface it; otherwise a summary row.
        epoch_log = st.session_state.get("epoch_telemetry")
        if epoch_log:
            st.dataframe(pd.DataFrame(epoch_log), width='stretch')
        else:
            st.dataframe(pd.DataFrame([{
                "Spent Epsilon": f"{epsilon_spent:.4f}",
                "Target Epsilon": f"{target_eps:.4f}",
                "Delta": delta_choice,
                "Clip Norm (C)": clip_disp,
            }]), width='stretch')
