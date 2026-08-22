"""
SYNTHGUARD Panel E - Red-Team Privacy Audit Results.

Renders the adaptive attacker's escalation-ladder results alongside the
formal epsilon-DP guarantee: proof (ceiling) + empirical validation.
"""
import json
import os
import streamlit as st

VERDICT_ICON = {"held": "&#9989;", "partial": "&#9888;&#65039;", "leak": "&#10060;"}
VERDICT_COLOR = {"held": "#4ADE80", "partial": "#FBBF24", "leak": "#F87171"}


def render_panel_e():
    """Render the Red-Team Audit tab on the OP Dashboard."""
    report = st.session_state.get("attack_report")

    st.markdown("""
    <div style="color:#94A3B8;font-size:0.85rem;margin-bottom:12px;">
      An adaptive rule-based attacker escalates through four attack levels after every
      generation run. The formal &epsilon;-DP proof is the <b>ceiling</b> of the privacy
      claim; this red-team audit <b>empirically validates</b> the implementation.
    </div>
    """, unsafe_allow_html=True)

    if not report:
        # Try loading from the session directory
        session_id = st.session_state.get("session_id", "default")
        path = os.path.join("sessions", session_id, "attack_report.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                st.session_state.attack_report = report
            except Exception:
                pass

    if not report:
        st.info("No red-team audit has run yet. Trigger generation on Screen 3 - "
                "the attacker runs automatically afterwards.")
        return

    certified = report.get("verdict") == "PRIVACY_CERTIFIED"
    color = "#4ADE80" if certified else "#F87171"
    eps = report.get("epsilon_claimed")

    st.markdown(f"""
    <div class="synth-card-highlight" style="border-left:4px solid {color};padding:16px 24px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-weight:800;color:{color};font-size:1.05rem;">
            {'&#9989; PRIVACY CERTIFIED' if certified else '&#10060; PRIVACY FLAGGED'}
          </div>
          <div style="color:#94A3B8;font-size:0.82rem;margin-top:4px;">
            Worst attack success rate: <b style="color:#F8FAFC;">{report.get('worst_success_rate', 0):.2%}</b>
            &bull; Flag threshold: {report.get('risk_threshold', 0.05):.0%}
            {f'&bull; Claimed &epsilon;: {eps}' if eps is not None else ''}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for r in report.get("results", []):
        v = r.get("verdict", "held")
        icon = VERDICT_ICON.get(v, "&#9744;")
        vcolor = VERDICT_COLOR.get(v, "#64748B")
        rate = r.get("success_rate", 0.0)
        bar_pct = int(min(rate, 1.0) * 100)
        st.markdown(f"""
        <div class="synth-card" style="padding:12px 18px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-weight:700;color:#F8FAFC;font-size:0.88rem;">
              {icon} {r.get('level','')} &bull; {r.get('attack','').replace('_',' ').title()}
            </div>
            <div style="font-weight:700;color:{vcolor};font-size:0.88rem;">{rate:.2%}</div>
          </div>
          <div style="height:5px;background:#1E293B;border-radius:3px;margin:8px 0;">
            <div style="width:{bar_pct}%;height:100%;background:{vcolor};border-radius:3px;"></div>
          </div>
          <div style="color:#94A3B8;font-size:0.78rem;">{r.get('detail','')}</div>
        </div>
        """, unsafe_allow_html=True)

    if not certified:
        st.warning("Recommendation: increase noise (lower epsilon), reduce epochs, or "
                   "re-route to the adapter path, then regenerate and re-audit.")