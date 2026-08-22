"""
SYNTHGUARD - Main Application Entry Point
Title: Privacy-Preserving Synthetic Healthcare Data Generation
"""
import os
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import streamlit as st
from ui.state_schema import init_session_state
from ui.components.kpi_ribbon import render_kpi_ribbon
from ui.components.sidebar import render_sidebar
from ui.screens.screen1_ingestion import render_screen1
from ui.screens.screen2_training import render_screen2
from ui.screens.screen3_generation import render_screen3
from ui.screens.screen4_output import render_screen4

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="Privacy-Preserving Synthetic Healthcare Data Generation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Theme CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. Initialize State
init_session_state()

# 4. Render Persistent Sidebar Navigation
render_sidebar()

# 5. Header Title & Branding
st.markdown("""
<div style="margin-bottom: 16px;">
  <div style="display: flex; align-items: center; gap: 12px;">
    <div style="font-size: 2.2rem;"></div>
    <div>
      <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
        Privacy-Preserving Synthetic Healthcare Data Generation
      </h1>
      <div style="color: #38BDF8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
        SYNTHGUARD Enterprise Clinical Synthesis &amp; Mathematical Audit Platform
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# 6. Render Persistent KPI Header Ribbon
render_kpi_ribbon()
st.markdown("<hr style='border-color: #2A3F60; margin: 12px 0 20px 0;'>", unsafe_allow_html=True)

# 7. Screen Router based on Workflow State
step = st.session_state.get("step", 1)

if step == 1:
    render_screen1()
elif step == 2:
    render_screen2()
elif step == 3:
    render_screen3()
elif step == 4:
    render_screen4()
else:
    render_screen1()
