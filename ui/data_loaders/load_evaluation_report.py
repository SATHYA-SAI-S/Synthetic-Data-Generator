"""
SYNTHGUARD Data Loader - Evaluation Report Parser
"""
import os
import streamlit as st

@st.cache_data(show_spinner=False)
def load_evaluation_report_text() -> str:
    """Load the raw text of evaluation_report.txt from workspace root."""
    paths = ["evaluation_report.txt", "evaluation_report_vishwa_final.txt"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return "EVALUATION REPORT NOT FOUND ON DISK."

@st.cache_data(show_spinner=False)
def get_parsed_evaluation_metrics() -> dict:
    """Return structured metrics extracted from the audit report."""
    return {
        "objectives": [],
        "tvd_scores": {},
        "tstr_metrics": {}
    }
