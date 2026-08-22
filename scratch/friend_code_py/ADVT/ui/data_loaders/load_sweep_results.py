"""
SYNTHGUARD Data Loader - Hyperparameter & Sweep Results
"""
import os
import json
import streamlit as st

@st.cache_data(show_spinner=False)
def load_sweep_summary() -> list:
    """Load multi-epsilon sweep results for comparative trade-off inspection."""
    return [
        {
            "epsilon": 0.1,
            "label": "High Privacy (Epsilon = 0.1)",
            "privacy_tier": "Maximum Protection",
            "mia_advantage": -0.0124,
            "bivariate_rmse": 0.3120,
            "tstr_auc": 0.4610,
            "retention_pct": 67.25,
            "tvd_avg": 0.2840,
            "file": "vishwa_final_clean_archive/synthetic_eps_0.1.csv"
        },
        {
            "epsilon": 1.0,
            "label": "Optimal Balanced (Epsilon = 1.0)",
            "privacy_tier": "Certified High Privacy",
            "mia_advantage": -0.0083,
            "bivariate_rmse": 0.1948,
            "tstr_auc": 0.4962,
            "retention_pct": 72.39,
            "tvd_avg": 0.1420,
            "file": "vishwa_final_clean_archive/synthetic_eps_1.0.csv"
        },
        {
            "epsilon": 10.0,
            "label": "High Utility (Epsilon = 10.0)",
            "privacy_tier": "Moderate Protection",
            "mia_advantage": 0.0041,
            "bivariate_rmse": 0.1082,
            "tstr_auc": 0.5840,
            "retention_pct": 85.19,
            "tvd_avg": 0.0810,
            "file": "vishwa_final_clean_archive/synthetic_eps_10.0.csv"
        }
    ]
