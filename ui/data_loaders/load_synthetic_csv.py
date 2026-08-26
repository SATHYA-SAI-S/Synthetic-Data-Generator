"""
SYNTHGUARD Data Loader - Dynamic Real & Synthetic CSV Datasets
"""
import os
import hashlib
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def load_dataset_sample(session_id: str = None, epsilon_str: str = "1.0", max_rows: int = None) -> tuple:
    """
    Load real and synthetic datasets for the current session dynamically.
    Returns (real_df, synth_df, metadata_dict). If max_rows is None, loads full uncapped datasets.
    """
    if session_id is None:
        session_id = st.session_state.get("session_id", "default")
        
    session_dir = f"sessions/{session_id}"
    real_path = os.path.join(session_dir, "raw_upload.csv")
    synth_path = os.path.join(session_dir, "synthetic_clean.csv")
    
    real_df = None
    synth_df = None
    
    # 1. Check if session has uploaded file
    if os.path.exists(real_path):
        try:
            real_df = pd.read_csv(real_path, nrows=max_rows) if max_rows else pd.read_csv(real_path)
        except Exception:
            pass
            
    # 2. Check if synthetic file exists for session
    if os.path.exists(synth_path):
        try:
            synth_df = pd.read_csv(synth_path, nrows=max_rows) if max_rows else pd.read_csv(synth_path)
        except Exception:
            pass
            
    # 3. No fallback synthesis. If it doesn't exist on disk, return None.
    # We do not fake the synthetic generation.
    if synth_df is not None and len(synth_df) == 0:
        synth_df = None
        
    # 4. Fallback if no upload in session yet
    if real_df is None:
        if os.path.exists("data/diabetic_data.csv"):
            try:
                real_df = pd.read_csv("data/diabetic_data.csv", nrows=max_rows) if max_rows else pd.read_csv("data/diabetic_data.csv")
            except Exception:
                pass
                
    meta = {
        "name": st.session_state.get("dataset_name", "Clinical Cohort"),
        "total_rows": len(real_df),
        "total_cols": len(synth_df.columns) if synth_df is not None else len(real_df.columns),
        "session_id": session_id
    }
    
    return real_df, synth_df, meta
