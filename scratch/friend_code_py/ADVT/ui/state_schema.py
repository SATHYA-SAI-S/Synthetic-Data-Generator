"""
SYNTHGUARD - Dynamic Session State Schema & Workflow Coordinator
"""
import os
import uuid
import streamlit as st

def init_session_state():
    """Initialize session state defaults if not already present."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session_" + str(uuid.uuid4())[:8]
        
    defaults = {
        "step": 1,  # 1: Ingest, 2: Train, 3: Generate, 4: OP Output
        "dataset_name": None,
        "raw_data_path": None,
        "synthetic_data_path": None,
        "num_rows": 0,
        "num_cols_raw": 0,
        "num_cols_clean": 0,
        "encoded_dim": 0,
        "hipaa_dropped": [],
        "missingness_flags_count": 0,
        
        # Screening & Progress Gates
        "profile_complete": False,
        "training_complete": False,
        "generation_complete": False,
        "sanitization_complete": False,
        
        # Privacy Configuration & Accounting
        "target_epsilon": 1.0,
        "target_delta": 1e-4,
        "noise_multiplier": 5.00,
        "clip_norm": 1.0,
        "epochs": 5,
        "batch_size": 256,
        "epsilon_spent": 0.3720,
        "mia_advantage": -0.0083,
        "mia_attack_auc": 0.4958,
        
        # Utility & Integrity Metrics
        "tvd_best": 0.0316,
        "bivariate_rmse": 0.1948,
        "trtr_auc": 0.6855,
        "tstr_auc": 0.4962,
        "tstr_retention_pct": 72.39,
        "unhandled_nans": 0,
        "domain_violations": 0,
        
        # Active Data Caches
        "active_epsilon_view": "1.0",
        "uploaded_file_hash": None,
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_session():
    """Reset session state to initial uncompleted state for new upload."""
    st.cache_data.clear()
    new_id = "session_" + str(uuid.uuid4())[:8]
    st.session_state.session_id = new_id
    st.session_state.step = 1
    st.session_state.dataset_name = None
    st.session_state.raw_data_path = None
    st.session_state.synthetic_data_path = None
    st.session_state.num_rows = 0
    st.session_state.num_cols_raw = 0
    st.session_state.num_cols_clean = 0
    st.session_state.encoded_dim = 0
    st.session_state.hipaa_dropped = []
    st.session_state.missingness_flags_count = 0
    st.session_state.profile_complete = False
    st.session_state.training_complete = False
    st.session_state.generation_complete = False
    st.session_state.sanitization_complete = False
    st.session_state.uploaded_file_hash = None
