"""
SYNTHGUARD Session State Schema.
Fully adaptive: metrics are NOT pre-seeded with fabricated values. They default
to None until the real run produces them, so every dashboard panel reflects the
user's actual dataset, not the diabetes demo.
"""
import os
import uuid
import json
import streamlit as st

def save_session():
    if "session_id" not in st.session_state:
        return
    sid = st.session_state.session_id
    os.makedirs(f"sessions/{sid}", exist_ok=True)
    with open("sessions/last_session.txt", "w") as f:
        f.write(sid)
    dump = {}
    for k, v in st.session_state.items():
        if isinstance(v, (str, int, float, bool, list, dict)) and k != "kaggle_credentials" and not k.endswith("_btn") and "step_btn" not in k and "FormSubmitter" not in k:
            dump[k] = v
    with open(f"sessions/{sid}/state.json", "w") as f:
        json.dump(dump, f)


def init_session_state():
    """Initialize session state defaults if not already present."""
    if "session_id" not in st.session_state:
        restored = False
        if os.path.exists("sessions/last_session.txt"):
            with open("sessions/last_session.txt", "r") as f:
                last_sid = f.read().strip()
            state_file = f"sessions/{last_sid}/state.json"
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        dump = json.load(f)
                    for k, v in dump.items():
                        st.session_state[k] = v
                    st.session_state.session_id = last_sid
                    restored = True
                except Exception:
                    pass
        if not restored:
            st.session_state.session_id = "session_" + str(uuid.uuid4())[:8]

    # NOTE: Runtime metrics (epsilon_spent, mia_advantage, etc.) default to None
    # (adaptive) — they are populated only by the actual training/generation run.
    defaults = {
        "step": 1,
        "dataset_name": None,
        "raw_data_path": None,
        "synthetic_data_path": None,
        "num_rows": 0,
        "num_cols_raw": 0,
        "num_cols_clean": 0,
        "encoded_dim": 0,
        "hipaa_dropped": [],
        "missingness_flags_count": 0,
        "profile_complete": False,
        "training_complete": False,
        "generation_complete": False,
        "sanitization_complete": False,
        "target_epsilon": 1.0,
        "target_delta": 1e-4,
        "delta_choice": "1.0e-4",
        "noise_multiplier": None,       # adaptive - set from run
        "clip_norm": 1.0,
        "epochs": 5,
        "batch_size": 256,
        "epsilon_spent": None,            # adaptive | None = not run yet
        "mia_advantage": None,            # adaptive
        "mia_attack_auc": None,           # adaptive
        "tvd_best": None,                 # adaptive
        "bivariate_rmse": None,           # adaptive
        "trtr_auc": None,                 # adaptive
        "tstr_auc": None,                 # adaptive
        "tstr_retention_pct": None,       # adaptive
        "unhandled_nans": None,           # adaptive
        "domain_violations": None,        # adaptive
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
    # Keys to clear: profile/metrics/artifacts from a previous run.
    clear_keys = [
        "dataset_name", "raw_data_path", "synthetic_data_path", "num_rows",
        "num_cols_raw", "num_cols_clean", "encoded_dim", "hipaa_dropped",
        "missingness_flags_count", "profile_complete", "training_complete",
        "generation_complete", "sanitization_complete", "uploaded_file_hash",
        "epsilon_spent", "mia_advantage", "mia_attack_auc", "tvd_best",
        "bivariate_rmse", "trtr_auc", "tstr_auc", "tstr_retention_pct",
        "unhandled_nans", "domain_violations", "generation_info",
        "attack_report", "route_decision", "pipeline_stages",
    ]
    for k in clear_keys:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.step = 1
    # Re-initialize adaptive defaults, keeping the fresh session_id.
    defaults = {
        "step": 1,
        "dataset_name": None, "raw_data_path": None, "synthetic_data_path": None,
        "num_rows": 0, "num_cols_raw": 0, "num_cols_clean": 0, "encoded_dim": 0,
        "hipaa_dropped": [], "missingness_flags_count": 0,
        "profile_complete": False, "training_complete": False,
        "generation_complete": False, "sanitization_complete": False,
        "target_epsilon": 1.0, "target_delta": 1e-4, "delta_choice": "1.0e-4",
        "noise_multiplier": None, "clip_norm": 1.0, "epochs": 5,
        "batch_size": 256,
        "epsilon_spent": None, "mia_advantage": None, "mia_attack_auc": None,
        "tvd_best": None, "bivariate_rmse": None, "trtr_auc": None,
        "tstr_auc": None, "tstr_retention_pct": None,
        "unhandled_nans": None, "domain_violations": None,
        "active_epsilon_view": "1.0", "uploaded_file_hash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    save_session()
