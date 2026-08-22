import os
import uuid
import json

with open('ui/state_schema.py', 'r', encoding='utf-8') as f:
    schema = f.read()

new_schema = """
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
        if isinstance(v, (str, int, float, bool, list, dict)) and k != "kaggle_credentials":
            dump[k] = v
    with open(f"sessions/{sid}/state.json", "w") as f:
        json.dump(dump, f)

def init_session_state():
    \"\"\"Initialize session state defaults if not already present.\"\"\"
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
        "noise_multiplier": 5.00,
        "clip_norm": 1.0,
        "epochs": 5,
        "batch_size": 256,
        "epsilon_spent": 0.3720,
        "mia_advantage": -0.0083,
        "mia_attack_auc": 0.4958,
        "tvd_best": 0.0316,
        "bivariate_rmse": 0.1948,
        "trtr_auc": 0.6855,
        "tstr_auc": 0.4962,
        "tstr_retention_pct": 72.39,
        "unhandled_nans": 0,
        "domain_violations": 0,
        "active_epsilon_view": "1.0",
        "uploaded_file_hash": None,
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_session():
    \"\"\"Reset session state to initial uncompleted state for new upload.\"\"\"
    st.cache_data.clear()
    new_id = "session_" + str(uuid.uuid4())[:8]
    st.session_state.session_id = new_id
    for k in [
        "dataset_name", "raw_data_path", "synthetic_data_path", "num_rows", 
        "num_cols_raw", "num_cols_clean", "encoded_dim", "hipaa_dropped",
        "missingness_flags_count", "profile_complete", "training_complete",
        "generation_complete", "sanitization_complete", "uploaded_file_hash"
    ]:
        st.session_state[k] = defaults.get(k, None) if "defaults" in locals() else (0 if "count" in k or "dim" in k or "num_" in k else ([] if "dropped" in k else False))
        if k == "dataset_name" or "path" in k or "hash" in k:
             st.session_state[k] = None

    st.session_state.step = 1
    save_session()
"""

with open('ui/state_schema.py', 'w', encoding='utf-8') as f:
    f.write(new_schema)
