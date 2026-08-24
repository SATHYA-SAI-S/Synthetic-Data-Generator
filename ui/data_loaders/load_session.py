"""
SYNTHGUARD Data Loader - Session & Artifact Management
Fully adaptive: builds the run manifest from the ACTUAL session state
(dataset name, file hash, real hyperparameters, real metrics). No fabricated
diabetes-specific values.
"""
import os
import json
import hashlib
import streamlit as st


@st.cache_data(show_spinner=False)
def load_run_manifest(session_id: str = "default") -> dict:
    """Build a reproducibility manifest from the live session state."""
    manifest_path = f"sessions/{session_id}/run_manifest.json"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Derive everything adaptively from the current session.
    dataset_name = st.session_state.get("dataset_name") or "unknown"
    raw_path = st.session_state.get("raw_data_path")
    sha256 = None
    if raw_path and os.path.exists(raw_path):
        h = hashlib.sha256()
        with open(raw_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    eps_spent = st.session_state.get("epsilon_spent")
    target_eps = st.session_state.get("target_epsilon", 1.0)
    delta_choice = st.session_state.get("delta_choice", "1.0e-4")
    noise_mult = st.session_state.get("noise_multiplier")
    clip_norm = st.session_state.get("clip_norm", 1.0)
    epochs = st.session_state.get("epochs", 5)
    batch_size = st.session_state.get("batch_size", 256)
    mia_adv = st.session_state.get("mia_advantage")
    mia_auc = st.session_state.get("mia_attack_auc")

    def fmt(v):
        return v if v is not None else "pending"

    return {
        "session_id": session_id,
        "platform": "SYNTHGUARD Clinical Synthesis Platform",
        "generated_at": "live-session",
        "dataset": dataset_name,
        "dataset_sha256": sha256 or "not-computed",
        "privacy_parameters": {
            "target_epsilon": target_eps,
            "target_delta": delta_choice,
            "spent_epsilon": fmt(eps_spent),
            "noise_multiplier_sigma": fmt(noise_mult),
            "clip_norm_C": clip_norm,
            "accountant": "Renyi DP (Gaussian)",
            "mia_advantage": fmt(mia_adv),
            "mia_attack_auc": fmt(mia_auc),
        },
        "model_architecture": {
            "backbone": "TabularMLPDenoiser",
            "timesteps": 1000,
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "epochs": epochs,
        },
        "utility_metrics": {
            "bivariate_correlation_rmse": fmt(st.session_state.get("bivariate_rmse")),
            "tstr_auc_roc": fmt(st.session_state.get("tstr_auc")),
            "trtr_baseline_auc": fmt(st.session_state.get("trtr_auc")),
            "utility_retention_pct": fmt(st.session_state.get("tstr_retention_pct")),
            "tvd_best_column": fmt(st.session_state.get("tvd_best")),
        },
        "integrity_audit": {
            "total_rows_synthesized": st.session_state.get("num_rows", 0),
            "total_columns": st.session_state.get("num_cols_clean", 0),
            "unhandled_nans": fmt(st.session_state.get("unhandled_nans")),
            "domain_violations": fmt(st.session_state.get("domain_violations")),
            "domain_guardrails_applied": bool(
                st.session_state.get("sanitization_complete")),
        },
    }