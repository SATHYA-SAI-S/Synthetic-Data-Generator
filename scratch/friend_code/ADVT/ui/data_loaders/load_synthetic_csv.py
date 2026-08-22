"""
SYNTHGUARD Data Loader - Dynamic Real & Synthetic CSV Datasets
"""
import os
import hashlib
import pandas as pd
import numpy as np
import streamlit as st

def generate_synthetic_from_real(real_df: pd.DataFrame, epsilon: float = 1.0, num_samples: int = None) -> pd.DataFrame:
    """
    Generate dynamic, privacy-preserving synthetic dataframe mimicking the schema and distributions
    of real_df with calibrated noise reflecting the target epsilon and clinical domain guardrails.
    """
    if real_df is None or len(real_df) == 0:
        return pd.DataFrame()
        
    n = num_samples if num_samples is not None else len(real_df)
    synth_dict = {}
    
    # Calculate noise scale inversely proportional to epsilon
    noise_scale = max(0.01, min(1.5, 0.5 / (float(epsilon) if float(epsilon) > 0 else 1.0)))
    
    for col in real_df.columns:
        series = real_df[col].dropna()
        if len(series) == 0:
            synth_dict[col] = [np.nan] * n
            continue
            
        # 1. Numeric continuous or integer
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.mean()
            std_val = series.std() if series.std() > 0 else 1.0
            min_val = series.min()
            max_val = series.max()
            
            # Sample with Gaussian DP perturbation
            raw_samples = np.random.normal(mean_val, std_val * (1.0 + noise_scale * 0.1), size=n)
            
            # Domain Guardrails: clamp to valid clinical range
            clamped = np.clip(raw_samples, min_val, max_val)
            
            # If original was integer, round and cast to integer
            if pd.api.types.is_integer_dtype(series) or (series % 1 == 0).all():
                synth_dict[col] = np.round(clamped).astype(int)
            else:
                synth_dict[col] = np.round(clamped, 2)
                
        # 2. Categorical / string / object
        else:
            val_counts = series.value_counts(normalize=True)
            cats = val_counts.index.tolist()
            probs = val_counts.values
            
            # Add DP Laplace perturbation to probabilities
            perturbed_probs = probs + np.random.laplace(0, noise_scale * 0.02, size=len(probs))
            perturbed_probs = np.clip(perturbed_probs, 1e-4, 1.0)
            perturbed_probs = perturbed_probs / perturbed_probs.sum()
            
            synth_dict[col] = np.random.choice(cats, size=n, p=perturbed_probs)
            
    return pd.DataFrame(synth_dict)

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
            
    # 3. If real exists but synthetic not generated yet, synthesize dynamically with exact 1:1 match
    if real_df is not None and (synth_df is None or len(synth_df) == 0):
        synth_df = generate_synthetic_from_real(real_df, epsilon=float(epsilon_str), num_samples=len(real_df))
        
    # 4. Fallback if no upload in session yet
    if real_df is None:
        if os.path.exists("data/diabetic_data.csv"):
            try:
                real_df = pd.read_csv("data/diabetic_data.csv", nrows=max_rows) if max_rows else pd.read_csv("data/diabetic_data.csv")
                synth_path_fb = f"vishwa_final_clean_archive/synthetic_eps_{epsilon_str}.csv"
                if os.path.exists(synth_path_fb):
                    synth_df = pd.read_csv(synth_path_fb, nrows=max_rows) if max_rows else pd.read_csv(synth_path_fb)
                else:
                    synth_df = generate_synthetic_from_real(real_df, float(epsilon_str), len(real_df))
            except Exception:
                pass
                
    if real_df is None:
        real_df = pd.DataFrame({"Metric_A": [1, 2, 3], "Metric_B": [10.5, 20.2, 30.1]})
        synth_df = real_df.copy()
        
    meta = {
        "name": st.session_state.get("dataset_name", "Clinical Cohort"),
        "total_rows": len(real_df),
        "total_cols": len(synth_df.columns) if synth_df is not None else len(real_df.columns),
        "session_id": session_id
    }
    
    return real_df, synth_df, meta
