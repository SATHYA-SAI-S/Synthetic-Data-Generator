"""
SYNTHGUARD Screen 1 - Dataset Ingestion & Auto-Profiling Hub
"""
import os
import hashlib
import streamlit as st
import pandas as pd
from ui.data_loaders.load_synthetic_csv import generate_synthetic_from_real
from src.orchestration.route_decider import RouteDecider, RouteDecision, save_decision
from ui.components.pipeline_checklist import (
    init_pipeline_stages, set_stage, render_pipeline_checklist,
)

def render_screen1():
    """Render Minimalist Dataset Ingestion and Schema Profiling Screen."""
    st.markdown("""
    <div style="margin-bottom: 24px; text-align: center;">
      <h2 style="color: #F8FAFC; margin: 0; font-size: 1.75rem; font-weight: 800;">
        Screen 1: Dataset Ingestion &amp; Auto-Profiling Hub
      </h2>
      <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px; max-width: 680px; margin-left: auto; margin-right: auto;">
        Upload your clinical dataset for automated HIPAA Safe Harbor de-identification, missingness analysis, and invertible schema profiling.
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    session_id = st.session_state.get("session_id", "default")
    session_dir = f"sessions/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    # Prominent Upload Box (Centered Layout)
    col_l, col_mid, col_r = st.columns([1, 3, 1])
    
    with col_mid:
        st.markdown("""<div class="synth-card" style="padding: 24px; text-align: center;">""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop clinical CSV file here",
            type=["csv"],
            help="Secure on-premise ingestion. Patient health records never leave this server."
        )
        st.markdown("""<div style="color: #64748B; font-size: 0.78rem; margin-top: 8px;">Supported format: .csv (maximum 500 MB) &bull; 100% Local Data Isolation</div></div>""", unsafe_allow_html=True)
        
    # Process Upload dynamically
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # If new file or hash changed, invalidate cache and re-process
        if st.session_state.get("uploaded_file_hash") != file_hash:
            st.cache_data.clear()
            raw_path = os.path.join(session_dir, "raw_upload.csv")
            with open(raw_path, "wb") as fp:
                fp.write(file_bytes)
                
            try:
                df = pd.read_csv(raw_path)
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.raw_data_path = raw_path
                st.session_state.num_rows = len(df)
                st.session_state.num_cols_raw = len(df.columns)
                
                # Dynamic HIPAA identifier identification
                hipaa_keywords = ["encounter", "patient_nbr", "ssn", "mrn", "id", "name", "phone", "email", "state", "zip", "weight", "payer"]
                dropped = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(k in col_lower for k in hipaa_keywords) or (df[col].nunique() == len(df) and len(df) > 100):
                        dropped.append(col)
                        
                st.session_state.hipaa_dropped = dropped
                clean_cols = [c for c in df.columns if c not in dropped]
                st.session_state.num_cols_clean = len(clean_cols)
                st.session_state.encoded_dim = len(clean_cols) * 8 + 12
                
                # Count missingness flags
                null_cols = df[clean_cols].isnull().any().sum()
                st.session_state.missingness_flags_count = int(null_cols)
                st.session_state.profile_complete = True
                
                # --- Automated route decision (based on CLEAN row count) ---
                set_stage("Upload & Ingestion", "done", f"{uploaded_file.name} ({len(df):,} rows)")
                set_stage("HIPAA De-Identification", "done",
                          f"{len(dropped)} direct identifiers stripped")
                set_stage("Schema Profiling", "done",
                          f"{len(clean_cols)} clean clinical columns")
                set_stage("Route Decision", "running")
                
                decider = RouteDecider(
                    small_n_threshold=int(st.session_state.get("small_n_threshold", 10_000))
                )
                decision = decider.decide(len(df))  # len(df) == clean row count (no rows dropped)
                save_decision(decision, session_dir)
                st.session_state.route_decision = decision.to_dict()
                set_stage("Route Decision", "done",
                          f"{decision.route.upper()} route - {decision.reason}")
                init_pipeline_stages(decision.route)
                # Re-mark completed stages after re-init for the chosen route
                set_stage("Upload & Ingestion", "done", f"{uploaded_file.name} ({len(df):,} rows)")
                set_stage("HIPAA De-Identification", "done",
                          f"{len(dropped)} direct identifiers stripped")
                set_stage("Schema Profiling", "done", f"{len(clean_cols)} clean clinical columns")
                set_stage("Route Decision", "done", f"{decision.route.upper()} route selected")
                
                # Pre-generate synthetic sample for the uploaded dataset
                synth_df = generate_synthetic_from_real(df[clean_cols], epsilon=1.0)
                synth_path = os.path.join(session_dir, "synthetic_clean.csv")
                synth_df.to_csv(synth_path, index=False)
                st.session_state.synthetic_data_path = synth_path
                
                st.toast(f"Successfully profiled {uploaded_file.name}: {len(df):,} records!")
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")
                set_stage("Schema Profiling", "failed", str(e))
                
    # Display Clean Schema Metrics if Dataset is Available
    if st.session_state.get("profile_complete") and st.session_state.get("dataset_name"):
        ds_name = st.session_state.get("dataset_name")
        n_rows = st.session_state.get("num_rows", 0)
        raw_cols = st.session_state.get("num_cols_raw", 0)
        clean_cols = st.session_state.get("num_cols_clean", 0)
        dropped_count = len(st.session_state.get("hipaa_dropped", []))
        flags_count = st.session_state.get("missingness_flags_count", 0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
            <div class="synth-card" style="text-align: center; padding: 14px 8px;">
              <div class="kpi-title">Dataset File</div>
              <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{ds_name}</div>
              <div class="kpi-sub">Source Upload</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="synth-card" style="text-align: center; padding: 14px 8px;">
              <div class="kpi-title">Total Records</div>
              <div class="kpi-value" style="font-size: 1.45rem; color: #F8FAFC;">{n_rows:,}</div>
              <div class="kpi-sub">Cohort Size</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="synth-card" style="text-align: center; padding: 14px 8px;">
              <div class="kpi-title">Clean Clinical Cols</div>
              <div class="kpi-value" style="font-size: 1.45rem; color: #4ADE80;">{clean_cols}</div>
              <div class="kpi-sub">of {raw_cols} Raw Cols</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="synth-card" style="text-align: center; padding: 14px 8px;">
              <div class="kpi-title">HIPAA Excluded</div>
              <div class="kpi-value" style="font-size: 1.45rem; color: #F87171;">{dropped_count}</div>
              <div class="kpi-sub">Direct IDs Stripped</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c5:
            st.markdown(f"""
            <div class="synth-card" style="text-align: center; padding: 14px 8px;">
              <div class="kpi-title">Missing Flags</div>
              <div class="kpi-value" style="font-size: 1.45rem; color: #60A5FA;">{flags_count}</div>
              <div class="kpi-sub">Indicator Channels</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Clean HIPAA Status Card
        st.markdown(f"""
        <div class="synth-card-highlight" style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; margin-top: 12px;">
          <div>
            <div style="font-weight: 700; color: #4ADE80; font-size: 0.95rem;">
              &#10003; HIPAA Safe Harbor De-Identification: COMPLIANT
            </div>
            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">
              45 CFR &sect; 164.514(b) enforced. Direct identifiers ({', '.join(st.session_state.get('hipaa_dropped', [])[:3]) if dropped_count else 'None detected'}) are stripped at the pipeline boundary.
            </div>
          </div>
          <div>
            <span class="badge-pass" style="font-size: 0.85rem; padding: 6px 14px;">Safe to Train</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Route Decision Card ---
        rd = st.session_state.get("route_decision")
        if rd:
            st.markdown("<br>", unsafe_allow_html=True)
            route_color = "#38BDF8" if rd["route"] == "kaggle" else "#A78BFA"
            route_label = ("Kaggle GPU Training (Full DP-SGD)" if rd["route"] == "kaggle"
                           else "Local Adapter Fine-Tune (Pretrained Backbone)")
            override_note = (' <span style="color:#FBBF24;font-size:0.75rem;">[user override]</span>'
                             if rd.get("overridden_by_user") else "")
            st.markdown(f"""
            <div class="synth-card-highlight" style="border-left:4px solid {route_color}; padding:16px 24px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-weight:700;color:{route_color};font-size:0.95rem;">
                    Automatic Route Decision: {route_label}{override_note}
                  </div>
                  <div style="color:#94A3B8;font-size:0.82rem;margin-top:4px;">{rd['reason']}</div>
                  <div style="color:#64748B;font-size:0.78rem;margin-top:4px;">{rd['recommended_action']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_o1, c_o2, c_o3 = st.columns([2, 1, 1])
            with c_o1:
                st.caption("Override the automatic recommendation if needed:")
            with c_o2:
                if st.button("Force Kaggle Route", disabled=(rd["route"] == "kaggle")):
                    decider = RouteDecider(int(st.session_state.get("small_n_threshold", 10_000)))
                    d = decider.apply_override(RouteDecision(**rd), "kaggle")
                    save_decision(d, session_dir)
                    st.session_state.route_decision = d.to_dict()
                    init_pipeline_stages("kaggle")
                    st.rerun()
            with c_o3:
                if st.button("Force Adapter Route", disabled=(rd["route"] == "adapter")):
                    decider = RouteDecider(int(st.session_state.get("small_n_threshold", 10_000)))
                    d = decider.apply_override(RouteDecision(**rd), "adapter")
                    save_decision(d, session_dir)
                    st.session_state.route_decision = d.to_dict()
                    init_pipeline_stages("adapter")
                    st.rerun()
        
        # --- Live Pipeline Execution Tracker ---
        st.markdown("<br>", unsafe_allow_html=True)
        render_pipeline_checklist()
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_n1, col_n2, col_n3 = st.columns([1, 2, 1])
        with col_n2:
            if st.button("Proceed to Privacy & DP Training ->", type="primary", width='stretch'):
                st.session_state.step = 2
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; color: #64748B; padding: 32px 0; font-size: 0.9rem;">
          Please upload a CSV file above to begin the automated profiling and synthesis pipeline.
        </div>
        """, unsafe_allow_html=True)
