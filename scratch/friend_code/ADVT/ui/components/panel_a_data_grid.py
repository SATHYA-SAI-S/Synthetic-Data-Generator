"""
SYNTHGUARD Component - Panel A: Cleaned Data Grid Viewer
"""
import streamlit as st
import pandas as pd
from ui.data_loaders.load_synthetic_csv import load_dataset_sample

def render_panel_a():
    """Render interactive, paginated view of sanitized synthetic patient records."""
    preset = st.session_state.get("dataset_preset", "UCI Diabetes")
    active_eps = st.session_state.get("active_epsilon_view", "1.0")
    
    st.markdown("""
    <div class="synth-card-highlight">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h3 style="margin: 0; color: #F8FAFC;">Panel A: Sanitized Synthetic Records</h3>
          <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
            Cleaned post-processed data stream. 0 NaN values, all integer counts domain-bounded.
          </div>
        </div>
        <div>
          <span class="badge-pass">Domain Clamped</span>
          <span class="badge-info">Paginated</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    real_df, synth_df, meta = load_dataset_sample(preset, active_eps, max_rows=5000)
    
    # Search and Filter Toolbar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("Search Synthetic Values / Diagnoses", "", placeholder="e.g. 250.0, Female, Caucasian...")
    with col2:
        page_size = st.selectbox("Records per Page", [25, 50, 100, 250], index=1)
    with col3:
        page_num = st.number_input("Page", min_value=1, max_value=max(1, len(synth_df)//page_size), value=1)
        
    filtered_df = synth_df.copy()
    if search_query:
        mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
        filtered_df = filtered_df[mask]
        
    # Pagination slicing
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    st.markdown(f"**Showing records {start_idx+1} to {min(end_idx, len(filtered_df))} of {len(filtered_df):,} total generated rows**")
    
    # Enhanced DataFrame display
    st.dataframe(
        page_df,
        use_container_width=True,
        height=400
    )
    
    # Patient Record Spot-Check Card
    st.markdown("---")
    st.markdown("### Clinical Record Spot-Check Inspection")
    spot_index = st.slider("Select Record Index for Deep Field Inspection", 0, len(synth_df)-1, 0)
    
    sample_record = synth_df.iloc[spot_index].to_dict()
    
    c1, c2, c3 = st.columns(3)
    keys = list(sample_record.keys())
    split_size = max(1, len(keys)//3)
    
    for i, col in enumerate([c1, c2, c3]):
        with col:
            st.markdown("""<div class="synth-card" style="padding: 12px;">""", unsafe_allow_html=True)
            chunk_keys = keys[i*split_size:(i+1)*split_size] if i < 2 else keys[i*split_size:]
            for k in chunk_keys:
                val = sample_record[k]
                st.markdown(f"<div style='font-size: 0.8rem; color: #94A3B8;'>{k}: <b style='color: #F8FAFC;'>{val}</b></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
