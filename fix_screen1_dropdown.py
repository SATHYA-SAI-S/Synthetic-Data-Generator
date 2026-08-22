import os

with open('ui/screens/screen1_ingestion.py', 'r', encoding='utf-8') as f:
    content = f.read()

injection_target = '    # Prominent Upload Box (Centered Layout)'

dropdown_code = """    # --- Recent Sessions Dropdown ---
    available_sessions = []
    if os.path.exists("sessions"):
        for d in os.listdir("sessions"):
            if d.startswith("session_") and os.path.exists(f"sessions/{d}/raw_upload.csv"):
                available_sessions.append(d)
    
    if available_sessions:
        col_s1, col_s2, col_s3 = st.columns([1, 3, 1])
        with col_s2:
            st.markdown("### Resume Previous Session")
            selected_session = st.selectbox("Select an existing session to resume:", ["-- Select --"] + sorted(available_sessions, reverse=True))
            if selected_session != "-- Select --" and selected_session != st.session_state.get("session_id"):
                with open("sessions/last_session.txt", "w") as f:
                    f.write(selected_session)
                st.session_state.clear()
                st.rerun()
                
"""

if '# --- Recent Sessions Dropdown ---' not in content:
    content = content.replace(injection_target, dropdown_code + injection_target)
    with open('ui/screens/screen1_ingestion.py', 'w', encoding='utf-8') as f:
        f.write(content)
