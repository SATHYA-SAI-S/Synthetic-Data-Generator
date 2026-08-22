import sys

with open('ui/screens/screen3_generation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove consent from _run_kaggle_route
old_consent = '''    consent = st.checkbox(
        "I understand the de-identified dataset will be uploaded to my private Kaggle dataset "
        "for GPU training.", key="kaggle_consent")
    if not consent:
        st.info("Consent required to start the Kaggle training job.")
        return None, None
'''
if old_consent in content:
    content = content.replace(old_consent, '')
else:
    # try slightly different whitespace
    pass

# 2. Add consent before the button
old_button = '        if st.button("Trigger Full Generation & Sanitization", type="primary", width=\'stretch\'):'
new_button = '''        if route == "kaggle":
            st.checkbox("I understand the de-identified dataset will be uploaded to my private Kaggle dataset for GPU training.", key="kaggle_consent")
            
        if st.button("Trigger Full Generation & Sanitization", type="primary", width='stretch'):'''
content = content.replace(old_button, new_button)

# 3. Add consent guard inside the button handler
old_call = '''            if route == "kaggle":
                got_raw, synth_df = _run_kaggle_route(session_dir, clean_cols, target_size)'''
new_call = '''            if route == "kaggle":
                if not st.session_state.get("kaggle_consent", False):
                    st.error("Consent required to start the Kaggle training job.")
                    return
                got_raw, synth_df = _run_kaggle_route(session_dir, clean_cols, target_size)'''
content = content.replace(old_call, new_call)

with open('ui/screens/screen3_generation.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed screen3_generation.py")
