import sys

with open('ui/screens/screen3_generation.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_button = '''        if route == "kaggle":
            st.checkbox("I understand the de-identified dataset will be uploaded to my private Kaggle dataset for GPU training.", key="kaggle_consent")'''

new_button = '''        if route == "kaggle":
            creds = st.session_state.get("kaggle_credentials", {})
            if creds.get("username") and creds.get("key"):
                st.markdown("<div style='color: #4ADE80; font-size: 0.9rem; margin-bottom: 8px;'>&#10003; Credentials loaded from Automation Settings.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: #F87171; font-size: 0.9rem; margin-bottom: 8px;'>&#10007; Missing Kaggle credentials in Automation Settings.</div>", unsafe_allow_html=True)
            
            st.checkbox("I understand the de-identified dataset will be uploaded to my private Kaggle dataset for GPU training.", key="kaggle_consent")'''

content = content.replace(old_button, new_button)

with open('ui/screens/screen3_generation.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated credentials UI")
