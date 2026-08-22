import os

with open('ui/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'save_session' not in content:
    content = content.replace('from ui.state_schema import init_session_state', 'from ui.state_schema import init_session_state, save_session')
    content += "\n\n# Persist session on every run\nsave_session()\n"
    with open('ui/app.py', 'w', encoding='utf-8') as f:
        f.write(content)
