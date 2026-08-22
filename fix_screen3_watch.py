import os

with open('ui/screens/screen3_generation.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_watch = "        final_job = bridge.watch(on_update=lambda job: None)"
new_watch = """        def _update_ui(job):
            prog = job.progress or {}
            pct = int(prog.get("pct", 0))
            loss = prog.get("loss", 0.0)
            stage = prog.get("stage", job.status)
            progress_bar.progress(min(pct, 99))
            status_text.markdown(f"**Kaggle {stage}... {pct}% (loss: {loss:.4f})**")

        final_job = bridge.watch(on_update=_update_ui)"""

content = content.replace(old_watch, new_watch)

with open('ui/screens/screen3_generation.py', 'w', encoding='utf-8') as f:
    f.write(content)
