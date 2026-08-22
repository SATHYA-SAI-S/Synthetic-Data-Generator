"""
SYNTHGUARD - Pipeline Execution Checklist Component.

Renders a persistent step-tracker (checkmark / in-progress / failed) so the
user always knows what stage the automation is at. Stages are stored in
session state as a list of dicts:
    {"name": str, "status": "pending"|"running"|"done"|"failed",
     "detail": str, "ts": float|None}
"""
import time
import streamlit as st

STATUS_ICON = {
    "pending": "&#9744;",   # ballot box
    "running": "&#9203;",   # hourglass
    "done": "&#9989;",      # check mark
    "failed": "&#10060;",   # cross mark
}
STATUS_COLOR = {
    "pending": "#64748B",
    "running": "#38BDF8",
    "done": "#4ADE80",
    "failed": "#F87171",
}

DEFAULT_STAGES = [
    "Upload & Ingestion",
    "HIPAA De-Identification",
    "Schema Profiling",
    "Route Decision",
    "Packaging & Push",       # Kaggle route only
    "Kaggle Queue",
    "DP-SGD Training",        # or Adapter Fine-Tune on adapter route
    "Synthetic Generation",
    "Red-Team Privacy Audit",
    "Artifact Delivery",
]


def init_pipeline_stages(route: str = "kaggle") -> None:
    """(Re)initialize the pipeline checklist for the active route."""
    stages = list(DEFAULT_STAGES)
    if route == "adapter":
        # Rename training stage; drop Kaggle-specific stages.
        stages = [s for s in stages if s not in ("Packaging & Push", "Kaggle Queue")]
        stages[stages.index("DP-SGD Training")] = "Adapter DP Fine-Tune"
    st.session_state.pipeline_stages = [
        {"name": s, "status": "pending", "detail": "", "ts": None} for s in stages
    ]


def set_stage(name: str, status: str, detail: str = "") -> None:
    """Update one stage's status/detail by exact name."""
    stages = st.session_state.get("pipeline_stages")
    if not stages:
        return
    for s in stages:
        if s["name"] == name:
            s["status"] = status
            s["detail"] = detail
            s["ts"] = time.time() if status in ("done", "failed") else s["ts"]
            return
    # Unknown stage: append it so nothing is silently dropped.
    stages.append({"name": name, "status": status, "detail": detail,
                   "ts": time.time() if status in ("done", "failed") else None})


def render_pipeline_checklist(title: str = "Pipeline Execution Tracker") -> None:
    """Render the live pipeline checklist card."""
    stages = st.session_state.get("pipeline_stages")
    if not stages:
        return

    rows = []
    for i, s in enumerate(stages):
        icon = STATUS_ICON.get(s["status"], "&#9744;")
        color = STATUS_COLOR.get(s["status"], "#64748B")
        detail = f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:2px;">{s["detail"]}</div>' if s["detail"] else ""
        connector = '<div style="border-left:2px solid #2A3F60;height:14px;margin-left:11px;"></div>' if i < len(stages) - 1 else ""
        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:10px;">'
            f'<div style="font-size:1rem;line-height:1.2;">{icon}</div>'
            f'<div><div style="color:{color};font-size:0.85rem;font-weight:600;">{s["name"]}</div>{detail}</div>'
            f'</div>{connector}'
        )

    done = sum(1 for s in stages if s["status"] == "done")
    total = len(stages)
    pct = int(100 * done / max(total, 1))

    st.markdown(f"""
    <div class="synth-card" style="padding:18px 22px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div style="font-weight:700;color:#F8FAFC;font-size:0.95rem;">{title}</div>
        <div style="color:#38BDF8;font-weight:700;font-size:0.85rem;">{done}/{total} stages</div>
      </div>
      <div style="height:6px;background:#1E293B;border-radius:3px;margin-bottom:12px;">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#38BDF8,#4ADE80);border-radius:3px;"></div>
      </div>
      {''.join(rows)}
    </div>
    """, unsafe_allow_html=True)