"""
SYNTHGUARD Screen 3 - Synthetic Generation & Sanitization Engine

Route-aware generation:
  - adapter route : local DP fine-tune of schema adapter (small-N)
  - kaggle route  : package -> push -> poll -> pull via KaggleBridge
Both routes finish with an automatic red-team privacy audit.
"""
import os
import shutil
import time
import streamlit as st
import pandas as pd
import numpy as np
from ui.data_loaders.load_synthetic_csv import generate_synthetic_from_real
from ui.components.pipeline_checklist import set_stage, render_pipeline_checklist


def _run_adapter_route(session_dir: str, clean_cols, target_size: int):
    """Local small-N route: DP fine-tune adapter + generate."""
    from src.diffusion.schema_adapter import adapter_route_run
    from src.adversary.attacker_engine import run_red_team

    set_stage("Adapter DP Fine-Tune", "running", "Registry lookup + DP fine-tuning...")
    raw_path = os.path.join(session_dir, "raw_upload.csv")
    raw_df = pd.read_csv(raw_path)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def cb(epoch, total, loss):
        pct = int(100 * epoch / max(total, 1))
        progress_bar.progress(min(pct, 99))
        status_text.markdown(f"**DP fine-tune epoch {epoch}/{total} - loss {loss:.4f}**")

    synth_df, info = adapter_route_run(
        raw_df[clean_cols],
        epsilon=float(st.session_state.get("target_epsilon", 1.0)),
        num_samples=target_size,
        epochs=int(st.session_state.get("epochs", 30)),
        batch_size=256,
        progress_cb=cb,
    )
    progress_bar.progress(100)
    status_text.markdown(f"**Adapter route complete** ({info['backbone_source']})")

    synth_path = os.path.join(session_dir, "synthetic_clean.csv")
    synth_df.to_csv(synth_path, index=False)
    st.session_state.synthetic_data_path = synth_path
    st.session_state.generation_info = info
    set_stage("Adapter DP Fine-Tune", "done",
              f"final loss {info.get('final_loss', float('nan')):.4f}")
    set_stage("Synthetic Generation", "done", f"{len(synth_df):,} rows generated")
    return raw_df, synth_df


def _run_kaggle_route(session_dir: str, clean_cols, target_size: int):
    """Kaggle route: package -> push -> poll -> pull."""
    from src.orchestration.kaggle_bridge import launch_job, KaggleErrorCategory

    creds = st.session_state.get("kaggle_credentials") or {}
    if not creds.get("username") or not creds.get("key"):
        st.error("Kaggle credentials missing. Add them in the sidebar settings before launching.")
        set_stage("Packaging & Push", "failed", "Missing Kaggle credentials")
        return None, None


    # Set env credentials for the kaggle CLI
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

    config = {
        "epsilon": float(st.session_state.get("target_epsilon", 1.0)),
        "delta": st.session_state.get("delta_choice", "1.0e-4"),
        "epochs": int(st.session_state.get("epochs", 5)),
        "batch_size": int(st.session_state.get("batch_size", 256)),
        "clip_norm": float(st.session_state.get("clip_norm", 1.0)),
        "num_samples": target_size,
        "clean_columns": clean_cols,
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_event(stage, detail):
        status_map = {"Packaging & Push": "running", "Kaggle Queue": "running"}
        if stage in status_map:
            set_stage(stage, status_map[stage], detail)

    try:
        set_stage("Packaging & Push", "running")
        bridge = launch_job(
            session_dir=session_dir,
            clean_csv_path=os.path.join(session_dir, "raw_upload.csv"),
            config=config,
            kaggle_user=creds["username"],
            on_event=on_event,
        )
        set_stage("Packaging & Push", "done", "Dataset + kernel pushed to Kaggle")
        set_stage("Kaggle Queue", "done", "Kernel accepted by Kaggle")
        set_stage("DP-SGD Training", "running", "Polling kernel status...")

        last_pct = 0
        final_job = bridge.watch(on_update=lambda job: None)
        prog = bridge.job.progress
        if prog:
            last_pct = int(prog.get("pct", 100))

        if bridge.job.status == "complete":
            progress_bar.progress(100)
            status_text.markdown("**Kaggle training complete - pulling artifacts...**")
            set_stage("DP-SGD Training", "done", f"progress {last_pct}%")
            results = bridge.pull_results()
            set_stage("Artifact Delivery", "done", str(results))
            # Locate synthetic output produced by the kernel
            synth_candidate = None
            for root, _, files in os.walk(results):
                for f in files:
                    if f.startswith("synthetic") and f.endswith(".csv"):
                        synth_candidate = os.path.join(root, f)
                        break
            raw_df = pd.read_csv(os.path.join(session_dir, "raw_upload.csv"))
            if synth_candidate:
                shutil.copy2(synth_candidate,
                             os.path.join(session_dir, "synthetic_clean.csv"))
                st.session_state.synthetic_data_path = os.path.join(
                    session_dir, "synthetic_clean.csv")
                set_stage("Synthetic Generation", "done",
                          f"Pulled {os.path.basename(synth_candidate)} from Kaggle")
                return raw_df, pd.read_csv(synth_candidate)
            # Kernel finished but no synthetic file -> backend issue
            set_stage("Synthetic Generation", "failed",
                      "Kernel completed but no synthetic CSV found in outputs")
            return raw_df, None
        else:
            err = bridge.job.error or {}
            set_stage("DP-SGD Training", "failed", err.get("message", "Unknown error"))
            category = err.get("category", "")
            if category == KaggleErrorCategory.QUOTA:
                st.warning("Weekly Kaggle GPU quota exhausted. Options: wait for reset, "
                           "switch to the Adapter route, or reduce epochs.")
            elif category == KaggleErrorCategory.AUTH:
                st.error("Kaggle authentication failed - re-enter credentials in settings.")
            else:
                st.error(f"Backend rework required: {err.get('message', 'see error ledger')}")
            return None, None
    except Exception as e:
        set_stage("DP-SGD Training", "failed", str(e))
        st.error(f"Kaggle job failed: {e}")
        return None, None


def _run_red_team_audit(session_dir: str, raw_df: pd.DataFrame, synth_df: pd.DataFrame):
    """Automatic red-team privacy audit after generation."""
    from src.adversary.attacker_engine import run_red_team
    set_stage("Red-Team Privacy Audit", "running", "Adaptive attacker escalating...")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(raw_df))
    cut = int(0.8 * len(raw_df))
    train_df, holdout_df = raw_df.iloc[idx[:cut]], raw_df.iloc[idx[cut:]]
    report = run_red_team(
        train_df, holdout_df, synth_df,
        epsilon_claimed=float(st.session_state.get("target_epsilon", 1.0)),
        report_path=os.path.join(session_dir, "attack_report.json"),
    )
    icon = "&#9989;" if report.verdict == "PRIVACY_CERTIFIED" else "&#9888;&#65039;"
    set_stage("Red-Team Privacy Audit", "done",
              f"{icon} {report.verdict} (worst attack success {report.worst_success_rate:.2%})")
    st.session_state.attack_report = report.to_dict()


def render_screen3():
    """Render Reverse Diffusion Sampling and Clinical Domain Guardrails Screen."""
    ds_name = st.session_state.get("dataset_name", "Clinical Cohort")
    n_rows = st.session_state.get("num_rows", 0)
    session_id = st.session_state.get("session_id", "default")
    session_dir = f"sessions/{session_id}"
    rd = st.session_state.get("route_decision") or {}
    route = rd.get("route", "adapter")

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
      <h2 style="color: #F8FAFC; margin: 0;">Screen 3: Synthetic Generation & Sanitization Engine</h2>
      <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px;">
        Active Dataset: <b style="color: #38BDF8;">{ds_name}</b> &bull; Exact 1:1 cohort scale ({n_rows:,} records)
        &bull; Route: <b style="color: {'#38BDF8' if route == 'kaggle' else '#A78BFA'};">
        {'Kaggle GPU Training' if route == 'kaggle' else 'Local Adapter Fine-Tune'}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### Sampling Parameters")
        st.markdown(f"""
        <div class="synth-card" style="margin-bottom: 16px; padding: 14px 18px; border-left: 4px solid #38BDF8;">
          <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Target Cohort Size (1:1 Match)</div>
          <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{n_rows:,} <span style="font-size: 0.85rem; font-weight: 500; color: #4ADE80;">records</span></div>
          <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Exact dimension matching from ingested <code>{ds_name}</code> (Zero row cap).</div>
        </div>
        """, unsafe_allow_html=True)

        timesteps = st.slider("Reverse Diffusion Timesteps (DDPM / DDIM)", 50, 1000, 1000, step=50)

        st.markdown("### Domain Guardrail Filters (Enforced)")
        st.checkbox("Round continuous duration/age to valid clinical integer units", value=True, disabled=True)
        st.checkbox("Clip negative counts to zero (non-negative labs/procedures)", value=True, disabled=True)
        st.checkbox("Cast categorical ID codes to integer types", value=True, disabled=True)
        st.checkbox("Restore natural clinical missingness flags", value=True, disabled=True)

        if route == "kaggle":
            creds = st.session_state.get("kaggle_credentials", {})
            if creds.get("username") and creds.get("key"):
                st.markdown("<div style='color: #4ADE80; font-size: 0.9rem; margin-bottom: 8px;'>&#10003; Credentials loaded from Automation Settings.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: #F87171; font-size: 0.9rem; margin-bottom: 8px;'>&#10007; Missing Kaggle credentials in Automation Settings.</div>", unsafe_allow_html=True)
            
            st.checkbox("I understand the de-identified dataset will be uploaded to my private Kaggle dataset for GPU training.", key="kaggle_consent")
            
        if st.button("Trigger Full Generation & Sanitization", type="primary", width='stretch'):
            raw_path = os.path.join(session_dir, "raw_upload.csv")
            if not os.path.exists(raw_path):
                st.error("No uploaded dataset found in this session.")
                return
            raw_df = pd.read_csv(raw_path)
            dropped = st.session_state.get("hipaa_dropped", [])
            clean_cols = [c for c in raw_df.columns if c not in dropped]
            target_size = len(raw_df)

            if route == "kaggle":
                if not st.session_state.get("kaggle_consent", False):
                    st.error("Consent required to start the Kaggle training job.")
                    return
                got_raw, synth_df = _run_kaggle_route(session_dir, clean_cols, target_size)
            else:
                got_raw, synth_df = _run_adapter_route(session_dir, clean_cols, target_size)

            if synth_df is not None and len(synth_df) > 0:
                try:
                    _run_red_team_audit(session_dir, got_raw, synth_df)
                except Exception as e:
                    set_stage("Red-Team Privacy Audit", "failed", str(e))
                st.session_state.generation_complete = True
                st.session_state.sanitization_complete = True
                st.session_state.step = 4
                st.rerun()

    with col2:
        st.markdown("### Post-Processing Execution Summary")
        st.markdown(f"""
        <div class="synth-card">
          <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem; margin-bottom: 8px;">
            Active Guardrail Rules for <code>{ds_name}</code>
          </div>
          <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
            <div>&bull; <b>Cohort Scale:</b> Generates exactly {n_rows:,} rows without artificial caps.</div>
            <div>&bull; <b>Integer Casting:</b> Continuous float drift rounded to integer codes.</div>
            <div>&bull; <b>Range Clamping:</b> Values clipped strictly to [min, max] domain boundaries.</div>
            <div>&bull; <b>Non-Negativity:</b> Counts, lab visits, and durations strictly non-negative.</div>
            <div>&bull; <b>Red-Team Audit:</b> Adaptive attacker runs automatically post-generation.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    render_pipeline_checklist()
    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("Open OP Dashboard ->", type="primary", width='stretch'):
            st.session_state.step = 4
            st.rerun()