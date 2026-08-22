"""
Kaggle Bridge - Automated push / poll / pull orchestration with error taxonomy.

Flow:
  1. package()      - zip de-identified clean CSV + run_config.json
  2. push_dataset() - create/version a PRIVATE Kaggle dataset
  3. push_kernel()  - generate kernel-metadata.json dynamically & push kernel
  4. poll_status()  - poll kernel status every POLL_INTERVAL seconds
  5. pull_results() - download outputs (synthetic CSV, progress.json, logs)

Error taxonomy (see kaggle_error_handler.py):
  - Quota/limit errors  -> auto-handled (retry scheduling, route downgrade offer)
  - Transient errors    -> exponential backoff auto-retry
  - Backend errors      -> classified as "backend_rework", surfaced to UI

The Kaggle kernel writes a progress.json to its output directory every few
steps: {"epoch": int, "total_epochs": int, "pct": float, "loss": float,
"epsilon_spent": float, "stage": str}. The UI polls this file via
`kaggle kernels output` to render a live progress bar.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 15
STUCK_WATCHDOG_SECONDS = 20 * 60          # no progress update for 20 min -> stuck
MAX_TRANSIENT_RETRIES = 3


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class KaggleErrorCategory:
    QUOTA = "quota"                # weekly 30h GPU limit etc. -> auto-handle
    TIMEOUT = "kernel_timeout"     # 9h/12h kernel limit -> auto-chunk/resume
    TRANSIENT = "transient"        # 429 rate limit, network -> backoff retry
    AUTH = "auth"                  # 401/403 -> prompt user for credentials
    SIZE_LIMIT = "size_limit"      # dataset too large -> shard or local route
    BACKEND_REWORK = "backend_rework"  # NaN loss, OOM, CUDA mismatch -> no retry


@dataclass
class KaggleJobError(Exception):
    category: str
    message: str
    raw: str = ""
    retriable: bool = False

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.category}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "message": self.message,
                "raw": self.raw[-4000:], "retriable": self.retriable}


def classify_kaggle_error(text: str) -> KaggleJobError:
    """Map raw Kaggle CLI/API error text to the taxonomy."""
    t = text.lower()
    if "401" in t or "403" in t or "unauthorized" in t or "forbidden" in t or "invalid api" in t:
        return KaggleJobError(KaggleErrorCategory.AUTH,
                              "Kaggle authentication failed. Re-enter credentials.", text)
    if "429" in t or "too many requests" in t or "rate limit" in t:
        return KaggleJobError(KaggleErrorCategory.TRANSIENT,
                              "Kaggle rate limit hit.", text, retriable=True)
    if "quota" in t or "30 hours" in t or "weekly" in t and "limit" in t \
            or "gpu quota" in t or "exceeded your" in t:
        return KaggleJobError(KaggleErrorCategory.QUOTA,
                              "Weekly GPU quota exhausted.", text)
    if "timeout" in t and ("kernel" in t or "9 hour" in t or "12 hour" in t):
        return KaggleJobError(KaggleErrorCategory.TIMEOUT,
                              "Kernel exceeded max runtime.", text)
    if "too large" in t or "max size" in t or "size limit" in t or "exceeds" in t and "gb" in t:
        return KaggleJobError(KaggleErrorCategory.SIZE_LIMIT,
                              "Dataset exceeds Kaggle size limits.", text)
    if "nan" in t or "out of memory" in t or "cuda" in t or "traceback" in t \
            or "assertionerror" in t or "runtimeerror" in t:
        return KaggleJobError(KaggleErrorCategory.BACKEND_REWORK,
                              "Training crashed - backend rework required.", text)
    return KaggleJobError(KaggleErrorCategory.BACKEND_REWORK,
                          "Unclassified Kaggle failure.", text)


def append_error_ledger(session_dir: str, err: KaggleJobError) -> None:
    """Persist every error so nothing is silently swallowed."""
    ledger = Path(session_dir) / "error_ledger.json"
    entries = []
    if ledger.exists():
        try:
            entries = json.loads(ledger.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries.append({"ts": time.time(), **err.to_dict()})
    tmp = ledger.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(ledger)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

@dataclass
class KaggleJob:
    session_dir: str
    slug_dataset: str
    slug_kernel: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "idle"           # idle|packaging|pushing|queued|running|complete|error
    progress: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run(cmd: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding='utf-8', timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


class KaggleBridge:
    """Automated Kaggle dataset/kernel lifecycle manager."""

    def __init__(self, job: KaggleJob, on_event: Optional[Callable[[str, str], None]] = None):
        """
        on_event(stage_name, detail) lets the UI checklist react to bridge events.
        """
        self.job = job
        self.on_event = on_event or (lambda s, d: None)

    # -- packaging ----------------------------------------------------------

    def package(self, clean_csv_path: str) -> Path:
        """Zip the de-identified clean CSV + run_config.json for upload."""
        self.job.status = "packaging"
        self.on_event("Packaging & Push", "Packaging de-identified dataset...")
        pkg_dir = Path(self.job.session_dir) / "kaggle_package"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(clean_csv_path, pkg_dir / "clean_data.csv")
        (pkg_dir / "run_config.json").write_text(
            json.dumps(self.job.config, indent=2), encoding="utf-8")

        zip_path = pkg_dir / "advt_session_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pkg_dir / "clean_data.csv", "clean_data.csv")
            z.write(pkg_dir / "run_config.json", "run_config.json")
        log.info("Packaged %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
        return zip_path

    # -- pushing ------------------------------------------------------------

    def _require_kaggle_cli(self) -> None:
        proc = _run("kaggle --version")
        if proc.returncode != 0:
            raise KaggleJobError(
                KaggleErrorCategory.BACKEND_REWORK,
                "kaggle CLI not installed. Run: pip install kaggle")

    def push_dataset(self, zip_path: Path) -> None:
        self._require_kaggle_cli()
        self.job.status = "pushing"
        folder = zip_path.parent
        meta = {
            "title": f"advt-session-{Path(self.job.session_dir).name}",
            "id": self.job.slug_dataset,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
            proc = _run(f'kaggle datasets create -p "{folder}" --dir-mode zip')
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                self.on_event("Packaging & Push", "Dataset pushed to Kaggle (private).")
                return
            err = classify_kaggle_error(out)
            if err.category == KaggleErrorCategory.TRANSIENT and attempt < MAX_TRANSIENT_RETRIES:
                wait = 2 ** attempt * 5
                log.warning("Transient push failure, retry %d/%d in %ds", attempt,
                            MAX_TRANSIENT_RETRIES, wait)
                time.sleep(wait)
                continue
            # Dataset may already exist -> try version upgrade instead
            if "already exists" in out.lower():
                proc_v = _run(f'kaggle datasets version -p "{folder}" -m "auto update"')
                if proc_v.returncode == 0:
                    return
                err = classify_kaggle_error((proc_v.stdout or "") + (proc_v.stderr or ""))
            append_error_ledger(self.job.session_dir, err)
            self.job.error = err.to_dict()
            self.job.status = "error"
            raise err

    def push_kernel(self, runner_script_path: str) -> None:
        """Generate kernel-metadata.json dynamically and push the kernel."""
        self._require_kaggle_cli()
        folder = Path(self.job.session_dir) / "kaggle_package"
        meta = {
            "id": self.job.slug_kernel,
            "title": Path(self.job.slug_kernel).name.replace("-", " "),
            "code_file": Path(runner_script_path).name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [self.job.slug_dataset],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
        shutil.copy2(runner_script_path, folder / Path(runner_script_path).name)

        proc = _run(f'kaggle kernels push -p "{folder}"')
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            err = classify_kaggle_error(out)
            append_error_ledger(self.job.session_dir, err)
            self.job.error = err.to_dict()
            self.job.status = "error"
            raise err
        self.on_event("Kaggle Queue", f"Kernel pushed: {self.job.slug_kernel}")

    # -- polling --------------------------------------------------------------

    def poll_once(self) -> str:
        """One status poll. Returns kernel status string."""
        proc = _run(f"kaggle kernels status {self.job.slug_kernel}")
        out = ((proc.stdout or "") + (proc.stderr or "")).strip().lower()
        if proc.returncode != 0 and "complete" not in out:
            err = classify_kaggle_error(out)
            if err.category in (KaggleErrorCategory.TRANSIENT,):
                return self.job.status  # ignore transient poll hiccups
            append_error_ledger(self.job.session_dir, err)
            self.job.error = err.to_dict()
            self.job.status = "error"
            raise err
        for state in ("complete", "error", "cancelacknowledged", "running", "queued"):
            if state in out:
                mapped = {"cancelacknowledged": "cancelled"}.get(state, state)
                self.job.status = mapped
                return mapped
        return self.job.status

    def fetch_progress(self) -> Dict[str, Any]:
        """Pull latest output files and read progress.json written by the kernel."""
        out_dir = Path(self.job.session_dir) / "kaggle_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run(f'kaggle kernels output {self.job.slug_kernel} -p "{out_dir}"')
        pj = out_dir / "progress.json"
        if pj.exists():
            try:
                self.job.progress = json.loads(pj.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self.job.progress

    def watch(self,
              on_update: Optional[Callable[[KaggleJob], None]] = None,
              stop_check: Optional[Callable[[], bool]] = None) -> KaggleJob:
        """
        Blocking watch loop: polls status + progress until complete/error.
        `on_update(job)` is invoked after each poll (UI refresh hook).
        `stop_check()` allows the caller to cancel the loop.
        """
        last_progress_ts = time.time()
        while True:
            if stop_check and stop_check():
                log.info("Watch loop cancelled by caller.")
                return self.job
            status = self.poll_once()
            prog = self.fetch_progress()
            if prog.get("ts"):
                last_progress_ts = float(prog["ts"])
            elif status == "running" and time.time() - last_progress_ts > STUCK_WATCHDOG_SECONDS:
                err = KaggleJobError(KaggleErrorCategory.TIMEOUT,
                                     "Kernel stuck: no progress for 20 minutes.")
                append_error_ledger(self.job.session_dir, err)
                self.job.error = err.to_dict()
                self.job.status = "error"
                _run(f"kaggle kernels status {self.job.slug_kernel}")
                raise err
            if on_update:
                on_update(self.job)
            if status in ("complete", "error", "cancelled"):
                return self.job
            time.sleep(POLL_INTERVAL_SECONDS)

    # -- pulling ---------------------------------------------------------------

    def pull_results(self) -> Path:
        """Download final artifacts into sessions/<id>/kaggle_results/."""
        results = Path(self.job.session_dir) / "kaggle_results"
        results.mkdir(parents=True, exist_ok=True)
        proc = _run(f'kaggle kernels output {self.job.slug_kernel} -p "{results}"')
        if proc.returncode != 0:
            err = classify_kaggle_error((proc.stdout or "") + (proc.stderr or ""))
            append_error_ledger(self.job.session_dir, err)
            raise err
        self.on_event("Artifact Delivery",
                      f"Artifacts downloaded to {results}")
        return results


def launch_job(session_dir: str,
               clean_csv_path: str,
               config: Dict[str, Any],
               kaggle_user: str,
               runner_script_path: str = "kaggle_runner/run_pipeline.py",
               on_event: Optional[Callable[[str, str], None]] = None) -> KaggleBridge:
    """Convenience: package + push dataset + push kernel in one call."""
    stamp = int(time.time())
    job = KaggleJob(
        session_dir=session_dir,
        slug_dataset=f"{kaggle_user}/advt-session-{stamp}",
        slug_kernel=f"{kaggle_user}/advt-dp-sgd-{stamp}",
        config=config,
    )
    bridge = KaggleBridge(job, on_event=on_event)
    zip_path = bridge.package(clean_csv_path)
    bridge.push_dataset(zip_path)
    bridge.push_kernel(runner_script_path)
    return bridge


if __name__ == "__main__":  # pragma: no cover - smoke test of classifier
    samples = [
        "403 Forbidden - You must accept the competition rules",
        "429 Too Many Requests",
        "You have exhausted your weekly GPU quota (30 hours)",
        "Kernel timed out after 9 hours",
        "Traceback ... RuntimeError: CUDA error",
    ]
    for s in samples:
        print(classify_kaggle_error(s))