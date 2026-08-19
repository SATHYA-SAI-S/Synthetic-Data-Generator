"""
setup_check.py — Phase 2 Environment Verification Script

Standalone script with zero dependency on the rest of this codebase.
Verifies: Python version, GPU availability, Opacus DP-SGD smoke test.
Writes a structured pass/fail report to stdout and optionally to a JSON file.

Usage:
    python setup_check.py
    python setup_check.py --report-file setup_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("setup_check")


# ── Result data structure ──────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    error: Optional[str] = None


@dataclass
class SetupReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=platform.platform)
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ── Individual checks ──────────────────────────────────────────────────────────

def check_python_version(report: SetupReport, min_major: int = 3, min_minor: int = 10) -> None:
    """Verify Python version meets minimum requirement."""
    major, minor = sys.version_info.major, sys.version_info.minor
    passed = (major, minor) >= (min_major, min_minor)
    report.checks.append(CheckResult(
        name="python_version",
        passed=passed,
        detail=f"Python {major}.{minor} ({'OK' if passed else f'FAIL: need >={min_major}.{min_minor}'})",
    ))


def check_core_imports(report: SetupReport) -> None:
    """Verify all required packages can be imported."""
    packages = [
        ("torch", "torch"),
        ("opacus", "opacus"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("pydantic", "pydantic"),
        ("scipy", "scipy"),
        ("joblib", "joblib"),
        ("pyarrow", "pyarrow"),
    ]
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            report.checks.append(CheckResult(
                name=f"import_{module_name}",
                passed=True,
                detail=f"{display_name}=={version} importable",
            ))
        except ImportError as exc:
            report.checks.append(CheckResult(
                name=f"import_{module_name}",
                passed=False,
                detail=f"{display_name} MISSING",
                error=str(exc),
            ))


def check_gpu(report: SetupReport) -> None:
    """Check CUDA GPU availability and VRAM."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            detail = (
                f"CUDA available | devices={device_count} | "
                f"device0={device_name} | VRAM={vram_gb:.1f} GB"
            )
            passed = True
        else:
            detail = "CUDA not available — CPU only. Training will be extremely slow."
            passed = True  # Warn but don't hard-fail; CPU training is technically possible

        report.checks.append(CheckResult(
            name="gpu_cuda",
            passed=passed,
            detail=detail,
        ))
    except Exception as exc:
        report.checks.append(CheckResult(
            name="gpu_cuda",
            passed=False,
            detail="GPU check failed with exception",
            error=traceback.format_exc(),
        ))


def check_opacus_dpsgd_smoke(report: SetupReport) -> None:
    """
    Run a minimal DP-SGD training step to confirm Opacus is functional.

    Creates a tiny linear model, wraps it with Opacus PrivacyEngine,
    runs one forward-backward pass, and verifies privacy accounting.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from opacus import PrivacyEngine

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Tiny synthetic data --
        N, D, C = 64, 8, 2  # 64 samples, 8 features, 2 classes
        X = torch.randn(N, D)
        y = torch.randint(0, C, (N,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        # -- Tiny model (moved to device BEFORE make_private) --
        model = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, C)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # -- Wrap with Opacus PrivacyEngine --
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # -- One training step --
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            break  # Only one step needed for smoke test

        # -- Verify privacy accountant --
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        assert epsilon > 0, "Epsilon must be positive after one DP step"

        report.checks.append(CheckResult(
            name="opacus_dpsgd_smoke",
            passed=True,
            detail=(
                f"DP-SGD smoke test PASSED | "
                f"epsilon={epsilon:.4f} at delta=1e-5 after 1 step"
            ),
        ))

    except Exception:
        report.checks.append(CheckResult(
            name="opacus_dpsgd_smoke",
            passed=False,
            detail="DP-SGD smoke test FAILED",
            error=traceback.format_exc(),
        ))


def check_pydantic_v2(report: SetupReport) -> None:
    """Verify Pydantic V2 API is available (not V1 compatibility shim)."""
    try:
        from pydantic import BaseModel
        from pydantic import __version__ as pydantic_version

        major = int(pydantic_version.split(".")[0])
        if major < 2:
            report.checks.append(CheckResult(
                name="pydantic_v2",
                passed=False,
                detail=f"Pydantic V1 detected ({pydantic_version}); V2 required",
            ))
            return

        # Test V2-specific feature: model_config
        class _TestModel(BaseModel):
            model_config = {"frozen": True}
            value: int = 42

        instance = _TestModel()
        assert instance.value == 42

        report.checks.append(CheckResult(
            name="pydantic_v2",
            passed=True,
            detail=f"Pydantic V2 ({pydantic_version}) functional",
        ))
    except Exception:
        report.checks.append(CheckResult(
            name="pydantic_v2",
            passed=False,
            detail="Pydantic V2 check failed",
            error=traceback.format_exc(),
        ))


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(report: SetupReport) -> None:
    """Print a human-readable summary to stdout."""
    SEP = "=" * 60
    log.info(SEP)
    log.info("ENVIRONMENT SETUP CHECK REPORT")
    log.info(f"Timestamp : {report.timestamp}")
    log.info(f"Python    : {report.python_version}")
    log.info(f"Platform  : {report.platform}")
    log.info(SEP)

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        log.info(f"[{status}] {check.name}: {check.detail}")
        if check.error:
            for line in check.error.strip().splitlines():
                log.error(f"       {line}")

    log.info(SEP)
    overall = "ALL CHECKS PASSED" if report.all_passed else "SOME CHECKS FAILED — see above"
    log.info(f"OVERALL: {overall}")
    log.info(SEP)


def write_json_report(report: SetupReport, path: str) -> None:
    """Write the report as JSON to the given file path."""
    data = asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"Report written to: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 2 environment setup")
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    report = SetupReport()

    check_python_version(report)
    check_core_imports(report)
    check_gpu(report)
    check_pydantic_v2(report)
    check_opacus_dpsgd_smoke(report)

    print_report(report)

    if args.report_file:
        write_json_report(report, args.report_file)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
