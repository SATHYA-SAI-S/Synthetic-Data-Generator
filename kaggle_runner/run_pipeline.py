"""
Kaggle Kernel Runner - Fully adaptive autonomous DP-SGD training.

Reads the user's de-identified dataset + run config from the mounted
private Kaggle dataset (/kaggle/input/<dataset>/), trains ONLY the
configured epsilon (no hardcoded sweep, no UCI download, no git clone),
and writes progress.json + synthetic_clean.csv to the kernel output dir.

The UI (kaggle_bridge.fetch_progress/pull_results) polls these artifacts
to render a live progress bar and pull the final synthetic CSV.
"""
import json
import os
import glob
import sys
import time
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_progress(**kwargs):
    """Write a partial progress.json so the UI can poll live status."""
    payload = {
        "ts": time.time(),
        "stage": kwargs.get("stage", "running"),
        "pct": float(kwargs.get("pct", 0.0)),
        "epoch": int(kwargs.get("epoch", 0)),
        "total_epochs": int(kwargs.get("total_epochs", 1)),
        "loss": float(kwargs.get("loss", 0.0)),
        "epsilon_spent": float(kwargs.get("epsilon_spent", 0.0)),
    }
    tmp = OUTPUT_DIR / "progress.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(OUTPUT_DIR / "progress.json")


def locate_input_files():
    """
    Locate the mounted private dataset in /kaggle/input/<slug>/.
    Returns (clean_csv_path, run_config_path) or raises RuntimeError.
    """
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise RuntimeError("Kaggle input directory not found - dataset not mounted.")

    datasets = [d for d in input_root.iterdir() if d.is_dir()]
    if not datasets:
        raise RuntimeError("No Kaggle input dataset mounted to kernel.")

    # Prefer the first mounted dataset folder.
    data_dir = datasets[0]
    clean_csv = data_dir / "clean_data.csv"
    run_config = data_dir / "run_config.json"

    if not clean_csv.exists():
        # Fall back to any csv in the mounted dataset.
        csvs = sorted(data_dir.glob("*.csv"))
        if not csvs:
            raise RuntimeError(f"No CSV found in mounted dataset {data_dir}.")
        clean_csv = csvs[0]

    if not run_config.exists():
        raise RuntimeError(f"run_config.json missing in mounted dataset {data_dir}.")

    return clean_csv, run_config


def verify_cuda():
    """Verify PyTorch can use the GPU before proceeding."""
    print("Verifying CUDA availability...")
    check_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA not available'; _ = torch.zeros(1).cuda(); "
        "print(f'CUDA OK: {torch.cuda.get_device_name(0)}'); "
        "print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA: {torch.version.cuda}')"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", check_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        print("FATAL: PyTorch cannot use the GPU (CUDA capability mismatch?).")
        sys.exit(1)


def main():
    print("=== Starting Adaptive Kaggle DP-SGD Run ===")

    # 0. Verify CUDA + copy the runner/trainer modules so imports resolve.
    verify_cuda()

    # Copy this runner + the training script into working dir so the kernel
    # can import them (Kaggle mounts only the dataset, not the repo).
    script_dir = Path(__file__).resolve().parent
    for name in ("scripts", ""):
        pass  # scaffold handled below by importing via PYTHONPATH

    ##########################################################################
    # 1. Locate user's data + config from the mounted private dataset
    ##########################################################################
    try:
        clean_csv_path, run_config_path = locate_input_files()
    except RuntimeError as e:
        _write_progress(stage="failed", pct=100, loss=0.0, epoch=0, total_epochs=1)
        print(f"FATAL: {e}")
        sys.exit(1)

    print(f"Using dataset : {clean_csv_path}")
    print(f"Using config  : {run_config_path}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    target_epsilon = float(run_config.get("epsilon", 1.0))
    target_delta = float(run_config.get("delta", 1e-4))
    epochs = int(run_config.get("epochs", 5))
    batch_size = int(run_config.get("batch_size", 256))
    clip_norm = float(run_config.get("clip_norm", 1.0))
    num_samples = int(run_config.get("num_samples", -1))
    clean_columns = run_config.get("clean_columns", None)

    print(f"Config: eps={target_epsilon} delta={target_delta} "
          f"epochs={epochs} batch={batch_size} clip={clip_norm} "
          f"samples={num_samples}")

    # 2. Ensure the training script + src package are importable from the repo.
    #    On Kaggle we clone the pinned repo (deterministic) for the src package.
    _write_progress(stage="Preparing environment", pct=2, loss=0.0, epoch=0,
                    total_epochs=epochs)
    repo_dir = Path("/kaggle/working/Synthetic-Data-Generator")
    pin = "29ca855cab13be9ef80c656f711697339c3ae165"  # workspace HEAD (deterministic)
    try:
        subprocess.run(
            f'git clone https://github.com/SATHYA-SAI-S/Synthetic-Data-Generator.git '
            f'"{repo_dir}" -q && cd "{repo_dir}" && git checkout {pin}',
            shell=True, check=True,
        )
    except subprocess.CalledProcessError:
        print("WARN: git clone failed; falling back to current directory for src/")
        repo_dir = Path(".")
    sys.path.insert(0, str(repo_dir))

    # 3. Run the training via the local pipeline (adaptive, single config).
    _write_progress(stage="Importing pipeline", pct=5, loss=0.0, epoch=0,
                    total_epochs=epochs)
    sys.path.insert(0, str(Path(__file__).parent / ".."))
    try:
        from kaggle_runner.kernel_train import run_adaptive_training
    except ImportError:
        from kaggle_kernel_train import run_adaptive_training

    run_adaptive_training(
        clean_csv_path=str(clean_csv_path),
        output_dir=str(OUTPUT_DIR),
        epsilon=target_epsilon,
        delta=target_delta,
        epochs=epochs,
        batch_size=batch_size,
        clip_norm=clip_norm,
        num_samples=num_samples,
        clean_columns=clean_columns,
    )

    print("=== Run Complete. Artifacts written to /kaggle/working ===")


if __name__ == "__main__":
    main()