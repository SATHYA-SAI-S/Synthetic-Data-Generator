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

    # Recursively search for run_config.json anywhere in /kaggle/input
    configs = list(input_root.rglob("run_config.json"))
    if not configs:
        # Print all files for debugging in the Kaggle logs
        all_files = list(input_root.rglob("*"))
        print(f"DEBUG: /kaggle/input contents: {all_files}")
        raise RuntimeError("run_config.json missing in all mounted datasets. Mount may have failed.")
        
    run_config = configs[0]
    data_dir = run_config.parent
    clean_csv = data_dir / "clean_data.csv"
    
    if not clean_csv.exists():
        csvs = sorted(data_dir.rglob("*.csv"))
        if not csvs:
            raise RuntimeError(f"No CSV found near run_config.json in {data_dir}.")
        clean_csv = csvs[0]

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

    # 0. Install compatible PyTorch for P100 (sm_60) before verifying CUDA
    print("Installing PyTorch with P100-compatible CUDA 11.8 build...")
    subprocess.run(
        "pip install --force-reinstall --no-cache-dir "
        "torch==2.3.1 torchvision==0.18.1 "
        "--index-url https://download.pytorch.org/whl/cu118 -q",
        shell=True, check=False
    )
    
    # 0b. Verify CUDA
    verify_cuda()

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
    _write_progress(stage="Preparing environment", pct=2, loss=0.0, epoch=0,
                    total_epochs=epochs)
    repo_dir = Path("/kaggle/working/Synthetic-Data-Generator")
    try:
        subprocess.run(
            f'git clone https://github.com/SATHYA-SAI-S/Synthetic-Data-Generator.git "{repo_dir}" -q',
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