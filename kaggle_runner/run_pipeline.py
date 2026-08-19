import os
import shutil
import subprocess
import sys
import zipfile

def run_cmd(cmd):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        sys.exit(1)

def verify_cuda():
    """Verify PyTorch can actually use the GPU before proceeding."""
    print("Verifying CUDA availability...")
    check_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA not available'; "
        "print(f'CUDA OK: {torch.cuda.get_device_name(0)}'); "
        "print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA: {torch.version.cuda}')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", check_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print("=" * 70)
        print("FATAL: PyTorch cannot use the GPU!")
        print("This is usually caused by a CUDA capability mismatch.")
        print("The Tesla P100 (sm_60) requires PyTorch built with CUDA 11.8 or earlier.")
        print("=" * 70)
        sys.exit(1)

def main():
    print("Starting Kaggle Autonomous Run...")
    
    # 1. Clone the repository (force clean clone to avoid stale code)
    if os.path.exists("Synthetic-Data-Generator"):
        print("Removing existing Synthetic-Data-Generator directory...")
        shutil.rmtree("Synthetic-Data-Generator")
    run_cmd("git clone https://github.com/SATHYA-SAI-S/Synthetic-Data-Generator.git")
    
    os.chdir("Synthetic-Data-Generator")
    
    # 2. Install requirements
    print("Installing requirements...")
    run_cmd("pip install -r requirements.txt")
    
    # 3. Force reinstall PyTorch with CUDA 11.8 which supports sm_60 (Tesla P100)
    #    --force-reinstall is CRITICAL: without it, pip sees torch already installed
    #    and skips the downgrade, leaving an incompatible build.
    print("Installing PyTorch with P100-compatible CUDA 11.8 build...")
    run_cmd(
        "pip install --force-reinstall --no-cache-dir "
        "torch==2.3.1 torchvision==0.18.1 "
        "--index-url https://download.pytorch.org/whl/cu118 -q"
    )
    
    # 3b. Verify CUDA works before proceeding
    verify_cuda()
    
    # 4. Download the dataset
    print("Downloading dataset...")
    os.makedirs("data", exist_ok=True)
    dataset_url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    dataset_path = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    run_cmd(f"wget -q {dataset_url} -O {dataset_path}")
    
    # Verify the download is a valid zip
    if not zipfile.is_zipfile(dataset_path):
        print("ERROR: Downloaded file is not a valid zip archive!")
        sys.exit(1)
    print("Dataset downloaded and verified.")
    
    # 5. Run the end-to-end pipeline sweep
    print("Executing End-to-End Pipeline...")
    run_cmd("PYTHONPATH=. python scripts/reproduce_end_to_end.py")
    
    print("Run Complete! Check the Kaggle Output for results.")

if __name__ == "__main__":
    main()
