import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        sys.exit(1)

def main():
    print("Starting Kaggle Autonomous Run...")
    
    # 1. Clone the repository
    if not os.path.exists("Synthetic-Data-Generator"):
        run_cmd("git clone https://github.com/SATHYA-SAI-S/Synthetic-Data-Generator.git")
    
    os.chdir("Synthetic-Data-Generator")
    
    # 2. Install requirements
    print("Installing requirements...")
    run_cmd("pip install -r requirements.txt")
    
    # 3. Download the Kaggle dataset if needed, but it's already in the repo's data folder!
    # Wait, the `data/diabetes+130-us+hospitals+for+years+1999-2008.zip` is in the repo.
    # So we can just run the end-to-end script!
    
    # 4. Run the end-to-end pipeline sweep
    print("Executing End-to-End Pipeline...")
    run_cmd("python scripts/reproduce_end_to_end.py")
    
    print("Run Complete! Check the Kaggle Output for results.")

if __name__ == "__main__":
    main()
