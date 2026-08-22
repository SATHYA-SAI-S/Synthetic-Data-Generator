import os
import shutil
import sys

# Setup paths
input_project_path = "/kaggle/input/datasets/vishwad007v/advt-healthcare-privacy-codebase/ADVT"
working_project_path = "/kaggle/working/ADVT"

if os.path.exists(working_project_path):
    shutil.rmtree(working_project_path)
    
shutil.copytree(input_project_path, working_project_path)
os.chdir(working_project_path)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Install differential privacy library
!pip install -q opacus==1.6.0

print("✓ Workspace set up successfully at:", os.getcwd())



import os
import zipfile

os.chdir("/kaggle/working/ADVT")
unpacked_dir = "/kaggle/input/datasets/vishwad007v/advt-healthcare-privacy-codebase/ADVT/data/diabetes+130-us+hospitals+for+years+1999-2008"
os.makedirs("data", exist_ok=True)
zip_target_path = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"

with zipfile.ZipFile(zip_target_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    csv_file = os.path.join(unpacked_dir, "diabetic_data.csv")
    if os.path.exists(csv_file):
        zf.write(csv_file, arcname="diabetic_data.csv")
        print("✓ Single-file dataset zip created successfully!")
    else:
        print("❌ Error: diabetic_data.csv not found in unpacked directory.")





#cell - 3

import os

os.chdir("/kaggle/working/ADVT")
print("Applying hotfixes to prevent NaN propagation...")

# Hotfix A: dataset_profiler.py
profiler_path = "src/profiling/dataset_profiler.py"
with open(profiler_path, "r") as f:
    profiler_code = f.read()

if "numeric_fraction < 1.0" not in profiler_code:
    old_p = "if numeric_fraction >= config.cardinality.continuous_numeric_fraction:\n            return InferredDtype.CONTINUOUS"
    new_p = """if numeric_fraction >= config.cardinality.continuous_numeric_fraction:
            if numeric_fraction < 1.0:
                if n_unique <= config.cardinality.low_card_max:
                    return InferredDtype.CATEGORICAL_LOW
                return InferredDtype.CATEGORICAL_HIGH
            return InferredDtype.CONTINUOUS"""
    profiler_code = profiler_code.replace(old_p, new_p)
    with open(profiler_path, "w") as f:
        f.write(profiler_code)
    print("✓ Profiler hotfix applied.")
else:
    print("✓ Profiler hotfix already present.")

# Hotfix B: pipeline.py
pipeline_path = "src/preprocessing/pipeline.py"
with open(pipeline_path, "r") as f:
    pipeline_code = f.read()

if "pd.to_numeric(working_df[col], errors=\"coerce\")" not in pipeline_code:
    old_pipe = """                scaler = self._scaler_factory(col)
                encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)"""
    new_pipe = """                import pandas as pd
                coerced = pd.to_numeric(working_df[col], errors="coerce")
                if coerced.isna().sum() > working_df[col].isna().sum():
                    encoder = self._encoder_factory(col)
                    encoded = encoder.fit_transform(working_df[col])
                    self._encoders[col] = encoder
                    self._column_types[col] = "categorical"
                    encoded_parts.append(encoded)
                    valid_training_columns.append(col)
                    for i in range(encoder.output_dim):
                        self._encoded_col_names.append(f"{col}__enc{i}")
                    continue
                
                scaler = self._scaler_factory(col)
                encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)"""
    pipeline_code = pipeline_code.replace(old_pipe, new_pipe)
    with open(pipeline_path, "w") as f:
        f.write(pipeline_code)
    print("✓ Pipeline hotfix applied.")
else:
    print("✓ Pipeline hotfix already present.")




import os
import sys
import shutil
import subprocess
import zipfile

print("🚀 Initializing completely isolated workspace with unique cache namespaces...")

# ---------------------------------------------------------
# 1. UNIQUE NAMESPACES (Zero Collision Guarantee)
# ---------------------------------------------------------
RUN_TAG = "fresh_run_vishwa"
WORKING_DIR = f"/kaggle/working/ADVT_{RUN_TAG}"
OUTPUT_DIR = f"{WORKING_DIR}/outputs_{RUN_TAG}"
REGISTRY_DIR = f"{WORKING_DIR}/registry_{RUN_TAG}"
ZIP_ARCHIVE_OUT = f"/kaggle/working/vishwa_final_clean_archive"

# Clean up working dir if it previously existed
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)

input_project_path = "/kaggle/input/datasets/vishwad007v/advt-healthcare-privacy-codebase/ADVT"
shutil.copytree(input_project_path, WORKING_DIR)
os.chdir(WORKING_DIR)

sys.path.insert(0, WORKING_DIR)
sys.path.insert(0, os.path.join(WORKING_DIR, "src"))

# ---------------------------------------------------------
# 2. INSTALL OPACUS & PREPARE CLEAN DATASET ZIP
# ---------------------------------------------------------
!pip install -q opacus==1.6.0

unpacked_dir = "/kaggle/input/datasets/vishwad007v/advt-healthcare-privacy-codebase/ADVT/data/diabetes+130-us+hospitals+for+years+1999-2008"
os.makedirs(f"{WORKING_DIR}/data_{RUN_TAG}", exist_ok=True)
clean_zip_path = f"{WORKING_DIR}/data_{RUN_TAG}/diabetes_dataset_clean.zip"

with zipfile.ZipFile(clean_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    csv_file = os.path.join(unpacked_dir, "diabetic_data.csv")
    if os.path.exists(csv_file):
        zf.write(csv_file, arcname="diabetic_data.csv")
        print("✓ Dataset packaged into fresh isolated zip.")

# Also update default data path location in data/
os.makedirs("data", exist_ok=True)
shutil.copy(clean_zip_path, "data/diabetes+130-us+hospitals+for+years+1999-2008.zip")

# ---------------------------------------------------------
# 3. APPLY IN-MEMORY CODE HOTFIXES
# ---------------------------------------------------------
print("Applying strict Dtype, Routing, and AMP stability shields...")

# Hotfix A: dataset_profiler.py (Strict alphanumeric categorical routing)
profiler_path = "src/profiling/dataset_profiler.py"
with open(profiler_path, "r") as f:
    p_code = f.read()

if "numeric_fraction < 1.0" not in p_code:
    old_p = "if numeric_fraction >= config.cardinality.continuous_numeric_fraction:\n            return InferredDtype.CONTINUOUS"
    new_p = """if numeric_fraction >= config.cardinality.continuous_numeric_fraction:
            if numeric_fraction < 1.0:
                if n_unique <= config.cardinality.low_card_max:
                    return InferredDtype.CATEGORICAL_LOW
                return InferredDtype.CATEGORICAL_HIGH
            return InferredDtype.CONTINUOUS"""
    p_code = p_code.replace(old_p, new_p)
    with open(profiler_path, "w") as f:
        f.write(p_code)
    print("✓ Profiler hotfix applied.")

# Hotfix B: pipeline.py (Defense-in-depth dynamic rerouting)
pipeline_path = "src/preprocessing/pipeline.py"
with open(pipeline_path, "r") as f:
    pipe_code = f.read()

if "errors=\"coerce\"" not in pipe_code:
    old_pipe = """                scaler = self._scaler_factory(col)
                encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)"""
    new_pipe = """                import pandas as pd
                coerced = pd.to_numeric(working_df[col], errors=\"coerce\")
                if coerced.isna().sum() > working_df[col].isna().sum():
                    encoder = self._encoder_factory(col)
                    encoded = encoder.fit_transform(working_df[col])
                    self._encoders[col] = encoder
                    self._column_types[col] = \"categorical\"
                    encoded_parts.append(encoded)
                    valid_training_columns.append(col)
                    for i in range(encoder.output_dim):
                        self._encoded_col_names.append(f\"{col}__enc{i}\")
                    continue
                
                scaler = self._scaler_factory(col)
                encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)"""
    pipe_code = pipe_code.replace(old_pipe, new_pipe)
    with open(pipeline_path, "w") as f:
        f.write(pipe_code)
    print("✓ Pipeline hotfix applied.")

# Hotfix C: schema.py (Force Float32 & Lower Learning Rate to prevent gradient blowup)
schema_path = "src/config/schema.py"
if os.path.exists(schema_path):
    with open(schema_path, "r") as f:
        s_code = f.read()
    s_code = s_code.replace("use_amp: bool = True", "use_amp: bool = False")
    s_code = s_code.replace("1e-3", "1e-4")
    s_code = s_code.replace("0.001", "0.0001")
    with open(schema_path, "w") as f:
        f.write(s_code)
    print("✓ Config schema patched: AMP disabled (Float32 enforced), LR set to 1e-4.")

# ---------------------------------------------------------
# 4. EXECUTE ISOLATED GENERATION SWEEP
# ---------------------------------------------------------
env = os.environ.copy()
env["PYTHONPATH"] = f"{WORKING_DIR}:{env.get('PYTHONPATH', '')}"

print("\n🚀 Starting completely fresh end-to-end training and generation sweep...")
result = subprocess.run(
    ["python", "scripts/reproduce_end_to_end.py"],
    env=env,
    capture_output=True,
    text=True
)

print("--- STDOUT (Last 1200 chars) ---")
print(result.stdout[-1200:])

if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise RuntimeError("Generation sweep failed. Check log output above.")

print("\n✓ Sweep completed successfully!")

# ---------------------------------------------------------
# 5. PACKAGE INTO BRAND NEW ZIP ARCHIVE
# ---------------------------------------------------------
possible_output_dirs = [
    f"{WORKING_DIR}/outputs/sweep_results",
    f"{WORKING_DIR}/sweep_results_archive",
    f"{WORKING_DIR}/outputs"
]

target_dir = None
for d in possible_output_dirs:
    if os.path.exists(d) and len(os.listdir(d)) > 0:
        target_dir = d
        break

if target_dir:
    shutil.make_archive(ZIP_ARCHIVE_OUT, "zip", target_dir)
    print(f"\n📦 Brand-new clean archive created at: {ZIP_ARCHIVE_OUT}.zip")
    print("Download 'vishwa_final_clean_archive.zip' from your Kaggle output panel.")
else:
    print("⚠️ Could not locate output directory to zip.")