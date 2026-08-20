import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("clean_synthetic")

# Clinical domain bounds and rounding rules
RULES_DIABETES = {
    # Durations
    "time_in_hospital": {"type": "int", "min": 1, "max": 14},
    # Counts
    "num_lab_procedures": {"type": "int", "min": 1, "max": 132},
    "num_procedures": {"type": "int", "min": 0, "max": 6},
    "num_medications": {"type": "int", "min": 1, "max": 81},
    "number_diagnoses": {"type": "int", "min": 1, "max": 16},
    "number_inpatient": {"type": "int", "min": 0, "max": 21},
    "number_outpatient": {"type": "int", "min": 0, "max": 42},
    "number_emergency": {"type": "int", "min": 0, "max": 76},
    # Integer Categorical IDs
    "admission_type_id": {"type": "int", "min": 1, "max": 8},
    "discharge_disposition_id": {"type": "int", "min": 1, "max": 28},
    "admission_source_id": {"type": "int", "min": 1, "max": 25},
}

RULES_CDC_HEART = {
    "SleepHours": {"type": "int", "min": 1, "max": 24},
    "PhysicalHealthDays": {"type": "int", "min": 0, "max": 30},
    "MentalHealthDays": {"type": "int", "min": 0, "max": 30},
    "RemovedTeeth": {"type": "int", "min": 0, "max": 32},
    "HeightInMeters": {"type": "float", "decimals": 2, "min": 1.20, "max": 2.40},
    "WeightInKilograms": {"type": "float", "decimals": 1, "min": 30.0, "max": 280.0},
    "BMI": {"type": "float", "decimals": 1, "min": 12.0, "max": 75.0},
}

ALL_RULES = {**RULES_DIABETES, **RULES_CDC_HEART}

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies clinical bounding, rounding, and nullable integer casting to a DataFrame."""
    cleaned = df.copy()
    
    for col in cleaned.columns:
        if col in ALL_RULES:
            rule = ALL_RULES[col]
            series = pd.to_numeric(cleaned[col], errors="coerce")
            
            # Non-null mask
            valid_mask = series.notna()
            
            if rule["type"] == "int":
                # Clip bounds and round
                clipped = series.clip(lower=rule.get("min", 0), upper=rule.get("max", 999999))
                rounded = np.round(clipped)
                # Store back as nullable Int64 (preserves natural NaNs without floats)
                cleaned[col] = rounded.astype("Int64")
                
            elif rule["type"] == "float":
                decimals = rule.get("decimals", 2)
                clipped = series.clip(lower=rule.get("min", 0.0), upper=rule.get("max", 999999.0))
                rounded = np.round(clipped, decimals)
                cleaned[col] = rounded

    return cleaned

def clean_file(file_path: str) -> None:
    """Loads CSV, applies domain cleaning, and saves the cleaned file."""
    if not os.path.exists(file_path):
        log.warning(f"File not found: {file_path}")
        return

    log.info(f"--- Cleaning {file_path} ---")
    df = pd.read_csv(file_path, low_memory=False)
    initial_shape = df.shape
    
    cleaned_df = clean_dataframe(df)
    
    # Overwrite with cleaned CSV
    cleaned_df.to_csv(file_path, index=False)
    log.info(f"Successfully cleaned and saved {file_path} (Shape: {cleaned_df.shape})")
    
    # Print sample verification
    preview_cols = [c for c in ["time_in_hospital", "num_medications", "admission_type_id", "SleepHours", "BMI", "GeneralHealth", "readmitted"] if c in cleaned_df.columns]
    log.info(f"Cleaned preview for {os.path.basename(file_path)}:\n{cleaned_df[preview_cols].head(5)}")

def run_all_cleanups():
    target_files = [
        "vishwa_final_clean_archive/synthetic_eps_0.1.csv",
        "vishwa_final_clean_archive/synthetic_eps_1.0.csv",
        "vishwa_final_clean_archive/synthetic_eps_10.0.csv",
        "outputs/adapter_finetuning/synthetic_cdc_heart_adapted.csv",
        "outputs/adapter_finetuning/synthetic_adapted_small_cohort.csv"
    ]
    
    for f in target_files:
        clean_file(f)

if __name__ == "__main__":
    run_all_cleanups()
