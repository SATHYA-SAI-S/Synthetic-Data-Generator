import os
import sys
import zipfile
import logging
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("evaluate_tstr")

def evaluate_tstr_benchmark(
    real_data_path: str = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    synthetic_data_path: str = "vishwa_final_clean_archive/synthetic_eps_1.0.csv",
    target_col: str = "readmitted",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Objective 6: Train on Synthetic, Test on Real (TSTR) Benchmark.
    
    Compares:
      1. TRTR (Train on Real, Test on Real) Baseline
      2. TSTR (Train on Synthetic, Test on Real) Utility Evaluation
    """
    log.info("=== Loading Real Dataset ===")
    if str(real_data_path).endswith(".zip"):
        with zipfile.ZipFile(real_data_path) as z:
            csv_candidates = [n for n in z.namelist() if n.endswith(".csv") and "mapping" not in n.lower()]
            csv_name = csv_candidates[0] if csv_candidates else z.namelist()[0]
            with z.open(csv_name) as f:
                real_df = pd.read_csv(f, low_memory=False)
    else:
        real_df = pd.read_csv(real_data_path, low_memory=False)

    log.info(f"Loaded Real DataFrame: {real_df.shape}")

    log.info(f"=== Loading Synthetic Dataset ({synthetic_data_path}) ===")
    synth_df = pd.read_csv(synthetic_data_path, low_memory=False)
    log.info(f"Loaded Synthetic DataFrame: {synth_df.shape}")

    # 1. Define Common Feature Columns and Target
    drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code"]
    feature_cols = [
        c for c in synth_df.columns 
        if c in real_df.columns and c != target_col and c not in drop_cols
    ]
    log.info(f"Using {len(feature_cols)} aligned feature columns for TSTR evaluation.")

    # 2. Extract Binary Target (<30 Days Readmission)
    def extract_target(df: pd.DataFrame) -> np.ndarray:
        s = df[target_col].astype(str).str.strip()
        return (s == "<30").astype(int).to_numpy()

    y_real = extract_target(real_df)
    X_real = real_df[feature_cols].copy()

    y_synth = extract_target(synth_df)
    X_synth = synth_df[feature_cols].copy()

    # 3. Categorical vs Numeric Feature Identification
    cat_features = []
    num_features = []
    for c in feature_cols:
        coerced = pd.to_numeric(X_real[c], errors="coerce")
        if (
            coerced.isna().sum() > X_real[c].isna().sum() 
            or X_real[c].dtype == "object" 
            or c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
        ):
            cat_features.append(c)
            X_real[c] = X_real[c].fillna("__null__").astype(str)
            X_synth[c] = X_synth[c].fillna("__null__").astype(str)
        else:
            num_features.append(c)
            X_real[c] = pd.to_numeric(X_real[c], errors="coerce")
            X_synth[c] = pd.to_numeric(X_synth[c], errors="coerce")

    log.info(f"Feature split: {len(num_features)} numeric features, {len(cat_features)} categorical features.")

    # 4. Split Real Data into Real Train & Real Test Holdout
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
    )
    log.info(f"Train/Test Splits: Real Train = {X_real_train.shape[0]}, Real Test Holdout = {X_real_test.shape[0]}, Synthetic Train = {X_synth.shape[0]}")

    # 5. Build Memory-Efficient Preprocessing Pipeline (Ordinal Encoding for High-Cardinality ICD-9)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__null__")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )

    # 6. Model 1: TRTR Baseline (Train on Real -> Test on Real Holdout)
    log.info("--- Training TRTR Baseline (Real -> Real) ---")
    trtr_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=random_state))
    ])
    trtr_model.fit(X_real_train, y_real_train)
    y_pred_trtr = trtr_model.predict(X_real_test)
    y_prob_trtr = trtr_model.predict_proba(X_real_test)[:, 1]

    auc_trtr = float(roc_auc_score(y_real_test, y_prob_trtr))
    f1_trtr = float(f1_score(y_real_test, y_pred_trtr, zero_division=0))
    acc_trtr = float(accuracy_score(y_real_test, y_pred_trtr))

    log.info(f"TRTR Baseline -> AUC-ROC: {auc_trtr:.4f} | F1-Score: {f1_trtr:.4f} | Accuracy: {acc_trtr:.4f}")

    # 7. Model 2: TSTR Benchmark (Train on Synthetic -> Test on Real Holdout)
    log.info("--- Training TSTR Benchmark (Synthetic -> Real) ---")
    tstr_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=random_state))
    ])
    tstr_model.fit(X_synth, y_synth)
    y_pred_tstr = tstr_model.predict(X_real_test)
    y_prob_tstr = tstr_model.predict_proba(X_real_test)[:, 1]

    auc_tstr = float(roc_auc_score(y_real_test, y_prob_tstr))
    f1_tstr = float(f1_score(y_real_test, y_pred_tstr, zero_division=0))
    acc_tstr = float(accuracy_score(y_real_test, y_pred_tstr))

    retention_auc = (auc_tstr / max(1e-6, auc_trtr)) * 100.0
    log.info(f"TSTR Benchmark -> AUC-ROC: {auc_tstr:.4f} | F1-Score: {f1_tstr:.4f} | Accuracy: {acc_tstr:.4f}")
    log.info(f"Downstream Utility Retention: {retention_auc:.2f}% of TRTR Baseline AUC-ROC")

    results = {
        "target_task": "30-Day Hospital Readmission (<30 days)",
        "classifier": "HistGradientBoostingClassifier(max_iter=150, lr=0.08)",
        "TRTR_baseline": {
            "auc_roc": round(auc_trtr, 4),
            "f1_score": round(f1_trtr, 4),
            "accuracy": round(acc_trtr, 4)
        },
        "TSTR_benchmark": {
            "auc_roc": round(auc_tstr, 4),
            "f1_score": round(f1_tstr, 4),
            "accuracy": round(acc_tstr, 4)
        },
        "utility_retention_pct": round(retention_auc, 2)
    }

    # Save benchmark json
    out_json = "outputs/tstr_evaluation_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved TSTR benchmark results to {out_json}")

    return results

if __name__ == "__main__":
    evaluate_tstr_benchmark()
