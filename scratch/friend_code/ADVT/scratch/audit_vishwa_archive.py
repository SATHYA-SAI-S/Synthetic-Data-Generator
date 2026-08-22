import zipfile, os, json
import pandas as pd
import numpy as np
from scipy import stats

# 1. Load Real Data
with zipfile.ZipFile('data/diabetes+130-us+hospitals+for+years+1999-2008.zip') as z:
    with z.open('diabetic_data.csv') as f:
        real_df = pd.read_csv(f)

print("=== 1. DATASET SHAPE & COLUMN INTEGRITY ===")
print(f"Real Data Shape: {real_df.shape}")

archive_dir = 'vishwa_final_clean_archive'
synth_files = {
    'eps_0.1': os.path.join(archive_dir, 'synthetic_eps_0.1.csv'),
    'eps_1.0': os.path.join(archive_dir, 'synthetic_eps_1.0.csv'),
    'eps_10.0': os.path.join(archive_dir, 'synthetic_eps_10.0.csv'),
}

synth_dfs = {}
for k, path in synth_files.items():
    df = pd.read_csv(path, low_memory=False)
    synth_dfs[k] = df
    total_cells = df.shape[0] * df.shape[1]
    null_count = df.isna().sum().sum()
    print(f"{k} Shape: {df.shape} | Null Cells: {null_count} ({null_count / total_cells * 100:.2f}%)")

# Check HIPAA identifiers
hipaa_identifiers = ['encounter_id', 'patient_nbr', 'patient_id', 'mrn', 'ssn']
for name, df in synth_dfs.items():
    present_hipaa = [c for c in hipaa_identifiers if c in df.columns]
    print(f"{name} HIPAA direct identifiers found: {present_hipaa}")

dropped_cols = [c for c in real_df.columns if c not in synth_dfs['eps_1.0'].columns]
print(f"Dropped columns ({len(dropped_cols)}): {dropped_cols}")
print(f"Preserved columns count: {len(synth_dfs['eps_1.0'].columns)}")

# 2. Demographic and Categorical Distribution Analysis (TVD)
print("\n=== 2. CATEGORICAL DISTRIBUTIONS & TVD ===")
cat_cols = ['gender', 'race', 'age', 'readmitted', 'change', 'diabetesMed']

for col in cat_cols:
    print(f"\n--- Distribution for {col} ---")
    real_counts = real_df[col].value_counts(normalize=True, dropna=False).to_dict()
    df_compare = pd.DataFrame({"Real": real_counts})
    for k, df in synth_dfs.items():
        synth_counts = df[col].value_counts(normalize=True, dropna=False).to_dict()
        df_compare[k] = pd.Series(synth_counts)
    
    # Fill NaN with 0 for comparison
    df_compare = df_compare.fillna(0.0)
    print((df_compare * 100).round(2))
    
    # Compute TVD (Total Variation Distance = 0.5 * sum |P - Q|)
    for k in synth_dfs.keys():
        tvd = 0.5 * np.sum(np.abs(df_compare['Real'].values - df_compare[k].values))
        print(f"  TVD(Real, {k}) = {tvd:.4f}")

# 3. Continuous Features Distribution Analysis (KS Test & Moments)
print("\n=== 3. CONTINUOUS FEATURE DISTRIBUTIONS (KS Test & Moments) ===")
num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses']

for col in num_cols:
    print(f"\n--- Metric Summary for {col} ---")
    real_series = pd.to_numeric(real_df[col], errors='coerce').dropna()
    metrics = {
        "Real": {
            "mean": real_series.mean(),
            "std": real_series.std(),
            "median": real_series.median(),
            "min": real_series.min(),
            "max": real_series.max()
        }
    }
    for k, df in synth_dfs.items():
        s_series = pd.to_numeric(df[col], errors='coerce').dropna()
        ks_stat, p_val = stats.ks_2samp(real_series, s_series)
        metrics[k] = {
            "mean": s_series.mean(),
            "std": s_series.std(),
            "median": s_series.median(),
            "min": s_series.min(),
            "max": s_series.max(),
            "ks_stat": ks_stat
        }
    print(pd.DataFrame(metrics).round(3))

# 4. Alphanumeric Diagnosis Code Robustness (diag_1, diag_2, diag_3)
print("\n=== 4. ALPHANUMERIC DIAGNOSIS CODES ROBUSTNESS ===")
diag_cols = ['diag_1', 'diag_2', 'diag_3']
for col in diag_cols:
    print(f"\n--- Diagnosis Column: {col} ---")
    real_v_codes = real_df[col].dropna().astype(str).apply(lambda x: x.startswith(('V', 'E', '?'))).sum()
    print(f"Real alphanumeric ('V', 'E', '?') count in {col}: {real_v_codes} / {len(real_df[col].dropna())} ({real_v_codes/len(real_df[col].dropna())*100:.2f}%)")
    
    for k, df in synth_dfs.items():
        s_vals = df[col].dropna().astype(str)
        s_v_codes = s_vals.apply(lambda x: x.startswith(('V', 'E', '?'))).sum()
        s_other_codes = (s_vals == '__other__').sum()
        s_unique = s_vals.nunique()
        print(f"  {k}: Alphanumeric codes: {s_v_codes} | __other__ tokens: {s_other_codes} | Unique codes: {s_unique} | Nulls: {df[col].isna().sum()}")
        print(f"  {k} Top 5 codes:\n{s_vals.value_counts().head(5).to_dict()}")

# 5. Bivariate Correlation Matrix RMSE
print("\n=== 5. BIVARIATE CORRELATION PRESERVATION (RMSE) ===")
real_num = real_df[num_cols].apply(pd.to_numeric, errors='coerce').dropna()
real_corr = real_num.corr().values

for k, df in synth_dfs.items():
    s_num = df[num_cols].apply(pd.to_numeric, errors='coerce').dropna()
    s_corr = s_num.corr().values
    corr_rmse = np.sqrt(np.mean((real_corr - s_corr)**2))
    print(f"Correlation Matrix RMSE (Real vs {k}): {corr_rmse:.4f}")

# 6. Distance-based MIA Evaluation (generalization vs memorization)
print("\n=== 6. DISTANCE-BASED MEMBERSHIP INFERENCE ATTACK (MIA) ===")
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Use first 1000 rows as train sample and next 1000 rows as holdout sample
train_sample = real_df[num_cols].head(1000).apply(pd.to_numeric, errors='coerce').fillna(0).values
holdout_sample = real_df[num_cols].tail(1000).apply(pd.to_numeric, errors='coerce').fillna(0).values

scaler = StandardScaler()
train_norm = scaler.fit_transform(train_sample)
holdout_norm = scaler.transform(holdout_sample)

for k, df in synth_dfs.items():
    synth_sample = df[num_cols].head(2000).apply(pd.to_numeric, errors='coerce').fillna(0).values
    synth_norm = scaler.transform(synth_sample)
    
    # Distance from train to synth vs holdout to synth
    d_train = cdist(train_norm, synth_norm, metric='euclidean').min(axis=1).mean()
    d_holdout = cdist(holdout_norm, synth_norm, metric='euclidean').min(axis=1).mean()
    
    mia_advantage = (d_holdout - d_train) / (d_holdout + 1e-8)
    print(f"{k} -> Mean Min Distance to Synth: Train = {d_train:.4f} | Holdout = {d_holdout:.4f} | MIA Advantage = {mia_advantage:.4f}")
