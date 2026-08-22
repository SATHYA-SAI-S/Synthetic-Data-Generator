import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any

class PrivacyEvaluator:
    """
    Evaluates empirical privacy risks using Distance-Based Membership Inference Attacks (D-MIA).
    """
    def __init__(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame, df_synth: pd.DataFrame):
        self.df_train = df_train
        self.df_holdout = df_holdout
        self.df_synth = df_synth
        
    def _factorize_data(self, df: pd.DataFrame, reference_cols: list) -> np.ndarray:
        """Robust conversion to numeric space for distance computation."""
        if df.empty or not reference_cols:
            return np.empty((len(df), len(reference_cols)), dtype=float)
            
        out = pd.DataFrame()
        for col in reference_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or pd.api.types.is_bool_dtype(df[col]):
                    out[col] = pd.factorize(df[col])[0]
                else:
                    out[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                out[col] = 0.0
        return out.to_numpy(dtype=float)
        
    def evaluate_mia_risk(self) -> Dict[str, Any]:
        """
        Computes the Distance-Based MIA vulnerability score.
        If synthetic data memorizes the training set, the distance from training records 
        to their nearest synthetic neighbor will be noticeably smaller than the distance 
        from holdout records to synthetic neighbors.
        """
        if self.df_train.empty or self.df_holdout.empty or self.df_synth.empty:
            return {
                "mean_dist_train_to_synth": 0.0,
                "mean_dist_holdout_to_synth": 0.0,
                "mia_risk_score": 0.0
            }

        cols = self.df_train.columns.tolist()
        
        train_num = self._factorize_data(self.df_train, cols)
        holdout_num = self._factorize_data(self.df_holdout, cols)
        synth_num = self._factorize_data(self.df_synth, cols)
        
        if len(synth_num) == 0 or len(train_num) == 0 or len(holdout_num) == 0:
            return {
                "mean_dist_train_to_synth": 0.0,
                "mean_dist_holdout_to_synth": 0.0,
                "mia_risk_score": 0.0
            }

        # Fit Nearest Neighbors on Synthetic Data
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(synth_num)
        
        # Distances from Train -> Synth
        dist_train, _ = nn.kneighbors(train_num)
        mean_dist_train = float(np.mean(dist_train))
        
        # Distances from Holdout -> Synth
        dist_holdout, _ = nn.kneighbors(holdout_num)
        mean_dist_holdout = float(np.mean(dist_holdout))
        
        # Risk Score (0 to 1)
        # If train distances are much smaller than holdout distances, risk is high.
        # If model generalized perfectly, train distances == holdout distances (Risk -> 0)
        dist_diff = max(0.0, mean_dist_holdout - mean_dist_train)
        
        # Normalize risk roughly by holdout distance
        mia_risk_score = min(1.0, dist_diff / max(mean_dist_holdout, 1e-9))
        
        return {
            "mean_dist_train_to_synth": float(mean_dist_train),
            "mean_dist_holdout_to_synth": float(mean_dist_holdout),
            "mia_risk_score": float(mia_risk_score)
        }
