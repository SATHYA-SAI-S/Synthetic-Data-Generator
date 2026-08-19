import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)

class UtilityEvaluator:
    """
    Evaluates the statistical fidelity of the synthetic data compared to real data.
    """
    def __init__(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> None:
        self.df_real = df_real
        self.df_synth = df_synth
        
    def _compute_categorical_tvd(self, col: str) -> float:
        """Computes Total Variation Distance (TVD) for categorical columns."""
        counts_real = self.df_real[col].value_counts(normalize=True)
        counts_synth = self.df_synth[col].value_counts(normalize=True)
        
        all_cats = set(counts_real.index).union(set(counts_synth.index))
        tvd = 0.5 * sum(abs(counts_real.get(c, 0.0) - counts_synth.get(c, 0.0)) for c in all_cats)
        return float(tvd)

    def _compute_continuous_ks(self, col: str) -> float:
        """Computes Kolmogorov-Smirnov (KS) statistic for continuous columns."""
        real_vals = self.df_real[col].dropna()
        synth_vals = self.df_synth[col].dropna()
        
        if len(real_vals) == 0 or len(synth_vals) == 0:
            return 1.0
            
        stat, _ = ks_2samp(real_vals, synth_vals)
        return float(stat)

    def evaluate_univariate(self) -> Dict[str, Any]:
        """
        Returns a dictionary of univariate distance metrics per column.
        Lower values (closer to 0) indicate higher utility/fidelity.
        """
        results = {}
        for col in self.df_real.columns:
            if col not in self.df_synth.columns:
                log.warning(f"Column {col} missing in synthetic data.")
                continue
                
            # Heuristic dtype split for metric selection
            if self.df_real[col].dtype == 'object' or pd.api.types.is_bool_dtype(self.df_real[col]):
                val = self._compute_categorical_tvd(col)
                results[col] = {"metric": "TVD", "value": val}
            else:
                try:
                    val = self._compute_continuous_ks(col)
                    results[col] = {"metric": "KS", "value": val}
                except Exception:
                    # Fallback if cast fails
                    val = self._compute_categorical_tvd(col)
                    results[col] = {"metric": "TVD", "value": val}
                    
        return results

    def evaluate_bivariate_correlation_rmse(self) -> float:
        """
        Computes the RMSE between the Pearson correlation matrices of the real and synthetic numeric data.
        Ensures columns are common, aligned, and matched in identical order.
        """
        real_num = self.df_real.select_dtypes(include=[np.number])
        synth_num = self.df_synth.select_dtypes(include=[np.number])
        
        # Intersect numeric columns to ensure identical order and alignment
        common_cols = [c for c in real_num.columns if c in synth_num.columns]
        if len(common_cols) < 2:
            return 0.0
            
        corr_real = real_num[common_cols].corr().fillna(0.0).values
        corr_synth = synth_num[common_cols].corr().fillna(0.0).values
        
        diff = corr_real - corr_synth
        rmse = np.sqrt(np.mean(diff ** 2))
        return float(rmse)
