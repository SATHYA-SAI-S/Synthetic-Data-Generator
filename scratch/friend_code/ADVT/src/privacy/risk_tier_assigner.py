import logging
from typing import Dict
import pandas as pd
import numpy as np

from src.privacy.base import AbstractRiskTierAssigner
from src.profiling.dataset_profiler import check_hipaa_identifier

log = logging.getLogger(__name__)

class HeuristicRiskTierAssigner(AbstractRiskTierAssigner):
    """
    Implements Section 3 heuristic assignment + correlation guard.
    
    Assigns:
        Tier1: Strict (HIPAA match OR high uniqueness)
        Tier2: Moderate (Medium uniqueness)
        Tier3: Loose (Low cardinality/uniqueness)
    """
    def __init__(self, correlation_threshold: float = 0.7) -> None:
        if correlation_threshold < 0.0 or correlation_threshold > 1.0:
            raise ValueError(f"correlation_threshold must be in [0.0, 1.0], got {correlation_threshold}")
        self.correlation_threshold = float(correlation_threshold)

    def assign_tiers(self, df: pd.DataFrame, column_profiles: list = None) -> Dict[str, str]:
        tiers = {}
        if df.empty or len(df.columns) == 0:
            return tiers
        
        # 1. Initial heuristic assignment
        for col in df.columns:
            hipaa = check_hipaa_identifier(col)
            
            series = df[col].dropna()
            n = len(series)
            uniqueness = (series.nunique() / max(n, 1)) if n > 0 else 0.0
                
            if hipaa.is_identifier or uniqueness > 0.8:
                tiers[col] = "Tier1" # Strict budget
            elif uniqueness > 0.15:
                tiers[col] = "Tier2" # Moderate budget
            else:
                tiers[col] = "Tier3" # Loose budget
                
        # 2. Correlated-feature leakage guard (Risk Register item 2)
        # We factorize all columns to compute a fast, rough correlation matrix.
        # This prevents a tight-budget field from being reconstructed via a loose-budget field.
        
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        numeric_df = pd.DataFrame()
        for col in df_sample.columns:
            numeric_df[col] = pd.factorize(df_sample[col])[0]
            
        corr_matrix = numeric_df.corr().abs().fillna(0.0)
        
        rank_map = {"Tier1": 1, "Tier2": 2, "Tier3": 3}
        rank_inv = {1: "Tier1", 2: "Tier2", 3: "Tier3"}
        
        # Resolve correlations by promoting to the stricter tier
        for col1 in df_sample.columns:
            for col2 in df_sample.columns:
                if col1 != col2 and col1 in tiers and col2 in tiers:
                    corr_val = float(corr_matrix.loc[col1, col2])
                    if corr_val > self.correlation_threshold:
                        r1 = rank_map.get(tiers[col1], 2)
                        r2 = rank_map.get(tiers[col2], 2)
                        tighter = rank_inv[min(r1, r2)]
                        
                        if tiers[col1] != tighter or tiers[col2] != tighter:
                            log.debug(
                                "Correlation guard: grouped '%s' and '%s' into %s due to corr=%.2f", 
                                col1, col2, tighter, corr_val
                            )
                        tiers[col1] = tighter
                        tiers[col2] = tighter
                        
        return tiers
