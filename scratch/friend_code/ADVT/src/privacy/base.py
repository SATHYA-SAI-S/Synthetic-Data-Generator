from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from pydantic import BaseModel
import torch

class RiskTier(BaseModel):
    name: str
    epsilon_budget: float
    clip_norm: float

class AbstractPrivacyAccountant(ABC):
    """
    Centralized DP accountant. All noise injection MUST be recorded here.
    """
    @abstractmethod
    def record_step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Record a single step of noise addition."""
        pass
        
    @abstractmethod
    def get_epsilon(self, target_delta: float) -> float:
        """Get the current epsilon spent."""
        pass

class AbstractRiskTierAssigner(ABC):
    """
    Assigns features to Risk Tiers based on cardinality, uniqueness, HIPAA, and correlation.
    """
    @abstractmethod
    def assign_tiers(self, df: pd.DataFrame, column_profiles: list) -> Dict[str, str]:
        """
        Return a mapping from column name to tier name.
        """
        pass
