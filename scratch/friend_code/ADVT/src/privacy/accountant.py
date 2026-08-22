from opacus.accountants import RDPAccountant
from src.privacy.base import AbstractPrivacyAccountant

class CentralPrivacyAccountant(AbstractPrivacyAccountant):
    """
    ONE accountant class; every caller routes through it.
    Wraps Opacus RDPAccountant to ensure standard composition is strictly tracked.
    """
    def __init__(self) -> None:
        self._accountant = RDPAccountant()
        
    def record_step(self, noise_multiplier: float, sample_rate: float) -> None:
        """
        Record a noise addition step.
        """
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier cannot be negative, got {noise_multiplier}")
        if sample_rate <= 0.0 or sample_rate > 1.0:
            raise ValueError(f"sample_rate must be in (0.0, 1.0], got {sample_rate}")
            
        self._accountant.history.append((float(noise_multiplier), float(sample_rate), 1))
        
    def get_epsilon(self, target_delta: float) -> float:
        """
        Get the current epsilon for the target delta.
        """
        if target_delta <= 0.0 or target_delta >= 1.0:
            raise ValueError(f"target_delta must be in (0.0, 1.0), got {target_delta}")
        if not self._accountant.history:
            return 0.0
            
        return float(self._accountant.get_epsilon(delta=target_delta))
        
    @property
    def steps(self) -> int:
        return len(self._accountant.history)
