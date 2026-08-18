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
        self._accountant.history.append((noise_multiplier, sample_rate, 1))
        
    def get_epsilon(self, target_delta: float) -> float:
        """
        Get the current epsilon for the target delta.
        """
        return self._accountant.get_epsilon(delta=target_delta)
        
    @property
    def steps(self) -> int:
        return len(self._accountant.history)
