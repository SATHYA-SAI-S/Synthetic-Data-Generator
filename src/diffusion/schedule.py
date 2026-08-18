import torch
from src.diffusion.base import AbstractNoiseSchedule

class LinearNoiseSchedule(AbstractNoiseSchedule):
    """Standard linear beta schedule for DDPM."""
    def __init__(
        self, 
        num_timesteps: int = 1000, 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02, 
        device: torch.device = torch.device('cpu')
    ) -> None:
        self._num_timesteps = num_timesteps
        self._device = device
        
        # Linear schedule
        self._betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self._alphas = 1.0 - self._betas
        self._alphas_cumprod = torch.cumprod(self._alphas, dim=0)
        
    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def get_betas(self) -> torch.Tensor:
        return self._betas

    def get_alphas(self) -> torch.Tensor:
        return self._alphas

    def get_alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod
