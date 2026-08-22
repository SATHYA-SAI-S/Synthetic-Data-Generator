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
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError(f"betas must be positive, got start={beta_start}, end={beta_end}")
        if beta_start >= beta_end or beta_end >= 1.0:
            raise ValueError(f"Invalid beta range: {beta_start} to {beta_end}")

        self._num_timesteps = num_timesteps
        self._device = device
        
        # Linear schedule
        self._betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self._alphas = 1.0 - self._betas
        self._alphas_cumprod = torch.cumprod(self._alphas, dim=0)
        
    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "LinearNoiseSchedule":
        """Move all schedule tensors to the specified device."""
        self._device = device
        self._betas = self._betas.to(device)
        self._alphas = self._alphas.to(device)
        self._alphas_cumprod = self._alphas_cumprod.to(device)
        return self

    def get_betas(self) -> torch.Tensor:
        return self._betas

    def get_alphas(self) -> torch.Tensor:
        return self._alphas

    def get_alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod
