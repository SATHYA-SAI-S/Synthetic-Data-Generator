from typing import Protocol
import torch

class AbstractNoiseSchedule(Protocol):
    """Protocol for a noise schedule (e.g., beta/sigma scheduling over time)."""
    @property
    def num_timesteps(self) -> int:
        ...

    def get_betas(self) -> torch.Tensor:
        ...

    def get_alphas(self) -> torch.Tensor:
        ...

    def get_alphas_cumprod(self) -> torch.Tensor:
        ...

class AbstractDenoiser(Protocol):
    """Protocol for a neural network that predicts noise (epsilon) given x_t and t."""
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...
        
    @property
    def input_dim(self) -> int:
        ...
