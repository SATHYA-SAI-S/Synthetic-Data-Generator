import torch
from typing import Tuple
from src.diffusion.base import AbstractNoiseSchedule

def forward_diffuse(
    x_0: torch.Tensor, 
    t: torch.Tensor, 
    schedule: AbstractNoiseSchedule
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply closed-form forward diffusion q(x_t | x_0).
    
    Args:
        x_0: Original data tensor of shape (batch_size, input_dim).
        t: Timesteps tensor of shape (batch_size,).
        schedule: The noise schedule.
        
    Returns:
        x_t: Noisy data at timestep t.
        noise: The true noise (epsilon) added, shape (batch_size, input_dim).
    """
    alphas_cumprod = schedule.get_alphas_cumprod().to(x_0.device)
    
    # Extract alpha_cumprod for the given timesteps
    # shape: (batch_size, 1) to broadcast over input_dim
    a_t = alphas_cumprod[t].view(-1, 1)
    
    # Generate Gaussian noise
    noise = torch.randn_like(x_0)
    
    # q(x_t | x_0) = sqrt(a_t) * x_0 + sqrt(1 - a_t) * noise
    x_t = torch.sqrt(a_t) * x_0 + torch.sqrt(1.0 - a_t) * noise
    
    return x_t, noise
