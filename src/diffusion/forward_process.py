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
    if x_0.ndim != 2:
        raise ValueError(f"Expected 2D tensor x_0 (batch_size, input_dim), got shape {x_0.shape}")
        
    # H-2 FIX: Move schedule tensors to x_0.device to prevent device mismatch.
    # Previously, if the schedule was constructed on CPU (default) but x_0 was
    # on CUDA, the multiplication below would crash with a device mismatch error.
    alphas_cumprod = schedule.get_alphas_cumprod().to(x_0.device)
    num_timesteps = schedule.num_timesteps
    
    # Clamp timesteps to valid range
    t_clamped = t.to(x_0.device).clamp(0, num_timesteps - 1).long()
    
    # Extract alpha_cumprod for the given timesteps
    # shape: (batch_size, 1) to broadcast over input_dim
    a_t = alphas_cumprod[t_clamped].view(-1, 1)
    
    # Generate Gaussian noise
    noise = torch.randn_like(x_0)
    
    # q(x_t | x_0) = sqrt(a_t) * x_0 + sqrt(1 - a_t) * noise
    x_t = torch.sqrt(torch.clamp(a_t, min=0.0)) * x_0 + torch.sqrt(torch.clamp(1.0 - a_t, min=0.0)) * noise
    
    return x_t, noise
