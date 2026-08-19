import torch
from src.diffusion.base import AbstractDenoiser, AbstractNoiseSchedule

@torch.no_grad()
def generate_samples(
    denoiser: AbstractDenoiser,
    schedule: AbstractNoiseSchedule,
    num_samples: int,
    device: torch.device = torch.device('cpu'),
    batch_size: int = 8192
) -> torch.Tensor:
    """
    Generate synthetic samples by running the reverse diffusion process (T -> 0).
    Supports chunked sampling to avoid VRAM exhaustion on large sample sizes.
    
    Args:
        denoiser: Trained denoising network.
        schedule: The noise schedule.
        num_samples: Number of synthetic rows to generate.
        device: Torch device to use.
        batch_size: Maximum batch size per sampling chunk.
        
    Returns:
        Tensor of shape (num_samples, input_dim) representing synthetic rows.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    denoiser.eval()
    
    # If sample count exceeds batch_size, generate in chunks
    if batch_size is not None and num_samples > batch_size:
        chunks = []
        remaining = num_samples
        while remaining > 0:
            current_batch = min(remaining, batch_size)
            chunk = _generate_single_batch(denoiser, schedule, current_batch, device)
            chunks.append(chunk)
            remaining -= current_batch
        return torch.cat(chunks, dim=0)
    else:
        return _generate_single_batch(denoiser, schedule, num_samples, device)


def _generate_single_batch(
    denoiser: AbstractDenoiser,
    schedule: AbstractNoiseSchedule,
    num_samples: int,
    device: torch.device
) -> torch.Tensor:
    # Start from pure Gaussian noise x_T
    x = torch.randn((num_samples, denoiser.input_dim), device=device)
    
    alphas = schedule.get_alphas().to(device)
    alphas_cumprod = schedule.get_alphas_cumprod().to(device)
    betas = schedule.get_betas().to(device)
    
    for t in reversed(range(schedule.num_timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict epsilon
        pred_noise = denoiser(x, t_tensor)
        
        a_t = alphas[t]
        a_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        # DDPM update rule with numerical stability clamp
        denom = torch.sqrt(torch.clamp(1.0 - a_cumprod_t, min=1e-8))
        sqrt_a_t = torch.sqrt(torch.clamp(a_t, min=1e-8))
        
        x = (1.0 / sqrt_a_t) * (
            x - ((1.0 - a_t) / denom) * pred_noise
        )
        
        # Add posterior noise for all steps except t=0
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(torch.clamp(beta_t, min=0.0)) * noise
            
    return x
