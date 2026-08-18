import torch
from typing import List
from src.privacy.base import AbstractPrivacyAccountant

def clip_and_noise_tier(
    params: List[torch.nn.Parameter],
    clip_norm: float,
    noise_multiplier: float,
    batch_size: int,
    dataset_size: int,
    accountant: AbstractPrivacyAccountant
) -> None:
    """
    Applies per-sample gradient clipping and Gaussian noise for a specific tier.
    Assumes params have .grad_sample populated (e.g., via Opacus GradSampleModule).
    
    Args:
        params: Parameters belonging to this tier.
        clip_norm: The clipping threshold C for this tier.
        noise_multiplier: sigma for this tier.
        batch_size: Number of samples in the batch.
        dataset_size: Total number of samples in the dataset (for accounting).
        accountant: Central DP accountant to record this specific event.
    """
    if not params:
        return
        
    # 1. Compute per-sample gradient norm across all params in this tier
    per_param_norms = [
        p.grad_sample.reshape(batch_size, -1).norm(2, dim=-1)
        for p in params if hasattr(p, "grad_sample")
    ]
    if not per_param_norms:
        return
        
    per_sample_norms = torch.stack(per_param_norms, dim=0).norm(2, dim=0)
    
    # 2. Compute clipping factor: max(1, C / ||g||)
    clip_factor = (clip_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)
    
    # 3. Clip per-sample gradients, sum into .grad, and add noise
    for p in params:
        if not hasattr(p, "grad_sample"):
            continue
            
        # Reshape clip_factor to broadcast over grad_sample dimensions
        broadcast_shape = [batch_size] + [1] * (p.grad_sample.dim() - 1)
        clipped_grad_sample = p.grad_sample * clip_factor.view(broadcast_shape)
        
        # Sum over batch
        summed_grad = clipped_grad_sample.sum(dim=0)
        
        # Add noise
        noise = torch.randn_like(summed_grad) * clip_norm * noise_multiplier
        
        # Assign to .grad (averaged over batch)
        p.grad = (summed_grad + noise) / batch_size
        
        # Clear grad_sample to conserve memory (must set to None, not del, for Opacus)
        p.grad_sample = None
        
    # 4. Record event in the single centralized accountant
    sample_rate = batch_size / max(1, dataset_size)
    accountant.record_step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
