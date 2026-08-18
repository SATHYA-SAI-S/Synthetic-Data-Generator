import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule
import logging
from typing import Dict, List

from src.diffusion.base import AbstractDenoiser
from src.privacy.base import AbstractPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.clip_and_noise import clip_and_noise_tier
from src.diffusion.forward_process import forward_diffuse

log = logging.getLogger(__name__)

class DPTrainer:
    """
    Phase 5 DP-SGD Trainer.
    Supports fixed sigma, adaptive sigma, and per-tier privacy heads.
    """
    def __init__(
        self,
        denoiser: AbstractDenoiser,
        schedule, # AbstractNoiseSchedule for diffusion
        optimizer: torch.optim.Optimizer,
        accountant: AbstractPrivacyAccountant,
        privacy_schedule: AdaptiveNoiseSchedule,
        dataset_size: int,
        tier_params: Dict[str, List[torch.nn.Parameter]],
        tier_clip_norms: Dict[str, float],
        device: torch.device = torch.device('cpu')
    ) -> None:
        # Fix for Opacus parameter attribute loss:
        # We must call .to(device) ON the original module before wrapping.
        # Calling it on GradSampleModule creates new parameter objects that lack 
        # the p.grad_sample attribute initialized by Opacus.
        denoiser = denoiser.to(device)
        self.denoiser = GradSampleModule(denoiser)
        self.schedule = schedule
        self.optimizer = optimizer
        self.accountant = accountant
        self.privacy_schedule = privacy_schedule
        self.dataset_size = dataset_size
        
        # C-2 Fix: Remap tier_params to the wrapped module's parameters.
        # The caller passed original parameter objects, but GradSampleModule
        # may have created new ones or the caller doesn't have the wrapped ones.
        # We can map them by size/shape and position, or just extract them directly 
        # from self.denoiser.parameters(). 
        # Actually, since all parameters are in self.denoiser, we can map by names if we had names.
        # But we only have parameter objects. Let's find the matching parameter in the wrapped module
        # by checking shape and data equality (or just rely on the order, which is deterministic).
        
        wrapped_params = list(self.denoiser.parameters())
        original_params = list(denoiser.parameters())
        param_map = {id(orig): wrapped for orig, wrapped in zip(original_params, wrapped_params)}
        
        self.tier_params = {}
        for tier_name, p_list in tier_params.items():
            self.tier_params[tier_name] = [param_map[id(p)] for p in p_list if id(p) in param_map]
            
        self.tier_clip_norms = tier_clip_norms
        self.device = device
        self.criterion = nn.MSELoss(reduction='none') # Need per-sample loss for opacus

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.denoiser.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch in dataloader:
            x_0 = batch[0].to(self.device)
            batch_size = x_0.shape[0]
            n_samples += batch_size
            
            # Uniformly sample timesteps
            t = torch.randint(
                0, self.schedule.num_timesteps, (batch_size,), device=self.device
            )
            
            # Apply forward diffusion process
            x_t, true_noise = forward_diffuse(x_0, t, self.schedule)
            x_t = x_t.to(self.device)
            x_t.requires_grad_()
            true_noise = true_noise.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Predict noise
            pred_noise = self.denoiser(x_t, t)
            
            # L2 loss per sample
            loss_per_sample = self.criterion(pred_noise, true_noise).mean(dim=1)
            
            # Backprop to populate .grad_sample
            loss_per_sample.mean().backward()
            
            # Apply distinct clip norm and noise per tier
            for tier_name, params in self.tier_params.items():
                clip_norm = self.tier_clip_norms[tier_name]
                
                # If using adaptive schedule, we need to decide the sigma.
                # Since DP-SGD normally expects a constant noise multiplier per step,
                # but we have an adaptive schedule over T. 
                # Wait, the paper specifies "Adaptive sigma per diffusion timestep".
                # A batch contains MULTIPLE timesteps `t`!
                # If we apply noise to the parameter gradients, the parameter gradients 
                # are a SUM over the batch. We can't apply different sigmas to different 
                # samples easily AFTER summation.
                # Wait! DP-SGD adds noise to the SUM of gradients.
                # If the batch has a uniform average `t`, we could use the average.
                # Or we group samples by `t`? No, Opacus adds noise to the parameter.
                # The prompt: "Adaptive sigma per diffusion timestep - extend accountant call to accept per-step sigma"
                # If the noise is added to the gradient, and the gradient is averaged over the batch,
                # we can just use the mean `t` of the batch to query the adaptive schedule!
                # Let's use the mean `t` of the batch.
                mean_t = int(t.float().mean().item())
                sigma_t = self.privacy_schedule.get_sigma(mean_t)
                
                clip_and_noise_tier(
                    params=params,
                    clip_norm=clip_norm,
                    noise_multiplier=sigma_t,
                    batch_size=batch_size,
                    dataset_size=self.dataset_size,
                    accountant=self.accountant
                )
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss_per_sample.sum().item()
            
        return total_loss / max(1, n_samples)
