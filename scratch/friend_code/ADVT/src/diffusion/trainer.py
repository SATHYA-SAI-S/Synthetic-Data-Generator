import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from src.diffusion.base import AbstractDenoiser, AbstractNoiseSchedule
from src.diffusion.forward_process import forward_diffuse

log = logging.getLogger(__name__)

class DiffusionTrainer:
    """
    Phase 4: Baseline (Non-DP) Diffusion Trainer.
    Trains the denoiser to predict added Gaussian noise.
    """
    def __init__(
        self,
        denoiser: AbstractDenoiser,
        schedule: AbstractNoiseSchedule,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu')
    ) -> None:
        self.denoiser = denoiser.to(device)
        self.schedule = schedule
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader yielding (x_0,) batches.
            
        Returns:
            Average MSE loss for the epoch.
        """
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
            true_noise = true_noise.to(self.device)
            
            # Predict noise
            pred_noise = self.denoiser(x_t, t)
            
            # L2 loss between predicted noise and true noise
            loss = self.criterion(pred_noise, true_noise)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_size
            
        return total_loss / max(1, n_samples)
