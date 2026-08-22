import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.dp_trainer import DPTrainer

class SplitDenoiser(MLPDenoiser):
    """
    Simulates a denoiser that explicitly partitions parameters for tiers.
    """
    def __init__(self, input_dim: int, hidden_dims: list, num_timesteps: int):
        super().__init__(input_dim, hidden_dims, num_timesteps)
        # Partition output layer conceptually
        # In reality, Phase 5 Step 3 requires separating params.
        # We can just partition the parameter *groups* in the dictionary.
        pass

def test_per_tier_heads():
    batch_size = 8
    dataset_size = 16
    input_dim = 10
    
    data = torch.randn((dataset_size, input_dim))
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    denoiser = MLPDenoiser(input_dim, [32, 32], num_timesteps=10)
    schedule = LinearNoiseSchedule(num_timesteps=10)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.01)
    
    accountant = CentralPrivacyAccountant()
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=10, strategy="constant")
    
    # Split parameters into two hypothetical tiers
    all_params = list(denoiser.parameters())
    mid_idx = len(all_params) // 2
    
    tier_params = {
        "Tier1": all_params[:mid_idx],
        "Tier2": all_params[mid_idx:]
    }
    
    # Apply different clip norms
    tier_clip_norms = {
        "Tier1": 0.1, # Strict
        "Tier2": 2.0  # Loose
    }
    
    trainer = DPTrainer(
        denoiser=denoiser,
        schedule=schedule,
        optimizer=optimizer,
        accountant=accountant,
        privacy_schedule=privacy_schedule,
        dataset_size=dataset_size,
        tier_params=tier_params,
        tier_clip_norms=tier_clip_norms
    )
    
    # 1 epoch
    trainer.train_epoch(loader)
    
    # Dataset size 16, batch size 8 -> 2 steps.
    # We have 2 tiers! The accountant is called twice per batch.
    # Total calls = 2 steps * 2 tiers = 4
    assert accountant.steps == 4
    
    eps = accountant.get_epsilon(target_delta=1e-5)
    assert eps > 0.0
