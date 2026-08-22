import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.dp_trainer import DPTrainer

def test_dp_trainer_fixed_sigma():
    batch_size = 8
    dataset_size = 32
    input_dim = 5
    
    data = torch.randn((dataset_size, input_dim))
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    denoiser = MLPDenoiser(input_dim, [16, 16], num_timesteps=10)
    schedule = LinearNoiseSchedule(num_timesteps=10)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.01)
    
    accountant = CentralPrivacyAccountant()
    # Constant strategy simulates Fixed-Sigma
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=10, strategy="constant")
    
    # Global tier
    tier_params = {"global": list(denoiser.parameters())}
    tier_clip_norms = {"global": 1.0}
    
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
    
    assert accountant.steps == 0
    loss = trainer.train_epoch(loader)
    
    # We had dataset_size=32, batch_size=8 -> 4 steps
    assert accountant.steps == 4
    eps = accountant.get_epsilon(target_delta=1e-5)
    assert eps > 0.0
    
    assert hasattr(trainer.denoiser._module, 'out_layer') # verify wrapper doesn't destroy access
