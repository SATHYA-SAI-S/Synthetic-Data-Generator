import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest

from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.dp_trainer import DPTrainer

def test_dp_trainer_adaptive():
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
    # Linear strategy = Adaptive schedule
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=10, strategy="linear")
    
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
    
    loss = trainer.train_epoch(loader)
    
    assert accountant.steps == 4
    eps = accountant.get_epsilon(target_delta=1e-5)
    
    # Check that eps is sensible
    assert eps > 0.0

    # Ensure the wrapper adds gradients successfully
    for p in denoiser.parameters():
        assert p.grad is not None
