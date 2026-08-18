import torch
from torch.utils.data import TensorDataset, DataLoader
from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.trainer import DiffusionTrainer
from src.diffusion.sampler import generate_samples

def test_trainer_smoke():
    # Tiny synthetic tensor
    batch_size = 8
    input_dim = 5
    num_samples = 32
    
    # Just random data
    data = torch.randn((num_samples, input_dim))
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Init components
    denoiser = MLPDenoiser(input_dim, [32, 32], num_timesteps=20)
    schedule = LinearNoiseSchedule(num_timesteps=20)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.01)
    
    trainer = DiffusionTrainer(denoiser, schedule, optimizer)
    
    # Run 1 epoch
    loss_0 = trainer.train_epoch(loader)
    
    # Run a few more to see loss decrease
    for _ in range(5):
        loss_n = trainer.train_epoch(loader)
        
    # Not strictly guaranteed to decrease on every single small run due to noise sampling, 
    # but we just want to ensure it runs without crashing.
    assert loss_n >= 0.0
    
    # Ensure sampling works
    samples = generate_samples(denoiser, schedule, num_samples=10)
    assert samples.shape == (10, input_dim)
