import torch
import pytest
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.forward_process import forward_diffuse

def test_forward_diffuse_shapes():
    schedule = LinearNoiseSchedule(num_timesteps=1000)
    batch_size = 32
    input_dim = 15
    
    x_0 = torch.zeros((batch_size, input_dim))
    t = torch.randint(0, 100, (batch_size,))
    
    x_t, noise = forward_diffuse(x_0, t, schedule)
    
    assert x_t.shape == (batch_size, input_dim)
    assert noise.shape == (batch_size, input_dim)

def test_forward_diffuse_noise_scale():
    schedule = LinearNoiseSchedule(num_timesteps=1000)
    # At t=99 (last step), data should be almost pure noise (mean ~0, std ~1)
    batch_size = 1000
    input_dim = 10
    
    x_0 = torch.ones((batch_size, input_dim)) * 5.0 # strong signal
    t = torch.full((batch_size,), 999, dtype=torch.long)
    
    x_t, noise = forward_diffuse(x_0, t, schedule)
    
    # Check that x_t has mean close to 0 and std close to 1
    assert torch.abs(x_t.mean()) < 0.5
    assert 0.8 < x_t.std() < 1.2
