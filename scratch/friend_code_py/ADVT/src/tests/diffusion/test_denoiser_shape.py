import torch
import pytest
from src.diffusion.denoiser import MLPDenoiser

def test_denoiser_shapes():
    # Synthetic schema 1
    input_dim_1 = 45
    hidden_dims_1 = [128, 64]
    net1 = MLPDenoiser(input_dim_1, hidden_dims_1, num_timesteps=50)
    
    x1 = torch.randn(16, input_dim_1)
    t1 = torch.randint(0, 50, (16,))
    
    out1 = net1(x1, t1)
    assert out1.shape == (16, input_dim_1)
    assert net1.input_dim == input_dim_1
    
    # Synthetic schema 2
    input_dim_2 = 12
    hidden_dims_2 = [256, 256, 256]
    net2 = MLPDenoiser(input_dim_2, hidden_dims_2, num_timesteps=1000)
    
    x2 = torch.randn(4, input_dim_2)
    t2 = torch.randint(0, 1000, (4,))
    
    out2 = net2(x2, t2)
    assert out2.shape == (4, input_dim_2)
    assert net2.input_dim == input_dim_2
