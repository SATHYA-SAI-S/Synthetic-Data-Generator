import torch
import torch.nn as nn
from typing import List
from src.diffusion.base import AbstractDenoiser

class MLPDenoiser(nn.Module, AbstractDenoiser):
    """
    Concrete denoising network for tabular data.
    Dynamically sized based on registry output dimension.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        num_timesteps: int = 1000
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        
        # Time embedding: project t into the hidden space
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
            
        # ModuleList so we can inject time embeddings inside the forward loop
        self.net = nn.ModuleList(layers)
        
        # Output layer predicts noise epsilon of same shape as input
        self.out_layer = nn.Linear(in_dim, input_dim)
        
    @property
    def input_dim(self) -> int:
        return self._input_dim
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise epsilon.
        
        Args:
            x: (batch_size, input_dim) - noisy data x_t
            t: (batch_size,) - diffusion timesteps
            
        Returns:
            epsilon_pred: (batch_size, input_dim)
        """
        t_emb = self.time_embed(t) # (batch_size, hidden_dims[0])
        
        h = x
        # 3 modules per logical layer: Linear, SiLU, LayerNorm
        for i in range(0, len(self.net), 3):
            h = self.net[i](h) # Linear
            
            # Inject time conditioning at the first hidden layer
            if i == 0:
                h = h + t_emb
                
            h = self.net[i+1](h) # SiLU
            h = self.net[i+2](h) # LayerNorm
            
        return self.out_layer(h)
