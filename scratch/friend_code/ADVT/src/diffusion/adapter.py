import torch
import torch.nn as nn
from typing import List
from src.diffusion.base import AbstractDenoiser

class SchemaAdapterModel(nn.Module, AbstractDenoiser):
    """
    Objective 4: Schema Adapter Network for Small-Dataset Fine-Tuning.
    
    Wraps a pre-trained tabular diffusion backbone (with input_dim D_backbone),
    freezes all of its weights, and attaches trainable input/output projection 
    layers to adapt to a new dataset schema (dimension D_new).
    
    This drastically reduces the number of parameters requiring DP-SGD gradient 
    clipping and noise addition, saving significant privacy budget on small-N cohorts.
    """
    def __init__(
        self,
        backbone: nn.Module,
        new_input_dim: int,
        freeze_backbone: bool = True
    ) -> None:
        super().__init__()
        if new_input_dim <= 0:
            raise ValueError(f"new_input_dim must be positive, got {new_input_dim}")
        if not hasattr(backbone, "input_dim") or backbone.input_dim <= 0:
            raise ValueError("Backbone must expose a positive input_dim property")
        if not hasattr(backbone, "num_timesteps") or backbone.num_timesteps <= 0:
            raise ValueError("Backbone must expose a positive num_timesteps property")

        self.backbone = backbone
        self._new_input_dim = int(new_input_dim)
        self._backbone_dim = int(backbone.input_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Trainable adapter layers
        self.input_adapter = nn.Sequential(
            nn.Linear(self._new_input_dim, self._backbone_dim),
            nn.SiLU()
        )
        self.output_adapter = nn.Linear(self._backbone_dim, self._new_input_dim)

    @property
    def input_dim(self) -> int:
        return self._new_input_dim

    @property
    def num_timesteps(self) -> int:
        return self.backbone.num_timesteps

    def adapter_parameters(self) -> List[torch.nn.Parameter]:
        """Returns only the trainable adapter parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Project x (new schema) -> backbone space -> predict noise -> project back.
        
        Args:
            x: (batch_size, new_input_dim) - Noisy data tensor
            t: (batch_size,) - Timestep indices
            
        Returns:
            pred_noise: (batch_size, new_input_dim)
        """
        if x.ndim != 2 or x.shape[-1] != self._new_input_dim:
            raise ValueError(
                f"Expected input x of shape (batch_size, {self._new_input_dim}), got {x.shape}"
            )

        # 1. Project new schema into backbone dimension
        h = self.input_adapter(x)

        # 2. Pass through frozen pre-trained diffusion backbone
        eps_backbone = self.backbone(h, t)

        # 3. Project back to new schema dimension
        pred_noise = self.output_adapter(eps_backbone)
        return pred_noise
