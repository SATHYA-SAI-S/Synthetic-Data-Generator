import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule
import logging
from typing import Dict, List
from pathlib import Path

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
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if not tier_params:
            raise ValueError("tier_params cannot be empty")
        for tier_name, norm in tier_clip_norms.items():
            if norm <= 0:
                raise ValueError(f"tier_clip_norms[{tier_name}] must be positive, got {norm}")
                
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
        
        # CRITICAL FIX (K-02): Remap tier_params to the WRAPPED module's parameters.
        # Previously, tier_params were extracted from the ORIGINAL denoiser before
        # wrapping with GradSampleModule. Opacus's GradSampleModule COPIES the module
        # and its parameters, so the original parameter objects never receive
        # .grad_sample. This caused clip_and_noise_tier to silently no-op, meaning
        # NO DP noise was ever added and NO privacy accounting was recorded.
        #
        # The fix: build a mapping from original parameter IDs to wrapped parameter
        # objects, then remap all tier_params through it. This ensures the wrapped
        # parameters (which DO get .grad_sample populated by Opacus) are used.
        wrapped_params = list(self.denoiser.parameters())
        original_params = list(denoiser.parameters())
        param_map = {id(orig): wrapped for orig, wrapped in zip(original_params, wrapped_params)}
        
        self.tier_params = {}
        for tier_name, p_list in tier_params.items():
            remapped = [param_map[id(p)] for p in p_list if id(p) in param_map]
            self.tier_params[tier_name] = remapped
            if len(remapped) != len(p_list):
                log.warning(
                    "DPTrainer: %d of %d parameters in tier '%s' could not be remapped "
                    "to the wrapped GradSampleModule. DP noise will NOT be applied to "
                    "unmapped parameters.",
                    len(p_list) - len(remapped), len(p_list), tier_name,
                )
            
        self.tier_clip_norms = tier_clip_norms
        self.device = device
        self.criterion = nn.MSELoss(reduction='none') # Need per-sample loss for opacus

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.denoiser.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch in dataloader:
            if not batch:
                continue
            x_0 = batch[0].to(self.device)
            batch_size = x_0.shape[0]
            if batch_size == 0:
                continue
            n_samples += batch_size
            
            # Uniformly sample timesteps
            t = torch.randint(
                0, self.schedule.num_timesteps, (batch_size,), device=self.device
            )
            
            # Apply forward diffusion process
            x_t, true_noise = forward_diffuse(x_0, t, self.schedule)
            x_t = x_t.to(self.device)
            true_noise = true_noise.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Predict noise
            pred_noise = self.denoiser(x_t, t)
            
            # L2 loss per sample
            loss_per_sample = self.criterion(pred_noise, true_noise).mean(dim=1)
            batch_loss = loss_per_sample.mean()
            
            # C-1 FIX: NaN guard — a NaN/Inf loss means divergence; fail fast
            # instead of checkpointing a poisoned model and exporting garbage.
            if not torch.isfinite(batch_loss):
                raise FloatingPointError(
                    "DPTrainer: loss became NaN/Inf. Training has diverged "
                    "(noise multiplier too high relative to clip norm, or lr too large). "
                    "Reduce base_sigma / learning rate and restart."
                )
            
            # Backprop to populate .grad_sample
            batch_loss.backward()
            
            # P-4 FIX: use the MINIMUM sigma over the batch's timesteps for both
            # noise addition and accounting. Per-sample timesteps differ; using the
            # min sigma is the conservative (privacy-preserving) choice and keeps
            # the RDP composition valid for the whole batch.
            t_min = int(t.min().item())
            t_min = max(0, min(t_min, self.privacy_schedule.num_timesteps - 1))
            sigma_t = self.privacy_schedule.get_sigma(t_min)
            
            # Apply distinct clip norm and noise per tier.
            # P-2 FIX: the accountant records ONE step per batch (passed only on
            # the first tier), not once per tier.
            for i, (tier_name, params) in enumerate(self.tier_params.items()):
                clip_norm = self.tier_clip_norms[tier_name]
                
                clip_and_noise_tier(
                    params=params,
                    clip_norm=clip_norm,
                    noise_multiplier=sigma_t,
                    batch_size=batch_size,
                    dataset_size=self.dataset_size,
                    accountant=self.accountant if i == 0 else None
                )
            
            # C-1 FIX: guard against NaN gradients before stepping
            grad_finite = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in self.denoiser.parameters()
            )
            if not grad_finite:
                raise FloatingPointError(
                    "DPTrainer: non-finite gradients detected after clipping/noise. "
                    "Aborting to prevent model corruption."
                )
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss_per_sample.sum().item()
            
        return total_loss / max(1, n_samples)

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        loss: float = 0.0,
        extra: Dict = None
    ) -> None:
        """
        Saves full trainer state to disk atomically.
        Unwraps GradSampleModule to store clean model weights.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")

        # Unwrap denoiser to save clean weights
        unwrapped = getattr(self.denoiser, "_module", self.denoiser)
        
        state = {
            "epoch": epoch,
            "loss": loss,
            "denoiser_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accountant_history": getattr(getattr(self.accountant, "_accountant", None), "history", []),
            "accountant_steps": getattr(self.accountant, "steps", 0),
            "extra": extra or {}
        }
        
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        log.info(f"Saved DP checkpoint to {checkpoint_path} (epoch {epoch})")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Loads trainer state from disk into denoiser, optimizer, and accountant.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(path, map_location=self.device)
        
        unwrapped = getattr(self.denoiser, "_module", self.denoiser)
        unwrapped.load_state_dict(checkpoint["denoiser_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if hasattr(self.accountant, "_accountant") and "accountant_history" in checkpoint:
            self.accountant._accountant.history = list(checkpoint["accountant_history"])
            
        log.info(f"Loaded DP checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 0)})")
        return checkpoint
