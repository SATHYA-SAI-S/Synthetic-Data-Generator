import torch

class AdaptiveNoiseSchedule:
    """
    Computes per-timestep sigma (noise multiplier).
    
    Since early diffusion steps (near T) contain mostly noise, they leak less 
    data and need less DP noise. Later steps (near 0) contain more signal 
    and need more DP noise.
    """
    def __init__(self, base_sigma: float, num_timesteps: int, strategy: str = "linear") -> None:
        if base_sigma < 0:
            raise ValueError(f"base_sigma cannot be negative, got {base_sigma}")
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
            
        self.base_sigma = float(base_sigma)
        self.num_timesteps = int(num_timesteps)
        self.strategy = strategy
        self.sigmas = self._compute_schedule()
        
    def _compute_schedule(self) -> torch.Tensor:
        sigmas = torch.zeros(self.num_timesteps)
        for t in range(self.num_timesteps):
            # t=0 is data (high signal), t=T is pure noise (low signal)
            # We want more DP noise at t=0, less at t=T
            ratio = 1.0 - (t / max(1, self.num_timesteps - 1))
            
            if self.strategy == "linear":
                # Linear from 0.5 * base (at T) to 1.5 * base (at 0)
                # Averages to base_sigma across all timesteps
                s = self.base_sigma * (0.5 + ratio)
            elif self.strategy == "constant":
                s = self.base_sigma
            else:
                s = self.base_sigma
                
            sigmas[t] = s
            
        return sigmas
        
    def get_sigma(self, t: int) -> float:
        clamped_t = int(max(0, min(int(t), self.num_timesteps - 1)))
        return float(self.sigmas[clamped_t])
