import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.clip_and_noise import clip_and_noise_tier
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.risk_tier_assigner import HeuristicRiskTierAssigner
from src.privacy.dp_trainer import DPTrainer
from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.sampler import generate_samples
from src.diffusion.forward_process import forward_diffuse
from src.orchestration.gpu_budget_guard import ComputeBudgetGuard
from src.evaluation.utility_metrics import UtilityEvaluator
from src.evaluation.privacy_metrics import PrivacyEvaluator


def test_accountant_input_validation():
    accountant = CentralPrivacyAccountant()
    # Empty history returns 0.0
    assert accountant.get_epsilon(target_delta=1e-5) == 0.0

    # Invalid noise multiplier
    with pytest.raises(ValueError, match="noise_multiplier cannot be negative"):
        accountant.record_step(noise_multiplier=-1.0, sample_rate=0.01)

    # Invalid sample rate
    with pytest.raises(ValueError, match="sample_rate must be in"):
        accountant.record_step(noise_multiplier=1.0, sample_rate=0.0)
    with pytest.raises(ValueError, match="sample_rate must be in"):
        accountant.record_step(noise_multiplier=1.0, sample_rate=1.5)

    # Invalid delta
    with pytest.raises(ValueError, match="target_delta must be in"):
        accountant.get_epsilon(target_delta=0.0)
    with pytest.raises(ValueError, match="target_delta must be in"):
        accountant.get_epsilon(target_delta=1.0)


def test_clip_and_noise_input_validation():
    accountant = CentralPrivacyAccountant()
    param = torch.nn.Parameter(torch.randn(5, 5))
    param.grad_sample = torch.randn(4, 5, 5)

    # Invalid clip_norm
    with pytest.raises(ValueError, match="clip_norm must be positive"):
        clip_and_noise_tier([param], clip_norm=0.0, noise_multiplier=1.0, batch_size=4, dataset_size=100, accountant=accountant)

    # Invalid noise_multiplier
    with pytest.raises(ValueError, match="noise_multiplier cannot be negative"):
        clip_and_noise_tier([param], clip_norm=1.0, noise_multiplier=-0.5, batch_size=4, dataset_size=100, accountant=accountant)

    # Invalid batch size
    with pytest.raises(ValueError, match="batch_size must be positive"):
        clip_and_noise_tier([param], clip_norm=1.0, noise_multiplier=1.0, batch_size=0, dataset_size=100, accountant=accountant)

    # Invalid dataset size
    with pytest.raises(ValueError, match="dataset_size must be positive"):
        clip_and_noise_tier([param], clip_norm=1.0, noise_multiplier=1.0, batch_size=4, dataset_size=0, accountant=accountant)

    # Null grad_sample param handling (should not crash)
    param_null = torch.nn.Parameter(torch.randn(3, 3))
    param_null.grad_sample = None
    clip_and_noise_tier([param_null], clip_norm=1.0, noise_multiplier=1.0, batch_size=4, dataset_size=100, accountant=accountant)


def test_adaptive_schedule_validation_and_clamping():
    # Negative base_sigma
    with pytest.raises(ValueError, match="base_sigma cannot be negative"):
        AdaptiveNoiseSchedule(base_sigma=-1.0, num_timesteps=10)

    # num_timesteps < 1
    with pytest.raises(ValueError, match="num_timesteps must be >= 1"):
        AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=0)

    schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=10, strategy="linear")
    # Clamping negative t
    assert schedule.get_sigma(-5) == schedule.get_sigma(0)
    # Clamping overflow t
    assert schedule.get_sigma(100) == schedule.get_sigma(9)


def test_denoiser_validation_and_clamping():
    with pytest.raises(ValueError, match="input_dim must be positive"):
        MLPDenoiser(input_dim=0, hidden_dims=[32])

    with pytest.raises(ValueError, match="hidden_dims must be a non-empty list"):
        MLPDenoiser(input_dim=10, hidden_dims=[])

    with pytest.raises(ValueError, match="num_timesteps must be positive"):
        MLPDenoiser(input_dim=10, hidden_dims=[32], num_timesteps=0)

    denoiser = MLPDenoiser(input_dim=4, hidden_dims=[16, 16], num_timesteps=10)
    
    # Wrong input shape
    with pytest.raises(ValueError, match="Expected input x of shape"):
        denoiser(torch.randn(8, 5), torch.zeros(8))

    # Out of range timesteps should clamp without CUDA/CPU embedding crash
    out = denoiser(torch.randn(4, 4), torch.tensor([-10, 5, 100, 9]))
    assert out.shape == (4, 4)


def test_sampler_and_forward_validation():
    denoiser = MLPDenoiser(input_dim=4, hidden_dims=[16, 16], num_timesteps=10)
    schedule = LinearNoiseSchedule(num_timesteps=10)

    # Invalid num_samples
    with pytest.raises(ValueError, match="num_samples must be positive"):
        generate_samples(denoiser, schedule, num_samples=0)

    # Invalid schedule parameters
    with pytest.raises(ValueError, match="num_timesteps must be positive"):
        LinearNoiseSchedule(num_timesteps=0)
    with pytest.raises(ValueError, match="Invalid beta range"):
        LinearNoiseSchedule(beta_start=0.5, beta_end=0.1)

    # forward_diffuse invalid input
    with pytest.raises(ValueError, match="Expected 2D tensor x_0"):
        forward_diffuse(torch.randn(4), torch.zeros(4), schedule)


def test_gpu_budget_guard_directory_and_atomic(tmp_path: Path):
    nested_path = tmp_path / "deep" / "nested" / "gpu_state.json"
    guard = ComputeBudgetGuard(state_file=str(nested_path), max_hours=1.0)
    
    # Check budget creates directory and saves atomic file
    guard.check_budget()
    assert nested_path.exists()
    
    # Negative max_hours raises
    with pytest.raises(ValueError, match="max_hours must be positive"):
        ComputeBudgetGuard(state_file=str(nested_path), max_hours=-1.0)


def test_utility_and_privacy_evaluators_edge_cases():
    # Empty dataframes
    empty_df = pd.DataFrame()
    non_empty_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    
    priv = PrivacyEvaluator(empty_df, empty_df, empty_df)
    mia_res = priv.evaluate_mia_risk()
    assert mia_res["mia_risk_score"] == 0.0

    # Utility evaluator with column misalignment
    df1 = pd.DataFrame({"col_a": [1.0, 2.0, 3.0], "col_b": [4.0, 5.0, 6.0], "col_c": [7.0, 8.0, 9.0]})
    df2 = pd.DataFrame({"col_b": [4.0, 5.0, 6.0], "col_a": [1.0, 2.0, 3.0], "col_d": [10.0, 11.0, 12.0]})
    
    util = UtilityEvaluator(df1, df2)
    rmse = util.evaluate_bivariate_correlation_rmse()
    # Since col_a and col_b are identical in df1 and df2, RMSE should be 0.0
    assert abs(rmse) < 1e-6


def test_dp_trainer_checkpointing(tmp_path: Path):
    denoiser = MLPDenoiser(input_dim=4, hidden_dims=[16, 16], num_timesteps=10)
    schedule = LinearNoiseSchedule(num_timesteps=10)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    accountant = CentralPrivacyAccountant()
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=10)
    
    trainer = DPTrainer(
        denoiser=denoiser,
        schedule=schedule,
        optimizer=optimizer,
        accountant=accountant,
        privacy_schedule=privacy_schedule,
        dataset_size=100,
        tier_params={"global": list(denoiser.parameters())},
        tier_clip_norms={"global": 1.0}
    )
    
    # Record dummy step
    accountant.record_step(noise_multiplier=1.0, sample_rate=0.01)
    
    ckpt_path = tmp_path / "checkpoints" / "dp_model.pt"
    trainer.save_checkpoint(str(ckpt_path), epoch=5, loss=0.42, extra={"custom_val": 123})
    assert ckpt_path.exists()
    
    # Create fresh trainer and load checkpoint
    new_denoiser = MLPDenoiser(input_dim=4, hidden_dims=[16, 16], num_timesteps=10)
    new_optimizer = torch.optim.Adam(new_denoiser.parameters(), lr=1e-3)
    new_accountant = CentralPrivacyAccountant()
    
    new_trainer = DPTrainer(
        denoiser=new_denoiser,
        schedule=schedule,
        optimizer=new_optimizer,
        accountant=new_accountant,
        privacy_schedule=privacy_schedule,
        dataset_size=100,
        tier_params={"global": list(new_denoiser.parameters())},
        tier_clip_norms={"global": 1.0}
    )
    
    loaded = new_trainer.load_checkpoint(str(ckpt_path))
    assert loaded["epoch"] == 5
    assert loaded["loss"] == 0.42
    assert loaded["extra"]["custom_val"] == 123
    assert new_accountant.steps == 1


def test_chunked_sampling():
    denoiser = MLPDenoiser(input_dim=4, hidden_dims=[16, 16], num_timesteps=5)
    schedule = LinearNoiseSchedule(num_timesteps=5)
    
    # Generate 15 samples in chunks of 4
    samples = generate_samples(denoiser, schedule, num_samples=15, batch_size=4)
    assert samples.shape == (15, 4)


def test_resource_stats():
    stats = ComputeBudgetGuard.get_resource_stats()
    assert isinstance(stats, dict)
    if "cpu_ram_used_gb" in stats:
        assert stats["cpu_ram_used_gb"] >= 0.0


def test_schedule_to_device():
    schedule = LinearNoiseSchedule(num_timesteps=10, device=torch.device('cpu'))
    assert schedule.device == torch.device('cpu')
    schedule.to(torch.device('cpu'))
    assert schedule.get_betas().device == torch.device('cpu')
    assert schedule.get_alphas_cumprod().device == torch.device('cpu')
