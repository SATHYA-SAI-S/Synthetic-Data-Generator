import os
import zipfile
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
import json

from src.config.schema import PipelineConfig
from src.profiling.dataset_profiler import DatasetProfiler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.encoders import OneHotEncoder
from src.preprocessing.scalers import StandardScaler
from src.registry.schema_registry import FileSchemaRegistry

def make_encoder_factory(config: PipelineConfig):
    return lambda col_name: OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq)

def make_scaler_factory(config: PipelineConfig):
    return lambda col_name: StandardScaler()

from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.sampler import generate_samples
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.risk_tier_assigner import HeuristicRiskTierAssigner
from src.privacy.dp_trainer import DPTrainer
from src.orchestration.gpu_budget_guard import ComputeBudgetGuard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_sweep(
    data_path: str,
    output_dir: str,
    config: PipelineConfig = None,
    epsilons: list = None,
) -> None:
    """
    Run the end-to-end DP-diffusion pipeline sweep across epsilon budgets.

    Args:
        data_path: Path to input data (CSV or zip containing diabetic_data.csv).
        output_dir: Directory to write outputs.
        config: Optional PipelineConfig override. Defaults to PipelineConfig().
        epsilons: Optional list of target epsilon values. Defaults to [0.1, 1.0, 10.0].
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize GPU Guard
    guard = ComputeBudgetGuard(state_file=os.path.join(output_dir, "gpu_state.json"))
    
    # 2. Load Raw Data
    log.info(f"Loading data from {data_path}")
    if data_path.endswith('.zip'):
        with zipfile.ZipFile(data_path, 'r') as z:
            with z.open('diabetic_data.csv') as f:
                raw_df = pd.read_csv(f)
    else:
        raw_df = pd.read_csv(data_path)
    
    if epsilons is None:
        epsilons = [0.1, 1.0, 10.0]
    if config is None:
        config = PipelineConfig()
    
    num_timesteps = config.diffusion.num_timesteps
    hidden_dims = config.diffusion.hidden_dims
    num_epochs = config.training.epochs
    results_report = []
    
    for target_eps in epsilons:
        log.info(f"=== Starting Sweep: Epsilon = {target_eps} ===")
        guard.check_budget()
        
        # 4. Profile & Preprocess
        dataset_name = f"diabetes_eps_{target_eps}"
        profiler = DatasetProfiler(config)
        profile = profiler.profile(raw_df, dataset_name=dataset_name)
        
        registry = FileSchemaRegistry(root_dir=Path(output_dir) / f"registry_eps_{target_eps}")
        
        handler = MissingnessHandler(config)
        encoder_factory = make_encoder_factory(config)
        scaler_factory = make_scaler_factory(config)
        
        pipeline = PreprocessingPipeline(
            config=config, 
            profiler=profiler,
            missingness_handler=handler,
            encoder_factory=encoder_factory,
            scaler_factory=scaler_factory,
            registry=registry
        )
        
        encoded_data = pipeline.fit_transform(raw_df, dataset_name=dataset_name)
        input_dim = encoded_data.shape[1]
        
        # 5. Risk Tier Assignment
        assigner = HeuristicRiskTierAssigner()
        tiers = assigner.assign_tiers(raw_df, profile.columns)
        
        # 6. Initialize Diffusion & DP Components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Training device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        denoiser = MLPDenoiser(input_dim=input_dim, hidden_dims=hidden_dims, num_timesteps=num_timesteps)
        
        tier_params = {"global": list(denoiser.parameters())}
        tier_clip_norms = {"global": 1.0} 
        
        schedule = LinearNoiseSchedule(num_timesteps=num_timesteps, device=device)
        
        # C-2 FIX: cap the noise multiplier. Previously sigma = 15/eps gave sigma=150
        # for eps=0.1 — pure-noise training that diverged to NaN. A capped sigma plus
        # the closed-loop epsilon stop below reaches the target budget safely.
        MAX_SIGMA = 4.0
        base_sigma = min(15.0 / target_eps, MAX_SIGMA)
        privacy_schedule = AdaptiveNoiseSchedule(base_sigma=base_sigma, num_timesteps=num_timesteps, strategy="linear")
        accountant = CentralPrivacyAccountant()
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
        
        trainer = DPTrainer(
            denoiser=denoiser,
            schedule=schedule,
            optimizer=optimizer,
            accountant=accountant,
            privacy_schedule=privacy_schedule,
            dataset_size=len(encoded_data),
            tier_params=tier_params,
            tier_clip_norms=tier_clip_norms,
            device=device
        )
        
        # 7. Train
        dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
        
        loss = 0.0
        eps_spent = 0.0
        for epoch in tqdm(range(num_epochs), desc=f"Training eps={target_eps}"):
            loss = trainer.train_epoch(loader)
            eps_spent = accountant.get_epsilon(target_delta=1e-5)
            # P-1 FIX: closed-loop epsilon stop — never train past the target budget.
            if eps_spent >= target_eps:
                log.info(f"Target epsilon {target_eps} reached at epoch {epoch+1} "
                         f"(eps spent={eps_spent:.4f}). Stopping training.")
                break
            
        # Save model checkpoint
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, f"model_eps_{target_eps}.pt")
        trainer.save_checkpoint(ckpt_path, epoch=num_epochs, loss=loss, extra={"target_epsilon": target_eps, "eps_spent": eps_spent})
        
        stats = guard.get_resource_stats()
        log.info(f"Training finished. Final Loss: {loss:.4f}, Epsilon spent: {eps_spent:.4f}")
        if stats:
            log.info(f"Resource Usage: CPU RAM: {stats.get('cpu_ram_used_gb', 'N/A')}/{stats.get('cpu_ram_total_gb', 'N/A')} GB | GPU VRAM: {stats.get('gpu_vram_allocated_gb', 'N/A')} GB")
        
        # 8. Generation
        log.info("Generating synthetic samples...")
        unwrapped_denoiser = getattr(trainer.denoiser, "_module", trainer.denoiser)
        synthetic_tensor = generate_samples(
            denoiser=unwrapped_denoiser, 
            schedule=schedule, 
            num_samples=len(raw_df), 
            device=device,
            batch_size=8192
        )
        
        # 9. Decoding
        log.info("Inverse transforming to CSV...")
        synthetic_df = pipeline.inverse_transform(synthetic_tensor.cpu().numpy())
        
        # C-3/C-4 FIX: output validation gate — refuse to export garbage.
        n_all_nan_cols = int(synthetic_df.isna().all().sum())
        if n_all_nan_cols > 0:
            raise ValueError(
                f"Generation failed: {n_all_nan_cols}/{synthetic_df.shape[1]} columns are "
                "entirely NaN. The model likely diverged; refusing to export."
            )
        uniqueness = synthetic_df.drop_duplicates().shape[0] / max(1, len(synthetic_df))
        if uniqueness < 0.01:
            raise ValueError(
                f"Generation failed: synthetic row uniqueness {uniqueness:.4f} < 0.01 "
                "(model collapse). Refusing to export."
            )
        
        out_file = os.path.join(output_dir, f"synthetic_eps_{target_eps}.csv")
        synthetic_df.to_csv(out_file, index=False)
        log.info(f"Saved: {out_file}\n")
        
        results_report.append({
            "target_epsilon": target_eps,
            "actual_epsilon": eps_spent,
            "loss": loss
        })
        
        guard.check_budget()
        
    with open(os.path.join(output_dir, "sweep_report.json"), "w") as f:
        json.dump(results_report, f, indent=4)

if __name__ == "__main__":
    run_sweep(data_path="data/diabetes+130-us+hospitals+for+years+1999-2008.zip", output_dir="outputs/sweep_results")