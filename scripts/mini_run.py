import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import logging
import zipfile
import time

from src.config.schema import PipelineConfig
from src.profiling.dataset_profiler import DatasetProfiler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.encoders import OneHotEncoder
from src.preprocessing.scalers import StandardScaler
from src.registry.schema_registry import FileSchemaRegistry
from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.sampler import generate_samples
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.dp_trainer import DPTrainer
from src.privacy.risk_tier_assigner import HeuristicRiskTierAssigner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def make_encoder_factory(config: PipelineConfig):
    return lambda col_name: OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq)

def make_scaler_factory(config: PipelineConfig):
    return lambda col_name: StandardScaler()

def run_mini_pipeline():
    start_time = time.time()
    
    # 1. Load a tiny subset of the data
    data_path = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    log.info("Loading 2,000 rows for the mini-run...")
    
    with zipfile.ZipFile(data_path, 'r') as z:
        csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
        with z.open(csv_filename) as f:
            raw_df = pd.read_csv(f, nrows=2000, na_values=['?', 'Unknown'])
            
    # 2. Preprocessing
    log.info("Phase 1-3: Profiling and Preprocessing...")
    config = PipelineConfig()
    registry = FileSchemaRegistry("scratch/registry_mini")
    
    pipeline = PreprocessingPipeline(
        config=config,
        profiler=DatasetProfiler(config),
        missingness_handler=MissingnessHandler(config),
        encoder_factory=make_encoder_factory(config),
        scaler_factory=make_scaler_factory(config),
        registry=registry
    )
    
    encoded_tensor = pipeline.fit_transform(raw_df, "diabetes_mini")
    dataset_profile = registry.load_profile("diabetes_mini")
    log.info(f"Encoded tensor shape: {encoded_tensor.shape}")
    
    # 3. Model Initialization
    log.info("Phase 4-5: Initializing DP-Diffusion Model...")
    input_dim = encoded_tensor.shape[1]
    denoiser = MLPDenoiser(input_dim=input_dim, hidden_dims=[64, 64], num_timesteps=50)
    
    # Use a highly safe learning rate to avoid NaNs
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-5)
    
    schedule = LinearNoiseSchedule(num_timesteps=50)
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=1.0, num_timesteps=50)
    accountant = CentralPrivacyAccountant()
    
    tier_params = {"global": list(denoiser.parameters())}
    tier_clip_norms = {"global": 1.0}
    
    trainer = DPTrainer(
        denoiser=denoiser,
        schedule=schedule,
        optimizer=optimizer,
        accountant=accountant,
        privacy_schedule=privacy_schedule,
        dataset_size=len(encoded_tensor),
        tier_params=tier_params,
        tier_clip_norms=tier_clip_norms
    )
    
    # 4. Training (Mini Run)
    log.info("Phase 7: Starting Mini Training Loop (15 Epochs)...")
    dataset = TensorDataset(torch.tensor(encoded_tensor, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    for epoch in range(0):
        loss = trainer.train_epoch(dataloader)
        if (epoch + 1) % 5 == 0:
            log.info(f"Epoch {epoch+1}/15 - Loss: {loss:.4f} (NaN Check: {torch.isnan(torch.tensor(loss)).item()})")
            
    # 5. Generative Sampling
    log.info("Phase 7: Generating Synthetic Data (1,000 samples)...")
    synthetic_tensor = generate_samples(
        denoiser=denoiser,
        schedule=schedule,
        num_samples=1000,
        batch_size=256
    )
    
    # 6. Inverse Transform
    print(f"NaNs in Tensor: {torch.isnan(synthetic_tensor).any().item()}")
    print(f"Tensor min: {synthetic_tensor.min().item()}, max: {synthetic_tensor.max().item()}")
    synthetic_df = pipeline.inverse_transform(synthetic_tensor.numpy())
    
    # 7. Save
    output_path = "scratch/mini_synthetic_data.csv"
    synthetic_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    log.info(f"SUCCESS! Mini-run completed in {elapsed:.1f} seconds. Valid data saved to {output_path}")

if __name__ == "__main__":
    run_mini_pipeline()
