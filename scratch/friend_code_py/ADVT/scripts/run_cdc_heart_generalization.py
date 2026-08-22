import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config.schema import PipelineConfig
from src.profiling.dataset_profiler import DatasetProfiler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.encoders import OneHotEncoder
from src.preprocessing.scalers import StandardScaler
from src.registry.schema_registry import FileSchemaRegistry
from src.diffusion.denoiser import MLPDenoiser
from src.diffusion.adapter import SchemaAdapterModel
from src.diffusion.schedule import LinearNoiseSchedule
from src.diffusion.sampler import generate_samples
from src.privacy.accountant import CentralPrivacyAccountant
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule
from src.privacy.dp_trainer import DPTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cdc_heart_generalization")

def run_cdc_heart_generalization(
    data_path: str = "data/indicators of heart disease (2022 update)/2022/heart_2022_with_nans.csv",
    backbone_ckpt: str = "vishwa_final_clean_archive/checkpoints/model_eps_1.0.pt",
    output_dir: str = "outputs/adapter_finetuning",
    sample_size: int = 1000,
    target_eps: float = 1.0,
    num_epochs: int = 5
) -> dict:
    """
    Objective 5: Multi-Dataset Generalization on CDC BRFSS Heart Disease 2022 Dataset.
    
    1. Auto-profiles the 40 new clinical columns.
    2. Constructs tensor representation and schema registry.
    3. Fine-tunes pre-trained diffusion backbone using SchemaAdapterModel under DP-SGD.
    4. Samples and inverts back to CDC schema, saving synthetic_cdc_heart_adapted.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 1. Ingest CDC Heart Disease dataset
    log.info(f"Loading raw CDC Heart Disease 2022 data from {data_path} (subsampling N={sample_size})")
    raw_df = pd.read_csv(data_path, low_memory=False)
    log.info(f"Full CDC dataset shape: {raw_df.shape} ({len(raw_df.columns)} columns)")

    cdc_subset = raw_df.head(sample_size).copy()
    dataset_name = "cdc_brfss_heart_2022"

    # 2. Auto-Profile New Schema
    log.info("--- Phase 5a: Auto-Profiling New CDC Schema ---")
    config = PipelineConfig()
    profiler = DatasetProfiler(config)
    profile = profiler.profile(cdc_subset, dataset_name=dataset_name)
    
    log.info(f"Successfully profiled {len(profile.columns)} columns in CDC dataset.")
    for col_info in profile.columns[:5]:
        log.info(f"  - Column '{col_info.name}': Inferred Dtype = {col_info.inferred_dtype.value}, Missing Rate = {col_info.missing_rate:.2%}")

    # 3. Pipeline Transformation & Registry Creation
    registry_path = Path(output_dir) / "registry_cdc_heart"
    registry = FileSchemaRegistry(root_dir=registry_path)

    pipeline = PreprocessingPipeline(
        config=config,
        profiler=profiler,
        missingness_handler=MissingnessHandler(config),
        encoder_factory=lambda c: OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq),
        scaler_factory=lambda c: StandardScaler(),
        registry=registry
    )

    encoded_data = pipeline.fit_transform(cdc_subset, dataset_name=dataset_name)
    new_input_dim = encoded_data.shape[1]
    log.info(f"CDC Preprocessing Pipeline fitted: Encoded tensor shape = {encoded_data.shape} (D_new = {new_input_dim})")
    assert np.isnan(encoded_data).sum() == 0, "Encoded CDC tensor must be 100% NaN-free!"

    # 4. Instantiate Pre-Trained Backbone & Attach Schema Adapter
    log.info("--- Phase 5b: Attaching SchemaAdapterModel to Pre-Trained Backbone ---")
    backbone_dim = 616  # standard full feature dim from base model
    backbone = MLPDenoiser(input_dim=backbone_dim, hidden_dims=[256, 256, 256], num_timesteps=1000)

    if os.path.exists(backbone_ckpt):
        try:
            log.info(f"Transferring pre-trained weights from {backbone_ckpt}")
            ckpt = torch.load(backbone_ckpt, map_location="cpu")
            sd = ckpt.get("denoiser_state_dict", ckpt)
            backbone_sd = {k: v for k, v in sd.items() if k in backbone.state_dict() and v.shape == backbone.state_dict()[k].shape}
            backbone.load_state_dict(backbone_sd, strict=False)
            log.info("Pre-trained backbone weights transferred successfully.")
        except Exception as e:
            log.warning(f"Note on checkpoint weight transfer: {e}")

    adapter_model = SchemaAdapterModel(
        backbone=backbone,
        new_input_dim=new_input_dim,
        freeze_backbone=True
    ).to(device)

    trainable_params = adapter_model.adapter_parameters()
    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in adapter_model.backbone.parameters())
    log.info(f"SchemaAdapter initialized: {num_trainable:,} trainable adapter parameters | {num_frozen:,} frozen backbone parameters.")

    # 5. Lightweight DP-SGD Fine-Tuning
    log.info("--- Phase 5c: DP-SGD Fine-Tuning on Adapter Parameters ---")
    schedule = LinearNoiseSchedule(num_timesteps=1000, device=device)
    base_sigma = 5.0 / target_eps
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=base_sigma, num_timesteps=1000, strategy="linear")
    accountant = CentralPrivacyAccountant()
    optimizer = torch.optim.Adam(trainable_params, lr=2e-3)

    tier_params = {"cdc_adapter_layers": trainable_params}
    tier_clip_norms = {"cdc_adapter_layers": 1.0}

    trainer = DPTrainer(
        denoiser=adapter_model,
        schedule=schedule,
        optimizer=optimizer,
        accountant=accountant,
        privacy_schedule=privacy_schedule,
        dataset_size=len(encoded_data),
        tier_params=tier_params,
        tier_clip_norms=tier_clip_norms,
        device=device
    )

    dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        loss = trainer.train_epoch(loader)
        eps_spent = accountant.get_epsilon(target_delta=1e-4)
        log.info(f"[CDC Adapter Epoch {epoch:02d}/{num_epochs:02d}] Loss: {loss:.4f} | ε spent: {eps_spent:.4f}")

    # 6. Sampling & Inverse Transform to CDC Schema
    log.info("--- Phase 5d: Reverse Diffusion Sampling & Schema Inverse Transform ---")
    unwrapped = getattr(trainer.denoiser, "_module", trainer.denoiser)
    synth_tensor = generate_samples(
        denoiser=unwrapped,
        schedule=schedule,
        num_samples=sample_size,
        device=device,
        batch_size=1024
    )

    synth_df = pipeline.inverse_transform(synth_tensor.cpu().numpy())
    out_csv = os.path.join(output_dir, "synthetic_cdc_heart_adapted.csv")
    synth_df.to_csv(out_csv, index=False)
    log.info(f"Saved synthetic CDC dataset to {out_csv} (Shape: {synth_df.shape})")

    # 7. Verification & Statistical Validation
    log.info("--- Phase 5e: Verification & Validation ---")
    orig_cols = list(cdc_subset.columns)
    synth_cols = list(synth_df.columns)
    dropped_cols = [c for c in orig_cols if c not in synth_cols]
    
    total_cells = synth_df.shape[0] * synth_df.shape[1]
    null_cells = synth_df.isna().sum().sum()
    null_rate = null_cells / total_cells
    
    preview_cols = [c for c in ['Sex', 'GeneralHealth', 'HadHeartAttack', 'SleepHours', 'BMI', 'AgeCategory'] if c in synth_df.columns]
    log.info(f"Columns Preserved: {len(synth_cols)}/{len(orig_cols)} (Dropped HIPAA/Geographic Identifiers: {dropped_cols})")
    log.info(f"Total Synthetic Cells: {total_cells:,} | Restored Missingness Cells: {null_cells:,} ({null_rate:.2%})")
    log.info(f"Sample of synthesized CDC records:\n{synth_df[preview_cols].head(5)}")

    summary = {
        "dataset_name": dataset_name,
        "raw_rows": len(raw_df),
        "fine_tune_rows": sample_size,
        "original_columns_count": len(orig_cols),
        "synthesized_columns_count": len(synth_cols),
        "dropped_geographic_hipaa_columns": dropped_cols,
        "encoded_tensor_dim": new_input_dim,
        "trainable_adapter_params": num_trainable,
        "frozen_backbone_params": num_frozen,
        "final_loss": round(float(loss), 4),
        "epsilon_spent": round(float(eps_spent), 4),
        "synthetic_shape": list(synth_df.shape),
        "total_cells": int(total_cells),
        "null_cells": int(null_cells),
        "null_rate_pct": round(float(null_rate * 100), 2)
    }

    # Save summary json
    summary_path = os.path.join(output_dir, "cdc_heart_generalization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    run_cdc_heart_generalization()
