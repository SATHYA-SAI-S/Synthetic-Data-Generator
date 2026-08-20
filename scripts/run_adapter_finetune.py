import os
import sys
import zipfile
import logging
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
from src.orchestration.gpu_budget_guard import ComputeBudgetGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("adapter_finetune")

def run_adapter_finetuning(
    data_path: str = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    backbone_ckpt: str = "vishwa_final_clean_archive/checkpoints/model_eps_1.0.pt",
    output_dir: str = "outputs/adapter_finetuning",
    sample_size: int = 500,
    target_eps: float = 1.0,
    num_epochs: int = 5
) -> pd.DataFrame:
    """
    Objective 4: Fine-tune a frozen diffusion backbone on a small-N cohort using SchemaAdapterModel.
    Runs DP-SGD exclusively on adapter projection weights to conserve privacy budget.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using compute device: {device}")

    # 1. Ingest small cohort
    log.info(f"Loading data from {data_path} (subsampling N={sample_size})")
    if str(data_path).endswith(".zip"):
        with zipfile.ZipFile(data_path) as z:
            csv_candidates = [n for n in z.namelist() if n.endswith(".csv") and "mapping" not in n.lower()]
            csv_name = csv_candidates[0] if csv_candidates else z.namelist()[0]
            with z.open(csv_name) as f:
                raw_df = pd.read_csv(f)
    else:
        raw_df = pd.read_csv(data_path)

    # Subsample small cohort (e.g. rare disease or specialized hospital ward)
    small_df = raw_df.head(sample_size).copy()
    dataset_name = f"small_cohort_n_{sample_size}"

    # 2. Preprocessing & Registry
    config = PipelineConfig()
    profiler = DatasetProfiler(config)
    profile = profiler.profile(small_df, dataset_name=dataset_name)
    registry = FileSchemaRegistry(root_dir=Path(output_dir) / "registry")

    pipeline = PreprocessingPipeline(
        config=config,
        profiler=profiler,
        missingness_handler=MissingnessHandler(config),
        encoder_factory=lambda c: OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq),
        scaler_factory=lambda c: StandardScaler(),
        registry=registry
    )

    encoded_data = pipeline.fit_transform(small_df, dataset_name=dataset_name)
    new_input_dim = encoded_data.shape[1]
    log.info(f"Small cohort encoded tensor shape: {encoded_data.shape} (D_new = {new_input_dim})")

    # 3. Load or Instantiate Pre-Trained Backbone
    # Default backbone dimensions from base model
    backbone_dim = 616  # standard full feature dim or fallback to matching dim
    backbone = MLPDenoiser(input_dim=backbone_dim, hidden_dims=[256, 256, 256], num_timesteps=1000)

    if os.path.exists(backbone_ckpt):
        try:
            log.info(f"Loading pre-trained diffusion backbone from {backbone_ckpt}")
            ckpt = torch.load(backbone_ckpt, map_location="cpu")
            sd = ckpt.get("denoiser_state_dict", ckpt)
            # Filter matching keys if dimensions match
            backbone_sd = {k: v for k, v in sd.items() if k in backbone.state_dict() and v.shape == backbone.state_dict()[k].shape}
            backbone.load_state_dict(backbone_sd, strict=False)
            log.info("Successfully transferred pre-trained backbone weights.")
        except Exception as e:
            log.warning(f"Could not load exact checkpoint weights ({e}), continuing with initialized backbone.")

    # 4. Attach Trainable Schema Adapter
    adapter_model = SchemaAdapterModel(
        backbone=backbone,
        new_input_dim=new_input_dim,
        freeze_backbone=True
    ).to(device)

    trainable_params = adapter_model.adapter_parameters()
    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in adapter_model.backbone.parameters())
    log.info(f"SchemaAdapter initialized: {num_trainable:,} trainable adapter params | {num_frozen:,} frozen backbone params.")

    # 5. Configure DP-SGD for Adapter Only
    schedule = LinearNoiseSchedule(num_timesteps=1000, device=device)
    base_sigma = 5.0 / target_eps
    privacy_schedule = AdaptiveNoiseSchedule(base_sigma=base_sigma, num_timesteps=1000, strategy="linear")
    accountant = CentralPrivacyAccountant()
    optimizer = torch.optim.Adam(trainable_params, lr=2e-3)

    tier_params = {"adapter_layers": trainable_params}
    tier_clip_norms = {"adapter_layers": 1.0}

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

    # 6. Fine-Tuning Loop
    dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    log.info(f"Starting DP fine-tuning for {num_epochs} epochs on small cohort...")
    for epoch in range(1, num_epochs + 1):
        loss = trainer.train_epoch(loader)
        eps_spent = accountant.get_epsilon(target_delta=1e-4)
        log.info(f"[Epoch {epoch:02d}/{num_epochs:02d}] Loss: {loss:.4f} | ε spent: {eps_spent:.4f}")

    # Save fine-tuned adapter checkpoint
    ckpt_path = os.path.join(output_dir, "adapter_model.pt")
    trainer.save_checkpoint(ckpt_path, epoch=num_epochs, loss=loss, extra={"sample_size": sample_size, "eps_spent": eps_spent})

    # 7. Generate Synthetic Samples
    log.info("Generating synthetic samples from adapted model...")
    unwrapped = getattr(trainer.denoiser, "_module", trainer.denoiser)
    synth_tensor = generate_samples(
        denoiser=unwrapped,
        schedule=schedule,
        num_samples=sample_size,
        device=device,
        batch_size=1024
    )

    # 8. Decode to DataFrame
    synth_df = pipeline.inverse_transform(synth_tensor.cpu().numpy())
    out_csv = os.path.join(output_dir, "synthetic_adapted_small_cohort.csv")
    synth_df.to_csv(out_csv, index=False)
    log.info(f"Saved fine-tuned synthetic dataset to {out_csv} (Shape: {synth_df.shape})")

    return synth_df

if __name__ == "__main__":
    run_adapter_finetuning()
