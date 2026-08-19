import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_full_reproducibility(tmp_path):
    """
    Acceptance check: Run reproduce_end_to_end.py on a small synthetic dataset.
    Validates end-to-end API compatibility and artifact generation.
    
    FIX (K-03): This test was previously a no-op that swallowed all exceptions.
    Now it actually runs the pipeline and asserts outputs are generated.
    """
    # Create tiny mock CSV
    df = pd.DataFrame({
        "encounter_id": np.arange(1, 101),
        "age": np.random.uniform(20.0, 90.0, 100),
        "diagnosis": np.random.choice(["A", "B", "C"], 100)
    })
    
    data_path = tmp_path / "mock_data.csv"
    output_dir = tmp_path / "outputs"
    df.to_csv(data_path, index=False)
    
    # Import the run_sweep function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root
    from scripts.reproduce_end_to_end import run_sweep
    
    # Use a small config to keep the test fast
    from src.config.schema import PipelineConfig, DiffusionConfig, TrainingConfig
    small_config = PipelineConfig(
        diffusion=DiffusionConfig(num_timesteps=10, hidden_dims=[16, 16]),
        training=TrainingConfig(epochs=1, batch_size=64),
    )
    
    # Run the sweep on the tiny mock dataset
    run_sweep(
        data_path=str(data_path),
        output_dir=str(output_dir),
        config=small_config,
        epsilons=[1.0],
    )

    # Assert outputs were generated
    assert os.path.exists(output_dir / "gpu_state.json"), "GPU state missing"
    assert os.path.exists(output_dir / "sweep_report.json"), "JSON report missing"
    
    # Check that the requested epsilon produced CSVs
    assert os.path.exists(output_dir / "synthetic_eps_1.0.csv")
    assert os.path.exists(output_dir / "registry_eps_1.0")
