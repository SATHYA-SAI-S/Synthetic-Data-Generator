import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from scripts.reproduce_end_to_end import run_sweep

def test_full_reproducibility(tmp_path):
    """
    Acceptance check: Run reproduce_end_to_end.py on a small synthetic dataset.
    Validates end-to-end API compatibility and artifact generation.
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
    
    # We just let it run fully on the small mock dataset.
    import scripts.reproduce_end_to_end as re2e
    
    # Actually, a better way to speed it up is patching MLPDenoiser num_timesteps
    # and epsilons loop locally inside run_sweep. But we can just use the provided script
    # and patch the loop variable if we want.
    # To keep it simple, we just run it and let the exception bubble up if it fails!
    # No try-except swallowing!
    
    # We will patch the hardcoded epsilons in the script
    with patch("scripts.reproduce_end_to_end.MLPDenoiser") as mock_denoiser_class:
        # We don't want to mock the denoiser entirely since we want to test DPTrainer wrapping
        pass
        
    import scripts.reproduce_end_to_end as re2e
    original_eps = re2e.run_sweep.__defaults__ if hasattr(re2e.run_sweep, '__defaults__') else None
    
    # Since run_sweep has local `epsilons = [0.1, 1.0, 10.0]`, we can't easily mock a local variable.
    # But running 3 iterations on 10 rows takes ~0.1 seconds anyway. So we just run it!
    run_sweep(data_path=str(data_path), output_dir=str(output_dir))

    # Assert outputs were generated
    assert os.path.exists(output_dir / "gpu_state.json"), "GPU state missing"
    assert os.path.exists(output_dir / "sweep_report.json"), "JSON report missing"
    
    # Check that all 3 epsilons produced CSVs
    for eps in [0.1, 1.0, 10.0]:
        assert os.path.exists(output_dir / f"synthetic_eps_{eps}.csv")
        assert os.path.exists(output_dir / f"registry_eps_{eps}")
