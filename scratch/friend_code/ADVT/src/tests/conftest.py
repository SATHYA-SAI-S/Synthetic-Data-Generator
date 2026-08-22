"""
tests/conftest.py — Shared pytest fixtures for all Phase 3 tests.

Fixtures here are available to all test files without import.
All fixtures use synthetic data with explicit column names — no
real dataset column names appear in any source file outside tests/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.schema import PipelineConfig


# ── Config ────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig.default()


@pytest.fixture
def low_card_threshold_config() -> PipelineConfig:
    """Config with very low cardinality threshold for testing edge cases."""
    from src.config.schema import CardinalityConfig
    return PipelineConfig(
        cardinality=CardinalityConfig(low_card_max=3, rare_category_min_freq=2)
    )


# ── Synthetic DataFrames ───────────────────────────────────────────────────────

@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """
    A tiny synthetic DataFrame with:
    - 'age':      continuous
    - 'bmi':      continuous with missingness
    - 'diagnosis': low-cardinality categorical
    - 'region':    high-cardinality categorical
    - 'readmitted': binary
    """
    rng = np.random.default_rng(42)
    n = 200

    age = rng.uniform(18, 90, n)
    bmi = rng.uniform(15, 50, n)
    bmi[rng.choice(n, size=20, replace=False)] = np.nan  # ~10% missing

    diagnosis = rng.choice(["TypeA", "TypeB", "TypeC", "TypeD"], n)
    # Create high-cardinality: 50 regions, some rare
    regions = [f"region_{i}" for i in range(50)]
    region_probs = np.ones(50)
    region_probs[:5] = 10  # first 5 are frequent
    region_probs = region_probs / region_probs.sum()
    region = rng.choice(regions, n, p=region_probs)

    readmitted = rng.choice(["Yes", "No"], n)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "diagnosis": diagnosis,
        "region": region,
        "readmitted": readmitted,
    })


@pytest.fixture
def df_with_hipaa_columns(tiny_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with HIPAA identifier column names added."""
    n = len(tiny_df)
    df = tiny_df.copy()
    df["patient_id"] = range(n)
    df["date_of_birth"] = pd.date_range("1950-01-01", periods=n, freq="D").astype(str)
    df["zip"] = [f"{50000 + i:05d}" for i in range(n)]
    return df


@pytest.fixture
def continuous_series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(100.0, 15.0, 500), name="test_continuous")


@pytest.fixture
def continuous_series_with_nan() -> pd.Series:
    rng = np.random.default_rng(1)
    vals = rng.normal(100.0, 15.0, 500).tolist()
    vals[0] = np.nan
    vals[100] = np.nan
    vals[250] = np.nan
    return pd.Series(vals, name="test_with_nan")


@pytest.fixture
def low_card_series() -> pd.Series:
    rng = np.random.default_rng(2)
    return pd.Series(
        rng.choice(["cat_A", "cat_B", "cat_C"], 200),
        name="low_card_col"
    )


@pytest.fixture
def high_card_series() -> pd.Series:
    """50 unique categories; only 5 are frequent."""
    rng = np.random.default_rng(3)
    cats = [f"cat_{i}" for i in range(50)]
    probs = np.ones(50)
    probs[:5] = 30
    probs = probs / probs.sum()
    return pd.Series(
        rng.choice(cats, 300, p=probs),
        name="high_card_col"
    )


@pytest.fixture
def registry_root(tmp_path: Path) -> Path:
    """Temporary directory for registry tests."""
    return tmp_path / "registry"
