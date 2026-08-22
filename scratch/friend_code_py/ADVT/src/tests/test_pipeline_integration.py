"""
tests/test_pipeline_integration.py — Full Phase 3 pipeline integration test.

Runs the complete pipeline (profiler -> missingness -> encoders/scalers ->
registry) on a synthetic tiny CSV, then verifies:
  1. fit_transform produces a 2D float32 numpy array
  2. transform on new data matches expected shape
  3. inverse_transform recovers a DataFrame with original column names
  4. Registry is populated and load() reconstructs a working entry
  5. Pipeline raises on schema violations (missing columns at transform time)
  6. fit_transform_from_file works end-to-end with an actual CSV on disk
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.schema import PipelineConfig
from src.preprocessing.encoders import FrequencyEncoder, OneHotEncoder
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.scalers import StandardScaler
from src.profiling.base import InferredDtype
from src.profiling.dataset_profiler import DatasetProfiler
from src.registry.schema_registry import FileSchemaRegistry


# ── Factories for the integration test ────────────────────────────────────────

def _make_encoder_factory(config: PipelineConfig):
    """
    Returns an EncoderFactory function.
    Selects OneHotEncoder vs FrequencyEncoder based on column profile.
    In integration tests we default to OneHotEncoder for simplicity.
    """
    def factory(col_name: str) -> OneHotEncoder:
        return OneHotEncoder(min_freq=config.cardinality.rare_category_min_freq)
    return factory


def _make_scaler_factory(config: PipelineConfig):
    def factory(col_name: str) -> StandardScaler:
        return StandardScaler()
    return factory


def _build_pipeline(
    config: PipelineConfig, registry_root: Path
) -> PreprocessingPipeline:
    return PreprocessingPipeline(
        config=config,
        profiler=DatasetProfiler(config=config),
        missingness_handler=MissingnessHandler(config=config),
        encoder_factory=_make_encoder_factory(config),
        scaler_factory=_make_scaler_factory(config),
        registry=FileSchemaRegistry(registry_root),
    )


# ── Integration tests ─────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_fit_transform_returns_float32_array(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        result = pipeline.fit_transform(tiny_df, "integration_test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.shape[0] == len(tiny_df)
        assert result.shape[1] > 0

    def test_fit_transform_no_nan_in_output(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        """After missingness handling, the encoded array must contain no NaN."""
        pipeline = _build_pipeline(default_config, registry_root)
        result = pipeline.fit_transform(tiny_df, "nan_check_test")
        assert not np.any(np.isnan(result)), (
            "Encoded output must not contain NaN — missingness handler should have "
            "imputed all missing values before encoding."
        )

    def test_transform_matches_shape(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        train_encoded = pipeline.fit_transform(tiny_df, "shape_test")

        # Use a different slice as "new data" for transform
        new_df = tiny_df.iloc[:50].copy()
        new_encoded = pipeline.transform(new_df)

        assert new_encoded.shape[1] == train_encoded.shape[1], (
            f"transform() output width {new_encoded.shape[1]} != "
            f"fit_transform() output width {train_encoded.shape[1]}"
        )
        assert new_encoded.shape[0] == 50

    def test_inverse_transform_returns_dataframe(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        encoded = pipeline.fit_transform(tiny_df, "inv_test")
        decoded = pipeline.inverse_transform(encoded)
        assert isinstance(decoded, pd.DataFrame)
        assert len(decoded) == len(tiny_df)

    def test_inverse_transform_column_names(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        """Decoded DataFrame must contain the original (non-indicator) column names."""
        pipeline = _build_pipeline(default_config, registry_root)
        pipeline.fit_transform(tiny_df, "colname_test")
        encoded = pipeline.transform(tiny_df)
        decoded = pipeline.inverse_transform(encoded)

        # Only check non-indicator columns (indicator columns are stripped by inverse_transform)
        original_cols = [c for c in pipeline._training_columns
                         if "__missing_flag" not in c]
        for col in original_cols:
            assert col in decoded.columns, (
                f"Column '{col}' missing from inverse_transform output"
            )

    def test_registry_populated_after_fit(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        pipeline.fit_transform(tiny_df, "registry_pop_test")

        registry = FileSchemaRegistry(registry_root)
        assert "registry_pop_test" in registry.list_datasets()
        entry = registry.load("registry_pop_test")
        assert entry.profile.n_rows == len(tiny_df)

    def test_get_profile_returns_dataset_profile(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        from src.profiling.base import DatasetProfile
        pipeline = _build_pipeline(default_config, registry_root)
        pipeline.fit_transform(tiny_df, "profile_test")
        profile = pipeline.get_profile()
        assert isinstance(profile, DatasetProfile)
        assert profile.n_rows == len(tiny_df)

    def test_get_profile_raises_before_fit(
        self, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.get_profile()

    def test_transform_raises_on_missing_column(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        pipeline = _build_pipeline(default_config, registry_root)
        pipeline.fit_transform(tiny_df, "col_check_test")

        # Drop a column that was used in training
        broken_df = tiny_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="age"):
            pipeline.transform(broken_df)

    def test_hipaa_columns_dropped_from_training(
        self,
        df_with_hipaa_columns: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        """HIPAA identifier columns must not appear in training_columns."""
        pipeline = _build_pipeline(default_config, registry_root)
        pipeline.fit_transform(df_with_hipaa_columns, "hipaa_drop_test")
        for hipaa_col in ["patient_id", "date_of_birth", "zip"]:
            assert hipaa_col not in pipeline._training_columns, (
                f"HIPAA column '{hipaa_col}' should be excluded from training"
            )

    def test_fit_transform_from_file(
        self,
        tiny_df: pd.DataFrame,
        default_config: PipelineConfig,
        registry_root: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: write CSV to disk, run fit_transform_from_file."""
        csv_path = tmp_path / "test_data.csv"
        tiny_df.to_csv(csv_path, index=False)

        pipeline = _build_pipeline(default_config, registry_root)
        result = pipeline.fit_transform_from_file(csv_path, "file_test")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(tiny_df)

    def test_full_roundtrip_continuous_values(
        self,
        default_config: PipelineConfig,
        registry_root: Path,
    ) -> None:
        """
        For a DataFrame with only continuous columns and no missingness,
        the full pipeline round-trip (fit_transform -> inverse_transform)
        must recover the original values within float tolerance.
        """
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "x1": rng.normal(50, 10, n),
            "x2": rng.uniform(0, 100, n),
            "x3": rng.exponential(5, n),
        })

        pipeline = _build_pipeline(default_config, registry_root)
        encoded = pipeline.fit_transform(df, "continuous_roundtrip_test")
        decoded = pipeline.inverse_transform(encoded)

        for col in ["x1", "x2", "x3"]:
            if col in decoded.columns:
                original = df[col].to_numpy(dtype=float)
                recovered = pd.to_numeric(decoded[col], errors="coerce").to_numpy(dtype=float)
                np.testing.assert_allclose(
                    recovered, original, atol=1e-4,
                    err_msg=f"Column '{col}' did not round-trip within tolerance"
                )
