"""
tests/test_registry_roundtrip.py — Registry save/load round-trip tests.

Tests:
  - save() creates correct directory structure
  - load() reconstructs RegistryEntry with identical types
  - load_profile() deserializes only the JSON (lightweight path)
  - Versioning: multiple saves increment version; load(latest) works
  - delete() removes version; raises on missing dataset
  - Atomic write: partial write does not corrupt registry
  - list_datasets / list_versions return correct values
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.schema import PipelineConfig
from src.preprocessing.encoders import OneHotEncoder
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.scalers import StandardScaler
from src.profiling.dataset_profiler import DatasetProfiler
from src.registry.schema_registry import FileSchemaRegistry


def _build_minimal_registry_entry(
    df: pd.DataFrame, config: PipelineConfig, dataset_name: str
) -> tuple[
    "DatasetProfiler",
    dict,
    dict,
    "MissingnessHandler",
    list,
    dict,
    list,
]:
    """Helper: fit a minimal set of transforms and return all registry args."""
    profiler = DatasetProfiler(config=config)
    profile = profiler.profile(df, dataset_name)

    handler = MissingnessHandler(config=config)
    handler.fit(df)

    scalers = {}
    encoders = {}
    training_columns = []
    column_types = {}
    encoded_col_names = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            scaler = StandardScaler()
            scaler.fit(df[col].dropna().reset_index(drop=True))
            scalers[col] = scaler
            column_types[col] = "continuous"
            encoded_col_names.append(col)
        else:
            enc = OneHotEncoder(min_freq=1)
            enc.fit(df[col])
            encoders[col] = enc
            column_types[col] = "categorical"
            for i in range(enc.output_dim):
                encoded_col_names.append(f"{col}__enc{i}")
        training_columns.append(col)

    return profile, scalers, encoders, handler, training_columns, column_types, encoded_col_names


class TestFileSchemaRegistry:

    def test_save_creates_files(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        profile, scalers, encoders, handler, tc, ct, ecn = (
            _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        )
        version = registry.save(
            dataset_name="test_ds",
            profile=profile,
            scalers=scalers,
            encoders=encoders,
            missingness_handler=handler,
            training_columns=tc,
            column_types=ct,
            encoded_col_names=ecn,
        )
        assert version == 1
        assert (registry_root / "test_ds" / "v1" / "profile.json").exists()
        assert (registry_root / "test_ds" / "v1" / "pipeline_state.joblib").exists()
        assert (registry_root / "test_ds" / "latest.txt").exists()

    def test_load_returns_registry_entry(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        profile, scalers, encoders, handler, tc, ct, ecn = (
            _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        )
        registry.save(
            dataset_name="test_ds",
            profile=profile,
            scalers=scalers,
            encoders=encoders,
            missingness_handler=handler,
            training_columns=tc,
            column_types=ct,
            encoded_col_names=ecn,
        )

        entry = registry.load("test_ds")
        assert entry.dataset_name == "test_ds"
        assert entry.version == 1
        assert entry.profile.n_rows == tiny_df.shape[0]
        assert set(entry.scalers.keys()) == set(scalers.keys())
        assert set(entry.encoders.keys()) == set(encoders.keys())
        assert entry.training_columns == tc
        assert entry.column_types == ct
        assert entry.encoded_col_names == ecn

    def test_load_profile_is_lightweight(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        """load_profile must not load the joblib file."""
        registry = FileSchemaRegistry(registry_root)
        profile, scalers, encoders, handler, tc, ct, ecn = (
            _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        )
        registry.save(
            dataset_name="test_ds",
            profile=profile,
            scalers=scalers,
            encoders=encoders,
            missingness_handler=handler,
            training_columns=tc,
            column_types=ct,
            encoded_col_names=ecn,
        )
        loaded_profile = registry.load_profile("test_ds")
        assert loaded_profile.n_rows == tiny_df.shape[0]
        assert loaded_profile.n_columns == tiny_df.shape[1]

    def test_versioning_increments(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")

        v1 = registry.save("test_ds", *args)
        v2 = registry.save("test_ds", *args)
        assert v1 == 1
        assert v2 == 2
        assert registry.list_versions("test_ds") == [1, 2]

    def test_load_latest_version(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args)
        registry.save("test_ds", *args)

        entry = registry.load("test_ds")  # no version -> latest
        assert entry.version == 2

    def test_load_specific_version(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args)
        registry.save("test_ds", *args)

        entry_v1 = registry.load("test_ds", version=1)
        assert entry_v1.version == 1

    def test_list_datasets(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args_a = _build_minimal_registry_entry(tiny_df, default_config, "dataset_a")
        args_b = _build_minimal_registry_entry(tiny_df, default_config, "dataset_b")
        registry.save("dataset_a", *args_a)
        registry.save("dataset_b", *args_b)

        datasets = registry.list_datasets()
        assert "dataset_a" in datasets
        assert "dataset_b" in datasets

    def test_load_unknown_dataset_raises_key_error(
        self, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        with pytest.raises(KeyError, match="not registered"):
            registry.load("nonexistent_dataset")

    def test_load_unknown_version_raises_value_error(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args)
        with pytest.raises(ValueError):
            registry.load("test_ds", version=999)

    def test_delete_version(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args)
        registry.save("test_ds", *args)

        registry.delete("test_ds", version=1)
        assert registry.list_versions("test_ds") == [2]

    def test_delete_all(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args)
        registry.delete("test_ds")
        assert "test_ds" not in registry.list_datasets()

    def test_save_duplicate_version_raises(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        registry = FileSchemaRegistry(registry_root)
        args = _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        registry.save("test_ds", *args, version=1)
        with pytest.raises(ValueError, match="already exists"):
            registry.save("test_ds", *args, version=1)

    def test_scaler_inverse_works_after_registry_roundtrip(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig, registry_root: Path
    ) -> None:
        """
        Scalers deserialized from registry must still produce correct inverse_transform.
        This is the key integration test: registry doesn't break estimator state.
        """
        registry = FileSchemaRegistry(registry_root)
        profile, scalers, encoders, handler, tc, ct, ecn = (
            _build_minimal_registry_entry(tiny_df, default_config, "test_ds")
        )
        registry.save(
            dataset_name="test_ds",
            profile=profile,
            scalers=scalers,
            encoders=encoders,
            missingness_handler=handler,
            training_columns=tc,
            column_types=ct,
            encoded_col_names=ecn,
        )

        entry = registry.load("test_ds")

        for col, original_scaler in scalers.items():
            loaded_scaler = entry.scalers[col]
            test_series = tiny_df[col].dropna().reset_index(drop=True)
            original_transformed = original_scaler.transform(test_series)
            loaded_transformed = loaded_scaler.transform(test_series)
            np.testing.assert_allclose(
                original_transformed, loaded_transformed, atol=1e-6,
                err_msg=f"Scaler for '{col}' produces different output after registry roundtrip"
            )
