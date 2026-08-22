"""
preprocessing/pipeline.py — Pipeline orchestrator for Phase 3.

Orchestrates:  profiler -> missingness handler -> encoders/scalers -> registry

Key design decisions:
  1. ALL dependencies (profiler, encoders factory, missingness handler,
     registry) are injected via constructor — nothing is imported or
     instantiated internally. This is what lets Phase 4/5 swap in a
     different registry or alternative encoder without touching this file.

  2. I/O boundary: the pipeline reads/writes DataFrames and files ONLY
     at the outermost edge (fit_transform_from_file / transform_from_file).
     All internal business logic receives DataFrames, not file paths.

  3. Column dropping (high-missingness, near-identifier) happens here,
     not inside the profiler or encoders. The profiler detects; the
     pipeline decides.

  4. The pipeline stores the fitted state in the registry at the end of
     fit_transform, so Phase 7 can reconstruct the full inverse pipeline
     from scratch given only the registry.

Phase 4/5 dependency:
    Phase 4 (AutoConfigEngine) calls pipeline.get_profile() after fitting
    to retrieve the DatasetProfile and make tier assignment decisions.
    Phase 7 calls pipeline.inverse_transform(generated_array) to decode
    diffusion model outputs back to a human-readable DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config.schema import PipelineConfig
from src.preprocessing.base import (
    AbstractEncoder,
    AbstractMissingnessHandler,
    AbstractScaler,
)
from src.profiling.base import AbstractProfiler, DatasetProfile, InferredDtype
from src.registry.base import AbstractSchemaRegistry

log = logging.getLogger(__name__)

# Type alias for the encoder/scaler factory callables
EncoderFactory = Callable[[str], AbstractEncoder]  # col_name -> encoder instance
ScalerFactory = Callable[[str], AbstractScaler]    # col_name -> scaler instance


class PreprocessingPipeline:
    """
    Orchestrates the full Phase 3 preprocessing pipeline.

    Constructed with injected dependencies — no hardcoded imports of
    concrete encoder/scaler/registry classes.

    Lifecycle:
        1. fit_transform(df) -> np.ndarray
           Profiles data, fits all transforms, encodes/scales everything,
           saves fitted state to registry. Returns encoded array.
        2. transform(df) -> np.ndarray
           Applies fitted transforms to new data (inference time).
        3. inverse_transform(arr) -> pd.DataFrame
           Decodes a numpy array back to a DataFrame in original space.
    """

    def __init__(
        self,
        config: PipelineConfig,
        profiler: AbstractProfiler,
        missingness_handler: AbstractMissingnessHandler,
        encoder_factory: EncoderFactory,
        scaler_factory: ScalerFactory,
        registry: AbstractSchemaRegistry,
    ) -> None:
        """
        Args:
            config:               Pipeline configuration (thresholds, seeds).
            profiler:             Fitted DatasetProfiler instance.
            missingness_handler:  MissingnessHandler instance.
            encoder_factory:      Callable that returns a new AbstractEncoder
                                  given a column name. Allows per-column encoder
                                  customization by Phase 4.
            scaler_factory:       Callable that returns a new AbstractScaler
                                  given a column name.
            registry:             Where fitted transforms and schema are persisted.
        """
        self._config = config
        self._profiler = profiler
        self._missingness_handler = missingness_handler
        self._encoder_factory = encoder_factory
        self._scaler_factory = scaler_factory
        self._registry = registry

        # Fitted state — populated after fit_transform()
        self._profile: Optional[DatasetProfile] = None
        self._training_columns: list[str] = []     # columns actually used (after drops)
        self._column_types: dict[str, str] = {}    # col -> "continuous" | "categorical" | "binary"
        self._scalers: dict[str, AbstractScaler] = {}
        self._encoders: dict[str, AbstractEncoder] = {}
        self._encoded_col_names: list[str] = []    # flat ordered list of output feature names
        self._fitted = False

    # ── Public interface ───────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        """
        Profile, fit all transforms, encode/scale the DataFrame.

        Args:
            df:           Raw input DataFrame (may contain NaN, mixed dtypes).
            dataset_name: Logical name for registry keys and logging.

        Returns:
            2D numpy array of shape (n_rows, n_encoded_features).

        Raises:
            ValueError: If df is empty, has < minimum_viable_n rows, or all
                        columns are dropped due to high missingness / near-id.
        """
        log.info("Pipeline fit_transform started for dataset '%s'", dataset_name)
        # B-10: Seed numpy for reproducibility
        if hasattr(self._config, "random_seed") and self._config.random_seed is not None:
            np.random.seed(self._config.random_seed)

        # Step 1: Profile
        self._profile = self._profiler.profile(df, dataset_name)

        # Step 2: Determine columns to drop
        drop_cols = set(
            self._profile.high_missing_columns
            + self._profile.near_identifier_columns
            + self._profile.hipaa_flagged_columns
        )
        # N-04: Drop constant columns
        constant_cols = {c for c in df.columns if df[c].nunique(dropna=False) <= 1}
        drop_cols.update(constant_cols)
        
        if drop_cols:
            log.warning(
                "Dropping %d columns (high-missing / near-identifier / HIPAA / constant): %s",
                len(drop_cols), sorted(drop_cols),
            )

        working_df = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors="ignore",
        )

        if working_df.shape[1] == 0:
            raise ValueError(
                f"All columns dropped for dataset '{dataset_name}'. "
                "Cannot proceed with empty training set."
            )

        # Step 3: Missingness handler fit + transform
        self._missingness_handler.fit(working_df)
        working_df = self._missingness_handler.transform(working_df)

        # Step 4: Fit and apply encoders/scalers per column
        initial_columns = list(working_df.columns)
        valid_training_columns = []
        encoded_parts: list[np.ndarray] = []
        self._encoded_col_names = []

        # B-06: O(1) profile lookup instead of O(n^2)
        profile_dict = {c.name: c for c in self._profile.columns}

        for col in initial_columns:
            col_profile = profile_dict.get(col)

            inferred = (
                col_profile.inferred_dtype
                if col_profile is not None
                else InferredDtype.UNKNOWN
            )

            if inferred in (InferredDtype.CONTINUOUS, InferredDtype.ORDINAL):
                # Verify column is actually numeric before scaling
                coerced = pd.to_numeric(working_df[col], errors="coerce")
                if coerced.isna().sum() > working_df[col].isna().sum():
                    encoder = self._encoder_factory(col)
                    encoded = encoder.fit_transform(working_df[col])
                    self._encoders[col] = encoder
                    self._column_types[col] = "categorical"
                    encoded_parts.append(encoded)
                    valid_training_columns.append(col)
                    for i in range(encoder.output_dim):
                        self._encoded_col_names.append(f"{col}__enc{i}")
                else:
                    scaler = self._scaler_factory(col)
                    encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)
                    self._scalers[col] = scaler
                    self._column_types[col] = "continuous"
                    encoded_parts.append(encoded)
                    self._encoded_col_names.append(col)
                    valid_training_columns.append(col)

            elif inferred in (
                InferredDtype.CATEGORICAL_LOW,
                InferredDtype.CATEGORICAL_HIGH,
                InferredDtype.BINARY,
            ):
                encoder = self._encoder_factory(col)
                # R-01: Pass raw series to encoder, let encoder handle NaNs
                encoded = encoder.fit_transform(working_df[col])
                self._encoders[col] = encoder
                self._column_types[col] = "categorical"
                encoded_parts.append(encoded)
                valid_training_columns.append(col)
                # Track one name per output dim
                for i in range(encoder.output_dim):
                    self._encoded_col_names.append(f"{col}__enc{i}")

            else:
                log.warning(
                    "Column '%s' has dtype '%s'; checking numeric viability.",
                    col, inferred.value,
                )
                coerced = pd.to_numeric(working_df[col], errors="coerce")
                if coerced.isna().sum() > working_df[col].isna().sum():
                    encoder = self._encoder_factory(col)
                    encoded = encoder.fit_transform(working_df[col])
                    self._encoders[col] = encoder
                    self._column_types[col] = "categorical"
                    encoded_parts.append(encoded)
                    valid_training_columns.append(col)
                    for i in range(encoder.output_dim):
                        self._encoded_col_names.append(f"{col}__enc{i}")
                else:
                    scaler = self._scaler_factory(col)
                    try:
                        encoded = scaler.fit_transform(working_df[col]).reshape(-1, 1)
                        self._scalers[col] = scaler
                        self._column_types[col] = "continuous"
                        encoded_parts.append(encoded)
                        self._encoded_col_names.append(col)
                        valid_training_columns.append(col)
                    except Exception as exc:
                        log.error(
                            "Cannot encode column '%s' (dtype=%s): %s. Skipping.",
                            col, inferred.value, exc,
                        )
                    
        # B-01: Update training columns to only those that successfully encoded
        self._training_columns = valid_training_columns

        if not encoded_parts:
            raise ValueError(
                f"No columns could be encoded for dataset '{dataset_name}'."
            )

        result = np.concatenate(encoded_parts, axis=1).astype(np.float32)

        self._fitted = True
        log.info(
            "Pipeline fit complete: %d input columns -> %d encoded features",
            len(self._training_columns), result.shape[1],
        )

        # Step 5: Persist to registry
        self._registry.save(
            dataset_name=dataset_name,
            profile=self._profile,
            scalers=self._scalers,
            encoders=self._encoders,
            missingness_handler=self._missingness_handler,
            training_columns=self._training_columns,
            column_types=self._column_types,
            encoded_col_names=self._encoded_col_names,
        )

        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted transforms to new data (inference / evaluation time).

        Args:
            df: Input DataFrame with same columns as training data.

        Returns:
            2D numpy array of shape (n_rows, n_encoded_features).
        """
        self._assert_fitted()
        
        # M-5 Fix: Warn about extra columns
        extra_cols = set(df.columns) - set(self._training_columns)
        if extra_cols:
            log.warning("PreprocessingPipeline.transform: Ignoring %d extra columns not seen during fit: %s", len(extra_cols), list(extra_cols)[:5])

        # Drop the same columns as during fit (they won't be in training_columns)
        working_df = self._missingness_handler.transform(df)

        encoded_parts: list[np.ndarray] = []

        for col in self._training_columns:
            if col not in working_df.columns:
                raise ValueError(
                    f"Column '{col}' expected but not found in input DataFrame. "
                    "Ensure the same schema is used as during fit_transform()."
                )

            if col in self._scalers:
                encoded = self._scalers[col].transform(working_df[col]).reshape(-1, 1)
                encoded_parts.append(encoded)
            elif col in self._encoders:
                encoded = self._encoders[col].transform(working_df[col])
                encoded_parts.append(encoded)
            else:
                log.warning("Column '%s' has no fitted transform; skipping.", col)

        return np.concatenate(encoded_parts, axis=1).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        """
        Decode a numpy array back to a human-readable DataFrame.

        Args:
            arr: 2D numpy array of shape (n_rows, n_encoded_features).

        Returns:
            DataFrame in original column space, with NaN positions restored.
        """
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)

        # R-08: Width validation
        expected_width = 0
        for col in self._training_columns:
            if col in self._scalers:
                expected_width += 1
            elif col in self._encoders:
                expected_width += self._encoders[col].output_dim
            else:
                expected_width += 1
        if arr.shape[1] != expected_width:
            raise ValueError(f"inverse_transform expected {expected_width} columns, got {arr.shape[1]}")

        reconstructed: dict[str, np.ndarray] = {}
        col_cursor = 0

        for col in self._training_columns:
            if col in self._scalers:
                col_arr = arr[:, col_cursor]
                reconstructed[col] = self._scalers[col].inverse_transform(col_arr)
                col_cursor += 1
            elif col in self._encoders:
                enc = self._encoders[col]
                dim = enc.output_dim
                col_arr = arr[:, col_cursor: col_cursor + dim]
                reconstructed[col] = enc.inverse_transform(col_arr)
                col_cursor += dim
            else:
                col_cursor += 1

        result_df = pd.DataFrame(reconstructed)

        # Restore NaN positions via missingness handler
        result_df = self._missingness_handler.inverse_transform(result_df)
        return result_df

    def fit_transform_from_file(self, csv_path: Path, dataset_name: str) -> np.ndarray:
        """
        Convenience: read CSV from disk, then fit_transform.
        This is the ONLY place file I/O happens in the pipeline.
        """
        log.info("Reading CSV from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        return self.fit_transform(df, dataset_name)

    def get_profile(self) -> DatasetProfile:
        """Return the DatasetProfile computed during fit_transform."""
        self._assert_fitted()
        return self._profile  # type: ignore[return-value]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit_transform() first."
            )
