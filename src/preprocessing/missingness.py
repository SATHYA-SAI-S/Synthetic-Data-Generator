"""
preprocessing/missingness.py — Missingness flag injection and reconstruction.

Implements AbstractMissingnessHandler. Learns which columns need a binary
indicator during fit(), injects those indicators during transform(), and
removes them and restores NaN positions during inverse_transform().

No I/O. No global state. Fully invertible.

Phase 4/5 dependency:
    The missingness handler output (injected DataFrame) is passed to the
    scaler/encoder pipeline. The registry stores the handler's fitted state
    (columns_requiring_indicator, indicator_column_names) so inverse_transform
    can be called at synthesis time in Phase 7/8.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config.schema import PipelineConfig
from src.preprocessing.base import AbstractMissingnessHandler

log = logging.getLogger(__name__)

# Naming convention for injected indicator columns
_INDICATOR_SUFFIX = "__missing_flag"


class MissingnessHandler(AbstractMissingnessHandler):
    """
    Concrete missingness handler.

    Behaviour:
      fit():
        For each column with missing_rate >= inject_indicator_above,
        record that it needs an indicator column named '<col>__missing_flag'.
        Does NOT drop columns above drop_if_missing_above — dropping is the
        pipeline orchestrator's responsibility (pipeline.py), not this class.

      transform():
        Adds binary indicator columns (1 = was_missing, 0 = was_observed)
        for each tracked column. Imputes NaN in original columns with the
        column's median (continuous) or mode (categorical) so downstream
        scalers/encoders receive no NaN inputs.

        NOTE: The imputed value is a placeholder. The true signal is
        preserved in the injected indicator column. The generative model
        learns the joint distribution of (value | indicator).

      inverse_transform():
        Removes all injected indicator columns. For each tracked column,
        uses the original indicator values to restore NaN positions in the
        reconstructed data.

    Round-trip contract:
        For a DataFrame df:
            handler.fit(df)
            transformed = handler.transform(df)
            reconstructed = handler.inverse_transform(transformed)
        Then for every column c in df:
            - NaN positions in df[c] are NaN in reconstructed[c]
            - Non-NaN positions in df[c] equal reconstructed[c] (same values)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        # Columns that require a missingness indicator
        self._indicator_columns: list[str] = []
        # Imputation values learned during fit (median for numeric, mode for object)
        self._imputation_values: dict[str, object] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "MissingnessHandler":
        """
        Learn which columns require indicator injection.

        Args:
            df: Training DataFrame. Not mutated.

        Returns:
            self
        """
        n_rows = len(df)
        self._indicator_columns = []
        self._imputation_values = {}

        for col in df.columns:
            missing_rate = df[col].isna().sum() / n_rows
            if missing_rate >= self._config.missingness.inject_indicator_above:
                self._indicator_columns.append(col)
                # Compute imputation value
                col_series = df[col].dropna()
                if pd.api.types.is_numeric_dtype(df[col]):
                    impute_val = float(col_series.median()) if len(col_series) > 0 else 0.0
                else:
                    impute_val = str(col_series.mode().iloc[0]) if len(col_series) > 0 else ""
                self._imputation_values[col] = impute_val
                log.debug(
                    "MissingnessHandler: column '%s' missing_rate=%.1f%%, "
                    "indicator will be injected, impute_val=%s",
                    col, missing_rate * 100, impute_val,
                )

        self._fitted = True
        log.info(
            "MissingnessHandler fitted: %d columns will receive indicators: %s",
            len(self._indicator_columns), self._indicator_columns,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject binary missingness indicators and impute NaN values.

        Args:
            df: Input DataFrame. Not mutated.

        Returns:
            New DataFrame with added indicator columns. Original columns
            have NaN replaced by imputed values.
        """
        self._assert_fitted()
        result = df.copy()

        for col in self._indicator_columns:
            if col not in result.columns:
                log.warning(
                    "MissingnessHandler.transform: column '%s' not found in DataFrame. "
                    "Skipping indicator injection.",
                    col,
                )
                continue

            indicator_col = f"{col}{_INDICATOR_SUFFIX}"
            # 1 where missing, 0 where observed — vectorized
            result[indicator_col] = result[col].isna().astype(np.int8)

            # Impute NaN with learned value
            result[col] = result[col].fillna(self._imputation_values[col])

        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove indicator columns and restore NaN positions.

        Args:
            df: DataFrame containing injected indicator columns.
                Not mutated.

        Returns:
            DataFrame with indicator columns removed and NaN positions
            restored in original columns.
        """
        self._assert_fitted()
        result = df.copy()

        for col in self._indicator_columns:
            indicator_col = f"{col}{_INDICATOR_SUFFIX}"

            if indicator_col not in result.columns:
                log.warning(
                    "MissingnessHandler.inverse_transform: indicator column '%s' "
                    "not found. Cannot restore NaN positions for '%s'.",
                    indicator_col, col,
                )
                continue

            if col not in result.columns:
                log.warning(
                    "MissingnessHandler.inverse_transform: original column '%s' "
                    "not found. Skipping NaN restoration.",
                    col,
                )
                result.drop(columns=[indicator_col], inplace=True)
                continue

            indicator_values = result[indicator_col]
            if not np.all(np.isin(indicator_values.round(5), [0.0, 1.0]) | np.isnan(indicator_values)):
                log.warning(
                    "MissingnessHandler: indicator column '%s' contains soft non-binary predictions. "
                    "Thresholding at 0.5.", indicator_col
                )
            # Restore NaN where indicator >= 0.5 (vectorized boolean mask)
            was_missing_mask = indicator_values >= 0.5
            result.loc[was_missing_mask, col] = np.nan

            # Drop the injected indicator column
            result.drop(columns=[indicator_col], inplace=True)

        return result

    @property
    def indicator_column_names(self) -> list[str]:
        """Names of injected indicator columns (after transform)."""
        self._assert_fitted()
        return [f"{col}{_INDICATOR_SUFFIX}" for col in self._indicator_columns]

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "MissingnessHandler has not been fitted. Call fit() first."
            )
