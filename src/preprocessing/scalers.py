"""
preprocessing/scalers.py — Concrete continuous feature scalers.

All scalers implement fit / transform / inverse_transform with exact
round-trip correctness. NaN values are preserved through the transform
(passed as-is; the missingness handler owns NaN injection/removal).

Available scalers:
    StandardScaler  — zero-mean, unit-variance (z-score)
    MinMaxScaler    — scales to [0, 1] range
    RobustScaler    — uses median/IQR; resistant to outliers

Phase 4/5 dependency:
    Phase 7 (DP-diffusion) calls transform() to prepare training tensors
    and inverse_transform() to decode generated samples back to original
    scale. The scaler objects are stored in the registry (schema_registry.py)
    and must be deserializable via joblib.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.preprocessing.base import AbstractScaler

log = logging.getLogger(__name__)

# Sentinel for unfitted state — raises on transform before fit
_NOT_FITTED = object()


class StandardScaler(AbstractScaler):
    """
    Zero-mean, unit-variance scaler (z-score normalization).

    transform:         z = (x - mean) / std
    inverse_transform: x = z * std + mean

    NaN handling: NaN values are propagated unchanged through both
    transform and inverse_transform.

    Edge case: If std == 0 (constant column), std is set to 1.0 and a
    warning is logged. The transform becomes a mean-subtraction only.
    """

    def __init__(self) -> None:
        self._mean: float = _NOT_FITTED  # type: ignore[assignment]
        self._std: float = _NOT_FITTED   # type: ignore[assignment]

    def fit(self, series: pd.Series) -> "StandardScaler":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            raise ValueError(
                f"StandardScaler.fit: no non-null numeric values in series '{series.name}'"
            )
        self._mean = float(numeric.mean())
        self._std = float(numeric.std(ddof=0))
        if self._std == 0.0:
            log.warning(
                "StandardScaler: column '%s' has zero variance (std=0). "
                "Setting std=1.0 to avoid division by zero.",
                series.name,
            )
            self._std = 1.0
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._assert_fitted()
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        # Vectorized: NaN propagates through arithmetic naturally
        return (values - self._mean) / self._std

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)
        return arr * self._std + self._mean

    def _assert_fitted(self) -> None:
        if self._mean is _NOT_FITTED:
            raise RuntimeError("StandardScaler has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        if self._mean is _NOT_FITTED:
            return "StandardScaler(unfitted)"
        return f"StandardScaler(mean={self._mean:.4f}, std={self._std:.4f})"


class MinMaxScaler(AbstractScaler):
    """
    Scales values to the [0, 1] range using observed min/max.

    transform:         x_scaled = (x - min) / (max - min)
    inverse_transform: x = x_scaled * (max - min) + min

    NaN handling: NaN values are propagated unchanged.

    Edge case: If min == max (constant column), range is set to 1.0.
    """

    def __init__(self) -> None:
        self._min: float = _NOT_FITTED  # type: ignore[assignment]
        self._max: float = _NOT_FITTED  # type: ignore[assignment]
        self._range: float = _NOT_FITTED  # type: ignore[assignment]

    def fit(self, series: pd.Series) -> "MinMaxScaler":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            raise ValueError(
                f"MinMaxScaler.fit: no non-null numeric values in series '{series.name}'"
            )
        self._min = float(numeric.min())
        self._max = float(numeric.max())
        self._range = self._max - self._min
        if self._range == 0.0:
            log.warning(
                "MinMaxScaler: column '%s' is constant (min==max=%.4f). "
                "Setting range=1.0.",
                series.name, self._min,
            )
            self._range = 1.0
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._assert_fitted()
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return (values - self._min) / self._range

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)
        return arr * self._range + self._min

    def _assert_fitted(self) -> None:
        if self._min is _NOT_FITTED:
            raise RuntimeError("MinMaxScaler has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        if self._min is _NOT_FITTED:
            return "MinMaxScaler(unfitted)"
        return f"MinMaxScaler(min={self._min:.4f}, max={self._max:.4f})"


class RobustScaler(AbstractScaler):
    """
    Scales using median and interquartile range (IQR).

    Resistant to outliers — suitable for skewed clinical measurements
    (e.g., lab values with extreme outliers).

    transform:         x_scaled = (x - median) / IQR
    inverse_transform: x = x_scaled * IQR + median

    where IQR = Q75 - Q25.

    NaN handling: NaN values are propagated unchanged.

    Edge case: If IQR == 0, IQR is set to 1.0 with a warning.
    """

    def __init__(self) -> None:
        self._median: float = _NOT_FITTED  # type: ignore[assignment]
        self._iqr: float = _NOT_FITTED     # type: ignore[assignment]

    def fit(self, series: pd.Series) -> "RobustScaler":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            raise ValueError(
                f"RobustScaler.fit: no non-null numeric values in series '{series.name}'"
            )
        self._median = float(numeric.median())
        q25, q75 = float(np.percentile(numeric, 25)), float(np.percentile(numeric, 75))
        self._iqr = q75 - q25
        if self._iqr == 0.0:
            log.warning(
                "RobustScaler: column '%s' has zero IQR (Q25=Q75=%.4f). "
                "Setting IQR=1.0.",
                series.name, q25,
            )
            self._iqr = 1.0
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._assert_fitted()
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return (values - self._median) / self._iqr

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)
        return arr * self._iqr + self._median

    def _assert_fitted(self) -> None:
        if self._median is _NOT_FITTED:
            raise RuntimeError("RobustScaler has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        if self._median is _NOT_FITTED:
            return "RobustScaler(unfitted)"
        return f"RobustScaler(median={self._median:.4f}, iqr={self._iqr:.4f})"
