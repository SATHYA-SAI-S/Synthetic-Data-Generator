"""
preprocessing/base.py — Abstract interfaces for all encoders and scalers.

CRITICAL DESIGN NOTE:
    Every encoder and scaler MUST implement inverse_transform.
    This is not optional — Phase 4/5 synthesis loops decode outputs back
    to original feature space, and Phase 8 evaluation compares synthetic
    data against original distributions in original space.

    An encoder/scaler that does not implement a tested inverse_transform
    will silently corrupt all downstream phases. The test suite enforces
    exact round-trip correctness.

Phase 4/5 dependency:
    - The DP-diffusion model (Phase 7) encodes training data using these
      interfaces and decodes generated samples using the registered inverses.
    - Phase 4 may inject alternative encoders (e.g., target-encoded for
      high-cardinality categoricals) by implementing these protocols.
    - Do not add state to these ABCs; all state lives in concrete classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


class AbstractScaler(ABC):
    """
    Protocol for continuous feature scalers.

    Scalers operate on single-column pandas Series and return numpy arrays.
    Both fit and inverse_transform are required.

    Contract:
        inverse_transform(transform(x)) == x  (within float tolerance)

    State:
        All learned parameters (mean, std, min, max) must be stored as
        instance attributes so the scaler can be serialized by the registry.
    """

    @abstractmethod
    def fit(self, series: pd.Series) -> "AbstractScaler":
        """
        Fit the scaler on the given Series. Returns self for chaining.

        Args:
            series: The training data column (may contain NaN; handle explicitly).

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Transform the given Series to a scaled numpy array.

        Args:
            series: Input column (may contain NaN at inference time).

        Returns:
            1D numpy array of scaled values. NaN inputs produce NaN outputs.
        """
        ...

    @abstractmethod
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Invert the transform. MUST satisfy:
            inverse_transform(transform(x)) == x  (within float tolerance)

        Args:
            arr: 1D numpy array of scaled values.

        Returns:
            1D numpy array in original space.
        """
        ...

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(series).transform(series)


class AbstractEncoder(ABC):
    """
    Protocol for categorical feature encoders.

    Encoders operate on single-column pandas Series (string/object dtype)
    and return numpy arrays. inverse_transform is required.

    Contract:
        inverse_transform(transform(x)) == x  (exact string equality for
        non-rare, non-null values; see individual encoder docs for
        rare/null behavior)

    State:
        Vocabulary, frequency maps, and grouping rules must be stored as
        instance attributes for registry serialization.
    """

    @abstractmethod
    def fit(self, series: pd.Series) -> "AbstractEncoder":
        """
        Fit the encoder on the given Series.

        Args:
            series: The training data column. May contain NaN.

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Encode the column.

        Args:
            series: Input categorical column.

        Returns:
            2D numpy array of shape (n_samples, n_encoded_features).
            One-hot: (n, n_categories). Ordinal: (n, 1).
        """
        ...

    @abstractmethod
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Decode back to original category strings.

        Args:
            arr: Encoded array of shape matching transform output.

        Returns:
            1D numpy array of string category labels.

        Contract:
            For values that were in the training vocabulary (excluding rare
            categories grouped to __other__), the round-trip must be exact.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of columns produced by transform()."""
        ...

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(series).transform(series)


class AbstractMissingnessHandler(ABC):
    """
    Protocol for missingness flag injection and reconstruction.

    The handler learns which columns require a binary indicator during fit,
    injects the indicators during transform, and strips/reconstructs
    missingness during inverse_transform.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "AbstractMissingnessHandler":
        """Learn which columns require missingness indicator injection."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject binary missingness indicators as new columns.
        Returns a new DataFrame; does not mutate input.
        """
        ...

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove injected indicator columns and reconstruct NaN positions.
        Returns a new DataFrame; does not mutate input.
        """
        ...
