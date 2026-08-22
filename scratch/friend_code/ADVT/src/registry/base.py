"""
registry/base.py — Abstract interface for the schema registry.

The registry is the persistence layer for all fitted pipeline state:
column order, dtypes, scaler/encoder objects, vocabulary sizes, and
dataset profile snapshots.

Phase 4/5 dependency:
    Phase 7 loads the registry to reconstruct the full inverse pipeline
    for decoding generated samples. The interface defined here is the
    contract that Phase 7 depends on — do not change method signatures
    without coordinating with Phase 7 developers.
    Phase 4 (AutoConfigEngine) calls registry.load_profile(dataset_name)
    to retrieve the DatasetProfile without needing to re-run the profiler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.preprocessing.base import AbstractEncoder, AbstractMissingnessHandler, AbstractScaler
from src.profiling.base import DatasetProfile


class RegistryEntry:
    """
    Container for all state associated with one fitted dataset pipeline.

    Intentionally a plain class (not Pydantic) because it holds non-serializable
    Python objects (fitted sklearn-style estimators). The registry backend
    handles serialization via joblib.
    """

    def __init__(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        scalers: dict[str, AbstractScaler],
        encoders: dict[str, AbstractEncoder],
        missingness_handler: AbstractMissingnessHandler,
        training_columns: list[str],
        column_types: dict[str, str],
        encoded_col_names: list[str],
        version: int = 1,
    ) -> None:
        self.dataset_name = dataset_name
        self.profile = profile
        self.scalers = scalers
        self.encoders = encoders
        self.missingness_handler = missingness_handler
        self.training_columns = training_columns
        self.column_types = column_types
        self.encoded_col_names = encoded_col_names
        self.version = version


class AbstractSchemaRegistry(ABC):
    """
    Protocol for schema registries.

    Concrete implementations may store state to disk (FileSchemaRegistry),
    a database, or in-memory (for testing). Phase 4/5 can inject a
    different backend without touching the pipeline.
    """

    @abstractmethod
    def save(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        scalers: dict[str, AbstractScaler],
        encoders: dict[str, AbstractEncoder],
        missingness_handler: AbstractMissingnessHandler,
        training_columns: list[str],
        column_types: dict[str, str],
        encoded_col_names: list[str],
        version: Optional[int] = None,
    ) -> int:
        """
        Persist the fitted pipeline state for a dataset.

        Args:
            dataset_name:        Logical name (used as primary key).
            profile:             The DatasetProfile from profiling.
            scalers:             Fitted scalers keyed by column name.
            encoders:            Fitted encoders keyed by column name.
            missingness_handler: Fitted missingness handler.
            training_columns:    Ordered list of columns used in training.
            column_types:        Mapping col -> "continuous" | "categorical".
            encoded_col_names:   Flat ordered list of output feature names.
            version:             Optional explicit version; auto-incremented if None.

        Returns:
            The version number under which the entry was saved.

        Raises:
            RuntimeError: If the save fails (disk full, permission error, etc.).
                          Must raise — never silently swallow.
        """
        ...

    @abstractmethod
    def load(self, dataset_name: str, version: Optional[int] = None) -> RegistryEntry:
        """
        Load the fitted pipeline state for a dataset.

        Args:
            dataset_name: Logical name.
            version:      Specific version to load. If None, loads the latest.

        Returns:
            RegistryEntry with all fitted state.

        Raises:
            KeyError:   If the dataset_name is not registered.
            ValueError: If the requested version does not exist.
        """
        ...

    @abstractmethod
    def load_profile(self, dataset_name: str, version: Optional[int] = None) -> DatasetProfile:
        """
        Load only the DatasetProfile for a dataset (lightweight; no estimator deserialization).

        Args:
            dataset_name: Logical name.
            version:      Specific version. If None, loads the latest.

        Returns:
            DatasetProfile.

        Raises:
            KeyError: If the dataset is not registered.
        """
        ...

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """Return all registered dataset names."""
        ...

    @abstractmethod
    def list_versions(self, dataset_name: str) -> list[int]:
        """
        Return all registered versions for a dataset, ascending.

        Raises:
            KeyError: If dataset_name is not registered.
        """
        ...

    @abstractmethod
    def delete(self, dataset_name: str, version: Optional[int] = None) -> None:
        """
        Delete a specific version (or all versions if version=None).

        Raises:
            KeyError: If the dataset is not registered.
        """
        ...
