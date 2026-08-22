"""
profiling/base.py — Abstract interface for dataset profilers.

Define the protocol BEFORE the concrete implementation so Phase 4/5 can
inject alternative profilers (e.g., a streaming profiler for large files)
without touching the pipeline orchestrator.

Phase 4/5 dependency:
    AutoConfigEngine expects ``DatasetProfile`` as its input contract.
    The risk-tier assignment logic (Phase 5) will consume ``.columns``
    (typed ColumnProfile objects) and ``.hipaa_flags``.
    Do NOT change field names on ``ColumnProfile`` or ``DatasetProfile``
    without a corresponding update to the AutoConfigEngine interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.config.schema import PipelineConfig


# ── Enums ──────────────────────────────────────────────────────────────────────

class InferredDtype(str, Enum):
    """Inferred semantic data type for a column."""
    CONTINUOUS = "continuous"
    CATEGORICAL_LOW = "categorical_low"    # low cardinality; one-hot suitable
    CATEGORICAL_HIGH = "categorical_high"  # high cardinality; embedding suitable
    ORDINAL = "ordinal"
    BINARY = "binary"
    NEAR_IDENTIFIER = "near_identifier"    # uniqueness ratio near 1; exclude from training
    UNKNOWN = "unknown"


class MissingnessPattern(str, Enum):
    """Heuristic classification of the missingness pattern in a column."""
    NONE = "none"                # < 1% missing
    MCAR_LIKE = "mcar_like"      # uniform random missing; no structural dependence detected
    STRUCTURAL = "structural"    # missingness correlated with another column's values
    HIGH = "high"                # high rate (above drop threshold); may be dropped
    UNKNOWN = "unknown"


# ── Typed output models ────────────────────────────────────────────────────────

class HipaaFlag(BaseModel):
    """Result of a HIPAA Safe Harbor identifier check on a single column name."""
    model_config = {"frozen": True}

    is_identifier: bool
    matched_category: Optional[str] = None
    matched_pattern: Optional[str] = None


class StructuralDependency(BaseModel):
    """Describes a detected structural missingness dependency."""
    model_config = {"frozen": True}

    predictor_column: str
    correlation: float   # point-biserial or Cramer-V, depending on predictor type
    method: str          # "point_biserial" | "cramers_v"


class ColumnProfile(BaseModel):
    """
    Complete statistical profile of a single column.

    This is the stable contract that the auto-config engine (Phase 4/5)
    consumes to assign privacy tiers and decide encoding strategies.
    All fields are required; use Optional + None explicitly for absent data.
    """
    model_config = {"frozen": True}

    name: str
    inferred_dtype: InferredDtype
    pandas_dtype: str                 # raw pandas dtype string, e.g. "float64"

    n_total: int                      # total number of rows in dataset
    n_non_null: int
    n_null: int
    missing_rate: float               # n_null / n_total

    n_unique: int
    uniqueness_ratio: float           # n_unique / n_non_null

    # Continuous-only stats (None for categorical)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None

    # Categorical-only stats (None for continuous)
    top_categories: Optional[list[tuple[str, int]]] = None  # [(value, count), ...]
    n_rare_categories: Optional[int] = None  # categories below rare_category_min_freq

    missingness_pattern: MissingnessPattern = MissingnessPattern.UNKNOWN
    structural_dependency: Optional[StructuralDependency] = None

    hipaa_flag: HipaaFlag = Field(
        default_factory=lambda: HipaaFlag(is_identifier=False)
    )


class DatasetProfile(BaseModel):
    """
    Complete profile of a dataset, containing per-column profiles and
    dataset-level statistics.

    Serializable (Pydantic model) — can be written to JSON/YAML for
    downstream consumption without reconstructing the full pandas DataFrame.
    """
    model_config = {"frozen": True}

    dataset_name: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]

    # Convenience views (derived from columns; populated by profiler)
    hipaa_flagged_columns: list[str]          # column names with is_identifier=True
    near_identifier_columns: list[str]        # column names with NEAR_IDENTIFIER dtype
    high_missing_columns: list[str]           # columns above drop_if_missing_above threshold
    small_n_flag: bool                        # True if n_rows < small_n_threshold

    profiler_config_snapshot: dict            # serialized PipelineConfig used for this profile

    def column_by_name(self, name: str) -> ColumnProfile:
        """Look up a ColumnProfile by column name. Raises KeyError if not found."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in profile.")


# ── Abstract profiler protocol ─────────────────────────────────────────────────

class AbstractProfiler(ABC):
    """
    Protocol for dataset profilers.

    Concrete implementations must:
    1. Accept a PipelineConfig at construction time.
    2. Return a DatasetProfile from profile().
    3. Not perform I/O — receive the DataFrame, return the profile.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    @abstractmethod
    def profile(self, df: pd.DataFrame, dataset_name: str) -> DatasetProfile:
        """
        Profile the given DataFrame and return a typed DatasetProfile.

        Args:
            df: The input DataFrame. Must not be mutated.
            dataset_name: A human-readable name for logging/registry keys.

        Returns:
            A fully populated DatasetProfile.

        Raises:
            ValueError: If df is empty or has zero columns.
            RuntimeError: If dtype inference confidence is below configured threshold.
        """
        ...
