"""
profiling/dataset_profiler.py — Concrete dataset profiler implementation.

Computes per-column statistics using vectorized pandas/numpy operations.
Performs HIPAA Safe Harbor name matching, missingness pattern classification,
and structural missingness heuristics.

No I/O is performed here. The profiler receives a DataFrame and returns
a DatasetProfile. All file reading/writing happens in pipeline.py.

Phase 4/5 dependency:
    AutoConfigEngine receives the DatasetProfile output. Do not change the
    field contract of ColumnProfile or DatasetProfile without coordinating
    with Phase 4/5 developers.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config.schema import PipelineConfig
from src.profiling.base import (
    AbstractProfiler,
    ColumnProfile,
    DatasetProfile,
    HipaaFlag,
    InferredDtype,
    MissingnessPattern,
    StructuralDependency,
)

log = logging.getLogger(__name__)


# ── HIPAA Safe Harbor — 18 identifier categories ───────────────────────────────
# Each entry: (category_name, compiled_regex_pattern)
# Matching is case-insensitive on the column name.
# Reference: HIPAA Safe Harbor — 18 identifier categories.
# Patterns use (?:^|_) ... (?:_|$) anchors so they match on token boundaries
# in underscore-delimited column names (e.g. zip_code, date_of_birth).
# We also test the full column name with a leading/trailing "_" pad so
# standalone names like "fax" or "zip" also match correctly.

_HIPAA_IDENTIFIERS: list[tuple[str, list[re.Pattern]]] = [
    ("Names", [
        re.compile(r"(?:^|_)(?:name|first_?name|last_?name|patient_?name|full_?name)(?:_|$)", re.I),
    ]),
    # IP addresses MUST come before geographic — "ip_address" contains "address"
    ("Internet protocol addresses", [
        re.compile(r"(?:^|_)(?:ip_?address|ip_?addr|ipv4|ipv6)(?:_|$)", re.I),
    ]),
    ("Geographic subdivisions smaller than state", [
        # "address" matches only when NOT preceded by "ip" (handled by ordering above)
        re.compile(r"(?:^|_)(?:zip|postal|address|street|city|county|tract|geo)(?:_|$|code)", re.I),
    ]),
    ("Dates (other than year)", [
        re.compile(r"(?:^|_)(?:dob|date(?:_of)?_birth|birth_?date|admit_?date|discharge_?date|"
                   r"death_?date|service_?date|visit_?date|encounter_?date|date|day|month)(?:_|$)", re.I),
    ]),
    ("Phone numbers", [
        re.compile(r"(?:^|_)(?:phone|tel|telephone|mobile|cell)(?:_|$)", re.I),
    ]),
    ("Fax numbers", [
        re.compile(r"(?:^|_)fax(?:_|$)", re.I),
    ]),
    ("Email addresses", [
        re.compile(r"(?:^|_)e?mail(?:_|$)", re.I),
    ]),
    ("Social Security Numbers", [
        re.compile(r"(?:^|_)(?:ssn|social_?security|ss_?number)(?:_|$)", re.I),
    ]),
    ("Medical record numbers", [
        re.compile(r"(?:^|_)(?:mrn|medical_?record(?:_?number)?|record_?number|chart_?number)(?:_|$)", re.I),
    ]),
    ("Health plan beneficiary numbers", [
        re.compile(r"(?:^|_)(?:beneficiary|member_?id|plan_?id|insurance_?id)(?:_|$)", re.I),
    ]),
    ("Account numbers", [
        re.compile(r"(?:^|_)(?:account_?(?:number|num|no|id)|acct)(?:_|$)", re.I),
    ]),
    ("Certificate/license numbers", [
        re.compile(r"(?:^|_)(?:license|certificate|cert_?number|npi)(?:_|$|number)", re.I),
    ]),
    ("Vehicle identifiers", [
        re.compile(r"(?:^|_)(?:vin|vehicle|license_?plate|plate)(?:_|$)", re.I),
    ]),
    ("Device identifiers", [
        re.compile(r"(?:^|_)(?:device_?(?:id|serial)|serial_?number|imei|udid)(?:_|$)", re.I),
    ]),
    ("Web universal resource locators", [
        re.compile(r"(?:^|_)(?:url|uri|website|web_?address)(?:_|$)", re.I),
    ]),
    ("Biometric identifiers", [
        re.compile(r"(?:^|_)(?:fingerprint|biometric|retina|iris|dna)(?:_|$)", re.I),
    ]),
    ("Full-face photographs", [
        re.compile(r"(?:^|_)(?:photo|photograph|image|picture|face|portrait)(?:_|$)", re.I),
    ]),
    ("Any other unique identifying number or code", [
        re.compile(r"(?:^|_)(?:patient_?id|subject_?id|participant_?id|uid|uuid|"
                   r"person_?id|individual_?id|case_?id|encounter_?(?:id)?|patient_?nbr|nbr)(?:_|$)", re.I),
    ]),
]



def check_hipaa_identifier(column_name: str) -> HipaaFlag:
    """
    Check if a column name matches any of the 18 HIPAA Safe Harbor identifier categories.

    This function is standalone and unit-testable in isolation from the profiler.

    Strategy:
        Column names are padded with leading/trailing underscores before matching
        so that patterns using (?:^|_)...(?:_|$) correctly match both standalone
        names (e.g. "fax" -> "_fax_") and compound names (e.g. "zip_code").

    Args:
        column_name: The column name to check (case-insensitive).

    Returns:
        HipaaFlag with is_identifier=True and matched details if a match is found,
        or HipaaFlag(is_identifier=False) if no match.
    """
    # Pad with underscores so boundary patterns match standalone tokens correctly
    padded = f"_{column_name.lower()}_"
    for category, patterns in _HIPAA_IDENTIFIERS:
        for pattern in patterns:
            match = pattern.search(padded)
            if match:
                return HipaaFlag(
                    is_identifier=True,
                    matched_category=category,
                    matched_pattern=match.group(0).strip("_"),
                )

    return HipaaFlag(is_identifier=False)



# ── Dtype inference ────────────────────────────────────────────────────────────

def _infer_dtype(
    series: pd.Series,
    config: PipelineConfig,
) -> InferredDtype:
    """
    Infer the semantic data type of a pandas Series using vectorized operations.

    Priority order:
    1. Near-identifier (uniqueness ratio check)
    2. Binary (exactly 2 unique non-null values)
    3. Ordinal (integer, small number of unique values)
    4. Continuous (numeric, passes confidence threshold)
    5. Low-cardinality categorical
    6. High-cardinality categorical
    7. Unknown (fallback)
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return InferredDtype.UNKNOWN

    n_unique = non_null.nunique()
    uniqueness_ratio = n_unique / len(non_null)
    is_numeric = pd.api.types.is_numeric_dtype(non_null)

    # Near-identifier check: only applies to non-numeric columns.
    # Continuous float columns are naturally all-unique; flagging them as
    # near-identifiers would incorrectly drop all continuous features.
    if not is_numeric and uniqueness_ratio > config.cardinality.near_identifier_ratio:
        return InferredDtype.NEAR_IDENTIFIER

    # Binary check
    if n_unique <= 2:
        return InferredDtype.BINARY

    # Numeric inference — vectorized coerce + count
    if not is_numeric:
        coerced = pd.to_numeric(non_null, errors="coerce")
        numeric_fraction = coerced.notna().sum() / len(non_null)
        if numeric_fraction < 1.0:
            # Not 100% numeric strings — contains alphanumeric codes/symbols -> treat as categorical
            if n_unique <= config.cardinality.low_card_max:
                return InferredDtype.CATEGORICAL_LOW
            return InferredDtype.CATEGORICAL_HIGH
        non_null = coerced.dropna()

    # It's numeric. Determine continuous vs ordinal.
    is_integer = False
    if pd.api.types.is_integer_dtype(non_null):
        is_integer = True
    elif pd.api.types.is_float_dtype(non_null):
        # M-3 Fix: Robust vectorized integer check without overflow risks
        is_integer = bool(np.all(np.mod(non_null.values, 1) == 0))

    if is_integer:
        if n_unique <= config.dtype_inference.ordinal_max_unique_int:
            return InferredDtype.ORDINAL

    return InferredDtype.CONTINUOUS



def _infer_dtype_safe(series: pd.Series, config: PipelineConfig) -> InferredDtype:
    """Wrapper that catches exceptions and returns UNKNOWN on failure."""
    try:
        return _infer_dtype(series, config)
    except Exception as exc:
        log.warning("Dtype inference failed for column '%s': %s", series.name, exc)
        return InferredDtype.UNKNOWN


# ── Continuous statistics ──────────────────────────────────────────────────────

def _continuous_stats(series: pd.Series) -> dict:
    """Compute continuous column statistics using vectorized numpy operations."""
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if len(non_null) == 0:
        return {}
    return {
        "mean": float(non_null.mean()),
        "std": float(non_null.std()),
        "min_val": float(non_null.min()),
        "max_val": float(non_null.max()),
        "median": float(non_null.median()),
        "skewness": float(non_null.skew()),
    }


# ── Categorical statistics ─────────────────────────────────────────────────────

def _categorical_stats(series: pd.Series, config: PipelineConfig) -> dict:
    """Compute categorical column statistics using vectorized value_counts."""
    value_counts = series.value_counts(dropna=True)
    top_n = 20
    top_categories = [
        (str(k), int(v)) for k, v in value_counts.head(top_n).items()
    ]

    # Compute rare_category_min_freq, honouring the fraction override
    min_freq = config.cardinality.rare_category_min_freq
    if config.cardinality.rare_category_min_frac is not None:
        frac_freq = int(len(series) * config.cardinality.rare_category_min_frac)
        min_freq = max(min_freq, frac_freq)

    n_rare = int((value_counts < min_freq).sum())
    return {
        "top_categories": top_categories,
        "n_rare_categories": n_rare,
    }


# ── Missingness pattern classification ────────────────────────────────────────

def _classify_missingness(
    target: pd.Series,
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[MissingnessPattern, Optional[StructuralDependency]]:
    """
    Classify the missingness pattern of ``target`` within the context of ``df``.

    Returns:
        A (MissingnessPattern, Optional[StructuralDependency]) tuple.

    Algorithm:
    - Compute missing rate. If < inject_indicator_above: NONE.
    - If > drop_if_missing_above: HIGH.
    - Structural test: correlate the binary missingness indicator of ``target``
      with all other columns. If point-biserial or Cramer-V exceeds the
      configured threshold, flag as STRUCTURAL.
    - Otherwise: MCAR_LIKE.
    """
    n_total = len(target)
    n_null = target.isna().sum()
    missing_rate = n_null / n_total

    if missing_rate < config.missingness.inject_indicator_above:
        return MissingnessPattern.NONE, None

    if missing_rate > config.missingness.drop_if_missing_above:
        return MissingnessPattern.HIGH, None

    n_non_null = n_total - n_null
    if n_non_null < config.missingness.structural_min_n:
        return MissingnessPattern.MCAR_LIKE, None

    # Binary missingness indicator: 1 = missing, 0 = observed
    missing_indicator = target.isna().astype(int)

    best_corr: float = 0.0
    best_dep: Optional[StructuralDependency] = None

    for col_name in df.columns:
        if col_name == target.name:
            continue
        other = df[col_name].dropna()
        # Align indices
        aligned_indicator = missing_indicator.loc[other.index]

        if len(aligned_indicator) < config.missingness.structural_min_n:
            continue

        try:
            if pd.api.types.is_numeric_dtype(other):
                corr, _ = stats.pointbiserialr(aligned_indicator, other)
                method = "point_biserial"
            else:
                # Cramer-V for categorical predictor
                contingency = pd.crosstab(aligned_indicator, other)
                chi2, _, _, _ = stats.chi2_contingency(contingency)
                n = contingency.sum().sum()
                k = min(contingency.shape) - 1
                corr = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
                method = "cramers_v"

            abs_corr = abs(float(corr))
            if abs_corr > best_corr:
                best_corr = abs_corr
                best_dep = StructuralDependency(
                    predictor_column=col_name,
                    correlation=abs_corr,
                    method=method,
                )
        except Exception as exc:
            log.debug(
                "Structural missingness test skipped for (%s, %s): %s",
                target.name, col_name, exc,
            )
            continue

        if best_corr >= config.missingness.structural_correlation_threshold:
            break

    if best_corr >= config.missingness.structural_correlation_threshold:
        return MissingnessPattern.STRUCTURAL, best_dep

    return MissingnessPattern.MCAR_LIKE, None


# ── Concrete profiler ──────────────────────────────────────────────────────────

class DatasetProfiler(AbstractProfiler):
    """
    Concrete implementation of AbstractProfiler.

    Computes dtype inference, cardinality, missingness patterns (including
    structural missingness heuristics), and HIPAA identifier matching
    for all columns in the input DataFrame.

    All computations are vectorized (pandas/numpy). No Python-level row loops.
    """

    def profile(self, df: pd.DataFrame, dataset_name: str) -> DatasetProfile:
        """
        Profile the given DataFrame.

        Args:
            df: Input DataFrame. Will not be mutated.
            dataset_name: Logical name for this dataset (used in registry keys).

        Returns:
            A fully populated DatasetProfile.

        Raises:
            ValueError: If df is empty or has zero columns.
        """
        if df.empty:
            raise ValueError(f"Cannot profile empty DataFrame for dataset '{dataset_name}'")
        if df.shape[1] == 0:
            raise ValueError(f"DataFrame has zero columns for dataset '{dataset_name}'")

        n_rows, n_columns = df.shape
        log.info(
            "Profiling dataset '%s': %d rows x %d columns",
            dataset_name, n_rows, n_columns,
        )

        if n_rows < self._config.small_n.minimum_viable_n:
            raise ValueError(
                f"Dataset '{dataset_name}' has only {n_rows} rows, below the minimum "
                f"viable threshold of {self._config.small_n.minimum_viable_n}. "
                "Cannot proceed with training pipeline."
            )

        column_profiles: list[ColumnProfile] = []

        for col_name in df.columns:
            series = df[col_name]
            col_profile = self._profile_column(series, df)
            column_profiles.append(col_profile)
            log.debug(
                "Column '%s': dtype=%s, missing=%.1f%%, n_unique=%d",
                col_name,
                col_profile.inferred_dtype.value,
                col_profile.missing_rate * 100,
                col_profile.n_unique,
            )

        # Build convenience views (vectorized list comprehensions)
        hipaa_flagged = [c.name for c in column_profiles if c.hipaa_flag.is_identifier]
        near_id = [c.name for c in column_profiles
                   if c.inferred_dtype == InferredDtype.NEAR_IDENTIFIER]
        high_missing = [c.name for c in column_profiles
                        if c.missing_rate > self._config.missingness.drop_if_missing_above]
        small_n_flag = n_rows < self._config.small_n.small_n_threshold

        if hipaa_flagged:
            log.warning(
                "HIPAA identifier columns detected: %s. These must be excluded from "
                "generative training or handled with Tier 1 budget allocation.",
                hipaa_flagged,
            )

        if small_n_flag:
            log.warning(
                "Dataset '%s' has %d rows, classified as small-N (threshold=%d). "
                "Phase 7 will require adjusted DP parameters.",
                dataset_name, n_rows, self._config.small_n.small_n_threshold,
            )

        return DatasetProfile(
            dataset_name=dataset_name,
            n_rows=n_rows,
            n_columns=n_columns,
            columns=column_profiles,
            hipaa_flagged_columns=hipaa_flagged,
            near_identifier_columns=near_id,
            high_missing_columns=high_missing,
            small_n_flag=small_n_flag,
            profiler_config_snapshot=self._config.model_dump(),
        )

    def _profile_column(self, series: pd.Series, df: pd.DataFrame) -> ColumnProfile:
        """Profile a single column. Called once per column; no I/O."""
        col_name = str(series.name)
        n_total = len(series)
        n_null = int(series.isna().sum())
        n_non_null = n_total - n_null
        missing_rate = n_null / n_total if n_total > 0 else 0.0

        n_unique = int(series.nunique(dropna=True))
        uniqueness_ratio = (n_unique / n_non_null) if n_non_null > 0 else 0.0

        inferred_dtype = _infer_dtype_safe(series, self._config)
        hipaa_flag = check_hipaa_identifier(col_name)

        # Compute type-specific stats
        extra_stats: dict = {}
        if inferred_dtype in (InferredDtype.CONTINUOUS, InferredDtype.ORDINAL):
            extra_stats = _continuous_stats(series)
        elif inferred_dtype in (
            InferredDtype.CATEGORICAL_LOW,
            InferredDtype.CATEGORICAL_HIGH,
            InferredDtype.BINARY,
            InferredDtype.NEAR_IDENTIFIER,
        ):
            extra_stats = _categorical_stats(series, self._config)

        # Missingness classification
        missingness_pattern, structural_dep = _classify_missingness(
            series, df, self._config
        )

        return ColumnProfile(
            name=col_name,
            inferred_dtype=inferred_dtype,
            pandas_dtype=str(series.dtype),
            n_total=n_total,
            n_non_null=n_non_null,
            n_null=n_null,
            missing_rate=missing_rate,
            n_unique=n_unique,
            uniqueness_ratio=uniqueness_ratio,
            missingness_pattern=missingness_pattern,
            structural_dependency=structural_dep,
            hipaa_flag=hipaa_flag,
            **extra_stats,
        )
