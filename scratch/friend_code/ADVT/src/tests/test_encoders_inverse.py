"""
tests/test_encoders_inverse.py — Round-trip correctness tests for all transforms.

THE SINGLE MOST IMPORTANT TEST FILE IN THE ENTIRE FRAMEWORK.

Every encoder and scaler must satisfy:
    inverse_transform(transform(x)) == x

Failure here means Phase 7 cannot decode generated samples back to
original feature space. This is a hard correctness requirement, not a
nice-to-have.

Tests cover:
  - StandardScaler, MinMaxScaler, RobustScaler: exact float round-trip
  - OneHotEncoder: exact string round-trip for non-rare, non-null values
  - FrequencyEncoder: exact string round-trip for non-rare, non-null values
  - MissingnessHandler: NaN position restoration
  - Edge cases: constant columns, all-null series, single-value series
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.schema import PipelineConfig
from src.preprocessing.encoders import FrequencyEncoder, OneHotEncoder
from src.preprocessing.missingness import MissingnessHandler
from src.preprocessing.scalers import MinMaxScaler, RobustScaler, StandardScaler

# Tolerance for float comparisons (accounts for float32 round-trip)
FLOAT_ATOL = 1e-5


# ── StandardScaler ─────────────────────────────────────────────────────────────

class TestStandardScalerRoundTrip:

    def test_round_trip_exact(self, continuous_series: pd.Series) -> None:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(continuous_series)
        recovered = scaler.inverse_transform(transformed)
        original = continuous_series.to_numpy(dtype=float)
        np.testing.assert_allclose(recovered, original, atol=FLOAT_ATOL,
                                   err_msg="StandardScaler round-trip failed")

    def test_nan_propagated_through_transform(
        self, continuous_series_with_nan: pd.Series
    ) -> None:
        scaler = StandardScaler()
        scaler.fit(continuous_series_with_nan)
        transformed = scaler.transform(continuous_series_with_nan)
        # NaN positions in input should be NaN in output
        original_nan_mask = continuous_series_with_nan.isna().to_numpy()
        assert np.all(np.isnan(transformed[original_nan_mask])), \
            "NaN values should propagate through StandardScaler.transform"

    def test_nan_propagated_through_inverse(
        self, continuous_series_with_nan: pd.Series
    ) -> None:
        scaler = StandardScaler()
        scaler.fit(continuous_series_with_nan)
        transformed = scaler.transform(continuous_series_with_nan)
        recovered = scaler.inverse_transform(transformed)
        original_nan_mask = continuous_series_with_nan.isna().to_numpy()
        assert np.all(np.isnan(recovered[original_nan_mask])), \
            "NaN values should propagate through StandardScaler.inverse_transform"

    def test_round_trip_non_nan_values(
        self, continuous_series_with_nan: pd.Series
    ) -> None:
        """Non-NaN values must round-trip exactly regardless of NaN positions."""
        scaler = StandardScaler()
        scaler.fit(continuous_series_with_nan)
        transformed = scaler.transform(continuous_series_with_nan)
        recovered = scaler.inverse_transform(transformed)
        not_nan_mask = ~continuous_series_with_nan.isna().to_numpy()
        original_vals = continuous_series_with_nan.to_numpy(dtype=float)[not_nan_mask]
        np.testing.assert_allclose(
            recovered[not_nan_mask], original_vals, atol=FLOAT_ATOL,
            err_msg="StandardScaler: non-NaN values did not round-trip correctly"
        )

    def test_constant_column_no_crash(self) -> None:
        """Constant column (zero variance) must not crash — just warn."""
        series = pd.Series([5.0] * 100, name="constant")
        scaler = StandardScaler()
        scaler.fit(series)
        transformed = scaler.transform(series)
        recovered = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, 5.0, atol=FLOAT_ATOL)

    def test_unfitted_raises(self) -> None:
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="fit"):
            scaler.transform(pd.Series([1.0, 2.0]))

    def test_unfitted_inverse_raises(self) -> None:
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="fit"):
            scaler.inverse_transform(np.array([0.0, 1.0]))


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

class TestMinMaxScalerRoundTrip:

    def test_round_trip_exact(self, continuous_series: pd.Series) -> None:
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(continuous_series)
        recovered = scaler.inverse_transform(transformed)
        original = continuous_series.to_numpy(dtype=float)
        np.testing.assert_allclose(recovered, original, atol=FLOAT_ATOL,
                                   err_msg="MinMaxScaler round-trip failed")

    def test_output_in_0_1_range(self, continuous_series: pd.Series) -> None:
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(continuous_series)
        valid = transformed[~np.isnan(transformed)]
        assert valid.min() >= -FLOAT_ATOL, "MinMaxScaler output should be >= 0"
        assert valid.max() <= 1.0 + FLOAT_ATOL, "MinMaxScaler output should be <= 1"

    def test_constant_column_no_crash(self) -> None:
        series = pd.Series([3.14] * 100, name="constant")
        scaler = MinMaxScaler()
        scaler.fit(series)
        transformed = scaler.transform(series)
        recovered = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, 3.14, atol=FLOAT_ATOL)

    def test_round_trip_with_nan(
        self, continuous_series_with_nan: pd.Series
    ) -> None:
        scaler = MinMaxScaler()
        scaler.fit(continuous_series_with_nan)
        transformed = scaler.transform(continuous_series_with_nan)
        recovered = scaler.inverse_transform(transformed)
        not_nan_mask = ~continuous_series_with_nan.isna().to_numpy()
        original_vals = continuous_series_with_nan.to_numpy(dtype=float)[not_nan_mask]
        np.testing.assert_allclose(
            recovered[not_nan_mask], original_vals, atol=FLOAT_ATOL
        )


# ── RobustScaler ──────────────────────────────────────────────────────────────

class TestRobustScalerRoundTrip:

    def test_round_trip_exact(self, continuous_series: pd.Series) -> None:
        scaler = RobustScaler()
        transformed = scaler.fit_transform(continuous_series)
        recovered = scaler.inverse_transform(transformed)
        original = continuous_series.to_numpy(dtype=float)
        np.testing.assert_allclose(recovered, original, atol=FLOAT_ATOL,
                                   err_msg="RobustScaler round-trip failed")

    def test_outlier_resistance(self) -> None:
        """RobustScaler should not be dominated by extreme outliers."""
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 200).tolist()
        vals.append(1000.0)   # extreme outlier
        vals.append(-1000.0)
        series = pd.Series(vals, name="outlier_col")

        scaler = RobustScaler()
        transformed = scaler.fit_transform(series)
        # Median of transformed values should be near 0
        non_nan = transformed[~np.isnan(transformed)]
        # Most transformed values should be in a reasonable range
        assert abs(float(np.median(non_nan))) < 0.5

    def test_round_trip_with_nan(
        self, continuous_series_with_nan: pd.Series
    ) -> None:
        scaler = RobustScaler()
        scaler.fit(continuous_series_with_nan)
        transformed = scaler.transform(continuous_series_with_nan)
        recovered = scaler.inverse_transform(transformed)
        not_nan_mask = ~continuous_series_with_nan.isna().to_numpy()
        original_vals = continuous_series_with_nan.to_numpy(dtype=float)[not_nan_mask]
        np.testing.assert_allclose(
            recovered[not_nan_mask], original_vals, atol=FLOAT_ATOL
        )


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

class TestOneHotEncoderRoundTrip:

    def test_round_trip_frequent_values(self, low_card_series: pd.Series) -> None:
        """All values in a low-cardinality series should round-trip exactly."""
        encoder = OneHotEncoder(min_freq=5)
        encoder.fit(low_card_series)
        transformed = encoder.transform(low_card_series)
        recovered = encoder.inverse_transform(transformed)

        # For each position that is not NaN in original, compare
        for orig, rec in zip(low_card_series, recovered):
            if pd.isna(orig):
                assert pd.isna(rec), f"NaN should decode to NaN, got {rec}"
            else:
                assert str(orig) == str(rec), (
                    f"Round-trip failed: original='{orig}', recovered='{rec}'"
                )

    def test_null_values_round_trip(self) -> None:
        """NaN in input -> NaN in decoded output."""
        series = pd.Series(["A", "B", None, "A", "B", None], name="col")
        encoder = OneHotEncoder(min_freq=1)
        encoder.fit(series)
        transformed = encoder.transform(series)
        recovered = encoder.inverse_transform(transformed)
        assert pd.isna(recovered[2]), "Position 2 should be NaN"
        assert pd.isna(recovered[5]), "Position 5 should be NaN"

    def test_output_shape(self, low_card_series: pd.Series) -> None:
        encoder = OneHotEncoder(min_freq=5)
        encoder.fit(low_card_series)
        transformed = encoder.transform(low_card_series)
        assert transformed.ndim == 2
        assert transformed.shape[0] == len(low_card_series)
        assert transformed.shape[1] == encoder.output_dim

    def test_output_is_binary(self, low_card_series: pd.Series) -> None:
        encoder = OneHotEncoder(min_freq=5)
        transformed = encoder.fit_transform(low_card_series)
        unique_vals = np.unique(transformed)
        assert set(unique_vals).issubset({0.0, 1.0}), \
            "One-hot output should contain only 0s and 1s"

    def test_each_row_sums_to_one(self, low_card_series: pd.Series) -> None:
        encoder = OneHotEncoder(min_freq=5)
        transformed = encoder.fit_transform(low_card_series)
        row_sums = transformed.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_unfitted_raises(self) -> None:
        encoder = OneHotEncoder()
        with pytest.raises(RuntimeError, match="fit"):
            encoder.transform(pd.Series(["A", "B"]))

    def test_rare_categories_grouped(self) -> None:
        """Categories below min_freq must be grouped into '__other__'."""
        series = pd.Series(
            ["common"] * 50 + ["rare_1"] * 2 + ["rare_2"] * 1,
            name="col"
        )
        encoder = OneHotEncoder(min_freq=10)
        encoder.fit(series)
        # "common" is in vocab; rare cats are not individually in vocab
        assert "common" in encoder._vocab
        assert "rare_1" not in encoder._vocab
        assert "rare_2" not in encoder._vocab
        assert "__other__" in encoder._vocab


# ── FrequencyEncoder ──────────────────────────────────────────────────────────

class TestFrequencyEncoderRoundTrip:

    def test_round_trip_frequent_values(self, high_card_series: pd.Series) -> None:
        """Frequent categories must round-trip exactly."""
        encoder = FrequencyEncoder(min_freq=10)
        encoder.fit(high_card_series)
        transformed = encoder.transform(high_card_series)
        recovered = encoder.inverse_transform(transformed)

        vc = high_card_series.value_counts()
        frequent_cats = set(vc[vc >= 10].index)

        for orig, rec in zip(high_card_series, recovered):
            if pd.isna(orig):
                continue
            if str(orig) in frequent_cats:
                assert str(orig) == str(rec), (
                    f"FrequencyEncoder round-trip failed for frequent value: "
                    f"original='{orig}', recovered='{rec}'"
                )

    def test_output_shape_is_n_by_1(self, high_card_series: pd.Series) -> None:
        encoder = FrequencyEncoder(min_freq=5)
        transformed = encoder.fit_transform(high_card_series)
        assert transformed.shape == (len(high_card_series), 1), \
            f"Expected shape ({len(high_card_series)}, 1), got {transformed.shape}"

    def test_output_dim_property(self, high_card_series: pd.Series) -> None:
        encoder = FrequencyEncoder()
        encoder.fit(high_card_series)
        assert encoder.output_dim == 1

    def test_vocab_size_accessible(self, high_card_series: pd.Series) -> None:
        encoder = FrequencyEncoder(min_freq=10)
        encoder.fit(high_card_series)
        assert encoder.vocab_size >= 2  # at least __null__ and one category

    def test_null_decodes_to_nan(self) -> None:
        series = pd.Series(["A", "B", None, "A"], name="col")
        encoder = FrequencyEncoder(min_freq=1)
        encoder.fit(series)
        transformed = encoder.transform(series)
        recovered = encoder.inverse_transform(transformed)
        assert pd.isna(recovered[2]), "NaN input should decode to NaN"

    def test_unfitted_raises(self) -> None:
        encoder = FrequencyEncoder()
        with pytest.raises(RuntimeError, match="fit"):
            encoder.transform(pd.Series(["A", "B"]))


# ── MissingnessHandler ────────────────────────────────────────────────────────

class TestMissingnessHandlerRoundTrip:

    def test_nan_positions_restored(self, default_config: PipelineConfig) -> None:
        """
        After transform + inverse_transform, NaN positions must match the original.
        """
        rng = np.random.default_rng(42)
        n = 200
        vals = rng.normal(0, 1, n).tolist()
        # Inject exactly 30 NaNs at known positions
        nan_positions = [0, 10, 50, 99, 150, 199]
        for pos in nan_positions:
            vals[pos] = np.nan

        df = pd.DataFrame({
            "value": vals,
            "label": ["A"] * 100 + ["B"] * 100,
        })

        handler = MissingnessHandler(config=default_config)
        handler.fit(df)
        transformed = handler.transform(df)
        reconstructed = handler.inverse_transform(transformed)

        for pos in nan_positions:
            assert pd.isna(reconstructed["value"].iloc[pos]), (
                f"NaN at position {pos} was not restored after inverse_transform"
            )

    def test_non_nan_values_unchanged(self, default_config: PipelineConfig) -> None:
        """Non-NaN values must be identical after transform + inverse."""
        rng = np.random.default_rng(7)
        vals = rng.normal(0, 1, 200).tolist()
        nan_positions = {5, 15, 100}
        for pos in nan_positions:
            vals[pos] = np.nan

        df = pd.DataFrame({"x": vals})
        handler = MissingnessHandler(config=default_config)
        handler.fit(df)
        transformed = handler.transform(df)
        reconstructed = handler.inverse_transform(transformed)

        not_nan_positions = [i for i in range(200) if i not in nan_positions]
        for pos in not_nan_positions:
            original_val = df["x"].iloc[pos]
            recovered_val = reconstructed["x"].iloc[pos]
            assert abs(float(original_val) - float(recovered_val)) < FLOAT_ATOL, (
                f"Non-NaN value at position {pos} changed after round-trip: "
                f"original={original_val}, recovered={recovered_val}"
            )

    def test_no_indicator_columns_after_inverse(
        self, tiny_df: pd.DataFrame, default_config: PipelineConfig
    ) -> None:
        """After inverse_transform, no __missing_flag columns should remain."""
        handler = MissingnessHandler(config=default_config)
        handler.fit(tiny_df)
        transformed = handler.transform(tiny_df)
        reconstructed = handler.inverse_transform(transformed)
        for col in reconstructed.columns:
            assert "__missing_flag" not in col, (
                f"Indicator column '{col}' found after inverse_transform"
            )

    def test_no_missing_when_no_missingness(
        self, default_config: PipelineConfig
    ) -> None:
        """If no column has missingness, transform should be a no-op."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0] * 50,
            "b": ["X", "Y", "Z"] * 50,
        })
        handler = MissingnessHandler(config=default_config)
        handler.fit(df)
        assert handler._indicator_columns == [], (
            "No columns should require indicators when there's no missingness"
        )
        transformed = handler.transform(df)
        assert set(transformed.columns) == set(df.columns)
