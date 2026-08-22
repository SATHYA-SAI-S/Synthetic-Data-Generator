"""
preprocessing/encoders.py — Categorical feature encoders with inverse_transform.

Two concrete encoders:

  OneHotEncoder   — for low-cardinality features (n_unique <= low_card_max).
                    Rare categories are grouped into '__other__' before encoding.
                    Inverse maps argmax back to original category string.

  FrequencyEncoder — for high-cardinality features. Uses frequency-based
                    grouping: rare categories (below min_freq threshold) are
                    bucketed into '__other__'. The encoded representation is
                    a single integer index into the frequency-sorted vocabulary,
                    making it embedding-table-friendly for Phase 7's diffusion
                    model. inverse_transform maps index back to category string.

Both implement the AbstractEncoder protocol exactly.
NaN values are encoded as index 0 / '__null__' category and reconstructed
as NaN on inverse.

Phase 4/5 dependency:
    Phase 7 uses FrequencyEncoder output as discrete token indices for
    embedding lookup. The vocabulary (self._vocab) must be accessible
    externally for embedding table size initialization:
        n_embeddings = len(encoder._vocab)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.base import AbstractEncoder

log = logging.getLogger(__name__)

# Reserved tokens — must not collide with real category names
_NULL_TOKEN = "__null__"
_OTHER_TOKEN = "__other__"


class OneHotEncoder(AbstractEncoder):
    """
    One-hot encoder for low-cardinality categorical columns.

    Encoding strategy:
      1. Fit: compute value_counts. Group categories below min_freq into
         '__other__'. Build ordered vocabulary list.
      2. Transform: map each value to its vocabulary index, then one-hot
         expand into a (n_samples, n_vocab) binary matrix.
      3. Inverse: argmax over one-hot rows -> vocabulary index -> category string.
         '__null__' token -> NaN. '__other__' -> '__other__' (cannot recover).

    Round-trip contract:
        For values that are (a) non-null and (b) above min_freq in the
        training data, inverse_transform(transform(x)) == x exactly.
        Rare values that were binned to '__other__' decode as '__other__'.
        NaN values decode as NaN.
    """

    def __init__(self, min_freq: int = 10) -> None:
        self._min_freq = min_freq
        self._vocab: list[str] = []           # ordered; index 0 = __null__
        self._cat_to_idx: dict[str, int] = {}
        self._fitted = False

    def fit(self, series: pd.Series) -> "OneHotEncoder":
        counts = series.value_counts(dropna=True)
        # B-14: Collision check for reserved tokens
        if _NULL_TOKEN in counts.index or _OTHER_TOKEN in counts.index:
            raise ValueError(f"Reserved tokens {_NULL_TOKEN} or {_OTHER_TOKEN} found in real data.")

        # Separate frequent from rare categories (vectorized boolean mask)
        frequent_mask = counts >= self._min_freq
        frequent_cats = counts[frequent_mask].index.tolist()
        rare_cats = counts[~frequent_mask].index.tolist()

        if rare_cats:
            log.debug(
                "OneHotEncoder: %d rare categories in '%s' grouped into '%s'",
                len(rare_cats), series.name, _OTHER_TOKEN,
            )

        # Vocabulary: null, other (if needed), then frequent cats sorted by freq desc
        vocab = [_NULL_TOKEN]
        if rare_cats:
            vocab.append(_OTHER_TOKEN)
        vocab.extend(frequent_cats)

        self._vocab = vocab
        self._cat_to_idx = {cat: idx for idx, cat in enumerate(vocab)}
        self._fitted = True
        log.debug(
            "OneHotEncoder fitted on '%s': vocab_size=%d (incl. null/other)",
            series.name, len(self._vocab),
        )
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._assert_fitted()
        n = len(series)
        
        # Vectorized stringification and mapping
        series_str = series.copy()
        series_str[series_str.isna()] = _NULL_TOKEN
        series_str = series_str.astype(str)
        
        # H-5 Fix: Map unseen values to __other__ if it exists, else __null__
        fallback_idx = self._cat_to_idx.get(_OTHER_TOKEN, self._cat_to_idx[_NULL_TOKEN])
        
        mapped = series_str.map(self._cat_to_idx)
        indices = mapped.fillna(fallback_idx).to_numpy(dtype=np.int32)

        # One-hot expand: vectorized using numpy indexing
        n_classes = len(self._vocab)
        one_hot = np.zeros((n, n_classes), dtype=np.float32)
        one_hot[np.arange(n), indices] = 1.0
        return one_hot

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Decode one-hot rows back to category strings.
        Uses argmax - ties broken by lowest index (i.e., __null__).
        """
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)

        # R-07: Warn on non-binary (soft) inputs
        if not np.all(np.isin(arr, [0.0, 1.0])):
            log.warning("OneHotEncoder.inverse_transform: input contains non-binary values (soft predictions).")

        indices = np.argmax(arr, axis=1)
        result = np.empty(len(indices), dtype=object)
        for i, idx in enumerate(indices):
            label = self._vocab[idx]
            result[i] = np.nan if label == _NULL_TOKEN else label
        return result

    @property
    def output_dim(self) -> int:
        self._assert_fitted()
        return len(self._vocab)

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("OneHotEncoder has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        if not self._fitted:
            return "OneHotEncoder(unfitted)"
        return f"OneHotEncoder(vocab_size={len(self._vocab)}, min_freq={self._min_freq})"


class FrequencyEncoder(AbstractEncoder):
    """
    Frequency-based integer encoder for high-cardinality categorical columns.

    Encoding strategy:
      1. Fit: compute value_counts descending. Group rare categories
         (below min_freq) into '__other__'. Assign integer indices in
         frequency-descending order (most frequent = index 1; index 0
         reserved for __null__; __other__ gets the last index).
      2. Transform: map each value to its integer index. Output shape: (n, 1).
         This is directly usable as a discrete token index for embedding tables.
      3. Inverse: integer index -> vocabulary string -> category (or NaN).

    Rationale for frequency ordering:
      Most ML embedding tables initialize frequent tokens with better
      representations. Ordering by frequency ensures the embedding indices
      carry implicit frequency signal, which can accelerate convergence
      in Phase 7's diffusion model.

    Round-trip contract:
      For non-null, non-rare values: inverse_transform(transform(x)) == x exactly.
      Rare values binned to '__other__' decode as '__other__'.
      NaN values decode as NaN.
    """

    def __init__(self, min_freq: int = 10) -> None:
        self._min_freq = min_freq
        self._vocab: list[str] = []           # index -> category
        self._cat_to_idx: dict[str, int] = {}
        self._fitted = False

    def fit(self, series: pd.Series) -> "FrequencyEncoder":
        counts = series.value_counts(dropna=True, ascending=False)
        # B-14: Collision check for reserved tokens
        if _NULL_TOKEN in counts.index or _OTHER_TOKEN in counts.index:
            raise ValueError(f"Reserved tokens {_NULL_TOKEN} or {_OTHER_TOKEN} found in real data.")

        frequent_mask = counts >= self._min_freq
        frequent_cats = counts[frequent_mask].index.tolist()
        rare_cats = counts[~frequent_mask].index.tolist()

        if rare_cats:
            log.debug(
                "FrequencyEncoder: %d rare categories in '%s' grouped into '%s'",
                len(rare_cats), series.name, _OTHER_TOKEN,
            )

        # index 0 = __null__, then frequent cats by freq, then __other__
        vocab = [_NULL_TOKEN] + frequent_cats
        if rare_cats:
            vocab.append(_OTHER_TOKEN)

        self._vocab = vocab
        self._cat_to_idx = {cat: idx for idx, cat in enumerate(vocab)}
        self._fitted = True
        log.debug(
            "FrequencyEncoder fitted on '%s': vocab_size=%d (incl. null/other)",
            series.name, len(self._vocab),
        )
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._assert_fitted()
        
        # R-01 / R-03: Safe stringification
        series_str = series.copy()
        series_str[series_str.isna()] = _NULL_TOKEN
        series_str = series_str.astype(str)
        
        mapped = series_str.map(self._cat_to_idx)
        fallback_idx = self._cat_to_idx.get(_OTHER_TOKEN, 0)
        
        indices = mapped.fillna(fallback_idx).to_numpy(dtype=np.int32)
        return indices.reshape(-1, 1)  # (n, 1) for consistent 2D output

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Decode integer indices back to category strings."""
        self._assert_fitted()
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        arr = np.round(arr).astype(int).ravel()  # flatten (n,1) or (n,)
        
        # B-08 / M-6 FIX: route out-of-range indices to __other__ (or NaN) instead
        # of silently clipping. Silent clipping masked model failure by mapping
        # garbage indices to the most-frequent category.
        other_idx = self._cat_to_idx.get(_OTHER_TOKEN)
        
        result = np.empty(len(arr), dtype=object)
        for i, idx in enumerate(arr):
            if idx < 0 or idx >= len(self._vocab):
                log.warning("FrequencyEncoder.inverse_transform: index %d out of range [0, %d]; routing to '%s'.",
                            idx, len(self._vocab) - 1, _OTHER_TOKEN if other_idx is not None else "NaN")
                result[i] = (_OTHER_TOKEN if other_idx is not None else np.nan)
                continue
            label = self._vocab[idx]
            result[i] = np.nan if label == _NULL_TOKEN else label
        return result

    @property
    def output_dim(self) -> int:
        """FrequencyEncoder always outputs 1 column (the integer index)."""
        return 1

    @property
    def vocab_size(self) -> int:
        """Size of vocabulary — needed by Phase 7 to initialize embedding tables."""
        self._assert_fitted()
        return len(self._vocab)

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FrequencyEncoder has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        if not self._fitted:
            return "FrequencyEncoder(unfitted)"
        return f"FrequencyEncoder(vocab_size={len(self._vocab)}, min_freq={self._min_freq})"
