"""
config/schema.py — Single source of truth for all pipeline thresholds and configuration.

All numeric thresholds, cardinality cutoffs, tier definitions, and tunable knobs
live here as typed Pydantic models. No magic numbers anywhere else in the codebase.

Phase 4/5 dependency:
    AutoConfigEngine will consume ``PipelineConfig`` and extend it with
    ``PrivacyTierConfig`` (per-feature epsilon budgets) without modifying
    this file — add new sub-models, do not mutate existing fields.
"""

from __future__ import annotations

from typing import Optional
import logging

from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)


class CardinalityConfig(BaseModel):
    """Thresholds governing categorical feature cardinality classification."""

    model_config = {"frozen": True}

    # Number of unique values below which a categorical field is treated as
    # "low-cardinality" and one-hot encoded.
    low_card_max: int = Field(
        default=15,
        ge=2,
        le=500,
        description="Max unique values for low-cardinality one-hot encoding.",
    )

    # Unique-value fraction above which a field is considered a near-identifier
    # (e.g., patient ID encoded as int). Expressed as fraction of N.
    near_identifier_ratio: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "If uniqueness_ratio > this, field is flagged as a near-identifier "
            "and excluded from generative training unless override is set."
        ),
    )

    # Minimum frequency a rare category must have to be kept as its own bin.
    # Categories below this are grouped into an '__other__' bucket.
    rare_category_min_freq: int = Field(
        default=10,
        ge=1,
        description="Minimum count for a category to retain its own bin.",
    )

    # Alternative: express rare threshold as a fraction of total N.
    rare_category_min_frac: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "If set, overrides rare_category_min_freq with N * rare_category_min_frac. "
            "The more restrictive of the two thresholds is used."
        ),
    )


class MissingnessConfig(BaseModel):
    """Thresholds governing missingness detection and handling strategy."""

    model_config = {"frozen": True}

    # Columns with a missing rate above this threshold are dropped from training
    # unless explicitly whitelisted in the dataset config.
    drop_if_missing_above: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description="Drop column if missingness rate exceeds this fraction.",
    )

    # Columns with missingness between these bounds get a binary missingness
    # indicator column injected (1 = was_missing, 0 = was_observed).
    inject_indicator_above: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Inject a binary missingness indicator if rate >= this.",
    )

    # Correlation threshold for structural missingness detection.
    # If the missingness pattern of column A correlates with the values of
    # column B above this threshold, A is flagged as structurally missing.
    structural_correlation_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Point-biserial correlation threshold above which a column's "
            "missingness pattern is flagged as structurally dependent on another column."
        ),
    )

    # Minimum N required to run the structural missingness heuristic.
    structural_min_n: int = Field(
        default=30,
        ge=10,
        description="Minimum number of non-null values required to test structural missingness.",
    )


class SmallNConfig(BaseModel):
    """Configuration for small-N cohort detection and handling."""

    model_config = {"frozen": True}

    # Below this row count, the dataset is considered small-N and triggers
    # more aggressive DP noise (or a different model architecture in Phase 7).
    small_n_threshold: int = Field(
        default=500,
        ge=50,
        description="Row count below which dataset is classified as small-N.",
    )

    # Below this count, training is refused outright — too few rows for
    # any meaningful DP guarantee at reasonable epsilon.
    minimum_viable_n: int = Field(
        default=100,
        ge=10,
        description="Minimum row count to proceed with training at all.",
    )

    @model_validator(mode="after")
    def validate_n_ordering(self) -> "SmallNConfig":
        if self.minimum_viable_n >= self.small_n_threshold:
            raise ValueError(
                f"minimum_viable_n ({self.minimum_viable_n}) must be less than "
                f"small_n_threshold ({self.small_n_threshold})"
            )
        return self


class DtypeInferenceConfig(BaseModel):
    """Settings for automatic dtype inference during profiling."""

    model_config = {"frozen": True}

    # Fraction of values in a column that must parse as numeric for it to be
    # classified as continuous. E.g., 0.95 means >=95% parseable as float.
    numeric_confidence_threshold: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Fraction of non-null values that must be numeric to classify as continuous.",
    )

    # Maximum unique integer values for a column to be treated as ordinal
    # rather than continuous (e.g., a 1-10 rating scale).
    ordinal_max_unique_int: int = Field(
        default=20,
        ge=2,
        description="Max unique integer values to classify as ordinal (not continuous).",
    )



class DiffusionConfig(BaseModel):
    """Phase 4: Diffusion model hyperparameters."""
    model_config = {"frozen": True}
    
    num_timesteps: int = Field(default=1000, ge=10, description="Total diffusion steps T")
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=0.02, gt=0.0)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256, 256], description="Denoiser MLP layer sizes")

class TrainingConfig(BaseModel):
    """Phase 4/6: Training loop hyperparameters."""
    model_config = {"frozen": True}
    
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    epochs: int = Field(default=50, ge=1)
    checkpoint_dir: str = Field(default="checkpoints")
    gpu_budget_hours: float = Field(default=30.0, gt=0.0)

class PrivacyConfig(BaseModel):
    """Phase 5: DP configurations."""
    model_config = {"frozen": True}
    
    target_epsilon: float = Field(default=1.0, gt=0.0)
    target_delta: float = Field(default=1e-5, gt=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)

class PipelineConfig(BaseModel):
    """
    Root configuration object for the Phase 3 preprocessing pipeline.

    This is the single config object passed to the pipeline orchestrator.
    All sub-configs are frozen (immutable) once constructed.

    Phase 4/5 extension pattern:
        from src.config.schema import PipelineConfig
        from pydantic import BaseModel

        class ExtendedConfig(PipelineConfig):
            privacy_tiers: PrivacyTierConfig = PrivacyTierConfig()

    Do NOT mutate fields of this model after construction.
    """

    model_config = {"frozen": True}

    cardinality: CardinalityConfig = Field(
        default_factory=CardinalityConfig,
        description="Cardinality classification thresholds.",
    )

    missingness: MissingnessConfig = Field(
        default_factory=MissingnessConfig,
        description="Missingness detection and handling configuration.",
    )

    small_n: SmallNConfig = Field(
        default_factory=SmallNConfig,
        description="Small-N cohort detection thresholds.",
    )

    dtype_inference: DtypeInferenceConfig = Field(
        default_factory=DtypeInferenceConfig,
        description="Dtype inference settings.",
    )

    # Random seed for reproducibility across all stochastic pipeline steps.
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Global random seed for pipeline reproducibility.",
    )


    diffusion: DiffusionConfig = Field(
        default_factory=DiffusionConfig,
        description="Diffusion model hyperparameters."
    )
    
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training loop hyperparameters."
    )
    
    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig,
        description="Differential Privacy (DP) configurations."
    )
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return a default config with sensible healthcare dataset defaults."""
        return cls()
