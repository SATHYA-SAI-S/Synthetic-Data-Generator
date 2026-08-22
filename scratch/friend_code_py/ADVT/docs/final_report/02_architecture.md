# 2. Architecture

The pipeline consists of strictly decoupled layers enforcing robust data transformations:

## Layer 0-1: Data Profiling
The `DatasetProfiler` dynamically scans incoming CSV data. It infers continuous vs. categorical types, identifies missingness patterns, and flags HIPAA identifiers. It produces an immutable `DatasetProfile` schema.

## Layer 2-3: Preprocessing & Serialization
The `PreprocessingPipeline` leverages specific encoders (One-Hot, Frequency) and Scalers (Standard, MinMax, Robust). The mapping states and vocabularies are serialized via `FileSchemaRegistry` to guarantee flawless dataset round-tripping.

## Layer 4: Diffusion Base
A non-adaptive DDPM `MLPDenoiser` dynamically sizes its neural architecture to match the dimension footprint provided by the registry.

## Layer 5: Privacy Engine
The DP-SGD mechanism is injected via `Opacus`. Gradients are captured per-sample, partitioned by the `HeuristicRiskTierAssigner`, and independently scaled and noised via `clip_and_noise_tier()`. A central `RDPAccountant` tracks the total compositional privacy spend.

## Layer 6-7: Orchestration & Evaluation
The `reproduce_end_to_end.py` script automatically loops across $\epsilon$ configurations, generating synthetic artifacts, while the `UtilityEvaluator` and `PrivacyEvaluator` calculate KS/TVD statistics and D-MIA vulnerability bounds.
