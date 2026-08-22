# Phase 1: Data Auto-Configuration & Preprocessing
Privacy-Preserving Synthetic Healthcare Data Generation

Sathya Sai S  ·  Sudharshini Manikandan  ·  Vishwa D
Document Version: 1.0

## Table of Contents
1. Executive Summary
2. Auto-Profiling Methodology
3. Missingness Handling & Imputation
4. Feature Engineering and Encoders
5. Deliverables and Success Criteria

## 1. Executive Summary
This phase establishes the foundational data pipeline. It automatically profiles the incoming healthcare dataset (e.g., the diabetic readmission dataset) to detect column types, HIPAA identifiers, and high-missingness columns. The goal is to prepare a clean, numeric tensor representation suitable for diffusion modeling without requiring hardcoded schema rules.

## 2. Auto-Profiling Methodology
The DatasetProfiler dynamically analyzes statistical properties of each column. 
- It flags near-constant columns and high-cardinality IDs.
- It detects Protected Health Information (PHI) like encounter_id and patient_nbr and automatically drops them or assigns them to Tier 1 privacy constraints.

## 3. Missingness Handling & Imputation
A highly robust MissingnessHandler was built to map NaN values to reserved tokens (e.g., __null__). This ensures that missing data patterns—which often carry clinical significance—are preserved and modeled by the diffusion network rather than being blindly mean-imputed.

## 4. Feature Engineering and Encoders
We implemented custom OneHotEncoder and StandardScaler classes that support inverse transformations.
- Categorical variables are mapped to one-hot vectors. Rare categories are binned into __other__ to prevent privacy leaks.
- Continuous variables are scaled to a standard Gaussian distribution.

## 5. Deliverables and Success Criteria
- Outcome: The raw dataset is fully transformed into a PyTorch-ready numeric tensor.
- Success: The pipeline successfully rejected invalid HIPAA columns and mapped the diabetic dataset into a robust numeric space without data leakage.
