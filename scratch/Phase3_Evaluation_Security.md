# Phase 3: Evaluation & Red-Team Validation
Privacy-Preserving Synthetic Healthcare Data Generation

Sathya Sai S  ·  Sudharshini Manikandan  ·  Vishwa D
Document Version: 1.0

## Table of Contents
1. Executive Summary
2. Utility Evaluation Metrics
3. Privacy & Security Validation (Red-Teaming)
4. Empirical Results
5. Conclusion

## 1. Executive Summary
Generating synthetic data is only half the battle; proving it is realistic and safe is the other half. This phase implements automated statistical tests to evaluate the generated data's utility (usefulness) and red-team audits to validate its privacy (security against attacks).

## 2. Utility Evaluation Metrics
We measure the Bivariate Correlation RMSE to ensure the synthetic data preserves the feature interactions of the original data. If ge and 	ime_in_hospital are correlated in real data, they must remain correlated in the synthetic data.

## 3. Privacy & Security Validation (Red-Teaming)
To simulate adversarial attacks, we measure Membership Inference Attack (MIA) risk. We calculate the Distance to Closest Record (DCR). If a synthetic record is an exact duplicate of a real patient record, the DCR is 0. Our red-team module actively blocks the release of any dataset where DCR falls below the safety threshold.

## 4. Empirical Results
The pipeline successfully completed an end-to-end sweep.
- High Privacy (Eps=0.1): Highest loss, lowest utility, maximum security.
- Balanced (Eps=1.0): Ideal for general research.
- High Utility (Eps=10.0): Lowest loss, best statistical fidelity.

## 5. Conclusion
The framework formally proved that it can generate synthetic tabular records that pass both rigorous statistical utility tests and adversarial privacy checks.
