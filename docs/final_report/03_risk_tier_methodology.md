# 3. Risk Tier Methodology

## Core Heuristics
The framework categorizes columns into three risk budgets prior to diffusion modeling:
1. **Tier 1 (Strict)**: Features flagged as direct HIPAA Safe Harbor Identifiers (e.g., `patient_id`, `SSN`), or features exhibiting extreme uniqueness ($\ge 80\%$).
2. **Tier 2 (Moderate)**: Features exhibiting moderate uniqueness ($> 15\%$) not flagged by HIPAA checks.
3. **Tier 3 (Loose)**: Low cardinality clinical variables (e.g., standard diagnosis groupers).

## Correlated-Feature Leakage Guard
To prevent reconstruction of a Tier 1 variable via a highly correlated Tier 3 proxy, a correlation matrix is evaluated (via rapid factorization). Pairs exhibiting a correlation threshold $\ge 0.70$ are jointly grouped into the strictest tier amongst them, neutralizing indirect MIA vulnerabilities.
