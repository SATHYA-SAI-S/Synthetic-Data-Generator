import pytest
import numpy as np
import pandas as pd
from src.evaluation.utility_metrics import UtilityEvaluator
from src.evaluation.privacy_metrics import PrivacyEvaluator

def test_utility_evaluator():
    # Setup mock data
    real_df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["A", "B", "A", "B", "C"]
    })
    
    syn_df = pd.DataFrame({
        "num": [1.1, 1.9, 3.0, 4.2, 4.8],
        "cat": ["A", "A", "A", "B", "C"] # Slightly different distribution
    })
    
    evaluator = UtilityEvaluator(real_df, syn_df)
    results = evaluator.evaluate_univariate()
    results["correlation_rmse"] = evaluator.evaluate_bivariate_correlation_rmse()
    
    # Assert metrics are computed
    assert "num" in results
    assert "cat" in results
    assert "correlation_rmse" in results
    
    # KS should be small for very similar numeric
    assert results["num"]["value"] < 0.5
    
    # TVD for cat: real has 40% A, syn has 60% A
    assert results["cat"]["value"] > 0

def test_privacy_evaluator():
    # D-MIA test
    train_df = pd.DataFrame({
        "f1": [1.0, 2.0],
        "f2": [0.5, 0.5]
    })
    holdout_df = pd.DataFrame({
        "f1": [10.0, 20.0],
        "f2": [5.0, 5.0]
    })
    syn_df = pd.DataFrame({
        "f1": [1.1, 1.9], # Very close to train
        "f2": [0.5, 0.4]
    })
    
    evaluator = PrivacyEvaluator(train_df, holdout_df, syn_df)
    results = evaluator.evaluate_mia_risk()
    
    assert "mia_risk_score" in results
    assert "mean_dist_train_to_synth" in results
    # Risk score should be high since syn is much closer to train than holdout
    assert results["mia_risk_score"] > 0.5
