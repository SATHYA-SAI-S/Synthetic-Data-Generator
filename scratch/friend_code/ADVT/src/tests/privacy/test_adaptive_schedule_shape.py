import pytest
from src.privacy.adaptive_schedule import AdaptiveNoiseSchedule

def test_adaptive_schedule_shape():
    num_steps = 100
    base_sigma = 1.0
    
    schedule = AdaptiveNoiseSchedule(base_sigma, num_steps, strategy="linear")
    
    # At t=0 (data), ratio is 1.0 -> sigma = 1.0 * (0.5 + 1.0) = 1.5
    assert abs(schedule.get_sigma(0) - 1.5) < 1e-5
    
    # At t=99 (noise), ratio is 0.0 -> sigma = 1.0 * (0.5 + 0.0) = 0.5
    assert abs(schedule.get_sigma(99) - 0.5) < 1e-5
    
    # Monotonicity check: sigma should decrease as t increases
    for t in range(1, num_steps):
        assert schedule.get_sigma(t) < schedule.get_sigma(t - 1)
        
    # Constant strategy check
    const_schedule = AdaptiveNoiseSchedule(base_sigma, num_steps, strategy="constant")
    assert const_schedule.get_sigma(0) == base_sigma
    assert const_schedule.get_sigma(99) == base_sigma
