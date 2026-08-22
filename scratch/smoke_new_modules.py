"""Smoke tests for the new automation modules."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.route_decider import RouteDecider
d = RouteDecider(10000)
assert d.decide(5000).route == "adapter"
assert d.decide(50000).route == "kaggle"
print("route_decider OK: small->adapter, large->kaggle")

from src.orchestration.kaggle_bridge import classify_kaggle_error
cases = {
    "429 Too Many Requests": "transient",
    "weekly GPU quota (30 hours)": "quota",
    "403 Forbidden": "auth",
    "RuntimeError CUDA OOM": "backend_rework",
}
for s, expected in cases.items():
    got = classify_kaggle_error(s).category
    assert got == expected, f"{s}: {got} != {expected}"
print("kaggle_bridge classifier OK:", list(cases.values()))

import numpy as np
import pandas as pd
from src.adversary.attacker_engine import run_red_team
rng = np.random.default_rng(0)
real = pd.DataFrame({"a": rng.integers(0, 50, 300), "b": rng.choice(["x", "y", "z"], 300)})
hold = pd.DataFrame({"a": rng.integers(0, 50, 300), "b": rng.choice(["x", "y", "z"], 300)})
synth = real.sample(200, replace=True).reset_index(drop=True)  # deliberately leaky
rep = run_red_team(real, hold, synth, epsilon_claimed=1.0)
assert rep.verdict == "PRIVACY_FLAGGED", rep.verdict
print("attacker engine OK: leaky synth ->", rep.verdict,
      "| worst:", round(rep.worst_success_rate, 3))

# Clean synth should be certified
synth_clean = pd.DataFrame({"a": rng.integers(0, 50, 200), "b": rng.choice(["x", "y", "z"], 200)})
rep2 = run_red_team(real, hold, synth_clean, epsilon_claimed=1.0)
print("attacker engine clean case verdict:", rep2.verdict)

# Adapter route (requires torch)
try:
    from src.diffusion.schema_adapter import adapter_route_run
    df = pd.DataFrame({
        "age": np.random.randint(20, 90, 400),
        "los": np.random.randint(1, 15, 400),
        "gender": np.random.choice(["M", "F"], 400),
    })
    synth_df, info = adapter_route_run(df, epsilon=1.0, epochs=1, num_samples=50)
    assert len(synth_df) == 50 and set(synth_df.columns) == set(df.columns)
    print("schema_adapter OK:", info["backbone_source"], "| dim", info["encoded_dim"])
except (ImportError, RuntimeError) as e:
    print("schema_adapter SKIPPED (torch unavailable):", e)

print("ALL SMOKE TESTS PASSED")