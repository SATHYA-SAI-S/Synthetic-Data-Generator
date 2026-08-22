"""Smoke test for audit fixes (no torch needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

# sklearn is not installed in this env — stub the one class we need
import types
_sk = types.ModuleType("sklearn"); _skn = types.ModuleType("sklearn.neighbors")
class _NN:
    def __init__(self, **kw): pass
    def fit(self, X): self.X = np.asarray(X, float); return self
    def kneighbors(self, Q):
        Q = np.asarray(Q, float)
        d = np.empty(len(Q))
        for i, q in enumerate(Q):
            d[i] = np.sqrt(((self.X - q) ** 2).sum(1).min())
        return d.reshape(-1, 1), None
_skn.NearestNeighbors = _NN
_sk.neighbors = _skn
sys.modules["sklearn"] = _sk; sys.modules["sklearn.neighbors"] = _skn

# stub scipy.stats.ks_2samp too
_sci = types.ModuleType("scipy"); _scist = types.ModuleType("scipy.stats")
def _ks(a, b):
    # crude two-sample KS
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return (np.abs(ca - cb).max(), 0.0)
_scist.ks_2samp = _ks
_sci.stats = _scist
sys.modules["scipy"] = _sci; sys.modules["scipy.stats"] = _scist

# stub torch (only needed for abstract base class imports)
_torch = types.ModuleType("torch")
class _device:
    def __init__(self, *a): pass
_torch.device = _device
class _Tensor: pass
_torch.Tensor = _Tensor
sys.modules["torch"] = _torch

# 1. PrivacyEvaluator shared-vocabulary fix (E-1)
from src.evaluation.privacy_metrics import PrivacyEvaluator
train = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1.0, 2.0, 3.0, 4.0]})
hold = pd.DataFrame({"cat": ["b", "c"], "num": [2.5, 3.5]})
synth = pd.DataFrame({"cat": ["a", "b"], "num": [1.5, 2.5]})
ev = PrivacyEvaluator(train, hold, synth)
res = ev.evaluate_mia_risk()
assert np.isfinite(res["mia_risk_score"]), res
# verify shared vocab: 'a' must map to same code in train and synth frames
v = ev._build_shared_vocab(["cat"])
t_codes = ev._factorize_data(train, ["cat"], v).ravel()
s_codes = ev._factorize_data(synth, ["cat"], v).ravel()
assert t_codes[0] == s_codes[0], f"vocab mismatch: {t_codes} vs {s_codes}"
print("E-1 OK:", res)

# 2. UtilityEvaluator NaN for <2 numeric cols (E-3)
from src.evaluation.utility_metrics import UtilityEvaluator
ue = UtilityEvaluator(pd.DataFrame({"x": [1.0]}), pd.DataFrame({"x": [2.0]}))
assert np.isnan(ue.evaluate_bivariate_correlation_rmse())
print("E-3 OK")

# 3. MissingnessHandler object-dtype imputation keeps numerics numeric (M-2)
from src.config.schema import PipelineConfig
from src.preprocessing.missingness import MissingnessHandler
cfg = PipelineConfig()
df = pd.DataFrame({"obj_num": ["1.0", None, "3.0", "5.0"]})
h = MissingnessHandler(cfg).fit(df)
out = h.transform(df)
imputed = out.loc[out["obj_num__missing_flag"] == 1, "obj_num"].iloc[0]
assert not isinstance(imputed, str), f"imputant still str: {imputed!r}"
back = h.inverse_transform(out)
assert back["obj_num"].isna().sum() == 1
print("M-2 OK")

# 4. FrequencyEncoder out-of-range routing (M-6)
from src.preprocessing.encoders import FrequencyEncoder
fe = FrequencyEncoder(min_freq=2).fit(pd.Series(["x"] * 10 + ["y"] * 5 + ["z"]))
assert fe.vocab_size == 4  # null, x, y, __other__
dec = fe.inverse_transform(np.array([[1.0], [50.0]]))
assert dec[0] == "x" and dec[1] == "__other__", dec
print("M-6 OK")

# 5. RiskTierAssigner order-independence (M-3)
from src.privacy.risk_tier_assigner import HeuristicRiskTierAssigner
rng = np.random.default_rng(0)
n = 500
dfa = pd.DataFrame({
    "id_like": rng.random(n),
    "const": ["k"] * n,
    "cat_low": rng.choice(["p", "q"], n),
})
ta = HeuristicRiskTierAssigner().assign_tiers(dfa)
tb = HeuristicRiskTierAssigner().assign_tiers(dfa[list(reversed(dfa.columns))])
assert ta["id_like"] == tb["id_like"] == "Tier1"
print("M-3 OK")

print("ALL SMOKE TESTS PASSED")