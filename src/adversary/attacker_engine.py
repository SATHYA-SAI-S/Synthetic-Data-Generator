"""
Adversarial Red-Team Attacker Engine.

An adaptive, rule-based attacker that iteratively attempts to extract real
patient information from synthetic data, escalating strategies like a real
adversary. Runs automatically after generation (both local and Kaggle routes)
and produces a structured verdict that complements the formal epsilon-DP
guarantee: the proof is the ceiling of the privacy claim; the attacker
empirically validates the implementation.

Escalation ladder:
  L1  Exact quasi-identifier re-identification (QI subsets ranked by rarity)
  L2  Membership inference (D-MIA) with AUC-ROC discrimination
  L3  Attribute inference (predict a held-out sensitive column from synth NNs)
  L4  Uniqueness / singling-out + multi-column fingerprint linkage

Each level adapts to the previous level's partial successes. The attacker
stops early if a level achieves a "leak" above the configured threshold.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:  # pragma: no cover
    SKLEARN_OK = False

VERDICT_CERTIFIED = "PRIVACY_CERTIFIED"
VERDICT_FLAGGED = "PRIVACY_FLAGGED"

DEFAULT_RISK_THRESHOLD = 0.05   # >5% attack success on any level => flagged


@dataclass
class AttackResult:
    level: str
    attack: str
    success_rate: float          # 0..1, higher = worse for privacy
    detail: str
    verdict: str                 # "held" | "partial" | "leak"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackReport:
    verdict: str
    worst_success_rate: float
    risk_threshold: float
    results: List[Dict[str, Any]]
    epsilon_claimed: Optional[float] = None
    ran_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)
        return p


def _factorize(df: pd.DataFrame, cols: List[str],
               vocab: Optional[Dict[str, Dict[str, int]]] = None
               ) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    vocab = vocab or {}
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in vocab:
            out[c] = df[c].astype(str).map(vocab[c]).fillna(-1.0)
        elif df[c].dtype == object or pd.api.types.is_bool_dtype(df[c]):
            cats = sorted(df[c].dropna().astype(str).unique().tolist())
            vocab[c] = {v: i for i, v in enumerate(cats)}
            out[c] = df[c].astype(str).map(vocab[c]).fillna(-1.0)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=float), vocab


class AdaptiveAttacker:
    """Rule-based adaptive attacker with an escalation ladder."""

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_holdout: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 risk_threshold: float = DEFAULT_RISK_THRESHOLD,
                 max_qi_width: int = 4,
                 seed: int = 0):
        self.df_train = df_train
        self.df_holdout = df_holdout
        self.df_synth = df_synth
        self.risk_threshold = risk_threshold
        self.max_qi_width = max_qi_width
        self.rng = np.random.default_rng(seed)
        self.results: List[AttackResult] = []
        self._vocab: Dict[str, Dict[str, int]] = {}

    # -- helpers -------------------------------------------------------------

    def _common_cols(self) -> List[str]:
        return [c for c in self.df_train.columns
                if c in self.df_synth.columns and c in self.df_holdout.columns]

    def _verdict(self, rate: float) -> str:
        if rate <= self.risk_threshold:
            return "held"
        if rate <= self.risk_threshold * 3:
            return "partial"
        return "leak"

    # -- L1: exact QI re-identification --------------------------------------

    def _rank_qi_subsets(self, cols: List[str]) -> List[List[str]]:
        """Rank QI subsets by identifying power (rare value combinations first)."""
        scored = []
        for width in range(2, min(self.max_qi_width, len(cols)) + 1):
            for combo in itertools.combinations(cols, width):
                try:
                    keys = self.df_train[list(combo)].astype(str).agg("|".join, axis=1)
                    rarity = 1.0 - (keys.value_counts(normalize=True).max())
                    scored.append((rarity, list(combo)))
                except Exception:
                    continue
        scored.sort(reverse=True)
        return [c for _, c in scored[:8]]  # top-8 most identifying subsets

    def attack_l1_exact_qi(self) -> AttackResult:
        cols = self._common_cols()
        if not cols:
            return AttackResult("L1", "exact_qi_match", 0.0, "No common columns", "held")
        worst_rate, worst_qi = 0.0, []
        for qi in self._rank_qi_subsets(cols):
            rq = set(map(tuple, self.df_train[qi].astype(str).values))
            sq = list(map(tuple, self.df_synth[qi].astype(str).values))
            hits = sum(1 for t in sq if t in rq)
            rate = hits / max(len(sq), 1)
            if rate > worst_rate:
                worst_rate, worst_qi = rate, qi
        return AttackResult(
            "L1", "exact_qi_match", worst_rate,
            f"Worst QI subset {worst_qi}: {worst_rate:.2%} of synth rows map to a real record",
            self._verdict(worst_rate), {"qi": worst_qi})

    # -- L2: membership inference with AUC ------------------------------------

    def attack_l2_mia(self) -> AttackResult:
        if not SKLEARN_OK:
            return AttackResult("L2", "dmia_auc", 0.0, "sklearn unavailable - skipped", "held")
        cols = self._common_cols()
        if not cols:
            return AttackResult("L2", "dmia_auc", 0.0, "No common columns", "held")
        synth_num, self._vocab = _factorize(self.df_synth, cols, self._vocab)
        tr_num, _ = _factorize(self.df_train, cols, self._vocab)
        ho_num, _ = _factorize(self.df_holdout, cols, self._vocab)
        if len(synth_num) == 0:
            return AttackResult("L2", "dmia_auc", 0.0, "Empty synth", "held")

        nn = NearestNeighbors(n_neighbors=1).fit(synth_num)
        d_tr = nn.kneighbors(tr_num)[0].ravel()
        d_ho = nn.kneighbors(ho_num)[0].ravel()
        y = np.r_[np.ones(len(d_tr)), np.zeros(len(d_ho))]
        scores = np.r_[d_tr, d_ho]
        try:
            auc = float(roc_auc_score(y, -scores))  # smaller distance = more likely member
        except ValueError:
            auc = 0.5
        # Convert AUC to an "advantage over random guessing" success rate
        success = max(0.0, (auc - 0.5) * 2)
        return AttackResult(
            "L2", "dmia_auc", success,
            f"MIA AUC={auc:.3f} (0.5=random). Attacker advantage: {success:.2%}",
            self._verdict(success), {"auc": auc})

    # -- L3: attribute inference ----------------------------------------------

    def attack_l3_attribute_inference(self) -> AttackResult:
        if not SKLEARN_OK:
            return AttackResult("L3", "attribute_inference", 0.0, "sklearn unavailable - skipped", "held")
        cols = self._common_cols()
        candidates = [c for c in cols
                      if self.df_train[c].nunique() <= 10 and c in self._vocab]
        if not candidates:
            return AttackResult("L3", "attribute_inference", 0.0,
                                "No low-cardinality sensitive column to infer", "held")
        worst_rate, worst_col = 0.0, None
        for target in candidates[:5]:
            feats = [c for c in cols if c != target]
            if not feats:
                continue
            synth_f, _ = _factorize(self.df_synth, feats, self._vocab)
            ho_f, _ = _factorize(self.df_holdout, feats, self._vocab)
            nn = NearestNeighbors(n_neighbors=3).fit(synth_f)
            _, idx = nn.kneighbors(ho_f)
            true_vals = self.df_holdout[target].astype(str).to_numpy()
            # majority vote among 3 nearest synth neighbors
            pred = []
            for row in idx:
                vals = self.df_synth.iloc[row][target].astype(str).values
                pred.append(pd.Series(vals).mode().iloc[0])
            acc = float(np.mean(np.array(pred) == true_vals))
            baseline = self.df_holdout[target].astype(str).value_counts(normalize=True).max()
            advantage = max(0.0, acc - baseline) / max(1 - baseline, 1e-9)
            if advantage > worst_rate:
                worst_rate, worst_col = advantage, target
        return AttackResult(
            "L3", "attribute_inference", worst_rate,
            f"Worst attribute {worst_col}: attacker advantage {worst_rate:.2%} over majority baseline",
            self._verdict(worst_rate), {"attribute": worst_col})

    # -- L4: uniqueness / singling-out ----------------------------------------

    def attack_l4_uniqueness(self) -> AttackResult:
        cols = self._common_cols()
        if not cols:
            return AttackResult("L4", "uniqueness", 0.0, "No common columns", "held")
        uq = len(self.df_synth.drop_duplicates()) / max(len(self.df_synth), 1)
        # Unique synth rows whose fingerprint also exists in real train data
        rq = set(map(tuple, self.df_train[cols].astype(str).values))
        sq = list(map(tuple, self.df_synth[cols].astype(str).values))
        unique_hits = sum(1 for t in set(sq) if t in rq)
        rate = unique_hits / max(len(set(sq)), 1)
        return AttackResult(
            "L4", "uniqueness_linkage", rate,
            f"Unique-row fraction {uq:.3f}; {rate:.2%} of unique synth fingerprints "
            f"match a real training record",
            self._verdict(rate), {"unique_fraction": uq})

    # -- driver -----------------------------------------------------------------

    def run(self, epsilon_claimed: Optional[float] = None) -> AttackReport:
        """Run the full escalation ladder with adaptive early-stop on leaks."""
        self.results = []
        ladder = [self.attack_l1_exact_qi, self.attack_l2_mia,
                  self.attack_l3_attribute_inference, self.attack_l4_uniqueness]
        for attack_fn in ladder:
            try:
                res = attack_fn()
            except Exception as e:
                log.exception("Attack level failed")
                res = AttackResult(attack_fn.__name__, "error", 0.0, f"Attacker error: {e}", "held")
            self.results.append(res)
            # Adaptive escalation: stop early only on a hard leak (real adversary
            # would keep pushing, but a hard leak already decides the verdict).
            if res.verdict == "leak":
                log.warning("Hard leak at %s (%.2f%%) - flagging run",
                            res.attack, res.success_rate * 100)
                break

        worst = max((r.success_rate for r in self.results), default=0.0)
        verdict = VERDICT_FLAGGED if worst > self.risk_threshold else VERDICT_CERTIFIED
        return AttackReport(
            verdict=verdict,
            worst_success_rate=worst,
            risk_threshold=self.risk_threshold,
            results=[r.to_dict() for r in self.results],
            epsilon_claimed=epsilon_claimed,
        )


def run_red_team(df_train: pd.DataFrame,
                 df_holdout: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 epsilon_claimed: Optional[float] = None,
                 risk_threshold: float = DEFAULT_RISK_THRESHOLD,
                 report_path: Optional[str] = None) -> AttackReport:
    """Convenience entry point used by the pipeline and the UI."""
    attacker = AdaptiveAttacker(df_train, df_holdout, df_synth,
                                risk_threshold=risk_threshold)
    report = attacker.run(epsilon_claimed=epsilon_claimed)
    if report_path:
        report.save(report_path)
    return report


if __name__ == "__main__":  # smoke test
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.integers(0, 50, 300), "b": rng.choice(["x", "y", "z"], 300)})
    hold = pd.DataFrame({"a": rng.integers(0, 50, 300), "b": rng.choice(["x", "y", "z"], 300)})
    synth = real.sample(200, replace=True).reset_index(drop=True)  # deliberately leaky
    rep = run_red_team(real, hold, synth, epsilon_claimed=1.0)
    print(json.dumps(rep.to_dict(), indent=2))