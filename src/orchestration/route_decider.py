"""
Route Decider - Automatic train-vs-pretrain routing based on dataset size.

Decides whether an uploaded dataset should:
  - go through the small-N adapter path (registry lookup -> schema adapter ->
    DP fine-tune adapter -> generate), or
  - be pushed to Kaggle for full DP-SGD training from scratch.

The decision is made AFTER preprocessing (HIPAA stripping), so it is based on
the clean row count, not the raw upload size.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default small-N threshold. Below this, full DP-SGD from scratch tends to
# underfit and waste privacy budget; adapter fine-tuning on a pretrained
# backbone is recommended instead.
DEFAULT_SMALL_N_THRESHOLD = 10_000


@dataclass
class RouteDecision:
    """Structured routing decision persisted to session state."""
    route: str                       # "adapter" | "kaggle"
    reason: str                      # human-readable explanation
    clean_rows: int
    threshold: int
    recommended_action: str
    overridden_by_user: bool = False
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteDecision":
        return cls(**d)


class RouteDecider:
    """
    Rule-based router. The framework recommends a route; the user may override
    (recorded via `overridden_by_user`).
    """

    def __init__(self, small_n_threshold: int = DEFAULT_SMALL_N_THRESHOLD):
        if small_n_threshold <= 0:
            raise ValueError(f"small_n_threshold must be positive, got {small_n_threshold}")
        self.small_n_threshold = int(small_n_threshold)

    def decide(self, clean_rows: int) -> RouteDecision:
        """Produce a RouteDecision for a cleaned dataset of `clean_rows` rows."""
        if clean_rows < self.small_n_threshold:
            return RouteDecision(
                route="adapter",
                reason=(
                    f"Dataset has {clean_rows:,} clean rows (< {self.small_n_threshold:,} "
                    f"threshold). Small-N DP-SGD from scratch underfits and wastes "
                    f"privacy budget; pretraining on a larger public backbone with a "
                    f"schema-adapter fine-tune is recommended."
                ),
                clean_rows=clean_rows,
                threshold=self.small_n_threshold,
                recommended_action=(
                    "Adapter route: registry lookup -> attach schema adapter -> "
                    "DP fine-tune adapter (frozen backbone) -> generate."
                ),
            )
        return RouteDecision(
            route="kaggle",
            reason=(
                f"Dataset has {clean_rows:,} clean rows (>= {self.small_n_threshold:,} "
                f"threshold). Sufficient scale for full DP-SGD training on Kaggle GPU."
            ),
            clean_rows=clean_rows,
            threshold=self.small_n_threshold,
            recommended_action=(
                "Kaggle route: package de-identified data + config -> push private "
                "dataset & kernel -> stream progress -> pull artifacts."
            ),
        )

    def apply_override(self, decision: RouteDecision, forced_route: str) -> RouteDecision:
        """Record a user override of the recommended route."""
        if forced_route not in ("adapter", "kaggle"):
            raise ValueError(f"Invalid forced route: {forced_route}")
        decision.route = forced_route
        decision.overridden_by_user = True
        decision.recommended_action += " [User override applied]"
        log.info("Route override by user: forced=%s (recommended reason kept)", forced_route)
        return decision


def save_decision(decision: RouteDecision, session_dir: str) -> Path:
    """Persist the decision alongside session artifacts."""
    out = Path(session_dir) / "route_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(out)
    return out


def load_decision(session_dir: str) -> Optional[RouteDecision]:
    p = Path(session_dir) / "route_decision.json"
    if not p.exists():
        return None
    try:
        return RouteDecision.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        log.warning("Failed to load route decision: %s", e)
        return None