import time
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

class ComputeBudgetGuard:
    """
    Enforces a strict weekly compute budget limit (e.g., 30 hours for Kaggle).
    Tracks wall-clock session time and persists elapsed time to disk so it 
    survives kernel restarts and multiple runs.
    """
    def __init__(self, state_file: str = "compute_budget_state.json", max_hours: float = 30.0) -> None:
        self.state_file = Path(state_file)
        self.max_seconds = max_hours * 3600.0
        self.session_start = time.time()
        self.previously_elapsed = self._load_state()

    def _load_state(self) -> float:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                return state.get("elapsed_seconds", 0.0)
            except Exception as e:
                log.warning(f"Failed to load compute budget state: {e}")
        return 0.0

    def _save_state(self, total_elapsed: float) -> None:
        try:
            with open(self.state_file, 'w') as f:
                json.dump({"elapsed_seconds": total_elapsed}, f)
        except Exception as e:
            log.error(f"Failed to save GPU budget state: {e}")

    def get_elapsed_seconds(self) -> float:
        session_elapsed = time.time() - self.session_start
        return self.previously_elapsed + session_elapsed

    def check_budget(self) -> None:
        """
        Check if the budget is exceeded. Saves state on every check.
        Raises TimeoutError if budget is exhausted.
        """
        total = self.get_elapsed_seconds()
        self._save_state(total)
        
        if total > self.max_seconds:
            hours_used = total / 3600.0
            log.error(f"GPU Budget Exhausted! ({hours_used:.2f}h used)")
            raise TimeoutError(f"GPU Weekly Budget Exceeded. Max: {self.max_seconds/3600}h")
            
        remaining = self.max_seconds - total
        # Only log periodically or explicitly in the trainer loop to avoid spam
