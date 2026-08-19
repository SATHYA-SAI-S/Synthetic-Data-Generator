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
        if max_hours <= 0.0:
            raise ValueError(f"max_hours must be positive, got {max_hours}")
        self.state_file = Path(state_file)
        self.max_seconds = float(max_hours * 3600.0)
        self.session_start = time.time()
        self.previously_elapsed = self._load_state()

    def _load_state(self) -> float:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return float(state.get("elapsed_seconds", 0.0))
            except Exception as e:
                log.warning(f"Failed to load compute budget state: {e}")
        return 0.0

    def _save_state(self, total_elapsed: float) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_file.with_suffix(".tmp")
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump({"elapsed_seconds": float(total_elapsed)}, f)
            tmp_path.replace(self.state_file)
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

    @staticmethod
    def get_resource_stats() -> dict:
        """
        Returns a dictionary of current CPU RAM and GPU VRAM statistics.
        Useful for monitoring resources in notebooks (e.g. Kaggle / Colab).
        """
        stats = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats["cpu_ram_used_gb"] = round((mem.total - mem.available) / (1024 ** 3), 2)
            stats["cpu_ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
            stats["cpu_ram_percent"] = mem.percent
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                stats["gpu_vram_allocated_gb"] = round(allocated, 3)
                stats["gpu_vram_reserved_gb"] = round(reserved, 3)
                stats["gpu_vram_max_allocated_gb"] = round(max_alloc, 3)
                stats["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return stats
