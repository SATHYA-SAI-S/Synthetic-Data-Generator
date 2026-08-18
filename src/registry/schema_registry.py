"""
registry/schema_registry.py — File-based schema registry implementation.

Persists fitted pipeline state to disk using joblib for estimator objects
and JSON for the DatasetProfile (which is a Pydantic model, hence
fully serializable without joblib).

Directory layout:
    <registry_root>/
        <dataset_name>/
            v1/
                profile.json
                pipeline_state.joblib   # scalers, encoders, handler, metadata
            v2/
                ...
            latest -> v2/              # symlink (or a latest.txt file on Windows)

Versioning:
    Each call to save() auto-increments the version unless an explicit
    version is provided. Versions are immutable once written.
    Reads are always from the latest version unless a version is specified.

Thread safety:
    save() uses an atomic write pattern (write to temp, rename). This
    protects against partial writes from session kills (Kaggle constraint).
    Concurrent writes to the same dataset_name are NOT supported — the
    pipeline is single-node, single-process.

Phase 4/5 dependency:
    Phase 4 (AutoConfigEngine) calls load_profile() — which reads only
    profile.json — without deserializing joblib estimators. Keep these
    two files separate for this reason.
"""

from __future__ import annotations

import json
import logging
import shutil
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Optional

import joblib

from src.preprocessing.base import AbstractEncoder, AbstractMissingnessHandler, AbstractScaler
from src.profiling.base import DatasetProfile
from src.registry.base import AbstractSchemaRegistry, RegistryEntry

log = logging.getLogger(__name__)

_PROFILE_FILENAME = "profile.json"
_STATE_FILENAME = "pipeline_state.joblib"
_LATEST_FILENAME = "latest.txt"



def _validate_dataset_name(name: str) -> None:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Invalid dataset_name: '{name}'")

class FileSchemaRegistry(AbstractSchemaRegistry):
    """
    File-system backed schema registry.

    Args:
        root_dir: Directory under which all registry entries are stored.
                  Will be created if it doesn't exist.

    Raises:
        RuntimeError: On any save failure (disk full, permission error, etc.).
    """

    def __init__(self, root_dir: Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        log.info("FileSchemaRegistry initialized at: %s", self._root.resolve())

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        scalers: dict[str, AbstractScaler],
        encoders: dict[str, AbstractEncoder],
        missingness_handler: AbstractMissingnessHandler,
        training_columns: list[str],
        column_types: dict[str, str],
        encoded_col_names: list[str],
        version: Optional[int] = None,
    ) -> int:
        _validate_dataset_name(dataset_name)
        dataset_dir = self._root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Determine version
        if version is None:
            existing = self.list_versions(dataset_name) if dataset_dir.exists() else []
            version = (max(existing) + 1) if existing else 1

        version_dir = dataset_dir / f"v{version}"
        if version_dir.exists():
            raise ValueError(
                f"Registry version v{version} already exists for dataset "
                f"'{dataset_name}'. Use a different version or delete first."
            )

        # Write to a temp directory first (atomic write pattern)
        with tempfile.TemporaryDirectory(dir=self._root) as tmp_str:
            tmp_dir = Path(tmp_str) / f"v{version}"
            tmp_dir.mkdir()

            # -- Write profile.json (Pydantic -> JSON) --
            profile_path = tmp_dir / _PROFILE_FILENAME
            profile_path.write_text(
                profile.model_dump_json(indent=2), encoding="utf-8"
            )

            # -- Write pipeline state via joblib --
            state = {
                "scalers": scalers,
                "encoders": encoders,
                "missingness_handler": missingness_handler,
                "training_columns": training_columns,
                "column_types": column_types,
                "encoded_col_names": encoded_col_names,
                "version": version,
                "dataset_name": dataset_name,
            }
            state_path = tmp_dir / _STATE_FILENAME
            joblib.dump(state, state_path, compress=3)
            # B-04: Write SHA-256 for integrity check
            hash_val = hashlib.sha256(state_path.read_bytes()).hexdigest()
            (tmp_dir / (_STATE_FILENAME + ".sha256")).write_text(hash_val, encoding="utf-8")

            # -- Atomic rename into final location --
            try:
                shutil.move(str(tmp_dir), str(version_dir))
            except Exception as exc:
                raise RuntimeError(
                    f"Registry save failed for dataset '{dataset_name}' v{version}: {exc}"
                ) from exc

        # R-05: Update latest.txt pointer atomically
        latest_path = dataset_dir / _LATEST_FILENAME
        tmp_latest = dataset_dir / f"{_LATEST_FILENAME}.tmp"
        tmp_latest.write_text(str(version), encoding="utf-8")
        os.replace(tmp_latest, latest_path)

        log.info(
            "Registry: saved dataset '%s' as v%d -> %s",
            dataset_name, version, version_dir,
        )
        return version

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, dataset_name: str, version: Optional[int] = None) -> RegistryEntry:
        _validate_dataset_name(dataset_name)
        version_dir = self._resolve_version_dir(dataset_name, version)

        profile = self._load_profile_from_dir(version_dir, dataset_name)
        
        state_path = version_dir / _STATE_FILENAME
        hash_path = version_dir / (_STATE_FILENAME + ".sha256")
        if hash_path.exists():
            expected = hash_path.read_text(encoding="utf-8").strip()
            actual = hashlib.sha256(state_path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Corrupted pipeline state for '{dataset_name}'. Checksum mismatch.")
        else:
            log.warning("Checksum file missing for dataset '%s'. State integrity unverified.", dataset_name)
        
        state = joblib.load(state_path)

        return RegistryEntry(
            dataset_name=dataset_name,
            profile=profile,
            scalers=state["scalers"],
            encoders=state["encoders"],
            missingness_handler=state["missingness_handler"],
            training_columns=state["training_columns"],
            column_types=state["column_types"],
            encoded_col_names=state["encoded_col_names"],
            version=state["version"],
        )

    def load_profile(self, dataset_name: str, version: Optional[int] = None) -> DatasetProfile:
        """Load only the profile."""
        _validate_dataset_name(dataset_name)
        version_dir = self._resolve_version_dir(dataset_name, version)
        return self._load_profile_from_dir(version_dir, dataset_name)

    # ── Listing & management ──────────────────────────────────────────────────

    def list_datasets(self) -> list[str]:
        return sorted(
            d.name for d in self._root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def list_versions(self, dataset_name: str) -> list[int]:
        _validate_dataset_name(dataset_name)
        dataset_dir = self._root / dataset_name
        if not dataset_dir.exists():
            return []
        versions = []
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    versions.append(int(child.name[1:]))
                except ValueError:
                    pass
        return sorted(versions)

    def delete(self, dataset_name: str, version: Optional[int] = None) -> None:
        _validate_dataset_name(dataset_name)
        dataset_dir = self._root / dataset_name
        if not dataset_dir.exists():
            raise KeyError(
                f"Dataset '{dataset_name}' is not registered."
            )
        if version is None:
            shutil.rmtree(dataset_dir)
            log.info("Registry: deleted all versions for dataset '%s'", dataset_name)
        else:
            version_dir = dataset_dir / f"v{version}"
            if not version_dir.exists():
                raise ValueError(
                    f"Version v{version} of dataset '{dataset_name}' does not exist."
                )
            shutil.rmtree(version_dir)
            # Update latest pointer if we deleted the latest
            remaining = self.list_versions(dataset_name)
            if remaining:
                latest_path = dataset_dir / _LATEST_FILENAME
                latest_path.write_text(str(max(remaining)), encoding="utf-8")
            else:
                (dataset_dir / _LATEST_FILENAME).unlink(missing_ok=True)
            log.info(
                "Registry: deleted v%d for dataset '%s'", version, dataset_name
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_version_dir(
        self, dataset_name: str, version: Optional[int]
    ) -> Path:
        dataset_dir = self._root / dataset_name
        if not dataset_dir.exists():
            raise KeyError(
                f"Dataset '{dataset_name}' is not registered in registry at "
                f"{self._root}."
            )
        if version is None:
            latest_path = dataset_dir / _LATEST_FILENAME
            if not latest_path.exists():
                raise KeyError(
                    f"No versions found for dataset '{dataset_name}'."
                )
            try:
                version = int(latest_path.read_text(encoding="utf-8").strip())
            except ValueError as e:
                raise RuntimeError(f"Corrupted latest.txt for dataset '{dataset_name}': {e}")

        version_dir = dataset_dir / f"v{version}"
        if not version_dir.exists():
            raise ValueError(
                f"Version v{version} of dataset '{dataset_name}' does not exist. "
                f"Available versions: {self.list_versions(dataset_name)}"
            )
        return version_dir

    @staticmethod
    def _load_profile_from_dir(version_dir: Path, dataset_name: str) -> DatasetProfile:
        profile_path = version_dir / _PROFILE_FILENAME
        if not profile_path.exists():
            raise RuntimeError(
                f"profile.json missing in registry at {version_dir}. "
                "Registry may be corrupted."
            )
        return DatasetProfile.model_validate_json(
            profile_path.read_text(encoding="utf-8")
        )
