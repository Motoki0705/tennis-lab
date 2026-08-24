"""Deterministic RNG ownership for scene datasets and data loaders."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from torch.utils.data import get_worker_info

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)

_MAX_RUN_SEED = 2**32 - 1
_MAX_DERIVED_SEED = 2**63 - 1


def validate_run_seed(value: object, *, path: str = "run.seed") -> int:
    """Return one repository-compatible seed or reject it explicitly."""
    if type(value) is not int:
        raise TypeError(f"{path} must be an int, got {type(value).__name__}.")
    seed = int(value)
    if not 0 <= seed <= _MAX_RUN_SEED:
        raise ValueError(
            f"{path} must be between 0 and {_MAX_RUN_SEED}, got {seed}."
        )
    return seed


def validate_seed(value: object, *, path: str = "seed") -> int:
    """Return one non-negative seed accepted by shared RNG derivation."""
    if type(value) is not int:
        raise TypeError(f"{path} must be an int, got {type(value).__name__}.")
    seed = int(value)
    if not 0 <= seed <= _MAX_DERIVED_SEED:
        raise ValueError(
            f"{path} must be between 0 and {_MAX_DERIVED_SEED}, got {seed}."
        )
    return seed


def require_run_seed(config: object) -> int:
    """Read and validate the required ``run.seed`` from a composed config."""
    root = as_config_mapping(config, path="configuration")
    run = require_config_mapping(root, "run", path="configuration")
    value = require_config_value(run, "seed", int, path="run")
    return validate_run_seed(value)


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a stable non-negative seed without Python's randomized ``hash``."""
    validated = validate_seed(base_seed, path="base_seed")
    digest = hashlib.blake2b(digest_size=8, person=b"scene-rng-v1")
    digest.update(validated.to_bytes(8, byteorder="little", signed=False))
    for component in components:
        if type(component) is int:
            if not 0 <= component <= 2**64 - 1:
                raise ValueError(
                    "Integer seed components must be between 0 and 2**64 - 1."
                )
            payload = int(component).to_bytes(8, byteorder="little", signed=False)
            type_tag = b"i"
        elif isinstance(component, str):
            payload = component.encode("utf-8")
            type_tag = b"s"
        else:
            raise TypeError(
                "Seed components must be str or int, got "
                f"{type(component).__name__}."
            )
        digest.update(type_tag)
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)
    return int.from_bytes(digest.digest(), byteorder="little") & _MAX_DERIVED_SEED


@runtime_checkable
class WorkerSeededSceneDataset(Protocol):
    """Dataset protocol required by the shared scene DataModule."""

    def seed_worker(self, *, worker_seed: int, worker_id: int) -> None:
        """Install the deterministic RNG stream assigned to one worker."""


def seed_scene_dataset_worker(worker_id: int) -> None:
    """Install a deterministic, decorrelated NumPy stream in a loader worker."""
    info = get_worker_info()
    if info is None:
        raise RuntimeError("seed_scene_dataset_worker must run inside a DataLoader worker.")
    dataset = info.dataset
    if not isinstance(dataset, WorkerSeededSceneDataset):
        raise TypeError(
            "SceneDirectoryDataModule datasets must implement seed_worker(); got "
            f"{type(dataset).__name__}."
        )
    dataset.seed_worker(worker_seed=int(info.seed), worker_id=worker_id)


def require_worker_seeded_dataset(dataset: object) -> WorkerSeededSceneDataset:
    """Validate the DataModule/dataset RNG boundary before workers are started."""
    if not isinstance(dataset, WorkerSeededSceneDataset):
        raise TypeError(
            "SceneDirectoryDataModule datasets must implement seed_worker(); got "
            f"{type(dataset).__name__}."
        )
    return dataset


__all__ = [
    "WorkerSeededSceneDataset",
    "derive_seed",
    "require_run_seed",
    "require_worker_seeded_dataset",
    "seed_scene_dataset_worker",
    "validate_seed",
    "validate_run_seed",
]
