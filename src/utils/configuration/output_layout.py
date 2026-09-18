"""Names for task artifacts; root authority remains owned by PathResolver."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

from omegaconf import OmegaConf

from src.utils.configuration.errors import PathContractError

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
OUTPUT_KINDS = frozenset(
    {"train", "evaluate", "visualize", "analyze", "generate", "precompute"}
)


def _component(value: str) -> str:
    if type(value) is not str or _COMPONENT.fullmatch(value) is None:
        raise PathContractError(
            "Output identity must be one non-empty path component using "
            f"letters, digits, '_', '-' or '.': {value!r}."
        )
    return value


def task_output_path(task: str, kind: str, experiment: str, run_id: str) -> str:
    """Return the configured role-relative task/kind/experiment/run hierarchy."""
    if kind not in OUTPUT_KINDS:
        raise PathContractError(f"Unsupported task output kind: {kind!r}.")
    return "/".join(_component(value) for value in (task, kind, experiment, run_id))


def dataset_output_path(task: str, version: str) -> str:
    """Name a durable, versioned dataset below the explicit data root."""
    return "/".join(_component(value) for value in (task, version))


def new_run_id() -> str:
    """Generate an identity once per composed config (OmegaConf caches it)."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{uuid4().hex[:8]}"


def register_output_resolvers() -> None:
    """Enable paths in programmatic compose as well as Hydra CLI applications."""
    OmegaConf.register_new_resolver("tennis_output", task_output_path)
    OmegaConf.register_new_resolver("tennis_dataset", dataset_output_path)
    OmegaConf.register_new_resolver("tennis_run_id", new_run_id, use_cache=True)
