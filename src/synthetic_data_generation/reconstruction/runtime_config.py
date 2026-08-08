"""Explicit NHT training runtime and resolved public-command config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class NHTTrainingRuntime:
    """Concrete GPU Python and NHT trainer selected by typed configuration."""

    python: Path
    trainer: Path

    def __post_init__(self) -> None:
        for name, path in (("python", self.python), ("trainer", self.trainer)):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"NHT training {name} must be an absolute path.")
            if not path.is_file():
                raise FileNotFoundError(
                    f"NHT training {name} must be a file: {path}"
                )
        if self.trainer.is_symlink():
            raise ValueError("NHT training trainer must be an ordinary file.")
        if self.trainer.name != "simple_trainer_nht.py":
            raise ValueError("NHT training trainer must be simple_trainer_nht.py.")


def resolved_nht_runtime_config(
    base_config_path: Path,
    *,
    runtime: NHTTrainingRuntime,
) -> dict[str, object]:
    """Return the strict base config with its two runtime paths made explicit."""
    if not base_config_path.is_absolute() or not base_config_path.is_file():
        raise FileNotFoundError(f"NHT base config is unavailable: {base_config_path}")
    loaded: object = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping) or any(
        not isinstance(key, str) for key in loaded
    ):
        raise ValueError("NHT base config must contain a string-keyed mapping.")
    result: dict[str, object] = dict(loaded)
    if "nht_training" not in result:
        raise ValueError("NHT base config must contain nht_training.")
    training = result["nht_training"]
    if not isinstance(training, Mapping) or any(
        not isinstance(key, str) for key in training
    ):
        raise ValueError("NHT nht_training config must be a string-keyed mapping.")
    resolved_training: dict[str, object] = dict(training)
    for key in ("python", "trainer"):
        if key not in resolved_training:
            raise ValueError(f"NHT nht_training config must contain {key}.")
    resolved_training["python"] = str(runtime.python)
    resolved_training["trainer"] = str(runtime.trainer)
    result["nht_training"] = resolved_training
    return result


def write_nht_runtime_config(
    base_config_path: Path,
    destination: Path,
    *,
    runtime: NHTTrainingRuntime,
) -> Path:
    """Atomically write the fully explicit config consumed by nht-reconstruct."""
    if not destination.is_absolute() or destination.name != "input-config.yaml":
        raise ValueError("NHT runtime config must use the fixed input-config.yaml path.")
    payload = resolved_nht_runtime_config(base_config_path, runtime=runtime)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "NHTTrainingRuntime",
    "resolved_nht_runtime_config",
    "write_nht_runtime_config",
]
