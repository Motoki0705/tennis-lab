"""
Provide shared typed boundaries for alignment stage orchestration.

Usage:
    python -m src.synthetic_data_generation.scripts.alignment.common --cfg job

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/alignment/pipeline.yaml`.
    - The CLI only composes and reports shared pipeline configuration.
    - Resume candidates must pass their strict artifact loader before reuse.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from src.synthetic_data_generation.alignment.scene_provider.bundle import sha256_file
from src.utils.hydra import hydra_main


@dataclass(frozen=True)
class AlignmentJob:
    """One serially executed court-alignment job."""

    alignment_id: str
    scene_id: str
    provider_bundle: Path
    output_root: Path
    config_overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ArtifactHandle:
    """Strict identity of one immutable artifact boundary."""

    path: Path
    artifact_id: str
    fingerprint: str
    file_sha256: str
    schema: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible artifact reference."""
        return {
            "path": str(self.path),
            "artifact_id": self.artifact_id,
            "fingerprint": self.fingerprint,
            "file_sha256": self.file_sha256,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class StageResult:
    """Machine-readable result shared by direct and pipeline execution."""

    stage: str
    status: str
    artifact_paths: tuple[Path, ...]
    primary_artifact: Path | None
    fingerprint: str | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible stage result."""
        return {
            "stage": self.stage,
            "status": self.status,
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "primary_artifact": (
                str(self.primary_artifact)
                if self.primary_artifact is not None
                else None
            ),
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
        }


class AlignmentStageError(RuntimeError):
    """Stage failure that retains every immutable artifact already published."""

    stage: str
    job_id: str
    preserved_artifacts: tuple[Path, ...]

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        job_id: str,
        preserved_artifacts: Sequence[Path] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.job_id = job_id
        self.preserved_artifacts = tuple(preserved_artifacts)


def json_artifact_handle(
    path: Path,
    payload: Mapping[str, Any],
) -> ArtifactHandle:
    """Build a handle for an already strict-loaded JSON artifact."""
    return ArtifactHandle(
        path=path.resolve(),
        artifact_id=_required_string(payload, "artifact_id"),
        fingerprint=_required_string(payload, "artifact_fingerprint"),
        file_sha256=str(sha256_file(path)),
        schema=_required_string(payload, "schema"),
    )


def directory_artifact_handle(
    path: Path,
    manifest: Mapping[str, Any],
) -> ArtifactHandle:
    """Build a handle for a strict-loaded directory artifact manifest."""
    manifest_path = path.resolve() / "manifest.json"
    return ArtifactHandle(
        path=path.resolve(),
        artifact_id=_required_string(manifest, "artifact_id"),
        fingerprint=_required_string(manifest, "artifact_fingerprint"),
        file_sha256=str(sha256_file(manifest_path)),
        schema=_required_string(manifest, "schema"),
    )


def find_matching_artifact(
    candidates: Sequence[Path],
    *,
    load: Callable[[Path], Mapping[str, Any]],
    matches: Callable[[Mapping[str, Any]], bool],
) -> tuple[Path, Mapping[str, Any]] | None:
    """Return the newest strictly verified candidate satisfying all inputs."""
    verified: list[tuple[str, Path, Mapping[str, Any]]] = []
    for candidate in sorted(candidates):
        payload = load(candidate)
        if matches(payload):
            created_at = _required_string(payload, "created_at_utc")
            verified.append((created_at, candidate.resolve(), payload))
    if not verified:
        return None
    _, path, payload = max(verified, key=lambda item: (item[0], str(item[1])))
    return path, payload


def write_json_summary(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic pipeline summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def print_stage_result(result: StageResult) -> None:
    """Print one stable JSON object for direct CLI users."""
    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Artifact field {key!r} must be a non-empty string.")
    return value


@hydra_main(
    version_base="1.3",
    config_path="../../configs/alignment",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Compose and report the shared pipeline configuration."""
    print(
        json.dumps(
            {
                "pipeline_id": str(cfg.pipeline_id),
                "max_parallel_jobs": int(cfg.max_parallel_jobs),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    cast(Any, main)()
