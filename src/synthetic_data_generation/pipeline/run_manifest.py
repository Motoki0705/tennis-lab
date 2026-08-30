"""Single mutable ``run.json`` state for a canonical scene workspace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.synthetic_data_generation.pipeline.contracts import (
    ScenePipelineRequest,
    StageName,
    StageStatus,
)
from src.utils.io import load_json, save_json_atomic, utc_now_iso

RUN_MANIFEST_SCHEMA = "synthetic_scene_run_v1"
_SOURCE_COMPARE_CHUNK_BYTES = 1024 * 1024


def _same_file_contents(left: Path, right: Path) -> bool:
    """Compare two source authorities without trusting host-specific paths."""
    try:
        if left.resolve(strict=True) == right.resolve(strict=True):
            return True
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            while True:
                left_chunk = left_stream.read(_SOURCE_COMPARE_CHUNK_BYTES)
                right_chunk = right_stream.read(_SOURCE_COMPARE_CHUNK_BYTES)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


@dataclass(slots=True)
class StageRecord:
    """Current state of one stage; historical attempts are not retained."""

    status: StageStatus = StageStatus.PENDING
    attempt: int = 0
    summary: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe current-state record."""
        return {
            "status": self.status.value,
            "attempt": self.attempt,
            "summary": self.summary,
            "error": self.error,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class MutableRunManifest:
    """The one current-state manifest for all canonical stages."""

    scene_id: str
    config_schema: str
    source_video: str
    targets: list[str]
    stages: dict[StageName, StageRecord]
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(cls, request: ScenePipelineRequest) -> MutableRunManifest:
        """Create pending state for every canonical stage."""
        return cls(
            scene_id=request.scene_id,
            config_schema=request.config_schema,
            source_video=str(request.source_video),
            targets=sorted(target.value for target in request.targets),
            stages={stage: StageRecord() for stage in StageName},
        )

    @classmethod
    def load(cls, path: Path) -> MutableRunManifest:
        """Load the strict current-state schema and reject ambiguity."""
        raw = load_json(path)
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema",
            "scene_id",
            "config_schema",
            "source_video",
            "targets",
            "stages",
            "updated_at",
        }:
            raise ValueError(f"Invalid run manifest schema at {path}.")
        if raw["schema"] != RUN_MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported run manifest schema: {raw['schema']!r}.")
        stages_raw = raw["stages"]
        if not isinstance(stages_raw, Mapping) or set(stages_raw) != {
            stage.value for stage in StageName
        }:
            raise ValueError(
                "run.json must contain exactly one record for every stage."
            )
        stages: dict[StageName, StageRecord] = {}
        for stage in StageName:
            value = stages_raw[stage.value]
            if not isinstance(value, Mapping) or set(value) != {
                "status",
                "attempt",
                "summary",
                "error",
                "updated_at",
            }:
                raise ValueError(f"Invalid run record for stage {stage.value}.")
            summary = value["summary"]
            if not isinstance(summary, dict):
                raise TypeError(f"Stage {stage.value} summary must be a mapping.")
            error = value["error"]
            if error is not None and not isinstance(error, str):
                raise TypeError(f"Stage {stage.value} error must be a string or null.")
            attempt = value["attempt"]
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
                raise TypeError(f"Stage {stage.value} attempt must be non-negative.")
            stages[stage] = StageRecord(
                status=StageStatus(str(value["status"])),
                attempt=attempt,
                summary=summary,
                error=error,
                updated_at=str(value["updated_at"]),
            )
        targets = raw["targets"]
        if not isinstance(targets, list) or any(
            not isinstance(item, str) for item in targets
        ):
            raise TypeError("run.json targets must be a string list.")
        return cls(
            scene_id=str(raw["scene_id"]),
            config_schema=str(raw["config_schema"]),
            source_video=str(raw["source_video"]),
            targets=targets,
            stages=stages,
            updated_at=str(raw["updated_at"]),
        )

    def assert_request_compatible(
        self,
        request: ScenePipelineRequest,
        *,
        canonical_source_video: Path | None = None,
    ) -> None:
        """Compare semantic request fields before destructive invalidation.

        Absolute source paths may legitimately change between the host and an
        isolated execution environment. A mismatch is accepted only when the
        requested file is byte-identical to the canonical ingested scene copy.
        """
        if request.scene_id != self.scene_id:
            raise ValueError("Request scene_id disagrees with the canonical workspace.")
        if request.config_schema != self.config_schema:
            raise ValueError(
                "Request config schema is incompatible with the current scene."
            )
        source_path_matches = str(request.source_video) == self.source_video
        source_content_matches = (
            canonical_source_video is not None
            and _same_file_contents(request.source_video, canonical_source_video)
        )
        if not source_path_matches and not source_content_matches:
            raise ValueError("Request source video disagrees with the current scene.")

    def begin(self, stage: StageName) -> None:
        """Begin a pending or explicitly retryable stage attempt."""
        record = self.stages[stage]
        allowed = {
            StageStatus.PENDING,
            StageStatus.FAILED,
            StageStatus.INVALIDATED,
            StageStatus.SKIPPED,
        }
        if record.status not in allowed:
            requirement = (
                "completed stages require explicit invalidation"
                if record.status is StageStatus.COMPLETED
                else "only pending, failed, invalidated, or skipped stages may begin"
            )
            raise ValueError(
                f"Stage {stage.value} cannot begin from {record.status.value}; "
                f"{requirement}."
            )
        record.status = StageStatus.RUNNING
        record.attempt += 1
        record.summary = {}
        record.error = None
        record.updated_at = utc_now_iso()
        self.updated_at = record.updated_at

    def complete(self, stage: StageName, summary: Mapping[str, object]) -> None:
        """Mark a running stage completed only after validation/publication."""
        record = self.stages[stage]
        if record.status is not StageStatus.RUNNING:
            raise ValueError(f"Stage {stage.value} can complete only from running.")
        record.status = StageStatus.COMPLETED
        record.summary = dict(summary)
        record.error = None
        record.updated_at = utc_now_iso()
        self.updated_at = record.updated_at

    def fail(self, stage: StageName, error: BaseException) -> None:
        """Mark a running stage failed without retaining a completed claim."""
        record = self.stages[stage]
        if record.status is not StageStatus.RUNNING:
            raise ValueError(f"Stage {stage.value} can fail only from running.")
        record.status = StageStatus.FAILED
        record.summary = {}
        record.error = f"{type(error).__name__}: {error}"
        record.updated_at = utc_now_iso()
        self.updated_at = record.updated_at

    def invalidate(self, stage: StageName) -> None:
        """Invalidate one stage after successful preflight and physical unpublish."""
        record = self.stages[stage]
        record.status = StageStatus.INVALIDATED
        record.summary = {}
        record.error = None
        record.updated_at = utc_now_iso()
        self.updated_at = record.updated_at

    def skip(self, stage: StageName) -> None:
        """Record an explicitly unrequested domain without publishing it."""
        record = self.stages[stage]
        record.status = StageStatus.SKIPPED
        record.summary = {}
        record.error = None
        record.updated_at = utc_now_iso()
        self.updated_at = record.updated_at

    def save(self, path: Path) -> None:
        """Atomically replace the one fixed run.json file."""
        payload = {
            "schema": RUN_MANIFEST_SCHEMA,
            "scene_id": self.scene_id,
            "config_schema": self.config_schema,
            "source_video": self.source_video,
            "targets": self.targets,
            "stages": {
                stage.value: self.stages[stage].to_dict() for stage in StageName
            },
            "updated_at": self.updated_at,
        }
        save_json_atomic(payload, path)


__all__ = ["MutableRunManifest", "RUN_MANIFEST_SCHEMA", "StageRecord"]
