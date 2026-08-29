"""Canonical infrastructure-stage handlers for ingest and final reporting."""

from __future__ import annotations

import html
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.dataset.contracts import (
    DatasetDomain,
    DatasetManifest,
    FrameInventory,
    TargetCourtBinding,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    ResolvedTargetCourtV2,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
    court_schema_from_dataset_schema,
)
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    StageExecutionContext,
    StageExecutionSummary,
    StageHandler,
    StageName,
)

_INGEST_SCHEMA = "canonical_scene_source_v1"
_REPORT_SCHEMA = "canonical_scene_report_v1"

if TYPE_CHECKING:
    from src.synthetic_data_generation.dataset.court.assembler import (
        CourtAssemblyReport,
    )


@dataclass(slots=True)
class DeferredStageHandler:
    """Construct one heavyweight handler only when its stage is first selected."""

    factory: Callable[[], StageHandler[StageExecutionSummary]]
    _resolved: StageHandler[StageExecutionSummary] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("Deferred stage handler factory must be callable.")

    def preflight(self, context: StageExecutionContext) -> None:
        """Resolve once and delegate preflight to the concrete handler."""
        self._resolve().preflight(context)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Delegate execution to the same concrete handler instance."""
        return self._resolve().execute(context)

    def validate(self, context: StageExecutionContext) -> None:
        """Delegate validation to the same concrete handler instance."""
        self._resolve().validate(context)

    def _resolve(self) -> StageHandler[StageExecutionSummary]:
        resolved = self._resolved
        if resolved is not None:
            return resolved
        resolved = self.factory()
        if resolved is self:
            raise TypeError("Deferred stage handler factory returned its own proxy.")
        for method in ("preflight", "execute", "validate"):
            if not callable(getattr(resolved, method, None)):
                raise TypeError(
                    "Deferred stage handler factory returned an incomplete lifecycle."
                )
        self._resolved = resolved
        return resolved


@dataclass(frozen=True, slots=True)
class VideoProperties:
    """Finite, positive video properties observed at the ingest boundary."""

    frame_count: int
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if min(self.frame_count, self.width, self.height) <= 0:
            raise ValueError("Video frame count and dimensions must be positive.")
        if self.fps <= 0.0:
            raise ValueError("Video FPS must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


@dataclass(frozen=True, slots=True)
class IngestStageHandler:
    """Copy one validated source video into the fixed scene workspace."""

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate source readability before any completed output is invalidated."""
        _require_stage(context, StageName.INGEST)
        _read_video_properties(context.request.source_video)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Write the video and semantic metadata only to attempt-local staging."""
        _require_stage(context, StageName.INGEST)
        properties = _read_video_properties(context.request.source_video)
        video_path = context.staging_path.joinpath("video.mp4")
        metadata_path = context.staging_path.joinpath("metadata.json")
        shutil.copyfile(context.request.source_video, video_path)
        payload = {
            "schema": _INGEST_SCHEMA,
            "scene_id": context.request.scene_id,
            "configured_source_video": str(context.request.source_video),
            **properties.to_dict(),
        }
        _write_json(metadata_path, payload)
        return StageExecutionSummary(properties.to_dict())

    def validate(self, context: StageExecutionContext) -> None:
        """Reopen the staged copy and require metadata to match observations."""
        _require_stage(context, StageName.INGEST)
        video_path = context.staging_path.joinpath("video.mp4")
        metadata_path = context.staging_path.joinpath("metadata.json")
        properties = _read_video_properties(video_path)
        payload = _read_json(metadata_path)
        expected = {
            "schema": _INGEST_SCHEMA,
            "scene_id": context.request.scene_id,
            "configured_source_video": str(context.request.source_video),
            **properties.to_dict(),
        }
        if payload != expected:
            raise ValueError("Staged source metadata disagrees with the copied video.")


@dataclass(frozen=True, slots=True)
class ReportStageHandler:
    """Validate requested domain manifests and publish the sole final report."""

    alignment_directory: Path
    dataset_manifests: Mapping[DatasetTarget, Path]
    _validated_inputs: dict[tuple[object, ...], dict[str, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if set(self.dataset_manifests) != set(DatasetTarget):
            raise ValueError(
                "Report handler requires one manifest path per dataset target."
            )
        if any(path.name != "dataset.json" for path in self.dataset_manifests.values()):
            raise ValueError(
                "Dataset report inputs must use the fixed dataset.json name."
            )
        if self.alignment_directory.name != "alignment":
            raise ValueError(
                "Report alignment input must be the fixed alignment directory."
            )

    def preflight(self, context: StageExecutionContext) -> None:
        """Require accepted alignment and all explicitly requested datasets."""
        _require_stage(context, StageName.REPORT)
        self._collect(context)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Write strict JSON and a small human-readable view into staging."""
        _require_stage(context, StageName.REPORT)
        payload = self._collect(context)
        _write_json(context.staging_path.joinpath("report.json"), payload)
        datasets = payload["datasets"]
        accepted_court_count = payload["accepted_court_count"]
        if not isinstance(datasets, dict) or not isinstance(accepted_court_count, int):
            raise TypeError("Internal report payload has invalid summary types.")
        rows = "".join(
            "<tr><td>"
            + html.escape(target)
            + "</td><td>"
            + str(summary["frame_inventory"]["source"])
            + "</td><td>"
            + str(len(summary["target_courts"]))
            + "</td></tr>"
            for target, summary in sorted(datasets.items())
        )
        document = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>Canonical scene report</title></head><body>"
            f"<h1>{html.escape(context.request.scene_id)}</h1>"
            "<table><thead><tr><th>dataset</th><th>frames</th><th>courts</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></body></html>\n"
        )
        context.staging_path.joinpath("index.html").write_text(
            document, encoding="utf-8"
        )
        return StageExecutionSummary(
            {
                "dataset_count": len(datasets),
                "accepted_court_count": accepted_court_count,
            }
        )

    def validate(self, context: StageExecutionContext) -> None:
        """Recompute report evidence and reject missing or stale staged content."""
        _require_stage(context, StageName.REPORT)
        actual = _read_json(context.staging_path.joinpath("report.json"))
        expected = self._collect(context)
        if actual != expected:
            raise ValueError("Staged report does not match current canonical datasets.")
        index_path = context.staging_path.joinpath("index.html")
        if (
            not index_path.is_file()
            or context.request.scene_id not in index_path.read_text(encoding="utf-8")
        ):
            raise ValueError("Staged HTML report is missing its scene identity.")

    def _collect(self, context: StageExecutionContext) -> dict[str, object]:
        cache_key = self._cache_key(context)
        cached = self._validated_inputs.get(cache_key)
        if cached is not None:
            return cached
        alignment = validate_alignment_outputs(self.alignment_directory)
        datasets: dict[str, object] = {}
        for target in sorted(context.request.targets, key=lambda item: item.value):
            path = self.dataset_manifests[target]
            manifest = _validate_domain_manifest(target, path)
            if manifest.scene_id != context.request.scene_id:
                raise ValueError(f"{target.value} dataset belongs to another scene.")
            if manifest.domain is not DatasetDomain(target.value):
                raise ValueError(f"{target.value} dataset declares the wrong domain.")
            for diagnostic in manifest.diagnostics:
                candidate = path.parent.joinpath(diagnostic).resolve(strict=False)
                if not candidate.is_relative_to(path.parent.resolve(strict=False)):
                    raise ValueError(
                        f"Dataset diagnostic escapes its owner: {diagnostic}."
                    )
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Dataset diagnostic is missing: {candidate}."
                    )
            datasets[target.value] = manifest.to_dict()
        payload: dict[str, object] = {
            "schema": _REPORT_SCHEMA,
            "scene_id": context.request.scene_id,
            "accepted_court_count": len(alignment.layout.courts),
            "primary_court_instance_id": alignment.layout.primary_court_instance_id,
            "datasets": datasets,
        }
        self._validated_inputs.clear()
        self._validated_inputs[cache_key] = payload
        return payload

    def _cache_key(self, context: StageExecutionContext) -> tuple[object, ...]:
        manifest_identities = tuple(
            (target.value, *_file_identity(self.dataset_manifests[target]))
            for target in sorted(context.request.targets, key=lambda item: item.value)
        )
        alignment_identities = tuple(
            (
                path.relative_to(self.alignment_directory).as_posix(),
                *_file_identity(path),
            )
            for path in sorted(self.alignment_directory.rglob("*"))
            if path.is_file()
        )
        return (
            context.request.scene_id,
            tuple(sorted(target.value for target in context.request.targets)),
            alignment_identities,
            manifest_identities,
        )


def _validate_domain_manifest(target: DatasetTarget, path: Path) -> DatasetManifest:
    """Run the sole strict domain reader before projecting report metadata."""
    if target is DatasetTarget.COURT:
        from src.synthetic_data_generation.dataset.court.assembler import (
            CourtArrayValidationMode,
            validate_court_dataset,
        )

        report = validate_court_dataset(
            path.parent,
            array_validation=CourtArrayValidationMode.HEADERS_ONLY,
        )
        payload = _read_json(path)
        return _court_report_manifest(
            payload,
            report=report,
            schema_version=court_schema_from_dataset_schema(
                payload.get("schema")
            ).version,
        )
    if target is DatasetTarget.BLCS:
        from src.synthetic_data_generation.dataset.blcs.assembler import (
            validate_blcs_dataset,
        )

        return validate_blcs_dataset(path.parent).manifest
    if target is DatasetTarget.PLCS:
        from src.synthetic_data_generation.dataset.plcs.validation import (
            validate_plcs_dataset,
        )

        validate_plcs_dataset(path.parent)
        payload = _read_json(path)
        common_keys = (
            "scene_id",
            "domain",
            "schema",
            "frame_inventory",
            "target_courts",
            "metadata",
            "diagnostics",
        )
        return DatasetManifest.from_dict({key: payload[key] for key in common_keys})
    raise ValueError(f"Unsupported report dataset target: {target!r}.")


def _court_report_manifest(
    payload: Mapping[str, object],
    *,
    report: CourtAssemblyReport,
    schema_version: CourtDatasetSchemaVersion | None = None,
) -> DatasetManifest:
    """Project the validated Court release into the common report contract."""
    version = schema_version or CourtDatasetSchemaVersion.V1
    bindings: dict[str, TargetCourtBinding] = {}
    if version is CourtDatasetSchemaVersion.V1:
        records = payload.get("trajectory_groups")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("Court trajectory_groups must be a sequence.")
        raw_bindings = []
        for group in records:
            if not isinstance(group, Mapping) or "target_court" not in group:
                raise TypeError(
                    "Court trajectory group lacks its target-court binding."
                )
            raw_bindings.append(TargetCourtBinding.from_dict(group["target_court"]))
    elif version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ):
        accepted = payload.get("samples")
        rejected = payload.get("rejected_samples")
        if (
            not isinstance(accepted, Sequence)
            or isinstance(accepted, (str, bytes))
            or not isinstance(rejected, Sequence)
            or isinstance(rejected, (str, bytes))
        ):
            raise TypeError("Court v2 sample inventories must be sequences.")
        raw_bindings = []
        for sample in (*accepted, *rejected):
            if not isinstance(sample, Mapping):
                raise TypeError("Court v2 sample must be a mapping.")
            raw_bindings.append(
                ResolvedTargetCourtV2.from_mapping(sample.get("target_court")).binding
            )
    else:  # pragma: no cover - exact schema registry is exhaustive
        raise TypeError("Unsupported Court report schema version.")
    for binding in raw_bindings:
        previous = bindings.setdefault(binding.court_instance_id, binding)
        if previous != binding:
            raise ValueError("Court records disagree on a target-court binding.")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        raise TypeError("Court diagnostics must be a sequence.")
    if any(not isinstance(value, str) for value in diagnostics):
        raise TypeError("Court diagnostic paths must be strings.")
    scene_id = payload.get("scene_id")
    schema = payload.get("schema")
    profile = payload.get("profile")
    if not isinstance(scene_id, str) or not isinstance(schema, str):
        raise TypeError("Court scene_id and schema must be strings.")
    if not isinstance(profile, str):
        raise TypeError("Court profile must be a string.")
    frame_indices = tuple(range(report.accepted_frame_count))
    return DatasetManifest(
        scene_id=scene_id,
        domain=DatasetDomain.COURT,
        schema=schema,
        frame_inventory=FrameInventory(
            source_count=report.accepted_frame_count,
            planned_indices=frame_indices,
            rendered_indices=frame_indices,
            labelled_indices=frame_indices,
        ),
        target_courts=tuple(bindings.values()),
        metadata={
            "profile": profile,
            "proposal_count": report.proposal_count,
            "accepted_frame_count": report.accepted_frame_count,
            "rejected_frame_count": report.rejected_frame_count,
            "trajectory_group_count": report.trajectory_group_count,
        },
        diagnostics=tuple(diagnostics),
    )


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _require_stage(context: StageExecutionContext, expected: StageName) -> None:
    if context.stage.name is not expected:
        raise ValueError(
            f"{type(context).__name__} is for {expected.value}, got {context.stage.name.value}."
        )


def _read_video_properties(path: Path) -> VideoProperties:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Source video is not an ordinary file: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Source video cannot be opened: {path}")
        return VideoProperties(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected an ordinary JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"JSON root must be a string-keyed object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DeferredStageHandler",
    "IngestStageHandler",
    "ReportStageHandler",
    "VideoProperties",
]
