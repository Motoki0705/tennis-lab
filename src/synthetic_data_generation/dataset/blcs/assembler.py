"""Fail-closed assembly and on-demand reading for compact BLCS datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.composition import (
    GaussianDeformationKind,
    GaussianForegroundComposition,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCS_DATASET_SCHEMA,
    BLCS_SAMPLE_SCHEMA,
    BLCSSampleRecord,
)
from src.synthetic_data_generation.dataset.blcs.diagnostics import (
    write_blcs_diagnostics,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSRenderAttempt,
    build_blcs_sample_metadata,
)
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan
from src.synthetic_data_generation.dataset.continuity import (
    FrameContinuityReport,
    TimelineFrameRecord,
    validate_frame_continuity,
)
from src.synthetic_data_generation.dataset.contracts import (
    DatasetDomain,
    DatasetManifest,
    FrameInventory,
    TargetCourtBinding,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
    FinalDatasetAssembler,
    LogicalRenderSample,
    PerformanceTimer,
    RenderSampleKey,
    SharedBackgroundStore,
    directory_size_bytes,
    materialize_logical_sample,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

MEASURED_DENSE_REFERENCE_BYTES = 175 * 1024**3

_DATASET_KEYS = {
    "schema",
    "scene_id",
    "domain",
    "frame_inventory",
    "target_courts",
    "metadata",
    "diagnostics",
    "performance",
    "trajectories",
    "samples",
}
_TRAJECTORY_KEYS = {
    "trajectory_id",
    "split",
    "source_frame_count",
    "global_frame_offset",
    "frame_inventory",
    "target_court",
    "candidate_id",
    "transform",
    "camera_profile",
    "camera_seed",
    "camera_ids",
    "attempt_token",
    "chunk_count",
    "chunk_directories",
    "background_store",
    "plan_json",
    "plan_npz",
}
_PLAN_KEYS = {
    "trajectory_id",
    "split",
    "fps",
    "source_frame_count",
    "global_frame_offset",
    "global_frame_indices",
    "tracks",
    "target_court",
    "camera_profile",
    "camera_seed",
    "cameras",
    "chunks",
    "composition",
    "source_metadata",
}
_SAMPLE_METADATA_KEYS = {
    "schema",
    "scene_id",
    "trajectory_id",
    "split",
    "global_frame_index",
    "source_frame_index",
    "chunk_index",
    "source_trajectory",
    "source_frame",
    "target_court",
    "candidate_id",
    "transform",
    "camera_profile",
    "camera_parameters",
    "seed",
    "objects",
    "semantic_arrays",
}


@dataclass(frozen=True, slots=True)
class BLCSAssemblyResult:
    """Validated whole-dataset inventory, continuity and performance evidence."""

    manifest: DatasetManifest
    continuity: FrameContinuityReport
    sample_records: tuple[BLCSSampleRecord, ...]
    metric_adapter: MetricSceneAdapter
    performance: DatasetPerformanceMetrics


@dataclass(frozen=True, slots=True)
class BLCSLogicalSample:
    """One on-demand full logical render plus compact semantic supervision."""

    record: BLCSSampleRecord
    render: LogicalRenderSample
    metadata: Mapping[str, object]
    semantic_arrays: Mapping[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class BLCSTrackIndex:
    """One strict active interval within a canonical trajectory."""

    trajectory_id: str
    split: str
    object_id: str
    object_index: int
    start_frame: int
    stop_frame: int


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryIndex:
    """Manifest-authoritative complete trajectory index."""

    trajectory_id: str
    split: str
    frame_count: int
    camera_ids: tuple[str, ...]
    tracks: tuple[BLCSTrackIndex, ...]


@dataclass(frozen=True, slots=True)
class BLCSAllViewTrajectory:
    """Dense semantic tensors for every generated view and source frame."""

    index: BLCSTrajectoryIndex
    ball_uv: NDArray[np.float32]
    ball_visible: NDArray[np.bool_]
    court_kp: NDArray[np.float32]
    court_visible: NDArray[np.bool_]
    positions_court_m: NDArray[np.float32]
    velocities_court_mps: NDArray[np.float32]
    present: NDArray[np.bool_]
    camera_R: NDArray[np.float32]
    camera_C: NDArray[np.float32]
    camera_f: NDArray[np.float32]
    camera_cx: NDArray[np.float32]
    camera_cy: NDArray[np.float32]
    camera_w: NDArray[np.float32]
    camera_h: NDArray[np.float32]


class BLCSCompactDatasetReader:
    """Strict current-schema reader with cached backgrounds and chunk payloads."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.result = validate_blcs_dataset(output_directory)
        self._records = {
            (
                record.trajectory_id,
                record.source_frame_index,
                record.camera_id,
            ): record
            for record in self.result.sample_records
        }
        self._backgrounds: dict[Path, SharedBackgroundStore] = {}
        self._chunks: dict[Path, ChunkReader] = {}
        payload = _mapping(
            _load_json(output_directory / "dataset.json"),
            keys=_DATASET_KEYS,
            name="BLCS dataset",
        )
        self._trajectory_records = {
            _string(record["trajectory_id"], name="trajectory_id"): record
            for value in _list(payload["trajectories"], name="trajectories")
            for record in (_mapping(value, keys=_TRAJECTORY_KEYS, name="trajectory"),)
        }
        self.trajectories = tuple(
            self._build_trajectory_index(record)
            for record in self._trajectory_records.values()
        )
        self.tracks = tuple(
            track for trajectory in self.trajectories for track in trajectory.tracks
        )

    def split_trajectories(self, split: str) -> tuple[BLCSTrajectoryIndex, ...]:
        """Return manifest-ordered trajectories for exactly one split."""
        canonical = "validation" if split == "val" else split
        if canonical not in {"train", "validation", "test"}:
            raise ValueError("BLCS split must be train, validation/val, or test.")
        return tuple(value for value in self.trajectories if value.split == canonical)

    def split_tracks(self, split: str) -> tuple[BLCSTrackIndex, ...]:
        """Return manifest-ordered object intervals for exactly one split."""
        selected = {value.trajectory_id for value in self.split_trajectories(split)}
        return tuple(value for value in self.tracks if value.trajectory_id in selected)

    def materialize_all_views(self, trajectory_id: str) -> BLCSAllViewTrajectory:
        """Load every generated camera and every source frame without selection."""
        try:
            record = self._trajectory_records[trajectory_id]
        except KeyError as error:
            raise KeyError(f"Unknown BLCS trajectory: {trajectory_id!r}.") from error
        plan = _mapping(
            _load_json(
                _contained_file(
                    self.output_directory,
                    _string(record["plan_json"], name="plan_json"),
                )
            ),
            keys=_PLAN_KEYS,
            name="BLCS plan",
        )
        archive_path = _contained_file(
            self.output_directory, _string(record["plan_npz"], name="plan_npz")
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        cameras = tuple(
            SceneCamera.from_dict(
                _mapping(
                    item,
                    keys={
                        "slot_id",
                        "court_local_center_m",
                        "court_local_look_at_m",
                        "hfov_degrees",
                        "camera",
                    },
                    name="planned camera",
                )["camera"]
            )
            for item in _list(plan["cameras"], name="cameras")
        )
        binding = TargetCourtBinding.from_dict(plan["target_court"])
        camera_to_court = tuple(
            binding.scene_from_court.inverse().matrix()
            @ camera.camera_to_scene.matrix()
            for camera in cameras
        )
        widths = np.asarray([camera.width for camera in cameras], dtype=np.float64)
        heights = np.asarray([camera.height for camera in cameras], dtype=np.float64)
        uv = arrays["camera_uv"].transpose(1, 0, 2, 3).copy()
        uv[..., 0] /= widths[:, None, None]
        uv[..., 1] /= heights[:, None, None]
        court_uv = arrays["court_uv"].copy()
        court_uv[..., 0] /= widths[:, None]
        court_uv[..., 1] /= heights[:, None]
        index = next(
            value for value in self.trajectories if value.trajectory_id == trajectory_id
        )
        rendered_visible = self._rendered_visibility(index)
        return BLCSAllViewTrajectory(
            index=index,
            ball_uv=uv.astype(np.float32),
            ball_visible=(rendered_visible & arrays["present"][None]),
            court_kp=court_uv.astype(np.float32),
            court_visible=arrays["court_visible"].astype(np.bool_),
            positions_court_m=arrays["positions_court_m"].astype(np.float32),
            velocities_court_mps=arrays["velocities_court_mps"].astype(np.float32),
            present=arrays["present"].astype(np.bool_),
            camera_R=np.stack([value[:3, :3].T for value in camera_to_court]).astype(
                np.float32
            ),
            camera_C=np.stack([value[:3, 3] for value in camera_to_court]).astype(
                np.float32
            ),
            camera_f=np.asarray(
                [camera.intrinsics[0] for camera in cameras], dtype=np.float32
            ),
            camera_cx=np.asarray(
                [camera.intrinsics[2] for camera in cameras], dtype=np.float32
            ),
            camera_cy=np.asarray(
                [camera.intrinsics[5] for camera in cameras], dtype=np.float32
            ),
            camera_w=widths.astype(np.float32),
            camera_h=heights.astype(np.float32),
        )

    def _rendered_visibility(self, index: BLCSTrajectoryIndex) -> NDArray[np.bool_]:
        """Load compact renderer visibility without materializing RGB/depth arrays."""
        result: NDArray[np.bool_] = np.zeros(
            (len(index.camera_ids), index.frame_count, len(index.tracks)),
            dtype=np.bool_,
        )
        for camera_index, camera_id in enumerate(index.camera_ids):
            for frame_index in range(index.frame_count):
                record = self._records[(index.trajectory_id, frame_index, camera_id)]
                chunk_path = _contained_directory(
                    self.output_directory, record.foreground_chunk
                )
                reader = self._chunks.setdefault(chunk_path, ChunkReader(chunk_path))
                metadata = reader.metadata()[record.chunk_sample_index]
                visible = _semantic_arrays(metadata)["rendered_visible"]
                if visible.shape != (len(index.tracks),):
                    raise ValueError(
                        "BLCS rendered visibility disagrees with its track axis."
                    )
                result[camera_index, frame_index] = visible
        return result

    def _build_trajectory_index(
        self, record: Mapping[str, object]
    ) -> BLCSTrajectoryIndex:
        plan = _mapping(
            _load_json(
                _contained_file(
                    self.output_directory,
                    _string(record["plan_json"], name="plan_json"),
                )
            ),
            keys=_PLAN_KEYS,
            name="BLCS plan",
        )
        trajectory_id = _string(record["trajectory_id"], name="trajectory_id")
        split = _string(record["split"], name="split")
        if split not in {"train", "validation", "test"}:
            raise ValueError("BLCS trajectory split is unsupported.")
        frame_count = _integer(record["source_frame_count"], name="frame_count")
        camera_ids = tuple(
            _string(value, name="camera_id")
            for value in _list(record["camera_ids"], name="camera_ids")
        )
        mappings, object_ids, _ = _plan_tracks(plan["tracks"], frame_count=frame_count)
        tracks = []
        for object_index, (object_id, mapping) in enumerate(
            zip(object_ids, mappings, strict=True)
        ):
            active = tuple(
                index for index, value in enumerate(mapping) if value is not None
            )
            tracks.append(
                BLCSTrackIndex(
                    trajectory_id=trajectory_id,
                    split=split,
                    object_id=object_id,
                    object_index=object_index,
                    start_frame=active[0],
                    stop_frame=active[-1] + 1,
                )
            )
        return BLCSTrajectoryIndex(
            trajectory_id=trajectory_id,
            split=split,
            frame_count=frame_count,
            camera_ids=camera_ids,
            tracks=tuple(tracks),
        )

    def materialize(
        self,
        *,
        trajectory_id: str,
        source_frame_index: int,
        camera_id: str,
    ) -> BLCSLogicalSample:
        """Materialize only the requested RGB/depth/label sample."""
        try:
            record = self._records[(trajectory_id, source_frame_index, camera_id)]
        except KeyError as error:
            raise KeyError("Unknown BLCS logical sample key.") from error
        background_path = _contained_directory(
            self.output_directory, record.background_store
        )
        store = self._backgrounds.setdefault(
            background_path, SharedBackgroundStore(background_path)
        )
        chunk_path = _contained_directory(
            self.output_directory, record.foreground_chunk
        )
        reader = self._chunks.setdefault(chunk_path, ChunkReader(chunk_path))
        deltas = reader.deltas()
        metadata = reader.metadata()
        if record.chunk_sample_index >= len(deltas):
            raise ValueError("BLCS sample index exceeds its foreground chunk.")
        delta = deltas[record.chunk_sample_index]
        expected_key = RenderSampleKey(record.source_frame_index, record.camera_id)
        if delta.key != expected_key:
            raise ValueError("BLCS record and foreground key disagree.")
        sample_metadata = metadata[record.chunk_sample_index]
        return BLCSLogicalSample(
            record=record,
            render=materialize_logical_sample(store.load(camera_id), delta),
            metadata=sample_metadata,
            semantic_arrays=_semantic_arrays(sample_metadata),
        )


def assemble_blcs_dataset(
    output_directory: Path,
    *,
    plans: Sequence[BLCSTrajectoryPlan],
    metric_adapter: MetricSceneAdapter,
    render_attempt: BLCSRenderAttempt,
    performance_timer: PerformanceTimer,
    performance_budget: DatasetPerformanceBudget,
) -> BLCSAssemblyResult:
    """Assemble current-attempt compact files without reopening every sample."""
    plan_tuple = tuple(plans)
    if not plan_tuple:
        raise ValueError("BLCS assembly requires at least one resolved plan.")
    if not isinstance(metric_adapter, MetricSceneAdapter):
        raise TypeError("BLCS assembly requires the accepted MetricSceneAdapter.")
    if (
        output_directory.name != "snapshot"
        or output_directory.parent.name != "blcs_dataset"
        or output_directory.parent.parent.name != ".transactions"
        or output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise ValueError("BLCS output must be its transaction snapshot directory.")
    samples_root = output_directory / "samples"
    if samples_root != render_attempt.trajectories[0].directory.parent:
        raise ValueError("BLCS render attempt is outside the canonical samples root.")
    rendered_by_id = {
        rendered.trajectory_id: rendered for rendered in render_attempt.trajectories
    }
    if set(rendered_by_id) != {plan.source.trajectory_id for plan in plan_tuple}:
        raise ValueError(
            "BLCS render attempt differs from the resolved plan inventory."
        )
    initial_render_bytes = directory_size_bytes(samples_root)
    all_records: list[BLCSSampleRecord] = []
    trajectory_entries: list[dict[str, object]] = []
    global_expected: list[int] = []
    rendered_visible_object_views = 0
    for plan in plan_tuple:
        rendered = rendered_by_id[plan.source.trajectory_id]
        trajectory_root = rendered.directory
        if set(path.name for path in trajectory_root.iterdir()) != {
            "backgrounds",
            "chunks",
        }:
            raise ValueError("BLCS trajectory contains non-canonical render files.")
        plan_json = trajectory_root / "plan.json"
        plan_npz = trajectory_root / "plan.npz"
        _write_json(plan_json, plan.to_dict())
        np.savez(
            plan_npz,
            positions_court_m=plan.source.positions_court_m,
            velocities_court_mps=plan.source.velocities_court_mps,
            present=plan.source.present,
            positions_scene=plan.positions_scene,
            camera_uv=plan.camera_uv,
            camera_depth=plan.camera_depth,
            geometric_visible=plan.geometric_visible,
            court_uv=plan.court_uv,
            court_visible=plan.court_visible,
        )
        camera_ids = tuple(
            camera.scene_camera.camera_id for camera in plan.camera_rig.cameras
        )
        expected_chunk_ids = tuple(
            f"chunk-{chunk.chunk_index:06d}" for chunk in plan.chunks
        )
        if (
            tuple(reader.directory.name for reader in rendered.chunk_readers)
            != expected_chunk_ids
        ):
            raise ValueError("BLCS rendered chunk inventory differs from the plan.")
        validated_chunks = FinalDatasetAssembler(
            frame_count=plan.source.frame_count,
            camera_ids=camera_ids,
            attempt_token=render_attempt.attempt_token,
        ).validate(rendered.chunk_readers)
        background_reference = rendered.background_directory.relative_to(
            output_directory
        ).as_posix()
        for plan_chunk, reader, validated in zip(
            plan.chunks,
            rendered.chunk_readers,
            validated_chunks,
            strict=True,
        ):
            expected_keys = tuple(
                RenderSampleKey(frame_index, camera_id)
                for frame_index in plan_chunk.frame_indices
                for camera_id in camera_ids
            )
            if validated.keys != expected_keys:
                raise ValueError(
                    "BLCS compact chunk has incomplete frame-camera coverage."
                )
            deltas = reader.deltas()
            chunk_metadata = reader.metadata()
            for key, delta, metadata in zip(
                validated.keys, deltas, chunk_metadata, strict=True
            ):
                camera_index = camera_ids.index(key.camera_id)
                expected_metadata = build_blcs_sample_metadata(
                    plan=plan,
                    source_frame_index=key.frame_index,
                    camera_index=camera_index,
                    chunk_index=plan_chunk.chunk_index,
                    delta=delta,
                )
                if metadata != expected_metadata:
                    raise ValueError(
                        "BLCS retained chunk metadata differs from its semantic plan."
                    )
            chunk_reference = validated.directory.relative_to(
                output_directory
            ).as_posix()
            for chunk_sample_index, key in enumerate(validated.keys):
                all_records.append(
                    BLCSSampleRecord(
                        trajectory_id=plan.source.trajectory_id,
                        split=plan.source.split,
                        global_frame_index=plan.global_frame_offset + key.frame_index,
                        source_frame_index=key.frame_index,
                        chunk_index=plan_chunk.chunk_index,
                        camera_id=key.camera_id,
                        background_store=background_reference,
                        foreground_chunk=chunk_reference,
                        chunk_sample_index=chunk_sample_index,
                    )
                )
        expected_source = tuple(range(plan.source.frame_count))
        actual_source = tuple(
            dict.fromkeys(
                record.source_frame_index
                for record in all_records
                if record.trajectory_id == plan.source.trajectory_id
            )
        )
        if actual_source != expected_source:
            raise ValueError("BLCS trajectory render/label inventory is incomplete.")
        trajectory_entries.append(
            {
                "trajectory_id": plan.source.trajectory_id,
                "split": plan.source.split,
                "source_frame_count": plan.source.frame_count,
                "global_frame_offset": plan.global_frame_offset,
                "frame_inventory": {
                    "source": plan.source.frame_count,
                    "planned": plan.source.frame_count,
                    "rendered": plan.source.frame_count,
                    "labelled": plan.source.frame_count,
                    "first_frame": 0,
                    "last_frame": plan.source.frame_count - 1,
                },
                "target_court": plan.target_court.court_instance_id,
                "candidate_id": plan.target_court.candidate_id,
                "transform": plan.target_court.scene_from_court.to_list(),
                "camera_profile": plan.camera_rig.profile,
                "camera_seed": plan.camera_rig.seed,
                "camera_ids": list(camera_ids),
                "attempt_token": render_attempt.attempt_token,
                "chunk_count": len(plan.chunks),
                "chunk_directories": [
                    reader.directory.relative_to(output_directory).as_posix()
                    for reader in rendered.chunk_readers
                ],
                "background_store": background_reference,
                "plan_json": plan_json.relative_to(output_directory).as_posix(),
                "plan_npz": plan_npz.relative_to(output_directory).as_posix(),
            }
        )
        rendered_visible_object_views += rendered.rendered_visible_object_views
        global_expected.extend(plan.global_frame_indices)
    expected_global = list(range(sum(plan.source.frame_count for plan in plan_tuple)))
    if global_expected != expected_global:
        raise ValueError("BLCS plan global frames are not exact contiguous 0..T-1.")
    continuity = _continuity(plan_tuple, all_records)
    inventory = FrameInventory(
        source_count=len(expected_global),
        planned_indices=tuple(expected_global),
        rendered_indices=tuple(expected_global),
        labelled_indices=tuple(expected_global),
    )
    diagnostic_paths = write_blcs_diagnostics(
        output_directory,
        plans=plan_tuple,
        inventory=inventory,
        continuity=continuity,
        records=all_records,
        rendered_visible_object_views=rendered_visible_object_views,
    )
    performance_reference = "diagnostics/performance.json"
    diagnostics = (*diagnostic_paths, performance_reference)
    court_ids = tuple(
        dict.fromkeys(plan.target_court.court_instance_id for plan in plan_tuple)
    )
    bindings = tuple(
        next(
            plan.target_court
            for plan in plan_tuple
            if plan.target_court.court_instance_id == court_id
        )
        for court_id in court_ids
    )
    metadata = {
        "trajectory_count": len(plan_tuple),
        "sample_count": len(all_records),
        "camera_profile": sorted({plan.camera_rig.profile for plan in plan_tuple}),
        "selection_seed": sorted(
            {plan.target_court.selection_seed for plan in plan_tuple}
        ),
        "metric_scene_adapter": metric_adapter.to_dict(),
    }
    manifest = DatasetManifest(
        scene_id=plan_tuple[0].dataset_scene_id,
        domain=DatasetDomain.BLCS,
        schema=BLCS_DATASET_SCHEMA,
        frame_inventory=inventory,
        target_courts=bindings,
        metadata=metadata,
        diagnostics=diagnostics,
    )
    _write_json(
        output_directory / "dataset.json",
        {
            "schema": manifest.schema,
            "scene_id": manifest.scene_id,
            "domain": manifest.domain.value,
            "frame_inventory": manifest.frame_inventory.to_dict(),
            "target_courts": [binding.to_dict() for binding in manifest.target_courts],
            "metadata": dict(manifest.metadata),
            "diagnostics": list(manifest.diagnostics),
            "performance": performance_reference,
            "trajectories": trajectory_entries,
            "samples": [record.to_dict() for record in all_records],
        },
    )
    performance = _write_performance(
        output_directory,
        timer=performance_timer,
        render_attempt=render_attempt,
        initial_render_bytes=initial_render_bytes,
        frame_count=len(expected_global),
        camera_count=len(plan_tuple[0].camera_rig.cameras),
        sample_count=len(all_records),
    )
    if render_attempt.nht_invocations != len(plan_tuple):
        raise ValueError("BLCS performance evidence lacks one NHT call per trajectory.")
    expected_misses = sum(len(plan.camera_rig.cameras) for plan in plan_tuple)
    if render_attempt.background_cache_misses != expected_misses:
        raise ValueError("BLCS performance evidence lacks one cache miss per camera.")
    if performance_budget.require_cuda and render_attempt.cuda_peak_bytes <= 0:
        raise ValueError("BLCS CUDA performance evidence lacks a peak allocation.")
    if performance.generated_bytes < performance.published_bytes:
        raise ValueError("BLCS generated-byte evidence is smaller than publication.")
    performance.validate_budget(performance_budget)
    return BLCSAssemblyResult(
        manifest=manifest,
        continuity=continuity,
        sample_records=tuple(all_records),
        metric_adapter=metric_adapter,
        performance=performance,
    )


def validate_blcs_dataset(output_directory: Path) -> BLCSAssemblyResult:
    """Perform the sole final streaming validation pass over compact chunks."""
    is_owner = (
        output_directory.name == "blcs" and output_directory.parent.name == "datasets"
    )
    is_transaction = (
        output_directory.name == "snapshot"
        and output_directory.parent.name == "blcs_dataset"
        and output_directory.parent.parent.name == ".transactions"
    )
    if (
        not (is_owner or is_transaction)
        or output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise ValueError(
            "BLCS dataset must be its canonical owner or transaction snapshot."
        )
    if set(path.name for path in output_directory.iterdir()) != {
        "dataset.json",
        "samples",
        "diagnostics",
    }:
        raise ValueError(
            "BLCS dataset contains stale or non-canonical top-level paths."
        )
    payload = _mapping(
        _load_json(output_directory / "dataset.json"),
        keys=_DATASET_KEYS,
        name="BLCS dataset",
    )
    if payload["schema"] != BLCS_DATASET_SCHEMA or payload["domain"] != "blcs":
        raise ValueError("Unsupported canonical compact BLCS schema/domain.")
    scene_id = _string(payload["scene_id"], name="scene_id")
    trajectories = tuple(
        _mapping(value, keys=_TRAJECTORY_KEYS, name="BLCS trajectory")
        for value in _list(payload["trajectories"], name="trajectories")
    )
    if not trajectories:
        raise ValueError("BLCS dataset must contain at least one trajectory.")
    trajectory_ids = tuple(
        _string(value["trajectory_id"], name="trajectory_id") for value in trajectories
    )
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("BLCS dataset contains duplicate trajectory IDs.")
    samples_root = output_directory / "samples"
    if set(path.name for path in samples_root.iterdir()) != set(trajectory_ids):
        raise ValueError("BLCS samples root differs from trajectory inventory.")
    records = tuple(
        _record(value) for value in _list(payload["samples"], name="samples")
    )
    by_trajectory: dict[str, list[BLCSSampleRecord]] = defaultdict(list)
    for record in records:
        by_trajectory[record.trajectory_id].append(record)
    binding_payloads = tuple(
        _mapping(
            value,
            keys={
                "court_instance_id",
                "candidate_id",
                "scene_from_court",
                "selection_seed",
            },
            name="target court",
        )
        for value in _list(payload["target_courts"], name="target_courts")
    )
    binding_by_id = {
        _string(value["court_instance_id"], name="court_instance_id"): value
        for value in binding_payloads
    }
    if len(binding_by_id) != len(binding_payloads):
        raise ValueError("BLCS target-court inventory contains duplicates.")
    expected_global: list[int] = []
    continuity_records: list[TimelineFrameRecord] = []
    continuity_reports: list[FrameContinuityReport] = []
    for trajectory in trajectories:
        trajectory_id = _string(trajectory["trajectory_id"], name="trajectory_id")
        frame_count = _integer(
            trajectory["source_frame_count"], name="source_frame_count"
        )
        offset = _integer(trajectory["global_frame_offset"], name="global_frame_offset")
        camera_ids = tuple(
            _string(value, name="camera_id")
            for value in _list(trajectory["camera_ids"], name="camera_ids")
        )
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("BLCS camera IDs must be non-empty and unique.")
        expected_global.extend(range(offset, offset + frame_count))
        _validate_trajectory_inventory(trajectory, frame_count=frame_count)
        target_id = _string(trajectory["target_court"], name="target_court")
        if target_id not in binding_by_id:
            raise ValueError("BLCS trajectory references an undeclared target court.")
        binding = binding_by_id[target_id]
        if (
            trajectory["candidate_id"] != binding["candidate_id"]
            or trajectory["transform"] != binding["scene_from_court"]
        ):
            raise ValueError("BLCS trajectory target-court metadata is inconsistent.")
        trajectory_root = samples_root / trajectory_id
        if set(path.name for path in trajectory_root.iterdir()) != {
            "backgrounds",
            "chunks",
            "plan.json",
            "plan.npz",
        }:
            raise ValueError(
                "BLCS trajectory contains a legacy or stale publication path."
            )
        plan_path = _contained_file(
            output_directory, _string(trajectory["plan_json"], name="plan_json")
        )
        plan_npz_path = _contained_file(
            output_directory, _string(trajectory["plan_npz"], name="plan_npz")
        )
        plan = _mapping(_load_json(plan_path), keys=_PLAN_KEYS, name="BLCS plan")
        _validate_plan_header(
            plan,
            trajectory=trajectory,
            binding=binding,
            frame_count=frame_count,
            offset=offset,
            camera_ids=camera_ids,
        )
        track_mappings, track_ids, source_trajectory_ids = _plan_tracks(
            plan["tracks"], frame_count=frame_count
        )
        with np.load(plan_npz_path, allow_pickle=False) as archive:
            plan_arrays = _validated_plan_arrays(
                archive,
                frame_count=frame_count,
                camera_count=len(camera_ids),
                transform=RigidTransform(
                    _float_tuple(binding["scene_from_court"], name="scene_from_court")
                ),
                track_mappings=track_mappings,
            )
        _validate_plan_composition(
            plan["composition"],
            scene_id=scene_id,
            trajectory_id=trajectory_id,
            track_ids=track_ids,
            track_mappings=track_mappings,
            present=plan_arrays["present"],
            positions_scene=plan_arrays["positions_scene"],
            scene_from_court=RigidTransform(
                _float_tuple(binding["scene_from_court"], name="scene_from_court")
            ),
        )
        background_path = _contained_directory(
            output_directory,
            _string(trajectory["background_store"], name="background_store"),
        )
        store = SharedBackgroundStore(background_path)
        if store.camera_ids != camera_ids:
            raise ValueError("BLCS shared backgrounds differ from camera inventory.")
        raw_chunk_paths = tuple(
            _string(value, name="chunk_directory")
            for value in _list(
                trajectory["chunk_directories"], name="chunk_directories"
            )
        )
        chunk_paths = tuple(
            _contained_directory(output_directory, value) for value in raw_chunk_paths
        )
        if len(chunk_paths) != _integer(trajectory["chunk_count"], name="chunk_count"):
            raise ValueError("BLCS chunk count differs from its directory inventory.")
        attempt_token = _string(trajectory["attempt_token"], name="attempt_token")
        chunks_root = trajectory_root / "chunks"
        if set(chunks_root.iterdir()) != set(chunk_paths):
            raise ValueError(
                "BLCS chunks directory contains stale or unexpected paths."
            )
        chunk_readers = tuple(ChunkReader(path) for path in chunk_paths)
        validated_chunks = FinalDatasetAssembler(
            frame_count=frame_count,
            camera_ids=camera_ids,
            attempt_token=attempt_token,
        ).validate(chunk_readers)
        planned_chunks = _list(plan["chunks"], name="plan.chunks")
        for chunk_index, validated in enumerate(validated_chunks):
            planned_chunk = _mapping(
                planned_chunks[chunk_index],
                keys={"chunk_index", "frame_indices"},
                name="plan chunk",
            )
            expected_chunk_keys = tuple(
                RenderSampleKey(_integer(frame_index, name="frame_index"), camera_id)
                for frame_index in _list(
                    planned_chunk["frame_indices"], name="frame_indices"
                )
                for camera_id in camera_ids
            )
            if validated.keys != expected_chunk_keys:
                raise ValueError(
                    "BLCS compact chunk boundaries differ from the resolved plan."
                )
        trajectory_records = by_trajectory.get(trajectory_id, [])
        expected_records: list[BLCSSampleRecord] = []
        trajectory_continuity_records: list[TimelineFrameRecord] = []
        camera_index = {camera_id: index for index, camera_id in enumerate(camera_ids)}
        for chunk_index, (chunk_path, reader, validated) in enumerate(
            zip(chunk_paths, chunk_readers, validated_chunks, strict=True)
        ):
            deltas = reader.deltas()
            chunk_metadata = reader.metadata()
            for sample_index, (key, delta, sample_metadata) in enumerate(
                zip(validated.keys, deltas, chunk_metadata, strict=True)
            ):
                record = BLCSSampleRecord(
                    trajectory_id=trajectory_id,
                    split=_string(trajectory["split"], name="split"),
                    global_frame_index=offset + key.frame_index,
                    source_frame_index=key.frame_index,
                    chunk_index=chunk_index,
                    camera_id=key.camera_id,
                    background_store=_relative(output_directory, background_path),
                    foreground_chunk=_relative(output_directory, chunk_path),
                    chunk_sample_index=sample_index,
                )
                expected_records.append(record)
                _validate_sample_metadata(
                    sample_metadata,
                    scene_id=scene_id,
                    record=record,
                    trajectory=trajectory,
                    binding=binding,
                    plan=plan,
                    plan_arrays=plan_arrays,
                    camera_index=camera_index[key.camera_id],
                    track_ids=track_ids,
                    source_trajectory_ids=source_trajectory_ids,
                    track_mappings=track_mappings,
                    rendered_instance_ids=set(
                        int(value) for value in np.unique(delta.instance_ids)
                    ),
                )
                for object_index, object_id in enumerate(track_ids):
                    source_frame = track_mappings[object_index][key.frame_index]
                    trajectory_continuity_records.append(
                        TimelineFrameRecord(
                            frame_index=record.source_frame_index,
                            chunk_index=record.chunk_index,
                            track_id=f"{trajectory_id}-{object_id}",
                            present=source_frame is not None,
                            source_frame_index=source_frame,
                            camera_id=record.camera_id,
                            label_id=(
                                f"{trajectory_id}-{key.frame_index:06d}-"
                                f"{record.camera_id}-{object_index + 1:03d}"
                            ),
                            court_instance_id=target_id,
                        )
                    )
        if trajectory_records != expected_records:
            raise ValueError("BLCS dataset sample records differ from compact chunks.")
        continuity_reports.append(
            validate_frame_continuity(
                trajectory_continuity_records,
                frame_count=frame_count,
            )
        )
        continuity_records.extend(trajectory_continuity_records)
    source_count = sum(
        _integer(value["source_frame_count"], name="source_frame_count")
        for value in trajectories
    )
    expected_indices = list(range(source_count))
    if expected_global != expected_indices:
        raise ValueError("BLCS global frame offsets are not contiguous 0..T-1.")
    inventory = FrameInventory(
        source_count=source_count,
        planned_indices=tuple(expected_indices),
        rendered_indices=tuple(expected_indices),
        labelled_indices=tuple(expected_indices),
    )
    if payload["frame_inventory"] != inventory.to_dict():
        raise ValueError("BLCS dataset frame inventory is inconsistent.")
    continuity = _aggregate_continuity_reports(
        continuity_reports,
        records=continuity_records,
    )
    diagnostics = tuple(
        _string(value, name="diagnostic")
        for value in _list(payload["diagnostics"], name="diagnostics")
    )
    for diagnostic in diagnostics:
        _contained_file(output_directory, diagnostic)
    performance_path = _string(payload["performance"], name="performance")
    if performance_path not in diagnostics:
        raise ValueError("BLCS performance evidence is not in diagnostics inventory.")
    performance = DatasetPerformanceMetrics.from_dict(
        _load_json(_contained_file(output_directory, performance_path))
    )
    if (
        performance.domain != "blcs"
        or performance.frame_count != source_count
        or performance.sample_count != len(records)
        or performance.published_bytes != directory_size_bytes(output_directory)
    ):
        raise ValueError("BLCS performance evidence differs from final publication.")
    metadata = _mapping(
        payload["metadata"],
        keys={
            "trajectory_count",
            "sample_count",
            "camera_profile",
            "selection_seed",
            "metric_scene_adapter",
        },
        name="BLCS dataset metadata",
    )
    if metadata["trajectory_count"] != len(trajectories) or metadata[
        "sample_count"
    ] != len(records):
        raise ValueError("BLCS dataset metadata counts are inconsistent.")
    metric_adapter = MetricSceneAdapter.from_dict(metadata["metric_scene_adapter"])
    bindings = tuple(
        TargetCourtBinding(
            court_instance_id=court_id,
            candidate_id=_string(value["candidate_id"], name="candidate_id"),
            scene_from_court=RigidTransform(
                _float_tuple(value["scene_from_court"], name="scene_from_court")
            ),
            selection_seed=_integer(value["selection_seed"], name="selection_seed"),
        )
        for court_id, value in binding_by_id.items()
    )
    manifest = DatasetManifest(
        scene_id=scene_id,
        domain=DatasetDomain.BLCS,
        schema=BLCS_DATASET_SCHEMA,
        frame_inventory=inventory,
        target_courts=bindings,
        metadata=metadata,
        diagnostics=diagnostics,
    )
    _validate_court_balance(trajectories, court_ids=set(binding_by_id))
    return BLCSAssemblyResult(
        manifest=manifest,
        continuity=continuity,
        sample_records=records,
        metric_adapter=metric_adapter,
        performance=performance,
    )


def validate_blcs_dataset_envelope(
    output_directory: Path,
) -> DatasetPerformanceMetrics:
    """Validate the post-assembly envelope without rescanning retained chunks."""
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("BLCS dataset must be an ordinary directory.")
    if set(path.name for path in output_directory.iterdir()) != {
        "dataset.json",
        "samples",
        "diagnostics",
    }:
        raise ValueError(
            "BLCS dataset contains stale or non-canonical top-level paths."
        )
    payload = _mapping(
        _load_json(output_directory / "dataset.json"),
        keys=_DATASET_KEYS,
        name="BLCS dataset",
    )
    if payload["schema"] != BLCS_DATASET_SCHEMA or payload["domain"] != "blcs":
        raise ValueError("Unsupported canonical compact BLCS schema/domain.")
    _string(payload["scene_id"], name="scene_id")
    diagnostics = tuple(
        _string(value, name="diagnostic")
        for value in _list(payload["diagnostics"], name="diagnostics")
    )
    for diagnostic in diagnostics:
        _contained_file(output_directory, diagnostic)
    performance_reference = _string(payload["performance"], name="performance")
    if performance_reference not in diagnostics:
        raise ValueError("BLCS performance evidence is not in diagnostics inventory.")
    performance = DatasetPerformanceMetrics.from_dict(
        _load_json(_contained_file(output_directory, performance_reference))
    )
    if performance.published_bytes != directory_size_bytes(output_directory):
        raise ValueError("BLCS performance bytes differ from final publication.")
    if performance.sample_count != len(_list(payload["samples"], name="samples")):
        raise ValueError("BLCS performance sample count differs from dataset.json.")
    return performance


def _write_performance(
    output_directory: Path,
    *,
    timer: PerformanceTimer,
    render_attempt: BLCSRenderAttempt,
    initial_render_bytes: int,
    frame_count: int,
    camera_count: int,
    sample_count: int,
) -> DatasetPerformanceMetrics:
    path = output_directory / "diagnostics" / "performance.json"
    wall_seconds, cpu_seconds, peak_rss_bytes = timer.elapsed()
    published_bytes = 0
    result: DatasetPerformanceMetrics | None = None
    for _ in range(8):
        result = DatasetPerformanceMetrics(
            domain="blcs",
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
            execution_device=render_attempt.execution_device,
            cuda_peak_bytes=render_attempt.cuda_peak_bytes,
            nht_invocations=render_attempt.nht_invocations,
            background_cache_misses=render_attempt.background_cache_misses,
            complete_array_scans=sample_count,
            generated_bytes=(
                render_attempt.generated_bytes
                + max(0, published_bytes - initial_render_bytes)
            ),
            published_bytes=published_bytes,
            dense_reference_bytes=MEASURED_DENSE_REFERENCE_BYTES,
            frame_count=frame_count,
            camera_count=camera_count,
            sample_count=sample_count,
        )
        _write_json(path, result.to_dict())
        measured = directory_size_bytes(output_directory)
        if measured == published_bytes:
            return result
        published_bytes = measured
    raise RuntimeError("BLCS performance byte accounting did not converge.")


def _continuity(
    plans: Sequence[BLCSTrajectoryPlan],
    records: Sequence[BLCSSampleRecord],
) -> FrameContinuityReport:
    plans_by_id = {plan.source.trajectory_id: plan for plan in plans}
    records_by_trajectory: dict[str, list[BLCSSampleRecord]] = defaultdict(list)
    for record in records:
        if record.trajectory_id not in plans_by_id:
            raise ValueError("BLCS sample record references an unknown trajectory.")
        records_by_trajectory[record.trajectory_id].append(record)

    continuity_records: list[TimelineFrameRecord] = []
    continuity_reports: list[FrameContinuityReport] = []
    for plan in plans:
        trajectory_records: list[TimelineFrameRecord] = []
        for record in records_by_trajectory[plan.source.trajectory_id]:
            for object_index, track in enumerate(plan.source.tracks):
                source_frame = track.source_frame_indices[record.source_frame_index]
                trajectory_records.append(
                    TimelineFrameRecord(
                        frame_index=record.source_frame_index,
                        chunk_index=record.chunk_index,
                        track_id=f"{record.trajectory_id}-{track.object_id}",
                        present=source_frame is not None,
                        source_frame_index=source_frame,
                        camera_id=record.camera_id,
                        label_id=(
                            f"{record.trajectory_id}-{record.source_frame_index:06d}-"
                            f"{record.camera_id}-{object_index + 1:03d}"
                        ),
                        court_instance_id=plan.target_court.court_instance_id,
                    )
                )
        continuity_reports.append(
            validate_frame_continuity(
                trajectory_records,
                frame_count=plan.source.frame_count,
            )
        )
        continuity_records.extend(trajectory_records)
    return _aggregate_continuity_reports(
        continuity_reports,
        records=continuity_records,
    )


def _aggregate_continuity_reports(
    reports: Sequence[FrameContinuityReport],
    *,
    records: Sequence[TimelineFrameRecord],
) -> FrameContinuityReport:
    if not reports or not records:
        raise ValueError("BLCS continuity aggregation requires validated trajectories.")
    label_ids = [record.label_id for record in records]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("label_id values must be globally unique.")
    return FrameContinuityReport(
        frame_count=sum(report.frame_count for report in reports),
        chunk_count=len({record.chunk_index for record in records}),
        track_count=len({record.track_id for record in records}),
        camera_count=len({record.camera_id for record in records}),
        record_count=sum(report.record_count for report in reports),
    )


def _validate_trajectory_inventory(
    trajectory: Mapping[str, object], *, frame_count: int
) -> None:
    inventory = _mapping(
        trajectory["frame_inventory"],
        keys={
            "source",
            "planned",
            "rendered",
            "labelled",
            "first_frame",
            "last_frame",
        },
        name="trajectory frame_inventory",
    )
    expected = {
        "source": frame_count,
        "planned": frame_count,
        "rendered": frame_count,
        "labelled": frame_count,
        "first_frame": 0,
        "last_frame": frame_count - 1,
    }
    if inventory != expected:
        raise ValueError("BLCS trajectory frame counts are not exactly equal.")


def _validate_plan_header(
    plan: Mapping[str, object],
    *,
    trajectory: Mapping[str, object],
    binding: Mapping[str, object],
    frame_count: int,
    offset: int,
    camera_ids: tuple[str, ...],
) -> None:
    if (
        plan["trajectory_id"] != trajectory["trajectory_id"]
        or plan["split"] != trajectory["split"]
        or plan["source_frame_count"] != frame_count
        or plan["global_frame_offset"] != offset
        or plan["target_court"] != binding
        or plan["camera_profile"] != trajectory["camera_profile"]
        or plan["camera_seed"] != trajectory["camera_seed"]
    ):
        raise ValueError("BLCS plan.json disagrees with dataset.json.")
    if tuple(_list(plan["global_frame_indices"], name="global_frame_indices")) != tuple(
        range(offset, offset + frame_count)
    ):
        raise ValueError("BLCS plan global frame inventory is incomplete.")
    cameras = _list(plan["cameras"], name="plan.cameras")
    planned_camera_ids = tuple(
        _string(
            _mapping(
                _mapping(
                    value,
                    keys={
                        "slot_id",
                        "court_local_center_m",
                        "court_local_look_at_m",
                        "hfov_degrees",
                        "camera",
                    },
                    name="planned camera",
                )["camera"],
                keys={
                    "camera_id",
                    "source_frame_index",
                    "width",
                    "height",
                    "intrinsics",
                    "camera_to_scene",
                    "image_path",
                },
                name="scene camera",
            )["camera_id"],
            name="camera_id",
        )
        for value in cameras
    )
    if planned_camera_ids != camera_ids:
        raise ValueError("BLCS plan camera inventory is inconsistent.")
    chunks = tuple(
        _mapping(
            value,
            keys={"chunk_index", "frame_indices"},
            name="plan chunk",
        )
        for value in _list(plan["chunks"], name="plan.chunks")
    )
    if len(chunks) != trajectory["chunk_count"]:
        raise ValueError("BLCS plan chunk count is inconsistent.")
    flattened = tuple(
        _integer(frame, name="frame_index")
        for chunk_index, chunk in enumerate(chunks)
        for frame in (
            _list(chunk["frame_indices"], name="frame_indices")
            if chunk["chunk_index"] == chunk_index
            else _raise_chunk_index()
        )
    )
    if flattened != tuple(range(frame_count)):
        raise ValueError(
            "BLCS plan chunks do not cover every source frame exactly once."
        )


def _raise_chunk_index() -> list[object]:
    raise ValueError("BLCS plan chunk indices are not contiguous.")


def _plan_tracks(
    value: object, *, frame_count: int
) -> tuple[tuple[tuple[int | None, ...], ...], tuple[str, ...], tuple[str, ...]]:
    mappings: list[tuple[int | None, ...]] = []
    object_ids: list[str] = []
    source_ids: list[str] = []
    for item in _list(value, name="plan.tracks"):
        track = _mapping(
            item,
            keys={"object_id", "source_trajectory_id", "source_frame_indices"},
            name="planned track",
        )
        object_ids.append(_string(track["object_id"], name="object_id"))
        source_ids.append(
            _string(track["source_trajectory_id"], name="source_trajectory_id")
        )
        mapping = tuple(
            None if frame is None else _integer(frame, name="source_frame")
            for frame in _list(
                track["source_frame_indices"], name="source_frame_indices"
            )
        )
        if len(mapping) != frame_count:
            raise ValueError("BLCS track mapping does not cover every source frame.")
        mappings.append(mapping)
    if not mappings or len(object_ids) != len(set(object_ids)):
        raise ValueError("BLCS plan tracks must be non-empty and unique.")
    return tuple(mappings), tuple(object_ids), tuple(source_ids)


def _validated_plan_arrays(
    archive: np.lib.npyio.NpzFile,
    *,
    frame_count: int,
    camera_count: int,
    transform: RigidTransform,
    track_mappings: tuple[tuple[int | None, ...], ...],
) -> dict[str, NDArray[np.generic]]:
    expected_keys = {
        "positions_court_m",
        "velocities_court_mps",
        "present",
        "positions_scene",
        "camera_uv",
        "camera_depth",
        "geometric_visible",
        "court_uv",
        "court_visible",
    }
    if set(archive.files) != expected_keys:
        raise ValueError("BLCS plan array inventory is invalid.")
    result = {name: np.asarray(archive[name]) for name in archive.files}
    positions = result["positions_court_m"]
    if (
        positions.dtype != np.float64
        or positions.ndim != 3
        or positions.shape[0] != frame_count
        or positions.shape[-1] != 3
    ):
        raise ValueError("BLCS plan positions have an invalid contract.")
    object_count = positions.shape[1]
    expected = {
        "velocities_court_mps": (positions.shape, np.dtype(np.float64)),
        "present": ((frame_count, object_count), np.dtype(np.bool_)),
        "positions_scene": (positions.shape, np.dtype(np.float64)),
        "camera_uv": (
            (frame_count, camera_count, object_count, 2),
            np.dtype(np.float64),
        ),
        "camera_depth": (
            (frame_count, camera_count, object_count),
            np.dtype(np.float64),
        ),
        "geometric_visible": (
            (frame_count, camera_count, object_count),
            np.dtype(np.bool_),
        ),
        "court_uv": ((camera_count, 20, 2), np.dtype(np.float64)),
        "court_visible": ((camera_count, 20), np.dtype(np.bool_)),
    }
    for name, (shape, dtype) in expected.items():
        if result[name].shape != shape or result[name].dtype != dtype:
            raise ValueError(f"BLCS plan {name} has an invalid contract.")
    for name, value in result.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"BLCS plan {name} contains non-finite values.")
    mapped_presence = np.asarray(
        [
            [mapping[frame_index] is not None for mapping in track_mappings]
            for frame_index in range(frame_count)
        ],
        dtype=np.bool_,
    )
    if not np.array_equal(mapped_presence, result["present"]):
        raise ValueError("BLCS source mappings disagree with presence.")
    if not np.allclose(
        transform.apply(positions), result["positions_scene"], atol=1.0e-8, rtol=0.0
    ):
        raise ValueError("BLCS trajectory and camera use different court transforms.")
    return result


def _validate_plan_composition(
    value: object,
    *,
    scene_id: str,
    trajectory_id: str,
    track_ids: tuple[str, ...],
    track_mappings: tuple[tuple[int | None, ...], ...],
    present: NDArray[np.generic],
    positions_scene: NDArray[np.generic],
    scene_from_court: RigidTransform,
) -> None:
    """Bind the published semantic plan to the exact rigid ball trajectory."""
    composition = GaussianForegroundComposition.from_dict(value)
    if (
        composition.scene_id != scene_id
        or composition.composition_id != f"blcs-{trajectory_id}"
    ):
        raise ValueError("BLCS Gaussian composition identity is inconsistent.")
    if len(composition.assets) != 1 or composition.assets[0].asset_class != "ball":
        raise ValueError("BLCS composition must contain exactly one ball Gaussian asset.")
    if tuple(item.object_id for item in composition.objects) != track_ids:
        raise ValueError("BLCS Gaussian objects differ from the trajectory tracks.")
    if tuple(item.instance_id for item in composition.objects) != tuple(
        range(1, len(track_ids) + 1)
    ):
        raise ValueError("BLCS Gaussian instance IDs must equal the track order.")
    if any(
        item.deformation_kind is not GaussianDeformationKind.RIGID
        for item in composition.objects
    ):
        raise ValueError("BLCS ball Gaussian objects must use rigid deformation.")
    if len(composition.frames) != present.shape[0]:
        raise ValueError("BLCS Gaussian composition frame count is inconsistent.")

    expected_rotation = scene_from_court.matrix()[:3, :3]
    object_index = {object_id: index for index, object_id in enumerate(track_ids)}
    for frame in composition.frames:
        instances = {instance.object_id: instance for instance in frame.instances}
        expected_ids = {
            track_ids[index]
            for index in range(len(track_ids))
            if bool(present[frame.frame_index, index])
        }
        if set(instances) != expected_ids:
            raise ValueError("BLCS Gaussian composition presence is inconsistent.")
        for object_id, instance in instances.items():
            index = object_index[object_id]
            if instance.source_frame_index != track_mappings[index][frame.frame_index]:
                raise ValueError("BLCS Gaussian source-frame mapping is inconsistent.")
            transform = instance.scene_from_asset
            if not np.isclose(transform.scale, 1.0, atol=1.0e-12, rtol=0.0):
                raise ValueError("BLCS ball Gaussian assets must retain metric scale.")
            matrix = transform.rigid.matrix()
            if not np.allclose(
                matrix[:3, :3], expected_rotation, atol=1.0e-8, rtol=0.0
            ) or not np.allclose(
                matrix[:3, 3],
                positions_scene[frame.frame_index, index],
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError("BLCS Gaussian rigid placement differs from its trajectory.")


def _validate_sample_metadata(
    value: Mapping[str, object],
    *,
    scene_id: str,
    record: BLCSSampleRecord,
    trajectory: Mapping[str, object],
    binding: Mapping[str, object],
    plan: Mapping[str, object],
    plan_arrays: Mapping[str, NDArray[np.generic]],
    camera_index: int,
    track_ids: tuple[str, ...],
    source_trajectory_ids: tuple[str, ...],
    track_mappings: tuple[tuple[int | None, ...], ...],
    rendered_instance_ids: set[int],
) -> None:
    sample = _mapping(value, keys=_SAMPLE_METADATA_KEYS, name="BLCS sample metadata")
    source_frame = record.source_frame_index
    present = plan_arrays["present"][source_frame]
    geometric = plan_arrays["geometric_visible"][source_frame, camera_index]
    rendered = [index + 1 in rendered_instance_ids for index in range(len(track_ids))]
    objects = [
        {
            "object_id": object_id,
            "instance_id": object_index + 1,
            "present": bool(present[object_index]),
            "source_trajectory": source_trajectory_ids[object_index],
            "source_frame": track_mappings[object_index][source_frame],
            "geometric_visible": bool(geometric[object_index]),
            "rendered_visible": rendered[object_index],
        }
        for object_index, object_id in enumerate(track_ids)
    ]
    cameras = _list(plan["cameras"], name="plan.cameras")
    expected = {
        "schema": BLCS_SAMPLE_SCHEMA,
        "scene_id": scene_id,
        "trajectory_id": record.trajectory_id,
        "split": record.split,
        "global_frame_index": record.global_frame_index,
        "source_frame_index": source_frame,
        "chunk_index": record.chunk_index,
        "source_trajectory": record.trajectory_id,
        "source_frame": source_frame,
        "target_court": trajectory["target_court"],
        "candidate_id": binding["candidate_id"],
        "transform": binding["scene_from_court"],
        "camera_profile": trajectory["camera_profile"],
        "camera_parameters": cameras[camera_index],
        "seed": {
            "court_assignment": binding["selection_seed"],
            "camera_sampling": trajectory["camera_seed"],
        },
        "objects": objects,
        "semantic_arrays": {
            "ball_uv": plan_arrays["camera_uv"][source_frame, camera_index].tolist(),
            "ball_depth": plan_arrays["camera_depth"][
                source_frame, camera_index
            ].tolist(),
            "geometric_visible": geometric.tolist(),
            "rendered_visible": rendered,
            "positions_court_m": plan_arrays["positions_court_m"][
                source_frame
            ].tolist(),
            "positions_scene": plan_arrays["positions_scene"][source_frame].tolist(),
            "velocities_court_mps": plan_arrays["velocities_court_mps"][
                source_frame
            ].tolist(),
            "present": present.tolist(),
            "source_frame_indices": [
                mapping[source_frame] for mapping in track_mappings
            ],
            "instance_ids": list(range(1, len(track_ids) + 1)),
        },
    }
    if sample != expected:
        raise ValueError("BLCS compact sample metadata is inconsistent.")


def _semantic_arrays(
    metadata: Mapping[str, object],
) -> dict[str, NDArray[np.generic]]:
    raw = _mapping(
        metadata.get("semantic_arrays"),
        keys={
            "ball_uv",
            "ball_depth",
            "geometric_visible",
            "rendered_visible",
            "positions_court_m",
            "positions_scene",
            "velocities_court_mps",
            "present",
            "source_frame_indices",
            "instance_ids",
        },
        name="semantic_arrays",
    )
    return {
        "ball_uv": np.asarray(raw["ball_uv"], dtype=np.float32),
        "ball_depth": np.asarray(raw["ball_depth"], dtype=np.float32),
        "geometric_visible": np.asarray(raw["geometric_visible"], dtype=np.bool_),
        "rendered_visible": np.asarray(raw["rendered_visible"], dtype=np.bool_),
        "positions_court_m": np.asarray(raw["positions_court_m"], dtype=np.float32),
        "positions_scene": np.asarray(raw["positions_scene"], dtype=np.float32),
        "velocities_court_mps": np.asarray(
            raw["velocities_court_mps"], dtype=np.float32
        ),
        "present": np.asarray(raw["present"], dtype=np.bool_),
        "source_frame_indices": np.asarray(
            [-1 if value is None else value for value in raw["source_frame_indices"]],
            dtype=np.int64,
        ),
        "instance_ids": np.asarray(raw["instance_ids"], dtype=np.int64),
    }


def _validate_court_balance(
    trajectories: Sequence[Mapping[str, object]], *, court_ids: set[str]
) -> None:
    counts = Counter(
        _string(value["target_court"], name="target_court") for value in trajectories
    )
    values = [counts[court_id] for court_id in court_ids]
    if len(trajectories) >= len(court_ids) and set(counts) != court_ids:
        raise ValueError("BLCS dataset leaves an accepted court unused.")
    if max(values) - min(values) > 1:
        raise ValueError("BLCS dataset court assignment differs by more than one.")
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for value in trajectories:
        by_split[_string(value["split"], name="split")][
            _string(value["target_court"], name="target_court")
        ] += 1
    for split, split_counts in by_split.items():
        split_values = [split_counts[court_id] for court_id in court_ids]
        if max(split_values) - min(split_values) > 1:
            raise ValueError(f"BLCS court assignment is imbalanced in split {split!r}.")


def _record(value: object) -> BLCSSampleRecord:
    raw = _mapping(
        value,
        keys={
            "trajectory_id",
            "split",
            "global_frame_index",
            "source_frame_index",
            "chunk_index",
            "camera_id",
            "background_store",
            "foreground_chunk",
            "chunk_sample_index",
        },
        name="BLCS sample record",
    )
    return BLCSSampleRecord(
        trajectory_id=_string(raw["trajectory_id"], name="trajectory_id"),
        split=_string(raw["split"], name="split"),
        global_frame_index=_integer(
            raw["global_frame_index"], name="global_frame_index"
        ),
        source_frame_index=_integer(
            raw["source_frame_index"], name="source_frame_index"
        ),
        chunk_index=_integer(raw["chunk_index"], name="chunk_index"),
        camera_id=_string(raw["camera_id"], name="camera_id"),
        background_store=_string(raw["background_store"], name="background_store"),
        foreground_chunk=_string(raw["foreground_chunk"], name="foreground_chunk"),
        chunk_sample_index=_integer(
            raw["chunk_sample_index"], name="chunk_sample_index"
        ),
    )


def _contained_file(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=False)


def _contained_directory(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=True)


def _contained(root: Path, relative: str, *, directory: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("BLCS dataset reference is not a relative POSIX path.")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise ValueError("BLCS dataset reference must not be a symbolic link.")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("BLCS dataset reference escapes its root.")
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise ValueError("BLCS dataset reference has the wrong file type.")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Expected an ordinary BLCS JSON file: {path}")

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON value {value!r} is forbidden.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}.")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _mapping(value: object, *, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{name} schema mismatch; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    result: list[float] = []
    for item in _list(value, name=name):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only numeric values.")
        result.append(float(item))
    return tuple(result)


__all__ = [
    "BLCSAllViewTrajectory",
    "BLCSAssemblyResult",
    "BLCSCompactDatasetReader",
    "BLCSLogicalSample",
    "BLCSTrackIndex",
    "BLCSTrajectoryIndex",
    "MEASURED_DENSE_REFERENCE_BYTES",
    "assemble_blcs_dataset",
    "validate_blcs_dataset",
    "validate_blcs_dataset_envelope",
]
