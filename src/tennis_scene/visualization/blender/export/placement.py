"""Export one identity's court placement trace as a Blender input bundle.

The exporter is a read-only consumer of an adopted component store.  It never
runs HMR/GVHMR inference: SMPL vertices and COCO17 joints are reconstructed
deterministically from the stored GVHMR parameters by
:class:`~src.tennis_scene.pipeline.components.body_placement.SmplGeometry`, on
the CPU, and the production placement path
(:func:`~src.tennis_scene.pipeline.components.player_reconstruction.trace_player_placement`)
is re-run so the saved ``body_placement`` artifact is reproduced exactly.

Bundle layout (``tennis_scene_blender_placement_v1``)::

    <output_dir>/
    ├── manifest.json
    ├── smpl_faces.npy
    ├── body_local.pc2         # (F,V,3) u_t; invalid frames are zero
    ├── joints_local.npy       # (F,17,3) J_src
    ├── target_joints.npy      # (F,17,3) triangulated COCO17, NaN invalid
    ├── target_weights.npy     # (F,17)
    ├── hip_position.npy       # (F,3) q_t
    ├── yaw_correction.npy     # (F,) psi_t
    ├── valid.npy              # (F,)
    ├── placed_vertices_t0.npy # (V,3) saved body_placement at t0, court frame
    └── plate/%05d.jpg         # plate camera over the requested frame interval

Blender world coordinates are court coordinates (metres, XY ground, +Z up).
The exported transform is ``world = hip_position + scale * Rz(yaw) @ u_t``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from scipy.spatial.transform import Rotation

from src.submodules.configuration import SmplAssetPaths
from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.components.body_placement import (
    BodyPlacementOutput,
    SmplGeometry,
)
from src.tennis_scene.pipeline.components.camera_alignment import (
    CameraAlignmentOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.components.identity import (
    IDENTITIES_PORT,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    BodyGeometryModel,
    PlacementTrace,
    PlayerSkeleton,
    ReconstructedPlayers,
    trace_player_placement,
)
from src.tennis_scene.pipeline.components.triangulation import (
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    SourceVideo,
)
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyPlacementInputAssembler,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.visualization.blender.export.bundle import (
    PLATE_PATTERN,
    _write_plate,
    court_geometry,
)
from src.tennis_scene.visualization.blender.export.camera import (
    BlenderCameraPose,
    to_blender_camera,
)
from src.tennis_scene.visualization.blender.export.camera import (
    PinholeCamera as BlenderPinholeCamera,
)
from src.tennis_scene.visualization.blender.export.pointcache import write_pc2
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
    smpl_vertices_to_court,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import (
    PinholeCamera as TriangulationPinholeCamera,
)
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.reader import probe_video_info

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.pipeline.components.court_calibration import (
        CourtCalibrationOutput,
    )

PLACEMENT_BUNDLE_SCHEMA = "tennis_scene_blender_placement_v1"
PLACEMENT_TOLERANCE_M = 1e-4
# The reveal exporter's standard horizontal sensor width.  It is recorded in
# the manifest together with the derived lens, so Blender can reproduce the fit
# without guessing.
PLACEMENT_SENSOR_WIDTH_MM = 36.0

# Store assets the deterministic SMPL reconstruction reads.  Paths and
# digests are taken from the adopted artifact, never discovered by glob.
_REQUIRED_BODY_ASSETS = {
    "body_model": "body_model",
    "smpl_hmr2_mean_params": "hmr2_mean_params",
    "smpl_smplx_to_smpl": "smplx_to_smpl",
    "smpl_smpl_coco17_regressor": "smpl_coco17_regressor",
    "smpl_smplx_verts437": "smplx_verts437",
    "smpl_smpl_neutral_joint_regressor": "smpl_neutral_joint_regressor",
}


@dataclass(frozen=True, slots=True)
class PlacementExportConfig:
    """Explicit placement-export choices; nothing here is inferred from data."""

    identity: int
    frames: tuple[int, int]
    t0: int
    plate_camera_id: str
    plate_video_path: Path
    human_vis_threshold: float
    plate_jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if type(self.identity) is not int or self.identity < 0:
            raise ValueError("identity must be a non-negative integer.")
        if (
            not isinstance(self.frames, tuple)
            or len(self.frames) != 2
            or any(type(value) is not int for value in self.frames)
            or not 0 <= self.frames[0] < self.frames[1]
        ):
            raise ValueError("frames must be (start, stop) with 0 <= start < stop.")
        if type(self.t0) is not int or not self.frames[0] <= self.t0 < self.frames[1]:
            raise ValueError("t0 must be an integer inside frames.")
        if not self.plate_camera_id.strip():
            raise ValueError("plate_camera_id must be non-empty.")
        if not isinstance(self.plate_video_path, Path):
            raise TypeError("plate_video_path must be a Path.")
        threshold = self.human_vis_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("human_vis_threshold must be finite and in [0, 1].")
        if type(self.plate_jpeg_quality) is not int or not 1 <= self.plate_jpeg_quality <= 100:
            raise ValueError("plate_jpeg_quality must be in [1, 100].")


class PlacementStore(Protocol):
    """The read-only subset of :class:`StoreReader` the exporter consumes."""

    root: Path
    index_path: Path
    source: Mapping[str, Any]
    source_sha256: str
    camera_ids: tuple[str, ...]
    references: Mapping[str, Mapping[str, Any]]

    def payload(self, node: str) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def stale_dependencies(self, node: str) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class PlacementInputs:
    """Validated store inputs and the production placement configuration."""

    source: ClipSource
    cameras: tuple[TriangulationPinholeCamera, ...]
    grouped: GroupedObservations
    skeleton: PlayerSkeleton
    recovered: GVHMROutput
    placed: ReconstructedPlayers
    placement_config: TemporalPlacementConfig
    joint_regressor: NDArray[np.float32]
    body_assets: SmplAssetPaths
    body_models_dir: Path
    asset_provenance: Mapping[str, Any]
    artifact_provenance: Mapping[str, Any]
    store_provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CpuRuntime:
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class _BodyModelBinding:
    """The subset of ``PeopleModelConfig`` that ``SmplGeometry`` reads."""

    body_models_dir: Path
    smpl_assets: SmplAssetPaths
    runtime: _CpuRuntime


def _read_artifact(
    reader: PlacementStore,
    node: str,
    *,
    schema: str,
    version: int,
    output_type: type[Any],
) -> tuple[Any, dict[str, Any]]:
    """Read one adopted artifact, enforcing its declared schema and lineage."""
    reference = reader.references.get(node)
    if reference is None:
        raise ValueError(f"Component store is missing required artifact {node!r}.")
    if reference["schema"] != schema or int(reference["version"]) != int(version):
        raise ValueError(
            f"Artifact {node!r} is {reference['schema']}:{reference['version']}, "
            f"but this exporter requires {schema}:{version}."
        )
    stale = reader.stale_dependencies(node)
    if stale:
        raise ValueError(
            f"Artifact {node!r} has superseded dependencies: {'; '.join(stale)}"
        )
    payload, descriptor = reader.payload(node)
    location = (reader.root / str(reference["path"])).resolve().parent
    codec = ArtifactCodec(output_type)
    value = codec.load(payload, location, descriptor["arrays"])
    return value, descriptor


def _clip_source(document: Mapping[str, Any]) -> ClipSource:
    videos = tuple(
        SourceVideo(
            camera_id=str(video["camera_id"]),
            path=Path(str(video["path"])),
            sha256=str(video["sha256"]),
            num_frames=int(video["num_frames"]),
            fps=float(video["fps"]),
            width=int(video["width"]),
            height=int(video["height"]),
        )
        for video in document["videos"]
    )
    return ClipSource(str(document["clip_id"]), videos)


def _asset_binding(
    descriptor: Mapping[str, Any],
) -> tuple[SmplAssetPaths, Path, dict[str, Any]]:
    """Bind the saved SMPL assets, verifying every recorded SHA-256."""
    identity = descriptor.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("body_placement artifact lacks an identity block.")
    settings = identity.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("body_placement artifact lacks identity settings.")
    assets = settings.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("body_placement artifact lacks saved model assets.")
    resolved: dict[str, Path] = {}
    provenance: dict[str, Any] = {}
    for store_key, name in _REQUIRED_BODY_ASSETS.items():
        entry = assets.get(store_key)
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            raise ValueError(f"Saved model asset {store_key!r} is missing path/sha256.")
        path = Path(str(entry["path"]))
        digest = str(entry["sha256"])
        if not path.is_file():
            raise FileNotFoundError(f"Saved model asset is missing: {path}")
        actual = dual_sha256(path)
        if actual != digest:
            raise ValueError(
                f"Saved model asset {store_key!r} SHA-256 mismatch: "
                f"recorded {digest}, actual {actual}."
            )
        resolved[name] = path
        provenance[name] = {"path": str(path), "sha256": actual}
    body_model = resolved.pop("body_model")
    if body_model.name != "SMPLX_NEUTRAL.npz" or body_model.parent.name != "smplx":
        raise ValueError(
            "Saved body_model asset must be <body_models>/smplx/SMPLX_NEUTRAL.npz, "
            f"got {body_model}."
        )
    smpl_assets = SmplAssetPaths(
        hmr2_mean_params=resolved["hmr2_mean_params"],
        smplx_to_smpl=resolved["smplx_to_smpl"],
        smpl_coco17_regressor=resolved["smpl_coco17_regressor"],
        smplx_verts437=resolved["smplx_verts437"],
        smpl_neutral_joint_regressor=resolved["smpl_neutral_joint_regressor"],
    )
    return smpl_assets, body_model.parent.parent, provenance


def _placement_config(descriptor: Mapping[str, Any], source: ClipSource) -> TemporalPlacementConfig:
    identity = descriptor.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("body_placement artifact lacks an identity block.")
    settings = identity.get("settings")
    if not isinstance(settings, Mapping) or not isinstance(settings.get("config"), Mapping):
        raise ValueError("body_placement artifact lacks its placement config.")
    saved = TemporalPlacementConfig(**dict(settings["config"]))
    scale = source.pixel_threshold_scale
    return replace(
        saved,
        max_reprojection_rms_px=saved.max_reprojection_rms_px * scale,
        reprojection_weight_sigma_px=saved.reprojection_weight_sigma_px * scale,
    )


def read_placement_inputs(
    store_dir: Path,
    *,
    human_vis_threshold: float,
    joint_regressor: NDArray[np.float32] | None = None,
) -> PlacementInputs:
    """Read an adopted component store and assemble production placement inputs."""
    if not 0.0 <= human_vis_threshold <= 1.0:
        raise ValueError("human_vis_threshold must be in [0, 1].")
    reader = StoreReader(store_dir / "scene.json")
    return _read_placement_inputs(
        reader,
        human_vis_threshold=human_vis_threshold,
        joint_regressor=joint_regressor,
    )


def _read_placement_inputs(
    reader: PlacementStore,
    *,
    human_vis_threshold: float,
    joint_regressor: NDArray[np.float32] | None = None,
) -> PlacementInputs:
    """Assemble inputs from an already opened reader (shared with tests)."""
    source = _clip_source(reader.source)
    if source.camera_ids != tuple(reader.camera_ids):
        raise ValueError("Store source camera order disagrees with its index.")

    calibration: CourtCalibrationOutput
    calibration, _ = _read_artifact(
        reader,
        "court_calibration",
        schema="local_court_calibration",
        version=1,
        output_type=_court_calibration_type(),
    )
    alignment, _ = _read_artifact(
        reader,
        "camera_alignment",
        schema="aligned_cameras",
        version=1,
        output_type=CameraAlignmentOutput,
    )
    skeleton_output, _ = _read_artifact(
        reader,
        "player_triangulation",
        schema="player_skeletons",
        version=1,
        output_type=PlayerTriangulationOutput,
    )
    recovered, _ = _read_artifact(
        reader,
        "gvhmr",
        schema="body_parameters",
        version=1,
        output_type=GVHMROutput,
    )
    placed_output, placed_descriptor = _read_artifact(
        reader,
        "body_placement",
        schema="placed_bodies",
        version=1,
        output_type=BodyPlacementOutput,
    )
    identities, _ = _read_artifact(
        reader,
        "player_association",
        schema=IDENTITIES_PORT.schema,
        version=IDENTITIES_PORT.version,
        output_type=PlayerIdentitiesOutput,
    )
    poses: dict[str, ObjectObservations] = {}
    for camera_id in source.camera_ids:
        node = f"pose_estimation/{camera_id}"
        pose, _ = _read_artifact(
            reader,
            node,
            schema="person_poses",
            version=1,
            output_type=ObjectObservations,
        )
        poses[camera_id] = cast(ObjectObservations, pose)

    geometry = alignment.geometry
    if geometry is None:
        raise ValueError("camera_alignment artifact has no accepted geometry.")
    if skeleton_output.skeleton is None:
        raise ValueError("player_triangulation artifact has no player skeleton.")
    if placed_output.players is None:
        raise ValueError("body_placement artifact has no placed bodies.")

    artifacts: dict[str, Any] = {
        "calibration": calibration,
        "alignment": alignment,
        "skeleton": skeleton_output,
        "recovered": recovered,
        "identities": identities,
        **{f"pose_{camera_id}": poses[camera_id] for camera_id in source.camera_ids},
    }
    assembled = BodyPlacementInputAssembler(human_vis_threshold).assemble(
        AssemblyContext(source), artifacts
    )
    placement_config = _placement_config(placed_descriptor, source)
    assets, body_models_dir, asset_provenance = _asset_binding(placed_descriptor)
    if joint_regressor is None:
        joint_regressor = load_smpl_joint_regressor(
            assets.smpl_neutral_joint_regressor
        )
    artifact_provenance: dict[str, Any] = {}
    for node in (
        "court_calibration",
        "camera_alignment",
        "player_triangulation",
        "gvhmr",
        "body_placement",
        "player_association",
        *(f"pose_estimation/{camera_id}" for camera_id in source.camera_ids),
    ):
        reference = reader.references[node]
        artifact_provenance[node] = {
            "artifact_id": reference["artifact_id"],
            "sha256": reference["sha256"],
            "schema": reference["schema"],
            "version": reference["version"],
            "path": reference["path"],
        }
    store_provenance = {
        "scene_index": str(reader.index_path),
        "index_sha256": dual_sha256(reader.index_path),
        "source_sha256": reader.source_sha256,
        "clip_id": source.clip_id,
        "camera_ids": list(source.camera_ids),
    }
    return PlacementInputs(
        source=source,
        cameras=tuple(geometry.cameras),
        grouped=assembled.grouped,
        skeleton=skeleton_output.skeleton,
        recovered=recovered,
        placed=placed_output.players,
        placement_config=placement_config,
        joint_regressor=np.asarray(joint_regressor, np.float32),
        body_assets=assets,
        body_models_dir=body_models_dir,
        asset_provenance=asset_provenance,
        artifact_provenance=artifact_provenance,
        store_provenance=store_provenance,
    )


def _court_calibration_type() -> type[Any]:
    from src.tennis_scene.pipeline.components.court_calibration import (
        CourtCalibrationOutput,
    )

    return cast(type[Any], CourtCalibrationOutput)


def _select_identity(
    inputs: PlacementInputs, identity: int
) -> tuple[int, Any]:
    identities = np.asarray(inputs.grouped.identities)
    matches = np.flatnonzero(identities == identity)
    if len(matches) != 1:
        raise ValueError(
            f"identity {identity} is not present exactly once in the placement inputs."
        )
    player = int(matches[0])
    bodies = [
        body for body in inputs.recovered.bodies if int(body.person_id) == identity
    ]
    if len(bodies) != 1:
        raise ValueError(
            f"identity {identity} has {len(bodies)} recovered GVHMR bodies; expected 1."
        )
    return player, bodies[0]


def _range_indices(trace: PlacementTrace, frames: tuple[int, int]) -> NDArray[np.int64]:
    start, stop = frames
    requested: NDArray[np.int64] = np.arange(start, stop, dtype=np.int64)
    positions = np.searchsorted(trace.frames, requested)
    if (positions >= len(trace.frames)).any() or not np.array_equal(
        trace.frames[positions], requested
    ):
        missing = requested[~np.isin(requested, trace.frames)]
        raise ValueError(
            "frames must lie inside one recovered GVHMR segment; missing "
            f"source frames: {missing.tolist()}."
        )
    segment = trace.segment_ids[positions]
    if len(np.unique(segment)) != 1:
        raise ValueError(
            "frames cross GVHMR segments; a single segment is required "
            f"(found {sorted(set(segment.tolist()))})."
        )
    return positions


def _trace_world(trace: PlacementTrace, scale: float) -> NDArray[np.float64]:
    correction = Rotation.from_euler(
        "z", trace.yaw_correction.astype(np.float64)[:, None]
    ).as_matrix()
    rotated = np.einsum("tij,tvj->tvi", correction, trace.vertices_local.astype(np.float64))
    return trace.hip_position[:, None, :] + scale * rotated


def _saved_world(
    placed: ReconstructedPlayers,
    player: int,
    frames: NDArray[np.int64],
    joint_regressor: NDArray[np.float32],
) -> NDArray[np.float32]:
    if placed.vertices_local is None:
        raise ValueError("body_placement artifact lacks placed vertices.")
    return smpl_vertices_to_court(
        placed.vertices_local[player, frames][None],
        placed.global_orient[player, frames][None],
        placed.position[player, frames][None],
        placed.yaw[player, frames][None],
        joint_regressor,
    )[0]


def _assert_world_matches(
    recomputed: NDArray[np.float64],
    saved: NDArray[np.float64] | NDArray[np.float32],
    mask: NDArray[np.bool_],
    frames: NDArray[np.int64],
    identity: int,
    *,
    context: str,
) -> float:
    """Reject non-finite or >= tolerance differences for selected frames.

    Returns the maximum absolute difference over the inspected frames so callers
    that report diagnostics (the ranking exporter) do not recompute it.
    """
    if recomputed.shape != saved.shape or mask.shape != frames.shape:
        raise ValueError(
            "World comparison arrays disagree: "
            f"recomputed={recomputed.shape}, saved={saved.shape}, "
            f"mask={mask.shape}, frames={frames.shape}."
        )
    if mask.shape[0] != recomputed.shape[0]:
        raise ValueError("World comparison mask does not match the frame axis.")
    selected = np.flatnonzero(mask)
    if not len(selected):
        return 0.0
    recomputed_values = np.asarray(recomputed[selected], np.float64)
    saved_values = np.asarray(saved[selected], np.float64)
    for label, values in (("recomputed", recomputed_values), ("saved", saved_values)):
        finite = np.isfinite(values)
        if not finite.all():
            masked_frame, vertex, component = np.unravel_index(
                int(np.argmin(finite)), values.shape
            )
            local_frame = int(selected[int(masked_frame)])
            source_frame = int(frames[local_frame])
            raise ValueError(
                f"{context} {label} world position is non-finite: "
                f"identity={identity}, source_frame={source_frame}, "
                f"vertex={int(vertex)}, component={int(component)}, "
                f"recomputed_world_m="
                f"{recomputed_values[int(masked_frame), int(vertex)].tolist()}, "
                f"saved_world_m="
                f"{saved_values[int(masked_frame), int(vertex)].tolist()}."
            )
    difference = np.abs(recomputed_values - saved_values)
    worst_flat = int(np.argmax(difference))
    worst = float(difference.reshape(-1)[worst_flat])
    if worst >= PLACEMENT_TOLERANCE_M:
        masked_frame, vertex, _ = np.unravel_index(worst_flat, difference.shape)
        local_frame = int(selected[int(masked_frame)])
        source_frame = int(frames[local_frame])
        raise ValueError(
            f"{context} differs from the saved body_placement artifact by at "
            f"least {PLACEMENT_TOLERANCE_M:g} m: identity={identity}, "
            f"source_frame={source_frame}, vertex={int(vertex)}, "
            f"max_abs_difference_m={worst:.9g}, "
            f"recomputed_world_m="
            f"{recomputed_values[int(masked_frame), int(vertex)].tolist()}, "
            f"saved_world_m="
            f"{saved_values[int(masked_frame), int(vertex)].tolist()}."
        )
    return worst


def _validate_root_validity(
    placed: ReconstructedPlayers,
    trace: PlacementTrace,
    player: int,
    identity: int,
) -> None:
    """The saved root mask and the public trace validity must agree on trace.frames."""
    saved = np.asarray(placed.root_valid[player, trace.frames], bool)
    recomputed = np.asarray(trace.valid, bool)
    if saved.shape != recomputed.shape:
        raise ValueError(
            "Saved root_valid and recomputed trace.valid differ in length: "
            f"identity={identity}, saved={saved.shape}, trace={recomputed.shape}."
        )
    mismatch = saved != recomputed
    if mismatch.any():
        frames = trace.frames[mismatch]
        raise ValueError(
            "Saved body_placement root_valid disagrees with the recomputed "
            f"trace.valid: identity={identity}, "
            f"source_frames={frames.tolist()}, "
            f"saved_root_valid={saved[mismatch].tolist()}, "
            f"trace_valid={recomputed[mismatch].tolist()}."
        )


def _validate_reproduction(
    inputs: PlacementInputs,
    placed: ReconstructedPlayers,
    trace: PlacementTrace,
    player: int,
    scale: float,
    identity: int,
) -> tuple[NDArray[np.float64], float]:
    """Stop when current code no longer reproduces the saved placement artifact."""
    frames = trace.frames
    smpl_valid = np.asarray(placed.smpl_valid[player, frames])
    recomputed = _trace_world(trace, scale)
    saved = _saved_world(placed, player, frames, inputs.joint_regressor)
    max_difference = _assert_world_matches(
        recomputed,
        saved,
        smpl_valid,
        frames,
        identity,
        context="Recomputed placement",
    )
    return recomputed, max_difference


def _placed_vertices_t0(
    inputs: PlacementInputs,
    placed: ReconstructedPlayers,
    trace: PlacementTrace,
    positions: NDArray[np.int64],
    recomputed_world: NDArray[np.float64],
    t0: int,
    player: int,
    identity: int,
) -> NDArray[np.float32]:
    local = int(np.flatnonzero(trace.frames[positions] == t0)[0])
    trace_index = int(positions[local])
    if not bool(trace.valid[trace_index]):
        raise ValueError(
            f"t0={t0} is not valid for identity {identity} "
            f"(rejection code {int(trace.reasons[trace_index])})."
        )
    if not bool(placed.smpl_valid[player, t0]):
        raise ValueError(
            f"t0={t0} has no saved SMPL body for identity {identity}; "
            "placed_vertices_t0 would be undefined."
        )
    saved = _saved_world(
        placed, player, np.asarray([t0], np.int64), inputs.joint_regressor
    )
    _assert_world_matches(
        recomputed_world[trace_index : trace_index + 1],
        saved,
        np.ones(1, bool),
        np.asarray([t0], np.int64),
        identity,
        context="t0 placement",
    )
    return np.asarray(saved[0], np.float32)


def _git_commit() -> str:
    from src.utils.paths import PROJECT_ROOT

    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return cast(
        dict[str, Any], json.loads(json.dumps(dict(value), default=str))
    )


def export_placement_bundle(
    inputs: PlacementInputs,
    *,
    body: BodyGeometryModel,
    output_dir: Path,
    config: PlacementExportConfig,
    smpl_faces_path: Path,
    resolved_config: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
) -> Path:
    """Write one identity's placement bundle and return its manifest path."""
    if output_dir.exists():
        raise FileExistsError(f"Bundle output already exists: {output_dir}")
    if not smpl_faces_path.is_file():
        raise FileNotFoundError(f"SMPL faces asset is missing: {smpl_faces_path}")
    player, recovered_body = _select_identity(inputs, config.identity)
    trace = trace_player_placement(
        recovered_body,
        inputs.skeleton,
        inputs.grouped,
        inputs.cameras,
        player=player,
        body=body,
        fps=inputs.source.fps,
        placement_config=inputs.placement_config,
    )
    scale = trace.scale
    if scale is None:
        raise ValueError(
            f"Placement for identity {config.identity} has no fitted scale; "
            "the bundle cannot encode the object transform."
        )
    _validate_root_validity(inputs.placed, trace, player, config.identity)
    positions = _range_indices(trace, config.frames)
    world, _ = _validate_reproduction(
        inputs,
        inputs.placed,
        trace,
        player,
        float(scale),
        config.identity,
    )
    t0_vertices = _placed_vertices_t0(
        inputs,
        inputs.placed,
        trace,
        positions,
        world,
        config.t0,
        player,
        config.identity,
    )

    valid = np.asarray(trace.valid[positions], bool)
    body_local = np.where(
        valid[:, None, None],
        trace.vertices_local[positions],
        0.0,
    ).astype(np.float32, copy=False)
    joints_local = np.asarray(trace.joints_local[positions], np.float64)
    target_joints = np.asarray(trace.target_joints[positions], np.float32)
    target_weights = np.asarray(trace.target_weights[positions], np.float64)
    hip_position = np.asarray(trace.hip_position[positions], np.float64)
    yaw_correction = np.asarray(trace.yaw_correction[positions], np.float64)
    segment_id = int(trace.segment_ids[positions][0])
    gvhmr_camera_id = trace.segment_camera_ids[segment_id]

    faces = np.asarray(load_smpl_faces(smpl_faces_path), np.int32)
    vertex_count = body_local.shape[1]
    if faces.size == 0 or faces.min() < 0 or int(faces.max()) >= vertex_count:
        raise ValueError(
            f"SMPL faces reference {faces.max()} but the body has {vertex_count} vertices."
        )

    plate_camera, plate_pose = _plate_camera(inputs, config)
    plate_video = _validate_plate_video(
        inputs,
        plate_camera_id=config.plate_camera_id,
        plate_video_path=config.plate_video_path,
    )
    plate_sha256 = dual_sha256(plate_video)
    smpl_faces_sha256 = dual_sha256(smpl_faces_path)

    output_dir.mkdir(parents=True)
    try:
        np.save(output_dir / "smpl_faces.npy", faces)
        write_pc2(output_dir / "body_local.pc2", body_local, start_frame=0.0)
        np.save(output_dir / "joints_local.npy", joints_local)
        np.save(output_dir / "target_joints.npy", target_joints)
        np.save(output_dir / "target_weights.npy", target_weights)
        np.save(output_dir / "hip_position.npy", hip_position)
        np.save(output_dir / "yaw_correction.npy", yaw_correction)
        np.save(output_dir / "valid.npy", valid)
        np.save(output_dir / "placed_vertices_t0.npy", t0_vertices)
        _write_plate(
            plate_video,
            output_dir / "plate",
            frame_range=range(config.frames[0], config.frames[1]),
            max_width=None,
            quality=config.plate_jpeg_quality,
        )

        start, stop = config.frames
        source_frames = list(range(start, stop))
        provenance = {
            "data_origin": "measured",
            "elements": {
                "body_local": "measured",
                "joints_local": "measured",
                "target_joints": "measured",
                "target_weights": "measured",
                "hip_position": "measured",
                "yaw_correction": "measured",
                "valid": "measured",
                "placed_vertices_t0": "measured",
                "court": "measured",
                "plate": "measured",
            },
            "store": dict(inputs.store_provenance),
            "artifacts": dict(inputs.artifact_provenance),
            "assets": dict(inputs.asset_provenance),
            "smpl_faces": {
                "path": str(smpl_faces_path),
                "sha256": smpl_faces_sha256,
            },
            "git_commit": git_commit if git_commit is not None else _git_commit(),
            "resolved_config": _json_mapping(resolved_config),
            "implementation": {
                "trace": "src.tennis_scene.pipeline.components.player_reconstruction.trace_player_placement",
                "geometry": "src.tennis_scene.pipeline.components.body_placement.SmplGeometry",
                "world": "src.tennis_scene.visualization.rendering.smpl_court.smpl_vertices_to_court",
            },
        }
        manifest = {
            "schema": PLACEMENT_BUNDLE_SCHEMA,
            "fps": float(inputs.source.fps),
            "width": plate_camera.width,
            "height": plate_camera.height,
            "camera": {
                "opencv": {
                    "K": plate_camera.K.tolist(),
                    "R": plate_camera.R.tolist(),
                    "t": plate_camera.t.tolist(),
                },
                "blender": plate_pose.to_dict(),
            },
            "frames": {
                "start": start,
                "stop": stop,
                "source_frames": source_frames,
            },
            "t0": config.t0,
            "identity": config.identity,
            "gvhmr_camera_id": gvhmr_camera_id,
            "scale": float(scale),
            "coco17_skeleton": [list(edge) for edge in COCO17_SKELETON],
            "court": court_geometry(),
            "plate": {
                "camera_id": config.plate_camera_id,
                "video": str(plate_video),
                "video_sha256": plate_sha256,
                "pattern": PLATE_PATTERN,
                "jpeg_quality": config.plate_jpeg_quality,
                "frames": source_frames,
            },
            "source": {
                "clip_id": inputs.source.clip_id,
                "camera_ids": list(inputs.source.camera_ids),
                "pixel_threshold_scale": float(inputs.source.pixel_threshold_scale),
            },
            "files": {
                "smpl_faces": "smpl_faces.npy",
                "body_local": "body_local.pc2",
                "joints_local": "joints_local.npy",
                "target_joints": "target_joints.npy",
                "target_weights": "target_weights.npy",
                "hip_position": "hip_position.npy",
                "yaw_correction": "yaw_correction.npy",
                "valid": "valid.npy",
                "placed_vertices_t0": "placed_vertices_t0.npy",
            },
            "provenance": provenance,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return manifest_path


def _plate_camera(
    inputs: PlacementInputs, config: PlacementExportConfig
) -> tuple[BlenderPinholeCamera, BlenderCameraPose]:
    """Return the calibrated plate camera and its Blender pose, or stop."""
    matches = [
        camera
        for camera in inputs.cameras
        if camera.camera_id == config.plate_camera_id
    ]
    if not matches:
        raise ValueError(
            f"plate_camera_id {config.plate_camera_id!r} has no calibrated camera "
            "in camera_alignment; calibrated cameras: "
            f"{[camera.camera_id for camera in inputs.cameras]}."
        )
    camera = matches[0]
    width, height = inputs.source.size
    export_camera = BlenderPinholeCamera(
        K=np.asarray(camera.intrinsic, np.float64),
        R=np.asarray(camera.rotation, np.float64),
        t=np.asarray(camera.translation, np.float64).reshape(3),
        width=width,
        height=height,
    )
    pose = to_blender_camera(
        export_camera, sensor_width_mm=PLACEMENT_SENSOR_WIDTH_MM
    )
    return export_camera, pose


def _validate_plate_video(
    inputs: PlacementInputs,
    *,
    plate_camera_id: str,
    plate_video_path: Path,
) -> Path:
    if plate_camera_id not in inputs.source.camera_ids:
        raise ValueError(
            f"plate_camera_id {plate_camera_id!r} is not a source camera "
            f"{list(inputs.source.camera_ids)}."
        )
    video = inputs.source.video(plate_camera_id)
    path = plate_video_path
    if not path.is_file():
        raise FileNotFoundError(f"Plate video is missing: {path}")
    actual = dual_sha256(path)
    if actual != video.sha256:
        raise ValueError(
            f"Plate video SHA-256 mismatch: config {path} has {actual}, "
            f"store source {plate_camera_id} has {video.sha256}."
        )
    info = probe_video_info(path)
    if (info.width, info.height) != (video.width, video.height):
        raise ValueError(
            f"Plate video is {info.width}x{info.height}, store source "
            f"{video.width}x{video.height}."
        )
    if info.frame_count != video.num_frames:
        raise ValueError(
            f"Plate video has {info.frame_count} frames, store source has "
            f"{video.num_frames}."
        )
    if not math.isclose(info.fps, video.fps, rel_tol=1e-5):
        raise ValueError(
            f"Plate video fps {info.fps} differs from store source fps {video.fps}."
        )
    return path


def _cpu_body_model(inputs: PlacementInputs) -> BodyGeometryModel:
    """Build the CPU SMPL reconstructor bound to the saved store assets."""
    from src.tennis_scene.pipeline.model_assets import PeopleModelConfig

    binding = _BodyModelBinding(
        inputs.body_models_dir,
        inputs.body_assets,
        _CpuRuntime(),
    )
    return cast(BodyGeometryModel, SmplGeometry(cast(PeopleModelConfig, binding)))


def export_blender_placement_bundle(
    *,
    store_dir: Path,
    output_dir: Path,
    config: PlacementExportConfig,
    smpl_faces_path: Path,
    resolved_config: Mapping[str, Any] | None = None,
    body: BodyGeometryModel | None = None,
) -> Path:
    """Read ``store_dir``, reconstruct on CPU and write the placement bundle."""
    inputs = read_placement_inputs(
        store_dir, human_vis_threshold=config.human_vis_threshold
    )
    owned = body is None
    if body is None:
        body = _cpu_body_model(inputs)
    try:
        return export_placement_bundle(
            inputs,
            body=body,
            output_dir=output_dir,
            config=config,
            smpl_faces_path=smpl_faces_path,
            resolved_config=resolved_config,
        )
    finally:
        if owned:
            body.unload()


__all__ = [
    "PLACEMENT_BUNDLE_SCHEMA",
    "PLACEMENT_SENSOR_WIDTH_MM",
    "PLACEMENT_TOLERANCE_M",
    "PlacementExportConfig",
    "PlacementInputs",
    "PlacementStore",
    "export_blender_placement_bundle",
    "export_placement_bundle",
    "read_placement_inputs",
]
