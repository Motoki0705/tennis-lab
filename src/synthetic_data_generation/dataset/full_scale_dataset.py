"""Atomic multi-trajectory, multi-camera 3DGS synthetic dataset publication."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.dataset.single_frame_pilot import (
    TRACKNET_COLUMNS,
)
from src.synthetic_data_generation.rendering.renderer_port import (
    RendererPort,
    RenderRequest,
    SpherePrimitive,
)
from src.synthetic_data_generation.scene_contract import SceneCamera, SceneContract
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData

DATASET_SCHEMA = "blcs_3dgs_full_scale_dataset_v1"
_MANIFEST_KEYS = {
    "schema",
    "dataset_fingerprint",
    "identity",
    "label_statistics",
    "diversity",
    "publication",
}
_INVENTORY_SCHEMA = "blcs_3dgs_payload_inventory_v1"


@dataclass(frozen=True)
class TrajectorySamplingSpec:
    """Identity and simulator inputs for one independently seeded BLCS rally."""

    seed: int
    from_cell: int
    side: str
    scene_id: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("Trajectory seed must be non-negative.")
        if not 0 <= self.from_cell <= 8:
            raise ValueError("Trajectory from_cell must lie in [0, 8].")
        if self.side not in {"near", "far"}:
            raise ValueError("Trajectory side must be 'near' or 'far'.")
        if not self.scene_id.strip():
            raise ValueError("Trajectory scene_id must not be empty.")


@dataclass(frozen=True)
class StaticSceneProvenance:
    """Verified static 3DGS RGB/depth artifact used for one captured camera."""

    camera_id: str
    uri: str
    sha256: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or not self.uri.strip():
            raise ValueError("Static-scene camera_id and URI must not be empty.")
        for name, digest in (
            ("sha256", self.sha256),
            ("request_fingerprint", self.request_fingerprint),
        ):
            _validate_sha256(digest, f"Static-scene {name}")


@dataclass(frozen=True)
class FullScaleDatasetConfig:
    """Frozen rendering and contiguous-clip selection settings."""

    camera_ids: tuple[str, ...]
    trajectories: tuple[TrajectorySamplingSpec, ...]
    clip_length: int
    ball_radius_m: float = 0.0335
    ball_color_rgb: tuple[int, int, int] = (64, 192, 64)
    supersampling: int = 4
    jpeg_quality: int = 95
    expected_renderer_backend_id: str = "deterministic-cpu-sphere-reference"
    expected_renderer_backend_version: str = "1"

    def __post_init__(self) -> None:
        camera_ids = tuple(self.camera_ids)
        trajectories = tuple(self.trajectories)
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Dataset camera_ids must be non-empty and unique.")
        if not trajectories:
            raise ValueError("Dataset trajectories must not be empty.")
        if len({spec.scene_id for spec in trajectories}) != len(trajectories):
            raise ValueError("Dataset trajectory scene_ids must be unique.")
        if self.clip_length < 8:
            raise ValueError("Dataset clip_length must be at least eight frames.")
        if not np.isfinite(self.ball_radius_m) or self.ball_radius_m <= 0.0:
            raise ValueError("Dataset ball_radius_m must be finite and positive.")
        if len(self.ball_color_rgb) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.ball_color_rgb
        ):
            raise ValueError("Dataset ball_color_rgb must contain three uint8 values.")
        if not 1 <= self.supersampling <= 16:
            raise ValueError("Dataset supersampling must lie in [1, 16].")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Dataset jpeg_quality must lie in [1, 100].")
        if (
            not self.expected_renderer_backend_id.strip()
            or not self.expected_renderer_backend_version.strip()
        ):
            raise ValueError("Expected renderer identity must not be empty.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "trajectories", trajectories)


@dataclass(frozen=True)
class FullScaleProvenance:
    """Immutable scene, code, and static-frame identities."""

    scene_contract_uri: str
    scene_contract_sha256: str
    static_scenes: tuple[StaticSceneProvenance, ...]
    git_revision: str
    git_dirty: bool
    code_diff_sha256: str

    def __post_init__(self) -> None:
        if not self.scene_contract_uri.strip():
            raise ValueError("SceneContract URI must not be empty.")
        _validate_sha256(self.scene_contract_sha256, "SceneContract SHA-256")
        _validate_sha256(self.code_diff_sha256, "Code diff SHA-256")
        if len(self.git_revision) != 40:
            raise ValueError("Git revision must contain 40 characters.")
        static_scenes = tuple(self.static_scenes)
        if not static_scenes:
            raise ValueError("At least one static scene provenance is required.")
        camera_ids = [source.camera_id for source in static_scenes]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Static-scene camera IDs must be unique.")
        object.__setattr__(self, "static_scenes", static_scenes)


def publish_full_scale_dataset(
    *,
    scenes: Sequence[BLCSSceneData],
    scene_contract: SceneContract,
    renderer: RendererPort,
    config: FullScaleDatasetConfig,
    provenance: FullScaleProvenance,
    output_root: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Render and atomically publish grouped TrackNet clips for training only."""
    scene_tuple = tuple(scenes)
    if len(scene_tuple) != len(config.trajectories):
        raise ValueError("BLCS scene count must match trajectory specifications.")
    cameras = _resolve_cameras(scene_contract, config.camera_ids)
    if len({camera.group_id for camera in cameras}) != len(cameras):
        raise ValueError("Full-scale cameras must belong to distinct camera groups.")
    static_by_camera = {source.camera_id: source for source in provenance.static_scenes}
    if set(static_by_camera) != set(config.camera_ids):
        raise ValueError("Static-scene provenance must exactly cover dataset cameras.")

    trajectory_records: list[dict[str, Any]] = []
    transformed_positions: list[np.ndarray[Any, Any]] = []
    segments: list[tuple[int, int]] = []
    for scene, spec in zip(scene_tuple, config.trajectories, strict=True):
        _validate_scene(scene, spec)
        frame_count = int(scene.ball_pos_world.shape[0])
        if frame_count < config.clip_length:
            raise ValueError(
                f"BLCS trajectory {scene.scene_id!r} has {frame_count} frames, "
                f"shorter than clip_length={config.clip_length}."
            )
        start = _deterministic_segment_start(
            seed=spec.seed,
            frame_count=frame_count,
            clip_length=config.clip_length,
        )
        stop = start + config.clip_length
        court_positions = scene.ball_pos_world.detach().cpu().numpy().astype(np.float64)
        scene_positions = scene_contract.alignment.scene_from_court.apply(
            court_positions
        )
        transformed_positions.append(scene_positions)
        segments.append((start, stop))
        trajectory_records.append(
            {
                "spec": asdict(spec),
                "trajectory_sha256": _array_sha256(court_positions),
                "velocity_sha256": _array_sha256(
                    scene.ball_vel_world.detach().cpu().numpy()
                ),
                "frame_count": frame_count,
                "segment_start": start,
                "segment_stop_exclusive": stop,
                "rally_length": scene.rally_length,
                "end_reason": scene.end_reason,
                "winner_side": scene.winner_side,
                "shots": scene.shots,
                "fps_out": scene.fps_out,
                "sim_fps": scene.sim_fps,
                "physics_config": scene.physics_config_dict,
                "court_config": scene.court_config_dict,
            }
        )

    camera_records = [
        _camera_identity(camera, scene_contract=scene_contract) for camera in cameras
    ]
    identity: dict[str, Any] = {
        "scene_fingerprint": scene_contract.scene_fingerprint,
        "scene_id": scene_contract.scene_id,
        "scene_contract_uri": provenance.scene_contract_uri,
        "scene_contract_sha256": provenance.scene_contract_sha256,
        "alignment_id": scene_contract.alignment.alignment_id,
        "alignment_manifest_sha256": scene_contract.alignment.manifest.sha256,
        "court_to_scene_scale": scene_contract.alignment.scene_from_court.scale,
        "cameras": camera_records,
        "static_scenes": [
            asdict(static_by_camera[camera_id]) for camera_id in config.camera_ids
        ],
        "trajectories": trajectory_records,
        "rendering": {
            "ball_radius_m": config.ball_radius_m,
            "ball_radius_scene_units": (
                scene_contract.alignment.scene_from_court.scale * config.ball_radius_m
            ),
            "ball_color_rgb": list(config.ball_color_rgb),
            "supersampling": config.supersampling,
            "jpeg_quality": config.jpeg_quality,
            "renderer_backend_id": config.expected_renderer_backend_id,
            "renderer_backend_version": config.expected_renderer_backend_version,
        },
        "split_policy": {
            "synthetic_usage": "train_only",
            "grouping_unit": "trajectory_and_camera_group",
            "clip_length": config.clip_length,
            "clip_count": len(scene_tuple) * len(cameras),
        },
        "code": {
            "git_revision": provenance.git_revision,
            "git_dirty": provenance.git_dirty,
            "code_diff_sha256": provenance.code_diff_sha256,
        },
    }
    dataset_fingerprint = _fingerprint_json(identity)
    output_dir = output_root / dataset_fingerprint
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_fingerprint}.", dir=output_root))

    total_frames = len(scene_tuple) * len(cameras) * config.clip_length
    rendered_frames = 0
    payloads: list[Path] = []
    split_entries: list[str] = []
    annotations_path = staging / "annotations.jsonl"
    label_statistics: Counter[str] = Counter()
    diameter_values: list[float] = []
    speed_values: list[float] = []
    displacement_values: list[float] = []
    visible_displacement_values: list[float] = []
    background_std_values: list[float] = []
    contrast_values: list[float] = []
    occupancy_positions: list[np.ndarray[Any, Any]] = []
    clip_duration_values: list[float] = []
    radius_scene_units = (
        scene_contract.alignment.scene_from_court.scale * config.ball_radius_m
    )
    try:
        split_dir = staging / "splits"
        split_dir.mkdir()
        with annotations_path.open("w", encoding="utf-8") as annotations_handle:
            for trajectory_index, (
                scene,
                spec,
                scene_positions,
                segment,
            ) in enumerate(
                zip(
                    scene_tuple,
                    config.trajectories,
                    transformed_positions,
                    segments,
                    strict=True,
                )
            ):
                start, stop = segment
                court_positions = (
                    scene.ball_pos_world.detach().cpu().numpy().astype(np.float64)
                )
                velocities = (
                    scene.ball_vel_world.detach().cpu().numpy().astype(np.float64)
                )
                occupancy_positions.append(court_positions[start:stop])
                clip_duration_values.append(config.clip_length / scene.fps_out)
                for camera in cameras:
                    relative_clip = Path(
                        "train",
                        f"trajectory_{trajectory_index:03d}",
                        (f"Clip_group_{camera.group_id:03d}_{camera.camera_id}"),
                    )
                    clip_dir = staging / relative_clip
                    clip_dir.mkdir(parents=True)
                    split_entries.append(relative_clip.as_posix())
                    frame_records: list[dict[str, Any]] = []
                    previous_projected: tuple[float, float] | None = None
                    previous_visible_projected: tuple[float, float] | None = None
                    for clip_index, source_index in enumerate(range(start, stop)):
                        sphere = SpherePrimitive(
                            primitive_id="ball-b001",
                            center_scene=(
                                float(scene_positions[source_index, 0]),
                                float(scene_positions[source_index, 1]),
                                float(scene_positions[source_index, 2]),
                            ),
                            radius_scene_units=radius_scene_units,
                            color_rgb=config.ball_color_rgb,
                        )
                        result = renderer.render(
                            RenderRequest(
                                scene_fingerprint=scene_contract.scene_fingerprint,
                                frame_index=source_index,
                                camera=camera,
                                spheres=(sphere,),
                                supersampling=config.supersampling,
                            )
                        )
                        _validate_render_metadata(result.metadata, config=config)
                        evidence = result.spheres[0]
                        projected = evidence.projected_center_xy
                        center_in_frame = bool(
                            projected is not None
                            and 0.0 <= projected[0] < camera.width
                            and 0.0 <= projected[1] < camera.height
                        )
                        positive = bool(
                            center_in_frame and evidence.visible_pixel_equivalent > 0.0
                        )
                        x = float(projected[0]) if positive and projected else 0.0
                        y = float(projected[1]) if positive and projected else 0.0
                        displacement = _pixel_displacement(
                            previous=previous_projected,
                            current=projected,
                        )
                        previous_projected = projected
                        visible_displacement = _pixel_displacement(
                            previous=previous_visible_projected,
                            current=projected if positive else None,
                        )
                        previous_visible_projected = projected if positive else None
                        local = _local_background_statistics(
                            rgb=result.rgb,
                            alpha=result.alpha,
                            projected_center=projected,
                            apparent_diameter_px=evidence.apparent_diameter_px,
                            ball_color_rgb=config.ball_color_rgb,
                        )
                        file_name = f"{clip_index:06d}.jpg"
                        frame_path = clip_dir / file_name
                        _write_jpeg(
                            frame_path,
                            result.rgb,
                            quality=config.jpeg_quality,
                        )
                        payloads.append(frame_path)
                        speed = float(np.linalg.norm(velocities[source_index]))
                        record: dict[str, Any] = {
                            "trajectory_index": trajectory_index,
                            "trajectory_scene_id": spec.scene_id,
                            "trajectory_seed": spec.seed,
                            "camera_id": camera.camera_id,
                            "camera_group_id": camera.group_id,
                            "clip_path": relative_clip.as_posix(),
                            "file_name": file_name,
                            "clip_frame_index": clip_index,
                            "source_trajectory_frame_index": source_index,
                            "court_position_m": [
                                float(value) for value in court_positions[source_index]
                            ],
                            "scene_position": [
                                float(value) for value in scene_positions[source_index]
                            ],
                            "speed_mps": speed,
                            "projected_center_xy": (
                                list(projected) if projected is not None else None
                            ),
                            "center_in_frame": center_in_frame,
                            "tracknet_visibility": 1.0 if positive else 0.0,
                            "tracknet_x": x,
                            "tracknet_y": y,
                            "visibility_state": evidence.visibility.value,
                            "visible_pixel_fraction": (evidence.visible_pixel_fraction),
                            "covered_pixel_equivalent": (
                                evidence.covered_pixel_equivalent
                            ),
                            "visible_pixel_equivalent": (
                                evidence.visible_pixel_equivalent
                            ),
                            "apparent_diameter_px": evidence.apparent_diameter_px,
                            "pixel_displacement_from_previous": displacement,
                            "visible_pixel_displacement_from_previous": (
                                visible_displacement
                            ),
                            "local_background_mean_rgb": local["mean_rgb"],
                            "local_background_std": local["std"],
                            "ball_background_contrast": local["contrast"],
                        }
                        annotations_handle.write(
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        frame_records.append(record)
                        label_statistics["frame_count"] += 1
                        label_statistics[
                            "positive_frames" if positive else "negative_frames"
                        ] += 1
                        label_statistics[f"{evidence.visibility.value}_frames"] += 1
                        if evidence.in_frame:
                            diameter_values.append(evidence.apparent_diameter_px)
                        speed_values.append(speed)
                        if displacement is not None:
                            displacement_values.append(displacement)
                        if visible_displacement is not None:
                            visible_displacement_values.append(visible_displacement)
                        if local["std"] is not None:
                            background_std_values.append(float(local["std"]))
                        if local["contrast"] is not None:
                            contrast_values.append(float(local["contrast"]))
                        rendered_frames += 1
                        if progress is not None:
                            progress(rendered_frames, total_frames)
                    label_path = clip_dir / "Label.csv"
                    _write_label_csv(label_path, frame_records)
                    payloads.append(label_path)

        payloads.append(annotations_path)
        train_split = split_dir / "train.txt"
        train_split.write_text(
            "".join(f"{entry}\n" for entry in split_entries),
            encoding="utf-8",
        )
        val_split = split_dir / "val.txt"
        val_split.write_text(
            "# synthetic samples are training-only\n",
            encoding="utf-8",
        )
        test_split = split_dir / "test.txt"
        test_split.write_text(
            "# synthetic samples are training-only\n",
            encoding="utf-8",
        )
        payloads.extend((train_split, val_split, test_split))
        if label_statistics["positive_frames"] == 0:
            raise ValueError(
                "Full-scale dataset contains no trainable positive frames."
            )
        if label_statistics["negative_frames"] == 0:
            raise ValueError("Full-scale dataset contains no useful negative frames.")

        inventory_path = staging / "payload_inventory.jsonl"
        _write_inventory(
            inventory_path,
            root=staging,
            payloads=payloads,
        )
        occupancy = np.concatenate(occupancy_positions, axis=0)
        counts, x_edges, y_edges = np.histogram2d(
            occupancy[:, 0],
            occupancy[:, 1],
            bins=(6, 8),
            range=((-6.0, 6.0), (-12.0, 12.0)),
        )
        diversity = {
            "camera_pose_count": len(cameras),
            "camera_group_count": len({camera.group_id for camera in cameras}),
            "trajectory_count": len(scene_tuple),
            "court_occupancy": {
                "x_edges_m": x_edges.tolist(),
                "y_edges_m": y_edges.tolist(),
                "counts": counts.astype(int).tolist(),
                "outside_grid_positions": int(len(occupancy) - int(counts.sum())),
            },
            "ball_pixel_diameter": _distribution(diameter_values),
            "speed_mps": _distribution(speed_values),
            "projected_center_displacement": _distribution(displacement_values),
            "visible_ball_displacement": _distribution(visible_displacement_values),
            "visibility_states": {
                state: label_statistics[f"{state}_frames"]
                for state in (
                    "fully_visible",
                    "partially_occluded",
                    "fully_occluded",
                    "out_of_frame",
                )
            },
            "local_background_std": _distribution(background_std_values),
            "ball_background_contrast": _distribution(contrast_values),
            "clip_duration_seconds": _distribution(clip_duration_values),
        }
        manifest = {
            "schema": DATASET_SCHEMA,
            "dataset_fingerprint": dataset_fingerprint,
            "identity": identity,
            "label_statistics": dict(sorted(label_statistics.items())),
            "diversity": diversity,
            "publication": {
                "frame_count": total_frames,
                "clip_count": len(split_entries),
                "split_files": {
                    "train": "splits/train.txt",
                    "val": "splits/val.txt",
                    "test": "splits/test.txt",
                },
                "annotations": "annotations.jsonl",
                "payload_inventory": {
                    "schema": _INVENTORY_SCHEMA,
                    "path": inventory_path.name,
                    "entry_count": len(payloads),
                    "sha256": _sha256_file(inventory_path),
                    "size_bytes": inventory_path.stat().st_size,
                },
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_and_validate_full_scale_dataset(staging)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def load_and_validate_full_scale_dataset(path: Path) -> dict[str, Any]:
    """Strictly validate identity, inventory, splits, labels, and JPEG decoding."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Full-scale manifest keys do not match v1 schema.")
    if manifest["schema"] != DATASET_SCHEMA:
        raise ValueError(f"Unsupported full-scale schema: {manifest['schema']!r}.")
    identity = manifest["identity"]
    if not isinstance(identity, dict):
        raise ValueError("Full-scale identity must be an object.")
    if _fingerprint_json(identity) != manifest["dataset_fingerprint"]:
        raise ValueError("Full-scale dataset fingerprint mismatch.")
    if path.name != manifest["dataset_fingerprint"] and not path.name.startswith("."):
        raise ValueError("Full-scale directory name does not match fingerprint.")
    publication = manifest["publication"]
    inventory_record = publication["payload_inventory"]
    inventory_path = path / inventory_record["path"]
    if inventory_path.stat().st_size != inventory_record["size_bytes"]:
        raise ValueError("Payload inventory size mismatch.")
    if _sha256_file(inventory_path) != inventory_record["sha256"]:
        raise ValueError("Payload inventory SHA-256 mismatch.")

    inventory_entries = []
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            inventory_entries.append(json.loads(line))
    if len(inventory_entries) != inventory_record["entry_count"]:
        raise ValueError("Payload inventory entry count mismatch.")
    for entry in inventory_entries:
        if set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Payload inventory record keys are invalid.")
        payload = path / entry["path"]
        if not payload.is_file():
            raise ValueError(f"Dataset payload is missing: {entry['path']}")
        if payload.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Dataset payload size mismatch: {entry['path']}")
        if _sha256_file(payload) != entry["sha256"]:
            raise ValueError(f"Dataset payload SHA-256 mismatch: {entry['path']}")
        if payload.suffix.lower() == ".jpg":
            with Image.open(payload) as image:
                image.verify()

    train_entries = _split_entries(path / publication["split_files"]["train"])
    if len(train_entries) != publication["clip_count"]:
        raise ValueError("Training split clip count mismatch.")
    if _split_entries(path / publication["split_files"]["val"]):
        raise ValueError("Synthetic validation split must be empty.")
    if _split_entries(path / publication["split_files"]["test"]):
        raise ValueError("Synthetic test split must be empty.")
    frame_count = 0
    for entry in train_entries:
        clip_dir = path / entry
        if not clip_dir.is_dir() or not clip_dir.name.startswith("Clip"):
            raise ValueError(f"Invalid TrackNet clip entry: {entry}")
        jpg_names = {payload.name for payload in clip_dir.glob("*.jpg")}
        with (clip_dir / "Label.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        if tuple(rows[0]) != TRACKNET_COLUMNS:
            raise ValueError(f"TrackNet columns are invalid: {entry}")
        if {row["file name"] for row in rows} != jpg_names:
            raise ValueError(f"TrackNet rows do not exactly cover JPEGs: {entry}")
        frame_count += len(rows)
    if frame_count != publication["frame_count"]:
        raise ValueError("Published frame count mismatch.")
    with (path / publication["annotations"]).open("r", encoding="utf-8") as handle:
        annotation_count = sum(1 for line in handle if line.strip())
    if annotation_count != frame_count:
        raise ValueError("Annotation JSONL frame count mismatch.")
    return manifest


def _resolve_cameras(
    scene_contract: SceneContract,
    camera_ids: tuple[str, ...],
) -> tuple[SceneCamera, ...]:
    by_id = {camera.camera_id: camera for camera in scene_contract.cameras}
    missing = sorted(set(camera_ids).difference(by_id))
    if missing:
        raise ValueError(f"Dataset cameras are absent from SceneContract: {missing}")
    return tuple(by_id[camera_id] for camera_id in camera_ids)


def _validate_scene(scene: BLCSSceneData, spec: TrajectorySamplingSpec) -> None:
    if scene.scene_id != spec.scene_id:
        raise ValueError("BLCS scene_id differs from trajectory specification.")
    if (
        scene.initial_from_cell != spec.from_cell
        or scene.initial_from_side != spec.side
    ):
        raise ValueError("BLCS launch metadata differs from trajectory specification.")
    for name, tensor in (
        ("ball_pos_world", scene.ball_pos_world),
        ("ball_vel_world", scene.ball_vel_world),
    ):
        array = tensor.detach().cpu().numpy()
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError(f"BLCS {name} must have finite shape (T, 3).")
    if scene.ball_pos_world.shape != scene.ball_vel_world.shape:
        raise ValueError("BLCS position and velocity shapes differ.")
    if scene.fps_out <= 0 or scene.sim_fps <= 0:
        raise ValueError("BLCS frame rates must be positive.")


def _deterministic_segment_start(
    *,
    seed: int,
    frame_count: int,
    clip_length: int,
) -> int:
    rng = np.random.default_rng(seed ^ 0x3D65_4253)
    return int(rng.integers(0, frame_count - clip_length + 1))


def _camera_identity(
    camera: SceneCamera,
    *,
    scene_contract: SceneContract,
) -> dict[str, Any]:
    camera_to_scene = np.asarray(camera.camera_to_scene).reshape(4, 4)
    camera_center_court = scene_contract.alignment.court_from_scene.apply(
        camera_to_scene[None, :3, 3]
    )[0]
    return {
        "camera_id": camera.camera_id,
        "camera_group_id": camera.group_id,
        "source_camera_id": camera.source_camera_id,
        "source_frame_index": camera.source_frame_index,
        "image_uri": camera.image_uri,
        "width": camera.width,
        "height": camera.height,
        "camera_center_court_m": [float(value) for value in camera_center_court],
    }


def _validate_render_metadata(metadata: Any, *, config: FullScaleDatasetConfig) -> None:
    if metadata.backend_id != config.expected_renderer_backend_id:
        raise ValueError("Renderer backend ID differs from frozen dataset config.")
    if metadata.backend_version != config.expected_renderer_backend_version:
        raise ValueError("Renderer backend version differs from frozen dataset config.")
    if not metadata.deterministic:
        raise ValueError("Full-scale baseline requires a deterministic renderer.")


def _pixel_displacement(
    *,
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
) -> float | None:
    if previous is None or current is None:
        return None
    return float(np.linalg.norm(np.asarray(current) - np.asarray(previous)))


def _local_background_statistics(
    *,
    rgb: np.ndarray[Any, Any],
    alpha: np.ndarray[Any, Any],
    projected_center: tuple[float, float] | None,
    apparent_diameter_px: float,
    ball_color_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    if projected_center is None:
        return {"mean_rgb": None, "std": None, "contrast": None}
    x, y = projected_center
    if not 0.0 <= x < rgb.shape[1] or not 0.0 <= y < rgb.shape[0]:
        return {"mean_rgb": None, "std": None, "contrast": None}
    radius = max(3, int(np.ceil(apparent_diameter_px)))
    x_start = max(0, int(np.floor(x)) - radius)
    x_stop = min(rgb.shape[1], int(np.floor(x)) + radius + 1)
    y_start = max(0, int(np.floor(y)) - radius)
    y_stop = min(rgb.shape[0], int(np.floor(y)) + radius + 1)
    patch: NDArray[np.float64] = rgb[y_start:y_stop, x_start:x_stop].astype(np.float64)
    alpha_patch = alpha[y_start:y_stop, x_start:x_stop]
    background_pixels = patch[alpha_patch == 0.0]
    if background_pixels.size == 0:
        return {"mean_rgb": None, "std": None, "contrast": None}
    mean = background_pixels.mean(axis=0)
    std = float(background_pixels.std())
    contrast = float(
        np.linalg.norm(np.asarray(ball_color_rgb, dtype=np.float64) - mean)
        / np.sqrt(3.0)
    )
    return {
        "mean_rgb": [float(value) for value in mean],
        "std": std,
        "contrast": contrast,
    }


def _write_label_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKNET_COLUMNS)
        writer.writeheader()
        for frame in frames:
            positive = float(frame["tracknet_visibility"]) > 0.0
            writer.writerow(
                {
                    "file name": frame["file_name"],
                    "visibility": frame["tracknet_visibility"],
                    "x-coordinate": frame["tracknet_x"],
                    "y-coordinate": frame["tracknet_y"],
                    "status": 0,
                    "instance id": "b001" if positive else "",
                    "role": "target",
                    "ball state": frame["visibility_state"],
                    "visible pixel fraction": frame["visible_pixel_fraction"],
                }
            )


def _write_jpeg(path: Path, rgb: np.ndarray[Any, Any], *, quality: int) -> None:
    Image.fromarray(rgb, mode="RGB").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
    )


def _write_inventory(
    path: Path,
    *,
    root: Path,
    payloads: Sequence[Path],
) -> None:
    relatives = sorted(payload.relative_to(root).as_posix() for payload in payloads)
    if len(relatives) != len(set(relatives)):
        raise ValueError("Dataset payload inventory contains duplicate paths.")
    with path.open("w", encoding="utf-8") as handle:
        for relative in relatives:
            payload = root / relative
            record = {
                "path": relative,
                "sha256": _sha256_file(payload),
                "size_bytes": payload.stat().st_size,
            }
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _split_entries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q10": None,
            "median": None,
            "mean": None,
            "q90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _array_sha256(array: np.ndarray[Any, Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _fingerprint_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
