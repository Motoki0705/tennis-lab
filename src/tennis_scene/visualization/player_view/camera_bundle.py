"""Persist one authoritative head-camera sequence per player from a C1 bundle."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.visualization.blender.export.pointcache import read_pc2
from src.utils.checksum import dual_sha256

from .bundle import BUNDLE_SCHEMA
from .head_camera import (
    OPENCV_TO_HEAD,
    CameraRejectionCode,
    HeadCameraConfig,
    HeadCameraSeries,
    build_head_camera_series,
    head_joint_chain,
)

CAMERA_SCHEMA = "tennis_scene_player_view_head_cameras_v1"


def _file(directory: Path, relative: str) -> Path:
    path = directory / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        directory.resolve()
    ):
        raise ValueError("Bundle file paths must stay inside the bundle directory")
    return path


class _InputBundle:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.files = {"manifest.json"}
        manifest = self.manifest
        if manifest["schema"] != BUNDLE_SCHEMA:
            raise ValueError("Head cameras require a player-view scene bundle v1")
        coordinates = manifest["coordinates"]
        if any(
            coordinates[key] != value
            for key, value in {
                "units": "m",
                "world": "court",
                "up": "+Z",
                "vectors": "column",
                "handedness": "right",
            }.items()
        ):
            raise ValueError("Bundle coordinates must be right-handed court metres")
        self.frames = manifest["num_frames"]
        self.players = manifest["players"]
        if type(self.frames) is not int or self.frames <= 0 or not self.players:
            raise ValueError("Bundle must contain players and a nonempty timeline")
        identities = [player["player_id"] for player in self.players]
        if (
            any(type(v) is not int or v < 0 for v in identities)
            or len(set(identities)) != len(identities)
            or [p["index"] for p in self.players] != list(range(len(self.players)))
        ):
            raise ValueError("Bundle requires unique player IDs and ordered indices")

    def array(self, name: str, shape: tuple[int, ...], dtype: str) -> NDArray:
        record = self.manifest["arrays"][name]
        path = _file(self.directory, record["path"])
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            value.shape != shape
            or value.dtype != np.dtype(dtype)
            or record["shape"] != list(shape)
            or record["dtype"] != str(value.dtype)
        ):
            raise ValueError(
                f"Bundle array {name} disagrees with its shape/dtype contract"
            )
        self.files.add(record["path"])
        return value


def export_head_cameras(
    *,
    bundle_dir: Path,
    output_dir: Path,
    config: HeadCameraConfig,
) -> Path:
    """Read only C1, publish atomically, and refuse any existing output."""
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Camera output already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(bundle_dir.resolve()):
        raise ValueError("Camera output must be outside the read-only source bundle")
    source = _InputBundle(bundle_dir)
    manifest = source.manifest
    frames, players = source.frames, len(source.players)
    source_frames = source.array("source_frames", (frames,), "int64")
    times = source.array("source_time_seconds", (frames,), "float64")
    interval = manifest["frames"]
    fps = manifest["fps"]
    if (
        not np.isfinite(fps)
        or fps <= 0
        or interval["start"] < 0
        or interval["stop"] > interval["source_num_frames"]
        or not np.array_equal(
            source_frames, np.arange(interval["start"], interval["stop"])
        )
        or not np.allclose(times, source_frames / fps, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Bundle timeline must preserve consecutive source frames and FPS"
        )
    pose = source.array("smpl_body_pose", (players, frames, 63), "float32")
    rotation = source.array(
        "body_to_court_rotation", (players, frames, 3, 3), "float32"
    )
    valid = source.array("player_smpl_valid", (players, frames), "bool")
    source_reasons = source.array("player_rejection_code", (players, frames), "uint8")
    parents = source.array("smplx_parents", (22,), "int32")
    names = manifest["smpl"]["body_joint_names"]
    mesh_names = manifest["smpl"]["mesh_joint_names"]
    chain = head_joint_chain(parents, names, mesh_names)
    regressor_record = manifest["arrays"]["smpl_joint_regressor"]
    vertices = regressor_record["shape"][1]
    regressor = source.array("smpl_joint_regressor", (24, vertices), "float32")
    if manifest["pc2"]["start_frame"] != 0 or manifest["pc2"]["sample_rate"] != 1:
        raise ValueError("Bundle PC2 timeline must start at zero with sample rate one")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        records = []
        for index, player in enumerate(source.players):
            cache_path = _file(bundle_dir, player["pc2"])
            world, start, rate = read_pc2(cache_path)
            if world.shape != (frames, vertices, 3) or start != 0 or rate != 1:
                raise ValueError(
                    "Placed PC2 dimensions/timing disagree with the bundle"
                )
            source.files.add(player["pc2"])
            series = build_head_camera_series(
                placed_vertices=world,
                body_pose=pose[index],
                body_to_court_rotation=rotation[index],
                mesh_valid=valid[index],
                scale=player["scale"],
                joint_regressor=regressor,
                parents=parents,
                body_joint_names=names,
                mesh_joint_names=mesh_names,
                config=config,
            )
            del world  # bound resident geometry to one player
            filename = f"player_{index}.npz"
            np.savez(
                staging / filename,
                K=series.K,
                camera_to_court=series.camera_to_court,
                camera_valid=series.camera_valid,
                camera_rejection_code=series.camera_rejection_code,
                head_position=series.head_position,
                source_frames=source_frames,
                source_time_seconds=times,
                source_player_rejection_code=source_reasons[index],
            )
            records.append(
                {
                    "index": index,
                    "source_index": player["source_index"],
                    "player_id": player["player_id"],
                    "path": filename,
                    "scale": player["scale"],
                    "valid_frames": int(series.camera_valid.sum()),
                    "rejection_counts": {
                        code.name: int((series.camera_rejection_code == code).sum())
                        for code in CameraRejectionCode
                    },
                }
            )
        result = {
            "schema": CAMERA_SCHEMA,
            "source_bundle": {
                "path": os.path.relpath(bundle_dir.resolve(), output_dir.resolve()),
                "schema": BUNDLE_SCHEMA,
                "files": {
                    name: dual_sha256(_file(bundle_dir, name))
                    for name in sorted(source.files)
                },
            },
            "num_frames": frames,
            "fps": fps,
            "frames": interval,
            "players": records,
            "config": asdict(config),
            "coordinates": manifest["coordinates"],
            "camera": {
                "axes": {"x": "right", "y": "down", "z": "forward"},
                "camera_to_court": "4x4 homogeneous, column vectors; X_court = T @ X_camera",
                "projection": "X_camera = T[:3,:3].T @ (X_court - T[:3,3]); uv = K @ X_camera / z",
                "K": "constant per player; zero skew, square pixels, principal point (width/2,height/2)",
                "distortion": "none",
                "shutter": "static pinhole per source frame",
                "near_far": "camera +Z depth in court metres",
            },
            "head": {
                "joint": "head",
                "rotation_chain": [names[j] for j in chain],
                "rotation": "body_to_court_rotation @ product(parent-relative body_pose rotations)",
                "position": "smpl_joint_regressor[mesh_joint_names.index('head')] @ placed_pc2_vertices",
                "head_axes": {"up": "+Y", "forward": "+Z", "subject_left": "+X"},
                "opencv_to_head": OPENCV_TO_HEAD.tolist(),
                "center": "regressed_head + head_rotation @ (player_scale * face_offset_head_m)",
                "roll": "preserved",
                "smoothing": "none",
                "interpolation": "none",
                "ball_tracking": "none",
            },
            "validity": {
                "mask": "camera_valid",
                "reasons": "camera_rejection_code",
                "reason_codes": {
                    str(int(code)): code.name for code in CameraRejectionCode
                },
                "invalid_values": "camera_to_court and head_position are all NaN; K stays defined",
                "reason_precedence": [
                    code.name for code in CameraRejectionCode if code != 0
                ],
                "source_player_rejection_code": "unchanged C1 player reason; independent of camera reason",
                "fallback": "none",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Camera output already exists: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output_dir / "manifest.json"


@dataclass(frozen=True, slots=True)
class SavedHeadCameras:
    """Portable reader result; consumers reuse these poses without recalculating FK."""

    player_id: int
    series: HeadCameraSeries
    source_frames: NDArray[np.int64]
    source_time_seconds: NDArray[np.float64]
    source_player_rejection_code: NDArray[np.uint8]
    fps: float


def load_head_cameras(directory: Path, *, player_id: int) -> SavedHeadCameras:
    """Load a selected original player ID, preserving missing camera samples."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != CAMERA_SCHEMA:
        raise ValueError("Expected a head-camera bundle v1")
    matches = [p for p in manifest["players"] if p["player_id"] == player_id]
    if len(matches) != 1:
        raise ValueError(f"Camera bundle requires exactly one player ID {player_id}")
    config = HeadCameraConfig(**manifest["config"])
    with np.load(_file(directory, matches[0]["path"]), allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    frames = manifest["num_frames"]
    expected = {
        "K": ((3, 3), "float64"),
        "camera_to_court": ((frames, 4, 4), "float64"),
        "camera_valid": ((frames,), "bool"),
        "camera_rejection_code": ((frames,), "uint8"),
        "head_position": ((frames, 3), "float64"),
        "source_frames": ((frames,), "int64"),
        "source_time_seconds": ((frames,), "float64"),
        "source_player_rejection_code": ((frames,), "uint8"),
    }
    if any(
        values[name].shape != shape or values[name].dtype != np.dtype(dtype)
        for name, (shape, dtype) in expected.items()
    ):
        raise ValueError("Camera archive shape/dtype disagrees with the schema")
    valid = values["camera_valid"]
    transform = values["camera_to_court"]
    reasons = values["camera_rejection_code"]
    if (
        not np.array_equal(valid, reasons == 0)
        or not np.isin(reasons, list(CameraRejectionCode)).all()
        or not np.isnan(transform[~valid]).all()
        or not np.isnan(values["head_position"][~valid]).all()
        or not np.isfinite(transform[valid]).all()
        or not np.isfinite(values["head_position"][valid]).all()
        or not np.allclose(values["K"], config.intrinsic_matrix(), rtol=0, atol=1e-9)
        or not np.allclose(transform[valid, 3], [0, 0, 0, 1], atol=1e-9)
    ):
        raise ValueError(
            "Camera archive has inconsistent validity, intrinsics or transforms"
        )
    rotation = transform[valid, :3, :3]
    if not np.allclose(
        rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6
    ) or not np.allclose(
        np.linalg.det(rotation),
        1,
        atol=1e-6,
    ):
        raise ValueError("Camera archive contains an improper rotation")
    interval = manifest["frames"]
    if (
        not np.isfinite(manifest["fps"])
        or manifest["fps"] <= 0
        or not np.array_equal(
            values["source_frames"], np.arange(interval["start"], interval["stop"])
        )
        or not np.allclose(
            values["source_time_seconds"],
            values["source_frames"] / manifest["fps"],
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError("Camera archive source timeline disagrees with its manifest")
    if "stabilization" in manifest or manifest["head"]["smoothing"] != "none":
        from .stabilization import validate_stabilization_record

        validate_stabilization_record(manifest, values)
    series = HeadCameraSeries(
        K=values["K"],
        camera_to_court=values["camera_to_court"],
        camera_valid=values["camera_valid"],
        camera_rejection_code=values["camera_rejection_code"],
        head_position=values["head_position"],
        config=config,
    )
    return SavedHeadCameras(
        player_id,
        series,
        values["source_frames"],
        values["source_time_seconds"],
        values["source_player_rejection_code"],
        manifest["fps"],
    )
