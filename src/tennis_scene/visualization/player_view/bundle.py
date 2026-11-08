"""Export a camera-independent, read-only snapshot of a saved SceneResult v2."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SCENE_MASK_FIELDS, SCENE_REASON_FIELDS, SceneResult
from src.tennis_scene.visualization.blender.export.bundle import court_geometry
from src.tennis_scene.visualization.blender.export.pointcache import write_pc2
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
    smpl_vertices_to_court,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.matrices import (
    SMPL_Y_UP_TO_COURT_Z_UP,
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
)
from src.utils.schema.player import SMPLH_BODY_JOINT_NAMES

from .assets import load_body_kinematics

BUNDLE_SCHEMA = "tennis_scene_player_view_bundle_v1"


@dataclass(frozen=True, slots=True)
class PlayerViewBundleConfig:
    """Source-frame interval and player-ID selection, preserving explicit gaps."""

    frames: tuple[int, int] | None = None
    player_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.frames is not None and (
            type(self.frames) is not tuple
            or len(self.frames) != 2
            or any(type(v) is not int for v in self.frames)
            or not 0 <= self.frames[0] < self.frames[1]
        ):
            raise ValueError("frames must be (start, stop), 0 <= start < stop")
        if self.player_ids is not None and (
            type(self.player_ids) is not tuple
            or not self.player_ids
            or any(type(v) is not int or v < 0 for v in self.player_ids)
            or len(set(self.player_ids)) != len(self.player_ids)
        ):
            raise ValueError("player_ids must be a non-empty tuple of unique IDs")


def _select(
    scene: SceneResult, config: PlayerViewBundleConfig
) -> tuple[list[int], range]:
    track_ids = scene.player_track_ids
    count = scene.player_position.shape[0]
    if (
        track_ids is None
        or track_ids.shape != (count,)
        or track_ids.dtype.kind not in "iu"
        or count == 0
        or (track_ids < 0).any()
        or len(set(track_ids.tolist())) != count
    ):
        raise ValueError("Scene requires unique non-negative player_track_ids (P,)")
    identities = track_ids.tolist()
    selected = identities if config.player_ids is None else config.player_ids
    unknown = set(selected) - set(identities)
    if unknown:
        raise ValueError(f"Unknown player IDs: {sorted(unknown)}")
    start, stop = (0, scene.num_frames) if config.frames is None else config.frames
    if stop > scene.num_frames:
        raise ValueError("Requested frames exceed the source scene timeline")
    return [identities.index(identity) for identity in selected], range(start, stop)


def _scales(scene: SceneResult, indices: list[int]) -> list[float | None]:
    """V2 has already scaled local mesh; require its recorded scale for FK."""
    placement = scene.metadata.get("body_placement", {})
    records = placement.get("players", {}) if isinstance(placement, dict) else {}
    result: list[float | None] = []
    assert scene.player_track_ids is not None and scene.player_smpl_valid is not None
    for index in indices:
        identity = int(scene.player_track_ids[index])
        record = records.get(str(identity), {}) if isinstance(records, dict) else {}
        scale = record.get("scale") if isinstance(record, dict) else None
        if scale is None and not scene.player_smpl_valid[index].any():
            result.append(None)
        elif (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(
                f"Player {identity} requires a recorded positive body_placement scale"
            )
        else:
            result.append(float(scale))
    return result


def _source(input_path: Path) -> tuple[Path, dict[str, Any]]:
    source: dict[str, Any] = {"input": str(input_path.resolve())}
    scene_path = input_path
    if input_path.is_dir() or input_path.name == "scene.json":
        index_path = input_path / "scene.json" if input_path.is_dir() else input_path
        scene_path = indexed_scene_path(index_path)
        index = json.loads(index_path.read_text())
        source["store_index"] = {
            "path": str(index_path.resolve()),
            "sha256": dual_sha256(index_path),
            "source_sha256": index["source_sha256"],
            "export": index["exports"]["scene"],
        }
    source.update(
        scene=str(scene_path.resolve()),
        scene_sha256=dual_sha256(scene_path),
        metadata_sha256=dual_sha256(scene_path.with_suffix(".metadata.json")),
    )
    return scene_path, source


def _write_array(
    directory: Path, arrays: dict[str, Any], name: str, value: NDArray
) -> None:
    filename = f"{name}.npy"
    np.save(directory / filename, value, allow_pickle=False)
    arrays[name] = {
        "path": filename,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _write_player(
    directory: Path,
    scene: SceneResult,
    source_index: int,
    index: int,
    frames: range,
    regressor: NDArray[np.float32],
) -> tuple[dict[str, Any], NDArray[np.float32], NDArray[np.float32]]:
    assert (
        scene.smpl_vertices_local is not None and scene.smpl_global_orient is not None
    )
    assert scene.player_smpl_valid is not None and scene.player_track_ids is not None
    selection = slice(frames.start, frames.stop)
    valid = scene.player_smpl_valid[source_index, selection]
    local = scene.smpl_vertices_local[source_index, selection]
    orient = scene.smpl_global_orient[source_index, selection]
    yaw = scene.player_yaw[source_index, selection]
    # Renderer applies the transpose of the encoded global rotation.
    rotation = (
        rotation_matrix_z(yaw)
        @ SMPL_Y_UP_TO_COURT_Z_UP
        @ axis_angle_to_rotation_matrix(orient).transpose(0, 2, 1)
    )
    root = np.einsum("v,tvc->tc", regressor[0], local)
    rotation[~valid] = 0
    root[~valid] = 0
    pc2_name = f"player_{index}.pc2"
    # Bound resident geometry to one 32-frame chunk. The cache is disk-backed,
    # and write_pc2 owns the existing PC2 format/header contract.
    cache_path = directory / ".vertices.npy"
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=local.shape
    )
    for offset in range(0, len(frames), 32):
        start, stop = (
            frames.start + offset,
            min(frames.start + offset + 32, frames.stop),
        )
        world = smpl_vertices_to_court(
            scene.smpl_vertices_local[source_index : source_index + 1, start:stop],
            scene.smpl_global_orient[source_index : source_index + 1, start:stop],
            scene.player_position[source_index : source_index + 1, start:stop],
            scene.player_yaw[source_index : source_index + 1, start:stop],
            regressor,
        )[0]
        world[~valid[offset : offset + len(world)]] = 0
        cache[offset : offset + len(world)] = world
    write_pc2(directory / pc2_name, cache, start_frame=0.0)
    del cache
    cache_path.unlink()
    return (
        {
            "index": index,
            "source_index": source_index,
            "player_id": int(scene.player_track_ids[source_index]),
            "pc2": pc2_name,
            "valid_frames": int(valid.sum()),
        },
        root,
        rotation,
    )


def export_player_view_bundle(
    *,
    scene_path: Path,
    smpl_faces_path: Path,
    smpl_joint_regressor_path: Path,
    smplx_model_path: Path,
    output_dir: Path,
    config: PlayerViewBundleConfig,
) -> Path:
    """Publish a complete bundle; refuse existing outputs and remove partial work.

    Store inputs are resolved exclusively through indexed_scene_path. Neither
    source videos nor store component arrays are read to fill missing inputs.
    """
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Bundle output already exists: {output_dir}")
    archive_path, source = _source(scene_path)
    scene = load_scene_result(archive_path)
    if scene.schema_version != 2:
        raise ValueError(
            "Player-view export requires SceneResult v2 with validity masks"
        )
    indices, frames = _select(scene, config)
    for name in (
        "smpl_vertices_local",
        "smpl_global_orient",
        "smpl_body_pose",
        "smpl_betas",
        "ball_3d",
    ):
        if getattr(scene, name) is None:
            raise ValueError(f"Player-view export requires {name}")
    scales = _scales(scene, indices)
    assert scene.smpl_vertices_local is not None
    faces = load_smpl_faces(smpl_faces_path.resolve()).astype(np.int32)
    regressor = load_smpl_joint_regressor(smpl_joint_regressor_path)
    vertices = scene.smpl_vertices_local.shape[2]
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.size == 0
        or faces.min() < 0
        or faces.max() >= vertices
    ):
        raise ValueError(
            "SMPL faces must be non-empty triangles indexing the scene mesh"
        )
    if (
        regressor.ndim != 2
        or regressor.shape[0] != 24
        or regressor.shape[1] != vertices
        or not np.isfinite(regressor).all()
    ):
        raise ValueError(
            "SMPL joint regressor must be finite (24, V) for the scene mesh"
        )
    if not np.allclose(regressor.sum(1), 1.0, atol=1e-4):
        raise ValueError("SMPL joint regressor rows must sum to one")
    kinematics = load_body_kinematics(smplx_model_path)
    assets = {
        name: {"path": str(path.resolve()), "sha256": dual_sha256(path)}
        for name, path in {
            "smpl_faces": smpl_faces_path,
            "smpl_joint_regressor": smpl_joint_regressor_path,
            "smplx_model": smplx_model_path,
        }.items()
    }
    # Serialize metadata before creating any output to catch invalid provenance.
    metadata_text = json.dumps(scene.metadata, indent=2, allow_nan=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        arrays: dict[str, Any] = {}
        selection = slice(frames.start, frames.stop)
        for name in (
            *SCENE_MASK_FIELDS,
            *SCENE_REASON_FIELDS,
            "smpl_body_pose",
            "smpl_global_orient",
            "player_position",
            "player_yaw",
        ):
            value = getattr(scene, name)
            assert value is not None
            selected = (
                value[selection]
                if name.startswith("ball_")
                else value[indices, selection]
            )
            _write_array(staging, arrays, name, selected)
        assert scene.smpl_betas is not None and scene.ball_3d is not None
        _write_array(staging, arrays, "smpl_betas", scene.smpl_betas[indices])
        _write_array(staging, arrays, "ball_positions", scene.ball_3d[selection])
        source_frames: NDArray[np.int64] = np.arange(
            frames.start, frames.stop, dtype=np.int64
        )
        _write_array(staging, arrays, "source_frames", source_frames)
        _write_array(staging, arrays, "source_time_seconds", source_frames / scene.fps)
        _write_array(staging, arrays, "smpl_faces", faces)
        _write_array(staging, arrays, "smpl_joint_regressor", regressor)
        for name, value in kinematics.items():
            _write_array(staging, arrays, name, value)
        players, roots, rotations = [], [], []
        for index, source_index in enumerate(indices):
            player, root, rotation = _write_player(
                staging, scene, source_index, index, frames, regressor
            )
            player["scale"] = scales[index]
            players.append(player)
            roots.append(root)
            rotations.append(rotation)
        _write_array(staging, arrays, "smpl_root_local", np.stack(roots))
        _write_array(staging, arrays, "body_to_court_rotation", np.stack(rotations))
        (staging / "source_metadata.json").write_text(metadata_text, encoding="utf-8")
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "source": source,
            "source_metadata": "source_metadata.json",
            "config": asdict(config),
            "frames": {
                "start": frames.start,
                "stop": frames.stop,
                "source_num_frames": scene.num_frames,
            },
            "num_frames": len(frames),
            "fps": scene.fps,
            "coordinates": {
                "units": "m",
                "world": "court",
                "ground": "XY",
                "origin": "court center on ground below net center",
                "x": "court width",
                "y": "court length",
                "up": "+Z",
                "handedness": "right",
                "angles": "radians",
                "vectors": "column",
            },
            "players": players,
            "arrays": arrays,
            "smpl": {
                "parameters": {
                    "body_pose": "smpl_body_pose",
                    "betas": "smpl_betas",
                    "global_orient": "smpl_global_orient",
                    "transl": "player_position",
                },
                "body_joint_names": list(SMPLH_BODY_JOINT_NAMES),
                "mesh_joint_names": [
                    *SMPLH_BODY_JOINT_NAMES,
                    "left_hand",
                    "right_hand",
                ],
                "body_pose": "SMPL-X 21 body joints, parent-relative axis-angle, radians",
                "global_orient": "SceneResult v2 renderer compensation, not raw incam orientation",
                "transl": "player_position: SMPL joint0 in court metres, not raw incam translation",
                "local_vertices": "scale already applied; root-centered canonical posed SMPL mesh",
                "mesh_transform": "world = body_to_court_rotation @ (local_vertex - smpl_root_local) + player_position",
                "rotation": "Rz(player_yaw) @ y_up_to_z_up @ Rodrigues(smpl_global_orient).T",
                "y_up_to_z_up": SMPL_Y_UP_TO_COURT_Z_UP.tolist(),
                "joint_position": "Regress the placed SMPL mesh with smpl_joint_regressor; SMPL-X FK positions use a different topology",
                "joint_rotation": "body_to_court_rotation @ product of parent-relative body_pose rotations",
                "assets": assets,
            },
            "pc2": {
                "start_frame": 0.0,
                "sample_rate": 1.0,
                "frame_mapping": "cache sample i = source_frames[i]",
            },
            "validity": {
                "mesh": "player_smpl_valid",
                "ball": "ball_3d_valid",
                "invalid_values": {
                    "pc2": 0,
                    "ball_positions": 0,
                    "body_to_court_rotation": 0,
                },
                "interpolation": "none",
            },
            "court": court_geometry(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Bundle output already exists: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output_dir / "manifest.json"
