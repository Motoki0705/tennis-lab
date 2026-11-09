"""Bind saved scene/camera inputs into a portable, non-interpolating Blender shot."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.tennis_scene.visualization.blender.export.camera import to_blender_camera
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2, write_pc2
from src.utils.checksum import dual_sha256

from .camera_bundle import _file, _InputBundle, load_head_cameras
from .head_camera import HeadCameraConfig
from .lenses import EquidistantLensConfig

SHOT_SCHEMA = "tennis_scene_player_view_blender_v1"


@dataclass(frozen=True)
class PlayerViewRenderConfig:
    """Prototype appearance; camera intrinsics and FPS come from saved inputs."""

    ball_radius_m: float = 0.05
    samples: int = 16
    player_color: tuple[float, float, float] = (0.45, 0.45, 0.45)
    ball_color: tuple[float, float, float] = (0.12, 0.8, 0.025)

    def __post_init__(self) -> None:
        if not np.isfinite(self.ball_radius_m) or self.ball_radius_m <= 0:
            raise ValueError("ball_radius_m must be finite and positive")
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("samples must be a positive integer")
        for color in (self.player_color, self.ball_color):
            if len(color) != 3 or not all(
                np.isfinite(v) and 0 <= v <= 1 for v in color
            ):
                raise ValueError("Colors must contain three finite values in [0,1]")


def prepare_blender_shot(
    *,
    bundle_dir: Path,
    cameras_dir: Path,
    output_dir: Path,
    config: PlayerViewRenderConfig,
    frame_range: tuple[int, int] | None = None,
    lens: EquidistantLensConfig | None = None,
) -> Path:
    """Check C2 provenance against C1 and publish copied inputs atomically."""
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Shot input already exists: {output_dir}")
    if any(
        output_dir.resolve().is_relative_to(p.resolve())
        for p in (bundle_dir, cameras_dir)
    ):
        raise ValueError("Shot output must be outside its read-only inputs")
    source = _InputBundle(bundle_dir)
    manifest = source.manifest
    camera_manifest = json.loads((cameras_dir / "manifest.json").read_text())
    original_camera_config = HeadCameraConfig(**camera_manifest["config"])
    render_camera_config = (
        original_camera_config
        if lens is None
        else lens.pinhole_config(original_camera_config)
    )
    for name, digest in camera_manifest["source_bundle"]["files"].items():
        if dual_sha256(_file(bundle_dir, name)) != digest:
            raise ValueError(f"Saved camera input digest differs: {name}")
    if "manifest.json" not in camera_manifest["source_bundle"]["files"]:
        raise ValueError("Camera provenance must include the scene manifest")
    if camera_manifest["coordinates"] != manifest["coordinates"]:
        raise ValueError("Camera and scene coordinate conventions differ")
    identities = [p["player_id"] for p in source.players]
    if identities != [p["player_id"] for p in camera_manifest["players"]]:
        raise ValueError("Camera and scene player IDs/order differ")
    count, frames = len(identities), source.frames
    source_frames = source.array("source_frames", (frames,), "int64")
    times = source.array("source_time_seconds", (frames,), "float64")
    mesh_valid = source.array("player_smpl_valid", (count, frames), "bool")
    ball_valid = source.array("ball_3d_valid", (frames,), "bool")
    ball = source.array("ball_positions", (frames, 3), "float32")
    faces_record = manifest["arrays"]["smpl_faces"]
    faces = source.array("smpl_faces", tuple(faces_record["shape"]), "int32")
    if frame_range is None:
        first, last = int(source_frames[0]), int(source_frames[-1]) + 1
    else:
        if len(frame_range) != 2 or any(type(v) is not int for v in frame_range):
            raise ValueError("Frame range requires integer [start, stop)")
        first, last = frame_range
        if not int(source_frames[0]) <= first < last <= int(source_frames[-1]) + 1:
            raise ValueError(
                "Frame range must be a nonempty subset of the source timeline"
            )
    selection = slice(first - int(source_frames[0]), last - int(source_frames[0]))
    selected_indices = np.arange(frames)[selection]
    selected_count = last - first
    if not np.isfinite(ball[ball_valid]).all() or (ball[~ball_valid] != 0).any():
        raise ValueError("Ball values disagree with validity")
    if manifest["pc2"]["start_frame"] != 0 or manifest["pc2"]["sample_rate"] != 1:
        raise ValueError("PC2 must start at zero with sample rate one")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        players = []
        camera_files = {"manifest.json"}
        written_files = set()
        for index, player in enumerate(source.players):
            saved = load_head_cameras(cameras_dir, player_id=player["player_id"])
            if (
                saved.fps != manifest["fps"]
                or not np.array_equal(saved.source_frames, source_frames)
                or not np.array_equal(saved.source_time_seconds, times)
                or (saved.series.camera_valid & ~mesh_valid[index]).any()
            ):
                raise ValueError("Camera and scene timeline/masks differ")
            world, start, rate = read_pc2(_file(bundle_dir, player["pc2"]))
            if (
                world.shape[0] != frames
                or start != 0
                or rate != 1
                or not np.isfinite(world[mesh_valid[index]]).all()
                or (world[~mesh_valid[index]] != 0).any()
                or faces.ndim != 2
                or faces.shape[1] != 3
                or (faces < 0).any()
                or (faces >= world.shape[1]).any()
            ):
                raise ValueError("PC2 geometry/timing disagrees with scene")
            if frame_range is not None:
                write_pc2(
                    staging / player["pc2"],
                    world[selection],
                    start_frame=0,
                    sample_rate=1,
                )
                written_files.add(player["pc2"])
            del world
            locations = np.full((selected_count, 3), np.nan)
            rotations = np.full((selected_count, 3, 3), np.nan)
            render_series = replace(
                saved.series,
                config=render_camera_config,
                K=render_camera_config.intrinsic_matrix(),
            )
            focal_mm = render_series.K[0, 0] * 36.0 / render_camera_config.width
            for output_index, frame in enumerate(selected_indices):
                if not saved.series.camera_valid[frame]:
                    continue
                pose = to_blender_camera(
                    render_series.pinhole(int(frame)), sensor_width_mm=36.0
                )
                # Preserve the saved centre exactly; R.T @ R is only approximately
                # identity for rotations originating from float32 scene placement.
                locations[output_index] = saved.series.camera_to_court[frame, :3, 3]
                rotations[output_index] = pose.rotation_matrix
            filename = f"camera_{index}.npz"
            np.savez(
                staging / filename,
                location=locations,
                rotation=rotations,
                valid=saved.series.camera_valid[selection],
                reason=saved.series.camera_rejection_code[selection],
            )
            camera_files.add(camera_manifest["players"][index]["path"])
            source.files.add(player["pc2"])
            players.append(
                {
                    **player,
                    "valid_frames": int(mesh_valid[index, selection].sum()),
                    "camera": filename,
                    "lens_mm": float(focal_mm),
                    "camera_valid_frames": int(
                        saved.series.camera_valid[selection].sum()
                    ),
                }
            )
        selected_arrays = {
            "source_frames": source_frames[selection],
            "source_time_seconds": times[selection],
            "player_smpl_valid": mesh_valid[:, selection],
            "ball_positions": ball[selection],
            "ball_3d_valid": ball_valid[selection],
        }
        array_records = {
            key: dict(manifest["arrays"][key])
            for key in (*selected_arrays, "smpl_faces")
        }
        for key, value in selected_arrays.items():
            record = array_records[key]
            np.save(staging / record["path"], value)
            record["shape"] = list(value.shape)
            written_files.add(record["path"])
        for name in source.files - {"manifest.json"} - written_files:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_file(bundle_dir, name), destination)
        result = {
            "schema": SHOT_SCHEMA,
            "num_frames": selected_count,
            "fps": manifest["fps"],
            "frames": {**manifest["frames"], "start": first, "stop": last},
            "selection": {
                "source_frame_interval": [first, last],
                "camera_context": "saved input camera sequence; filtering precedes frame selection",
            },
            "players": players,
            "court": manifest["court"],
            "arrays": array_records,
            "camera_config": asdict(render_camera_config),
            "camera_processing": camera_manifest.get("stabilization"),
            "config": asdict(config),
            "policy": {
                "prototype_settings": True,
                "viewer_mesh": "hide_entire_mesh",
                "alpha": 1.0,
                "emission_strength": 0.0,
                "interpolation": "none",
                "motion_blur": False,
                "object_missing": "hide_object",
                "camera_missing": "blank_labeled_frame_preserving_time",
                "frame_mapping": "Blender/cache frame i = source_frames[i]",
                "camera_estimation": "none; saved C2 poses only",
                "human_acceptance": "pending",
            },
            "inputs": {
                "scene_files": {
                    name: dual_sha256(_file(bundle_dir, name))
                    for name in sorted(source.files)
                },
                "camera_files": {
                    name: dual_sha256(_file(cameras_dir, name))
                    for name in sorted(camera_files)
                },
            },
        }
        if lens is not None:
            result["lens"] = lens.metadata(original_camera_config)
        result["files"] = {
            str(p.relative_to(staging)): dual_sha256(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Shot input already exists: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output_dir / "manifest.json"
