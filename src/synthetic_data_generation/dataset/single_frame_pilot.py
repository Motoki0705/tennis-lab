"""Deterministic positive/negative TrackNet pilot publication."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.synthetic_data_generation.rendering.renderer_port import (
    RendererPort,
    RenderRequest,
    SpherePrimitive,
    VisibilityState,
)
from src.synthetic_data_generation.scene_contract import SceneContract
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData

PILOT_SCHEMA = "blcs_3dgs_single_frame_pilot_v1"
TRACKNET_COLUMNS = (
    "file name",
    "visibility",
    "x-coordinate",
    "y-coordinate",
    "status",
    "instance id",
    "role",
    "ball state",
    "visible pixel fraction",
)
_MANIFEST_KEYS = {
    "schema",
    "dataset_fingerprint",
    "identity",
    "frames",
    "label_statistics",
    "files",
}


@dataclass(frozen=True)
class SingleFramePilotConfig:
    """Frozen sampling and rendering settings for one positive plus one negative."""

    camera_id: str
    trajectory_frame_index: int
    ball_radius_m: float = 0.0335
    ball_color_rgb: tuple[int, int, int] = (64, 192, 64)
    supersampling: int = 4
    jpeg_quality: int = 100

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("Pilot camera_id must not be empty.")
        if self.trajectory_frame_index < 0:
            raise ValueError("Pilot trajectory_frame_index must be non-negative.")
        if not np.isfinite(self.ball_radius_m) or self.ball_radius_m <= 0.0:
            raise ValueError("Pilot ball_radius_m must be finite and positive.")
        if len(self.ball_color_rgb) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.ball_color_rgb
        ):
            raise ValueError("Pilot ball_color_rgb must contain three uint8 values.")
        if not 1 <= self.supersampling <= 16:
            raise ValueError("Pilot supersampling must lie in [1, 16].")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Pilot jpeg_quality must lie in [1, 100].")


@dataclass(frozen=True)
class PilotProvenance:
    """Immutable identities supplied by the orchestration layer."""

    seed: int
    scene_contract_uri: str
    scene_contract_sha256: str
    static_scene_uri: str
    static_scene_sha256: str
    static_scene_request_fingerprint: str
    git_revision: str
    git_dirty: bool
    code_diff_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("scene_contract_sha256", self.scene_contract_sha256),
            ("static_scene_sha256", self.static_scene_sha256),
            ("static_scene_request_fingerprint", self.static_scene_request_fingerprint),
            ("code_diff_sha256", self.code_diff_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Pilot provenance {name} must be SHA-256.")
        if len(self.git_revision) != 40:
            raise ValueError("Pilot git_revision must be a 40-character commit.")
        if not self.scene_contract_uri or not self.static_scene_uri:
            raise ValueError("Pilot provenance artifact URIs must not be empty.")


def publish_single_frame_pilot(
    *,
    scene: BLCSSceneData,
    scene_contract: SceneContract,
    renderer: RendererPort,
    config: SingleFramePilotConfig,
    provenance: PilotProvenance,
    output_root: Path,
) -> Path:
    """Render and atomically publish one supervised positive and one negative."""
    _validate_scene(scene)
    try:
        camera = next(
            camera
            for camera in scene_contract.cameras
            if camera.camera_id == config.camera_id
        )
    except StopIteration as exc:
        raise ValueError(
            f"Pilot camera {config.camera_id!r} is absent from SceneContract."
        ) from exc
    frame_index = config.trajectory_frame_index
    if frame_index >= int(scene.ball_pos_world.shape[0]):
        raise ValueError("Pilot trajectory_frame_index exceeds BLCS trajectory.")

    court_position = (
        scene.ball_pos_world[frame_index].detach().cpu().numpy().astype(np.float64)
    )
    scene_position = scene_contract.alignment.scene_from_court.apply(
        court_position[None, :]
    )[0]
    radius_scene_units = (
        scene_contract.alignment.scene_from_court.scale * config.ball_radius_m
    )
    sphere = SpherePrimitive(
        primitive_id="ball-b001",
        center_scene=(
            float(scene_position[0]),
            float(scene_position[1]),
            float(scene_position[2]),
        ),
        radius_scene_units=radius_scene_units,
        color_rgb=config.ball_color_rgb,
    )
    positive = renderer.render(
        RenderRequest(
            scene_fingerprint=scene_contract.scene_fingerprint,
            frame_index=frame_index,
            camera=camera,
            spheres=(sphere,),
            supersampling=config.supersampling,
        )
    )
    negative = renderer.render(
        RenderRequest(
            scene_fingerprint=scene_contract.scene_fingerprint,
            frame_index=frame_index,
            camera=camera,
            spheres=(),
            supersampling=config.supersampling,
        )
    )
    evidence = positive.spheres[0]
    if evidence.visibility is not VisibilityState.FULLY_VISIBLE:
        raise ValueError(
            "Pilot positive must be fully visible; got "
            f"{evidence.visibility.value} with fraction "
            f"{evidence.visible_pixel_fraction:.6f}."
        )
    if evidence.projected_center_xy is None:
        raise ValueError("Pilot positive lacks a projected centre.")
    if negative.spheres or float(negative.coverage.max()) != 0.0:
        raise ValueError("Pilot negative render is not empty.")
    if np.array_equal(negative.rgb, positive.rgb) or not np.any(positive.alpha > 0.0):
        raise ValueError("Pilot positive has no rendered ball signal.")

    trajectory_sha256 = _array_sha256(scene.ball_pos_world.detach().cpu().numpy())
    identity: dict[str, object] = {
        "scene_id": scene.scene_id,
        "trajectory_sha256": trajectory_sha256,
        "trajectory_frame_count": int(scene.ball_pos_world.shape[0]),
        "trajectory_frame_index": frame_index,
        "court_position_m": [float(value) for value in court_position],
        "scene_position": [float(value) for value in scene_position],
        "ball_radius_m": config.ball_radius_m,
        "ball_radius_scene_units": radius_scene_units,
        "ball_color_rgb": list(config.ball_color_rgb),
        "camera_id": camera.camera_id,
        "camera_group_id": camera.group_id,
        "supersampling": config.supersampling,
        "fps_out": scene.fps_out,
        "sim_fps": scene.sim_fps,
        "physics_config": scene.physics_config_dict,
        "court_config": scene.court_config_dict,
        "scene_fingerprint": scene_contract.scene_fingerprint,
        "alignment_id": scene_contract.alignment.alignment_id,
        "alignment_manifest_sha256": scene_contract.alignment.manifest.sha256,
        "renderer_backend_id": positive.metadata.backend_id,
        "renderer_backend_version": positive.metadata.backend_version,
        "depth_convention": positive.metadata.depth_convention,
        "seed": provenance.seed,
        "scene_contract_uri": provenance.scene_contract_uri,
        "scene_contract_sha256": provenance.scene_contract_sha256,
        "static_scene_uri": provenance.static_scene_uri,
        "static_scene_sha256": provenance.static_scene_sha256,
        "static_scene_request_fingerprint": (
            provenance.static_scene_request_fingerprint
        ),
        "git_revision": provenance.git_revision,
        "git_dirty": provenance.git_dirty,
        "code_diff_sha256": provenance.code_diff_sha256,
    }
    dataset_fingerprint = _fingerprint_json(identity)
    output_dir = output_root / dataset_fingerprint
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_fingerprint}.", dir=output_root))
    try:
        clip_dir = staging / "train" / "Clip1"
        split_dir = staging / "splits"
        clip_dir.mkdir(parents=True)
        split_dir.mkdir()
        _write_jpeg(
            clip_dir / "0000.jpg",
            positive.rgb,
            quality=config.jpeg_quality,
        )
        _write_jpeg(
            clip_dir / "0001.jpg",
            negative.rgb,
            quality=config.jpeg_quality,
        )
        center_x, center_y = evidence.projected_center_xy
        frame_records = [
            {
                "file_name": "0000.jpg",
                "visibility": 1.0,
                "x": center_x,
                "y": center_y,
                "state": evidence.visibility.value,
                "visible_pixel_fraction": evidence.visible_pixel_fraction,
                "covered_pixel_equivalent": evidence.covered_pixel_equivalent,
                "visible_pixel_equivalent": evidence.visible_pixel_equivalent,
                "apparent_diameter_px": evidence.apparent_diameter_px,
            },
            {
                "file_name": "0001.jpg",
                "visibility": 0.0,
                "x": 0.0,
                "y": 0.0,
                "state": "absent",
                "visible_pixel_fraction": 0.0,
                "covered_pixel_equivalent": 0.0,
                "visible_pixel_equivalent": 0.0,
                "apparent_diameter_px": 0.0,
            },
        ]
        _write_label_csv(clip_dir / "Label.csv", frame_records)
        (split_dir / "train.txt").write_text(
            "train/Clip1\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            clip_dir / "render_evidence.npz",
            positive_alpha=positive.alpha,
            positive_coverage=positive.coverage,
            positive_sphere_depth=positive.sphere_depth,
            negative_alpha=negative.alpha,
            negative_coverage=negative.coverage,
        )
        relative_files = (
            "train/Clip1/0000.jpg",
            "train/Clip1/0001.jpg",
            "train/Clip1/Label.csv",
            "train/Clip1/render_evidence.npz",
            "splits/train.txt",
        )
        files = {
            relative: {
                "sha256": _sha256_file(staging / relative),
                "size_bytes": (staging / relative).stat().st_size,
            }
            for relative in relative_files
        }
        manifest = {
            "schema": PILOT_SCHEMA,
            "dataset_fingerprint": dataset_fingerprint,
            "identity": identity,
            "frames": frame_records,
            "label_statistics": {
                "frame_count": 2,
                "positive_frames": 1,
                "negative_frames": 1,
                "fully_visible_frames": 1,
                "partially_occluded_frames": 0,
                "fully_occluded_frames": 0,
                "out_of_frame_frames": 0,
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_and_validate_single_frame_pilot(staging)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def load_and_validate_single_frame_pilot(path: Path) -> dict[str, Any]:
    """Strictly reload a published or staged pilot and verify all payloads."""
    manifest_path = path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise ValueError("Pilot manifest keys do not match v1 schema.")
    if raw["schema"] != PILOT_SCHEMA:
        raise ValueError(f"Unsupported pilot schema: {raw['schema']!r}.")
    identity = raw["identity"]
    if not isinstance(identity, dict):
        raise ValueError("Pilot identity must be an object.")
    if _fingerprint_json(identity) != raw["dataset_fingerprint"]:
        raise ValueError("Pilot dataset fingerprint mismatch.")
    if path.name != raw["dataset_fingerprint"] and not path.name.startswith("."):
        raise ValueError("Pilot directory name does not match fingerprint.")
    files = raw["files"]
    if not isinstance(files, dict):
        raise ValueError("Pilot files must be an object.")
    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise ValueError("Pilot file record is invalid.")
        if set(record) != {"sha256", "size_bytes"}:
            raise ValueError("Pilot file record keys are invalid.")
        file_path = path / relative
        if not file_path.is_file():
            raise ValueError(f"Pilot payload is missing: {relative}")
        if file_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Pilot payload size mismatch: {relative}")
        if _sha256_file(file_path) != record["sha256"]:
            raise ValueError(f"Pilot payload SHA-256 mismatch: {relative}")
    _validate_labels(path / "train" / "Clip1" / "Label.csv", raw["frames"])
    with np.load(
        path / "train" / "Clip1" / "render_evidence.npz",
        allow_pickle=False,
    ) as evidence:
        expected_arrays = {
            "positive_alpha",
            "positive_coverage",
            "positive_sphere_depth",
            "negative_alpha",
            "negative_coverage",
        }
        if set(evidence.files) != expected_arrays:
            raise ValueError("Pilot evidence arrays do not match v1 schema.")
        positive_alpha = evidence["positive_alpha"]
        positive_coverage = evidence["positive_coverage"]
        if positive_alpha.dtype != np.float32 or positive_coverage.dtype != np.float32:
            raise ValueError("Pilot evidence alpha/coverage must use float32.")
        if positive_alpha.shape != positive_coverage.shape:
            raise ValueError("Pilot evidence shapes disagree.")
        if not np.any(positive_alpha > 0.0):
            raise ValueError("Pilot positive has no visible pixels.")
        if np.any(evidence["negative_alpha"] != 0.0) or np.any(
            evidence["negative_coverage"] != 0.0
        ):
            raise ValueError("Pilot negative evidence is not empty.")
    for filename in ("0000.jpg", "0001.jpg"):
        with Image.open(path / "train" / "Clip1" / filename) as image:
            image.verify()
    return raw


def _validate_scene(scene: BLCSSceneData) -> None:
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


def _write_jpeg(path: Path, rgb: np.ndarray[Any, Any], *, quality: int) -> None:
    Image.fromarray(rgb, mode="RGB").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
    )


def _write_label_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKNET_COLUMNS)
        writer.writeheader()
        for frame in frames:
            positive = float(frame["visibility"]) > 0.0
            writer.writerow(
                {
                    "file name": frame["file_name"],
                    "visibility": frame["visibility"],
                    "x-coordinate": frame["x"],
                    "y-coordinate": frame["y"],
                    "status": 0,
                    "instance id": "b001" if positive else "",
                    "role": "target",
                    "ball state": frame["state"],
                    "visible pixel fraction": frame["visible_pixel_fraction"],
                }
            )


def _validate_labels(path: Path, frames: object) -> None:
    if not isinstance(frames, list) or len(frames) != 2:
        raise ValueError("Pilot manifest must contain exactly two frame records.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or tuple(rows[0]) != TRACKNET_COLUMNS:
        raise ValueError("Pilot Label.csv does not match TrackNet pilot schema.")
    if rows[0]["visibility"] != "1.0" or rows[0]["instance id"] != "b001":
        raise ValueError("Pilot positive TrackNet label is invalid.")
    if rows[1]["visibility"] != "0.0" or rows[1]["instance id"]:
        raise ValueError("Pilot negative TrackNet label is invalid.")


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
