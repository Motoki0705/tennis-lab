"""Read-only export of saved ball observations and their court-space rays.

All reconstruction geometry is computed here. Blender only displays the saved
arrays; neither this exporter nor the shot reruns inference or triangulation.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.visualization.blender.export.bundle import (
    _write_plate,
    court_geometry,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints
from src.utils.video.reader import probe_video_info

RAYS_SCHEMA = "tennis_scene_blender_rays_v1"


class RayState(IntEnum):
    UNOBSERVED = 0
    INLIER = 1
    OUTLIER = 2
    SINGLE_VIEW = 3
    REJECTED = 4


class RayLengthRule(IntEnum):
    UNOBSERVED = 0
    BALL = 1
    GROUND = 2
    FALLBACK = 3


@dataclass(frozen=True)
class RaysConfig:
    window_frames: int = 90
    ranking_suppression: str = "non_overlapping"
    fallback_ray_length_m: float = 60.0
    plate_max_width: int = 960
    plate_jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.ranking_suppression not in ("non_overlapping", "t0_distance"):
            raise ValueError(
                "ranking_suppression must be non_overlapping or t0_distance"
            )
        if self.window_frames < 3 or self.plate_max_width < 1:
            raise ValueError("window_frames must be >= 3 and plate_max_width positive")
        if (
            not math.isfinite(self.fallback_ray_length_m)
            or self.fallback_ray_length_m <= 0
        ):
            raise ValueError("fallback_ray_length_m must be finite and positive")
        if not 1 <= self.plate_jpeg_quality <= 100:
            raise ValueError("plate_jpeg_quality must be in [1, 100]")


def validate_ball(ball: BallReconstructionResult, views: int) -> None:
    """Reject inconsistent saved masks rather than inventing observations."""
    if ball.status != "ok":
        raise ValueError(f"ball.status must be ok, got {ball.status!r}")
    trajectory = ball.trajectory
    frames = len(trajectory.valid)
    for name, array, shape, dtype in (
        ("uv_px", ball.uv_px, (views, frames, 2), None),
        ("visibility", ball.visibility, (views, frames), np.bool_),
        ("positions", trajectory.positions, (frames, 3), None),
        ("valid", trajectory.valid, (frames,), np.bool_),
        ("reasons", trajectory.reasons, (frames,), np.uint8),
        ("inliers", trajectory.inliers, (views, frames), np.bool_),
        ("reprojection_px", trajectory.reprojection_px, (views, frames), None),
    ):
        if array.shape != shape or (dtype is not None and array.dtype != dtype):
            raise ValueError(
                f"Invalid ball {name} shape/dtype: {array.shape}/{array.dtype}"
            )
    if not np.isfinite(ball.uv_px[ball.visibility]).all():
        raise ValueError("Visible detections must be finite")
    if not np.isfinite(trajectory.positions[trajectory.valid]).all():
        raise ValueError("Valid ball positions must be finite")
    if np.any(trajectory.inliers & ~ball.visibility):
        raise ValueError("Ball inliers must be observed")
    if np.any(trajectory.valid & (ball.visibility.sum(axis=0) < 2)):
        raise ValueError("A valid ball needs at least two observed views")
    if not np.array_equal(trajectory.reasons == 0, trajectory.valid):
        raise ValueError("Ball rejection reasons disagree with valid")


def ray_arrays(
    cameras: tuple[PinholeCamera, ...],
    ball: BallReconstructionResult,
    *,
    fallback_ray_length_m: float,
) -> dict[str, NDArray[Any]]:
    """Calculate directions, closest points and explicit length/state rules."""
    validate_ball(ball, len(cameras))
    if not math.isfinite(fallback_ray_length_m) or fallback_ray_length_m <= 0:
        raise ValueError("fallback_ray_length_m must be finite and positive")
    visible = ball.visibility
    trajectory = ball.trajectory
    views, frames = visible.shape
    origins = np.array([camera.center for camera in cameras], dtype=np.float64)
    directions = np.full((views, frames, 3), np.nan)
    lengths = np.full((views, frames), np.nan)
    rules = np.zeros((views, frames), dtype=np.uint8)
    states = np.zeros((views, frames), dtype=np.uint8)
    closest = np.full((views, frames, 3), np.nan)
    reprojected = np.full((views, frames, 2), np.nan)
    positions = np.where(trajectory.valid[:, None], trajectory.positions, np.nan)
    for view, camera in enumerate(cameras):
        observed = visible[view]
        pixels = np.column_stack((ball.uv_px[view, observed], np.ones(observed.sum())))
        d = pixels @ np.linalg.inv(camera.intrinsic).T @ camera.rotation
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        directions[view, observed] = d
        states[view, observed] = RayState.REJECTED
        states[view, observed & (visible.sum(axis=0) == 1)] = RayState.SINGLE_VIEW
        valid = observed & trajectory.valid
        states[view, valid & trajectory.inliers[view]] = RayState.INLIER
        states[view, valid & ~trajectory.inliers[view]] = RayState.OUTLIER
        distance = np.einsum(
            "ij,ij->i", positions[valid] - origins[view], directions[view, valid]
        )
        if np.any(distance <= 0):
            raise ValueError("Valid ball lies behind an observed ray")
        lengths[view, valid] = distance * 1.15
        rules[view, valid] = RayLengthRule.BALL
        closest[view, valid] = (
            origins[view] + distance[:, None] * directions[view, valid]
        )
        invalid = observed & ~trajectory.valid
        ground_distance = np.full(frames, np.nan)
        np.divide(
            -origins[view, 2],
            directions[view, :, 2],
            out=ground_distance,
            where=np.abs(directions[view, :, 2]) > 1e-12,
        )
        ground = invalid & np.isfinite(ground_distance) & (ground_distance > 0)
        fallback = invalid & ~ground
        lengths[view, ground] = ground_distance[ground]
        lengths[view, fallback] = fallback_ray_length_m
        rules[view, ground] = RayLengthRule.GROUND
        rules[view, fallback] = RayLengthRule.FALLBACK
        projected, front = camera.project(positions)
        reprojected[view, trajectory.valid & front] = projected[
            trajectory.valid & front
        ]
    return {
        "ray_origin": origins,
        "ray_direction": directions,
        "ray_length": lengths,
        "ray_length_rule": rules,
        "ray_state": states,
        "closest_point": closest,
        "ball_position": positions,
        "ball_reason": trajectory.reasons,
        "reprojection_uv": reprojected,
        "detection_uv": np.where(visible[..., None], ball.uv_px, np.nan),
    }


def rank_windows(
    ball: BallReconstructionResult,
    *,
    window_frames: int,
    suppression: str = "non_overlapping",
) -> list[dict[str, Any]]:
    """Greedily suppress similar windows after sorting by validity and start.

    Keep at most ten. Non-overlap permits touching half-open intervals;
    t0_distance requires representative frames at least window_frames apart.
    Neither rule relaxes the eligibility requirements to fill the result list.
    """
    validate_ball(ball, ball.visibility.shape[0])
    if window_frames < 3:
        raise ValueError("window_frames must be >= 3")
    if suppression not in ("non_overlapping", "t0_distance"):
        raise ValueError("Unknown ranking suppression rule")
    valid = ball.trajectory.valid
    support = ball.trajectory.inliers.sum(axis=0)
    outlier = valid & (ball.visibility & ~ball.trajectory.inliers).any(axis=0)
    single = ball.visibility.sum(axis=0) == 1
    rejected = ~valid & (ball.visibility.sum(axis=0) >= 2)
    candidates = []
    for start in range(len(valid) - window_frames + 1):
        stop = start + window_frames
        if not outlier[start:stop].any() or not single[start:stop].any():
            continue
        middle = np.arange(start + window_frames // 3, stop - window_frames // 3)
        supported = middle[valid[middle] & (support[middle] >= 3)]
        if not len(supported):
            continue
        center = (start + stop - 1) / 2
        t0 = int(supported[np.argmin(np.abs(supported - center))])
        candidates.append(
            {
                "start": start,
                "stop": stop,
                "t0": t0,
                "valid_ratio": float(valid[start:stop].mean()),
                "t0_inlier_views": int(support[t0]),
                "outlier_frames": (
                    np.flatnonzero(outlier[start:stop]) + start
                ).tolist(),
                "single_view_frames": (
                    np.flatnonzero(single[start:stop]) + start
                ).tolist(),
                "rejected_frames": (
                    np.flatnonzero(rejected[start:stop]) + start
                ).tolist(),
                "event_counts": {
                    "outlier": int(outlier[start:stop].sum()),
                    "single_view": int(single[start:stop].sum()),
                    "rejected": int(rejected[start:stop].sum()),
                },
            }
        )
    retained: list[dict[str, Any]] = []
    for item in sorted(
        candidates, key=lambda item: (-item["valid_ratio"], item["start"])
    ):
        if suppression == "non_overlapping":
            distinct = all(
                item["stop"] <= other["start"] or item["start"] >= other["stop"]
                for other in retained
            )
        else:
            distinct = all(
                abs(item["t0"] - other["t0"]) >= window_frames for other in retained
            )
        if distinct:
            retained.append(item)
            if len(retained) == 10:
                break
    return retained


@dataclass(frozen=True)
class RaysInput:
    reader: StoreReader
    cameras: tuple[PinholeCamera, ...]
    ball: BallReconstructionResult
    fps: float
    provenance: dict[str, Any]
    ball_source: dict[str, Any]


def _payload(
    reader: StoreReader, node: str, schema: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    stale = reader.stale_dependencies(node)
    if stale:
        raise ValueError(f"Stale {node}: {stale}")
    payload, descriptor = reader.payload(node)
    if (descriptor["output_schema"], descriptor["output_version"]) != (schema, 1):
        raise ValueError(f"Unsupported {node} output schema/version")
    return payload, descriptor


def validate_scene_ball(ball: BallReconstructionResult, scene: dict[str, Any]) -> None:
    """The optional scene export must have the same court coordinates/mask."""
    if not np.array_equal(scene["ball_3d_valid"], ball.trajectory.valid):
        raise ValueError("scene_export ball_3d_valid disagrees with trajectory.valid")
    if not np.allclose(scene["ball_3d"], ball.trajectory.positions, atol=1e-6, rtol=0):
        raise ValueError("scene_export ball_3d disagrees with trajectory.positions")


def load_rays_input(store: Path) -> RaysInput:
    """Load adopted artifacts through StoreReader; no component store writes."""
    index = store / "scene.json" if store.is_dir() else store
    reader = StoreReader(index)
    alignment, _ = _payload(reader, "camera_alignment", "aligned_cameras")
    saved_ball, _ = _payload(reader, "ball_triangulation", "ball_trajectory")
    geometry = alignment["geometry"]
    if geometry is None or saved_ball["ball"] is None:
        raise ValueError("Store lacks aligned cameras or ball reconstruction")
    cameras = tuple(PinholeCamera(**value) for value in geometry["cameras"])
    if (
        tuple(camera.camera_id for camera in cameras) != reader.camera_ids
        or tuple(geometry["camera_ids"]) != reader.camera_ids
    ):
        raise ValueError("Aligned camera order differs from source camera order")
    value = saved_ball["ball"]
    ball = BallReconstructionResult(
        value["uv_px"],
        value["visibility"],
        TriangulatedPoints(**value["trajectory"]),
        value["status"],
    )
    validate_ball(ball, len(cameras))
    if len(ball.trajectory.valid) != reader.frame_count:
        raise ValueError("Ball frame count differs from source")
    scene_export = None
    if "scene" in reader.document["exports"]:
        scene_path = indexed_scene_path(reader.index_path)
        with np.load(scene_path, allow_pickle=False) as scene:
            validate_scene_ball(
                ball, {name: scene[name] for name in ("ball_3d", "ball_3d_valid")}
            )
        scene_export = reader.document["exports"]["scene"]
    videos = reader.source["videos"]
    fps = float(videos[0]["fps"])
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError("Source FPS must be finite and in [1, 240]")
    for video in videos:
        if (
            (video["width"], video["height"]) != reader.size
            or video["num_frames"] != reader.frame_count
            or not math.isclose(video["fps"], fps, rel_tol=1e-6)
        ):
            raise ValueError("Source cameras must have the same dimensions/timebase")
    # Capture every transitive input, including the ball_points producer. The
    # reader has already checked dependency references against the adopted index.
    by_id = {ref["artifact_id"]: node for node, ref in reader.references.items()}
    nodes: set[str] = set()

    def visit(node: str) -> None:
        if node in nodes:
            return
        nodes.add(node)
        for dependency in reader.descriptor(node)["dependencies"].values():
            visit(by_id[dependency["artifact_id"]])

    visit("camera_alignment")
    visit("ball_triangulation")
    artifacts = {
        node: {
            "reference": reader.references[node],
            "descriptor": reader.descriptor(node),
        }
        for node in sorted(nodes)
    }
    provenance = {
        "store_index": str(reader.index_path),
        "store_index_sha256": dual_sha256(reader.index_path),
        "source_sha256": reader.source_sha256,
        "source": reader.source,
        "artifacts": artifacts,
        "scene_export": scene_export,
    }
    ball_sources = {}
    for node in sorted(nodes):
        descriptor = reader.descriptor(node)
        if (
            node.startswith("ball_detection/")
            and descriptor["output_schema"] == "ball_detections"
        ):
            origin = descriptor["provenance"]["origin"]
            if origin == "import":
                if descriptor["provenance"]["importer"] != "video_ball_annotation_v2":
                    raise ValueError(f"Unsupported ball annotation importer: {node}")
                kind = "imported_annotations"
            elif origin == "component":
                kind = "detector"
            else:
                raise ValueError(f"Unknown ball origin for {node}: {origin}")
            ball_sources[node] = {
                "artifact_id": descriptor["artifact_id"],
                "kind": kind,
                "output_schema": descriptor["output_schema"],
                "output_version": descriptor["output_version"],
                "provenance": descriptor["provenance"],
                "identity": descriptor["identity"],
            }
    if not ball_sources:
        raise ValueError("No upstream ball detector/import provenance found")
    return RaysInput(
        reader, cameras, ball, fps, provenance, {"artifacts": ball_sources}
    )


def _document(
    inputs: RaysInput, config: RaysConfig, resolved_config: dict[str, Any]
) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "provenance": inputs.provenance,
        "ball_source": inputs.ball_source,
        "git_commit": commit,
        "resolved_config": resolved_config,
        "origins": {
            "observations": "measured",
            "reconstruction": "measured",
            "rays_and_frusta": "schematic",
        },
        "fallback_ray_length_m": config.fallback_ray_length_m,
    }


def export_ranking(
    inputs: RaysInput,
    output: Path,
    *,
    config: RaysConfig,
    resolved_config: dict[str, Any],
) -> Path:
    """Write ranking.json without decoding source videos."""
    document = _document(inputs, config, resolved_config)
    document.update(
        schema="tennis_scene_blender_rays_ranking_v1",
        window_frames=config.window_frames,
        ranking_suppression=config.ranking_suppression,
        candidate_limit=10,
        t0_rule="nearest frame to center within middle third with >=3 inliers",
        candidates=rank_windows(
            inputs.ball,
            window_frames=config.window_frames,
            suppression=config.ranking_suppression,
        ),
    )
    document["returned_candidates"] = len(document["candidates"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    return output


def export_rays_bundle(
    inputs: RaysInput,
    output: Path,
    *,
    start: int,
    stop: int,
    t0: int,
    config: RaysConfig,
    resolved_config: dict[str, Any],
) -> Path:
    """Export [start, stop) into a new bundle; remove partial results on failure."""
    if output.exists():
        raise FileExistsError(f"Bundle output already exists: {output}")
    if not 0 <= start <= t0 < stop <= inputs.reader.frame_count:
        raise ValueError("Require 0 <= start <= t0 < stop <= source frame count")
    if (
        not inputs.ball.trajectory.valid[t0]
        or inputs.ball.trajectory.inliers[:, t0].sum() < 3
    ):
        raise ValueError("t0 must have a valid ball and at least three inlier views")
    arrays = ray_arrays(
        inputs.cameras, inputs.ball, fallback_ray_length_m=config.fallback_ray_length_m
    )
    manifest = _document(inputs, config, resolved_config)
    manifest.update(
        schema=RAYS_SCHEMA,
        fps=inputs.fps,
        frames=[start, stop],
        t0=t0,
        num_frames=stop - start,
        court=court_geometry(),
        cameras=[],
        ray_states={item.name: int(item) for item in RayState},
        ray_length_rules={item.name: int(item) for item in RayLengthRule},
    )
    output.mkdir(parents=True)
    try:
        for name, array in arrays.items():
            sliced = (
                array
                if name == "ray_origin"
                else (
                    array[start:stop]
                    if name.startswith("ball_")
                    else array[:, start:stop]
                )
            )
            np.save(output / f"{name}.npy", sliced, allow_pickle=False)
        for camera, video in zip(
            inputs.cameras, inputs.reader.source["videos"], strict=True
        ):
            if Path(camera.camera_id).name != camera.camera_id or camera.camera_id in (
                ".",
                "..",
            ):
                raise ValueError("Camera ID must be a safe directory name")
            path = inputs.reader.videos[camera.camera_id]
            info = probe_video_info(path)
            if (info.width, info.height, info.frame_count) != (
                *inputs.reader.size,
                inputs.reader.frame_count,
            ) or not math.isclose(info.fps, inputs.fps, rel_tol=1e-5):
                raise ValueError(f"Source video metadata differs from store: {path}")
            if dual_sha256(path) != video["sha256"]:
                raise ValueError(f"Source video SHA-256 differs from store: {path}")
            plate_dir = f"plates/{camera.camera_id}"
            _write_plate(
                path,
                output / plate_dir,
                frame_range=range(start, stop),
                max_width=config.plate_max_width,
                quality=config.plate_jpeg_quality,
            )
            manifest["cameras"].append(
                {
                    "id": camera.camera_id,
                    "K": camera.intrinsic.tolist(),
                    "R": camera.rotation.tolist(),
                    "t": camera.translation.tolist(),
                    "center": camera.center.tolist(),
                    "width": info.width,
                    "height": info.height,
                    "plate_dir": plate_dir,
                }
            )
        target = output / "manifest.json"
        target.write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return target
