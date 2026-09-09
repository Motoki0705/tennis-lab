"""Validated external ball import and DINO/ViTPose observation caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.utils.schema.court import CourtConfig, court_keypoints_3d


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_clip(path: Path) -> dict[str, Any]:
    clip = json.loads((path / "clip.json").read_text())
    if not isinstance(clip, dict):
        raise ValueError("clip.json must be an object")
    if not 3 <= len(clip["camera_ids"]) <= 4 or len(set(clip["camera_ids"])) != len(
        clip["camera_ids"]
    ):
        raise ValueError("Reference reconstruction requires 3-4 unique camera IDs")
    if len(clip["video_paths"]) != len(clip["camera_ids"]):
        raise ValueError("Each camera must have exactly one video path")
    return cast(dict[str, Any], clip)


def import_ball(clip_dir: Path, output: Path) -> None:
    """Include observed, interpolated and occlusion-estimated coordinates explicitly."""
    clip = read_clip(clip_dir)
    n, t = len(clip["camera_ids"]), clip["num_frames"]
    px = np.zeros((n, t, 2), np.float32)
    valid = np.zeros((n, t), bool)
    states: list[list[str]] = []
    sources = {}
    for i, cam in enumerate(clip["camera_ids"]):
        source = clip_dir / "outsource" / f"{cam}_annotations.json"
        data = json.loads(source.read_text())
        if data["schema_version"] != "video_ball_annotation.v2":
            raise ValueError("Unsupported outsourced ball schema")
        video = clip_dir / clip["video_paths"][i]
        if sha256(video) != data["source"]["sha256"]:
            raise ValueError(f"Ball source video mismatch: {cam}")
        if (
            data["source"]["width"],
            data["source"]["height"],
            data["source"]["frame_count"],
        ) != (clip["width"], clip["height"], t):
            raise ValueError(f"Ball source dimensions mismatch: {cam}")
        frames = data["frames"]
        if [f["frame_index"] for f in frames] != list(range(t)):
            raise ValueError(f"Ball frame IDs missing/duplicated/unordered: {cam}")
        states.append([f["status"] for f in frames])
        for j, frame in enumerate(frames):
            status = frame["status"]
            if status not in {
                "observed",
                "interpolated",
                "occlusion_estimated",
                "unresolved",
            }:
                raise ValueError(f"Unknown ball status: {status}")
            if status == "unresolved":
                continue
            center = frame["center_px"]
            xy = np.asarray([center["x"], center["y"]], np.float32)
            if (
                not np.isfinite(xy).all()
                or (xy < 0).any()
                or (xy >= [clip["width"], clip["height"]]).any()
            ):
                raise ValueError(f"Invalid ball coordinate: {cam}:{j}")
            px[i, j], valid[i, j] = xy, True
        sources[cam] = {"path": str(source), "sha256": sha256(source)}
    uv = px / np.asarray([clip["width"], clip["height"]], np.float32)
    # Binary availability, not a fabricated detector probability.
    result = BallDetectionResult(
        uv.astype(np.float32), px, valid, valid.astype(np.float32)
    )
    ok, errors = result.validate()
    if not ok:
        raise ValueError(errors)
    result.save(output / "ball_detection_result.json")
    (output / "ball_import.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_ids": clip["camera_ids"],
                "sources": sources,
                "status": states,
                "score_semantics": "binary coordinate availability; not detector confidence",
                "included_statuses": [
                    "observed",
                    "interpolated",
                    "occlusion_estimated",
                ],
            },
            indent=2,
        )
    )


def court_homographies(clip_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ground homographies in each view's own near/far coordinates."""
    clip = read_clip(clip_dir)
    data = json.loads(
        (clip_dir / "annotations/manual_court_kp_result.json").read_text()
    )
    kp = np.asarray(data["keypoints"], np.float32)
    if (
        kp.shape != (len(clip["camera_ids"]), clip["num_frames"], 14, 2)
        or not np.isfinite(kp).all()
    ):
        raise ValueError("Court must contain finite (V,T,14,2) observations")
    visibility = np.asarray(data["visibility"])
    if visibility.shape != kp.shape[:-1] or not (visibility == 1).all():
        raise ValueError("This manual-court pipeline requires all 14 points visible")
    if ((kp < 0) | (kp > 1)).any():
        raise ValueError("Court coordinates must be normalized UV within [0, 1]")
    if data["frame_indices"] != list(range(clip["num_frames"])):
        raise ValueError("Court frame IDs do not match clip")
    if not np.allclose(kp, kp[:, :1]):
        raise ValueError(
            "This fixed-camera pipeline requires constant court observations"
        )
    world = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    matrices = []
    for camera in kp:
        h, _ = cv2.findHomography(world, camera[0] * [clip["width"], clip["height"]], 0)
        if h is None or not np.isfinite(h).all():
            raise ValueError("Cannot estimate court homography")
        matrices.append(h)
    return kp, np.asarray(matrices)


def observe_people(cfg: DictConfig, clip_dir: Path, output: Path) -> None:
    """Run DINO/BoT-SORT within the playing area, then ViTPose; cache each camera."""
    import torch

    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models import (
        DinoPersonTracker,
        Pose2DRequest,
        TrackRequest,
        ViTPosePose2D,
    )

    clip = read_clip(clip_dir)
    _, hs = court_homographies(clip_dir)
    head = ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ())
    for camera, video_name, h in zip(
        clip["camera_ids"], clip["video_paths"], hs, strict=True
    ):
        cache = output / f"{camera}_people.npz"
        video = (clip_dir / video_name).resolve()
        signature = {
            "video_sha256": sha256(video),
            "tracker": "DINO+BoT-SORT",
            "pose": "ViTPose-H",
            "settings": str(cfg.people),
        }
        if cache.exists():
            if json.loads(cache.with_suffix(".metadata.json").read_text()) != signature:
                raise ValueError(f"Stale observation cache: {cache}")
            print(f"Using validated cache {cache}", flush=True)
            continue
        track_cache = output / f"{camera}_tracks.npz"
        if track_cache.exists():
            if (
                json.loads(track_cache.with_suffix(".metadata.json").read_text())
                != signature
            ):
                raise ValueError(f"Stale track cache: {track_cache}")
            with np.load(track_cache) as saved:
                boxes = saved["boxes"]
                ids = saved["track_ids"]
        else:
            roi_xy = np.array([[-7, -18], [7, -18], [7, 18], [-7, 18]], np.float32)
            polygon = cv2.perspectiveTransform(roi_xy[None], h)[0]
            tracker = DinoPersonTracker(
                Path(cfg.people.dino_checkpoint),
                Path(cfg.people.dino_repository),
                device=cfg.device,
                confidence=float(cfg.people.confidence),
                short_side=int(cfg.people.short_side),
                max_long_side=int(cfg.people.max_long_side),
            )
            tracks = tracker.predict(
                TrackRequest(video, 2, False, tuple(map(tuple, polygon.tolist())))
            )
            ids = np.array(tracks.track_ids)
            boxes = np.stack([tracks.tracks[int(i)].numpy() for i in ids])
            np.savez_compressed(track_cache, boxes=boxes, track_ids=ids)
            track_cache.with_suffix(".metadata.json").write_text(json.dumps(signature))
            tracker.unload()
            del tracker
            torch.cuda.empty_cache()
        from src.submodules.models import TrackResult

        tracks = TrackResult(
            {int(i): torch.from_numpy(b) for i, b in zip(ids, boxes, strict=True)},
            clip["num_frames"],
        )
        pose = ViTPosePose2D(
            Path(cfg.people.vitpose_checkpoint),
            device=cfg.device,
            flip_test=True,
            batch_size=int(cfg.people.batch_size),
            head_config=head,
        )
        keypoints = np.stack(
            [
                pose.predict(
                    Pose2DRequest(video, tracks.bbx_xys(int(i), base_enlarge=1.2))
                ).keypoints.numpy()
                for i in ids
            ]
        )
        np.savez_compressed(cache, boxes=boxes, track_ids=ids, keypoints=keypoints)
        cache.with_suffix(".metadata.json").write_text(json.dumps(signature))
        pose.unload()
        del pose
        torch.cuda.empty_cache()
        print(f"Saved {cache}", flush=True)
