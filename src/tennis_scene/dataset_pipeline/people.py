"""Auditable singles-player observations, selected separately in each court half."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast
from zipfile import BadZipFile

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.submodules.models.tracker.common import (
    TrackRequest,
    TrackResult,
    select_and_complete_tracks,
    stitch_single_subject_tracklets,
)
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.reference_pipeline.observations import read_clip, sha256
from src.utils.io import save_json_atomic
from src.utils.video.reader import OpenCVVideoFrameReader


def _people_cache_settings(cfg: DictConfig) -> dict[str, Any]:
    """Normalize explicit/default error to the original schema-2 settings identity."""
    settings = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    if "long_gap_policy" in settings and settings["long_gap_policy"] == "error":
        del settings["long_gap_policy"]
    return settings


def pose_support_mask(observed: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Support detector frames and short interior gaps, never held boundaries."""
    supported = observed.copy()
    indices = np.flatnonzero(observed)
    for left, right in zip(indices[:-1], indices[1:], strict=True):
        if right - left - 1 <= max_gap_frames:
            supported[left : right + 1] = True
    return supported


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Publish complete archives; a missing receipt still fails validation."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, allow_pickle=False, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_court_halves(
    history: list[list[dict[str, Any]]],
    homography: np.ndarray,
    *,
    half_width: float,
    half_length: float,
    sample_indices: np.ndarray,
    min_coverage: float,
    max_gap_frames: int,
    long_gap_policy: str = "error",
    camera_id: str = "unknown",
) -> tuple[TrackResult, np.ndarray]:
    """Choose one subject per end; retain the exact detector/tracklet provenance.

    This policy is for singles clips without an end change. The largest box in
    each half is selected after the playing-area filter, including fragmented
    tracklets. Detector-free frames remain marked as interpolation.
    """
    if long_gap_policy not in {"error", "mask"}:
        raise ValueError(f"Invalid people.long_gap_policy: {long_gap_policy!r}")
    halves: list[list[list[dict[str, Any]]]] = [[[] for _ in history] for _ in range(2)]
    inverse = np.linalg.inv(homography)
    for frame_index, detections in enumerate(history):
        for detection in detections:
            box = np.asarray(detection["bbx_xyxy"], np.float32)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or (box[2:] <= box[:2]).any()
            ):
                raise ValueError("Person boxes must be finite positive xyxy boxes")
            foot = np.array([[[(box[0] + box[2]) / 2, box[3]]]], np.float32)
            x, y = cv2.perspectiveTransform(foot, inverse)[0, 0]
            if abs(x) <= half_width and 0.05 <= abs(y) <= half_length:
                halves[int(y > 0)][frame_index].append(detection)
    tracks, masks = {}, {}
    source_ids: np.ndarray = np.full((2, len(history)), -1, np.int64)
    for end, half in enumerate(halves):
        stitched = stitch_single_subject_tracklets(half)
        observed = np.array([bool(frame) for frame in stitched])
        coverage = float(observed[sample_indices].mean())
        observed_indices = np.flatnonzero(observed)
        # Include unsupported clip boundaries, not only interior gaps.
        gaps = np.diff(np.r_[-1, observed_indices, len(history)]) - 1
        if (
            not observed_indices.size
            or coverage < min_coverage
            or (long_gap_policy == "error" and gaps.max() > max_gap_frames)
        ):
            raise ValueError(
                f"{camera_id}: Court half {end} has insufficient person evidence: "
                f"sample coverage={coverage:.3f}, longest gap={gaps.max()} frames"
            )
        selected = select_and_complete_tracks(
            stitched, TrackRequest(Path("court-half"), 1, False), len(history)
        )
        tracks[end], masks[end] = selected.tracks[0], selected.observed_mask(0)
        for index, frame in enumerate(stitched):
            if frame:
                source_ids[end, index] = frame[0]["source_track_id"]
    return TrackResult(tracks, len(history), masks), source_ids


def _detections(
    cfg: DictConfig,
    paths: ReferenceClipPaths,
    video: Path,
    cache: Path,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.submodules.models.dino.person_detector import (
        DinoPersonDetector,
        PersonDetectionRequest,
    )

    identity = {
        "schema_version": 1,
        "video_sha256": sha256(video),
        "checkpoint_sha256": sha256(paths.dino_checkpoint),
        "confidence": float(cfg.confidence),
        "short_side": int(cfg.short_side),
        "max_long_side": int(cfg.max_long_side),
        "stride": int(cfg.detection_stride),
        "total_frames": total,
    }
    receipt = cache.with_suffix(".metadata.json")
    if cache.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError(f"Stale person detections: {cache}")
        try:
            with np.load(cache, allow_pickle=False) as saved:
                return (
                    saved["frame_indices"],
                    saved["offsets"],
                    saved["boxes"],
                    saved["scores"],
                )
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            raise ValueError(f"Invalid person detections: {cache}: {exc}") from exc
    indices = np.unique(
        np.r_[np.arange(0, total, int(cfg.detection_stride)), total - 1]
    )
    selected = set(indices.tolist())
    boxes, scores, offsets = [], [], [0]
    detector = DinoPersonDetector(
        paths.dino_checkpoint,
        paths.dino_repository,
        device=str(cfg.device),
        confidence=float(cfg.confidence),
        short_side=int(cfg.short_side),
        max_long_side=int(cfg.max_long_side),
    )
    decoded = 0
    try:
        progress = tqdm(total=len(indices), desc=f"DINO sampled {video.stem}")
        for frame_index, packet in enumerate(
            OpenCVVideoFrameReader(video, max_frames=total)
        ):
            decoded += 1
            if frame_index not in selected:
                continue
            result = detector.predict(PersonDetectionRequest(packet.frame))
            boxes.append(result.boxes_xyxy)
            scores.append(result.scores)
            offsets.append(offsets[-1] + len(result.scores))
            progress.update()
        progress.close()
    finally:
        detector.unload()
        torch.cuda.empty_cache()
    if decoded != total:
        raise ValueError(f"Decoded {decoded}/{total} frames: {video}")
    box_array, score_array = np.concatenate(boxes), np.concatenate(scores)
    offset_array = np.asarray(offsets, np.int64)
    _save_npz_atomic(
        cache,
        frame_indices=indices,
        offsets=offset_array,
        boxes=box_array,
        scores=score_array,
    )
    save_json_atomic(identity, receipt)
    return indices, offset_array, box_array, score_array


def observe_singles_people(
    cfg: DictConfig,
    paths: ReferenceClipPaths,
    clip_dir: Path,
    output: Path,
    *,
    homographies: np.ndarray,
) -> None:
    """Cache raw detections separately so selection changes never rerun DINO."""
    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models import Pose2DRequest, ViTPosePose2D

    clip = read_clip(clip_dir, minimum_views=1)
    settings = OmegaConf.create(OmegaConf.to_container(cfg.people, resolve=True))
    if not isinstance(settings, DictConfig):
        raise TypeError("people settings must be a mapping")
    settings.device = cfg.device
    for cam, relative, homography in zip(
        clip["camera_ids"], clip["video_paths"], homographies, strict=True
    ):
        video = (clip_dir / relative).resolve()
        cache = output / f"{cam}_people.npz"
        receipt = cache.with_suffix(".metadata.json")
        long_gap_policy = str(settings.get("long_gap_policy", "error"))
        identity = {
            "schema_version": 3 if long_gap_policy == "mask" else 2,
            "video_sha256": sha256(video),
            "detector_sha256": sha256(paths.dino_checkpoint),
            "pose_sha256": sha256(paths.vitpose_checkpoint),
            "settings": _people_cache_settings(cfg.people),
            "homography": homography.tolist(),
            "policy": "largest detected person per court half; singles without end changes",
        }
        if cache.exists():
            if json.loads(receipt.read_text()) != identity:
                raise ValueError(f"Stale person observations: {cache}")
            try:
                with np.load(cache, allow_pickle=False) as saved:
                    kp = saved["keypoints"]
                    if kp.shape != (2, clip["num_frames"], 17, 3):
                        raise ValueError(f"invalid keypoint shape {kp.shape}")
                    if long_gap_policy == "mask":
                        support = saved["pose_supported_mask"]
                        if support.shape != kp.shape[:2] or support.dtype != np.bool_:
                            raise ValueError("invalid pose_supported_mask")
                        if np.any(kp[..., 2][~support] != 0):
                            raise ValueError(
                                "unsupported poses have nonzero confidence"
                            )
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                raise ValueError(
                    f"Invalid person observations: {cache}: {exc}"
                ) from exc
            print(f"Using validated people {cache}", flush=True)
            continue
        indices, offsets, boxes, _ = _detections(
            settings, paths, video, output / f"{cam}_detections.npz", clip["num_frames"]
        )
        history: list[list[dict[str, Any]]] = [[] for _ in range(clip["num_frames"])]
        for sample, frame_index in enumerate(indices):
            history[frame_index] = [
                {"id": int(index), "bbx_xyxy": boxes[index]}
                for index in range(offsets[sample], offsets[sample + 1])
            ]
        # IDs identify actual raw detections, not invented persistent identities.
        tracks, source_ids = select_court_halves(
            history,
            homography,
            half_width=float(settings.court_half_width_m),
            half_length=float(settings.court_half_length_m),
            sample_indices=indices,
            min_coverage=float(settings.min_sample_coverage),
            max_gap_frames=int(float(settings.max_gap_seconds) * clip["fps"]),
            long_gap_policy=long_gap_policy,
            camera_id=cam,
        )
        pose = ViTPosePose2D(
            paths.vitpose_checkpoint,
            device=cfg.device,
            flip_test=True,
            batch_size=int(settings.batch_size),
            precision=cast(Literal["float32", "bfloat16"], settings.precision),
            head_config=ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ()),
        )
        try:
            keypoints = np.stack(
                [
                    pose.predict(
                        Pose2DRequest(video, tracks.bbx_xys(index, base_enlarge=1.2))
                    ).keypoints.numpy()
                    for index in tracks.track_ids
                ]
            )
        finally:
            pose.unload()
            torch.cuda.empty_cache()
        observed_masks = np.stack(
            [tracks.observed_mask(i).numpy() for i in tracks.track_ids]
        )
        supported = np.stack(
            [
                pose_support_mask(
                    mask, int(float(settings.max_gap_seconds) * clip["fps"])
                )
                for mask in observed_masks
            ]
        )
        keypoints[..., 2] = np.where(supported[..., None], keypoints[..., 2], 0)
        _save_npz_atomic(
            cache,
            boxes=np.stack([tracks.tracks[i].numpy() for i in tracks.track_ids]),
            track_ids=np.array(tracks.track_ids),
            keypoints=keypoints,
            observed_masks=observed_masks,
            pose_supported_mask=supported,
            source_detection_ids=source_ids,
            detection_frame_indices=indices,
        )
        save_json_atomic(identity, receipt)
        print(f"Saved court-half people {cache}", flush=True)
