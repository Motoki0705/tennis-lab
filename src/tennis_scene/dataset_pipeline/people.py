"""Auditable singles-player observations, selected separately in each court half."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from src.tennis_scene.dataset_pipeline.checkpoint_integrity import (
    validate_checkpoint_sha256,
)
from src.tennis_scene.dataset_pipeline.person_association import (
    PersonAssociation,
    associate_single_person,
    association_settings,
)
from src.tennis_scene.reference_pipeline.observations import read_clip, sha256
from src.utils.checksum import FileIntegrityError
from src.utils.io import save_json_atomic
from src.utils.video.reader import OpenCVVideoFrameReader


def _require_digest(actual: object, expected: str, *, path: Path, role: str) -> None:
    if actual != expected:
        raise FileIntegrityError(
            f"Checkpoint SHA-256 mismatch for {role}",
            details={"path": str(path), "expected": expected, "actual": actual},
        )


def _checkpoint_digest(
    path: Path, role: str, pins: Mapping[str, str] | None, *, before: str | None = None
) -> str:
    actual: str = sha256(path)
    if pins is not None:
        _require_digest(actual, pins[role], path=path, role=role)
    if before is not None:
        _require_digest(actual, before, path=path, role=f"{role} pre/post")
    return actual


def _read_checkpoint_receipt(receipt: Path) -> dict[str, Any]:
    try:
        saved = json.loads(receipt.read_text())
    except (OSError, ValueError) as exc:
        raise FileIntegrityError(
            "Cannot verify checkpoint receipt",
            details={"path": str(receipt), "error": str(exc)},
        ) from exc
    if not isinstance(saved, dict):
        raise FileIntegrityError(
            "Checkpoint receipt must be a JSON object", details={"path": str(receipt)}
        )
    return cast(dict[str, Any], saved)


def _validate_receipt_digests(receipt: Path, expected: Mapping[str, str]) -> None:
    saved = _read_checkpoint_receipt(receipt)
    for field, digest in expected.items():
        _require_digest(
            saved.get(field),
            digest,
            path=receipt,
            role=field,
        )


def validate_people_receipts(
    camera_ids: Sequence[str],
    observations: Path,
    *,
    checkpoint_sha256: Mapping[str, str] | None = None,
) -> None:
    """Authenticate stored producer digests before inference without loading models."""
    pins = (
        validate_checkpoint_sha256(checkpoint_sha256)
        if checkpoint_sha256 is not None
        else None
    )
    for camera in camera_ids:
        receipt = observations / f"{camera}_people.metadata.json"
        saved = _read_checkpoint_receipt(receipt)
        for field, role in (("detector_sha256", "dino"), ("pose_sha256", "vitpose")):
            digest = saved.get(field)
            if not isinstance(digest, str) or not digest:
                raise FileIntegrityError(
                    "Missing checkpoint digest in people receipt",
                    details={"path": str(receipt), "field": field, "actual": digest},
                )
            if pins is not None:
                _require_digest(digest, pins[role], path=receipt, role=role)
        _validate_receipt_digests(
            observations / f"{camera}_detections.metadata.json",
            {"checkpoint_sha256": saved["detector_sha256"]},
        )


def _validate_cache_identity(
    receipt: Path, expected: dict[str, Any], *, cache: Path, prefix: str
) -> None:
    """Keep strict receipt equality and expose the precise reason for a mismatch."""
    try:
        saved = json.loads(receipt.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{prefix}: {cache}; invalid receipt JSON: {exc}") from exc
    if not isinstance(saved, dict):
        raise ValueError(
            f"{prefix}: {cache}; receipt must be a JSON object, got {type(saved).__name__}"
        )
    if saved == expected:
        return

    changes: list[str] = []

    def compare(old: dict[str, Any], new: dict[str, Any], path: str = "") -> None:
        for key in sorted(old.keys() | new.keys()):
            field = f"{path}.{key}" if path else key
            if key not in old:
                changes.append(f"{field}: saved=<missing>, expected={new[key]!r}")
            elif key not in new:
                changes.append(f"{field}: saved={old[key]!r}, expected=<missing>")
            elif isinstance(old[key], dict) and isinstance(new[key], dict):
                compare(old[key], new[key], field)
            elif old[key] != new[key]:
                changes.append(f"{field}: saved={old[key]!r}, expected={new[key]!r}")

    compare(saved, expected)
    raise ValueError(
        f"{prefix}: {cache}; changed identity fields: {'; '.join(changes)}"
    )


def _people_cache_settings(cfg: DictConfig) -> dict[str, Any]:
    """Normalize explicit/default error to the original schema-2 settings identity."""
    settings = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    association_settings(settings)
    if settings.get("selection_policy") == "largest":
        del settings["selection_policy"]
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
    association: PersonAssociation | None = None,
) -> tuple[TrackResult, np.ndarray]:
    """Choose one subject per end; retain the exact detector/tracklet provenance.

    Singles clips must have no end change. Legacy selection uses largest boxes;
    explicit association retains a spatially consistent incumbent per half.
    Detector-free frames remain marked as interpolation.
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
        stitched = (
            stitch_single_subject_tracklets(half)
            if association is None
            else associate_single_person(half, association)
        )
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
    *,
    checkpoint_sha256: Mapping[str, str] | None = None,
    detector_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.submodules.models.dino.person_detector import (
        DinoPersonDetector,
        PersonDetectionRequest,
    )

    pins = (
        validate_checkpoint_sha256(checkpoint_sha256)
        if checkpoint_sha256 is not None
        else None
    )
    detector_digest = _checkpoint_digest(
        paths.dino_checkpoint, "dino", pins, before=detector_sha256
    )
    identity = {
        "schema_version": 1,
        "video_sha256": sha256(video),
        "checkpoint_sha256": detector_digest,
        "confidence": float(cfg.confidence),
        "short_side": int(cfg.short_side),
        "max_long_side": int(cfg.max_long_side),
        "stride": int(cfg.detection_stride),
        "total_frames": total,
    }
    receipt = cache.with_suffix(".metadata.json")
    if cache.exists():
        _validate_receipt_digests(receipt, {"checkpoint_sha256": detector_digest})
        _validate_cache_identity(
            receipt, identity, cache=cache, prefix="Stale person detections"
        )
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
    _checkpoint_digest(paths.dino_checkpoint, "dino", pins, before=detector_digest)
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
    checkpoint_sha256: Mapping[str, str] | None = None,
) -> None:
    """Cache raw detections separately so selection changes never rerun DINO.

    Pins authenticate each camera's checkpoint reads. Without pins, matching
    pre/post reads and sibling receipts still prevent mixed provenance. These
    boundary checks cannot detect a checkpoint changed and restored between
    reads; they do not lock files or authenticate the model's in-memory state.
    """
    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models import Pose2DRequest, ViTPosePose2D

    pins = (
        validate_checkpoint_sha256(checkpoint_sha256)
        if checkpoint_sha256 is not None
        else None
    )
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
        cache_settings = _people_cache_settings(cfg.people)
        association = association_settings(cache_settings)
        detector_digest = _checkpoint_digest(paths.dino_checkpoint, "dino", pins)
        pose_digest = _checkpoint_digest(paths.vitpose_checkpoint, "vitpose", pins)
        raw_cache = output / f"{cam}_detections.npz"
        raw_receipt = raw_cache.with_suffix(".metadata.json")
        identity = {
            "schema_version": 4
            if association is not None
            else (3 if long_gap_policy == "mask" else 2),
            "video_sha256": sha256(video),
            "detector_sha256": detector_digest,
            "pose_sha256": pose_digest,
            "settings": cache_settings,
            "homography": homography.tolist(),
            "policy": (
                "temporal_continuity_v2; largest seed; fixed IoU and incumbent-diagonal center gates; bounded median velocity prediction; no gap reset"
                if association is not None
                else "largest detected person per court half; singles without end changes"
            ),
        }
        if cache.exists():
            _validate_receipt_digests(
                receipt,
                {"detector_sha256": detector_digest, "pose_sha256": pose_digest},
            )
            _validate_receipt_digests(
                raw_receipt, {"checkpoint_sha256": detector_digest}
            )
            _validate_cache_identity(
                receipt, identity, cache=cache, prefix="Stale person observations"
            )
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
            settings,
            paths,
            video,
            raw_cache,
            clip["num_frames"],
            checkpoint_sha256=pins,
            detector_sha256=detector_digest,
        )
        _validate_receipt_digests(raw_receipt, {"checkpoint_sha256": detector_digest})
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
            association=association,
        )
        _checkpoint_digest(
            paths.vitpose_checkpoint, "vitpose", pins, before=pose_digest
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
        _checkpoint_digest(paths.dino_checkpoint, "dino", pins, before=detector_digest)
        _checkpoint_digest(
            paths.vitpose_checkpoint, "vitpose", pins, before=pose_digest
        )
        _validate_receipt_digests(raw_receipt, {"checkpoint_sha256": detector_digest})
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
