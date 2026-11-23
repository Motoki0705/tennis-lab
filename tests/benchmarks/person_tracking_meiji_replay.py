"""Replay the supported run 8–10 trackers on the full source grid."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import person_tracking_meiji_features as appearance  # type: ignore[import-not-found]
from person_tracking_linking import (  # type: ignore[import-not-found]
    lab_relink,
    merge_tracks,
)
from person_tracking_meiji_inputs import (  # type: ignore[import-not-found]
    CLIP,
    KPR,
    SOLIDER,
    feature_path,
    load_detections,
    verify_digest,
    write_npz,
)

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.feature_tracks import scatter_tracks
from src.tasks.person_tracking.methods import build_tracker
from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort import InvalidPrediction
from src.tasks.person_tracking.strongsort_offline import (
    AFLink,
    ReconstructedTracks,
    gaussian_interpolation,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic

BASELINE = "saved_default"
DEEP = f"deep_ocsort_pose__{CLIP}"
LAB = f"deep_ocsort_pose_lab__{CLIP}"
STRONG = f"strongsort_pp__{CLIP}"
DEFAULT = f"strongsort_pp_pose__{CLIP}"
HYBRID = f"deep_ocsort_pose_aflink_gsi__{CLIP}"
HISTORICAL = (DEEP, f"deep_ocsort_pose__{SOLIDER}", LAB, STRONG, f"strongsort_pp__{KPR}", DEFAULT, HYBRID)
CLIP_TRACKERS = (DEEP, LAB, STRONG, DEFAULT, HYBRID)
NATIVE_ELIGIBLE = (STRONG, DEEP, LAB)


def track_path(report: Path, variant: str, clip: str, camera: str) -> Path:
    return report / "tracks" / variant / clip / f"{camera}.npz"


def read_tracks(path: Path, camera: PinholeCamera, size: tuple[int, int]) -> tuple[CameraTracks, np.ndarray, dict[str, Any]]:
    receipt = json.loads(path.with_suffix(".json").read_text())
    with np.load(verify_digest(path, receipt["sha256"]), allow_pickle=False) as saved:
        tracks = CameraTracks(camera, size, saved["track_ids"], saved["boxes"], saved["observed"])
        origins = saved["origins"]
        if not np.array_equal(origins >= 0, tracks.observed):
            raise ValueError(f"Track origin and observed mask disagree: {path}")
    return tracks, origins, receipt


def _online(method: str, fps: float, frames: list[DetectionFeatures],
            camera: PinholeCamera, size: tuple[int, int]) -> tuple[CameraTracks, np.ndarray]:
    tracker, repeat = (build_tracker(method, fps=fps) for _ in range(2))
    assignments = []
    for frame in frames:
        output, reference = tracker.update(frame), repeat.update(frame)
        np.testing.assert_array_equal(output.track_ids, reference.track_ids)
        np.testing.assert_array_equal(output.detection_rows, reference.detection_rows)
        assignments.append(output)
    result: tuple[CameraTracks, np.ndarray] = scatter_tracks(camera, frames, assignments, size)
    return result


def _replay(
    variant: str, frames: list[DetectionFeatures], frozen: FrozenClip, camera: PinholeCamera,
    aflink: AFLink, online_deep: tuple[CameraTracks, np.ndarray] | None, *, saved_clip: bool,
) -> tuple[CameraTracks, np.ndarray, ReconstructedTracks | None]:
    fps, size = frozen.clip.fps, (frozen.clip.width, frozen.clip.height)
    method, _ = variant.split("__")
    if variant == DEFAULT:
        result = track_sequence(frames, fps=fps, config=TrackingConfig(), aflink=aflink)
        tracks = CameraTracks(camera, size, result.track_ids, result.boxes, result.observed)
        if saved_clip:
            saved = frozen.raw(camera.camera_id, rows=True)
            for reproduced, field in ((tracks.track_ids, "track_ids"), (tracks.boxes_xyxy, "boxes_xyxy"),
                                      (tracks.observed, "observed"), (result.evidence.detection_rows, "evidence.detection_rows")):
                np.testing.assert_array_equal(reproduced, saved[field])
        return tracks, result.evidence.detection_rows, result.reconstruction
    raw, origins = online_deep if online_deep is not None else _online(
        "strongsort" if method == "strongsort_pp" else "deep_ocsort_pose", fps, frames, camera, size)
    if method == "deep_ocsort_pose_lab":
        media = frozen.clip.media_path(camera.camera_id)
        linked, rows, _ = lab_relink(raw, origins, {"path": str(media), "sha256": frozen.media_sha256[camera.camera_id]})
        return linked, rows, None
    if method in ("strongsort_pp", "deep_ocsort_pose_aflink_gsi"):
        roots, _ = aflink.links(raw.boxes_xyxy, raw.observed)
        raw, origins = merge_tracks(raw, origins, roots)
        return raw, origins, gaussian_interpolation(raw.boxes_xyxy, raw.observed)
    return raw, origins, None


def replay_clip(
    frozen: FrozenRun, clip: FrozenClip, adopted: FrozenClip, variants: tuple[str, ...],
    report: Path, aflink: AFLink, identity: str, *, saved_clip: bool,
) -> None:
    size = (clip.clip.width, clip.clip.height)
    local = clip.local_cameras()
    for name in clip.clip.camera_ids:
        camera = local[name]
        data = load_detections(adopted, name)
        encoders: dict[str, list[DetectionFeatures]] = {}
        online_deep = None
        for variant in variants:
            _, encoder = variant.split("__")
            target = track_path(report, variant, clip.clip.clip_id, name)
            generated_features = feature_path(report, encoder, clip.clip.clip_id, name)
            feature = (adopted.store.references[f"person_appearance_features/{name}"]["sha256"] if encoder == CLIP and saved_clip
                       else dual_sha256(generated_features.with_suffix(".json")))
            inputs = {"comparison_identity": identity, "variant": variant, "camera": name, "feature": feature,
                      "scene": dual_sha256(adopted.store.index_path)}
            if target.with_suffix(".json").exists():
                raw, rows, receipt = read_tracks(target, camera, size)
                if receipt["input"] != inputs:
                    raise ValueError(f"Tracking input changed: {target}")
                if variant == DEEP and receipt["stop_reason"] is None:
                    online_deep = raw, rows
                continue
            if encoder not in encoders:
                encoders[encoder] = appearance.load_encoder_frames(frozen, adopted, name, data, report, encoder, saved_clip=saved_clip)
            start = time.perf_counter()
            stop = None
            reconstruction = None
            try:
                reused = online_deep if variant in (LAB, HYBRID) else None
                raw, rows, reconstruction = _replay(variant, encoders[encoder], clip, camera, aflink, reused, saved_clip=saved_clip)
                if variant == DEEP:
                    online_deep = raw, rows
            except (ReconstructionUnavailable, InvalidPrediction) as error:
                stop = f"{type(error).__name__}: {error}"
                shape = (0, clip.clip.num_frames)
                raw = CameraTracks(camera, size, np.empty(0, np.int64), np.empty((*shape, 4), np.float32), np.empty(shape, bool))
                rows = np.full(shape, -1, np.int64)
            arrays = {"track_ids": raw.track_ids, "boxes": raw.boxes_xyxy, "observed": raw.observed, "origins": rows}
            if reconstruction is not None:
                arrays.update(gsi_boxes=reconstruction.boxes, interpolated=reconstruction.interpolated)
            write_npz(target, **arrays)
            write_json_atomic(target.with_suffix(".json"), {
                "input": inputs, "sha256": dual_sha256(target), "stop_reason": stop,
                "track_count": len(raw.track_ids), "observations": int(raw.observed.sum()), "elapsed_s": time.perf_counter() - start,
            })
            print(f"{variant} {clip.clip.clip_id}/{name}: {stop or len(raw.track_ids)}", flush=True)
