"""CPU-only diagnostic preparation, independent from common-H estimation."""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tasks.court_detection.geometry.video_drift import (
    DriftConfig,
    inspect_frame,
    summarize_clip,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.utils.checksum import dual_sha256
from src.utils.video import read_video_frame

from .diagnostics import DiagnosedVideoCourt, publish_video_drift
from .store import VideoCourtArtifact


def _measure(job: tuple[Path, int, dict[str, Any], DriftConfig]) -> dict[str, Any]:
    path, index, camera, config = job
    cv2.setNumThreads(1)
    packet = read_video_frame(path, index)
    if list(packet.original_size) != camera['size_wh']:
        raise ValueError('Drift source resolution changed')
    rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
    report = inspect_frame(rgb, np.asarray(camera['matrix'], float),
                           tuple(camera['diagnostics']['quality']['supported_lines']), config)
    return {'frame_index': index, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), **report}


def build_video_drift(base: VideoCourtArtifact, sources: tuple[ClipSource, ...], *,
                      frame_plan: dict[str, dict[str, list[int]]], directory: Path,
                      provenance: dict[str, Any], config: DriftConfig, workers: int = 1) -> DiagnosedVideoCourt:
    """Measure exactly the declared frames, publish worst-frame clip reports.

    Callers own sample/exclusion selection. Every camera/clip must be covered;
    missing media, source mismatch or an incomplete plan raises before publish.
    The base bundle is read and verified only; no court estimator is invoked.
    """
    base.verify()
    if workers < 1 or type(workers) is not int:
        raise ValueError('Drift workers must be a positive integer')
    clips = {source.clip_id: source for source in sources}
    if len(clips) != len(sources) or not clips or set(frame_plan) != set(base.cameras):
        raise ValueError('Drift input scope is empty, duplicate or missing cameras')
    for source in sources:
        base.bind(source)
        if set(source.camera_ids) != set(base.cameras):
            raise ValueError('Drift source must contain all shared cameras')
    jobs, keys = [], []
    inputs: dict[Path, str] = {}
    for camera_id, camera in sorted(base.cameras.items()):
        if camera['status'] != 'accepted':
            raise ValueError('Cannot inspect drift without an accepted common H')
        if set(frame_plan[camera_id]) != set(camera['sources']) or set(clips) != set(camera['sources']):
            raise ValueError('Drift frame plan must cover every shared-court clip')
        for clip, indices in sorted(frame_plan[camera_id].items()):
            video = clips[clip].video(camera_id)
            if (not indices or len(set(indices)) != len(indices)
                    or any(type(index) is not int or not 0 <= index < video.num_frames for index in indices)):
                raise ValueError('Drift plan contains empty, duplicate or invalid frame indices')
            if dual_sha256(video.path) != video.sha256:
                raise ValueError('Drift source media content changed')
            inputs[video.path] = video.sha256
            for index in sorted(indices):
                keys.append((camera_id, clip))
                jobs.append((video.path, index, camera, config))
    if workers == 1:
        frames = list(map(_measure, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_measure, jobs))
    if any(dual_sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError('Drift source inputs changed during measurement')
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {camera: {clip: [] for clip in clips} for camera in base.cameras}
    for (camera, clip), report in zip(keys, frames, strict=True):
        grouped[camera][clip].append(report)
    reports = {camera: {clip: summarize_clip(values) for clip, values in groups.items()} for camera, groups in grouped.items()}
    return publish_video_drift(base, directory, protocol={'settings': asdict(config), 'frame_plan': frame_plan,
        'provenance': provenance, 'temporal_pooling': False, 'common_h_refit': False}, reports=reports)
