"""Strict detector-only pilot assembly on the source video's actual PTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import av
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.model_io.contracts import BallCandidates
from src.tasks.ball_refiner.data.inputs import detector_only_sequence_input
from src.tasks.ball_refiner.data.temporal import window_owners, window_starts
from src.tasks.ball_refiner.deployment import InferenceBundle
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DInput
from src.tennis_scene.pipeline.contracts import AssemblyContext, SourceVideo
from src.utils.checksum import dual_sha256


def read_video_timeline(video: SourceVideo) -> tuple[NDArray[np.int64], str]:
    """Verify source bytes, size and presentation order; never infer missing PTS."""
    if dual_sha256(video.path) != video.sha256:
        raise ValueError("Refiner source video checksum mismatch")
    pts: list[int] = []
    with av.open(str(video.path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Refiner requires exactly one video stream")
        stream = container.streams.video[0]
        base = stream.time_base
        if base is None or base <= 0 or stream.average_rate is None or abs(float(stream.average_rate) - video.fps) > 1e-6:
            raise ValueError("Refiner source time base/FPS disagrees with declared video")
        for frame in container.decode(stream):
            if (frame.pts is None or frame.time_base != base
                    or (frame.width, frame.height) != (video.width, video.height)):
                raise ValueError("Refiner source frames require actual PTS and the declared image size")
            pts.append(frame.pts)
            if len(pts) == video.num_frames:
                break
    result = np.asarray(pts, dtype=np.int64)
    if len(result) != video.num_frames or (np.diff(result) <= 0).any():
        raise ValueError("Refiner decoded frame count/PTS disagrees with the source timeline")
    if dual_sha256(video.path) != video.sha256:
        raise ValueError("Refiner source video changed during timeline decoding")
    return result, str(base)


@dataclass(frozen=True)
class BallRefiner2DInputAssembler:
    bundle: InferenceBundle
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallRefiner2DInput:
        if context.camera_id is None or set(artifacts) != {"detections"}:
            raise ValueError("Refiner requires exactly its camera-local detection port")
        video = context.source.video(context.camera_id)
        detections = artifacts["detections"]
        if (not isinstance(detections, BallDetectionOutput) or detections.camera_id != video.camera_id
                or not np.array_equal(detections.frame_indices, np.arange(video.num_frames))):
            raise ValueError("Refiner detection camera/frame alignment mismatch")
        evidence = detections.evidence
        if evidence is None or detections.score_semantics != "model_score":
            raise ValueError("Refiner requires generated pre-gate detector evidence; no point fallback")
        required = self.bundle.detector
        if evidence.config != required.candidates or evidence.source_size_wh != (video.width, video.height):
            raise ValueError("Refiner detector candidate config/source size mismatch")
        starts = window_starts(video.num_frames, required.window_length, required.stride)
        expected = np.asarray(starts, dtype=np.int64)[window_owners(video.num_frames, starts, required.window_length)]
        if (not np.array_equal(evidence.selected_window_start, expected)
                or not np.array_equal(evidence.selected_time_index, np.arange(video.num_frames) - expected)):
            raise ValueError("Refiner detector windows do not match the training evidence policy")
        if min(video.width, video.height) <= 1:
            raise ValueError("Refiner source sizes must support endpoint normalization")
        pts, time_base = read_video_timeline(video)
        seconds = ((pts - pts[0]).astype(np.float64) * float(Fraction(time_base))).astype(np.float32)
        coords = evidence.candidate_uv_px / np.asarray([video.width - 1, video.height - 1], dtype=np.float32)
        candidates = BallCandidates(
            coords=torch.from_numpy(coords.copy())[None],
            scores=torch.from_numpy(evidence.candidate_scores.copy())[None],
            valid=torch.from_numpy(evidence.candidate_valid.copy())[None],
            cells=torch.from_numpy(evidence.candidate_cells.copy())[None],
            patches=torch.from_numpy(evidence.patches.copy())[None],
            patch_valid=torch.from_numpy(evidence.patch_valid.copy())[None], config=evidence.config,
        )
        return BallRefiner2DInput(
            video, pts, time_base,
            detector_only_sequence_input(candidates, torch.from_numpy(seconds)[None], self.bundle.model_config),
            evidence.selected_window_start.copy(), evidence.selected_time_index.copy(),
        )
