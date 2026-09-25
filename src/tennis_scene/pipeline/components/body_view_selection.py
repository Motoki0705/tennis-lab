"""An independently replaceable policy selecting camera and segments for a body."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.tennis_scene.pipeline.utilts.timeline import association_frame_indices


@dataclass(frozen=True)
class BodyViewSelectionInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    raw: ObjectObservations
    grouped: GroupedObservations


@dataclass(frozen=True)
class BodySelection:
    person_id: int
    camera_id: str
    requests: tuple[BodyRecoveryRequest, ...]
    observed_samples: tuple[int, ...]


@dataclass(frozen=True)
class BodyViewSelectionOutput:
    selections: tuple[BodySelection, ...]


def supported_segments(support: NDArray[np.bool_], source_frames: NDArray[np.int64],
                       observed_any: NDArray[np.bool_], fps: float) -> list[NDArray[np.int64]]:
    active = np.flatnonzero(support)
    if not len(active):
        return []
    starts = [0]
    for index in range(1, len(active)):
        previous, current = int(active[index - 1]), int(active[index])
        missing = np.arange(previous + 1, current)
        ambiguous = bool(observed_any[source_frames[missing]].any())
        gap = (source_frames[current] - source_frames[previous]) / fps
        if missing.size and (ambiguous or gap > .1):
            starts.append(index)
    ends = [*starts[1:], len(active)]
    return [np.arange(active[start], active[end - 1] + 1, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True) if end - start >= 2]


class BodyViewSelectionModule:
    def __init__(self, camera_ids: tuple[str, ...], *, max_frames: int, enabled: bool = True) -> None:
        self.max_frames, self.enabled = max_frames, enabled
        self.io = ComponentIO("body_view_selection", BodyViewSelectionInput, BodyViewSelectionOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             "identities": InputPort("person_identities"), **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}}, "body_view_selection")

    def process(self, inputs: BodyViewSelectionInput) -> BodyViewSelectionOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled or not len(inputs.grouped.identities):
            return BodyViewSelectionOutput(())
        raw, grouped = inputs.raw, inputs.grouped
        if raw.boxes_xys is None:
            raise ValueError("Body selection requires source track boxes")
        sample = association_frame_indices(inputs.source.num_frames, inputs.source.fps, max_frames=self.max_frames)
        selections: list[BodySelection] = []
        for player, identity in enumerate(grouped.identities):
            coverage = grouped.visibility[player].any(-1).sum(-1)
            scores = grouped.confidence[player].mean(axis=(1, 2))
            view = min(range(len(geometry.cameras)), key=lambda v: (-int(coverage[v]), -float(scores[v]), geometry.cameras[v].camera_id))
            camera = geometry.cameras[view]
            rows = grouped.raw_indices[player, view, sample]
            support = rows >= 0
            requests: list[BodyRecoveryRequest] = []
            counts: list[int] = []
            for segment in supported_segments(support, sample, raw.observed[view].any(-1), raw.fps):
                source = sample[segment]
                present = support[segment]
                original = rows[segment[present]]
                actual = raw.boxes_xys[view, source[present], original]
                boxes = np.stack([np.interp(source, source[present], actual[:, c]) for c in range(3)], -1).astype(np.float32)
                kp: NDArray[np.float32] = np.zeros((len(segment), 17, 3), np.float32)
                kp[present, :, :2] = raw.uv_px[view, source[present], original]
                kp[present, :, 2] = np.where(grouped.visibility[player, view, source[present]], raw.confidence[view, source[present], original], 0)
                requests.append(BodyRecoveryRequest(inputs.source.video(camera.camera_id).path, source, kp, boxes, raw.size, camera.intrinsic))
                counts.append(int(present.sum()))
            selections.append(BodySelection(int(identity), camera.camera_id, tuple(requests), tuple(counts)))
        return BodyViewSelectionOutput(tuple(selections))
