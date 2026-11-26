"""The four fixed candidate-selection conditions, with one real-observation contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    exclusive_region,
    select_linked_candidates,
)
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.utils.geometry.triangulation import PinholeCamera

CONDITIONS = ("all_tracks", "image_2d", "court_raw_dwell", "court_standard")
LINKING = LinkingConfig()
IMAGE_HEIGHT_SCALE = 0.15


def rank_candidates(
    track_ids: NDArray[np.int64],
    dwell: NDArray[np.int64],
    scores: NDArray[np.float64],
    required: float,
    cap: int,
) -> NDArray[np.int64]:
    eligible = np.flatnonzero(dwell >= required)
    order = np.lexsort((track_ids[eligible], -scores[eligible]))
    return eligible[order[:cap]]


def image_candidates(
    raw: PersonTrackingOutput, size: tuple[int, int], cfg: LinkingConfig
) -> NDArray[np.int64]:
    width, height = size
    dwell = raw.observed.sum(axis=1)
    scores: NDArray[np.float64] = np.zeros(len(raw.track_ids), np.float64)
    required = raw.observed.shape[1] * cfg.min_presence_fraction
    for i in np.flatnonzero(dwell >= required):
        boxes = raw.boxes_xyxy[i, raw.observed[i]].astype(np.float64)
        center_x = np.median((boxes[:, 0] + boxes[:, 2]) / 2)
        box_height = np.median(boxes[:, 3] - boxes[:, 1])
        centrality = np.clip(1 - abs(2 * center_x / width - 1), 0, 1)
        relative_height = np.clip(box_height / (IMAGE_HEIGHT_SCALE * height), 0, 1)
        # Using the frame count gives the same order as the observation fraction.
        scores[i] = dwell[i] * (0.5 + 0.5 * centrality) * (0.5 + 0.5 * relative_height)
    return rank_candidates(raw.track_ids, dwell, scores, required, cfg.max_candidates)


def selected_rows(
    raw: PersonTrackingOutput, rows: NDArray[np.int64]
) -> PlayerSelectionOutput:
    origins = np.where(raw.observed[rows], rows[:, None], -1)
    if np.array_equal(rows, np.arange(len(raw.track_ids))):
        return PlayerSelectionOutput(
            raw.camera_id, raw.track_ids, raw.observed, raw, origins, {}
        )
    mask = np.zeros_like(raw.observed)
    mask[rows] = raw.observed[rows]
    evidence = raw.evidence
    if evidence is None:
        raise ValueError(
            f"{raw.camera_id}: replay requires the saved pose/CLIP evidence"
        )
    group = PersonTrackingOutput(
        raw.camera_id,
        raw.track_ids[rows],
        raw.boxes_xyxy[rows],
        raw.observed[rows],
        tuple(raw.source_track_ids[i] for i in rows),
        (),
        evidence.regroup(origins),
    )
    return PlayerSelectionOutput(raw.camera_id, raw.track_ids, mask, group, origins, {})


def scaled_sampling(source: ClipSource) -> CropSamplingConfig:
    cfg = CropSamplingConfig()
    scale = source.pixel_threshold_scale
    return replace(
        cfg, min_height_px=cfg.min_height_px * scale, border_px=cfg.border_px * scale
    )


def track_appearance(
    tracks: PersonTrackingOutput,
    size: tuple[int, int],
    sampling: CropSamplingConfig,
    encoder: str,
) -> tuple[TrackAppearance, ...]:
    evidence = tracks.evidence
    if evidence is None or evidence.encoder != encoder:
        raise ValueError(
            f"{tracks.camera_id}: wrong or missing saved appearance evidence"
        )
    return tuple(
        evidence_appearance(
            tracks.boxes_xyxy, tracks.observed, evidence, size, sampling
        )
    )


def select(
    raw: PersonTrackingOutput,
    source: ClipSource,
    camera: PinholeCamera | None,
    footpoints: FootpointConfig,
    encoder: str,
) -> dict[str, PlayerSelectionOutput]:
    evidence = raw.evidence
    if evidence is None or evidence.poses is None or evidence.encoder != encoder:
        raise ValueError(
            f"{source.clip_id}/{raw.camera_id}: expected archived pose and CLIP"
        )
    if (
        raw.observed.shape[1] != source.num_frames
        or raw.camera_id not in source.camera_ids
    ):
        raise ValueError(
            f"{source.clip_id}/{raw.camera_id}: wrong track timeline or camera"
        )

    selection = {
        "all_tracks": selected_rows(raw, np.arange(len(raw.track_ids), dtype=np.int64)),
        "image_2d": selected_rows(
            raw, np.sort(image_candidates(raw, source.size, LINKING))
        ),
    }
    if camera is None:
        empty = selected_rows(raw, np.empty(0, np.int64))
        return {**selection, "court_raw_dwell": empty, "court_standard": empty}
    if camera.camera_id != raw.camera_id:
        raise ValueError(f"{source.clip_id}: selection camera axes disagree")

    footpoints = replace(
        footpoints,
        bottom_border_px=footpoints.bottom_border_px * source.pixel_threshold_scale,
    )
    points, valid = ground_footpoints(
        raw.boxes_xyxy, raw.observed, camera, source.size[1], footpoints
    )
    dwell = (valid & exclusive_region(points, LINKING, core=True)).sum(axis=1)
    rows = rank_candidates(
        raw.track_ids,
        dwell,
        dwell.astype(np.float64),
        LINKING.min_presence_fraction * source.num_frames,
        LINKING.max_candidates,
    )
    selection["court_raw_dwell"] = selected_rows(raw, np.sort(rows))

    appearances = track_appearance(raw, source.size, scaled_sampling(source), encoder)
    camera_tracks = CameraTracks(
        camera, source.size, raw.track_ids, raw.boxes_xyxy, raw.observed, appearances
    )
    mask, diagnostic = select_linked_candidates(
        camera_tracks, source.fps, LINKING, footpoints
    )
    linked, origins = linked_timeline(camera_tracks, diagnostic)
    grouped = PersonTrackingOutput(
        raw.camera_id,
        linked.track_ids,
        linked.boxes_xyxy,
        linked.observed,
        tuple((int(i),) for i in linked.track_ids),
        (),
        evidence.regroup(origins),
    )
    selection["court_standard"] = PlayerSelectionOutput(
        raw.camera_id, raw.track_ids, mask, grouped, origins, {}
    )
    return selection
