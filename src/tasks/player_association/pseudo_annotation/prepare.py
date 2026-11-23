"""Prepare blind or explicitly hinted image packets from the baseline detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.tasks.player_association.pseudo_annotation.images import prepare_images
from src.tasks.player_association.pseudo_annotation.job import (
    Candidate,
    Packet,
    frame_numbers,
)


def _hints(frozen: FrozenClip, camera: str) -> dict[int, tuple[int, int | None]]:
    assignment = frozen.assignments()
    if assignment is None:
        raise ValueError("Hints require saved final player assignments")
    by_group = assignment.get(camera, {})
    raw = frozen.raw(camera, rows=True)
    kept = frozen.selection(camera, origins=True)
    rows = raw["evidence.detection_rows"]
    if rows.shape != raw["observed"].shape or not np.array_equal(
        rows >= 0, raw["observed"]
    ):
        raise ValueError(
            f"Tracking evidence does not identify the actual detections: {camera}"
        )
    with_player: dict[int, int] = {}
    for group_row, frame in zip(*np.nonzero(kept["tracks.observed"]), strict=True):
        group = int(kept["tracks.track_ids"][group_row])
        carrier = by_group.get(group)
        if carrier is not None and carrier[frame] >= 0:
            raw_row = kept["origin_rows"][group_row, frame]
            source_row = int(rows[raw_row, frame])
            if source_row < 0:
                raise ValueError(f"Selected group has no source observation: {camera}")
            with_player[source_row] = int(carrier[frame])
    result = {
        int(rows[row, frame]): (
            int(raw["track_ids"][row]),
            with_player.get(int(rows[row, frame])),
        )
        for row, frame in zip(*np.nonzero(raw["observed"]), strict=True)
    }
    if len(result) != int(raw["observed"].sum()):
        raise ValueError(f"Source detection row is reused in {camera}")
    return result


def prepare_baseline(
    frozen: FrozenRun,
    plan: dict[str, Any],
    clip_id: str,
    frames: list[int],
    destination: Path,
    *,
    show_prior_association: bool = False,
) -> Packet:
    stored = frozen.clip(clip_id)
    indices = frame_numbers(frames, stored.clip.num_frames)
    selected = plan["clips"][clip_id]["sampled"]
    strata = {
        frame: [name for name, values in selected.items() if frame in values]
        for frame in indices
    }
    if any(not names for names in strata.values()):
        raise ValueError("Annotation frames must belong to the sampled baseline plan")
    candidates: dict[int, dict[str, list[Candidate]]] = {frame: {} for frame in indices}
    for camera in stored.clip.camera_ids:
        offsets, boxes, source_rows = stored.detections(camera)
        hints = _hints(stored, camera) if show_prior_association else {}
        if not hints.keys() <= set(source_rows.tolist()):
            raise ValueError(
                f"Track row does not identify a pinned detection: {camera}"
            )
        for frame in indices:
            frame_candidates = []
            for row in range(offsets[frame], offsets[frame + 1]):
                raw_track, prior = hints.get(int(source_rows[row]), (None, None))
                frame_candidates.append(
                    Candidate(
                        box=boxes[row].astype(float).tolist(),
                        raw_track=raw_track,
                        prior_player=prior,
                    )
                )
            candidates[frame][camera] = frame_candidates
    return prepare_images(
        stored.clip,
        candidates,
        strata,
        destination,
        player_count=4,
        input_snapshot_sha256=frozen.sha256,
        show_prior_association=show_prior_association,
        expected_media=stored.media_sha256,
    )
