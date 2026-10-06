"""On-demand annotation-only proposals for the dataset review timeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    infer_play_intervals,
    mask_intervals,
)
from src.tasks.ball_detection.visualization.review.datasets import (
    BallDatasetCatalog,
    split_scene_id,
)


def review_play_intervals(catalog: BallDatasetCatalog, scene: str) -> dict[str, Any]:
    """Use the training selection rule, without writing a manifest or images."""
    dataset, local = split_scene_id(scene)
    store, clip_id = catalog.store_clip(dataset, local)
    clip = store.clip_by_id(clip_id)
    config = PlayIntervalConfig()
    states = annotation_states(store, clip)
    presence, observed, eligible, times = states.evidence, states.supervision, states.target, states.times
    selection = infer_play_intervals(presence, observed, times, eligible, config)
    annotations = states.review_records()
    return dict(
        scene=scene, frames=clip.frame_count, config=asdict(config),
        pose_approved=catalog.spec(dataset).pose_approved,
        timestamps=times.tolist(), play=selection.intervals,
        excluded=selection.excluded, training=mask_intervals(selection.selected),
        presence=mask_intervals(presence), observed=mask_intervals(observed),
        annotations=annotations,
        bridged=mask_intervals(selection.bridged & selection.play),
        windows=selection.window_starts,
        counts=dict(play=int(selection.play.sum()), excluded=int((~selection.play).sum()),
                    training=int(selection.selected.sum()), windows=len(selection.window_starts)),
    )
