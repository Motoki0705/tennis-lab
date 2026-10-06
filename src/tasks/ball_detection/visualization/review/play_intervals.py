"""On-demand annotation-only proposals for the dataset review timeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    infer_play_intervals,
    mask_intervals,
)
from src.tasks.ball_detection.data.play_manifest import clip_evidence
from src.tasks.ball_detection.data.store import POINT_KIND_NAMES
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
    presence, observed, eligible, times = clip_evidence(store, clip)
    selection = infer_play_intervals(presence, observed, times, eligible, config)
    annotations = []
    for index, row in enumerate(store.clip_rows(clip)):
        instances = store.instances_of(int(row))
        kinds = [POINT_KIND_NAMES[int(kind)] for kind in instances.point_kind]
        reviewed = bool(store.frames["annotated"][row])
        reasons = []
        if not eligible[index]:
            reasons.append("reference_only")
        if not reviewed:
            reasons.append("unreviewed")
        if not kinds:
            reasons.append("no_ball")
        elif len(kinds) > 1:
            reasons.append("multiple_balls")
        elif kinds == ["out_of_frame"]:
            reasons.append("out_of_frame")
        # The viewer draws every finite coordinate, including reference frames,
        # estimated positions and multi-ball frames. Do not apply selection here.
        annotations.append(dict(
            kinds=kinds, located_count=int(np.isfinite(instances.xy).all(axis=1).sum()),
            reviewed=reviewed, is_target=bool(eligible[index]),
            evidence=bool(presence[index]), exclusion_reasons=reasons,
        ))
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
