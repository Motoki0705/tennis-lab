"""Offline ball-event candidates from a scene with reconstruction masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ball_events.rules import (
    DEFAULT_THRESHOLDS,
    BallEvent,
    PlayerEvidence,
    RuleThresholds,
    analyze_ball_events,
    detect_ball_events,
)
from src.utils.schema.player import COCO_KP_IDX

if TYPE_CHECKING:
    from src.tennis_scene.schema import SceneResult

_WRISTS = [COCO_KP_IDX["left_wrist"], COCO_KP_IDX["right_wrist"]]


def detect_scene_events(
    scene: SceneResult, *, thresholds: RuleThresholds = DEFAULT_THRESHOLDS
) -> list[BallEvent]:
    """Apply the court-space rules to ball, root and wrist observations."""
    return analyze_scene_events(scene, thresholds=thresholds)[0]


def analyze_scene_events(
    scene: SceneResult, *, thresholds: RuleThresholds = DEFAULT_THRESHOLDS
) -> tuple[list[BallEvent], NDArray[np.bool_]]:
    """Return rule predictions with coverage on the original timeline."""
    if (
        scene.schema_version < 2
        or scene.ball_3d is None
        or scene.ball_3d_valid is None
        or scene.player_valid is None
        or scene.player_kp_3d is None
        or scene.player_kp_3d_vis is None
    ):
        raise ValueError(
            "Event detection requires SceneResult v2, v3 or v4 ball, player and validity arrays"
        )
    if scene.ball_3d.shape != (scene.num_frames, 3):
        raise ValueError("Ball positions disagree with the SceneResult frame count")
    players = PlayerEvidence(
        positions=np.concatenate(
            (
                scene.player_position[:, :, None, :],
                scene.player_kp_3d[:, :, _WRISTS, :],
            ),
            axis=2,
        ),
        valid=np.concatenate(
            (scene.player_valid[:, :, None], scene.player_kp_3d_vis[:, :, _WRISTS]),
            axis=2,
        ),
        track_ids=scene.player_track_ids,
    )
    events, coverage = analyze_ball_events(
        scene.ball_3d,
        scene.ball_3d_valid,
        scene.fps,
        players=players,
        thresholds=thresholds,
    )
    return list(events), np.asarray(coverage, dtype=np.bool_)


__all__ = ["BallEvent", "RuleThresholds", "analyze_scene_events", "detect_ball_events", "detect_scene_events"]
