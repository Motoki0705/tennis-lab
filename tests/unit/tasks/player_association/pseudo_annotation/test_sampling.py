from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.pseudo_annotation.sampling import (
    ASSOCIATION,
    CONFIDENT,
    NEAR_REJECTED,
    SINGLE_VIEW,
    STRATA,
    TRACKING,
    SamplingConfig,
    near_court,
    reconstruction_suspicions,
    sample_strata,
    tracking_suspicions,
)
from src.utils.geometry.triangulation import PointRejection


def config() -> SamplingConfig:
    return SamplingConfig.load(
        Path("src/tasks/player_association/configs/pseudo_annotation_meiji.yaml")
    )


def test_controls_exclude_doubt_and_sampling_preserves_source_frames() -> None:
    cfg = config().model_copy(
        update={"doubt_blocks_per_clip_stratum": 1, "confident_frames_per_clip": 2}
    )
    masks: dict[str, NDArray[np.bool_]] = {
        name: np.zeros(15, dtype=bool) for name in STRATA
    }
    masks[NEAR_REJECTED][[0, 2, 3, 4, 7, 8]] = True
    masks[TRACKING][[4, 5]] = True
    masks[CONFIDENT][:] = True
    detections: NDArray[np.bool_] = np.ones(15, dtype=bool)
    detections[3] = False
    eligible, sample = sample_strata("video_000/clip_020", masks, detections, cfg)
    assert eligible[NEAR_REJECTED] == [0, 2, 4, 7, 8]
    assert eligible[CONFIDENT] == [1, 6, 9, 10, 11, 12, 13, 14]
    assert sample[TRACKING] == [4, 5]
    assert len(sample[CONFIDENT]) == 2 and set(sample[CONFIDENT]) <= set(
        eligible[CONFIDENT]
    )
    assert sample[NEAR_REJECTED] in ([0, 2], [4], [7, 8])
    assert sample_strata(
        "video_000/clip_020", dict(reversed(masks.items())), detections, cfg
    ) == (eligible, sample)


def test_switches_and_short_observation_spans_use_valid_near_court_footpoints() -> None:
    cfg = config().model_copy(update={"switch_window_s": 0.2})
    observed: NDArray[np.bool_] = np.ones((3, 24), dtype=bool)
    observed[1, :] = False
    observed[1, [3, 4, 6, 7]] = True
    points = np.zeros((3, 24, 2))
    points[0, 12:, 0] = 6
    points[1, :, 0] = 6.5  # doubles alley, at the declared boundary
    points[1, :, 1] = -18
    points[2, :, 1] = 18.01
    points[2, 12:, 0] = 6
    near = near_court(points, observed, cfg)
    assert near[1].sum() == 4 and not near[2].any()
    flagged = tracking_suspicions(observed, points, observed, near, 10, cfg)
    assert np.flatnonzero(flagged).tolist() == [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]


def test_saved_reprojection_and_one_view_masks_preserve_player_time_axes() -> None:
    cfg = config()
    views: NDArray[np.int64] = np.full((4, 5), 3, dtype=np.int64)
    valid: NDArray[np.bool_] = np.ones((4, 5, 17), dtype=bool)
    reasons: NDArray[np.uint8] = np.zeros((4, 5, 17), dtype=np.uint8)
    residual: NDArray[np.float32] = np.zeros((4, 4, 5, 17), dtype=np.float32)
    views[3, 1] = 1
    valid[3, 1] = False
    residual[3, 0, 2, :3] = 24.0  # exactly at threshold
    residual[3, 2, 3, :2] = 24.1
    reasons[3, 3, 2] = PointRejection.REPROJECTION
    valid[3, 3, 2] = False
    views[1, 4] = 0
    valid[1, 4] = False
    single, bad, support = reconstruction_suspicions(
        views, valid, reasons, residual, cfg
    )
    assert np.flatnonzero(single).tolist() == [1]
    assert np.flatnonzero(bad).tolist() == [3]
    assert np.flatnonzero(support).tolist() == [0, 2, 3]
    # A high residual can also have good 3D support; the control selector excludes it.
    masks: dict[str, NDArray[np.bool_]] = {
        name: np.zeros(5, dtype=bool) for name in STRATA
    }
    masks[SINGLE_VIEW], masks[ASSOCIATION], masks[CONFIDENT] = single, bad, support
    eligible, _ = sample_strata("video_000/clip_000", masks, np.ones(5, bool), cfg)
    assert eligible[CONFIDENT] == [0, 2]
