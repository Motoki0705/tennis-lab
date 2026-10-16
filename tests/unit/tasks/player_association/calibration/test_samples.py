from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.calibration.samples import (
    CalibrationClip,
    PairWindow,
    hierarchical_weights,
    pair_windows,
    prepare_clip,
)


def test_geometry_labels_do_not_depend_on_appearance_and_use_whole_windows(scene: Callable[..., CalibrationClip]) -> None:
    clip = scene(frames=27)  # 1.5 s tail is too short, earlier three windows are full.
    config = load_association_config(players_per_side=1)
    pairs, audit = pair_windows(clip, config)
    assert len(pairs) == 36 and sum(p.positive for p in pairs) == 18
    assert {p.start for p in pairs} == {0, 8, 16}
    assert all(len(p.frames) == 8 for p in pairs)
    assert any(r['reason'] == 'shared_less_than_2s' for r in audit)
    cameras = tuple(replace(c, appearance=tuple(TrackAppearance(a.frames, a.embeddings[:, ::-1].copy())
                    for a in c.appearance or ())) for c in clip.cameras)
    # Invert only camera 0's appearance: all geometric pseudo labels must remain identical.
    changed, _ = pair_windows(replace(clip, cameras=(cameras[0], *clip.cameras[1:])), config)
    assert [(p.rows, p.start, p.positive, p.distance_m) for p in changed] == [
        (p.rows, p.start, p.positive, p.distance_m) for p in pairs]
    assert any(a.cosine != b.cosine for a, b in zip(pairs, changed, strict=True))
    tail, _ = pair_windows(scene(frames=28), config)
    assert len(tail) == 48 and sum(p.start == 24 for p in tail) == 12  # exact 2 s tail


def test_handoff_invalid_feet_and_switch_have_explicit_exclusions(scene: Callable[..., CalibrationClip]) -> None:
    config = load_association_config(players_per_side=1)
    clip = scene()
    events = [x.copy() for x in clip.handoffs]
    events[0][0, 2] = True
    _, audit = pair_windows(replace(clip, handoffs=tuple(events)), config)
    assert sum(r['reason'] == 'switch_or_handoff' for r in audit) == 4
    boxes = clip.cameras[0].boxes_xyxy.copy()
    boxes[0, 2, 3] = 1080
    cameras = (replace(clip.cameras[0], boxes_xyxy=boxes), *clip.cameras[1:])
    _, audit = pair_windows(replace(clip, cameras=cameras), config)
    assert any(r['reason'] == 'invalid_footpoint' for r in audit)
    boxes = clip.cameras[0].boxes_xyxy.copy()
    boxes[0, 4:, [0, 2]] += 160  # persistent 8 m x jump; the existing switch detector decides its cut.
    _, audit = pair_windows(replace(clip, cameras=(replace(clip.cameras[0], boxes_xyxy=boxes), *clip.cameras[1:])), config)
    assert any(r['reason'] == 'switch_or_handoff' for r in audit)


def test_competing_close_candidate_blocks_positive(scene: Callable[..., CalibrationClip]) -> None:
    clip = scene()
    camera = clip.cameras[1]
    boxes = camera.boxes_xyxy.copy()
    boxes[1] = boxes[0]
    _, audit = pair_windows(replace(clip, cameras=(clip.cameras[0], replace(camera, boxes_xyxy=boxes), clip.cameras[2])),
                           load_association_config(players_per_side=1))
    assert any(r['reason'] == 'ambiguous_geometry' and r['distance_m'] < 4 for r in audit)


def test_weights_balance_recordings_clips_strata_windows_and_fragments(scene: Callable[..., CalibrationClip]) -> None:
    pairs, _ = pair_windows(scene(), load_association_config(players_per_side=1))
    pair = pairs[0]
    data = [pair, replace(pair, rows=(2, 2)),
            replace(pair, start=8, end=16, frames=tuple(range(8, 16))),
            replace(pair, clip='video_000/clip_011'),
            replace(pair, video='video_001', clip='video_001/clip_010')]
    np.testing.assert_allclose(hierarchical_weights(data), [.0625, .0625, .125, .25, .5])
    with pytest.raises(ValueError, match='Duplicate'):
        hierarchical_weights([pair, pair])


def test_prepare_uses_fixed_selection_and_marks_source_handoffs(scene: Callable[..., CalibrationClip]) -> None:
    clip = scene()
    cam = clip.cameras[0]
    # Split one person's track with a one-frame overlap (within .2 s at 10 fps).
    first: np.ndarray = np.zeros(24, bool)
    second: np.ndarray = np.zeros(24, bool)
    first[:13], second[12:] = True, True
    raw = replace(cam, track_ids=np.array([1, 2, 3], np.int64),
                  boxes_xyxy=np.stack([cam.boxes_xyxy[0], cam.boxes_xyxy[1], cam.boxes_xyxy[0]]),
                  observed=np.stack([first, cam.observed[1], second]),
                  appearance=(cam.appearance[0], cam.appearance[1], cam.appearance[0]))  # type: ignore[index]
    ready, _ = prepare_clip(clip.key, 10., (raw, *clip.cameras[1:]), load_association_config(players_per_side=1))
    assert ready.handoffs[0].any() and ready.cameras[0].observed.shape[0] == 2


def test_pair_window_rejects_ambiguous_thresholds() -> None:
    args = ('v', 'v/c', ('cam0', 'cam1'), (0, 0), (-1, -1), 0, 4, (0, 1, 2, 3))
    with pytest.raises(ValueError, match='unambiguous'):
        PairWindow(*args, 4., .5, True)
    with pytest.raises(ValueError, match='unambiguous'):
        PairWindow(*args, 10., .5, False)
