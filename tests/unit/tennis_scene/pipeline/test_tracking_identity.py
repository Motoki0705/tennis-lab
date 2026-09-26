"""Identity links must retain observations and expose their evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.tracking_identity import (
    link_tracklets,
    torso_appearance_lab,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def _observation(track_id: int, x: float, color: tuple[float, float, float]) -> dict[str, object]:
    return {"id": track_id, "bbx_xyxy": np.array([x, 10, x + 20, 50], np.float32),
            "appearance_lab": np.array(color, np.float64)}


def test_links_unique_matching_tracklets_and_keeps_source_evidence() -> None:
    history: list[list[dict[str, object]]] = [[] for _ in range(150)]
    for frame in range(40):
        history[frame] = [_observation(2, 100, (50, 125, 118)), _observation(1, 10, (180, 110, 130))]
    for frame in range(50, 95):
        history[frame] = [_observation(3, 103, (53, 125, 119))]
    for frame in range(120, 150):
        history[frame] = [_observation(5, 105, (55, 125, 118))]

    linked = link_tracklets(history)

    assert linked.source_ids == {1: (1,), 2: (2, 3, 5)}
    assert [(item.earlier_id, item.later_id, item.missing_frames) for item in linked.links] == [(2, 3, 10), (3, 5, 25)]
    assert [row["id"] for row in linked.history[50]] == [2]
    assert [row["id"] for row in linked.history[120]] == [2]
    assert linked.history[45] == []


def test_refuses_far_appearance_and_ambiguous_links() -> None:
    far = [[_observation(1, 10, (50, 125, 118))], [],
           [_observation(2, 200, (50, 125, 118))]]
    assert link_tracklets(far).source_ids == {1: (1,), 2: (2,)}
    changed_clothing = [[_observation(1, 10, (50, 125, 118))], [],
                        [_observation(2, 10, (150, 125, 118))]]
    assert link_tracklets(changed_clothing).source_ids == {1: (1,), 2: (2,)}
    ambiguous = [[_observation(1, 10, (50, 125, 118)), _observation(2, 13, (52, 125, 118))],
                 [], [_observation(3, 11, (51, 125, 118))]]
    with pytest.raises(ReconstructionUnavailable, match="Multiple plausible"):
        link_tracklets(ambiguous)


def test_one_frame_nested_duplicate_can_continue_an_existing_identity(tmp_path: Path) -> None:
    history: list[list[dict[str, object]]] = [[] for _ in range(16)]
    for frame in range(10):
        history[frame].append(_observation(2, 100, (110, 110, 120)))
    nested = _observation(9, 105, (112, 110, 120))
    nested["bbx_xyxy"] = np.array([105, 10, 120, 50], np.float32)
    history[8].insert(0, nested)
    for frame in range(11, 16):
        history[frame].append(nested)

    linked = link_tracklets(history)

    assert linked.source_ids == {2: (2, 9)}
    assert len(linked.history[8]) == 1
    assert linked.history[8][0]["source_track_id"] == 2
    assert linked.links[0].overlap_span_frames == 2
    assert linked.links[0].shared_observation_frames == 1
    assert linked.links[0].duplicate_containment == pytest.approx(1.)
    output = PersonTrackingOutput("cam0", np.array([2], np.int64),
        np.zeros((1, 16, 4), np.float32), np.ones((1, 16), bool),
        (linked.source_ids[2],), linked.links)
    codec = ArtifactCodec(PersonTrackingOutput)
    payload, arrays = codec.dump(output, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    assert restored.tracklet_links == linked.links

    distinct = [[_observation(2, 100, (110, 110, 120))] for _ in range(10)]
    distinct[8].append(_observation(9, 112, (112, 110, 120)))
    for _ in range(11, 16):
        distinct.append([_observation(9, 112, (112, 110, 120))])
    assert link_tracklets(distinct).source_ids == {2: (2,), 9: (9,)}


def test_torso_appearance_samples_inside_bounding_box() -> None:
    frame: np.ndarray = np.full((60, 60, 3), (0, 180, 0), np.uint8)
    frame[15:30, 15:35] = (160, 80, 40)
    actual = torso_appearance_lab(frame, np.array([10, 10, 40, 50], np.float32))
    assert actual.shape == (3,)
    assert np.isfinite(actual).all()
    with pytest.raises(ValueError, match="positive"):
        torso_appearance_lab(frame, np.array([10, 10, 10, 50], np.float32))
