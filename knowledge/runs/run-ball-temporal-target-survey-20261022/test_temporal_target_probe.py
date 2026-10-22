"""Small CPU checks for the survey's selection and teacher accounting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import AnnotationStates
from src.tasks.ball_detection.data.store import POINT_KIND_CODES
from src.tasks.ball_detection.data.windows.hybrid import eligible_spans
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.utils.checksum import dual_sha256

spec = importlib.util.spec_from_file_location("temporal_target_probe", Path(__file__).with_name("temporal_target_probe.py"))
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def states(count: list[int], kinds: list[str | None]) -> AnnotationStates:
    n = len(count)
    kind_counts: NDArray[np.int64] = np.zeros((n, len(POINT_KIND_CODES)), np.int64)
    for i, kind in enumerate(kinds):
        if kind is not None:
            kind_counts[i, POINT_KIND_CODES[kind]] = count[i]
    return AnnotationStates(
        reviewed=np.ones(n, np.bool_), target=np.ones(n, np.bool_),
        count=np.asarray(count, np.int64), kinds=kind_counts,
        located=np.zeros(n, np.int64), occluded=np.zeros(n, np.int64),
        occluded_located=np.zeros(n, np.int64), occluded_unlocated=np.zeros(n, np.int64),
        xy=np.full((n, 2), np.nan), track=np.zeros(n, np.int64),
        times=np.arange(n, dtype=np.float64), boundaries=np.zeros(n, np.bool_), events=np.zeros(n, np.uint8),
    )


def test_symmetric_context_and_nested_membership() -> None:
    centers = [8, 24, 40, 56, 72, 88]
    assert probe.context_starts(centers, 32, [(0, 112)]) == centers
    assert probe.context_starts(centers, 64, [(0, 112)]) == [24, 40, 56, 72]
    assert probe.context_starts(centers, 128, [(0, 112)]) == []
    assert probe.context_starts(centers, 32, [(0, 32), (48, 112)]) == [8, 56, 72, 88]
    with pytest.raises(ValueError, match="symmetrically"):
        probe.context_starts(centers, 33, [(0, 112)])


def test_safe_clip_context_keeps_reference_and_pts_barriers() -> None:
    s = states([0] * 120, [None] * 120)
    s.target[40:48] = False
    s.times[80:] += 20
    spans = eligible_spans(s, PlayIntervalConfig())
    assert spans == [(0, 40), (48, 80), (80, 120)]
    assert probe.context_starts([8, 24, 56, 72, 88, 104], 32, spans) == [8, 56, 88]


def test_unique_teachers_are_not_window_occurrences() -> None:
    codes = np.asarray([1, 1, 2, 3, 0, 1, 2, 0], np.uint8)
    stats, mask = probe.selection_stats(codes, np.asarray([True, False, False, False, False, False, False, False]), [0, 2, 4], 4)
    assert mask.all()
    assert [stats[k] for k in ("observed", "reviewed_empty", "out_of_frame", "masked")] == [3, 2, 1, 2]
    assert [stats[f"{k}_occurrences"] for k in ("observed", "reviewed_empty", "out_of_frame", "masked")] == [4, 3, 2, 3]
    assert stats["observed_occluded"] == 1
    assert stats["supervised_blocks"] == 3
    assert stats["negative_only_blocks"] == 0
    assert sum(stats[k] for k in ("observed", "reviewed_empty", "out_of_frame", "masked")) == stats["unique_frames"]
    stats, _ = probe.selection_stats(np.asarray([2, 3, 0, 0], np.uint8), np.zeros(4, np.bool_), [0, 2], 2)
    assert stats["negative_only_blocks"] == 1
    assert stats["zero_teacher_blocks"] == 1


def test_irregular_tail_is_preserved() -> None:
    assert probe.compress_starts([8, 24, 40, 43, 59]) == [[8, 40, 16], [43, 59, 16]]
    with pytest.raises(ValueError, match="unique"):
        probe.compress_starts([8, 8])


def test_fully_masked_centers_are_excluded_without_losing_teachers() -> None:
    codes = np.asarray([1, 0, 0, 0, 2, 3], np.uint8)
    starts, mask = probe.usable_centers(codes, [0, 2, 4], 2)
    assert starts == [0, 4]
    assert mask.tolist() == [True, True, False, False, True, True]
    assert np.array_equal(codes[mask & (codes != 0)], codes[codes != 0])


def test_chat_notes_and_unknowns_follow_existing_api(tmp_path: Path) -> None:
    counts = [0, 0, 1, 1, 2, 1, 0, 1]
    s = states(counts, [None, None, "out_of_frame", "unresolved", "observed", "observed", None, "observed"])
    s.reviewed[5] = False
    s.target[6] = False
    annotation = tmp_path / "annotation.json"
    raw = dict(schema_version="tennis_chat_ball_annotation.v1", clip_id="fixture", frame_count=len(counts), frames=[
        dict(frame_index=i, reviewed=bool(s.reviewed[i]), balls=[{}] * count, notes="uncertain" if i == 1 else "")
        for i, count in enumerate(counts)
    ])
    annotation.write_text(json.dumps(raw))
    clip = SimpleNamespace(source="chat_annotation", clip_id="chat_annotation/fixture", frame_count=len(counts),
                           annotation_sha256=dual_sha256(annotation))
    (tmp_path / "metadata.json").write_text(json.dumps(dict(clips=[dict(clip_id=clip.clip_id, annotation_path=str(annotation))])))
    store = SimpleNamespace(directory=tmp_path, frames=dict(frame_index=np.arange(len(counts))),
                            clip_rows=lambda clip: np.arange(clip.frame_count))
    codes, evidence = probe.classify(store, clip, s)
    assert codes.tolist() == [2, 0, 3, 0, 0, 0, 0, 1]
    assert evidence["excluded_noted_empty_frames"] == 1
    annotation.write_text(json.dumps(raw) + " ")
    with pytest.raises(ValueError, match="checksum"):
        probe.classify(store, clip, s)
