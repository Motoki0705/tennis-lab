"""Check selection boundaries, paired supervision, ownership and target encoding."""

# mypy: disallow_untyped_decorators=False

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.windows.hybrid import eligible_spans
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig

spec = importlib.util.spec_from_file_location("all_frame_probe", Path(__file__).with_name("all_frame_probe.py"))
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_common_spans_exclude_short_regions_for_every_length() -> None:
    target: NDArray[np.bool_] = np.ones(512, dtype=np.bool_)
    target[256] = False
    times: NDArray[np.float64] = np.arange(512, dtype=np.float64)
    times[128:] += 20
    times[384:] += 20
    states = SimpleNamespace(target=target, times=times)
    spans = eligible_spans(states, PlayIntervalConfig())
    assert spans == [(0, 128), (128, 256), (257, 384), (384, 512)]
    common = [(a, b) for a, b in spans if b - a >= 128]
    assert common == [(0, 128), (128, 256), (384, 512)]
    for t in probe.LENGTHS:
        starts = [s for a, b in common for s in probe.window_starts(a, b, t, t // 2)]
        assert all(any(a <= s and s + t <= b for a, b in common) for s in starts)
        assert not any(257 <= s < 384 for s in starts)


@pytest.mark.parametrize("n", [128, 129, 145, 256, 270, 573])
@pytest.mark.parametrize("t", [32, 64, 128])
def test_every_frame_has_the_bruteforce_center_owner(n: int, t: int) -> None:
    a, b = 5, 5 + n
    starts = probe.window_starts(a, b, t, t // 2)
    owners = probe.owner_starts(a, b, t)
    expected = [min((s for s in starts if s <= f < s + t),
                    key=lambda s: (abs(2 * f - (2 * s + t - 1)), s)) for f in range(a, b)]
    assert owners.tolist() == expected
    assert starts[0] == a and starts[-1] == b - t
    assert owners[0] == a and owners[-1] == b - t
    # Overlap midpoint tie goes to the earlier start: centers 20.5 and 36.5.
    if t == 32:
        assert owners[23] == 5
    if n == 129:
        # Odd tail offset places the midpoint on an integer frame.
        midpoint = (starts[-2] + starts[-1] + t - 1) // 2
        assert owners[midpoint - a] == starts[-2]


def test_stride_tail_and_insufficient_span() -> None:
    assert probe.window_starts(7, 280, 128, 16) == [*range(7, 152, 16), 152]
    assert probe.window_starts(7, 135, 128, 16) == [7]
    assert probe.window_starts(7, 134, 128, 16) == []
    with pytest.raises(ValueError, match="Invalid"):
        probe.window_starts(7, 280, 128, 129)
    with pytest.raises(ValueError, match="Uncovered"):
        probe.owner_starts(7, 134, 128)


def test_paired_updates_preserve_all_frames_and_teachers_even_with_empty_children() -> None:
    codes: NDArray[np.uint8] = np.zeros(273, dtype=np.uint8)
    codes[[0, 45, 90, 128, 180, 272]] = [1, 2, 3, 1, 3, 2]
    starts = probe.window_starts(0, len(codes), 128, 16)
    parent_counts = probe.window_counts(codes, starts, 128)
    accepted = parent_counts[:, 1:].sum(axis=1) > 0
    assert accepted.all()
    coverage: NDArray[np.bool_] = np.zeros(len(codes), dtype=np.bool_)
    for p in starts:
        coverage[p:p + 128] = True
        for t in probe.LENGTHS:
            child = probe.paired_starts(p, t)
            assert [f for s in child for f in range(s, s + t)] == list(range(p, p + 128))
            assert np.array_equal(probe.window_counts(codes, child, t).sum(axis=0), np.bincount(codes[p:p + 128], minlength=4))
    assert coverage.all()
    assert (probe.window_counts(codes, probe.paired_starts(0, 32), 32)[:, 1:].sum(axis=1) == 0).any()


def test_only_fully_masked_packs_are_rejected_without_dropping_known_targets() -> None:
    codes: NDArray[np.uint8] = np.zeros(400, dtype=np.uint8)
    codes[0] = 1
    codes[200] = 2
    codes[-1] = 3
    starts = probe.window_starts(0, len(codes), 128, 16)
    counts = probe.window_counts(codes, starts, 128)
    chosen = [s for s, c in zip(starts, counts, strict=True) if c[1:].sum() > 0]
    assert len(chosen) < len(starts)
    mask: NDArray[np.bool_] = np.zeros(len(codes), dtype=np.bool_)
    for s in chosen:
        mask[s:s + 128] = True
    assert np.flatnonzero(mask & (codes > 0)).tolist() == [0, 200, 399]
    compressed = probe.compress_starts(chosen)
    assert [s for first, last, step in compressed for s in range(first, last + 1, step)] == chosen


def test_negative_targets_roundtrip_and_do_not_include_unknowns_or_observed() -> None:
    codes = np.asarray([2, 2, 0, 3, 3, 1, 2, 3, 2], dtype=np.uint8)
    mask = np.asarray([True, True, True, True, True, True, True, True, False])
    runs = probe.class_runs(codes, mask)
    assert runs == [[0, 2, 2], [3, 5, 3], [6, 7, 2], [7, 8, 3]]
    restored: NDArray[np.uint8] = np.zeros(len(codes), dtype=np.uint8)
    for a, b, kind in runs:
        assert not restored[a:b].any()
        restored[a:b] = kind
    assert np.array_equal(restored, np.where(mask & (codes >= 2), codes, 0))


def test_absence_overlap_is_complete_for_the_full_safe_span_definition() -> None:
    # Logical consequence of the agreed common definition, not independent data.
    codes = np.asarray([2, 3, 0, 1] * 64, dtype=np.uint8)
    spans = [(0, 128), (128, 256)]
    common: NDArray[np.bool_] = np.zeros(256, dtype=np.bool_)
    for a, b in spans:
        common[a:b] = True
    absence = common & (codes >= 2)
    assert absence.sum() == 128
    assert not (absence & ~common).any()


def test_saved_common_and_negative_membership_reconstruct_the_published_counts_and_identities() -> None:
    root = Path(__file__).parent
    catalog = {int(c["clip_index"]): c for c in csv.DictReader((root / "clip-catalog.csv").open())}
    summary = json.loads((root / "summary.json").read_text())["rows"]
    for split in ("train", "val", "test"):
        intervals = dict(json.loads(line) for line in (root / f"common-intervals-{split}.jsonl").read_text().splitlines())
        negatives = dict(json.loads(line) for line in (root / f"absence-targets-{split}.jsonl").read_text().splitlines())
        common_row = next(r for r in summary if r["split"] == split and r["scope"] == "common")
        negative_row = next(r for r in summary if r["split"] == split and r["scope"] == "absence")
        overlap_row = next(r for r in summary if r["split"] == split and r["scope"] == "absence_overlap")
        exclusive_row = next(r for r in summary if r["split"] == split and r["scope"] == "absence_exclusive")
        assert exclusive_row["frames"] == 0
        hashes = {key: hashlib.sha256() for key in ("common", "absence", "absence_teacher")}
        class_counts = {2: 0, 3: 0}
        frames = 0
        for index in sorted(intervals, key=lambda i: catalog[i]["clip_id"]):
            c = catalog[index]
            assert c["split"] == split
            selected: set[int] = set()
            for a, b in intervals[index]:
                assert 0 <= a < b <= int(c["frames"]) and b - a >= 128
                assert not selected.intersection(range(a, b))
                selected.update(range(a, b))
            frames += len(selected)
            for f in sorted(selected):
                key = [c["clip_id"], f, 1, c["native_fps"]]
                hashes["common"].update((json.dumps(key, separators=(",", ":")) + "\n").encode())
            selected_negative: dict[int, int] = {}
            for a, b, kind in negatives.get(index, []):
                assert kind in (2, 3) and 0 <= a < b <= int(c["frames"])
                assert set(range(a, b)) <= selected
                assert not set(range(a, b)).intersection(selected_negative)
                selected_negative.update(dict.fromkeys(range(a, b), kind))
                class_counts[kind] += b - a
            for f, kind in sorted(selected_negative.items()):
                key = [c["clip_id"], f, 1, c["native_fps"]]
                hashes["absence"].update((json.dumps(key, separators=(",", ":")) + "\n").encode())
                hashes["absence_teacher"].update((json.dumps([*key, kind], separators=(",", ":")) + "\n").encode())
        assert set(negatives) <= set(intervals)
        assert frames == common_row["frames"]
        assert class_counts == {2: negative_row["reviewed_empty"], 3: negative_row["out_of_frame"]}
        assert class_counts == {2: common_row["reviewed_empty"], 3: common_row["out_of_frame"]}
        assert negative_row["target_identity_sha256"] == overlap_row["target_identity_sha256"]
        assert negative_row["teacher_identity_sha256"] == overlap_row["teacher_identity_sha256"]
        assert hashes["common"].hexdigest() == common_row["target_identity_sha256"]
        assert hashes["absence"].hexdigest() == negative_row["target_identity_sha256"]
        assert hashes["absence_teacher"].hexdigest() == negative_row["teacher_identity_sha256"]
        expected_classes = [common_row[name] for name in probe.LABELS]
        assert sum(expected_classes) == frames
        evaluation = list(csv.DictReader((root / "evaluation-support.csv").open()))
        assert {int(r["assigned_frames"]) for r in evaluation if r["split"] == split} == {frames}
        assert {int(r["assigned_labeled_frames"]) for r in evaluation if r["split"] == split} == {sum(expected_classes[1:])}
    budget = list(csv.DictReader((root / "budget.csv").open()))
    for split in ("train", "val", "test"):
        paired = [r for r in budget if r["split"] == split and r["rule"] == "paired_known1"]
        assert {int(r["frames_per_window"]) for r in paired} == {32, 64, 128}
        for label in probe.LABELS:
            assert len({r[f"expected_{label}"] for r in paired}) == 1
