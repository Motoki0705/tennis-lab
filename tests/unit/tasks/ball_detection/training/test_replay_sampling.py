from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.temporal import SampledWindow
from src.tasks.ball_detection.training.replay import ReplayFPSMixSampler, replay_recipe


def inputs(tmp_path: Path) -> tuple[FrozenBallWindowDataset, Path]:
    records = [dict(clip=dict(clip_id=name, split="train"), semantic_sha256=name,
                    rgb_shard_sha256=name, windows=[dict(start=0, frame_step=1)])
               for name in ("original", "old500", "new")]
    dataset = cast(FrozenBallWindowDataset, SimpleNamespace(
        frame_steps=(1, 2, 4), records=records,
        windows=[(record, SampledWindow(start=start, frame_step=step))
                 for record in range(3) for step in (1, 2, 4) for start in range(3)]))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(dict(clips=records[:1])))
    return dataset, previous


def test_replay_uses_training_history_and_exact_quotas(tmp_path: Path) -> None:
    dataset, previous = inputs(tmp_path)
    sampler = ReplayFPSMixSampler(dataset, windows_per_epoch=30, seed=42,
                                  previously_trained_manifest=previous, untrained_fraction=.6)
    first = list(sampler)
    groups = [(dataset.windows[i][1].frame_step,
               sampler.membership[dataset.records[dataset.windows[i][0]]["clip"]["clip_id"]]) for i in first]
    assert Counter(groups) == {(step, group): count for step in (1, 2, 4)
                               for group, count in (("untrained", 6), ("replay", 4))}
    assert sampler.membership == {"original": "replay", "old500": "untrained", "new": "untrained"}
    assert set(index for bucket in sampler.buckets.values() for index in bucket) == set(range(27))
    for key, available in sampler.buckets.items():
        selected = [i for i, group in zip(first, groups, strict=True) if group == key]
        assert len(set(selected[:len(available)])) == min(len(available), len(selected))
    assert sampler.last_report is not None
    assert sum(row["draws"] for row in sampler.last_report["groups"]) == 30
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first


@pytest.mark.parametrize("change,match", [
    ("removed", "removed previously"), ("changed", "identity changed"),
    ("holdout", "held-out"), ("empty_fps", "No eligible windows"),
])
def test_replay_rejects_missing_or_changed_training_inputs(tmp_path: Path, change: str, match: str) -> None:
    dataset, previous = inputs(tmp_path)
    if change == "removed":
        dataset.records[0]["clip"]["clip_id"] = "replacement"
    elif change == "changed":
        dataset.records[0]["semantic_sha256"] = "changed GT"
    elif change == "holdout":
        document = json.loads(previous.read_text())
        document["clips"].append(dict(clip=dict(clip_id="new", split="test")))
        previous.write_text(json.dumps(document))
    else:
        dataset.windows = [(r, w) for r, w in dataset.windows if r != 0 or w.frame_step != 4]
    with pytest.raises(ValueError, match=match):
        ReplayFPSMixSampler(dataset, windows_per_epoch=30, seed=42,
                            previously_trained_manifest=previous, untrained_fraction=.6)


def test_replay_ratio_and_manifest_are_both_explicit(tmp_path: Path) -> None:
    dataset, previous = inputs(tmp_path)
    for manifest, fraction in ((previous, None), (None, .6), (previous, 1.0), (previous, float("nan"))):
        with pytest.raises(ValueError):
            replay_recipe(manifest, fraction)
    with pytest.raises(ValueError, match="exact positive integer quota"):
        ReplayFPSMixSampler(dataset, windows_per_epoch=31, seed=42,
                            previously_trained_manifest=previous, untrained_fraction=.6)
    assert replay_recipe(None, None) is None
