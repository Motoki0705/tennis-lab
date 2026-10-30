"""Consumer receipts must reject changed teachers, packs and scoring owners."""

# mypy: disallow_untyped_decorators=False

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from src.tasks.ball_detection.data.distribution import (
    prepare_distribution,
    source_payloads,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    build_all_frame_manifests,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.utils.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("verify_window_consumers", Path(__file__).with_name("verify_window_consumers.py"))
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

store_spec = importlib.util.spec_from_file_location("ball_store_fixture", PROJECT_ROOT / "tests/support/tasks/ball_detection/store.py")
assert store_spec is not None and store_spec.loader is not None
store_fixture = importlib.util.module_from_spec(store_spec)
store_spec.loader.exec_module(store_fixture)
ball, frame, write_store_clip = store_fixture.ball, store_fixture.frame, store_fixture.write_store_clip


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = tmp_path / "source"
    write_store_clip(source, "tracknet/test/drop", [frame(i, ball()) for i in range(8)], split="test")
    write_store_clip(source, "tracknet/train/main", [frame(i, ball() if i == 2 else ball("unresolved", None))
                                                   for i in range(8)])
    write_store_clip(source, "tracknet/val/main", [frame(i) for i in range(8)], split="val")
    stage = tmp_path / "distribution"
    prepare_distribution(source, stage, version="subset-v1",
                         clip_ids=["tracknet/train/main", "tracknet/val/main"], inputs={})
    configs = [AllFrameWindowConfig(2, max_window_length=8, pack_length=8, pack_stride=2)]
    original = build_all_frame_manifests(BallFrameStore(source), configs)[2]
    portable = build_all_frame_manifests(BallFrameStore(stage, external_payloads=source_payloads(stage, source)), configs)[2]
    return original, portable


def test_receipt_matches_remapped_real_manifests(manifests: tuple[dict[str, Any], dict[str, Any]]) -> None:
    source, portable = manifests
    for mode in ("train", "evaluation"):
        result = verifier.compare_manifests(source[mode], portable[mode])
        assert result["clips"] == result["remapped_clip_indices"] == 2


@pytest.mark.parametrize("damage", ["interval", "kind", "pack", "owner", "notes", "rgb", "summary", "duplicate", "missing"])
def test_receipt_rejects_changed_membership_or_evidence(
    manifests: tuple[dict[str, Any], dict[str, Any]], damage: str,
) -> None:
    original, portable = manifests
    mode = "train" if damage == "pack" else "evaluation"
    changed = copy.deepcopy(portable[mode])
    record = changed["clips"][0]
    if damage == "interval":
        record["common_intervals"][0] = (0, 7)
    elif damage == "kind":
        record["label_runs"][0][2] = 3
    elif damage == "pack":
        record["packs"].clear()
    elif damage == "owner":
        record["windows"][0]["owner_range"][1] -= 1
    elif damage == "notes":
        record["negative_evidence"]["excluded_noted_empty_frames"] = 1
    elif damage == "rgb":
        record["rgb_shard_sha256"] = "changed"
    elif damage == "summary":
        changed["summaries"]["train"]["teacher_identity_sha256"] = "changed"
    elif damage == "duplicate":
        changed["clips"].append(copy.deepcopy(record))
    else:
        changed["clips"].pop()
    with pytest.raises(ValueError):
        verifier.compare_manifests(original[mode], changed)
