"""CPU fixture coverage for the v4 migration's narrowly scoped audit exceptions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.checkpoint_warning import (
    checkpoint_warning_policy,
)

ROOT = Path(__file__).resolve().parents[4]
DRIVER = ROOT / "knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/reuse.py"
spec = importlib.util.spec_from_file_location("observation_reuse_v4", DRIVER)
assert spec is not None and spec.loader is not None
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)
PIN = "a" * 64
ACTUAL = "b" * 64
PINS = dict.fromkeys(("court", "dino", "vitpose", "plcs", "blcs", "dinov3"), PIN)


@pytest.mark.parametrize("role", ["dino", "vitpose"])
def test_checkpoint_difference_warns_without_rewriting_evidence(tmp_path, role):
    model = str(tmp_path / "model.pth")
    video = str(tmp_path / "clip.mp4")
    before = {model: PIN, video: PIN}
    after = {model: ACTUAL, video: PIN}
    warning = tmp_path / "warnings.jsonl"
    with checkpoint_warning_policy(PINS, [role], warning):
        reuse.verify_input_snapshot(
            before, after, {model: role}, phase="prepublication"
        )
    assert before == {model: PIN, video: PIN}
    assert after == {model: ACTUAL, video: PIN}
    records = [json.loads(line) for line in warning.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["observed_sha256"] == ACTUAL
    assert records[0]["expected_sha256"] == PIN
    assert records[0]["path"] == model


@pytest.mark.parametrize("kind", ["video", "src", "receipt", "court", "unmapped_model"])
def test_noncheckpoint_changes_remain_fatal(tmp_path, kind):
    path = str(tmp_path / kind)
    with (
        checkpoint_warning_policy(
            PINS, ["dino", "vitpose"], tmp_path / "warnings.jsonl"
        ),
        pytest.raises(ValueError, match="Input changed"),
    ):
        reuse.verify_input_snapshot(
            {path: PIN}, {path: ACTUAL}, {}, phase="postpublication"
        )


def test_input_set_change_and_unapproved_roles_are_fatal(tmp_path):
    with checkpoint_warning_policy(PINS, ["dino"], tmp_path / "warnings.jsonl"):
        with pytest.raises(ValueError, match="Input set changed"):
            reuse.verify_input_snapshot({"a": PIN}, {}, {}, phase="prepublication")
        with pytest.raises(ValueError, match="Only DINO/ViTPose"):
            reuse.verify_input_snapshot(
                {"a": PIN}, {"a": ACTUAL}, {"a": "court"}, phase="prepublication"
            )


def test_no_implicit_warning_outside_explicit_policy():
    with pytest.raises(ValueError, match="Input changed"):
        reuse.verify_input_snapshot(
            {"model": PIN},
            {"model": ACTUAL},
            {"model": "dino"},
            phase="postpublication",
        )


@pytest.mark.parametrize("key", reuse.historical.SELECTION_KEYS)
@pytest.mark.parametrize("change", ["dtype", "shape", "value"])
def test_all_selection_arrays_still_require_exact_equality(key, change):
    old: dict[str, np.ndarray] = {
        name: np.zeros((2, 3), dtype=np.float32)
        for name in reuse.historical.SELECTION_KEYS
    }
    new = {name: value.copy() for name, value in old.items()}
    if change == "dtype":
        new[key] = new[key].astype(np.float64)
    elif change == "shape":
        new[key] = np.zeros((3, 2), dtype=np.float32)
    else:
        new[key][0, 0] = 1
    assert set(reuse.historical.exact_differences(old, new)) == {key}
