import hashlib
import json
from copy import deepcopy
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest


@pytest.fixture
def continuation(monkeypatch):
    directory = Path("knowledge/runs/run-court-scene-diversity-colab-s42").resolve()
    monkeypatch.syspath_prepend(str(directory))
    return run_path(str(directory / "continue_campaign.py"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def report(continuation):
    train, val = continuation["expected_counts"](2)
    synthetic = [f"synthetic-{n}" for n in range(500)]
    real = [f"real-{n}" for n in range(2211)]
    identity = {
        "scene_images": train,
        "synthetic_validation": {
            "scene_images": val,
            "images": 500,
            "ordered_ids_sha256": digest(synthetic),
        },
        "accepted_images": 5500,
        "training_seed": 42,
    }
    splits = {}
    for name, ids, pose in [
        ("synthetic_court-val", synthetic, True),
        ("tennis_court_detector-val", real, False),
    ]:
        splits[name] = {
            "sample_ids": ids,
            "count": len(ids),
            "pose_supervised": pose,
            "ordered_sample_ids_sha256": digest(ids),
            "batch_size": 8,
            "metrics": {
                key: 0.5
                for key in [
                    "kp_mean_distance_px",
                    "kp_median_distance_px",
                    "seg_miou",
                    "line_dice",
                    "semantic_line_miou",
                ]
            },
        }
    return {
        "schema": "court_data_scaling_evaluation_v1",
        "series": "scene-diversity",
        "global_step": 33160,
        "checkpoint_sha256": "f" * 64,
        "data_condition": identity,
        "splits": splits,
    }


def verify(continuation, report, identity=None, order=None):
    return continuation["validate_evaluation"](
        report,
        2,
        "f" * 64,
        identity or deepcopy(report["data_condition"]),
        order
        or report["splits"]["tennis_court_detector-val"]["ordered_sample_ids_sha256"],
    )


def test_valid_separate_domains_are_preserved(continuation, report):
    result = verify(continuation, report)
    assert result["synthetic_court-val"]["count"] == 500
    assert result["tennis_court_detector-val"]["count"] == 2211
    assert "sample_ids" not in result["synthetic_court-val"]


@pytest.mark.parametrize(
    "change", ["checkpoint", "step", "count", "duplicate", "order", "pose", "nan"]
)
def test_evaluation_gate_rejects_invalid_identity_or_metrics(
    continuation, report, change
):
    original = deepcopy(report["data_condition"])
    real_order = report["splits"]["tennis_court_detector-val"][
        "ordered_sample_ids_sha256"
    ]
    syn = report["splits"]["synthetic_court-val"]
    real = report["splits"]["tennis_court_detector-val"]
    if change == "checkpoint":
        report["checkpoint_sha256"] = "e" * 64
    elif change == "step":
        report["global_step"] = 29844
    elif change == "count":
        syn["count"] = 499
    elif change == "duplicate":
        syn["sample_ids"][1] = syn["sample_ids"][0]
    elif change == "order":
        real["sample_ids"].reverse()
        real["ordered_sample_ids_sha256"] = digest(real["sample_ids"])
    elif change == "pose":
        real["metrics"]["pose_translation_l2_m"] = 0.0
    else:
        syn["metrics"]["seg_miou"] = float("nan")
    with pytest.raises(ValueError):
        verify(continuation, report, original, real_order)


def test_three_scene_rounding_contract(continuation):
    train, val = continuation["expected_counts"](3)
    assert train == {"B00": 1833, "B01": 1833, "B02": 1834}
    assert val == {"B00": 167, "B01": 167, "B02": 166}


def test_only_same_condition_resume_is_allowed(continuation):
    expected: dict[str, Any] = {"run": {"resume": None}, "model": {"depth": 8}}
    actual = deepcopy(expected)
    actual["run"]["resume"] = {
        "role": "artifact",
        "path": "court_detection/train/scene-diversity-6000-s384-colab/scene-2-s42/logs/version_0/checkpoints/update-00026528.ckpt",
    }
    assert continuation["validate_config"](expected, actual, 2) == {"run.resume"}
    with pytest.raises(ValueError):
        continuation["validate_config"](expected, actual, 3)
    actual["model"]["depth"] = 4
    with pytest.raises(ValueError):
        continuation["validate_config"](expected, actual, 2)


@pytest.mark.parametrize("matches", [True, False])
def test_next_phase_requires_successful_drive_content_verification(
    continuation, tmp_path, monkeypatch, matches
):
    import src.utils.artifact_store as stores

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def sync_tree(self):
            pass

    def run(command, *, cwd, stdout, check):
        stdout.write(json.dumps({"matches": matches}))

    publish = continuation["publish_verified"]
    monkeypatch.setitem(publish.__globals__, "REPO", tmp_path)
    monkeypatch.setattr(stores, "RcloneArtifactStore", Store)
    monkeypatch.setattr(publish.__globals__["subprocess"], "run", run)
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "not-a-credential-fixture"))
    root = tmp_path / "outputs/result"
    root.mkdir(parents=True)
    receipt = tmp_path / "verification.json"
    if matches:
        publish(root, receipt)
    else:
        with pytest.raises(ValueError, match="Drive verification"):
            publish(root, receipt)
