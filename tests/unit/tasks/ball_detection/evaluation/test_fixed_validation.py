import copy
import json
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.evaluation.coordinates import (
    CoordinateMetrics,
    frame_membership_sha256,
)
from src.tasks.ball_detection.evaluation.fixed_validation import (
    compare_full_models,
    verify_support,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.learning import (
    full_model_checkpoint,
    training_manifests,
)


def test_equal_models_with_different_training_data_share_exact_fixed_validation(tmp_path: Path) -> None:
    baseline = full_model_checkpoint(tmp_path / "parent")
    candidate = tmp_path / "candidate.pt"
    saved = torch.load(baseline, weights_only=True)
    saved["recipe"]["manifest_sha256"] = "1" * 64
    torch.save(saved, candidate)
    _, validation = training_manifests(tmp_path)
    result = compare_full_models(baseline, candidate, validation, baseline_sha256=dual_sha256(baseline),
        candidate_sha256=dual_sha256(candidate), validation_sha256=dual_sha256(validation),
        runtime=CoordinateRuntime(input_verification="upfront"), device=torch.device("cpu"), batch_size=1)
    assert result["candidate_minus_baseline_px"] == dict(common=0., full=0.)
    assert result["baseline"]["evaluation"] == result["candidate"]["evaluation"]
    assert result["baseline"]["training_manifest_sha256"] != result["candidate"]["training_manifest_sha256"]
    assert result["support"]["common"]["frame_fps_pairs"] == 32
    assert result["support"]["common"]["unique_observed_frames"] == 32
    assert result["support"]["common"]["membership_sha256"] == frame_membership_sha256((1, "val/clip", i) for i in range(32))
    assert result["test_usage"] == "none" and result["profile"] == "clean"
    changed = copy.deepcopy(result["candidate"]["evaluation"])
    changed["scopes"]["common"]["membership_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frame membership"):
        verify_support(changed, result["support"])


@pytest.mark.parametrize("fault", ["hash", "common", "missing_fps"])
def test_invalid_fixed_evaluation_fails_before_loading_models(tmp_path: Path, fault: str) -> None:
    _, validation = training_manifests(tmp_path)
    raw = json.loads(validation.read_text())
    if fault == "common":
        raw["clips"][0]["common_evaluation"] = False
    elif fault == "missing_fps":
        raw["sampling"]["frame_steps"] = [1, 2]
    validation.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash changed|lacks teachers"):
        compare_full_models(tmp_path / "missing-parent", tmp_path / "missing-candidate", validation,
            baseline_sha256="a" * 64, candidate_sha256="b" * 64,
            validation_sha256="c" * 64 if fault == "hash" else dual_sha256(validation),
            runtime=CoordinateRuntime(), device=torch.device("cpu"), batch_size=1)


def test_equal_counts_on_different_frames_do_not_share_membership_hash() -> None:
    results = []
    for start in (0, 1):
        metrics = CoordinateMetrics((1,))
        metrics.add(dict(clip_id=["clip"], frame_step=[1], start=[start], common_evaluation=[True], source=["tracknet"],
            frame_indices=(torch.arange(32) + start)[None], position_valid=torch.ones(1, 32, dtype=torch.bool)), torch.zeros(1, 32))
        results.append(metrics.report()["scopes"]["common"])
    assert results[0]["frame_fps_pairs"] == results[1]["frame_fps_pairs"]
    assert results[0]["membership_sha256"] != results[1]["membership_sha256"]


def test_candidate_cannot_silently_change_input_preprocessing(tmp_path: Path) -> None:
    baseline = full_model_checkpoint(tmp_path / "parent")
    candidate = tmp_path / "changed-input.pt"
    saved = torch.load(baseline, weights_only=True)
    saved["recipe"]["input_contract"]["mdd_a"] = .4
    saved["recipe"]["parent"]["input_contract"] = saved["recipe"]["input_contract"]
    torch.save(saved, candidate)
    _, validation = training_manifests(tmp_path)
    with pytest.raises(ValueError, match="architecture/preprocessing differs"):
        compare_full_models(baseline, candidate, validation, baseline_sha256=dual_sha256(baseline),
            candidate_sha256=dual_sha256(candidate), validation_sha256=dual_sha256(validation),
            runtime=CoordinateRuntime(), device=torch.device("cpu"), batch_size=1)
