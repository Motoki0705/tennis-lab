"""Exercise the actual models/dataset/objective, not replacement training stubs."""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import pytest
import torch

from src.tasks.ball_detection.checkpoints.temporal_heatmap import initialize_finetuning
from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.temporal_heatmap import (
    LEGACY_EVALUATION_SCHEMA,
    evaluate_temporal_heatmap,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import restore_heatmap_model
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal_heatmap.runner import (
    train_temporal_heatmap,
)
from src.tasks.ball_detection.visualization.presentation.temporal_heatmap import (
    render_temporal_heatmap_comparison,
)
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.temporal_heatmap import (
    prepare_fixture,
    small_model,
    update_budget,
)


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    return Path(prepare_fixture(tmp_path, native_fps="30000/1001"))


def train(prepared: Path, output: Path, architecture: str, *, frames: int = 32, **kwargs: Any) -> dict[str, Any]:
    options = dict(architecture=architecture, phase="pretrain" if frames == 32 else "finetune",
                   manifest=prepared / f"train_T{frames}.json", evaluation_manifest=prepared / f"evaluation_T{frames}.json",
                   model_config=small_model(architecture, frames=frames), output=output,
                   budget=update_budget(), runtime=CoordinateRuntime(cpu_threads=1), device=torch.device("cpu"),
                   augmentation=None, distance_px=20.)
    options.update(kwargs)
    result: dict[str, Any] = train_temporal_heatmap(**options)
    return result


def checkpoint(output: Path) -> dict[str, Any]:
    latest = json.loads((output / "latest.json").read_text())
    saved: dict[str, Any] = torch.load(output / latest["checkpoint"], map_location="cpu", weights_only=True)
    return saved


def test_persistence_failure_stops_at_the_completed_checkpoint(prepared: Path, tmp_path: Path) -> None:
    output = tmp_path / "save-failure"
    stages = []

    def persist(directory: Path, stage: str) -> None:
        stages.append(stage)
        assert (directory / "run.json").is_file()
        assert (directory / "train_manifest.json").is_file()
        assert (directory / "evaluation_manifest.json").is_file()
        if stage == "checkpoint":
            assert checkpoint(directory)["updates_completed"] == 1
            raise ConnectionError("Injected Drive connection failure")

    with pytest.raises(ConnectionError, match="Injected Drive"):
        train(prepared, output, "hybrid", persist=persist)
    assert stages == ["initialized", "checkpoint"]
    assert [r["update"] for r in logs(output)] == [1]
    assert not (output / "COMPLETED.json").exists()
    assert not (output / "STOPPED.json").exists()
    train(prepared, output, "hybrid", resume=output / "checkpoint-update-00000001.pt")
    assert [r["update"] for r in logs(output)] == [1, 2, 3]


def logs(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]


def assert_state_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_state_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("architecture", ["dpt", "hybrid"])
def test_pretrain32_full_state_finetune64_and_fixed_test_evaluation(prepared: Path, tmp_path: Path, architecture: str) -> None:
    pre, post = tmp_path / "pre", tmp_path / "post"
    assert train(prepared, pre, architecture)["complete"]
    parent = pre / "weights-update-00000003.pt"
    model, receipt = initialize_finetuning(parent, expected_sha256=dual_sha256(parent), parent_update=3,
                                         config=small_model(architecture, frames=64))
    original = restore_heatmap_model(torch.load(parent, weights_only=True))
    assert_state_equal(original.state_dict(), model.state_dict())
    assert receipt["saved_frames"] == 32 and receipt["target_frames"] == 64
    assert train(prepared, post, architecture, frames=64, initial_artifact=parent,
                 initial_artifact_sha256=dual_sha256(parent), parent_update=3)["complete"]
    saved = checkpoint(post)
    assert saved["updates_completed"] == 3 and saved["sample_position"] == 3
    assert saved["recipe"]["initialization"]["artifact_sha256"] == dual_sha256(parent)
    assert logs(post)[0]["update"] == 1 and logs(post)[0]["sample_position"] == 1
    for a, b in zip(logs(pre), logs(post), strict=True):
        for key in ("pack_ids", "pack_indices", "input_frames", "teachers_by_kind", "sample_position"):
            assert a[key] == b[key]
    assert all(int(state["step"]) == 3 for state in saved["optimizer"]["state"].values())
    val = json.loads((post / "validation-update-00000003.json").read_text())
    artifact = post / "weights-update-00000003.pt"
    report = evaluate_temporal_heatmap(restore_heatmap_model(saved["model_artifact"]), prepared / "evaluation_T64.json",
        split="test", runtime=CoordinateRuntime(cpu_threads=1), device=torch.device("cpu"),
        artifact_sha256=dual_sha256(artifact), distance_px=20., calibration_report=val, output=post / "test.json")
    assert report["calibration"] == val["calibration"]
    legacy = {k: v for k, v in val.items() if k not in {"provenance", "evaluation_code"}}
    legacy["schema"] = LEGACY_EVALUATION_SCHEMA
    legacy_result = evaluate_temporal_heatmap(restore_heatmap_model(saved["model_artifact"]), prepared / "evaluation_T64.json",
        split="test", runtime=CoordinateRuntime(cpu_threads=1), device=torch.device("cpu"),
        artifact_sha256=dual_sha256(artifact), distance_px=20., calibration_report=legacy)
    assert legacy_result["calibration"] == val["calibration"]
    expected = json.loads((prepared / "evaluation_T32.json").read_text())["summaries"]["test"]
    assert all(expected[key] == value for key, value in report["identities"].items())
    metrics = report["metrics"]
    assert metrics["support"] == dict(frames=144, scored_frames=144, masked=0, observed=88, reviewed_empty=32, out_of_frame=24)
    assert metrics["mean_error_px"] is not None and metrics["median_error_px"] is not None and metrics["p95_error_px"] is not None
    assert set(metrics["by_negative_kind"]) == {"reviewed_empty", "out_of_frame"}
    assert set(metrics["by_edge_distance"]) == {"0", "1-7", "8-15", "16-31", "32-63", "64+"}
    assert metrics["by_source"]["tracknet"]["support"] == metrics["support"]
    assert report["predictions"]["frames"] == 144
    assert "NaN" not in (post / "test.json").read_text()
    video = tmp_path / "comparison.mp4"
    rendered = render_temporal_heatmap_comparison(
        [pre / "validation-update-00000003.json", post / "validation-update-00000003.json"],
        clip_id="tracknet/val/dense", start=0, stop=8, output=video, panel_width=160,
        snapshot_root=Path(val["snapshot"]["directory"]))
    assert rendered["frames"] == 8 and rendered["identities"] == val["identities"]
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 328
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(float(Fraction("30000/1001")), rel=1e-5)
    capture.release()
    with pytest.raises(ValueError, match="test requires"):
        evaluate_temporal_heatmap(model, prepared / "evaluation_T64.json", split="test", runtime=CoordinateRuntime(),
                                  device=torch.device("cpu"), artifact_sha256=dual_sha256(artifact), distance_px=20.)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_temporal_heatmap(model, prepared / "evaluation_T64.json", split="test", runtime=CoordinateRuntime(),
            device=torch.device("cpu"), artifact_sha256="different-checkpoint", distance_px=20., calibration_report=val)
    with pytest.raises(ValueError, match="Resume recipe changed"):
        train(prepared, post, architecture, frames=128, resume=post / "checkpoint-update-00000003.pt")
    store = BallFrameStore(Path(val["snapshot"]["directory"]))
    shard = store.image_path(store.clip_by_id("tracknet/val/dense"))
    payload = bytearray(shard.read_bytes())
    payload[len(payload) // 2] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(ValueError, match="RGB shard changed"):
        render_temporal_heatmap_comparison(
            [pre / "validation-update-00000003.json", post / "validation-update-00000003.json"],
            clip_id="tracknet/val/dense", start=0, stop=8, output=tmp_path / "changed-rgb.mp4",
            snapshot_root=Path(val["snapshot"]["directory"]))


def test_both_architectures_share_pack_draws_augmentation_and_input_budget(prepared: Path, tmp_path: Path) -> None:
    augmentation = AugmentationConfig.load(Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml").resolve())
    augmentation = replace(augmentation, camera_probability=1., occlusion_probability=0.)
    outputs = [tmp_path / "dpt", tmp_path / "hybrid"]
    for architecture, output in zip(("dpt", "hybrid"), outputs, strict=True):
        train(prepared, output, architecture, budget=update_budget(input_frames=256), augmentation=augmentation)
    keys = ("update", "pack_ids", "pack_indices", "sample_position", "input_frames", "cumulative_input_frames",
            "valid_teachers", "teachers_by_kind", "original_teachers_by_kind", "augmentation_sha256", "learning_rate")
    for first, second in zip(logs(outputs[0]), logs(outputs[1]), strict=True):
        assert {key: first[key] for key in keys} == {key: second[key] for key in keys}
        assert first["input_frames"] == 256 and first["child_windows"] == 8
        assert first["valid_teachers"] == sum(first["teachers_by_kind"][k] for k in ("observed", "reviewed_empty", "out_of_frame"))


def assert_resume_roundtrip(
    prepared: Path, tmp_path: Path, architecture: str, runtime: CoordinateRuntime,
) -> None:
    uninterrupted, interrupted = tmp_path / "continuous", tmp_path / "resumed"
    train(prepared, uninterrupted, architecture, runtime=runtime)
    stopped = train(prepared, interrupted, architecture, runtime=runtime, stop_after_update=1)
    assert stopped["updates_completed"] == 1 and not stopped["complete"]
    # Simulate an interruption after logging an update but before checkpoint publication.
    with (interrupted / "train.jsonl").open("ab") as stream:
        stream.write(b'{"update":2,"uncommitted":true}\ntruncated')
    train(prepared, interrupted, architecture, runtime=runtime, resume=interrupted / "checkpoint-update-00000001.pt")
    first, second = checkpoint(uninterrupted), checkpoint(interrupted)
    for key in ("model_artifact", "optimizer", "schedule", "rng", "updates_completed", "sample_position"):
        assert_state_equal(first[key], second[key])
    for a, b in zip(logs(uninterrupted), logs(interrupted), strict=True):
        for key in ("loss", "loss_sum", "pack_ids", "pack_indices", "teachers_by_kind", "augmentation_sha256", "learning_rate"):
            assert a[key] == b[key], key
    assert (interrupted / "COMPLETED.json").is_file()
    with pytest.raises(ValueError, match="latest committed checkpoint"):
        train(prepared, interrupted, architecture, runtime=runtime, resume=interrupted / "checkpoint-update-00000001.pt")
    content = (interrupted / "train.jsonl").read_bytes()
    (interrupted / "train.jsonl").write_bytes(b"X" + content[1:])
    with pytest.raises(ValueError, match="Committed train log changed"):
        train(prepared, interrupted, architecture, runtime=runtime, resume=interrupted / "checkpoint-update-00000003.pt")


@pytest.mark.parametrize("architecture", ["dpt", "hybrid"])
def test_resume_matches_uninterrupted_weights_optimizer_schedule_rng_and_pack_position(
    prepared: Path, tmp_path: Path, architecture: str,
) -> None:
    assert_resume_roundtrip(prepared, tmp_path, architecture,
        CoordinateRuntime(cpu_threads=1, num_workers=int(architecture == "hybrid"), prefetch_factor=2))


def test_hybrid_resume_with_inline_loading_matches_continuous_training(prepared: Path, tmp_path: Path) -> None:
    assert_resume_roundtrip(prepared, tmp_path, "hybrid", CoordinateRuntime(cpu_threads=1, num_workers=0, prefetch_factor=2))


def test_pretraining_cannot_change_t_and_finetuning_needs_explicit_parent(prepared: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="T=32"):
        train(prepared, tmp_path / "bad", "dpt", frames=64, phase="pretrain")
    with pytest.raises(ValueError, match="initial heatmap artifact"):
        train(prepared, tmp_path / "bad", "dpt", frames=64)
    assert not (tmp_path / "bad").exists()


def test_train_and_validation_must_pin_the_same_rgb_contents(prepared: Path, tmp_path: Path) -> None:
    raw = json.loads((prepared / "evaluation_T32.json").read_text())
    raw["clips"][0]["rgb_shard_sha256"] = "0" * 64
    changed = tmp_path / "changed-evaluation.json"
    changed.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="RGB shard identities differ"):
        train(prepared, tmp_path / "bad-rgb", "dpt", evaluation_manifest=changed)
    assert not (tmp_path / "bad-rgb").exists()


def test_resume_rejects_a_changed_source_identity(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.training.temporal_heatmap import runner

    output = tmp_path / "source-change"
    train(prepared, output, "dpt", stop_after_update=1)
    identity = runner.source_identity()
    identity["source_sha256"] = {**identity["source_sha256"], "changed.py": "0" * 64}
    monkeypatch.setattr(runner, "source_identity", lambda: identity)
    with pytest.raises(ValueError, match="Resume code or torch version changed"):
        train(prepared, output, "dpt", resume=output / "checkpoint-update-00000001.pt")
    assert checkpoint(output)["updates_completed"] == 1
