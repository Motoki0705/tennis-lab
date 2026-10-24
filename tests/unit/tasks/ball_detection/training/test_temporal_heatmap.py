"""Update-wide masking, model-independent pack transforms and explicit budgets."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.data.windows.dataset import (
    FrozenBallPackDataset,
    collate_pack_samples,
)
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
    UpdateBudget,
)
from src.tasks.ball_detection.training.temporal_heatmap.packs import (
    PackSampler,
    augment_pack,
    supervised_loss_sum,
)
from tests.support.tasks.ball_detection.temporal_heatmap import (
    model_yaml,
    prepare_fixture,
    update_budget,
)


def test_accumulated_loss_and_gradients_use_total_teacher_count_including_empty_child() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    source = torch.randn(4, 32, 6, 6)
    positive = torch.zeros(4, 32, dtype=torch.bool)
    valid = positive.clone()
    positive[0, 0] = True
    positive[1, :5] = True
    valid[0, 0] = True
    valid[1, :12] = True
    valid[3, :31] = True
    batch = dict(uv=torch.full((4, 32, 2), .5), position_valid=positive, heatmap_valid=valid)
    dense = source.clone().requires_grad_()
    expected, _ = heatmap_objective(dense, batch)
    expected.backward()
    accumulated = source.clone().requires_grad_()
    numerator, count = 0., 0
    for child in range(4):
        part = {key: value[child:child + 1] for key, value in batch.items()}
        loss_sum, teachers = supervised_loss_sum(accumulated[child:child + 1], part, sigma_ratio=.012, gamma=2.)
        loss_sum.backward()
        numerator += float(loss_sum.detach())
        count += teachers
    assert count == 44
    assert numerator / count == pytest.approx(float(expected.detach()), abs=1e-7)
    assert accumulated.grad is not None
    accumulated.grad.div_(count)
    torch.testing.assert_close(accumulated.grad, dense.grad, rtol=2e-6, atol=1e-8)
    assert accumulated.grad[2].count_nonzero() == 0


def test_pack_augmentation_is_identical_before_partition_at_every_t(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    prepared = prepare_fixture(tmp_path)
    cfg = AugmentationConfig.load(Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"))
    cfg = replace(cfg, camera_probability=1., occlusion_probability=0.)
    results = []
    for frames in (32, 64, 128):
        dataset = FrozenBallPackDataset(prepared / f"train_T{frames}.json", split="train")
        batch = collate_pack_samples([dataset[0]])
        rgb, transformed, audit = augment_pack(batch, batch["rgb"], VideoAugmenter(cfg, 42), position=9)
        results.append((rgb.flatten(0, 1), transformed["uv"].flatten(0, 1), transformed["label_kind"].flatten(), audit))
    for result in results[1:]:
        for first, second in zip(results[0][:-1], result[:-1], strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert result[-1] == results[0][-1]


def test_sampler_resume_ignores_global_model_rng_and_prefetch() -> None:
    expected = list(PackSampler(11, seed=42, position=0, count=20))
    torch.manual_seed(100)
    torch.rand(1000)
    prefetched = iter(PackSampler(11, seed=42, position=0, count=20))
    assert [next(prefetched) for _ in range(9)] == expected[:9]
    # Only two draws were committed, even if a loader prefetched nine.
    assert list(PackSampler(11, seed=42, position=2, count=18)) == expected[2:]


@pytest.mark.parametrize("field,value", [("input_frames", 127), ("input_frames", 129), ("updates", 0),
    ("warmup_updates", 3), ("evaluation_interval", 0), ("checkpoint_interval", 0), ("learning_rate", float("nan"))])
def test_invalid_budget_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        update_budget(**{field: value})


def test_missing_budget_has_no_training_default(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        UpdateBudget()  # type: ignore[call-arg]
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train", "--stage", "temporal_heatmap",
                             "--architecture", "dpt", "--phase", "pretrain", "--output", str(tmp_path / "run")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--updates" in result.stderr and "--warmup-updates" in result.stderr
    assert not (tmp_path / "run").exists()


def test_training_cli_validates_disjoint_inputs_and_optional_paths_before_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.training.temporal_heatmap import cli

    paths = {name: tmp_path / name / filename for name, filename in (
        ("manifest", "train.json"), ("evaluation_manifest", "val.json"), ("model_config", "model.yaml"),
        ("augmentation_config", "augmentation.yaml"), ("initial_artifact", "parent.pt"),
    )}
    for path in paths.values():
        path.parent.mkdir()
        path.touch()
    paths["model_config"].write_text(yaml.safe_dump(model_yaml("dpt", frames=64)))
    augmentation = Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    paths["augmentation_config"].write_text(augmentation.read_text())
    output = tmp_path / "new-run" / "output"
    values = {key.replace("_", "-"): str(path) for key, path in paths.items()}
    values.update({"architecture": "dpt", "phase": "finetune", "output": str(output), "augmentation": "video",
                   "initial-artifact-sha256": "f" * 64, "parent-update": "3", "distance-px": "20",
                   "device": "cpu", "precision": "fp32", "jpeg-decoder": "opencv", "compile-mode": "off",
                   "input-verification": "lazy"})
    values.update({key.replace("_", "-"): str(value) for key, value in vars(update_budget()).items()})
    called: dict[str, Any] = {}

    def capture(**kwargs: Any) -> None:
        called.update(kwargs)

    monkeypatch.setattr(cli, "train_temporal_heatmap", capture)
    argv = ["train", *(item for name, value in values.items() for item in (f"--{name}", value))]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert all(called[name] == paths[name] for name in ("manifest", "evaluation_manifest", "initial_artifact"))
    assert called["augmentation"] is not None and called["resume"] is None and called["output"] == output
    assert not output.parent.exists()
    called.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--resume", "relative/checkpoint.pt"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not called


@pytest.mark.parametrize("calibration", ["relative/calibration.json", "missing"])
def test_evaluation_cli_rejects_invalid_optional_calibration_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calibration: str,
) -> None:
    from src.tasks.ball_detection.evaluation import temporal_heatmap_cli as cli
    from src.utils.configuration import PathContractError

    artifact, manifest = tmp_path / "weights.pt", tmp_path / "evaluation.json"
    artifact.touch()
    manifest.touch()
    output = tmp_path / "new-output" / "report.json"
    selected = str(tmp_path / "missing.json") if calibration == "missing" else calibration
    monkeypatch.setattr(sys, "argv", ["evaluate", "--artifact", str(artifact), "--manifest", str(manifest),
        "--output", str(output), "--split", "test", "--calibration", selected, "--distance-px", "20",
        "--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv", "--compile-mode", "off",
        "--input-verification", "lazy"])

    def unexpected_runtime(_: Any) -> None:
        pytest.fail("Runtime must not be configured before path validation")

    monkeypatch.setattr(cli, "runtime_from_arguments", unexpected_runtime)
    with pytest.raises((SystemExit, PathContractError)):
        cli.main()
    assert not output.parent.exists()


def test_comparison_cli_accepts_independent_reports_and_rejects_duplicate_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.scripts import visualize_temporal_heatmaps as cli
    from src.utils.configuration import PathContractError

    reports = [tmp_path / name / "evaluation.json" for name in ("first-run", "second-run")]
    for report in reports:
        report.parent.mkdir()
        report.touch()
    output = tmp_path / "new-output" / "comparison.mp4"
    called: dict[str, Any] = {}

    def capture(evaluations: list[Path], **kwargs: Any) -> None:
        called.update(evaluations=evaluations, **kwargs)

    monkeypatch.setattr(cli, "render_temporal_heatmap_comparison", capture)
    argv = ["visualize", "--evaluations", *(str(path) for path in reports), "--clip-id", "clip",
            "--start", "0", "--stop", "32", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert called["evaluations"] == reports and called["output"] == output
    assert not output.parent.exists()
    called.clear()
    argv[3] = argv[2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(PathContractError, match="duplicate paths"):
        cli.main()
    assert not called and not output.parent.exists()
