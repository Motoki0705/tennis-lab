"""Update-wide masking, model-independent pack transforms and explicit budgets."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

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
