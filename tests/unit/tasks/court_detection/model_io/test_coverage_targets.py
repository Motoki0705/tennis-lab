"""Loss/validation contracts for fractional Court teacher coverage."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.tasks.court_detection.training.losses import DiceLoss
from tests.unit.tasks.court_detection.model_io.test_adapters import (
    _adapter,
    _batch,
    _bundle,
)


def _coverage_adapter():
    bundle = _bundle("seg")
    bundle = CourtTargetBundleSpec(
        {"seg": replace(bundle.targets["seg"], target_dtype=torch.float32)}
    )
    return _adapter(bundle), _batch(bundle)


def test_one_hot_coverage_loss_matches_integer_labels_and_soft_loss_backpropagates() -> (
    None
):
    torch.manual_seed(4)
    labels = torch.randint(0, 3, (1, 8, 8))
    coverage = F.one_hot(labels, 3).permute(0, 3, 1, 2).float()
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    hard = F.cross_entropy(logits, labels) + DiceLoss(3)(logits, labels)
    soft = F.cross_entropy(logits, coverage) + DiceLoss(3, coverage_targets=True)(
        logits, coverage
    )
    torch.testing.assert_close(hard, soft)
    adapter, batch = _coverage_adapter()
    batch["targets"]["seg"] = coverage * 0.75 + coverage.roll(1, dims=1) * 0.25
    call = adapter.prepare_training_batch(batch)
    result = adapter.training_result({"seg": logits}, call)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert (
        logits.grad is not None
        and torch.isfinite(logits.grad).all()
        and logits.grad.count_nonzero()
    )


@pytest.mark.parametrize("invalid", ["shape", "sum", "negative", "nan"])
def test_invalid_coverage_is_rejected_before_loss(invalid: str) -> None:
    adapter, batch = _coverage_adapter()
    values = torch.zeros(1, 3, 8, 8)
    values[:, 0] = 1
    if invalid == "shape":
        values = values[:, 0]
    elif invalid == "sum":
        values *= 0.5
    elif invalid == "negative":
        values[0, 1, 0, 0] = -0.25
    elif invalid == "nan":
        values[0, 1, 0, 0] = float("nan")
    batch["targets"]["seg"] = values
    with pytest.raises(CourtModelIOError, match="coverage"):
        adapter.prepare_training_batch(batch)
