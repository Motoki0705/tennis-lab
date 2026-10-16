from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from src.tasks.ball_detection.models.mdd import (
    DeepMDDHybridDetector,
    MDDHybridConfig,
    MDDHybridOutput,
)
from src.tasks.ball_detection.training.hybrid.configuration import resolve_recipe
from src.tasks.ball_detection.training.hybrid.objective import hybrid_objective
from src.tasks.ball_detection.training.hybrid.runner import (
    freeze_for_epoch,
    parameter_role,
)
from tests.support.tasks.ball_detection.hybrid import hybrid_recipe


@pytest.mark.parametrize("field", ["presence_threshold", "loss.heatmap_weight", "loss.presence_weight",
                                  "freeze.encoder_epochs", "freeze.query_epochs", "absence.negative_kinds",
                                  "absence.window_weights.play", "absence.window_weights.absent", "absence.window_weights.nonplay"])
def test_undecided_values_are_mandatory(field: str, tmp_path: Path) -> None:
    cfg = hybrid_recipe(tmp_path / "data.json", tmp_path / "query.pt", tmp_path / "run")
    OmegaConf.update(cfg, field, "???")
    with pytest.raises(MissingMandatoryValue):
        resolve_recipe(cfg)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(("field", "value"), [("presence_threshold", 0.), ("presence_threshold", float("nan")),
    ("loss.heatmap_weight", -1.), ("loss.presence_weight", 0.), ("freeze.query_epochs", True),
    ("freeze.encoder_epochs", 3), ("absence.negative_kinds", ["unresolved"]),
    ("absence.window_weights.absent", .8)])
def test_invalid_decisions_fail(field: str, value: object, tmp_path: Path) -> None:
    cfg = hybrid_recipe(tmp_path / "data.json", tmp_path / "query.pt", tmp_path / "run")
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        resolve_recipe(cfg)


def test_absence_supervises_every_heatmap_cell_and_presence_but_unknown_is_ignored() -> None:
    heatmaps = torch.zeros(1, 3, 4, 4, requires_grad=True)
    presence = torch.zeros(1, 3, requires_grad=True)
    batch = dict(uv=torch.full((1, 3, 2), .5), position_valid=torch.tensor([[True, False, False]]),
                 heatmap_valid=torch.tensor([[True, True, False]]))
    loss, parts = hybrid_objective(MDDHybridOutput(heatmaps, presence), batch,
        loss_config=dict(heatmap_weight=1., presence_weight=2., sigma_ratio=.1, focal_gamma=2.))
    torch.testing.assert_close(loss, parts["heatmap"] + 2 * parts["presence"])
    loss.backward()
    assert heatmaps.grad is not None and bool((heatmaps.grad[0, 1] > 0).all())
    assert presence.grad is not None and presence.grad[0, 0] < 0 and presence.grad[0, 1] > 0
    assert bool((heatmaps.grad[0, 2] == 0).all()) and presence.grad[0, 2] == 0


def test_independent_freezes_preserve_upstream_gradients(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    raw = resolve_recipe(hybrid_recipe(tmp_path / "data.json", tmp_path / "query.pt", tmp_path / "run"))
    model = DeepMDDHybridDetector(MDDHybridConfig(**raw["model"]))
    freeze_for_epoch(model, epoch=0, freeze=dict(encoder_epochs=0, query_epochs=1))
    assert model.encoder.training and not model.blocks.training
    output = model(torch.randint(256, (1, 32, 3, 32, 32), dtype=torch.uint8), torch.arange(32)[None].float() / 30)
    (output.heatmap_logits.mean() + output.presence_logits.mean()).backward()
    roles = {role: [p for n, p in model.named_parameters() if parameter_role(n) == role] for role in ("encoder", "query", "new")}
    assert all(p.grad is None for p in roles["query"])
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in roles["encoder"])
    freeze_for_epoch(model, epoch=1, freeze=dict(encoder_epochs=2, query_epochs=1))
    assert not model.encoder.training and model.blocks.training
    assert all(not p.requires_grad for p in roles["encoder"])
    assert all(p.requires_grad for p in roles["query"])
