"""BLCS real-loss training connectivity and representative fault injections."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.configuration import parse_model_config
from tests.integration.tasks.blcs.model_contracts import AXIAL, CONFIG_DIR, SELECTORS
from tests.support.model_contracts.exercise import exercise_family
from tests.support.model_contracts.parameters import names_under
from tests.support.model_contracts.registry import (
    assert_selector_coverage,
    configured_selectors,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("gan", [False, True], ids=["supervised", "gan"])
def test_axial_real_loss_and_training_update(tmp_path: Path, gan: bool) -> None:
    exercise_family(AXIAL, tmp_path, gan=gan)


def test_every_public_model_selector_is_registered() -> None:
    assert_selector_coverage(configured_selectors(CONFIG_DIR), SELECTORS)


def test_unregistered_valid_model_selector_is_detected(tmp_path: Path) -> None:
    configs = copytree(CONFIG_DIR, tmp_path / "configs")
    (configs / "model" / "new_axial.yaml").write_text(
        "defaults:\n  - multiview_axial_small\n  - _self_\n"
    )
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        config = compose(config_name="train", overrides=["model=new_axial"])
    # This is a valid production selector, without constructing a large model.
    parse_model_config(config)
    with pytest.raises(AssertionError, match="unregistered=.*new_axial"):
        assert_selector_coverage(configured_selectors(configs), SELECTORS)


@pytest.mark.parametrize("fault", ["detach", "optimizer", "freeze", "omit_loss"])
def test_contract_detects_disconnected_velocity_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    def mutate(module: Any) -> None:
        head = module.model.output_head.velocity
        if fault == "detach":
            head.register_forward_hook(lambda mod, inputs, output: output.detach())
        elif fault == "optimizer":
            excluded = names_under(module, "model.output_head.velocity")
            monkeypatch.setattr(
                module,
                "optimizer_param_groups",
                lambda: [
                    {
                        "params": [
                            param
                            for name, param in module.named_parameters()
                            if name not in excluded
                        ],
                    }
                ],
            )
        elif fault == "freeze":
            head.requires_grad_(False)
        else:
            for term in ("velocity", "rollout_position", "rollout_velocity"):
                module.loss_fn.weights[term] = 0.0

    error = (
        "optimizer missing=.*velocity"
        if fault == "optimizer"
        else "required group did not update: velocity"
        if fault == "omit_loss"
        else "missing gradients.*velocity"
    )
    with pytest.raises(AssertionError, match=error):
        exercise_family(AXIAL, tmp_path, mutate=mutate)
