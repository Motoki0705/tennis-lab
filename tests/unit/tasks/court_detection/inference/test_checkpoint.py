"""Inference-only parsing and strict loading without a Lightning training run."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.inference.checkpoint import (
    CourtInferenceSpec,
    file_sha256,
    load_court_checkpoint,
)
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from tests.unit.tasks.court_detection.inference.test_pose_output_predictors import (
    _bundle,
)
from tests.unit.tasks.court_detection.test_configuration import _compose


def test_inference_does_not_fabricate_unused_training_configuration() -> None:
    config = OmegaConf.to_container(_compose("synthetic_court"), resolve=True)
    assert isinstance(config, dict)
    config.pop("run")
    before = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    spec = CourtInferenceSpec.from_checkpoint_config(
        config, serialize_target_bundle(_bundle())
    )
    assert spec.target_bundle.kinds == ("kp", "seg", "line")
    assert spec.short_side > 0
    assert config == before
    with pytest.raises(Exception, match="run"):
        CourtTrainingConfig.from_config(config)


def test_saved_encoder_assets_use_the_explicit_external_asset_root(
    tmp_path: Path,
) -> None:
    config = OmegaConf.to_container(
        _compose(
            "synthetic_court", "model/encoder=dinov3", f"paths.project_root={tmp_path}"
        ),
        resolve=True,
    )
    assert isinstance(config, dict)
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(config["paths"], repository_root=tmp_path)
    )
    spec = CourtInferenceSpec.from_checkpoint_config(
        config, serialize_target_bundle(_bundle()), resolver=resolver
    )
    for path in (
        spec.model.encoder.repository_path,
        spec.model.encoder.checkpoint_path,
    ):
        assert path is not None
        assert resolver.validate(PathRole.EXTERNAL_ASSET, path) == path
    assert spec.model.encoder.repository_path == tmp_path / "third_party/dinov3"


@pytest.mark.parametrize("change", ["none", "missing", "unexpected", "foreign_prefix"])
def test_loader_requires_the_complete_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    import src.tasks.court_detection.inference.checkpoint as module

    model = torch.nn.Linear(1, 1)
    spec = SimpleNamespace(
        model=SimpleNamespace(encoder=SimpleNamespace(checkpoint_path=None)),
        loss=None,
        target_bundle=_bundle(),
        short_side=256,
        architecture={"name": "fixture"},
    )
    monkeypatch.setattr(
        CourtInferenceSpec, "from_checkpoint_config", lambda *args, **kwargs: spec
    )
    monkeypatch.setattr(
        module,
        "build_court_inference_pair",
        lambda **kwargs: SimpleNamespace(model=model),
    )
    state: dict[str, Any] = {
        "model.weight": torch.full((1, 1), 3.0),
        "model.bias": torch.full((1,), 2.0),
    }
    if change == "missing":
        del state["model.bias"]
    elif change == "unexpected":
        state["model.untrained_extra"] = torch.ones(1)
    elif change == "foreign_prefix":
        state["optimizer.value"] = torch.ones(1)
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {"config": {}, "target_bundle_state": {}},
            "state_dict": state,
        },
        path,
    )
    before = file_sha256(path)
    if change == "none":
        loaded = load_court_checkpoint(path)
        assert loaded.identity["checkpoint_sha256"] == before
        assert float(model.weight[0, 0]) == 3
        assert float(model.bias[0]) == 2
    else:
        with pytest.raises((RuntimeError, CourtModelIOError)):
            load_court_checkpoint(path)
    assert file_sha256(path) == before
