"""External Court assets participate in sharing without constructing a model."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.inference.checkpoint import CourtInferenceSpec
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.model_assets import asset_hashes
from tests.unit.tasks.court_detection.inference.test_pose_output_predictors import (
    _bundle,
)
from tests.unit.tasks.court_detection.test_configuration import _compose
from tests.unit.tennis_scene.pipeline.config_factories import make_court_kp_config


def external_court_pair(tmp_path: Path) -> tuple[CourtKPModule, CourtKPModule, list[Path], list[Path]]:
    configs = [make_court_kp_config(tmp_path / name) for name in ("first", "second")]
    checkpoint = configs[0].checkpoint
    checkpoint.parent.mkdir(parents=True)
    config = OmegaConf.to_container(_compose("synthetic_court"), resolve=True)
    bundle = serialize_target_bundle(_bundle())
    torch.save({"hyper_parameters": {"config": config, "target_bundle_state": bundle}}, checkpoint)
    configs[1] = replace(configs[1], checkpoint=checkpoint)
    backbones, implementations = [], []
    for cfg in configs:
        spec = CourtInferenceSpec.from_checkpoint_config(config, bundle, resolver=cfg.resolver)
        backbone, repo = spec.model.encoder.checkpoint_path, spec.model.encoder.repository_path
        assert backbone is not None and repo is not None
        backbone.parent.mkdir(parents=True, exist_ok=True)
        backbone.write_bytes(b"backbone-a")
        repo.mkdir(parents=True)
        implementation = repo / "hubconf.py"
        implementation.write_text("# implementation a\n")
        (repo / "dinov3").mkdir()
        (repo / "dinov3" / "model.py").write_text("# nested implementation a\n")
        backbones.append(backbone)
        implementations.append(implementation)
    return CourtKPModule(configs[0]), CourtKPModule(configs[1]), backbones, implementations


@pytest.mark.parametrize("change", ["none", "backbone", "implementation", "missing_backbone", "missing_repository"])
def test_external_assets_define_court_sharing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    first, second, backbones, implementations = external_court_pair(tmp_path)
    if change == "backbone":
        backbones[1].write_bytes(b"backbone-b")
    elif change == "implementation":
        implementations[1].write_text("# implementation b\n")
    created: list[Any] = []
    def create() -> object:
        value = object()
        created.append(value)
        return value
    monkeypatch.setattr(first, "_load_predictor", create)
    monkeypatch.setattr(second, "_load_predictor", create)
    a, b = ("first", "court"), ("second", "court")
    declarations = {a: first.model_specs(), b: second.model_specs()}
    with InferenceRuntime(declarations, cleanup=lambda _: None) as runtime:
        with runtime.node(a) as models:
            original = models.borrow("court")
        if change.startswith("missing"):
            if change == "missing_backbone":
                backbones[1].unlink()
            else:
                import shutil
                shutil.rmtree(implementations[1].parent)
            with pytest.raises(FileNotFoundError), runtime.node(b) as models:
                models.borrow("court")
        else:
            with runtime.node(b) as models:
                other = models.borrow("court")
            assert (original is other) == (change == "none")
    assert len(created) == (2 if change in {"backbone", "implementation"} else 1)


@pytest.mark.parametrize("change", ["backbone", "source_modified", "source_added", "source_removed"])
def test_court_asset_changes_after_binding_fail_before_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    module, _, backbones, implementations = external_court_pair(tmp_path)
    bound = asset_hashes({"assets": module.asset_identities()})
    source = implementations[0].parent / "dinov3" / "model.py"
    if change == "backbone":
        backbones[0].write_bytes(b"backbone-b")
    elif change == "source_added":
        source.with_name("extra.py").write_text("# extra implementation\n")
    elif change == "source_removed":
        source.unlink()
    else:
        source.write_text("# modified implementation\n")
    created: list[bool] = []
    monkeypatch.setattr(module, "_load_predictor", lambda: created.append(True))
    node = ("clip", "court")
    runtime = InferenceRuntime({node: module.model_specs()}, expected_assets={node: bound}, cleanup=lambda _: None)
    with pytest.raises((ValueError, FileNotFoundError)), runtime, runtime.node(node) as models:
        models.borrow("court")
    assert not created and not runtime.loaded_keys
