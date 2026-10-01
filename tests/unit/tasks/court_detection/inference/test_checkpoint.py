"""Inference-only parsing and strict loading without a Lightning training run."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from src.tasks.court_detection.configuration import (
    CourtModelConfig,
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference.checkpoint import (
    CourtInferenceSpec,
    file_sha256,
    load_court_checkpoint,
    load_court_pair,
)
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
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


def test_saved_encoder_assets_separate_checkpoint_and_external_source_roots(
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
    repository, weights = spec.model.encoder.repository_path, spec.model.encoder.checkpoint_path
    assert repository is not None and weights is not None
    assert resolver.validate(PathRole.EXTERNAL_ASSET, repository) == repository
    assert resolver.validate(PathRole.CHECKPOINT, weights) == weights
    assert spec.model.encoder.repository_path == tmp_path / "third_party/dinov3"
    assert spec.backbone_asset_migration is None


@pytest.mark.parametrize('omegaconf_input', [False, True])
def test_legacy_saved_asset_layout_is_explicit_and_does_not_mutate_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, omegaconf_input: bool,
) -> None:
    config = OmegaConf.to_container(_compose('synthetic_court', f'paths.project_root={tmp_path}',
        'paths.checkpoint_root=outputs'), resolve=True)
    assert isinstance(config, dict)
    model = config['model']
    assert isinstance(model, dict)
    encoder = model['encoder']
    assert isinstance(encoder, dict)
    filename = Path(str(encoder['checkpoint_path'])).name
    encoder['checkpoint_path'] = f'dinov3/checkpoints/{filename}'
    before = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    saved = OmegaConf.create(config) if omegaconf_input else config
    spec = CourtInferenceSpec.from_checkpoint_config(saved, serialize_target_bundle(_bundle()))
    assert spec.model.encoder.checkpoint_path == tmp_path / 'ckpt/dinov3' / filename
    assert spec.architecture['encoder']['checkpoint_path'] == encoder['checkpoint_path']
    assert spec.backbone_asset_migration == {
        'layout': 'external_dinov3_to_checkpoint',
        'saved_path': f'dinov3/checkpoints/{filename}',
        'runtime_path': f'dinov3/{filename}',
        'checkpoint_root': str(tmp_path / 'ckpt'),
    }
    assert 'Migrating saved DINOv3 asset path' in caplog.text
    assert config == before
    assert OmegaConf.to_container(OmegaConf.create(saved), resolve=True) == before


def test_legacy_asset_layout_rejects_parent_traversal() -> None:
    config = _compose('synthetic_court')
    config.model.encoder.checkpoint_path = 'dinov3/checkpoints/../unexpected.pth'
    with pytest.raises(CourtModelIOError, match='exactly one checkpoint'):
        CourtInferenceSpec.from_checkpoint_config(config, serialize_target_bundle(_bundle()))


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
        pose_long_side=False,
        patch_size=16,
        architecture={"name": "fixture"},
        backbone_asset_migration=None,
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
        state["criterion.training_only"] = torch.ones(1)
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {"config": {}, "target_bundle_state": {}},
            "state_dict": state,
        },
        path,
    )
    before = file_sha256(path)
    if change in {"none", "foreign_prefix"}:
        loaded = load_court_checkpoint(path)
        assert loaded.identity["checkpoint_sha256"] == before
        assert float(model.weight[0, 0]) == 3
        assert float(model.bias[0]) == 2
    else:
        with pytest.raises((RuntimeError, CourtModelIOError)):
            load_court_checkpoint(path)
    assert file_sha256(path) == before


_CONFIG = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


@pytest.mark.parametrize('legacy_layout', [False, True])
def test_inference_loads_exact_weights_without_training_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_layout: bool,
) -> None:
    bundle = CourtTargetBundleSpec(
        {"kp": CourtTargetSpec("kp", "test", 1, ("point",), torch.float32, False)}
    )
    with initialize_config_dir(config_dir=str(_CONFIG), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"paths.external_asset_root={tmp_path}/assets", f"paths.checkpoint_root={tmp_path}/published"],
        )
    runtime = CourtTrainingConfig.from_config(cfg)
    resolver = runtime.shared.resolver
    assert runtime.model.encoder.checkpoint_path is not None
    runtime.model.encoder.checkpoint_path.parent.mkdir(parents=True)
    runtime.model.encoder.checkpoint_path.write_bytes(b"fixture encoder identity")
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    raw.pop("run")
    raw.pop("training")
    data = raw["data"]
    assert isinstance(data, dict)
    augmentation = data["augmentation"]
    assert isinstance(augmentation, dict)
    augmentation["train_scales"] = [256, 384, 512]
    if legacy_layout:
        raw['model']['encoder']['checkpoint_path'] = f'dinov3/checkpoints/{runtime.model.encoder.checkpoint_path.name}'
        # Old training roots remain immutable metadata. Explicit runtime roots win.
        raw['paths']['checkpoint_root'] = str(tmp_path / 'old-training-outputs')
    original_config = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)

    def make_model(
        config: CourtModelConfig, target_bundle: CourtTargetBundleSpec
    ) -> CourtHierarchicalModel:
        assert target_bundle == bundle
        assert config.encoder.checkpoint_path == runtime.model.encoder.checkpoint_path
        model = object.__new__(CourtHierarchicalModel)
        nn.Module.__init__(model)
        model.in_channels = 3
        model.target_bundle_spec = target_bundle
        encoder = object.__new__(CourtDINOv3Encoder)
        nn.Module.__init__(encoder)
        model.encoder = encoder
        model.register_parameter("weight", nn.Parameter(torch.zeros(2)))
        return model

    monkeypatch.setattr(CourtHierarchicalModel, "from_config", staticmethod(make_model))
    state_dict = {
        "model.weight": torch.tensor([1.0, 3.0]),
        "criterion.training_only": torch.ones(1),
    }
    checkpoint = {
        "hyper_parameters": {
            "config": raw,
            "target_bundle_state": serialize_target_bundle(bundle),
        },
        "state_dict": state_dict,
    }
    path = tmp_path / "court.ckpt"
    torch.save(checkpoint, path)
    original_hash = file_sha256(path)
    pair = load_court_pair(path, resolver=resolver)
    torch.testing.assert_close(pair.model.weight, torch.tensor([1.0, 3.0]))
    loaded = load_court_checkpoint(path, resolver=resolver)
    assert loaded.identity['backbone_sha256'] == file_sha256(runtime.model.encoder.checkpoint_path)
    assert bool(loaded.identity['backbone_asset_migration']) == legacy_layout
    assert raw == original_config and file_sha256(path) == original_hash
    with pytest.raises(ValueError, match="strict"):
        load_court_pair(path, resolver=resolver, strict=False)
    state_dict["model.unexpected"] = torch.ones(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_court_pair(path, resolver=resolver)
    del state_dict['model.unexpected']
    torch.save(checkpoint, path)
    old_path = resolver.roots.external_asset_root / 'dinov3/checkpoints' / runtime.model.encoder.checkpoint_path.name
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(runtime.model.encoder.checkpoint_path.read_bytes())
    runtime.model.encoder.checkpoint_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_court_pair(path, resolver=resolver)


def test_pose_inference_keeps_saved_resize_and_padding_contract() -> None:
    config = _compose(
        "synthetic_court",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
        "data.augmentation.preserve_fx_fy=true",
        "data.augmentation.patch_size=32",
    )
    spec = CourtInferenceSpec.from_checkpoint_config(
        config, serialize_target_bundle(_bundle())
    )
    assert spec.pose_long_side is True
    assert spec.patch_size == 32
    assert spec.short_side == config.data.augmentation.val_short_side
