from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

import src.tasks.ball_detection.models.dinov3_rope as dinov3_module
from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_model
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.adapters import (
    BallModelIOAdapter,
    DINOv3BallExecutionBoundary,
)
from src.tasks.ball_detection.model_io.contracts import (
    BallModelInputSpec,
    BallModelIOError,
)
from src.tasks.ball_detection.model_io.evaluation import CheckpointBallHeatmapPredictor
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_detection.models.dinov3_rope import DINOv3RoPEBallDetector
from src.tasks.base.model_io import bind_model_io
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.models.components.ffn_layers import DeepSeekV4SwiGLU, FFNType
from src.utils.models.loading import DINOv3BackboneAdapter
from src.utils.models.lora import LoRAConfig
from src.utils.paths import PROJECT_ROOT


class _FakeDINOv3(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.grad_enabled: bool | None = None
        self.invalid_response = False

    def forward_features(self, inputs: torch.Tensor) -> object:
        self.grad_enabled = torch.is_grad_enabled()
        if self.invalid_response:
            return {}
        token_count = (inputs.shape[-2] // self.patch_size) * (
            inputs.shape[-1] // self.patch_size
        )
        return {
            "x_norm_patchtokens": inputs.new_zeros(
                inputs.shape[0],
                token_count,
                self.embed_dim,
            )
        }


def _model(
    fake: _FakeDINOv3,
    *,
    gradient_checkpointing: bool = False,
    ffn_type: FFNType = "swiglu",
) -> DINOv3RoPEBallDetector:
    return DINOv3RoPEBallDetector(
        backbone_repository_path=Path("/unused/dinov3"),
        backbone_checkpoint_path=Path("/unused/dinov3/checkpoint.pth"),
        in_channels=3,
        num_classes=1,
        num_frames=2,
        image_size=(8, 8),
        backbone_name="fake",
        backbone_strict=True,
        backbone_train_mode="frozen",
        backbone_last_n_blocks=0,
        backbone_lora=LoRAConfig(
            enabled=False,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_modules=("qkv",),
        ),
        decoder_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        decoder_head_dim=4,
        decoder_ffn_dim=16,
        decoder_rope_dim=4,
        decoder_rope_base=(10000.0, 10000.0, 10000.0),
        decoder_dropout=0.0,
        decoder_attention_type="mha",
        decoder_n_kv_heads=None,
        decoder_ffn_type=ffn_type,
        decoder_gradient_checkpointing=gradient_checkpointing,
        head_min_channels=2,
        backbone=DINOv3BackboneAdapter(fake),
    )


def _pair(
    model: DINOv3RoPEBallDetector,
) -> tuple[DINOv3RoPEBallDetector, BallModelIOAdapter]:
    adapter = BallModelIOAdapter(
        BallModelInputSpec(
            model_name="dinov3_rope",
            input_mode="rgb",
            input_layout="btchw",
            in_channels=3,
            num_classes=1,
            configured_frames=2,
            image_size_hw=(8, 8),
            minimum_spatial_size=None,
            mdd_gain=1.0,
            mdd_offset=0.0,
        ),
        expected_model_type=DINOv3RoPEBallDetector,
        minimum_frames=1,
        execution_boundary=DINOv3BallExecutionBoundary(frozen_backbone=True),
    )
    adapter.validate_model_pair(model)
    return model, adapter


def test_dinov3_boundary_prepares_tokens_and_valid_shape_forward() -> None:
    fake = _FakeDINOv3()
    model, adapter = _pair(_model(fake, ffn_type="deepseek_v4_swiglu"))
    pair = bind_model_io(model, adapter)

    probability = pair.run(torch.zeros(1, 2, 3, 8, 8))

    assert isinstance(model.decoder[0].ffn, DeepSeekV4SwiGLU)
    assert fake.grad_enabled is False
    assert probability.shape == (1, 2, 8, 8)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))


def test_invalid_dinov3_response_fails_before_detector_forward() -> None:
    fake = _FakeDINOv3()
    fake.invalid_response = True
    model, adapter = _pair(_model(fake))
    pair = bind_model_io(model, adapter)
    model_entries = 0

    def count_entry(
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        nonlocal model_entries
        _ = (module, args)
        model_entries += 1

    model.register_forward_pre_hook(count_entry)

    with pytest.raises(BallModelIOError, match="missing required x_norm_patchtokens"):
        pair.run(torch.zeros(1, 2, 3, 8, 8))

    assert model_entries == 0


def test_decoder_checkpoint_execution_is_selected_on_mode_change() -> None:
    model = _model(_FakeDINOv3(), gradient_checkpointing=True)

    assert model._decoder_block_executor.__name__ == "_checkpoint_decoder_block"
    model.eval()
    assert model._decoder_block_executor.__name__ == "_run_decoder_block"
    model.train()
    assert model._decoder_block_executor.__name__ == "_checkpoint_decoder_block"


def test_composed_backbone_loads_weights_from_checkpoint_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "weights"
    external_root = tmp_path / "vendor"
    with initialize_config_dir(
        version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks/ball_detection/configs"),
    ):
        config = compose(config_name="train", overrides=[
            "model=dinov3_rope", f"paths.checkpoint_root={checkpoint_root}",
            f"paths.external_asset_root={external_root}",
        ])
    loaded: dict[str, object] = {}

    def load_backbone(**kwargs: object) -> DINOv3BackboneAdapter:
        loaded.update(kwargs)
        return DINOv3BackboneAdapter(_FakeDINOv3())

    monkeypatch.setattr(dinov3_module, "load_dinov3_backbone", load_backbone)
    validate_model(config, paths=BallRuntimePaths.from_config(config))
    DINOv3RoPEBallDetector.from_config(config)
    assert loaded["repository_path"] == external_root / "dinov3"
    assert loaded["checkpoint_path"] == (
        checkpoint_root / "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )


@pytest.mark.parametrize("explicit_root", [False, True])
def test_legacy_backbone_metadata_migrates_explicitly_and_preserves_saved_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, explicit_root: bool,
) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks/ball_detection/configs")):
        config = compose(config_name="train", overrides=[
            "model=dinov3_rope", f"paths.project_root={tmp_path}", "paths.checkpoint_root=ckpt",
            "model.image_size=[8,8]", "data.image_size=[8,8]", "model.num_frames=2", "model.decoder.dim=8",
            "model.decoder.num_layers=1", "model.decoder.num_heads=2", "model.decoder.head_dim=4",
            "model.decoder.ffn_dim=16", "model.decoder.rope_dim=4", "model.heatmap_head.min_channels=2",
        ])
    filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    weights_root = tmp_path / ("custom-weights" if explicit_root else "ckpt")
    canonical = weights_root / "dinov3" / filename
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    loaded_paths: list[Path] = []

    def load_backbone(**kwargs: Any) -> DINOv3BackboneAdapter:
        path = kwargs["checkpoint_path"]
        assert isinstance(path, Path)
        loaded_paths.append(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return DINOv3BackboneAdapter(_FakeDINOv3())

    monkeypatch.setattr(dinov3_module, "load_dinov3_backbone", load_backbone)
    config.paths.checkpoint_root = str(weights_root)
    original = build_ball_detection_pair(config)
    config.paths.checkpoint_root = "outputs"
    config.model.backbone.checkpoint_path = f"dinov3/checkpoints/{filename}"
    saved = OmegaConf.to_container(config, resolve=True)
    checkpoint = tmp_path / "outputs/detector.ckpt"
    checkpoint.parent.mkdir()
    torch.save({"hyper_parameters": {"config": saved},
                "state_dict": {f"model.{key}": tensor for key, tensor in original.model.state_dict().items()}}, checkpoint)
    roots = RuntimePathRoots.from_mapping({**dict(config.paths), "checkpoint_root": str(weights_root)}, repository_root=tmp_path)
    resolver = PathResolver(roots) if explicit_root else None
    loaded = load_ball_checkpoint(checkpoint, resolver=resolver)
    assert loaded_paths[-1] == canonical
    assert OmegaConf.to_container(loaded.config, resolve=True) == saved
    assert loaded.backbone_asset_migration == {
        "layout": "external_dinov3_to_checkpoint", "saved_path": f"dinov3/checkpoints/{filename}",
        "runtime_path": f"dinov3/{filename}", "checkpoint_root": str(weights_root),
    }
    assert "Migrating saved DINOv3 asset path" in caplog.text
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(loaded.model_io.model.state_dict()[name], value, rtol=0, atol=0)
    runtime_resolver = PathResolver(roots)
    reference = BallRuntimePaths(runtime_resolver).checkpoint_input(
        {"checkpoint": {"role": "artifact", "path": "detector.ckpt"}}, "checkpoint", path="inference",
    )
    assert reference.path == checkpoint and reference.role is PathRole.ARTIFACT
    predictor = BallDetectionPredictor.load_from_checkpoint(
        reference.path, resolver=runtime_resolver, checkpoint_role=reference.role,
        device="cpu", subpixel_refine=True, strict=True, weights_only=False,
    )
    evaluator = CheckpointBallHeatmapPredictor.load(
        reference.path, resolver=runtime_resolver, device=torch.device("cpu"), strict=True, weights_only=False,
    )
    assert loaded_paths[-2:] == [canonical, canonical]
    for model in (predictor.model, evaluator.model):
        for name, value in original.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert torch.load(checkpoint, weights_only=False)["hyper_parameters"]["config"] == saved
    legacy = tmp_path / "third_party/dinov3/checkpoints" / filename
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    canonical.unlink()
    with pytest.raises(FileNotFoundError, match=str(canonical)):
        load_ball_checkpoint(checkpoint, resolver=resolver)
