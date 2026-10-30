"""Court: fake pretrained DINOv3 only; real trainability, DPT and five losses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import nn

from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.models.loading import DINOv3BackboneAdapter
from tests.integration.tasks.court_detection.test_composable_pipeline import (
    _prepare_court_roots,
)
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint

CONFIG_DIR = Path("src/tasks/court_detection/configs")
SELECTORS = {"dinov3_dpt": "court_hierarchical"}


class _ExternalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.proj(self.qkv(x).tanh())
        return x + self.fc2(self.fc1(x).tanh())


class _PretrainedBoundary(nn.Module):
    """Tiny external feature provider; this is not an implementation of DINOv3."""

    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 4, 4)
        self.blocks = nn.ModuleList(_ExternalBlock() for _ in range(4))

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        *,
        n: tuple[int, ...],
        reshape: bool,
        return_class_token: bool,
        norm: bool,
    ) -> Any:
        assert not reshape and not return_class_token and norm
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in n:
                outputs.append(x)
        return tuple(outputs)


def _compose(root: Path, gan: bool, *, mode: str = "frozen") -> DictConfig:
    assert not gan, "Court has no GAN selector"
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        return compose(
            config_name="train",
            overrides=[
                f"paths.project_root={root}",
                "paths.data_root=data",
                "paths.checkpoint_root=checkpoints",
                "paths.artifact_root=artifacts",
                "paths.output_root=outputs",
                "paths.cache_root=cache",
                "paths.external_asset_root=external",
                "run.output_dir=court/contract",
                "run.gpus=0",
                "data.source.scene_ids=[V3]",
                "data.batch_size=2",
                "data.num_workers=0",
                "data.pin_memory=false",
                "data.augmentation.train_scales=[32]",
                "data.augmentation.val_short_side=32",
                "mixed.sources.tennis_court_detector.excluded_sample_ids=[]",
                "mixed.train_batch_counts.synthetic_court=1",
                "mixed.train_batch_counts.tennis_court_detector=1",
                "model.encoder.out_indices=[0,1,2,3]",
                f"model.encoder.train_mode={'full' if mode == 'full' else 'frozen'}",
                f"model.encoder.lora.enabled={str(mode == 'lora').lower()}",
                "model.encoder.lora.rank=2",
                "model.transformer_encoder.dim=16",
                "model.transformer_encoder.depth=1",
                "model.transformer_encoder.num_heads=2",
                "model.transformer_encoder.head_dim=8",
                "model.transformer_encoder.rope_dim=4",
                "model.transformer_encoder.ffn_dim=32",
                "model.decoder.size=tiny",
                "model.decoder.channels=64",
                "model.dense_head.normalization_groups=2",
                *[
                    f"model.dense_head.{kind}.{key}={value}"
                    for kind in ("kp", "seg", "line", "semantic_line")
                    for key, value in (("hidden_channels", 8), ("depth", 1))
                ],
                "training.compile.enabled=false",
                "training.learning_rate=0.001",
                "training.weight_decay=0.0",
                "training.warmup_steps=0",
                "training.warmup_epochs=null",
                "training.steps_per_epoch=1",
                "training.trainer.max_epochs=2",
                "training.qualitative_logging.enabled=false",
            ],
        )


def _new_module(config: DictConfig, **kwargs: Any) -> Any:
    with patch(
        "src.tasks.court_detection.models.dinov3_dpt.load_dinov3_backbone",
        side_effect=lambda **_kwargs: DINOv3BackboneAdapter(_PretrainedBoundary()),
    ):
        return CourtDetectionLightningModule(config, **kwargs)


def _build(config: DictConfig) -> tuple[Any, Any]:
    _prepare_court_roots(Path(config.paths.project_root), calibrated_pose=True)
    data = CourtDetectionDataModule(config)
    data.setup("fit")
    return _new_module(config, target_bundle=data.target_bundle_spec), data


def _batch(config: DictConfig) -> Any:
    data = CourtDetectionDataModule(config)
    data.setup("fit")
    return next(iter(data.train_dataloader()))


def _result(module: Any, batch: Any) -> Any:
    result = module._shared_step(batch, "val")
    return {"loss": result.loss, "outputs": result.output}


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    all_names = frozenset(dict(module.named_parameters()))
    unused = names_under(module, "model.decoder.fusion_blocks.3.skip_block")
    backbone = names_under(module, "model.encoder.backbone")
    encoder = module.config.model.encoder
    if encoder.lora.enabled:
        adaptation = frozenset(
            name for name in backbone if name.endswith((".lora_a", ".lora_b"))
        )
        assert adaptation
        frozen = backbone - adaptation
    else:
        frozen = backbone if encoder.train_mode == "frozen" else frozenset()
    active = all_names - frozen - unused
    groups = {
        key: names_under(module, f"model.{key}") - unused
        for key in (
            "feature_projections",
            "transformer_encoder",
            "decoder",
            "pose_head",
        )
    }
    groups.update(
        {
            kind: names_under(module, f"model.heads.{kind}")
            for kind in ("kp", "seg", "line", "semantic_line")
        }
    )
    if backbone - frozen:
        groups["backbone adaptation"] = backbone - frozen
    return ParameterContract(active, frozen, unused, optimizer_frozen=frozen), (
        OptimizerContract("generator", all_names, groups),
    )


def _assert_batch(batch: Any) -> None:
    assert batch["image"].dtype == torch.float32
    assert batch["image"].shape[0] == 2
    assert batch["pose_supervision_mask"].tolist() == [True, False]


def _assert_prediction(output: Any) -> None:
    assert {key: value.shape[1] for key, value in output.dense_logits.items()} == {
        "kp": 14,
        "seg": 7,
        "line": 1,
        "semantic_line": 12,
    }
    for tensor in output.dense_logits.values():
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    assert output.pose is not None
    assert (
        output.pose.values.shape == (2, 10)
        and output.pose.values.dtype == torch.float32
    )


def _predict(module: Any, batch: Any) -> Any:
    prepared = module.model_io.prepare_training_batch(batch)
    return module.model(*prepared.model_call.model_args)


def _restore(config: DictConfig, path: Path) -> Any:
    with patch(
        "src.tasks.court_detection.models.dinov3_dpt.load_dinov3_backbone",
        side_effect=lambda **_kwargs: DINOv3BackboneAdapter(_PretrainedBoundary()),
    ):
        return CourtDetectionLightningModule.load_from_checkpoint(
            path, config=config, map_location="cpu", weights_only=False
        )


FROZEN = ModelFamily(
    name="court_hierarchical/fake_pretrained",
    compose=_compose,
    build=_build,
    batch=_batch,
    supervised_result=_result,
    predict=_predict,
    save=save_checkpoint,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
LORA = replace(
    FROZEN,
    name="court_hierarchical/fake_pretrained_lora",
    compose=lambda root, gan: _compose(root, gan, mode="lora"),
)
FULL = replace(
    FROZEN,
    name="court_hierarchical/fake_pretrained_full",
    compose=lambda root, gan: _compose(root, gan, mode="full"),
)
