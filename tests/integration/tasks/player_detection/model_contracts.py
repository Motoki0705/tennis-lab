"""Player CPU registration with an explicit external DINO/criterion boundary.

Swin, deformable attention, CUDA denoising, matching and upstream criterion are
faked here. The real player factory, freeze rules, Dataset/collate, Lightning
loss aggregation, parameter groups and Trainer updates remain in use. This
registration does not qualify the upstream DINO graph or its CUDA kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss

from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
from src.tasks.player_detection.data.detection_dataset import collate_detection
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.models.dino_detector import decode_player_detections
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint
from tests.support.tasks.player_detection.chat_root import make_synthetic_root

CONFIG_DIR = Path("src/tasks/player_detection/configs")
SELECTORS = {"dino_4scale_swin_l": "dino_4scale_swin_l"}


class _SwinBoundary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 3, padding=1)
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(images).mean(dim=(-1, -2))
        for layer in self.layers:
            value = value + layer(value).tanh()
        return value


class _DinoBoundary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(_SwinBoundary(), nn.Identity())
        self.query_embed = nn.Embedding(3, 8)
        self.class_embed = nn.Linear(8, 91)
        self.bbox_embed = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, images: Any, targets: Any) -> Any:
        features = (
            self.backbone(torch.stack(images))[:, None] + self.query_embed.weight[None]
        )
        return {
            "pred_logits": self.class_embed(features),
            "pred_boxes": self.bbox_embed(features).sigmoid(),
        }


def _xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1
    )


class _CriterionBoundary(nn.Module):
    """Differentiable external loss stand-in; query-to-target matching is fixed."""

    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}

    def forward(self, outputs: Any, targets: Any) -> Any:
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        classification = torch.zeros_like(logits)
        predicted, expected = [], []
        for index, target in enumerate(targets):
            count = len(target["labels"])
            classification[index, torch.arange(count), target["labels"]] = 1.0
            predicted.append(boxes[index, :count])
            expected.append(target["boxes"])
        prediction, truth = torch.cat(predicted), torch.cat(expected)
        losses = {
            "loss_ce": sigmoid_focal_loss(logits, classification, reduction="mean"),
            "loss_bbox": F.l1_loss(prediction, truth),
            "loss_giou": (
                1 - generalized_box_iou(_xyxy(prediction), _xyxy(truth)).diag()
            ).mean(),
        }
        losses.update({f"{key}_dn": logits.sum() * 0 for key in self.weight_dict})
        return losses


def _compose(root: Path, gan: bool) -> DictConfig:
    assert not gan, "Player has no GAN selector"
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        return compose(
            config_name="train",
            overrides=[
                f"paths.project_root={root}",
                f"paths.data_root={root / 'data'}",
                "data.dataset_dir=player_detection/test-v1",
                "data.batch_size=2",
                "data.eval_batch_size=2",
                "data.num_workers=0",
                "data.train_samples_per_epoch=2",
                "data.input_size.short_side=32",
                "data.input_size.max_long_side=64",
                "data.augmentation.short_side_choices=[32]",
                "data.augmentation.hflip_prob=0.0",
                "data.augmentation.brightness=0.0",
                "data.augmentation.contrast=0.0",
                "data.augmentation.saturation=0.0",
                "training.qualitative_logging.enabled=false",
                "training.compile.enabled=false",
                "training.learning_rate=0.001",
                "training.weight_decay=0.0",
                "training.warmup_steps=0",
                "training.steps_per_epoch=1",
                "training.trainer.max_epochs=2",
                "evaluation.max_detections=3",
            ],
        )


def _new_module(config: DictConfig) -> Any:
    # Only the constructor's external CUDA availability/model/weights boundary is
    # replaced. CUDA availability is restored before a CPU Trainer is created.
    external = _DinoBoundary()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "src.tasks.player_detection.models.dino_detector.build_dino",
            return_value=(external, _CriterionBoundary()),
        ),
        patch(
            "src.tasks.player_detection.models.dino_detector.load_dino_state_dict",
            return_value=external.state_dict(),
        ),
    ):
        return PlayerDetectionLightningModule(
            config, PlayerTrainingConfig.from_config(config)
        )


def _loader(config: DictConfig) -> Any:
    data = PlayerDetectionDataModule(PlayerTrainingConfig.from_config(config).data)
    # A confirmed negative followed by a two-player frame, drawn through the
    # real Dataset. A fixed subset avoids randomly drawing two negative frames.
    return DataLoader(
        Subset(data._dataset("train"), [0, 1]),
        batch_size=2,
        collate_fn=collate_detection,
    )


def _build(config: DictConfig) -> tuple[Any, Any]:
    fixture = make_synthetic_root(Path(config.paths.project_root))
    build_dataset(fixture.config)
    return _components(config)


class _ContractPlayerData(PlayerDetectionDataModule):
    def setup(self, stage: str | None = None) -> None:
        # This fixture has a training split only; no validation is requested.
        assert stage == "fit"
        self._dataset("train")


def _components(config: DictConfig) -> tuple[Any, Any]:
    data: Any = _ContractPlayerData(PlayerTrainingConfig.from_config(config).data)
    # The real sampler permutes a negative/positive pair on each epoch.
    # Keep the criterion boundary's fixed query matching meaningful at every step.
    data._datasets["train"] = Subset(data._dataset("train"), [0, 1])
    return _new_module(config), data


def _result(module: Any, batch: Any) -> Any:
    outputs = []
    handle = module.model.register_forward_hook(
        lambda _module, _args, output: outputs.append(output)
    )
    try:
        loss = module.training_step(batch, 0)
    finally:
        handle.remove()
    assert len(outputs) == 1
    return {"loss": loss, "outputs": outputs[0]}


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    frozen = names_under(
        module,
        "model.dino.backbone.0.patch_embed",
        "model.dino.backbone.0.layers.0",
        "model.dino.backbone.0.layers.1",
    )
    active = frozenset(dict(module.named_parameters())) - frozen
    groups = {
        key: names_under(module, f"model.dino.{key}")
        for key in ("backbone.0.layers.2", "query_embed", "class_embed", "bbox_embed")
    }
    return ParameterContract(active, frozen, frozenset()), (
        OptimizerContract("generator", active, groups),
    )


def _assert_batch(batch: Any) -> None:
    assert [len(labels) for labels in batch.labels] == [0, 2]
    assert batch.boxes_cxcywh[0].shape == (0, 4)
    assert all(
        image.dtype == torch.float32 and image.device.type == "cpu"
        for image in batch.images
    )


def _assert_prediction(output: Any) -> None:
    assert output["pred_logits"].shape == (2, 3, 91)
    assert output["pred_boxes"].shape == (2, 3, 4)
    assert all(
        value.dtype == torch.float32 and torch.isfinite(value).all()
        for value in output.values()
    )


def _restore(config: DictConfig, path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = _new_module(payload["hyper_parameters"]["config"])
    module.load_state_dict(
        payload["state_dict"],
        strict=True,
    )
    return module


def _predict(module: Any, batch: Any) -> Any:
    raw = module.model(batch.images, None)
    decoded = decode_player_detections(
        raw,
        batch.original_sizes,
        max_detections=module.runtime.evaluation.max_detections,
    )
    return dict(raw=raw, decoded=decoded, evaluation=module.runtime.evaluation)


DINO = ModelFamily(
    name="player/dino_external_cuda_boundary",
    compose=_compose,
    build=_build,
    rebuild=_components,
    batch=lambda config: next(iter(_loader(config))),
    supervised_result=_result,
    predict=_predict,
    save=save_checkpoint,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
