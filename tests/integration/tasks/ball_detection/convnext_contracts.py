"""The legacy ConvNeXt heatmap selector through its actual Lightning loop."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.checkpoints.heatmap import load_ball_checkpoint
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.training.convnext_heatmap.module import (
    BallDetectionLightningModule,
)
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint
from tests.support.tasks.ball_detection.store import (
    ball,
    frame,
    store_config,
    write_store_clip,
)


def _compose(root: Path, gan: bool) -> DictConfig:
    assert not gan, "This profile exercises supervised ConvNeXt"
    cfg = store_config(root)
    cfg.data.sources = ["tracknet"]
    cfg.data.train_sampling = None
    cfg.data.image_size = [64, 64]
    cfg.model.dims = [4, 8, 16, 32]
    cfg.model.depth = 1
    cfg.training.learning_rate = 0.001
    cfg.training.weight_decay = 0.0
    cfg.training.warmup_steps = 0
    cfg.training.compile.enabled = False
    cfg.training.qualitative_logging.enabled = False
    cfg.training.steps_per_epoch = 1
    return cfg


def _data(config: DictConfig) -> Any:
    data = BallStoreDataModule(config)
    data.setup("fit")
    return data


def _build(config: DictConfig) -> tuple[Any, Any]:
    root = Path(config.paths.data_root) / "ball_detection/test-v1"
    for split in ("train", "val"):
        write_store_clip(
            root,
            f"tracknet/{split}/clip",
            [frame(0, ball()), frame(1, ball("unresolved", None))],
            split=split,
        )
    return BallDetectionLightningModule(config), _data(config)


def _parameters(module: Any) -> Any:
    active = names_under(module, "model")
    groups = {
        key: names_under(module, f"model.{key}")
        for key in (
            "downsampling_layers",
            "encoder_layers",
            "bottleneck",
            "upsampling_layers",
            "decoder_layers",
            "final_conv",
        )
    }
    return ParameterContract(active, frozenset(), frozenset()), (
        OptimizerContract("generator", active, groups),
    )


def _batch(config: DictConfig) -> Any:
    return next(iter(_data(config).train_dataloader()))


def _result(module: Any, batch: Any) -> Any:
    result = module._compute_supervised_result(batch, "val")
    return {**result, "outputs": result["pred_heatmaps"]}


def _assert_batch(batch: Any) -> None:
    assert batch["supervised"].dtype == torch.bool
    assert batch["supervised"].any() and not batch["supervised"].all()


def _assert_prediction(output: Any) -> None:
    assert output.ndim == 4 and output.shape[1] == 2
    assert output.dtype == torch.float32 and torch.isfinite(output).all()


def _predict(module: Any, batch: Any) -> Any:
    call = module.model_io.prepare_training_batch(
        batch, image_normalization=module.image_normalization
    )
    logits = module.model(*call.model_call.model_args)
    public = module.model_io.prediction(
        logits,
        call.model_call,
        subpixel_refine=bool(module.config.metrics.subpixel_refine),
    )
    return dict(
        logits=logits,
        public=public,
        supervised=call.supervised,
        original_size=call.original_size,
        normalization=module.image_normalization,
    )


def _predict_artifact(path: Path, batch: Any) -> Any:
    loaded = load_ball_checkpoint(path, weights_only=False)
    loaded.model_io.model.eval()
    return _predict(
        SimpleNamespace(
            model=loaded.model_io.model,
            model_io=loaded.model_io.adapter,
            config=loaded.config,
            image_normalization=loaded.image_normalization,
        ),
        batch,
    )


CONVNEXT = ModelFamily(
    name="conv_next_unet",
    compose=_compose,
    build=_build,
    rebuild=lambda config: (BallDetectionLightningModule(config), _data(config)),
    batch=_batch,
    supervised_result=_result,
    predict=_predict,
    predict_artifact=_predict_artifact,
    save=save_checkpoint,
    restore=lambda config, path: BallDetectionLightningModule.load_from_checkpoint(
        path, strict=True, map_location="cpu", weights_only=False
    ),
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
