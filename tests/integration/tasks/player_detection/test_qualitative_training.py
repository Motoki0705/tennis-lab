"""Render original validation clips, including unstored frames, through Lightning."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image, ImageSequence
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
from src.tasks.player_detection.data.detection_dataset import (
    PLAYER_CLASS_ID,
    DetectionBatch,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.models.dino_detector import PlayerDinoModel
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)
from src.tasks.player_detection.visualization.qualitative import PlayerClipRenderer
from src.tennis_scene.chat_annotation.layout import published_video_path
from src.tennis_scene.chat_annotation.manifests import load_prepared_manifests
from tests.support.tasks.player_detection.chat_root import (
    FRAMES,
    make_synthetic_root,
    player,
    publish_player_annotation,
    write_clip,
)

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).resolve().parents[4]


class _Dino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.frame_count = 0
        self.batch_sizes: list[int] = []

    def forward(self, images: list[torch.Tensor], targets: Any) -> dict[str, torch.Tensor]:
        assert not self.training
        assert targets is None
        assert all(image.device.type == "cpu" for image in images)
        self.frame_count += len(images)
        self.batch_sizes.append(len(images))
        logits = torch.full((len(images), 2, 91), -20.0)
        logits[:, 0, PLAYER_CLASS_ID] = 20.0
        boxes = torch.tensor([[0.5, 0.7, 0.5, 0.4], [0.9, 0.1, 0.1, 0.1]])
        return {"pred_logits": logits, "pred_boxes": boxes[None].expand(len(images), -1, -1)}


@pytest.fixture
def clip_runtime(tmp_path: Path) -> tuple[DictConfig, PlayerTrainingConfig, PlayerDetectionDataModule]:
    source = make_synthetic_root(tmp_path)
    # Three non-empty sources ensure a genuine source-isolated validation split.
    for name in ("src_c", "src_d"):
        manifest = write_clip(source.root, name, "f000-006")
        publish_player_annotation(
            source.root, manifest,
            {index: [player("p1", [10.0, 5.0, 30.0, 50.0])] for index in (1, 2, 4)},
        )
    build_dataset(replace(source.config, val_ratio=0.25, test_ratio=0.0))
    with initialize_config_dir(
        version_base="1.3", config_dir=str(_ROOT / "src/tasks/player_detection/configs")
    ):
        config = compose(config_name="train", overrides=[
            f"paths.project_root={tmp_path}",
            "data.dataset_dir=player_detection/test-v1",
            "data.num_workers=0", "data.pin_memory=false", "data.eval_batch_size=1",
            "data.val_frame_stride=1", "data.input_size.short_side=64",
            "data.input_size.max_long_side=96", "data.augmentation.short_side_choices=[64]",
            "evaluation.max_detections=2", "qualitative.frame_stride=1",
            "qualitative.max_frames=6", "qualitative.batch_size=2",
        ])
    runtime = PlayerTrainingConfig.from_config(config)
    data = PlayerDetectionDataModule(runtime.data)
    data.setup("fit")
    return config, runtime, data


def test_validation_writes_bbox_gif_using_unstored_source_frames(
    clip_runtime: tuple[DictConfig, PlayerTrainingConfig, PlayerDetectionDataModule],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, runtime, data = clip_runtime
    dino = _Dino()
    criterion = torch.nn.Module()
    criterion.weight_dict = {}
    # Only model construction is substituted; preprocessing, decoding, datamodule,
    # validation hook, callback, source-video reader and GIF writer are production code.
    with monkeypatch.context() as setup:
        setup.setattr(torch.cuda, "is_available", lambda: True)
        setup.setattr(
            "src.tasks.player_detection.training.lightning_module.build_player_dino",
            lambda *args, **kwargs: (PlayerDinoModel(dino), criterion),
        )
        module = PlayerDetectionLightningModule(config, runtime)
    callback = QualitativeLoggingCallback(
        enabled=True, every_n_epochs=1, num_samples=2,
        selection_mode="fixed_indices", selected_indices=[0, 1],
    )
    logger = TensorBoardLogger(str(tmp_path / "logs"), name="", version="validation")
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=logger, callbacks=[callback],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.validate(module, datamodule=data, verbose=False)
    batch = next(iter(data.val_dataloader()))
    assert isinstance(batch, DetectionBatch)
    clip = data.store.clip_of(batch.frames[0])
    assert data.store.frames["frame_index"][data.store.clip_frames(clip)].tolist() == [1, 2, 4]
    files = list((Path(logger.log_dir) / "qualitative/epoch_0000").glob("*.gif"))
    assert len(files) == 1  # The same clip selected by two batches is not duplicated.
    assert files[0].name == f"player_clip{clip.index:04d}.gif"
    with Image.open(files[0]) as gif:
        frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(gif)]
    assert len(frames) == FRAMES  # Includes unannotated source frames 0, 3, 5.
    assert all((frame[40, 24] == [255, 0, 0]).all() for frame in frames)  # Original-pixel bbox.
    assert dino.frame_count == len(data.val_dataloader().dataset) + FRAMES
    assert list(Path(logger.log_dir).glob("events.out.tfevents.*"))


def test_clip_sampling_is_bounded_and_restores_model_mode(
    clip_runtime: tuple[DictConfig, PlayerTrainingConfig, PlayerDetectionDataModule], tmp_path: Path,
) -> None:
    config, _, data = clip_runtime
    config.qualitative.max_frames = 2
    config.qualitative.frame_stride = 2
    config.qualitative.batch_size = 1
    renderer = PlayerClipRenderer(PlayerTrainingConfig.from_config(config))
    dino = _Dino()
    model = PlayerDinoModel(dino)
    renderer.render(model, [next(iter(data.val_dataloader()))], tmp_path / "gifs", None, 0)
    assert model.training
    assert dino.batch_sizes == [1, 1]
    with Image.open(next((tmp_path / "gifs").glob("*.gif"))) as gif:
        assert gif.n_frames == 2


@pytest.mark.parametrize("failure", ["missing_video", "wrong_hash"])
def test_source_video_errors_do_not_fall_back_to_sparse_store_images(
    clip_runtime: tuple[DictConfig, PlayerTrainingConfig, PlayerDetectionDataModule],
    tmp_path: Path, failure: str,
) -> None:
    _, runtime, data = clip_runtime
    assert runtime.qualitative is not None
    renderer = PlayerClipRenderer(runtime)
    batch = next(iter(data.val_dataloader()))
    clip = renderer.store.clip_of(batch.frames[0])
    if failure == "missing_video":
        manifests = load_prepared_manifests(runtime.qualitative.annotation_root)
        published_video_path(runtime.qualitative.annotation_root, manifests[clip.clip_id]).unlink()
    else:
        renderer.store.clips = tuple(
            replace(item, video_sha256="0" * 64) if item.index == clip.index else item
            for item in renderer.store.clips
        )
    model = PlayerDinoModel(_Dino())
    with pytest.raises((ValueError, FileNotFoundError)):
        renderer.render(model, [batch], tmp_path / "gifs", None, 0)
    assert model.training
    assert not (tmp_path / "gifs").exists()
