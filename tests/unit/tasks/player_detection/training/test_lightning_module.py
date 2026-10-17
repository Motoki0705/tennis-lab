"""Exercise player validation hooks with Lightning and a tiny CPU detector."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.detection_dataset import (
    PLAYER_CLASS_ID,
    DetectionSample,
    collate_detection,
)
from src.tasks.player_detection.models.dino_detector import PlayerDinoModel
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/player_detection/configs"


class _Dino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.ones(()))

    def forward(self, images, targets):
        assert all(image.device.type == "cpu" for image in images)
        boxes = self.logit.new_tensor([0.5, 0.5, 0.5, 0.5])
        return {
            "pred_logits": self.logit.expand(len(images), 1, PLAYER_CLASS_ID + 1),
            "pred_boxes": boxes.expand(len(images), 1, 4),
        }


class _Criterion(torch.nn.Module):
    weight_dict = {"loss_ce": 1.0}

    def forward(self, outputs, targets):
        loss = outputs["pred_logits"].square().mean()
        return dict.fromkeys(
            ("loss_ce", "loss_bbox", "loss_giou", "loss_ce_dn", "loss_bbox_dn", "loss_giou_dn"),
            loss,
        )


@pytest.fixture
def module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlayerDetectionLightningModule:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name="train", overrides=[
            f"paths.project_root={tmp_path}",
            "training.qualitative_logging.enabled=false",
            "training.warmup_steps=0",
            "training.trainer.max_epochs=1",
            "evaluation.max_detections=1",
        ])
    runtime = PlayerTrainingConfig.from_config(config)
    # Replace only CUDA model construction. Training, decoding, metrics, and
    # Lightning's sanity/validation lifecycle use the production implementation.
    with monkeypatch.context() as setup:
        setup.setattr(torch.cuda, "is_available", lambda: True)
        setup.setattr(
            "src.tasks.player_detection.training.lightning_module.build_player_dino",
            lambda *args, **kwargs: (PlayerDinoModel(_Dino()), _Criterion()),
        )
        return PlayerDetectionLightningModule(config, runtime)


def _loader(positives: list[bool]) -> DataLoader:
    samples = [
        DetectionSample(
            image=torch.zeros(3, 8, 8),
            boxes_cxcywh=torch.tensor([[0.5, 0.5, 0.5, 0.5]]) if positive else torch.empty(0, 4),
            labels=torch.full((int(positive),), PLAYER_CLASS_ID, dtype=torch.int64),
            boxes_xyxy_px=torch.tensor([[2.0, 2.0, 6.0, 6.0]]) if positive else torch.empty(0, 4),
            original_size=(8, 8),
            frame=index,
        )
        for index, positive in enumerate(positives)
    ]
    return DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_detection)


def _trainer(tmp_path: Path, **kwargs) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tmp_path, **kwargs,
    )


class _SanityCheck(pl.Callback):
    def __init__(self) -> None:
        self.batches: list[int] = []
        self.metric_keys: set[str] = set()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            self.batches.append(batch_idx)

    def on_sanity_check_end(self, trainer, pl_module):
        self.metric_keys = set(trainer.callback_metrics)


@pytest.mark.parametrize("sanity_has_positive", [False, True])
def test_fit_discards_sanity_metrics_before_full_validation(
    module: PlayerDetectionLightningModule, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, sanity_has_positive: bool,
) -> None:
    sanity = _SanityCheck()
    trainer = _trainer(tmp_path, callbacks=[sanity])
    assert trainer.num_sanity_val_steps == 2  # Keep the production Lightning default.
    computed: list[tuple[bool, dict[str, float]]] = []
    compute = module.val_metrics.compute

    def record_compute():
        headline, diagnostics = compute()
        computed.append((trainer.sanity_checking, diagnostics))
        return headline, diagnostics

    monkeypatch.setattr(module.val_metrics, "compute", record_compute)
    trainer.fit(
        module, train_dataloaders=_loader([True]),
        val_dataloaders=_loader([sanity_has_positive, False, True]),
    )

    assert sanity.batches == [0, 1]
    assert not any(key.startswith("val/") for key in sanity.metric_keys)
    assert trainer.global_step == 1
    assert module.model.dino.logit.item() < 1.0
    assert len(computed) == 1
    during_sanity, diagnostics = computed[0]
    assert not during_sanity
    assert diagnostics["images"] == 3
    assert diagnostics["target_instances"] == 1 + int(sanity_has_positive)
    assert diagnostics["negative_images"] == 2 - int(sanity_has_positive)
    assert diagnostics["false_positives"] == 2 - int(sanity_has_positive)
    assert trainer.callback_metrics["val/recall"].item() == 1.0
    assert trainer.callback_metrics["val/precision"].item() == pytest.approx(
        (1 + int(sanity_has_positive)) / 3,
    )


@pytest.mark.parametrize("stage", ["fit", "validate", "test"])
def test_full_negative_only_evaluation_still_fails(
    module: PlayerDetectionLightningModule, tmp_path: Path, stage: str,
) -> None:
    trainer = _trainer(tmp_path)
    negatives = _loader([False, False, False])
    with pytest.raises(ValueError, match="negative-only evaluation"):
        if stage == "fit":
            trainer.fit(module, train_dataloaders=_loader([True]), val_dataloaders=negatives)
        elif stage == "validate":
            trainer.validate(module, dataloaders=negatives, verbose=False)
        else:
            trainer.test(module, dataloaders=negatives, verbose=False)
    # The all-negative sanity subset must not be what rejects a fit.
    assert trainer.global_step == (1 if stage == "fit" else 0)
