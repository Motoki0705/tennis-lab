"""Exercise the shared disk writers with each task's composed config on CPU.

Only task data/model hooks are replaced by a two-sample linear fixture. The
runner, Trainer, callbacks, logger and artifact writers are production code.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageSequence
from torch.utils.data import DataLoader, TensorDataset

import src.utils.hydra  # noqa: F401 -- public config resolvers
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.paths import PROJECT_ROOT


class _FixtureData(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.test_dataset = TensorDataset(torch.ones(2, 1))

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.test_dataset, batch_size=2)

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.test_dataset, batch_size=2)

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.test_dataset, batch_size=2)


class _FixtureModel(BaseLightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.model = torch.nn.Linear(1, 1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return cast(torch.Tensor, self.model(batch[0]).square().mean())

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        self.log("val/loss", self.model(batch[0]).square().mean())

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        self.collect_test_predictions(batch, {})

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {"fixture": batch[0].numpy()}

    def on_test_epoch_end(self) -> None:
        self.save_test_predictions(
            metrics={"fixture_count": 2}, diagnostic_metrics={"fixture_batches": 1}
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any,
        global_step: int,
        epoch: int,
    ) -> None:
        save_qualitative_clip(
            frames_rgb=[
                np.zeros((8, 8, 3), np.uint8),
                np.full((8, 8, 3), 255, np.uint8),
            ],
            artifact_dir=artifact_dir,
            name="fixture",
            tb_writer=tb_writer,
            tag="fixture",
            global_step=global_step,
        )


class _FixtureRunner(BaseTrainingRunner):
    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return _FixtureData()

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return _FixtureModel(config)


@pytest.mark.parametrize(
    "task", ["ball_detection", "court_detection", "blcs", "plcs", "slcs"]
)
def test_named_run_writes_real_artifacts_on_cpu(
    task: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    monkeypatch.delenv("TENNIS_LAB_COLAB_PROGRESS_PATH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks" / task / "configs"),
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                f"paths.output_root={tmp_path / 'runs'}",
                f"paths.artifact_root={tmp_path / 'media'}",
                f"paths.checkpoint_root={tmp_path / 'weights'}",
                "run.gpus=0",
                "run.test_after_fit=true",
                "run.fast_dev_run=false",
                "training.compile.enabled=false",
                "training.trainer.max_epochs=1",
                "training.warmup_steps=0",
                "training.warmup_epochs=null",
                "training.trainer.precision=32-true",
                "training.trainer.check_val_every_n_epoch=1",
                "training.trainer.log_every_n_steps=1",
                "training.trainer.enable_progress_bar=false",
                "training.trainer.enable_model_summary=false",
                "training.early_stopping.enabled=false",
                "training.checkpoint.monitor=val/loss",
                "training.checkpoint.save_last=true",
                "training.qualitative_logging.enabled=true",
                "training.qualitative_logging.every_n_epochs=1",
                "training.qualitative_logging.num_samples=1",
            ],
        )
    runner = _FixtureRunner()
    runner.run(cfg)
    run = tmp_path / "runs" / cfg.run.output_dir
    saved = OmegaConf.load(run / "config.yaml")
    assert saved.run.output_dir == cfg.run.output_dir
    assert Path(saved.paths.output_root) == tmp_path / "runs"
    assert "${" not in (run / "config.yaml").read_text()
    log = run / "logs/version_0"
    checkpoint = log / "checkpoints/last.ckpt"
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert loaded["global_step"] == 1
    assert loaded["state_dict"]["model.weight"].device.type == "cpu"
    assert len(list((log / "checkpoints").glob("*.ckpt"))) == 2
    assert (log / "repro/output_dir.txt").read_text().strip() == str(checkpoint.parent)
    assert list(log.glob("events.out.tfevents.*"))
    with Image.open(log / "qualitative/epoch_0000/fixture.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 2
    with np.load(run / "predictions/pred_test.npz", allow_pickle=False) as prediction:
        np.testing.assert_array_equal(prediction["fixture"], np.ones((2, 1)))
        assert prediction["scene_ids"].tolist() == ["sample_000000", "sample_000001"]
    assert json.loads((run / "predictions/metrics.json").read_text()) == {
        "fixture_count": 2
    }
    assert json.loads((run / "predictions/diagnostic_metrics.json").read_text()) == {
        "fixture_batches": 1
    }
    assert not (tmp_path / "media").exists()
    assert not (tmp_path / "weights").exists()
    before = (run / "config.yaml").read_bytes()
    with pytest.raises(FileExistsError, match="Training output already exists"):
        runner.run(cfg)
    assert (run / "config.yaml").read_bytes() == before
