"""Opt-in CPU smoke with the published inputs and production checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.model_io.contracts import CourtPoseTrainingResult
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)


@pytest.mark.local_data
def test_published_two_source_batch_trains_all_five_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root_value = os.environ.get("TENNIS_COURT_TEST_ROOT")
    if root_value is None:
        pytest.skip(
            "Set TENNIS_COURT_TEST_ROOT to the repository holding data/ and ckpt/."
        )
    assert root_value is not None
    root = Path(root_value).resolve(strict=True)
    checkpoint_path = root / "ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config_dir = (
        Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                f"paths.project_root={root}",
                f"paths.output_root={tmp_path / 'runs'}",
                "run.gpus=0",
                "run.output_dir=court_detection/smoke/published",
                "training.compile.enabled=false",
                "training.qualitative_logging.enabled=false",
                "data.batch_size=2",
                "data.num_workers=0",
                "data.pin_memory=false",
                "mixed.train_batch_counts.synthetic_court=1",
                "mixed.train_batch_counts.tennis_court_detector=1",
                "data.augmentation.train_scales=[64]",
                "data.augmentation.val_short_side=64",
            ],
        )
    config.model = OmegaConf.create(checkpoint["hyper_parameters"]["config"]["model"])

    # Validate one actual image per available split, without scanning all JPEGs.
    def first_record(self: Any, split: str) -> Any:
        return self._records[split][:1]

    monkeypatch.setattr(SyntheticCourtInput, "records", first_record)
    monkeypatch.setattr(TennisCourtDetectorInput, "records", first_record)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        data = CourtDetectionDataModule(config)
        data.setup("fit")
        batch = next(iter(data.train_dataloader()))
        assert batch["pose_supervision_mask"].sum().item() == 1
        module = CourtDetectionLightningModule(
            config, target_bundle=data.target_bundle_spec
        )
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
        module.model.load_state_dict(state, strict=True)
        module.eval()
        monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
        result = module._shared_step(batch, "train")
        assert isinstance(result, CourtPoseTrainingResult)
        assert result.output.pose is not None
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert set(result.output.dense_logits) == {"kp", "seg", "line", "semantic_line"}
        assert result.output.pose.values.shape == (2, 10)
        for branch in [*module.model.heads.values(), module.model.pose_head]:
            gradients = [parameter.grad for parameter in branch.parameters()]
            assert all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
            assert any(gradient.count_nonzero() for gradient in gradients)
    finally:
        torch.set_num_threads(previous_threads)
