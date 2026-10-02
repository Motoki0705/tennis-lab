"""Normalized store batches must render and checkpoint at validation epoch end."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image, ImageSequence

from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_mixed_profile_renders_normalized_validation_and_saves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    monkeypatch.delenv("TENNIS_LAB_COLAB_PROGRESS_PATH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    store = tmp_path / "data/ball_detection/fixture"
    for source in ("tracknet", "meiji", "chat_annotation"):
        for split in ("train", "val"):
            write_store_clip(
                store, f"{source}/{split}/clip", [frame(0, ball()), frame(1, ball())],
                source=source, split=split,
            )
    with initialize_config_dir(
        config_dir=str(_PROJECT_ROOT / "src/tasks/ball_detection/configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="train_meiji_mixed",
            overrides=[
                f"paths.project_root={_PROJECT_ROOT}",
                f"paths.data_root={tmp_path / 'data'}",
                f"paths.output_root={tmp_path / 'outputs'}",
                f"paths.artifact_root={tmp_path / 'artifacts'}",
                f"paths.checkpoint_root={tmp_path / 'ckpt'}",
                "run.output_dir=qualitative-smoke",
                "run.init_weights=null",
                "run.gpus=0",
                "data.data_dir=ball_detection/fixture",
                "data.image_size=[64,64]",
                "data.heatmap_size=[16,16]",
                "data.batch_size=3",
                "data.num_workers=0",
                "data.pin_memory=false",
                "data.train_sampling.windows_per_epoch=6",
                "model.num_frames=2",
                "data.eval_stride=2",
                "model.dims=[4,8,16,32]",
                "model.depth=1",
                "training.trainer.max_epochs=1",
                "training.trainer.precision=32-true",
                "training.trainer.log_every_n_steps=1",
                "training.trainer.enable_model_summary=false",
                "training.warmup_steps=0",
                "training.qualitative_logging.num_samples=1",
            ],
        )
    for name, augmentation in cfg.data.augmentation.items():
        augmentation.enabled = name == "normalize_imagenet"
    # Keep epoch-0 rendering enabled even though the production interval is 4.
    assert cfg.training.qualitative_logging.enabled
    assert cfg.model.input_mode == "mdd"
    BallDetectionTrainingRunner().run(cfg)

    log_dir = tmp_path / "outputs/qualitative-smoke/logs/version_0"
    with Image.open(log_dir / "qualitative/epoch_0000/ball_batch00.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 2
    checkpoints = log_dir / "checkpoints"
    assert len(list(checkpoints.glob("*.ckpt"))) == 2
    saved = torch.load(checkpoints / "last.ckpt", map_location="cpu", weights_only=False)
    assert saved["global_step"] == 2
    assert saved["epoch"] == 0
    state = next(
        value for key, value in saved["callbacks"].items()
        if key.startswith("ModelCheckpoint")
    )
    assert state["monitor"] == "val/meiji/candidate_recall_at_8_20px"
    assert torch.isfinite(state["best_model_score"])
    assert Path(state["best_model_path"]).is_file()
    assert not (tmp_path / "outputs/qualitative-smoke/predictions").exists()
