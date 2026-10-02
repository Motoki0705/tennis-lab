"""CPU qualification of ft-e13's real mixed-store validation visualization."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image, ImageSequence
from torch.utils.data import default_collate

from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ASSET_ROOT = Path("/home/kamimura/projects/tennis-lab")
_CHECKPOINT = _ASSET_ROOT / "ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt"
_STORE = _ASSET_ROOT / "data/ball_detection/ball-mix-v2"

pytestmark = [
    pytest.mark.local_data,
    pytest.mark.skipif(
        not _CHECKPOINT.is_file() or not _STORE.is_dir(),
        reason="ft-e13 or the three-source store is unavailable",
    ),
]


def test_ft_e13_renders_all_three_validation_sources_on_cpu(tmp_path: Path) -> None:
    with initialize_config_dir(
        config_dir=str(_PROJECT_ROOT / "src/tasks/ball_detection/configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_meiji_mixed",
            overrides=[
                f"paths.project_root={_PROJECT_ROOT}",
                f"paths.data_root={_ASSET_ROOT / 'data'}",
                f"paths.checkpoint_root={_ASSET_ROOT / 'ckpt'}",
            ],
        )
    runner = BallDetectionTrainingRunner()
    module = BallDetectionLightningModule(config).eval()
    runner.maybe_load_init_weights(runner.validate_runtime_config(config), module)
    assert module.device.type == "cpu"
    data = BallStoreDataModule(config)
    data.setup("validate")
    assert data.val_dataset is not None
    samples = []
    for source in config.data.sources:
        index = next(
            i for i in range(len(data.val_dataset))
            if data.val_dataset.source_of(i) == source
        )
        batch = default_collate([data.val_dataset[index]])
        images = batch["images"]
        assert images.min() < 0 or images.max() > 1
        with torch.no_grad():
            features = module.model_io.mdd_features(
                images, image_normalization=module.image_normalization, preprocessed=True,
            )
            call = module.model_io.prepare_model_call(
                images, image_normalization=module.image_normalization, preprocessed=True,
            )
            torch.testing.assert_close(features, call.model_input, rtol=0, atol=0)
            module.render_qualitative_samples(
                batches=[batch], outputs=[], artifact_dir=tmp_path / source,
                tb_writer=None, global_step=0, epoch=0,
            )
        artifact = tmp_path / source / "ball_batch00.gif"
        with Image.open(artifact) as gif:
            count = len(list(ImageSequence.Iterator(gif)))
        assert count == config.model.num_frames == 8
        samples.append({
            "source": source, "window_id": batch["window_id"][0],
            "input_min": float(images.min()), "input_max": float(images.max()),
            "gif_frames": count, "mdd_matches_model_input": True,
            "artifact": str(artifact.relative_to(tmp_path)),
        })
    report = {
        "device": "cpu", "samples": samples,
        "checkpoint_sha256": hashlib.sha256(_CHECKPOINT.read_bytes()).hexdigest(),
        "store_sha256": {
            name: hashlib.sha256((_STORE / name).read_bytes()).hexdigest()
            for name in ("metadata.json", "index.npz")
        },
    }
    (tmp_path / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
