"""Small, fully labelled CPU fixtures for the three-stage recipe."""

from dataclasses import asdict
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from tests.integration.tasks.ball_detection.test_rgb_mdd_contract import config


def hybrid_recipe(manifest: Path, checkpoint: Path, output: Path) -> DictConfig:
    root = Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="train_mdd_hybrid")
    cfg.model = dict(name="mdd_hybrid", **asdict(config()))
    cfg.manifest, cfg.output = str(manifest), str(output)
    cfg.initialization = dict(mode="query", checkpoint=str(checkpoint))
    cfg.presence_threshold = .5
    cfg.loss.heatmap_weight, cfg.loss.presence_weight = 1., 1.
    cfg.freeze = dict(encoder_epochs=1, query_epochs=1)
    cfg.absence = dict(negative_kinds=["reviewed_empty", "out_of_frame"],
                      window_weights=dict(play=.5, absent=.25, nonplay=.25))
    cfg.epochs, cfg.windows_per_epoch, cfg.warmup_updates = 2, 4, 1
    cfg.learning_rate, cfg.seed = .001, 42
    cfg.device = "cpu"
    cfg.runtime.precision, cfg.runtime.jpeg_decoder = "fp32", "opencv"
    cfg.runtime.compile_mode, cfg.runtime.input_verification = "off", "lazy"
    cfg.runtime.num_workers = 0
    return cfg
