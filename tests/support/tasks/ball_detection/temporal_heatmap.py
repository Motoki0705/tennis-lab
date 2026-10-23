"""Small complete-pack stores and explicit recipes for CPU/GPU diagnostics."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig, MDDHeatmapHybridConfig
from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
    UpdateBudget,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def prepare_fixture(directory: Path) -> Path:
    root = directory / "store"
    for split in ("train", "val", "test"):
        frames = [frame(i, ball(xy=(10., 12.))) if i < 40 or i >= 96 else
                  frame(i) if i < 72 else frame(i, ball("out_of_frame", None)) for i in range(144)]
        write_store_clip(root, f"tracknet/{split}/dense", frames, split=split, size=(24, 24))
    # Three of the T=32 children have no teachers but must still be processed.
    sparse = [frame(i, ball(xy=(10., 12.))) if i == 5 else frame(i, annotated=False) for i in range(128)]
    write_store_clip(root, "tracknet/train/sparse", sparse, size=(24, 24))
    prepared = directory / "prepared"
    prepare_all_frame_training(root, prepared, configs=[AllFrameWindowConfig(t) for t in (32, 64, 128)])
    return prepared


def small_model(architecture: str, *, frames: int = 32, dropout: float = .2) -> MDDHeatmapConfig:
    cls = MDDHeatmapHybridConfig if architecture == "hybrid" else MDDHeatmapConfig
    return cls(stem_channels=(4, 4, 8, 8), mixed_channels=(8, 16), residual_blocks=(0,) * 6,
               decoder_channels=8, dim=16, heads=2, layers=1, ffn_dim=64, dropout=dropout,
               rope_base=10000., activation_checkpointing=False, encoder_variant="convnext_v2",
               temporal_mixing="factorized", input_channels=5, frames=frames, frame_chunk_size=None)


def update_budget(**overrides: Any) -> UpdateBudget:
    values: dict[str, Any] = dict(updates=3, input_frames=128, microbatch_windows=1, learning_rate=.001, warmup_updates=1,
                  minimum_lr_ratio=.1, weight_decay=.01, gradient_clip=1., evaluation_interval=3,
                  checkpoint_interval=1, seed=42, sigma_ratio=.012, gamma=2.)
    values.update(overrides)
    return UpdateBudget(**values)


def model_yaml(architecture: str, frames: int = 32) -> dict[str, Any]:
    return dict(name="mdd_heatmap_hybrid" if architecture == "hybrid" else "mdd_dpt_heatmap",
                **asdict(small_model(architecture, frames=frames)))
