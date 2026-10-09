import json
from pathlib import Path

import torch

from src.tasks.ball_detection.data.coordinate_dataset import collate_coordinate_windows
from src.tasks.ball_detection.data.coordinate_manifest import build_coordinate_manifest
from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.temporal_sampling import TemporalSamplingConfig
from src.tasks.ball_detection.training.heatmap_pretraining.data import (
    HeatmapWindowDataset,
)
from src.tasks.ball_detection.training.heatmap_pretraining.evaluation import (
    HeatmapMetrics,
)
from src.tasks.ball_detection.training.heatmap_pretraining.runner import learning_rate
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def test_masks_follow_review_states_and_preserve_existing_sampling(tmp_path: Path) -> None:
    frames = [frame(i, ball(xy=(8., 9.))) for i in range(128)]
    frames[4] = frame(4)
    frames[5] = frame(5, ball("out_of_frame", None))
    frames[6] = frame(6, ball("unresolved", None))
    frames[7] = frame(7, ball("interpolated", (8., 9.)))
    frames[8] = frame(8, ball(), annotated=False)
    frames[9] = frame(9, ball(), ball(track="b2"))
    root = write_store_clip(tmp_path / "store", "train/clip", frames, size=(32, 32))
    store = BallFrameStore(root)
    manifest = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig(), input_kind="mdd_only")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = HeatmapWindowDataset(path, split="train")
    sample = dataset[next(i for i, (_, w) in enumerate(dataset.windows) if w.start == 0 and w.frame_step == 1)]
    assert sample["heatmap_valid"][4:10].tolist() == [True, True, False, False, False, False]
    assert sample["position_valid"][4:10].tolist() == [False] * 6
    assert torch.equal(collate_coordinate_windows([sample])["heatmap_valid"][0], sample["heatmap_valid"])


def test_evaluation_ownership_avoids_duplicate_loss_and_confident_wrong_position() -> None:
    metrics = HeatmapMetrics((1,))
    batch = dict(clip_id=["clip"], frame_step=[1], start=[0], source=["meiji"], common_evaluation=[True],
        frame_indices=torch.arange(32)[None], position_valid=torch.ones(1, 32, dtype=torch.bool),
        heatmap_valid=torch.ones(1, 32, dtype=torch.bool))
    error = torch.zeros(1, 32)
    error[0, 3] = 100
    metrics.add(batch, error, torch.ones(1, 32), torch.full((1, 32), .1))
    metrics.add(batch, error, torch.zeros(1, 32), torch.full((1, 32), 9.))
    group = metrics.report()["scopes"]["full"]["by_frame_step"]["1"]
    assert group["supervised_frames"] == 32
    assert group["true_positive"] == 31 and group["false_positive"] == 1 and group["false_negative"] == 1
    assert abs(group["heatmap_loss"] - .1) < 1e-6


def test_warmup_cosine_schedule_boundaries() -> None:
    assert learning_rate(0, peak=.001, total=100, warmup=10) == .0001
    assert learning_rate(9, peak=.001, total=100, warmup=10) == .001
    assert learning_rate(10, peak=.001, total=100, warmup=10) == .001
    assert abs(learning_rate(99, peak=.001, total=100, warmup=10) - .0001) < 1e-12
