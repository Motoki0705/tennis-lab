"""Local-asset tests for the ball-detection backend (CPU only).

These exercise the real curated checkpoint and real TrackNet frames on CPU.
They never request CUDA: the shared web layer owns GPU scheduling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.ball_detection.visualization.review.checkpoints import BallCheckpointInfo

REPO_ROOT = Path("/home/kamimura/projects/tennis-lab")
DATA_ROOT = REPO_ROOT / "data"
CURATED_CHECKPOINT = REPO_ROOT / "ckpt" / "ball_detection"
TRACKNET_CLIP = DATA_ROOT / "tennis" / "tracknet" / "game1" / "Clip1"

pytestmark = [
    pytest.mark.local_data,
    pytest.mark.skipif(
        not TRACKNET_CLIP.is_dir(),
        reason="TrackNet dataset is unavailable",
    ),
]


def curated_checkpoint(service: DetectionService) -> BallCheckpointInfo:
    """Return the description of the first curated checkpoint on disk."""
    candidates = sorted(CURATED_CHECKPOINT.glob("*.ckpt"))
    if not candidates:
        pytest.skip("No curated ball-detection checkpoint is available")
    wanted = candidates[0].resolve()
    for info in service.checkpoints().values():
        if info.path.resolve() == wanted:
            return info
    raise pytest.skip.Exception(
        f"{candidates[0].name} is not within the service's checkpoint roots"
    )


def test_curated_checkpoint_is_described_from_its_own_config() -> None:
    service = DetectionService(REPO_ROOT)
    info = curated_checkpoint(service)
    assert info.usable, info.error
    assert info.model == "conv_next_unet"
    assert info.num_frames == 8
    assert info.image_size_hw == (288, 512)
    assert info.minimum_window == 2


def test_cpu_window_inference_matches_labels_in_original_pixels() -> None:
    service = DetectionService(REPO_ROOT)
    info = curated_checkpoint(service)
    checkpoint = info.id
    scenes = service.scenes("tracknet", limit=1)
    if not scenes["items"]:
        pytest.skip("No TrackNet scene is available")
    scene = scenes["items"][0]["id"]
    start = 20
    service.validate(checkpoint, scene, start=start, count=8, device="cpu")
    result = service.infer(
        checkpoint, scene, start=start, count=8, threshold=0.5, device="cpu"
    )
    preview = service.preview(scene, start=start, count=8)
    assert result["metrics"]["window"]["mode"] == "temporal"
    assert [item["index"] for item in result["items"]] == list(range(start, start + 8))

    ground_truth = {
        item["index"]: item["gt"]["points"] for item in preview["items"]
    }
    matched = 0
    distances: list[float] = []
    for item in result["items"]:
        predictions = item["pred"]["points"]
        targets = ground_truth[item["index"]]
        assert item["pred"]["rasters"][0]["name"] == "probability"
        if not predictions or not targets:
            continue
        best = min(
            float(((p["x"] - t["x"]) ** 2 + (p["y"] - t["y"]) ** 2) ** 0.5)
            for p in predictions
            for t in targets
        )
        distances.append(best)
        if best <= info.metrics.ball_distance_threshold:
            matched += 1
    # The checkpoint's own metric threshold is 4 px in *original* pixels, which
    # is strict on a 1280x720 clip; the epoch-13 checkpoint still lands most
    # frames inside it. A broken resize or MDD path collapses this to near zero.
    assert len(distances) == 8
    assert matched >= 5
    assert float(sorted(distances)[len(distances) // 2]) <= 2.0 * (
        info.metrics.ball_distance_threshold
    )
