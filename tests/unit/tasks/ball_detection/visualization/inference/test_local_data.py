"""Local-asset tests for the ball-detection backend (CPU only).

These exercise the real curated checkpoint and real TrackNet frames on CPU.
They never request CUDA: the shared web layer owns GPU scheduling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.visualization.inference.loader import load_ball_model
from src.tasks.ball_detection.visualization.inference.peaks import decode_frame_peaks
from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.ball_detection.visualization.review.checkpoints import BallCheckpointInfo
from src.utils.data.augmentation import normalize_frames_imagenet

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


def test_cpu_window_matches_dataset_preprocessing_and_original_pixels() -> None:
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
    # Independently apply the dataset's NumPy normalization. The public UI
    # accepts raw RGB, while this reference supplies dataset-preprocessed RGB.
    resolved = service._resolve_scene(scene)
    plan = service._plan_window(info, resolved, start=start, count=8)
    loaded = load_ball_model(info.path, device="cpu")
    assert loaded.image_size_hw is not None
    raw = service._window_tensor(resolved.frames, plan, size=loaded.image_size_hw)
    normalization = loaded.image_normalization
    frames = list(raw[0].permute(0, 2, 3, 1).numpy())
    if normalization.enabled:
        frames = normalize_frames_imagenet(
            frames,
            mean=np.asarray(normalization.mean, dtype=np.float32).reshape(1, 1, 3),
            std=np.asarray(normalization.std, dtype=np.float32).reshape(1, 1, 3),
        )
    prepared = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    with torch.no_grad():
        call = loaded.adapter.prepare_model_call(
            prepared, image_normalization=normalization, preprocessed=True,
        )
        heatmaps = loaded.adapter.probability_heatmaps(loaded.model(*call.model_args), call)[0]
    reference = decode_frame_peaks(
        heatmaps, original_size=resolved.frames.original_size(start), threshold=.5,
        nms_kernel=info.metrics.nms_kernel,
        max_peaks=info.metrics.max_predictions_per_frame,
        subpixel_refine=info.metrics.subpixel_refine,
    )
    distances: list[float] = []
    for item, expected in zip(result["items"], reference, strict=True):
        predictions = item["pred"]["points"]
        targets = ground_truth[item["index"]]
        np.testing.assert_allclose(
            [(point["x"], point["y"]) for point in predictions], expected.points,
            rtol=0, atol=1e-4,
        )
        assert item["pred"]["rasters"][0]["name"] == "probability"
        if not predictions or not targets:
            continue
        best = min(
            float(((p["x"] - t["x"]) ** 2 + (p["y"] - t["y"]) ** 2) ** 0.5)
            for p in predictions
            for t in targets
        )
        distances.append(best)
    # Keep a GT-distance smoke check. Exact recall at the checkpoint's 4px
    # threshold is an experiment metric, not a preprocessing specification:
    # restoring its saved normalization changes this window from 5 to 4 matches.
    # Pixel-exact agreement with the independent dataset path guards the fix.
    assert len(distances) == 8
    assert float(sorted(distances)[len(distances) // 2]) <= 2.0 * (
        info.metrics.ball_distance_threshold
    )
