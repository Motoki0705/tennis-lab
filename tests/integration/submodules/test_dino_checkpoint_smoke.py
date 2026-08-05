"""GPU smoke test for the deployed DINO checkpoint and custom CUDA op."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.models.dino import DinoPersonDetector, PersonDetectionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = PROJECT_ROOT / "ckpt/dino/checkpoint0029_4scale_swin.pth"
REPOSITORY = PROJECT_ROOT / "third_party/DINO"


@pytest.mark.cuda
@pytest.mark.slow
def test_dino_checkpoint_loads_strictly_and_runs_one_frame() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by DINO multi-scale deformable attention")
    if not CHECKPOINT.exists():
        pytest.skip(f"DINO checkpoint is unavailable: {CHECKPOINT}")

    detector = DinoPersonDetector(
        checkpoint=CHECKPOINT,
        repository=REPOSITORY,
        device="cuda",
        allow_device_fallback=False,
        confidence=0.35,
        short_side=800,
        max_long_side=1333,
    )
    try:
        result = detector.predict(
            PersonDetectionRequest(frame_bgr=np.zeros((180, 320, 3), dtype=np.uint8))
        )
        assert result.boxes_xyxy.ndim == 2
        assert result.boxes_xyxy.shape[1] == 4
        assert result.scores.shape == (len(result.boxes_xyxy),)
    finally:
        detector.unload()
