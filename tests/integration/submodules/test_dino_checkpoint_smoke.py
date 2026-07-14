"""GPU smoke test for the deployed DINO checkpoint and custom CUDA op."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.models.dino import DinoPersonDetector, PersonDetectionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = PROJECT_ROOT / "ckpt/dino/checkpoint0029_4scale_swin.pth"


@pytest.mark.gpu
@pytest.mark.slow
def test_dino_checkpoint_loads_strictly_and_runs_one_frame() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by DINO multi-scale deformable attention")
    if not CHECKPOINT.exists():
        pytest.skip(f"DINO checkpoint is unavailable: {CHECKPOINT}")

    detector = DinoPersonDetector(checkpoint=CHECKPOINT, device="cuda")
    try:
        result = detector.predict(
            PersonDetectionRequest(frame_bgr=np.zeros((180, 320, 3), dtype=np.uint8))
        )
        assert result.boxes_xyxy.ndim == 2
        assert result.boxes_xyxy.shape[1] == 4
        assert result.scores.shape == (len(result.boxes_xyxy),)
    finally:
        detector.unload()
