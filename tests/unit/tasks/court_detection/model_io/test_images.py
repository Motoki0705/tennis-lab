"""Court pose inference preserves the validation long-side/padding geometry."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.model_io.images import prepare_court_input


@pytest.mark.parametrize('height,width,expected', [(1080, 1920, (288, 512)), (1920, 1080, (512, 288)), (361, 848, (224, 512))])
def test_pose_input_has_long_side_512_and_only_minimum_patch_padding(height, width, expected):
    bundle = CourtTargetBundleSpec({'kp': CourtTargetSpec('kp', 'test', 1, ('point',), torch.float32, False)})
    spec = CourtModelSpec(bundle, 3, 512, pose_long_side=True, patch_size=16)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    prepared = prepare_court_input(image, spec=spec, device=torch.device('cpu'))
    assert prepared.images.shape == (1, 3, *expected)
    assert prepared.original_size_hw == (height, width)
    assert prepared.source_from_model_xy == pytest.approx((max(height, width) / 512,) * 2)
    assert all(0 <= padded - content < 16 for padded, content in zip(expected, prepared.content_size_hw, strict=True))
    pil = prepare_court_input(Image.fromarray(image), spec=spec, device=torch.device('cpu'))
    torch.testing.assert_close(pil.images, prepared.images)
