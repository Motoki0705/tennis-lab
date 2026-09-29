import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.person_tracking.features import FeatureConfig, FeatureExtractor


class Pose:
    calls = 0

    def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult:
        self.calls += 1
        frame, _ = list(request.frames_bgr)[0]
        assert (request.frame_indices == frame).all()
        keypoints = torch.zeros(len(request.bbx_xys), 17, 3)
        keypoints[..., :2] = request.bbx_xys[:, None, :2]
        keypoints[..., 2] = 1
        return Pose2DResult(keypoints)


class Encoder:
    name = 'test-prompt-aware'
    input_size = (16, 8)
    dimension = 2
    prompts: list[torch.Tensor]

    def __init__(self) -> None:
        self.prompts = []

    def embed(self, crops: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        assert crops.shape[1:] == (3, 16, 8)
        self.prompts.append(prompts)
        return torch.tensor([[1., 0.]]).repeat(len(crops), 1)


def test_multiple_detections_share_pose_pass_and_keep_rows_and_crop_relative_prompts() -> None:
    pose, encoder = Pose(), Encoder()
    extractor = FeatureExtractor(pose, encoder, FeatureConfig(appearance_batch_size=1))
    result = extractor.extract(5, np.zeros((1080, 1920, 3), np.uint8), np.array([12, 15, 16], np.int64),
                               np.array([[0, 0, 20, 40], [20, 30, 40, 70], [10, 10, 12, 12]], np.float32),
                               np.full(3, .9, np.float32))
    assert pose.calls == 1 and result.rows.tolist() == [12, 15, 16]
    assert result.appearance_valid.tolist() == [True, True, False]
    assert np.all(result.embeddings[2] == 0)
    assert len(encoder.prompts) == 2
    assert all(torch.allclose(prompt[..., :2], torch.full((1, 17, 2), .5)) for prompt in encoder.prompts)


def test_empty_frames_do_not_call_models_and_preserve_dimension() -> None:
    pose, encoder = Pose(), Encoder()
    result = FeatureExtractor(pose, encoder).extract(0, np.zeros((10, 10, 3), np.uint8),
              np.empty(0, np.int64), np.empty((0, 4), np.float32), np.empty(0, np.float32))
    assert pose.calls == 0 and encoder.prompts == []
    assert result.embeddings.shape == (0, 2) and result.poses.shape == (0, 17, 3)


def test_invalid_detection_fails_before_model_call() -> None:
    pose, encoder = Pose(), Encoder()
    with pytest.raises(ValueError, match='positive area'):
        FeatureExtractor(pose, encoder).extract(0, np.zeros((10, 10, 3), np.uint8), np.array([0], np.int64),
                  np.array([[3, 3, 1, 1]], np.float32), np.ones(1, np.float32))
    assert pose.calls == 0
