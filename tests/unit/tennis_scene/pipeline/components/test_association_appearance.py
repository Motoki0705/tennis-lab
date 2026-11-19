"""A saved row is admissible only when the inference can be reproduced."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.person_tracking.sequence import CLIP_ENCODER, TrackEvidence
from src.tasks.player_association.appearance import sampling
from src.tennis_scene.pipeline.components import person_features
from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
    camera_track_appearances,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.utils.video import FramePacket
from tests.unit.tennis_scene.pipeline.config_factories import fake_encoder_spec


class PixelEncoder:
    name = CLIP_ENCODER
    input_size = (8, 4)

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(len(pixels))
        embedding = torch.zeros((len(pixels), 1280))
        embedding[:, :3] = pixels[:, :, 0, 0]
        return torch.nn.functional.normalize(embedding, dim=1)


@pytest.mark.parametrize(('change', 'reason', 'misses'), [
    ('none', '', []),
    ('box', 'box_mismatch', [1]),
    ('image', 'image_mismatch', [1]),
    ('weights', 'feature_identity_mismatch', [0, 1, 3]),
    ('preprocessing', 'feature_identity_mismatch', [0, 1, 3]),
    ('row_from_another_frame', 'detection_row_mismatch', [1]),
    ('invalid_rows', 'appearance_invalid', [0, 3]),
    ('image_evidence_missing', 'image_evidence_missing', [0, 1, 3]),
    ('missing_detection_evidence', 'missing_detection_evidence', [0, 1, 3]),
])
def test_only_matching_samples_skip_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             change: str, reason: str, misses: list[int]) -> None:
    image_size = 64, 48
    images: list[NDArray[np.uint8]] = [np.full((48, 64, 3), [5, 20 + 10 * frame, 10], np.uint8) for frame in range(4)]

    def reader(path: Path, *, max_frames: int, **kwargs: Any) -> list[FramePacket[np.ndarray]]:
        return [FramePacket(frame, image, image_size) for frame, image in enumerate(images[:max_frames])]

    monkeypatch.setattr(sampling, 'OpenCVVideoFrameReader', reader)
    monkeypatch.setattr(person_features, 'OpenCVVideoFrameReader', reader)
    video = SourceVideo('cam0', tmp_path / 'input.mp4', 'media', 4, 30., *image_size)
    # Sparse source rows and varying positions on the frame axis.
    detections = PersonDetectionOutput(video.camera_id, np.array([0, 1, 2, 2, 3], np.int64),
        np.tile(np.array([5.25, 6, 36, 40], np.float32), (3, 1)), np.ones(3, np.float32), np.array([3, 8, 19], np.int64))
    weights = tmp_path / 'weights'
    weights.write_bytes(b'CLIP weights A')
    encoder = PixelEncoder()
    spec = replace(fake_encoder_spec(lambda: encoder), weights={'checkpoint': weights})
    features_module = person_features.PersonAppearanceFeaturesModule(spec, capture_frame_sha256=change != 'image_evidence_missing')
    feature_data = features_module.process(person_features.PersonAppearanceFeaturesInput(video, detections))
    boxes: NDArray[np.float32] = np.zeros((1, 4, 4), np.float32)
    boxes[0, [0, 1, 3]] = detections.boxes_xyxy
    observed = np.array([[True, True, False, True]])
    # Track/selection evidence deliberately carries different appearance values.
    track_embeddings: NDArray[np.float32] = np.zeros((1, 4, 1280), np.float32)
    track_embeddings[0, [0, 1, 3], 99] = 1
    evidence = TrackEvidence(np.array([[3, 8, -1, 19]], np.int64), None, track_embeddings, observed, CLIP_ENCODER)
    tracks = PersonTrackingOutput(video.camera_id, np.array([11], np.int64), boxes, observed, ((11,),), (), evidence)
    preprocessing = features_module.preprocessing_settings()

    if change == 'box':
        boxes[0, 1, 0] += .125  # Different raw box with identical rounded pixels.
    elif change == 'image':
        images[1][0, 0, 0] = 254  # Outside the crop, within the decoded source image.
    elif change == 'weights':
        other_weights = tmp_path / 'other-weights'
        other_weights.write_bytes(b'CLIP weights B')
        spec = replace(spec, weights={'checkpoint': other_weights})
    elif change == 'preprocessing':
        preprocessing = {**preprocessing, 'min_appearance_height_px': 19.}
    elif change == 'row_from_another_frame':
        evidence.detection_rows[0, 1] = 3
    elif change == 'invalid_rows':
        feature_data.appearance_valid[[0, 2]] = False
        feature_data.embeddings[[0, 2]] = 0
    elif change == 'missing_detection_evidence':
        tracks = replace(tracks, evidence=None)

    encoder.calls.clear()
    appearances, diagnostics, reasons = camera_track_appearances(video, tracks, sampling.CropSamplingConfig(min_height_px=10.),
        saved=SavedAssociationAppearance(detections, feature_data), preprocessing=preprocessing,
        encoder=spec, name=CLIP_ENCODER, borrow_encoder=lambda: encoder)

    assert encoder.calls == ([len(misses)] if misses else [])
    assert dict(reasons) == ({reason: len(misses)} if misses else {})
    assert diagnostics == [{'track_id': 11, 'samples': 3, 'rejected': {},
                            'reused_frames': [frame for frame in (0, 1, 3) if frame not in misses],
                            'recomputed_frames': {reason: misses} if misses else {}}]
    appearance, = appearances
    np.testing.assert_array_equal(appearance.frames, [0, 1, 3])
    expected = np.array([[10, 20, 5], [10, 30, 5], [10, 50, 5]], np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(appearance.embeddings[:, :3], expected, atol=1e-7)
    assert not appearance.embeddings[:, 3:].any()
