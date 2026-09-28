"""Camera-local full GMM output; never fall back to detector point estimates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner.deployment import InferenceBundle
from src.tasks.ball_refiner.inference import SequencePrediction, predict_sequence
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.utils.device import resolve_device


@dataclass(frozen=True)
class BallRefiner2DInput:
    video: SourceVideo
    pts: NDArray[np.int64]
    time_base: str
    model_input: Refiner2DInput
    detector_window_start: NDArray[np.int64]
    detector_time_index: NDArray[np.int64]


@dataclass(frozen=True)
class BallRefiner2DOutput:
    """source-grid uv GMM and exact PTS; covariance/weights derive losslessly."""

    camera_id: str
    source_size_wh: tuple[int, int]
    frame_indices: NDArray[np.int64]
    pts: NDArray[np.int64]
    time_base: str
    timestamps_seconds: NDArray[np.float32]
    prediction: SequencePrediction
    detector_window_start: NDArray[np.int64]
    detector_time_index: NDArray[np.int64]
    detector_window_length: int
    calibration: str

    def __post_init__(self) -> None:
        n = self.prediction.distribution.means.shape[1]
        if (not self.camera_id or len(self.source_size_wh) != 2
                or any(type(x) is not int or x <= 1 for x in self.source_size_wh)
                or self.calibration != "uncalibrated"):
            raise ValueError("Invalid refiner source size, camera or calibration status")
        for array, dtype in ((self.frame_indices, np.int64), (self.pts, np.int64),
                             (self.timestamps_seconds, np.float32), (self.detector_window_start, np.int64),
                             (self.detector_time_index, np.int64)):
            if array.shape != (n,) or array.dtype != dtype or not np.isfinite(array).all():
                raise ValueError("Refiner artifact must cover the complete source timeline")
        if not np.array_equal(self.frame_indices, np.arange(n)) or (np.diff(self.pts) <= 0).any():
            raise ValueError("Refiner source frame indices/PTS must be ordered and complete")
        base = Fraction(self.time_base)
        expected = ((self.pts - self.pts[0]).astype(np.float64) * float(base)).astype(np.float32)
        if base <= 0 or not np.array_equal(self.timestamps_seconds, expected) or (np.diff(expected) <= 0).any():
            raise ValueError("Refiner seconds must derive from actual presentation timestamps")
        length = self.detector_window_length
        if (type(length) is not int or not 1 <= length <= n or (self.detector_window_start < 0).any()
                or (self.detector_window_start + length > n).any() or (self.detector_time_index < 0).any()
                or (self.detector_time_index >= length).any()
                or not np.array_equal(self.detector_window_start + self.detector_time_index, self.frame_indices)):
            raise ValueError("Refiner artifact detector provenance must address real source frames")


class BallRefiner2DModule:
    io = ComponentIO("ball_refiner_2d", BallRefiner2DInput, BallRefiner2DOutput,
                     {"detections": InputPort("ball_detections", version=2)}, "ball_distribution_2d", version=1)

    def __init__(self, bundle: InferenceBundle, *, device: str, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Refiner batch size must be a positive integer")
        self.bundle, self.device, self.batch_size = bundle, device, batch_size

    def process(self, inputs: BallRefiner2DInput) -> BallRefiner2DOutput:
        video = inputs.video
        if inputs.model_input.timestamps_seconds.shape != (1, video.num_frames):
            raise ValueError("Refiner input must preserve one camera's full source timeline")
        device = resolve_device(self.device)
        pair = self.bundle.load_model()
        try:
            pair.model.to(device)
            prediction = predict_sequence(
                pair, inputs.model_input, window_length=self.bundle.window_length, stride=self.bundle.stride,
                batch_size=self.batch_size, device=device,
            )
        finally:
            del pair
            release_inference_memory(self.device)
        return BallRefiner2DOutput(
            video.camera_id, (video.width, video.height), np.arange(video.num_frames, dtype=np.int64),
            inputs.pts, inputs.time_base, inputs.model_input.timestamps_seconds[0].numpy().copy(), prediction,
            inputs.detector_window_start, inputs.detector_time_index, self.bundle.detector.window_length, "uncalibrated",
        )
