"""Shared synthetic GMM fixture; actual model/bundle IO is integration-tested.

Only the model recipe is replaced. Production unfiltered point projection, consumer
joins, side decisions, triangulation, storage and reload remain real here.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.inference import SequencePrediction
from src.tasks.ball_refiner.refiner_2d.calibration import CovarianceCalibration
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.components.ball_refiner import (
    BallRefiner2DOutput,
    CalibratedBallRefiner2DOutput,
)
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    BallDetectionInputAssembler,
)
from src.tennis_scene.pipeline.runner import ComponentNode


@dataclass
class FixtureInput:
    version: int = 1

    def assemble(self, context: Any, artifacts: Any) -> BallDetectionOutput:
        return artifacts["detections"]  # Synthetic test input, never a production adapter.


@dataclass
class FixtureRefiner:
    size: tuple[int, int]
    io = ComponentIO("ball_refiner_2d", BallDetectionOutput, BallRefiner2DOutput,
                     {"detections": InputPort("ball_detections", 2)}, "ball_distribution_2d")

    def process(self, inputs: BallDetectionOutput) -> BallRefiner2DOutput:
        n = len(inputs.frame_indices)
        uv = np.clip(inputs.uv_px / (np.array(self.size) - 1), 0, 1).astype(np.float32)
        gmm = BallGMM2D(torch.from_numpy(uv)[None, :, None], torch.eye(2).repeat(1, n, 1, 1, 1) * .00001,
                        torch.zeros(1, n, 1), torch.from_numpy(np.where(inputs.observed, 20., -20.).astype(np.float32))[None])
        frames: NDArray[np.int64] = np.arange(n, dtype=np.int64)
        zeros: NDArray[np.int64] = np.zeros(n, np.int64)
        return BallRefiner2DOutput(inputs.camera_id, self.size, frames, frames, "1/30",
                                   (frames / 30).astype(np.float32), SequencePrediction(gmm, frames, zeros, 1),
                                   frames, zeros, 1, "uncalibrated")


class CalibratedFixtureRefiner(FixtureRefiner):
    io = ComponentIO("ball_refiner_2d", BallDetectionOutput, CalibratedBallRefiner2DOutput,
                     {"detections": InputPort("ball_detections", 2)}, "ball_distribution_2d", version=2)

    def process(self, inputs: BallDetectionOutput) -> CalibratedBallRefiner2DOutput:
        original = super().process(inputs)
        values = {field.name: getattr(original, field.name) for field in fields(original)}
        values["calibration"] = "covariance_scale_v1"
        return CalibratedBallRefiner2DOutput(**values, covariance_calibration=CovarianceCalibration(1., "0" * 64),
                                             calibration_artifact_sha256="1" * 64)


@pytest.fixture(autouse=True)
def synthetic_model_recipe(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    version = getattr(request, "param", 1)
    if version not in (1, 2):
        raise ValueError("Synthetic recipe supports only distribution schema v1/v2")
    def recipe(source: ClipSource, **kwargs: Any) -> tuple[ComponentNode, ...]:
        result: list[ComponentNode] = []
        for camera in source.camera_ids:
            context = AssemblyContext(source, camera)
            detector = BallDetectionModule(kwargs["detector_config"])
            refiner = FixtureRefiner(source.size) if version == 1 else CalibratedFixtureRefiner(source.size)
            result.extend((ComponentNode(f"ball_detection/{camera}", detector, detector.io, BallDetectionInputAssembler(),
                                         {}, context, {}, kwargs["code_identity"], kwargs["execution_source"]),
                           ComponentNode(f"ball_refiner_2d/{camera}", refiner, refiner.io, FixtureInput(),
                                         {"detections": f"ball_detection/{camera}"}, context, {},
                                         kwargs["code_identity"], kwargs["execution_source"])))
        return tuple(result)

    monkeypatch.setattr("src.tennis_scene.pipeline.definition.ball_refiner_definition", recipe)
