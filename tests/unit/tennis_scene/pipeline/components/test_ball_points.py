from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.inference import SequencePrediction
from src.tasks.ball_refiner.refiner_2d.confidence import PointConfidenceRule
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsModule,
    BallPointsOutput,
)
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.ball_points import (
    BallPointsInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.observations import gather_balls
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def output() -> BallRefiner2DOutput:
    distribution = BallGMM2D(torch.full((1, 4, 1, 2), .5),
                            torch.eye(2).repeat(1, 4, 1, 1, 1) * torch.tensor([.001, .001, .5, .5])[None, :, None, None, None],
                            torch.zeros(1, 4, 1), torch.tensor([[20., -20., 20., -20.]]))
    frame: NDArray[np.int64] = np.arange(4, dtype=np.int64)
    zeros: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    return BallRefiner2DOutput("cam0", (1920, 1080), frame, frame, "1/30", (frame / 30).astype(np.float32),
                               SequencePrediction(distribution, frame, zeros, 1), frame, zeros, 1, "uncalibrated")


def test_rejection_is_missing_and_survives_codec_and_consumer_join(tmp_path: Path) -> None:
    module = BallPointsModule(PointConfidenceRule(.9, 30000.), distribution_version=1)
    points = module.process(output())
    np.testing.assert_array_equal(points.rejection_codes, [0, 1, 2, 3])
    assert points.observed.tolist() == [True, False, False, False]
    assert not points.uv_px[1:].any() and not points.confidence[1:].any()
    codec = ArtifactCodec(BallPointsOutput)
    payload, arrays = codec.dump(points, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    source = ClipSource("test", (SourceVideo("cam0", tmp_path / "x.mp4", "hash", 4, 30., 1920, 1080),))
    joined = gather_balls(source, {"ball_cam0": restored})
    np.testing.assert_array_equal(joined.observed[0, :, 0], points.observed)
    assert not joined.visibility(0)[0, 1:].any()
    with pytest.raises(ValueError, match="missing mask"):
        replace(points, observed=np.ones(4, bool))
    with pytest.raises(ValueError, match="zero coordinates"):
        replace(points, uv_px=np.ones((4, 2), np.float32))


def test_assembler_rejects_another_camera_and_detector_point_fallback(tmp_path: Path) -> None:
    source = ClipSource("test", (SourceVideo("cam0", tmp_path / "x.mp4", "hash", 4, 30., 1920, 1080),))
    assembler = BallPointsInputAssembler()
    context = AssemblyContext(source, "cam0")
    assert assembler.assemble(context, {"distribution": output()}).camera_id == "cam0"
    with pytest.raises(ValueError, match="mismatch"):
        assembler.assemble(context, {"distribution": replace(output(), camera_id="cam1")})
    with pytest.raises(TypeError, match="detector/annotation"):
        assembler.assemble(context, {"distribution": object()})
    with pytest.raises(TypeError, match="confidence-filtered"):
        gather_balls(source, {"ball_cam0": object()})
