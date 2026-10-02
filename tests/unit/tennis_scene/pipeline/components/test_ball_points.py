from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.inference import SequencePrediction
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
    distribution = BallGMM2D(
        torch.tensor([[[[.2, .3], [.8, .7]]] * 4]),
        torch.eye(2).repeat(1, 4, 2, 1, 1) * torch.tensor([.001, .001, .5, .5])[None, :, None, None, None],
        torch.tensor([[[2., 0.], [0., 2.], [1., 1.], [0., 2.]]]),
        torch.tensor([[20., -1000., 20., -1000.]]),
    )
    frame: NDArray[np.int64] = np.arange(4, dtype=np.int64)
    zeros: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    # Nonuniform PTS and overlapping windows must survive point projection intact.
    pts = np.array([11, 13, 16, 20], np.int64)
    starts = np.array([0, 0, 1, 1], np.int64)
    return BallRefiner2DOutput("cam0", (1920, 1080), frame, pts, "1/60", ((pts - pts[0]) / 60).astype(np.float32),
                              SequencePrediction(distribution, starts, frame - starts, 3), frame, zeros, 1, "uncalibrated")


def test_low_presence_and_large_covariance_keep_all_points_through_storage_and_join(tmp_path: Path) -> None:
    raw = output()
    codec = ArtifactCodec(BallRefiner2DOutput)
    payload, arrays = codec.dump(raw, tmp_path)
    restored_raw = codec.load(payload, tmp_path, arrays)
    for field in fields(BallGMM2D):
        torch.testing.assert_close(getattr(restored_raw.prediction.distribution, field.name), getattr(raw.prediction.distribution, field.name), rtol=0, atol=0)
    for key in ("pts", "timestamps_seconds", "detector_window_start", "detector_time_index"):
        np.testing.assert_array_equal(getattr(restored_raw, key), getattr(raw, key))
    np.testing.assert_array_equal(restored_raw.prediction.window_start, raw.prediction.window_start)
    np.testing.assert_array_equal(restored_raw.prediction.time_index, raw.prediction.time_index)
    module = BallPointsModule(distribution_version=1)
    assert module.io.version == 2
    points = module.process(restored_raw)
    expected = (raw.prediction.distribution.means[0, np.arange(4), [0, 1, 0, 1]].double().numpy() * [1919, 1079]).astype(np.float32)
    np.testing.assert_array_equal(points.uv_px, expected)
    assert points.presence_probability.tolist() == [pytest.approx(1.), 0., pytest.approx(1.), 0.]
    point_dir = tmp_path / "points"
    point_dir.mkdir()
    point_codec = ArtifactCodec(BallPointsOutput)
    payload, arrays = point_codec.dump(points, point_dir)
    restored = point_codec.load(payload, point_dir, arrays)
    source = ClipSource("test", (SourceVideo("cam0", tmp_path / "x.mp4", "hash", 4, 30., 1920, 1080),))
    joined = gather_balls(source, {"ball_cam0": restored})
    np.testing.assert_array_equal(joined.uv_px[0, :, 0, 0], expected)
    assert joined.observed.all() and joined.visibility(0).all()
    assert (joined.confidence == 1).all()  # Presence never becomes a point mask/weight.
    with pytest.raises(ValueError, match="finite"):
        replace(points, uv_px=np.full((4, 2), np.nan, np.float32))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        replace(points, presence_probability=np.full(4, -1., np.float64))
    # The old artifact contains confidence, observed, area, rule and rejection fields.
    with pytest.raises(ValueError, match="Fields disagree"):
        point_codec.load({**payload, "observed": [True, False, False, False]}, point_dir, arrays)


def test_assembler_rejects_another_camera_and_detector_point_fallback(tmp_path: Path) -> None:
    source = ClipSource("test", (SourceVideo("cam0", tmp_path / "x.mp4", "hash", 4, 30., 1920, 1080),))
    assembler = BallPointsInputAssembler()
    context = AssemblyContext(source, "cam0")
    assert assembler.assemble(context, {"distribution": output()}).camera_id == "cam0"
    with pytest.raises(ValueError, match="mismatch"):
        assembler.assemble(context, {"distribution": replace(output(), camera_id="cam1")})
    with pytest.raises(TypeError, match="detector/annotation"):
        assembler.assemble(context, {"distribution": object()})
    with pytest.raises(TypeError, match="unfiltered"):
        gather_balls(source, {"ball_cam0": object()})
