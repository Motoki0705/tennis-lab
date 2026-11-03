from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
    BallPointsOutput,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    reconstruct_ball,
    single_ball_observations,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.ball_points import (
    BallPointsInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.observations import gather_balls
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.geometry.triangulation import PinholeCamera


def output() -> BallDetectionOutput:
    return BallDetectionOutput('cam0', np.arange(4, dtype=np.int64),
        np.array([[0., 0.], [20., 30.], [40., 50.], [60., 70.]], np.float32),
        np.array([.9, .8, .7, .6], np.float32), np.array([True, False, False, False]),
        np.array([1, 0, 2, 3], np.uint8), 'annotation_acceptance_not_probability', None)


def test_only_observed_points_survive_storage_and_join(tmp_path: Path) -> None:
    source = ClipSource('test', (SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 4, 30., 1920, 1080),))
    module = BallPointsModule()
    assert module.io.version == 4
    assembled = BallPointsInputAssembler().assemble(AssemblyContext(source, 'cam0'), {'detections': output()})
    points = module.process(assembled)
    # Even a real observation at pixel (0, 0) remains observed. Interpolation/occlusion estimates do not.
    assert points.observed.tolist() == [True, False, False, False]
    assert (points.uv_px == 0).all()
    np.testing.assert_allclose(points.confidence, [.9, 0, 0, 0])
    codec = ArtifactCodec(BallPointsOutput)
    payload, arrays = codec.dump(points, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    joined = gather_balls(source, {'ball_cam0': restored})
    np.testing.assert_array_equal(joined.observed[0, :, 0], points.observed)
    np.testing.assert_array_equal(joined.visibility(0)[0, :, 0, 0], points.observed)
    np.testing.assert_array_equal(joined.confidence[0, :, 0, 0], points.confidence)
    with pytest.raises(ValueError, match='finite'):
        replace(points, uv_px=np.full((4, 2), np.nan, np.float32))
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        replace(points, confidence=np.full(4, -1., np.float32))
    with pytest.raises(ValueError, match='Missing points'):
        replace(points, uv_px=np.ones((4, 2), np.float32))
    with pytest.raises(ValueError, match='Fields disagree'):
        codec.load({**payload, 'presence_probability': [1.] * 4}, tmp_path, arrays)


def test_assembler_checks_camera_timeline_and_payload_type(tmp_path: Path) -> None:
    source = ClipSource('test', (SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 4, 30., 1920, 1080),))
    context = AssemblyContext(source, 'cam0')
    with pytest.raises(ValueError, match='mismatch'):
        BallPointsInputAssembler().assemble(context, {'detections': replace(output(), camera_id='cam1')})
    with pytest.raises(TypeError, match='detector observations'):
        BallPointsInputAssembler().assemble(context, {'detections': object()})
    with pytest.raises(TypeError, match='observed masks'):
        gather_balls(source, {'ball_cam0': object()})


def test_filled_and_rejected_points_cannot_supply_the_second_real_view(tmp_path: Path) -> None:
    frames = 70
    source = ClipSource('test', tuple(SourceVideo(c, tmp_path / f'{c}.mp4', 'hash', frames, 60., 1280, 720)
                                   for c in ('cam0', 'cam1', 'cam2')))
    intrinsic = np.array([[500., 0, 640], [0, 500, 360], [0, 0, 1]])
    cameras = tuple(PinholeCamera(c, intrinsic, np.eye(3), np.array([shift, 0., 0.]))
                    for c, shift in zip(source.camera_ids, (-1., 1., 0.), strict=True))
    truth = np.column_stack((np.linspace(0, 1, frames), np.zeros(frames), np.full(frames, 2.)))
    rows = {}
    for index, camera in enumerate(cameras):
        uv = camera.project(truth)[0].astype(np.float32)
        observed: NDArray[np.bool_] = np.ones(frames, dtype=np.bool_)
        if index == 1:
            observed[20:24] = False
            uv[40] += 500
        if index == 2:
            observed[:] = False
        detections = BallDetectionOutput(camera.camera_id, np.arange(frames, dtype=np.int64), uv,
            np.ones(frames, np.float32), observed, np.where(observed, 1, 0).astype(np.uint8),
            'annotation_acceptance_not_probability', None)
        rows[f'ball_{camera.camera_id}'] = BallPointsModule().process(BallPointsInput(detections, source.size, source.fps))
    assert rows['ball_cam1'].smoothing.interpolated[20:24].all()
    assert rows['ball_cam1'].smoothing.rejected[40]
    grouped = single_ball_observations(gather_balls(source, rows), threshold=0)
    result = reconstruct_ball(grouped, cameras, fps=60., reprojection_px=2)
    assert not result.trajectory.valid[20:24].any()
    assert not result.trajectory.valid[40]
    assert not result.visibility[2].any()
    supported: NDArray[np.bool_] = np.ones(frames, dtype=np.bool_)
    supported[20:24] = supported[40] = False
    np.testing.assert_array_equal(result.trajectory.valid, supported)
    np.testing.assert_allclose(result.trajectory.positions[supported], truth[supported], atol=1e-4)
