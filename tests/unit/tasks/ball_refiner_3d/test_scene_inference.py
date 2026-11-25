"""SceneResult frame identity and the explicit near-nominal FPS policy."""

import hashlib

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor
from src.tasks.ball_refiner_3d.inference.scene import predict_scene_events, scene_input
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerPrediction
from src.tasks.ball_refiner_3d.model_io.factory import build_refiner
from src.tennis_scene.schema import SceneResult


def test_scene_prediction_preserves_mask_frame_grid_and_actual_fps(
    tmp_path, monkeypatch
):
    scene = SceneResult(
        num_frames=3,
        fps=60000 / 1001,
        width=1280,
        height=720,
        court_kp=np.empty((0, 3, 0, 2), np.float32),
        court_vis=np.empty((0, 3, 0), np.float32),
        player_position=np.empty((0, 3, 3), np.float32),
        player_yaw=np.empty((0, 3), np.float32),
        ball_3d=np.array([[1, -2, 3], [np.nan, np.nan, np.nan], [4, 5, 6]], np.float32),
        ball_3d_valid=np.array([True, False, True]),
        metadata={"scene_schema_version": 2},
    )
    source = tmp_path / "scene.json"
    source.write_bytes(b"fixed index")
    monkeypatch.setattr(
        "src.tasks.ball_refiner_3d.inference.scene.load_scene_result", lambda _: scene
    )
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0, 2, 64, 8, 10000, "swiglu", False)
    )
    predictor = RefinerPredictor(model, {"fps": 60}, torch.device("cpu"))
    coordinates, missing, fps = scene_input(scene)
    torch.testing.assert_close(
        coordinates[0],
        torch.tensor([[1, -2, 3], [0, 0, 0], [4, 5, 6]], dtype=torch.float32),
    )
    assert missing.tolist() == [[False, True, False]]
    with pytest.raises(ValueError, match="does not match checkpoint FPS"):
        predictor.predict(coordinates, missing, fps=fps, seed=0)

    probability = torch.tensor([[[0.1, 0.1, 0.1], [0.7, 0.2, 0.8], [0.1, 0.6, 0.1]]])
    monkeypatch.setattr(
        "src.tasks.ball_refiner_3d.inference.predictor.refine_coordinates",
        lambda _, xy, *args, **kwargs: RefinerPrediction(xy, probability),
    )
    prediction, document = predict_scene_events(
        predictor,
        source,
        clip_id="video_000/clip_000",
        checkpoint_sha256="a" * 64,
        seed=0,
        fps_tolerance=0.06,
    )
    assert prediction.event_probability.shape == (1, 3, 3)
    assert document["schema"] == "ball_event_predictions.v1"
    assert document["fps"] == fps and document["frames"] == 3
    assert document["inputs"] == {
        "scene_sha256": hashlib.sha256(b"fixed index").hexdigest()
    }
    assert [
        (event["kind"], event["frame"], event["flags"]) for event in document["events"]
    ] == [
        ("net", 1, ["ball_3d_missing"]),
        ("shot", 1, ["ball_3d_missing"]),
        ("bounce", 2, []),
    ]
    assert document["events"][0]["time_s"] == pytest.approx(1001 / 60000)
    assert document["events"][0]["score"] == pytest.approx(0.8)
