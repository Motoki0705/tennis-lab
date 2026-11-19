"""Native artifact restoration, input contracts and unthresholded evidence."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from src.tasks.ball_detection.inference.x3d_dpt import (
    X3DDPTPredictor,
    scene_dpt_input_contract,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import heatmap_model_payload
from src.tasks.ball_detection.models.mdd import MDDDPTDetector
from src.utils.inference_recording import InferenceRecorder
from tests.support.tasks.ball_detection.temporal import small_model


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    config = replace(small_model("dpt", frames=2), encoder_variant="x3d_csn", temporal_mixing="depthwise3d")
    model = MDDDPTDetector(config)
    saved = heatmap_model_payload(model)
    saved["scene_input_contract"] = scene_dpt_input_contract(image_size=(24, 40), frames=2, input_channels=5)
    path = tmp_path / "dpt.pt"
    torch.save(saved, path)
    return path


def load(path: Path, **changes) -> X3DDPTPredictor:
    settings: dict[str, Any] = dict(device="cpu", image_size=(24, 40), frames=2, input_channels=5,
                    normalize_imagenet=False, precision="fp32", subpixel_refine=True, compile_mode="off")
    settings.update(changes)
    return X3DDPTPredictor.load_from_checkpoint(path, **settings)


def test_strict_restoration_keeps_all_prediction_fields(artifact: Path, tmp_path: Path) -> None:
    first, second = load(artifact), load(artifact)
    rgb = torch.arange(2 * 3 * 24 * 40).remainder(256).to(torch.uint8).reshape(1, 2, 3, 24, 40)
    a = first.predict(rgb)
    recording = tmp_path / "recording.json"
    with InferenceRecorder(recording):
        b = second.predict(rgb)
    events = json.loads(recording.read_text())["events"]
    assert [(event["stage"], event["attributes"]["model"]) for event in events if "model" in event["attributes"]] == [
        ("transfer", "Ball"), ("inference", "Ball"), ("postprocess", "Ball"),
    ]
    assert next(event["attributes"]["batch_shape"] for event in events if event["stage"] == "inference") == [1, 2, 3, 24, 40]
    assert a.heatmaps.shape == (1, 2, 6, 10)
    for field in ("coords", "confidence", "heatmaps"):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    assert a.candidates.coords.device.type == "cpu"
    assert a.candidates.valid.any()
    saved = torch.load(artifact, weights_only=True)
    saved["state_dict"].pop(next(iter(saved["state_dict"])))
    torch.save(saved, artifact)
    with pytest.raises(RuntimeError, match="Missing key"):
        load(artifact)


@pytest.mark.parametrize("changes", [
    dict(image_size=(40, 24)), dict(frames=3), dict(input_channels=2),
    dict(normalize_imagenet=True), dict(precision="fp16"),
])
def test_checkpoint_contract_disagreement_stops(artifact: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        load(artifact, **changes)


@pytest.mark.parametrize("field,value", [("normalization", "imagenet"), ("frames", 3), ("first_mdd_frame", "repeat")])
def test_saved_native_preprocessing_is_not_reinterpreted(artifact: Path, field: str, value: object) -> None:
    saved = torch.load(artifact, weights_only=True)
    saved["input_contract"][field] = value
    torch.save(saved, artifact)
    with pytest.raises(ValueError, match="input contract mismatch"):
        load(artifact)


@pytest.mark.parametrize("shape,dtype", [((1, 2, 3, 24, 40), torch.float32), ((1, 1, 3, 24, 40), torch.uint8),
                                         ((1, 2, 5, 24, 40), torch.uint8), ((1, 2, 3, 40, 24), torch.uint8)])
def test_inference_rejects_bad_rgb_before_forward(artifact: Path, shape: tuple, dtype: torch.dtype) -> None:
    with pytest.raises(ValueError):
        load(artifact).predict(torch.zeros(shape, dtype=dtype))
