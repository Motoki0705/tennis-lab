"""Track-query inference-UI assembly, matching, and CPU mock verification."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io.factory import build_plcs_model_io
from src.tasks.plcs.visualization.inference.service import (
    InferenceService,
    PredictionRequest,
    SceneCatalogError,
)
from src.tasks.plcs.visualization.inference.tracking import (
    TrackingSceneError,
    TrackingWindow,
    build_tracking_batch,
    match_tracks,
    tracking_metric_config,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_DIR = PROJECT_ROOT / "src/tasks/plcs/configs"
# The canonical dataset lives in the primary checkout; worktrees omit it.
_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "plcs",
    Path("/home/kamimura/projects/tennis-lab/data/plcs"),
)
DATA_ROOT = next((path for path in _DATA_CANDIDATES if path.is_dir()), None)
SCENE = "scene_000000"
MULTI_FORMS = (
    "multi_object",
    "multi_object_broadcast",
    "multi_object_camera_view_v2",
)

TINY_TRACKING_OVERRIDES = (
    "model.hidden_dim=16",
    "model.num_heads=2",
    "model.ffn_dim=32",
    "model.rope_dim=6",
    "model.num_stages=4",
    "model.dropout=0.0",
    "model.mhc.coefficient_dim=8",
    "model.mhc.sinkhorn_iters=3",
    "model.cswa.compression_ratio=2",
    "model.cswa.window_radius=1",
    "model.cswa.backend=reference",
    "model.num_queries=4",
)

_CASE_ARGS: dict[str, tuple[str, tuple[int, ...], str | None]] = {
    "multi_object": ("data=tracking", (0, 1, 2), None),
    "multi_object_broadcast": ("data=tracking_broadcast", (0, 1), None),
    "multi_object_camera_view_v2": (
        "data=tracking_camera_view_v2",
        (0, 1, 2),
        "camera_1",
    ),
}


# --------------------------------------------------------------- pure units


def test_match_tracks_pairs_objects_and_measures_errors() -> None:
    gt_position = np.asarray([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=np.float64)
    pred_position = np.asarray([[[5.1, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=np.float64)
    # Every matched pair differs by 90 degrees so the yaw error is unambiguous.
    gt_rotation = np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float64)
    pred_rotation = np.asarray([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float64)
    match = match_tracks(
        pred_position_m=pred_position,
        pred_present=np.ones((1, 2), dtype=bool),
        pred_rotation=pred_rotation,
        gt_position_m=gt_position,
        gt_present=np.ones((1, 2), dtype=bool),
        gt_rotation=gt_rotation,
    )
    assert match.matched_frames == 1
    assert match.matched_pairs == 2
    assert match.unmatched_prediction == 0
    assert match.unmatched_ground_truth == 0
    # The nearest assignment is chosen, so error stays at the small offsets.
    assert match.position_error_m.max() == pytest.approx(0.2, abs=1e-6)
    assert match.yaw_error_deg.max() == pytest.approx(90.0, abs=1e-4)


def test_match_tracks_reports_surplus_tracks_as_unmatched() -> None:
    match = match_tracks(
        pred_position_m=np.zeros((2, 1, 3), dtype=np.float64),
        pred_present=np.ones((2, 1), dtype=bool),
        pred_rotation=None,
        gt_position_m=np.zeros((2, 3, 3), dtype=np.float64),
        gt_present=np.ones((2, 3), dtype=bool),
        gt_rotation=None,
    )
    assert match.matched_pairs == 2
    assert match.unmatched_ground_truth == 4
    assert match.unmatched_prediction == 0
    assert match.yaw_error_deg.size == 0


def test_match_tracks_gate_rejects_far_pairs() -> None:
    match = match_tracks(
        pred_position_m=np.zeros((1, 1, 3), dtype=np.float64),
        pred_present=np.ones((1, 1), dtype=bool),
        pred_rotation=None,
        gt_position_m=np.full((1, 1, 3), 100.0),
        gt_present=np.ones((1, 1), dtype=bool),
        gt_rotation=None,
        max_distance_m=1.0,
    )
    assert match.matched_pairs == 0
    assert match.unmatched_prediction == 1
    assert match.unmatched_ground_truth == 1


def test_match_tracks_rejects_shape_mismatch() -> None:
    with pytest.raises(TrackingSceneError, match="time axis"):
        match_tracks(
            pred_position_m=np.zeros((2, 1, 3), dtype=np.float64),
            pred_present=np.ones((2, 1), dtype=bool),
            pred_rotation=None,
            gt_position_m=np.zeros((3, 1, 3), dtype=np.float64),
            gt_present=np.ones((3, 1), dtype=bool),
            gt_rotation=None,
        )
    with pytest.raises(TrackingSceneError, match=r"\(T, N, 3\)"):
        match_tracks(
            pred_position_m=np.zeros((2, 3), dtype=np.float64),
            pred_present=np.ones((2, 1), dtype=bool),
            pred_rotation=None,
            gt_position_m=np.zeros((2, 1, 3), dtype=np.float64),
            gt_present=np.ones((2, 1), dtype=bool),
            gt_rotation=None,
        )
    with pytest.raises(TrackingSceneError, match="rotation"):
        match_tracks(
            pred_position_m=np.zeros((2, 1, 3), dtype=np.float64),
            pred_present=np.ones((2, 1), dtype=bool),
            pred_rotation=np.zeros((2, 1, 3), dtype=np.float64),
            gt_position_m=np.zeros((2, 1, 3), dtype=np.float64),
            gt_present=np.ones((2, 1), dtype=bool),
            gt_rotation=None,
        )


def test_tracking_metric_config_requires_a_block() -> None:
    with pytest.raises(TrackingSceneError, match="tracking_metrics"):
        tracking_metric_config({})
    resolved = tracking_metric_config(
        {
            "tracking_metrics": {
                "presence_threshold": 0.4,
                "duplicate_distance": 0.05,
                "id_switch_distance": 0.05,
            }
        }
    )
    assert resolved.presence_threshold == pytest.approx(0.4)


# ---------------------------------------------------- real-data assemblies


@lru_cache(maxsize=4)
def _plain_config(form: str) -> dict[str, Any]:
    data_override, _, _ = _CASE_ARGS[form]
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        composed = compose(
            config_name="train_tracking",
            overrides=[data_override, *TINY_TRACKING_OVERRIDES],
        )
    return cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True))


@lru_cache(maxsize=4)
def _tiny_predictor(form: str) -> PLCSTrackingPredictor:
    runtime = PLCSTrainingConfig.from_config(OmegaConf.create(_plain_config(form)))
    bound = build_plcs_model_io(runtime)
    return PLCSTrackingPredictor(
        model=bound.model,
        adapter=bound.adapter,
        device=torch.device("cpu"),
    )


def _service_with_mock_predictor(tmp_path: Path, form: str) -> InferenceService:
    checkpoint_root = tmp_path / "outputs" / "plcs"
    log_dir = checkpoint_root / form / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    (log_dir / "hparams.yaml").write_text(
        yaml.safe_dump({"config": _plain_config(form)}), encoding="utf-8"
    )
    (log_dir / "checkpoints" / "best.ckpt").write_bytes(b"")
    service = InferenceService(
        data_root=cast("Path", DATA_ROOT),
        checkpoint_root=checkpoint_root,
        device="cpu",
        project_root=PROJECT_ROOT,
    )
    predictor = _tiny_predictor(form)
    service._tracking_predictor = lambda *args, **kwargs: predictor  # type: ignore[method-assign]
    return service


def _tracking_request(form: str) -> PredictionRequest:
    _, cameras, reference = _CASE_ARGS[form]
    return PredictionRequest(
        checkpoint=f"{form}/logs/version_0/checkpoints/best.ckpt",
        family=form,
        scene=SCENE,
        cameras=cameras,
        reference_camera_id=reference,
        window_start=0,
        window_length=8,
        canonical_pose_source="gt",
        device="cpu",
    )


@pytest.mark.skipif(DATA_ROOT is None, reason="PLCS dataset is unavailable")
@pytest.mark.parametrize("form", MULTI_FORMS)
def test_tracking_batch_builds_from_real_scene(form: str) -> None:
    _, cameras, reference = _CASE_ARGS[form]
    batch = build_tracking_batch(
        config=_plain_config(form),
        window=TrackingWindow(
            family_dir=cast("Path", DATA_ROOT) / form,
            scene_id=SCENE,
            cameras=cameras,
            start=0,
            length=8,
            reference_camera_id=reference,
        ),
    )
    human_kp = cast("torch.Tensor", batch["human_kp"])
    assert human_kp.shape == (1, len(cameras), 8, 4, 17, 2)
    assert cast("torch.Tensor", batch["court_kp"]).shape == (
        1,
        len(cameras),
        8,
        14,
        2,
    )
    instance_id = cast("torch.Tensor", batch["target_instance_id"])
    assert instance_id.dtype == torch.long
    assert int(instance_id.min().item()) >= -1
    if reference is not None:
        assert "reference_view_selection" in batch
        assert "reference_from_physical" in batch


@pytest.mark.skipif(DATA_ROOT is None, reason="PLCS dataset is unavailable")
@pytest.mark.parametrize("form", MULTI_FORMS)
def test_tracking_predict_end_to_end_on_cpu(tmp_path: Path, form: str) -> None:
    service = _service_with_mock_predictor(tmp_path, form)
    request = _tracking_request(form)
    # The queue preflight must accept the same request without touching the GPU.
    resolved = service.validate_prediction_request(request)
    assert resolved["mode"] == "tracking"

    result = service.predict(request)
    header = result.header
    assert header["mode"] == "tracking"
    assert header["metrics"]["scope"] == "multi_object_tracks"
    assert header["metrics"]["matching"] == "hungarian_per_frame"
    assert header["metrics"]["position_error_m"]["mean"] is not None
    assert cast("Mapping[str, Any]", header["scene"])["num_persons"] == 9

    gt_tracks = [track for track in header["tracks"] if track["kind"] == "gt"]
    pred_tracks = [track for track in header["tracks"] if track["kind"] == "pred"]
    assert len(gt_tracks) == 10
    assert len(pred_tracks) == 4
    for track in gt_tracks:
        assert track["has_joints"] is True
        assert track["position"]["shape"] == [8, 3]
        assert track["joints"]["shape"] == [8, 17, 3]
        assert track["presence"]["shape"] == [8]
        assert track["rotation"]["shape"] == [8, 2]
    for track in pred_tracks:
        # Query slots are reused over time, so no per-object skeleton is drawn.
        assert track["has_joints"] is False
        assert track["joints"] is None
        assert track["presence"]["shape"] == [8]
        assert track["rotation"]["shape"] == [8, 2]
    assert any("予測骨格" in warning for warning in header["warnings"])
    assert result.payload.dtype == np.float32
    assert header["payload_elements"] == result.payload.size


@pytest.mark.skipif(DATA_ROOT is None, reason="PLCS dataset is unavailable")
def test_tracking_rejects_out_of_range_window(tmp_path: Path) -> None:
    service = _service_with_mock_predictor(tmp_path, "multi_object")
    request = _tracking_request("multi_object")
    with pytest.raises(SceneCatalogError, match="num_frames"):
        service.validate_prediction_request(
            PredictionRequest(
                checkpoint=request.checkpoint,
                family=request.family,
                scene=request.scene,
                cameras=request.cameras,
                reference_camera_id=None,
                window_start=100_000,
                window_length=8,
                canonical_pose_source="gt",
                device="cpu",
            )
        )
