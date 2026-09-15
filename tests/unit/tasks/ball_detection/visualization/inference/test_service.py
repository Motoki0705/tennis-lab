"""Unit tests for the ball-detection review/inference service backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.ball_detection.visualization.inference.loader import load_ball_model
from src.tasks.ball_detection.visualization.inference.peaks import (
    decode_frame_peaks,
    peaks_to_points,
)
from src.tasks.ball_detection.visualization.inference.service import (
    PREVIEW_FRAME_LIMIT,
    DetectionRequestError,
    DetectionService,
)
from src.tasks.base.visualization.detection.web import create_detection_app


def build_service(
    tmp_path: Path,
    *,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
    make_web_store: Callable[[Path], Path] | None = None,
    clips: int = 2,
    frames: int = 6,
    num_frames: int = 2,
) -> DetectionService:
    data_root = tmp_path / "data"
    make_clip_dataset(data_root / "tennis" / "tracknet", clips=clips, frames=frames)
    if make_web_store is not None:
        make_web_store(data_root)
    make_tiny_checkpoint(
        tmp_path / "outputs" / "ball_detection" / "run-local.ckpt", num_frames=num_frames
    )
    return DetectionService(
        tmp_path,
        data_root=data_root,
        outputs_root=tmp_path / "outputs" / "ball_detection",
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )


def test_catalog_reports_sources_and_compatibility(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    catalog = service.catalog()
    assert catalog["task"] == "ball_detection"
    assert catalog["title"] == "Ball Detection"
    datasets = {entry["id"]: entry for entry in catalog["datasets"]}
    assert datasets["tracknet"]["available"] is True
    assert datasets["tracknet"]["count"] == 2
    assert datasets["tracknet"]["mode"] == "temporal"
    assert datasets["web_static"]["available"] is False
    assert "reason" in datasets["web_static"]
    assert any("web_static" in warning for warning in catalog["warnings"])

    checkpoint = catalog["checkpoints"][0]
    assert checkpoint["id"] == "run-local.ckpt"
    assert checkpoint["model"] == "conv_next_unet"
    assert checkpoint["settings"] == {"count": 2, "threshold": 0.5}
    assert checkpoint["window"] == {"min": 2, "max": 2}
    assert checkpoint["compatible_datasets"] == ["tracknet"]


def test_catalog_reports_unusable_checkpoint(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
) -> None:
    data_root = tmp_path / "data"
    make_clip_dataset(data_root / "tennis" / "tracknet")
    broken = tmp_path / "outputs" / "ball_detection" / "broken.ckpt"
    broken.parent.mkdir(parents=True)
    broken.write_text("not a checkpoint", encoding="utf-8")
    service = DetectionService(
        tmp_path,
        data_root=data_root,
        outputs_root=broken.parent,
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )
    catalog = service.catalog()
    entry = catalog["checkpoints"][0]
    assert entry["id"] == "broken.ckpt"
    assert entry["compatible_datasets"] == []
    assert "error" in entry
    assert any("broken.ckpt" in warning for warning in catalog["warnings"])
    with pytest.raises(DetectionRequestError, match="unusable"):
        service.validate("broken.ckpt", "tracknet::game1/Clip1", device="cpu")


def test_scenes_paging_search_and_checkpoint_filter(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    page = service.scenes("tracknet", limit=1)
    assert page["total"] == 2
    assert [item["id"] for item in page["items"]] == ["tracknet::game1/Clip1"]
    second = service.scenes("tracknet", offset=1, limit=1)
    assert [item["id"] for item in second["items"]] == ["tracknet::game1/Clip2"]
    assert service.scenes("tracknet", search="clip2")["total"] == 1
    assert service.scenes("tracknet", search="nope")["total"] == 0
    assert service.scenes("tracknet", checkpoint="run-local.ckpt")["total"] == 2

    with pytest.raises(DetectionRequestError, match="Unknown dataset"):
        service.scenes("missing")
    with pytest.raises(DetectionRequestError, match="cannot run on dataset"):
        service.scenes("web_static", checkpoint="run-local.ckpt")
    with pytest.raises(DetectionRequestError, match="offset"):
        service.scenes("tracknet", offset=-1)
    with pytest.raises(DetectionRequestError, match="limit"):
        service.scenes("tracknet", limit=0)


def test_preview_returns_original_pixel_labels_without_rasters(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    preview = service.preview("tracknet::game1/Clip1", start=1, count=2)
    assert preview["scene"] == "tracknet::game1/Clip1"
    assert preview["label"] == "game1/Clip1"
    assert preview["frames"] == 6
    assert (preview["width"], preview["height"]) == (64, 48)
    assert [item["index"] for item in preview["items"]] == [1, 2]
    assert [item["name"] for item in preview["items"]] == ["0001.jpg", "0002.jpg"]
    point = preview["items"][0]["gt"]["points"][0]
    assert (point["x"], point["y"]) == (11.0, 21.0)
    assert point["label"] == "b001"
    assert point["visible"] is True
    assert preview["items"][0]["gt"]["rasters"] == []
    assert preview["warnings"] == []


def test_preview_rejects_bad_ranges_and_unknown_scene(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    scene = "tracknet::game1/Clip1"
    with pytest.raises(DetectionRequestError, match="exceeds"):
        service.preview(scene, start=5, count=2)
    with pytest.raises(DetectionRequestError, match="count must be within"):
        service.preview(scene, start=0, count=PREVIEW_FRAME_LIMIT + 1)
    with pytest.raises(DetectionRequestError, match="count must be within"):
        service.preview(scene, start=0, count=0)
    with pytest.raises(DetectionRequestError, match="non-negative"):
        service.preview(scene, start=-1, count=1)
    with pytest.raises(DetectionRequestError, match="Unknown scene"):
        service.preview("tracknet::game1/Clip999", start=0, count=1)
    with pytest.raises(DetectionRequestError, match="must look like"):
        service.preview("no-separator", start=0, count=1)


def test_jpeg_image_is_served_in_original_resolution(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    payload = service.image("tracknet::game1/Clip1", 2)
    assert payload[:2] == b"\xff\xd8"
    decoded = np.frombuffer(payload, dtype=np.uint8)
    import cv2

    frame = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape[:2] == (48, 64)
    with pytest.raises(DetectionRequestError, match="out of range"):
        service.image("tracknet::game1/Clip1", 6)


def test_validate_enforces_window_contract(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    scene = "tracknet::game1/Clip1"
    service.validate("run-local.ckpt", scene, start=4, count=2, device="cpu")
    with pytest.raises(DetectionRequestError, match="Unknown checkpoint"):
        service.validate("missing.ckpt", scene, device="cpu")
    with pytest.raises(DetectionRequestError, match="exceeds the scene"):
        service.validate("run-local.ckpt", scene, start=5, count=2, device="cpu")
    with pytest.raises(DetectionRequestError, match="temporal window"):
        service.validate("run-local.ckpt", scene, start=0, count=3, device="cpu")
    with pytest.raises(DetectionRequestError, match="temporal window"):
        service.validate("run-local.ckpt", scene, start=0, count=1, device="cpu")
    with pytest.raises(DetectionRequestError, match="threshold"):
        service.validate("run-local.ckpt", scene, count=2, threshold=1.5, device="cpu")
    if torch.cuda.is_available():
        # An explicit CUDA request keeps meaning CUDA; it never degrades to CPU.
        assert DetectionService._resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(DetectionRequestError, match="CUDA is unavailable"):
            service.validate("run-local.ckpt", scene, count=2, device="cuda")


def test_cuda_request_without_cuda_is_rejected(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    monkeypatch.setattr(
        "src.utils.device.torch.cuda.is_available", lambda: False
    )
    with pytest.raises(DetectionRequestError, match="CUDA is unavailable"):
        service.validate(
            "run-local.ckpt", "tracknet::game1/Clip1", count=2, device="cuda"
        )


def test_infer_runs_real_tiny_model_and_returns_scaled_points(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    result = service.infer("run-local.ckpt", "tracknet::game1/Clip1", start=1, count=2, device="cpu")
    assert result["scene"] == "tracknet::game1/Clip1"
    assert result["start"] == 1
    assert [item["index"] for item in result["items"]] == [1, 2]
    for item in result["items"]:
        pred = item["pred"]
        assert len(pred["rasters"]) == 1
        raster = pred["rasters"][0]
        assert raster["name"] == "probability"
        assert raster["data"].startswith("data:image/png;base64,")
        for point in pred["points"]:
            assert 0.0 <= point["x"] <= 63.0
            assert 0.0 <= point["y"] <= 47.0
            assert point["visible"] is True
            assert 0.0 <= point["score"] <= 1.0
    metrics = result["metrics"]
    assert metrics["window"] == {
        "mode": "temporal",
        "start": 1,
        "end": 2,
        "count": 2,
        "repeat": 1,
        "checkpoint_frames": 2,
    }
    assert 0.0 <= metrics["f1"] <= 1.0
    assert result["warnings"] == []


def test_infer_static_scene_repeats_frame_and_labels_it(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
    make_web_store: Callable[[Path], Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
        make_web_store=make_web_store,
    )
    assert service.catalog()["checkpoints"][0]["compatible_datasets"] == [
        "tracknet",
        "web_static",
        "web_temporal",
    ]
    scene = "web_static::0"
    preview = service.preview(scene, start=0, count=1)
    assert preview["frames"] == 1
    assert preview["items"][0]["gt"]["points"][0]["x"] == 12.5

    result = service.infer("run-local.ckpt", scene, start=0, count=2, device="cpu")
    assert result["metrics"]["window"] == {
        "mode": "static_repeat",
        "start": 0,
        "end": 0,
        "count": 2,
        "repeat": 2,
        "checkpoint_frames": 2,
    }
    assert any("canonical" in warning for warning in result["warnings"])
    # Repeating one frame is the store's own static sampling mode; asking for a
    # shorter window is refused instead of being padded quietly.
    with pytest.raises(DetectionRequestError, match="count must"):
        service.validate("run-local.ckpt", scene, start=0, count=1, device="cpu")


def test_infer_uses_checkpoint_image_size_and_accepts_tiny_windows(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model window is built at the checkpoint's saved input size."""
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    seen: dict[str, tuple[int, ...]] = {}
    original = load_ball_model

    def spy(path: Path, *, device: str | torch.device) -> object:
        loaded = original(path, device=device)
        seen["size"] = tuple(loaded.image_size_hw or ())
        return loaded

    monkeypatch.setattr(
        "src.tasks.ball_detection.visualization.inference.service.load_ball_model",
        spy,
    )
    result = service.infer("run-local.ckpt", "tracknet::game1/Clip1", start=0, count=2, device="cpu")
    assert seen["size"] == (64, 128)
    assert len(result["items"]) == 2


def test_infer_reports_missing_ground_truth_without_inventing_points(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    result = service.infer("run-local.ckpt", "tracknet::game1/Clip1", start=0, count=2, device="cpu")
    # GT lives in the preview payload only; inference items carry predictions.
    assert set(result["items"][0]) == {"index", "pred"}
    assert set(result["items"][0]["pred"]) == {"points", "rasters"}


def _client(
    service: DetectionService, *, mode: str
) -> TestClient:
    return TestClient(
        create_detection_app(
            service,
            task="ball_detection",
            mode=mode,  # type: ignore[arg-type]
            service_config={"project_root": str(service.project_root)},
        )
    )


def test_shared_app_serves_catalog_preview_image_and_cpu_inference(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    client = _client(service, mode="inference")
    catalog = client.get("/api/catalog").json()
    assert catalog["task"] == "ball_detection"
    assert catalog["mode"] == "inference"

    scenes = client.get("/api/scenes", params={"dataset": "tracknet"}).json()
    assert scenes["total"] == 2
    scene = scenes["items"][0]["id"]
    preview = client.get("/api/preview", params={"scene": scene, "count": 2}).json()
    assert preview["items"][0]["gt"]["points"][0]["x"] == 10.0
    image = client.get("/api/image", params={"scene": scene, "frame": 0})
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"

    response = client.post(
        "/api/infer",
        json={
            "checkpoint": "run-local.ckpt",
            "scene": scene,
            "start": 0,
            "count": 2,
            "threshold": 0.5,
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["window"]["count"] == 2
    assert len(payload["items"]) == 2


def test_shared_app_rejects_invalid_request_before_inference(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
    )
    client = _client(service, mode="inference")
    response = client.post(
        "/api/infer",
        json={
            "checkpoint": "run-local.ckpt",
            "scene": "tracknet::game1/Clip1",
            "start": 0,
            "count": 3,
            "device": "cpu",
        },
    )
    assert response.status_code == 422
    assert "temporal window" in response.json()["detail"]

    review_client = _client(service, mode="review")
    assert (
        review_client.post(
            "/api/infer",
            json={"checkpoint": "run-local.ckpt", "scene": "tracknet::game1/Clip1"},
        ).status_code
        == 403
    )


# ------------------------------------------------ 4) per-scene window filtering


def test_scenes_filter_clips_shorter_than_the_checkpoint_window(
    tmp_path: Path,
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    """A short clip is dropped even though its dataset is compatible overall."""
    from tests.unit.tasks.ball_detection.visualization.conftest import write_clip

    data_root = tmp_path / "data"
    game = data_root / "tennis" / "tracknet" / "game1"
    # Clip1 holds a single frame, Clip2 holds three; the checkpoint needs two.
    for name, frames in (("Clip1", 1), ("Clip2", 3)):
        write_clip(
            game / name,
            frames=frames,
            rows=[
                {
                    "file name": f"{index:04d}.jpg",
                    "instance id": "b001",
                    "visibility": 1,
                    "x-coordinate": 10.0,
                    "y-coordinate": 20.0,
                }
                for index in range(frames)
            ],
        )
    make_tiny_checkpoint(
        tmp_path / "outputs" / "ball_detection" / "run-local.ckpt", num_frames=2
    )
    service = DetectionService(
        tmp_path,
        data_root=data_root,
        outputs_root=tmp_path / "outputs" / "ball_detection",
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )

    unfiltered = service.scenes("tracknet", limit=10)
    assert unfiltered["total"] == 2
    assert sorted(item["frames"] for item in unfiltered["items"]) == [1, 3]

    filtered = service.scenes("tracknet", limit=10, checkpoint="run-local.ckpt")
    assert filtered["total"] == 1
    assert filtered["items"][0]["frames"] == 3
    assert filtered["items"][0]["id"] == "tracknet::game1/Clip2"

    # The dataset stays compatible overall, and the search interacts with the
    # filter instead of bypassing it.
    catalog = service.catalog()
    assert catalog["checkpoints"][0]["compatible_datasets"] == ["tracknet"]
    assert (
        service.scenes("tracknet", search="Clip1", checkpoint="run-local.ckpt")["total"]
        == 0
    )


# ------------------------------------------------- 5) missing vs negative labels


def _write_partially_labelled_clip(tmp_path: Path, *, frames: int = 5) -> None:
    from tests.unit.tasks.ball_detection.visualization.conftest import write_clip

    rows = [
        {
            "file name": f"{index:04d}.jpg",
            "instance id": "b001",
            "visibility": 1,
            "x-coordinate": 10.0 + index,
            "y-coordinate": 20.0 + index,
        }
        for index in (0, 1, 2)
    ]
    write_clip(
        tmp_path / "data" / "tennis" / "tracknet" / "game1" / "Clip1",
        frames=frames,
        rows=rows,
    )


def _partial_service(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> DetectionService:
    _write_partially_labelled_clip(tmp_path)
    make_tiny_checkpoint(
        tmp_path / "outputs" / "ball_detection" / "run-local.ckpt", num_frames=2
    )
    return DetectionService(
        tmp_path,
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs" / "ball_detection",
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )


def test_missing_annotation_rows_are_warned_and_not_treated_as_negatives(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    service = _partial_service(tmp_path, make_tiny_checkpoint)
    scene = "tracknet::game1/Clip1"
    preview = service.preview(scene, start=0, count=5)
    assert [item["annotated"] for item in preview["items"]] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert preview["items"][3]["gt"]["points"] == []
    assert any("frame 3 has no annotation row" in text for text in preview["warnings"])
    assert any("not an annotated negative" in text for text in preview["warnings"])

    # A fully annotated window is scored normally.
    scored = service.infer("run-local.ckpt", scene, start=0, count=2, device="cpu")
    assert scored["metrics"]["available"] is True
    assert scored["metrics"]["scored_frames"] == [0, 1]
    assert scored["metrics"]["excluded_frames"] == []

    # A boundary window keeps the annotated frame and excludes the missing one.
    boundary = service.infer("run-local.ckpt", scene, start=2, count=2, device="cpu")
    assert boundary["metrics"]["scored_frames"] == [2]
    assert boundary["metrics"]["excluded_frames"] == [3]
    assert [item["index"] for item in boundary["items"]] == [2, 3]
    assert boundary["metrics"]["available"] is True
    assert any("frame 3 has no annotation row" in text for text in boundary["warnings"])

    # A window with no annotation at all reports the metric as unavailable.
    unlabelled = service.infer(
        "run-local.ckpt", scene, start=3, count=2, device="cpu"
    )
    assert unlabelled["metrics"]["available"] is False
    assert "cannot be computed" in unlabelled["metrics"]["reason"]
    assert unlabelled["metrics"]["scored_frames"] == []
    assert len(unlabelled["items"]) == 2
    assert all("no annotation row" in text for text in unlabelled["warnings"])


def test_non_finite_label_is_rejected_instead_of_emitting_nan(
    tmp_path: Path,
) -> None:
    from tests.unit.tasks.ball_detection.visualization.conftest import write_clip

    write_clip(
        tmp_path / "data" / "tennis" / "tracknet" / "game1" / "Clip1",
        frames=2,
        rows=[
            {
                "file name": "0000.jpg",
                "instance id": "b001",
                "visibility": 1,
                "x-coordinate": float("nan"),
                "y-coordinate": 20.0,
            },
            {
                "file name": "0001.jpg",
                "instance id": "b001",
                "visibility": 1,
                "x-coordinate": 11.0,
                "y-coordinate": float("inf"),
            },
        ],
    )
    service = DetectionService(
        tmp_path,
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs" / "ball_detection",
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )
    with pytest.raises(DetectionRequestError, match="non-finite"):
        service.preview("tracknet::game1/Clip1", start=0, count=2)


# ------------------------------------------------------ 6) static aggregation


def test_static_repeat_aggregates_to_one_unique_frame(
    tmp_path: Path,
    make_clip_dataset: Callable[..., Path],
    make_tiny_checkpoint: Callable[..., Path],
    make_web_store: Callable[[Path], Path],
) -> None:
    service = build_service(
        tmp_path,
        make_clip_dataset=make_clip_dataset,
        make_tiny_checkpoint=make_tiny_checkpoint,
        make_web_store=make_web_store,
    )
    result = service.infer("run-local.ckpt", "web_static::0", count=2, device="cpu")
    assert [item["index"] for item in result["items"]] == [0]
    assert result["metrics"]["scored_frames"] == [0]
    assert result["metrics"]["excluded_frames"] == []
    # The window metadata still records the repetition the model consumed.
    assert result["metrics"]["window"]["repeat"] == 2
    assert result["metrics"]["window"]["count"] == 2
    assert any("Averaged" in text for text in result["warnings"])

    # The single prediction is the decode of the window's mean probability map,
    # recomputed here from the same public loader and peak decoder.
    import cv2

    from src.tasks.ball_detection.visualization.review.datasets import (
        BallDatasetCatalog,
    )

    info = service.checkpoints()["run-local.ckpt"]
    loaded = load_ball_model(info.path, device="cpu")
    assert loaded.image_size_hw is not None
    height, width = loaded.image_size_hw
    scene = BallDatasetCatalog(service.data_root).resolve("web_static", "0")
    rgb = scene.read_rgb(0)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    frame = resized.astype(np.float32) / 255.0
    window = (
        torch.from_numpy(np.stack([frame, frame]))
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .contiguous()
    )
    with torch.no_grad():
        call = loaded.adapter.prepare_model_call(window)
        probability = loaded.adapter.probability_heatmaps(
            loaded.model(*call.model_args), call
        )[0]
    mean_heatmap = probability.mean(dim=0, keepdim=True)
    peaks = decode_frame_peaks(
        mean_heatmap,
        original_size=(64, 48),
        threshold=0.5,
        nms_kernel=info.metrics.nms_kernel,
        max_peaks=info.metrics.max_predictions_per_frame,
        subpixel_refine=info.metrics.subpixel_refine,
    )
    expected = peaks_to_points(peaks[0])
    produced = result["items"][0]["pred"]["points"]
    assert len(produced) == len(expected)
    for actual, wanted in zip(produced, expected, strict=True):
        assert actual["label"] == wanted["label"]
        assert actual["x"] == pytest.approx(wanted["x"], abs=1e-4)
        assert actual["y"] == pytest.approx(wanted["y"], abs=1e-4)
        assert actual["score"] == pytest.approx(wanted["score"], abs=1e-6)
