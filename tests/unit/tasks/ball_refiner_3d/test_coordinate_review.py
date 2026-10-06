"""Review scientific identity, compatibility, and inference HTTP boundaries."""

import hashlib
import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.config import CorruptionConfig, ModelConfig
from src.tasks.ball_refiner_3d.data import SharedDataset, prepare
from src.tasks.ball_refiner_3d.evaluation import evaluate
from src.tasks.ball_refiner_3d.generation import (
    SCHEMA,
    CameraSampling,
    sample_visible_cameras,
)
from src.tasks.ball_refiner_3d.inference import checkpoint_metadata
from src.tasks.ball_refiner_3d.model import CoordinateRefiner
from src.tasks.ball_refiner_3d.review.artifacts import (
    bundle_path,
    cache_root,
    sha256,
    write_bundle,
)
from src.tasks.ball_refiner_3d.review.contracts import (
    ReviewRequest,
    evaluation_profile,
)
from src.tasks.ball_refiner_3d.review.prepare import prepare_saved_predictions
from src.tasks.ball_refiner_3d.review.service import ReviewService
from src.tasks.ball_refiner_3d.review.web import create_app
from src.tasks.base.visualization.inference_queue import execute_request
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def review(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "single_object"
    (data / "rallies").mkdir(parents=True)
    frames, records = 96, []
    time = np.arange(frames) / 60
    for index, split in enumerate(("train", "val", "test")):
        xyz = np.column_stack((np.sin(time) + index * 0.3, -4 + 5 * time, 0.5 + np.sin(3 * time) ** 2)).astype(np.float32)
        camera_config = CameraSampling(4, 1280, 720, 3, 12, 60, 120, 2, 12, 0.5, 2, 1024)
        cameras = sample_visible_cameras(camera_config, xyz, np.random.default_rng(index + 88))
        assert cameras is not None
        events: np.ndarray = np.zeros(frames, dtype=np.uint8)
        events[[15, 85]], events[45] = 1, 2
        path = data / "rallies" / f"rally_{index:06d}.npz"
        np.savez(path, xyz_m=xyz, uv_px=np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32),
                 visible=np.ones((4, frames), dtype=bool), projection=np.stack([c.matrix for c in cameras]),
                 events=events, time_s=time, camera_centers=np.stack([c.center for c in cameras]),
                 intrinsic=np.stack([c.intrinsic for c in cameras]), rotation=np.stack([c.rotation for c in cameras]))
        records.append({"id": path.stem, "path": str(path.relative_to(data)), "frames": frames, "split": split,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (data / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "image_size_wh": [1280, 720], "fps": 60, "views": 4, "records": records}))
    (data / "state.json").write_text(json.dumps({"status": "complete"}))
    dataset = SharedDataset(data)
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs")):
        raw = OmegaConf.to_container(compose(config_name="train_coordinates"), resolve=True)
    # Keep noisy review coverage as well as the new zero-noise training profile.
    raw["corruption"].update(noise_p95_px=200.0, jitter_sigma_px=3.0, outlier_probability=0.1, isolated_probability=0.04)
    corruption = CorruptionConfig(**raw["corruption"])
    outputs = tmp_path / "outputs"
    for dim, architecture in ((3, "regression"), (3, "flow")):
        model = CoordinateRefiner(ModelConfig(dim, architecture, 16, 1, 2, 0.0, 32, 3, 64, 8, 10000.0, "swiglu")).eval()
        run = outputs / f"{dim}d-{architecture}" / "run"
        checkpoints = run / "logs/version_0/checkpoints"
        checkpoints.mkdir(parents=True)
        config = {**raw, "model": asdict(model.config), "training": {**raw["training"], "gan": {**raw["training"]["gan"], "enabled": False}}}
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        payload = {**checkpoint_metadata(model, event_sigma_frames=2.0), "model": model.state_dict(), "manifest_sha256": dataset.manifest_hash,
                   "fps": 60, "step": 100, "validation_rmse": 0.2 if architecture == "regression" else 0.3, "discriminator": None}
        torch.save(payload, checkpoints / "best.ckpt")
        torch.save({**payload, "validation_rmse": 0.4, "step": 200}, checkpoints / "last.ckpt")
        report, predictions = evaluate(model, prepare(dataset.split("test"), dim, corruption, 20991, event_sigma_frames=2.0), torch.device("cpu"), seed=20991, batch_size=32)
        (run / "predictions").mkdir()
        np.savez(run / "predictions/pred_test.npz", allow_pickle=False, **predictions)
        (run / "predictions/metrics.json").write_text(json.dumps({**report, "best_step": 100}))
        (run / "data_contract.json").write_text(json.dumps({"manifest_sha256": dataset.manifest_hash, "test_ids": ["rally_000002"]}))
        digest, profile = sha256(checkpoints / "best.ckpt"), evaluation_profile(config)
        write_bundle(bundle_path(cache_root(outputs), digest, profile), predictions, checkpoint=checkpoints / "best.ckpt",
                     checkpoint_hash=digest, manifest_hash=dataset.manifest_hash, profile=profile)
    service = ReviewService(data, outputs, tmp_path / "curated")
    catalog = service.catalog()
    request = ReviewRequest(rally="rally_000002", manifest_sha256=catalog["manifest_sha256"],
                            checkpoint_3d=catalog["default_checkpoints"]["3"],
                            **evaluation_profile(raw))
    return service, request


def test_checkpoint_suggestions_are_compatible_and_validation_selected(review):
    service, request = review
    catalog = service.catalog()
    assert len(catalog["checkpoints"]) == 4
    selected = [p for p in catalog["checkpoints"] if p["recommended"]]
    assert {(p["dimensions"], p["method"], p["filename"]) for p in selected} == {(3, "regression", "best.ckpt")}
    torch.save({"legacy": True}, service.outputs_root / "legacy.ckpt")
    excluded = [p for p in service.catalog(refresh=True)["checkpoints"] if not p["compatible"]]
    assert len(excluded) == 1 and "形式" in excluded[0]["reason"]
    with pytest.raises(ValueError, match="カタログ"):
        service.preview(request.model_copy(update={"checkpoint_3d": "../../outside.ckpt"}))


def test_saved_and_real_cpu_inference_share_exact_inputs_and_all_frames(review):
    service, request = review
    saved, live = service.saved(request), service.infer(request)
    assert saved["input_sha256"] == live["input_sha256"]
    for dim in (3,):
        a, b = np.asarray(saved["scene"][f"prediction_{dim}d"]), np.asarray(live["scene"][f"prediction_{dim}d"])
        np.testing.assert_allclose(a, b, atol=1e-5)
        assert a.shape == np.asarray(saved["scene"][f"gt_{dim}d"]).shape
    np.testing.assert_allclose(saved["scene"]["event_probability"], live["scene"]["event_probability"], atol=1e-6)
    assert np.asarray(live["scene"]["event_probability"]).shape == (96,)
    assert np.asarray(live["scene"]["event_target"])[[15, 45, 85]].tolist() == [1., 1., 1.]
    observed = ~np.asarray(live["scene"]["missing_3d"])
    assert not np.allclose(np.asarray(live["scene"]["prediction_3d"])[observed], np.asarray(live["scene"]["input_3d"])[observed])
    assert live["scene"]["metrics_3d"]["all"]["count"] == 96


def test_flow_is_single_seeded_reproducible_trajectory(review):
    service, request = review
    flow = next(item.info["id"] for item in service.checkpoints.entries.values() if item.info["method"] == "flow" and item.info["filename"] == "best.ckpt")
    request = request.model_copy(update={"checkpoint_3d": flow})
    first, second = service.infer(request), service.infer(request)
    np.testing.assert_array_equal(first["scene"]["prediction_3d"], second["scene"]["prediction_3d"])
    other = service.infer(request.model_copy(update={"flow_seed": request.flow_seed + 1}))
    assert first["input_sha256"] == other["input_sha256"]
    assert not np.array_equal(first["scene"]["prediction_3d"], other["scene"]["prediction_3d"])


def test_noise_switch_does_not_move_event_gaps_and_clean_3d_is_triangulated(review):
    service, request = review
    original = service.preview(request)
    no_noise = service.preview(request.model_copy(update={"noise_enabled": False}))
    np.testing.assert_array_equal(original["scene"]["missing_2d"], no_noise["scene"]["missing_2d"])
    assert original["scene"]["intervals"] == no_noise["scene"]["intervals"]
    assert no_noise["scene"]["audit"]["noise_p95_px"] == 0
    clean = service.preview(request.model_copy(update={"noise_enabled": False, "missing_enabled": False}))
    assert not np.asarray(clean["scene"]["missing_2d"]).any()
    assert not np.asarray(clean["scene"]["missing_3d"]).any()
    np.testing.assert_allclose(clean["scene"]["input_3d"], clean["scene"]["gt_3d"], atol=1e-4)
    assert not np.allclose(original["scene"]["input_3d"], clean["scene"]["input_3d"])
    assert service.preview(request)["input_sha256"] == original["input_sha256"]


@pytest.mark.parametrize("change", [{"augmentation_seed": 1}, {"flow_seed": 1}, {"noise_enabled": False}, {"rally": "rally_000000"}])
def test_saved_predictions_reject_different_conditions(review, change):
    service, request = review
    with pytest.raises(ValueError, match="保存済み"):
        service.saved(request.model_copy(update=change))


def test_mutated_dataset_or_checkpoint_cannot_reuse_stale_result(review):
    service, request = review
    checked = ReviewRequest.model_validate(service.saved(request)["request"])
    entry = service.checkpoints.entries[request.checkpoint_3d]
    with entry.path.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint"):
        service.saved(checked)
    with (service.data_root / "rallies/rally_000002.npz").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(RuntimeError, match="ラリー"):
        service.preview(request)


def test_replaced_checkpoint_with_same_step_cannot_adopt_old_predictions_after_refresh(review):
    service, request = review
    before = service.saved(request)
    entry = service.checkpoints.entries[request.checkpoint_3d]
    payload = torch.load(entry.path, weights_only=True, map_location="cpu")
    payload["model"]["output.1.bias"] += 0.5
    payload["validation_rmse"] = 123.0
    torch.save(payload, entry.path)
    service.catalog(refresh=True)
    changed = service.checkpoints.entries[request.checkpoint_3d]
    assert changed.info["step"] == entry.info["step"]
    assert changed.info["sha256"] != entry.info["sha256"]
    assert not changed.info["saved_available"]
    request = request.model_copy(update={"checkpoint_hashes": {"3": changed.info["sha256"]}})
    with pytest.raises(ValueError, match="保存済み"):
        service.saved(request)
    restarted = ReviewService(service.data_root, service.outputs_root, service.checkpoints_root)
    with pytest.raises(ValueError, match="保存済み"):
        restarted.saved(request)
    after = service.infer(request)
    assert np.max(np.abs(np.asarray(after["scene"]["prediction_3d"]) - before["scene"]["prediction_3d"])) > 1


def test_prediction_bytes_and_receipt_are_checked_even_with_warm_array_cache(review):
    service, request = review
    service.saved(request)
    entry = service.checkpoints.entries[request.checkpoint_3d]
    with (entry.predictions / "pred_test.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="内容hash"):
        service.saved(request)
    assert not next(item for item in service.catalog(refresh=True)["checkpoints"] if item["id"] == request.checkpoint_3d)["saved_available"]


def test_legacy_predictions_require_fresh_evaluation_in_separate_review_cache(review):
    service, request = review
    entry = service.checkpoints.entries[request.checkpoint_3d]
    original = {path: sha256(path) for path in entry.run.rglob("*") if path.is_file()}
    shutil.rmtree(entry.predictions)
    service.catalog(refresh=True)
    assert not service.checkpoints.entries[request.checkpoint_3d].info["saved_available"]
    with pytest.raises(ValueError, match="保存済み"):
        service.saved(request)
    prepare_saved_predictions(service)
    saved, live = service.saved(request), service.infer(request)
    np.testing.assert_allclose(saved["scene"]["prediction_3d"], live["scene"]["prediction_3d"], atol=1e-5)
    assert original == {path: sha256(path) for path in entry.run.rglob("*") if path.is_file()}


def test_review_api_rejects_invalid_contract_and_uses_queue_for_cuda(review, monkeypatch):
    from src.tasks.ball_refiner_3d.review import web

    service, request = review
    client = TestClient(create_app(service))
    assert client.get("/").status_code == 200
    assert client.get("/static/no-such.js").status_code == 404
    assert client.get("/shared/scene3d.mjs").status_code == 200
    assert client.post("/api/preview", json=request.model_dump() | {"confidence": [1]}).status_code == 422
    wrong = request.model_dump() | {"checkpoint_2d": "retired"}
    assert client.post("/api/infer", json=wrong).status_code == 422
    assert client.post("/api/saved", json=request.model_dump()).json()["source"] == "saved"
    calls = []
    def queued(task, *, service, request):
        calls.append((task, service, request))
        return b'{"source":"gpu-proof"}'
    monkeypatch.setattr(web, "run_queued_inference", queued)
    result = client.post("/api/infer", json=request.model_dump() | {"device": "cuda"})
    assert result.json()["source"] == "gpu-proof"
    assert calls[0][0] == "ball_refiner_3d" and calls[0][2]["device"] == "cuda"
    assert calls[0][1]["data_root"] == str(service.data_root)
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="queue"):
        service.infer(request.model_copy(update={"device": "cuda"}))


def test_shared_queue_worker_dispatches_coordinate_review_contract(review):
    service, request = review
    result = json.loads(execute_request({"task": "ball_refiner_3d", "service": {
        "data_root": str(service.data_root), "outputs_root": str(service.outputs_root),
        "checkpoints_root": str(service.checkpoints_root)}, "request": request.model_dump()}))
    assert result["source"] == "live" and result["scene"]["rally"] == request.rally
    assert result["scene"]["metrics_3d"]["all"]["count"] == 96


def test_review_accepts_zero_noise_event_only_profile(review):
    service, request = review
    augmentation = request.augmentation.model_copy(update={"noise_p95_px": 0.0, "jitter_sigma_px": 0.0,
                                                           "outlier_probability": 0.0, "isolated_probability": 0.0})
    request = request.model_copy(update={"augmentation": augmentation})
    client = TestClient(create_app(service))
    response = client.post("/api/preview", json=request.model_dump())
    assert response.status_code == 200
    scene = response.json()["scene"]
    assert scene["audit"]["noise_p95_px"] == 0
    assert not np.asarray(scene["isolated_missing"]).any()
    mask = np.asarray(scene["missing_2d"])
    np.testing.assert_array_equal(np.asarray(scene["input_2d"])[~mask], np.asarray(scene["gt_2d"])[~mask])
