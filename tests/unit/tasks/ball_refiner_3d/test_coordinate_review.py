"""Review scientific identity, compatibility, and inference HTTP boundaries."""

import hashlib
import json
import shutil
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.configuration.generation import CameraSampling
from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.data.preprocessing import prepare
from src.tasks.ball_refiner_3d.data.schema import SCHEMA
from src.tasks.ball_refiner_3d.evaluation.evaluator import evaluate
from src.tasks.ball_refiner_3d.generate_dataset.cameras import sample_visible_cameras
from src.tasks.ball_refiner_3d.model_io.checkpoint import checkpoint_metadata
from src.tasks.ball_refiner_3d.model_io.factory import build_refiner
from src.tasks.ball_refiner_3d.physics.targets import FlightClock
from src.tasks.ball_refiner_3d.visualization.dataset_review.artifacts import (
    bundle_path,
    cache_root,
    sha256,
    write_bundle,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.contracts import (
    ReviewRequest,
    evaluation_profile,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.prepare import (
    prepare_saved_predictions,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.service import ReviewService
from src.tasks.ball_refiner_3d.visualization.dataset_review.web import create_app
from src.tasks.base.visualization.inference_queue import execute_request
from src.utils.paths import PROJECT_ROOT
from tests.support.physics.ball_record import simulated_record


@pytest.fixture
def review(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "single_object"
    (data / "rallies").mkdir(parents=True)
    frames, records = 96, []
    time = np.arange(frames) / 60
    for index, split in enumerate(("train", "val", "test")):
        xyz = np.column_stack(
            (np.sin(time) + index * 0.3, -4 + 5 * time, 0.5 + np.sin(3 * time) ** 2)
        ).astype(np.float32)
        camera_config = CameraSampling(
            4, 1280, 720, 3, 12, 60, 120, 2, 12, 0.5, 2, 1024
        )
        cameras = sample_visible_cameras(
            camera_config, xyz, np.random.default_rng(index + 88)
        )
        assert cameras is not None
        _, record = simulated_record(
            frames,
            output_fps=60,
            sim_fps=240,
            hits=(15 * 4, 45 * 4, 85 * 4),
            hit_kinds=("shot", "bounce", "shot"),
        )
        path = data / "rallies" / f"rally_{index:06d}.npz"
        np.savez(
            path,
            xyz_m=xyz,
            uv_px=np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32),
            visible=np.ones((4, frames), dtype=bool),
            projection=np.stack([c.matrix for c in cameras]),
            time_s=time,
            camera_centers=np.stack([c.center for c in cameras]),
            intrinsic=np.stack([c.intrinsic for c in cameras]),
            rotation=np.stack([c.rotation for c in cameras]),
            **cast("dict[str, Any]", record.to_arrays()),
        )
        records.append(
            {
                "id": path.stem,
                "path": str(path.relative_to(data)),
                "frames": frames,
                "split": split,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "image_size_wh": [1280, 720],
                "fps": 60,
                "views": 4,
                "records": records,
            }
        )
    )
    (data / "state.json").write_text(json.dumps({"status": "complete"}))
    dataset = SharedDataset(data)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs"),
    ):
        raw = OmegaConf.to_container(compose(config_name="train"), resolve=True)
    # Keep noisy review coverage as well as the new zero-noise training profile.
    raw["augmentation"].update(
        noise_p95_px=200.0,
        jitter_sigma_px=3.0,
        outlier_probability=0.1,
        isolated_probability=0.04,
    )
    corruption = CorruptionConfig(**raw["augmentation"])
    outputs = tmp_path / "outputs"
    for dim, architecture, physics in (
        (3, "regression", False),
        (3, "flow", False),
        (3, "regression", True),
    ):
        torch.manual_seed(len(architecture) + physics)
        model = build_refiner(
            ModelConfig(
                dim, architecture, 16, 1, 2, 0.0, 3, 64, 8, 10000.0, "swiglu", physics
            )
        ).eval()
        name = f"{dim}d-{architecture}" + ("-physics" if physics else "")
        run = outputs / name / "run"
        checkpoints = run / "logs/version_0/checkpoints"
        checkpoints.mkdir(parents=True)
        config = {
            **raw,
            "model": asdict(model.config),
            "training": {
                **raw["training"],
                "gan": {**raw["training"]["gan"], "enabled": False},
            },
        }
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        payload = {
            **checkpoint_metadata(
                model, event_sigma_frames=2.0, clock=FlightClock(9.8, 1 / 240, 4)
            ),
            "model": model.state_dict(),
            "manifest_sha256": dataset.manifest_hash,
            "fps": 60,
            "step": 100,
            "validation_rmse": 0.25
            if physics
            else 0.2
            if architecture == "regression"
            else 0.3,
            "discriminator": None,
        }
        torch.save(payload, checkpoints / "best.ckpt")
        torch.save(
            {**payload, "validation_rmse": 0.4, "step": 200}, checkpoints / "last.ckpt"
        )
        report, predictions = evaluate(
            model,
            prepare(
                dataset.split("test"), dim, corruption, 20991, event_sigma_frames=2.0
            ),
            torch.device("cpu"),
            seed=20991,
            batch_size=32,
            physics=False,
        )
        (run / "predictions").mkdir()
        np.savez(run / "predictions/pred_test.npz", allow_pickle=False, **predictions)
        (run / "predictions/metrics.json").write_text(
            json.dumps({**report, "best_step": 100})
        )
        (run / "data_contract.json").write_text(
            json.dumps(
                {"manifest_sha256": dataset.manifest_hash, "test_ids": ["rally_000002"]}
            )
        )
        digest, profile = sha256(checkpoints / "best.ckpt"), evaluation_profile(config)
        write_bundle(
            bundle_path(cache_root(outputs), digest, profile),
            predictions,
            checkpoint=checkpoints / "best.ckpt",
            checkpoint_hash=digest,
            manifest_hash=dataset.manifest_hash,
            profile=profile,
        )
    service = ReviewService(data, outputs, tmp_path / "curated")
    catalog = service.catalog()
    request = ReviewRequest(
        rally="rally_000002",
        manifest_sha256=catalog["manifest_sha256"],
        checkpoint_3d=catalog["default_checkpoints"]["3"],
        **evaluation_profile(raw),
    )
    return service, request


def test_checkpoint_suggestions_are_compatible_and_validation_selected(review):
    service, request = review
    catalog = service.catalog()
    assert len(catalog["checkpoints"]) == 6
    selected = [p for p in catalog["checkpoints"] if p["recommended"]]
    assert [(p["run_name"], p["method"], p["filename"]) for p in selected] == [
        ("3d-regression/run", "regression", "best.ckpt")
    ]
    torch.save({"legacy": True}, service.outputs_root / "legacy.ckpt")
    excluded = [
        p for p in service.catalog(refresh=True)["checkpoints"] if not p["compatible"]
    ]
    assert len(excluded) == 1 and "形式" in excluded[0]["reason"]
    with pytest.raises(ValueError, match="カタログ"):
        service.preview(
            request.model_copy(update={"checkpoint_3d": "../../outside.ckpt"})
        )


def test_saved_and_real_cpu_inference_share_exact_inputs_and_all_frames(review):
    service, request = review
    saved, live = service.saved(request), service.infer(request)
    assert saved["input_sha256"] == live["input_sha256"]
    for dim in (3,):
        a, b = (
            np.asarray(saved["scene"][f"prediction_{dim}d"]),
            np.asarray(live["scene"][f"prediction_{dim}d"]),
        )
        np.testing.assert_allclose(a, b, atol=1e-5)
        assert a.shape == np.asarray(saved["scene"][f"gt_{dim}d"]).shape
    np.testing.assert_allclose(
        saved["scene"]["event_probability"],
        live["scene"]["event_probability"],
        atol=1e-6,
    )
    assert np.asarray(live["scene"]["event_probability"]).shape == (96,)
    assert np.asarray(live["scene"]["event_target"])[[15, 45, 85]].tolist() == [
        1.0,
        1.0,
        1.0,
    ]
    observed = ~np.asarray(live["scene"]["missing_3d"])
    assert not np.allclose(
        np.asarray(live["scene"]["prediction_3d"])[observed],
        np.asarray(live["scene"]["input_3d"])[observed],
    )
    assert live["scene"]["metrics"]["prediction"]["all"]["count"] == 96
    # A coordinate-only model has no physics series or predicted parameters.
    assert live["scene"]["integrated_3d"] is None
    assert live["scene"]["segments"]["predicted"] is None
    assert live["scene"]["physics"]["predicted"] is None
    assert set(live["scene"]["metrics"]) == {"prediction", "linear"}


def test_flow_is_single_seeded_reproducible_trajectory(review):
    service, request = review
    flow = next(
        item.info["id"]
        for item in service.checkpoints.entries.values()
        if item.info["method"] == "flow" and item.info["filename"] == "best.ckpt"
    )
    request = request.model_copy(update={"checkpoint_3d": flow})
    first, second = service.infer(request), service.infer(request)
    np.testing.assert_array_equal(
        first["scene"]["prediction_3d"], second["scene"]["prediction_3d"]
    )
    other = service.infer(
        request.model_copy(update={"flow_seed": request.flow_seed + 1})
    )
    assert first["input_sha256"] == other["input_sha256"]
    assert not np.array_equal(
        first["scene"]["prediction_3d"], other["scene"]["prediction_3d"]
    )


def test_noise_switch_does_not_move_event_gaps_and_clean_3d_is_triangulated(review):
    service, request = review
    original = service.preview(request)
    no_noise = service.preview(request.model_copy(update={"noise_enabled": False}))
    np.testing.assert_array_equal(
        original["scene"]["event_missing"], no_noise["scene"]["event_missing"]
    )
    assert original["scene"]["intervals"] == no_noise["scene"]["intervals"]
    assert no_noise["scene"]["audit"]["noise_p95_px"] == 0
    clean = service.preview(
        request.model_copy(update={"noise_enabled": False, "missing_enabled": False})
    )
    assert not np.asarray(clean["scene"]["missing_3d"]).any()
    np.testing.assert_allclose(
        clean["scene"]["input_3d"], clean["scene"]["gt_3d"], atol=1e-4
    )
    assert not np.allclose(original["scene"]["input_3d"], clean["scene"]["input_3d"])
    assert service.preview(request)["input_sha256"] == original["input_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"augmentation_seed": 1},
        {"flow_seed": 1},
        {"noise_enabled": False},
        {"rally": "rally_000000"},
    ],
)
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


def test_replaced_checkpoint_with_same_step_cannot_adopt_old_predictions_after_refresh(
    review,
):
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
    request = request.model_copy(
        update={"checkpoint_hashes": {"3": changed.info["sha256"]}}
    )
    with pytest.raises(ValueError, match="保存済み"):
        service.saved(request)
    restarted = ReviewService(
        service.data_root, service.outputs_root, service.checkpoints_root
    )
    with pytest.raises(ValueError, match="保存済み"):
        restarted.saved(request)
    after = service.infer(request)
    assert (
        np.max(
            np.abs(
                np.asarray(after["scene"]["prediction_3d"])
                - before["scene"]["prediction_3d"]
            )
        )
        > 1
    )


def test_prediction_bytes_and_receipt_are_checked_even_with_warm_array_cache(review):
    service, request = review
    service.saved(request)
    entry = service.checkpoints.entries[request.checkpoint_3d]
    with (entry.predictions / "pred_test.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="内容hash"):
        service.saved(request)
    assert not next(
        item
        for item in service.catalog(refresh=True)["checkpoints"]
        if item["id"] == request.checkpoint_3d
    )["saved_available"]


def test_legacy_predictions_require_fresh_evaluation_in_separate_review_cache(review):
    service, request = review
    entry = service.checkpoints.entries[request.checkpoint_3d]
    original = {path: sha256(path) for path in entry.run.rglob("*") if path.is_file()}
    shutil.rmtree(entry.predictions)
    service.catalog(refresh=True)
    assert not service.checkpoints.entries[request.checkpoint_3d].info[
        "saved_available"
    ]
    with pytest.raises(ValueError, match="保存済み"):
        service.saved(request)
    prepare_saved_predictions(service)
    saved, live = service.saved(request), service.infer(request)
    np.testing.assert_allclose(
        saved["scene"]["prediction_3d"], live["scene"]["prediction_3d"], atol=1e-5
    )
    assert original == {
        path: sha256(path) for path in entry.run.rglob("*") if path.is_file()
    }


def test_review_api_rejects_invalid_contract_and_uses_queue_for_cuda(
    review, monkeypatch
):
    from src.tasks.ball_refiner_3d.visualization.dataset_review import web

    service, request = review
    client = TestClient(create_app(service))
    assert client.get("/").status_code == 200
    assert client.get("/static/no-such.js").status_code == 404
    assert client.get("/shared/scene3d.mjs").status_code == 200
    assert (
        client.post(
            "/api/preview", json=request.model_dump() | {"confidence": [1]}
        ).status_code
        == 422
    )
    wrong = request.model_dump() | {"checkpoint_2d": "retired"}
    assert client.post("/api/infer", json=wrong).status_code == 422
    assert (
        client.post("/api/saved", json=request.model_dump()).json()["source"] == "saved"
    )
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
    result = json.loads(
        execute_request(
            {
                "task": "ball_refiner_3d",
                "service": {
                    "data_root": str(service.data_root),
                    "outputs_root": str(service.outputs_root),
                    "checkpoints_root": str(service.checkpoints_root),
                },
                "request": request.model_dump(),
            }
        )
    )
    assert result["source"] == "live" and result["scene"]["rally"] == request.rally
    assert result["scene"]["metrics"]["prediction"]["all"]["count"] == 96


def test_review_accepts_zero_noise_event_only_profile(review):
    service, request = review
    augmentation = request.augmentation.model_copy(
        update={
            "noise_p95_px": 0.0,
            "jitter_sigma_px": 0.0,
            "outlier_probability": 0.0,
            "isolated_probability": 0.0,
        }
    )
    request = request.model_copy(update={"augmentation": augmentation})
    client = TestClient(create_app(service))
    response = client.post("/api/preview", json=request.model_dump())
    assert response.status_code == 200
    scene = response.json()["scene"]
    assert scene["audit"]["noise_p95_px"] == 0
    assert (
        not {"gt_2d", "input_2d", "missing_2d", "court_2d", "isolated_missing"}
        & scene.keys()
    )
    mask = np.asarray(scene["missing_3d"])
    np.testing.assert_array_equal(mask, scene["event_missing"])
    np.testing.assert_allclose(
        np.asarray(scene["input_3d"])[~mask],
        np.asarray(scene["gt_3d"])[~mask],
        atol=1e-4,
    )
    assert response.json()["schema"] == "ball_refiner_3d.event_review.v2"
    html = client.get("/").text
    assert (
        'id="view-2d"' not in html
        and 'id="camera"' not in html
        and 'id="graph-2d"' not in html
    )


def _physics_request(service, request):
    entry = next(
        item.info
        for item in service.checkpoints.entries.values()
        if item.info["physics_heads"] and item.info["filename"] == "best.ckpt"
    )
    return entry, request.model_copy(
        update={"checkpoint_3d": entry["id"], "checkpoint_hashes": {}}
    )


def test_physics_heads_show_integrated_flights_parameters_and_segments(review):
    service, request = review
    entry, request = _physics_request(service, request)
    assert entry["saved_available"]
    assert (
        "3d-regression-physics/run" in entry["label"] and "物理head" in entry["label"]
    )
    saved, live = service.saved(request), service.infer(request)
    assert saved["input_sha256"] == live["input_sha256"]
    # The saved bundle (evaluator) and the public predictor agree on every output.
    for key in ("prediction_3d", "integrated_3d", "integrated_truth_3d"):
        np.testing.assert_allclose(saved["scene"][key], live["scene"][key], atol=1e-4)
        assert np.asarray(saved["scene"][key]).shape == (96, 3)
    assert saved["scene"]["segments"] == live["scene"]["segments"]
    for key in ("predicted", "surface_probability"):
        np.testing.assert_allclose(
            saved["scene"]["physics"][key], live["scene"]["physics"][key], atol=1e-5
        )
    scene = live["scene"]
    truth = np.asarray(scene["segments"]["truth"])
    assert np.flatnonzero(np.diff(truth)).tolist() == [14, 44, 84]
    # Flights integrated under the true and the predicted segmentation differ
    # unless the predicted events happen to reproduce the true split.
    if scene["segments"]["predicted"] != scene["segments"]["truth"]:
        assert not np.allclose(scene["integrated_3d"], scene["integrated_truth_3d"])
    assert scene["physics"]["columns"] == [
        "wind_x_mps",
        "wind_y_mps",
        "k_drag",
        "k_magnus",
    ]
    np.testing.assert_allclose(scene["physics"]["truth"], [1.0, -0.5, 0.01, 0.001])
    assert scene["physics"]["surface"] == "hard"
    assert sum(scene["physics"]["surface_probability"]) == pytest.approx(1, abs=1e-5)
    assert set(scene["metrics"]) == {
        "linear",
        "prediction",
        "integrated",
        "integrated_truth",
    }
    kinematics = scene["metrics"]["integrated_truth"]["kinematics"]["all"]
    assert {"acceleration_rmse", "implausible_acceleration_rate", "jerk_ratio"} <= set(
        kinematics
    )
    observed = ~np.asarray(scene["missing_3d"])
    np.testing.assert_allclose(
        np.asarray(scene["linear_3d"])[observed],
        np.asarray(scene["input_3d"])[observed],
    )


def test_saved_physics_bundle_must_carry_physics_outputs(review):
    service, request = review
    entry, request = _physics_request(service, request)
    checkpoint = service.checkpoints.entries[entry["id"]]
    bundle = checkpoint.predictions
    with np.load(bundle / "pred_test.npz") as data:
        arrays = {key: data[key] for key in data.files if key != "integrated"}
    (bundle / "pred_test.npz").unlink()
    np.savez_compressed(bundle / "pred_test.npz", allow_pickle=False, **arrays)
    receipt = json.loads((bundle / "receipt.json").read_text())
    receipt["predictions_sha256"] = sha256(bundle / "pred_test.npz")
    (bundle / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="物理head"):
        service.saved(request)


def test_catalog_binds_saved_test_summary_to_checkpoint_and_dataset(review):
    service, request = review
    entry, _ = _physics_request(service, request)
    run = service.checkpoints.entries[entry["id"]].run
    assert entry["test_summary"] is None and "test評価" in entry["test_summary_reason"]
    (run / "predictions/metrics.json").write_text(
        json.dumps({"test_rmse_m": 0.5, "test_integrated_rmse_m": 1.5, "note": "x"})
    )
    report = {
        "checkpoint_sha256": entry["sha256"],
        "dataset_manifest_sha256": service.dataset.manifest_hash,
    }
    (run / "predictions/diagnostic_metrics.json").write_text(json.dumps(report))
    refreshed = next(
        item
        for item in service.catalog(refresh=True)["checkpoints"]
        if item["id"] == entry["id"]
    )
    assert refreshed["test_summary"] == {
        "test_rmse_m": 0.5,
        "test_integrated_rmse_m": 1.5,
    }
    (run / "predictions/diagnostic_metrics.json").write_text(
        json.dumps({**report, "checkpoint_sha256": "0" * 64})
    )
    refreshed = next(
        item
        for item in service.catalog(refresh=True)["checkpoints"]
        if item["id"] == entry["id"]
    )
    assert refreshed["test_summary"] is None
    assert "別のcheckpoint" in refreshed["test_summary_reason"]
