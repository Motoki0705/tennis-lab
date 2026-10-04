"""Small fixtures test storage semantics; review screenshots use real datasets."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.ball_refiner.refiner_3d.review.data import DatasetReview, sha256


def save(root: Path, manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    folder = root / "saved"
    folder.mkdir(exist_ok=True)
    path = folder / "train-00000.npz"
    np.savez_compressed(path, **arrays)
    record = manifest["rallies"][0]
    record.update(npz_bytes=path.stat().st_size, npz_sha256=sha256(path))
    (folder / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def saved(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    t, v, k = 4, 3, 2
    combinations = list(itertools.product([-1, 0, 1], repeat=v))
    subsets = np.array([[x >= 0 for x in row] for row in combinations], dtype=bool)
    p, w = np.array([0.7, 0.8, 0.2]), np.array([0.6, 0.4])
    weights = np.array(
        [
            np.prod([(1 - p[i]) if x < 0 else p[i] * w[x] for i, x in enumerate(row)])
            for row in combinations
        ]
    )
    components = len(weights)
    arrays = {
        "timestamps_seconds": np.arange(t) * 1001 / 60000,
        "positions_3d_m": np.tile([1.0, 2.0, 3.0], (t, 1)),
        "source_size_wh": np.tile([101, 81], (v, 1)),
        "occlusion_mask": np.tile([False, True, False, False], (v, 1)),
        "out_of_frame_mask": np.zeros((v, t), dtype=bool),
        "event_labels": np.zeros((t, 2), dtype=bool),
        "event_region_mask": np.ones(t, dtype=bool),
        "free_flight_mask": np.zeros(t, dtype=bool),
        "gmm2d_means_uv": np.tile([0.25, 0.5], (v, t, k, 1)),
        "gmm2d_scale_tril_uv": np.tile([[0.01, 0], [0.005, 0.02]], (v, t, k, 1, 1)),
        "gmm2d_mixture_logits": np.tile(np.log(w), (v, t, 1)),
        "gmm2d_presence_logits": np.tile(np.log(p / (1 - p))[:, None], (1, t)),
        "gmm3d_means_m": np.zeros((t, components, 3)),
        "gmm3d_covariance_m2": np.tile(np.eye(3), (t, components, 1, 1)),
        "gmm3d_weights": np.tile(weights, (t, 1)),
        "gmm3d_camera_subsets": np.tile(subsets, (t, 1, 1)),
        "gmm3d_method_codes": np.tile(
            np.where(subsets.any(-1), 1, 0).astype(np.uint8), (t, 1)
        ),
        "prior_only_probability": np.full(t, np.prod(1 - p)),
        "integration_converged": np.zeros(t, dtype=bool),
        "integration_convergence_assessed": np.zeros(t, dtype=bool),
        "integration_rounds": np.ones(t, dtype=np.uint8),
    }
    arrays["event_labels"][0, 0] = True
    for prefix in ("true", "estimated"):
        arrays[f"camera_{prefix}_K"] = np.tile(
            [[100, 0, 50], [0, 120, 40], [0, 0, 1]], (v, 1, 1)
        ).astype(float)
        arrays[f"camera_{prefix}_R"] = np.tile(np.eye(3), (v, 1, 1))
        arrays[f"camera_{prefix}_t"] = np.tile([0.0, 0.0, 10.0], (v, 1))
    manifest = {
        "schema": "ball_refiner_3d.synthetic.v2",
        "status": "complete",
        "mode": "dev",
        "counts": {"train": 1, "val": 0, "test": 0},
        "plan": {
            "schema_version": 2,
            "sampling": {
                "fps_numerator": 60000,
                "fps_denominator": 1001,
                "physics_event_mask_radius_frames": 5,
            },
            "geometry": {"sources": [{"camera_keys": ["cam0", "cam1", "cam2"]}]},
            "degradation": {
                "components_per_camera": k,
                "status": "test-only",
                "boundary_convergence": {"method": "fixed_hybrid"},
            },
        },
        "rallies": [
            {
                "rally_id": "train-00000",
                "split": "train",
                "frames": t,
                "seed": 936,
                "geometry_clip": "test-only",
                "components_per_frame": components,
                "fps_numerator": 60000,
                "fps_denominator": 1001,
                "component_method_labels": ["prior", "laplace"],
                "all_camera_occluded_frames": 1,
                "events": [{"kind": "hit", "frame": 0, "seconds": 0.0}],
            }
        ],
    }
    save(tmp_path, manifest, arrays)
    return tmp_path, manifest, arrays


def test_pixel_covariance_uses_both_axes_and_keeps_all_modes(saved):
    root, _, _ = saved
    frame = DatasetReview(root).frame("saved", "train-00000", 1)
    camera = frame["cameras"][0]
    np.testing.assert_allclose(camera["means_px"], [[25, 40], [25, 40]])
    np.testing.assert_allclose(camera["covariance_px2"], [[[1, 0.4], [0.4, 2.72]]] * 2)
    np.testing.assert_allclose(camera["truth_px"], [50 + 100 / 13, 40 + 240 / 13])
    assert len(frame["weights"]) == 27
    assert len(frame["subset_mass"]) == 8
    assert sum(row["mass"] for row in frame["subset_mass"]) == pytest.approx(1)


def test_gap_does_not_become_absence_or_prior_only(saved):
    root, _, _ = saved
    service = DatasetReview(root)
    frame = service.frame("saved", "train-00000", 1)
    assert all(c["occluded"] and not c["out_of_frame"] for c in frame["cameras"])
    assert [c["presence"] for c in frame["cameras"]] == pytest.approx([0.7, 0.8, 0.2])
    assert frame["prior_only_probability"] == pytest.approx(0.3 * 0.2 * 0.8)
    assert frame["convergence"] == "unassessed"
    assert (
        service.sequence("saved", "train-00000")["timestamps_seconds"][1]
        == 1001 / 60000
    )


@pytest.mark.parametrize(
    "diagnostic,expected", [(False, "not_saved"), (True, "nonconverged")]
)
def test_older_diagnostics_are_distinct_from_unassessed(saved, diagnostic, expected):
    root, manifest, arrays = saved
    del arrays["integration_convergence_assessed"]
    if diagnostic:
        manifest["plan"]["degradation"]["boundary_convergence"] = {"method": "ray"}
    else:
        del manifest["plan"]["degradation"]["boundary_convergence"]
        del arrays["integration_converged"]
        del arrays["integration_rounds"]
    save(root, manifest, arrays)
    assert (
        DatasetReview(root).frame("saved", "train-00000", 0)["convergence"] == expected
    )


def test_behind_camera_truth_is_null_not_zero(saved):
    root, manifest, arrays = saved
    arrays["camera_true_t"][:, 2] = -20
    arrays["out_of_frame_mask"][:] = True
    save(root, manifest, arrays)
    frame = DatasetReview(root).frame("saved", "train-00000", 0)
    assert all(
        c["truth_px"] is None and not c["truth_in_front"] for c in frame["cameras"]
    )


def test_failed_orphan_and_stopped_counts_are_visible(saved):
    root, manifest, _ = saved
    folder = root / "failed"
    folder.mkdir()
    manifest["status"] = "failed"
    manifest["rallies"] = []
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "val-00001.npz").write_bytes(b"unregistered")
    service = DatasetReview(root)
    row = next(x for x in service.catalog() if x["id"] == "failed")
    assert not row["available"] and row["orphan_npz"] == ["val-00001.npz"]
    assert row["counts"]["train"] == 0 and row["planned_counts"]["train"] == 1
    with pytest.raises(ValueError, match="not registered"):
        service.load("failed", "val-00001")
    manifest["status"] = "stopped"
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert DatasetReview(root).catalog()[0]["status"] == "stopped"


def test_corruption_is_detected_even_after_caching(saved):
    root, _, _ = saved
    service = DatasetReview(root)
    service.load("saved", "train-00000")
    path = root / "saved/train-00000.npz"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        service.load("saved", "train-00000")


def test_manifest_edit_requires_restart(saved):
    root, manifest, _ = saved
    service = DatasetReview(root)
    manifest["status"] = "stopped"
    (root / "saved/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest changed"):
        service.catalog()


@pytest.mark.parametrize(
    "failure",
    [
        "axes",
        "fps",
        "covariance",
        "subset",
        "presence",
        "assessed",
        "nonfinite",
        "bool",
        "prior",
        "outside",
        "event",
    ],
)
def test_semantic_corruption_fails_explicitly(saved, failure):
    root, manifest, arrays = saved
    if failure == "axes":
        arrays["gmm2d_means_uv"] = arrays["gmm2d_means_uv"].swapaxes(0, 1)
    elif failure == "fps":
        arrays["timestamps_seconds"] = np.arange(4) / 60
    elif failure == "covariance":
        arrays["gmm3d_covariance_m2"][:, :, 0, 0] = -1
    elif failure == "subset":
        arrays["gmm3d_camera_subsets"][:, 0, 0] = True
    elif failure == "presence":
        arrays["gmm2d_presence_logits"][0] = 0
    elif failure == "assessed":
        arrays["integration_convergence_assessed"][:] = True
    elif failure == "nonfinite":
        arrays["gmm3d_means_m"][0, 0, 0] = np.nan
    elif failure == "bool":
        arrays["occlusion_mask"] = arrays["occlusion_mask"].astype(int)
    elif failure == "outside":
        arrays["out_of_frame_mask"][:] = True
    elif failure == "event":
        manifest["rallies"][0]["events"][0]["frame"] = 3
    else:
        arrays["prior_only_probability"][:] = 1
    save(root, manifest, arrays)
    with pytest.raises(ValueError):
        DatasetReview(root).load("saved", "train-00000")


@pytest.mark.parametrize("index", [-1, 4])
def test_frame_bounds_and_unregistered_identity(saved, index):
    root, _, _ = saved
    service = DatasetReview(root)
    with pytest.raises(ValueError, match="outside"):
        service.frame("saved", "train-00000", index)
    with pytest.raises(ValueError, match="not registered"):
        service.load("saved", "../../other")
    with pytest.raises(ValueError, match="Unknown dataset"):
        service.load("../../outside", "train-00000")


def test_complete_counts_cannot_be_planned_counts(saved):
    root, manifest, _ = saved
    manifest["counts"]["train"] = 2
    (root / "saved/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count mismatch"):
        DatasetReview(root)


def test_empty_root_does_not_generate_or_fall_back(tmp_path):
    with pytest.raises(ValueError, match="No saved"):
        DatasetReview(tmp_path)


def test_http_contract_errors_and_static_resources(saved):
    from fastapi.testclient import TestClient

    from src.tasks.ball_refiner.refiner_3d.review.web import create_app

    root, _, _ = saved
    with TestClient(create_app(DatasetReview(root))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
        assert client.get("/api/catalog").json()[0]["frames"] == 4
        assert (
            client.get(
                "/api/sequence", params={"dataset": "saved", "rally": "train-00000"}
            ).json()["record"]["seed"]
            == 936
        )
        frame = client.get(
            "/api/frame",
            params={"dataset": "saved", "rally": "train-00000", "frame": 1},
        )
        assert frame.status_code == 200 and frame.json()["convergence"] == "unassessed"
        assert (
            client.get(
                "/api/frame",
                params={"dataset": "saved", "rally": "train-00000", "frame": 99},
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/api/rallies", params={"dataset": "saved", "split": "holdout"}
            ).status_code
            == 400
        )
        assert client.post("/api/frame").status_code == 405
        assert client.get("/manifest.json").status_code == 404
