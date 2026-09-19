"""Numerical regression tests for the CPU camera retrieval baseline."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.synthetic_data_generation.court_calibration.database import (
    POINTS,
    DatabaseConfig,
    LineDatabase,
    generate,
)
from src.synthetic_data_generation.court_calibration.matching import (
    query_database,
    refine,
    resize_query,
    validate_mask,
)
from src.synthetic_data_generation.dataset.camera_profiles import CameraSlotConfig


@pytest.fixture
def config() -> DatabaseConfig:
    return DatabaseConfig(
        3,
        42,
        256,
        144,
        1,
        CameraSlotConfig(
            "test",
            (-2.0, 2.0),
            (-32.0, -28.0),
            (10.0, 12.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (45.0, 50.0),
        ),
    )


def test_deterministic_roundtrip_and_cache_identity(
    config: DatabaseConfig, tmp_path: Path
) -> None:
    a, b = generate(config), generate(config)
    for name in ("K", "R", "t", "H", "masks", "descriptors"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    path = tmp_path / "database.npz"
    a.save(path)
    restored = LineDatabase.load(path, expected_config=config)
    np.testing.assert_array_equal(restored.H, a.H)
    with pytest.raises(FileExistsError):
        a.save(path)
    with pytest.raises(ValueError, match="identity mismatch"):
        LineDatabase.load(path, expected_config=replace(config, seed=43))
    restored.H[0, 0, 2] += 5
    with pytest.raises(ValueError, match="disagrees"):
        restored.validate()


def test_axes_and_retrieval(config: DatabaseConfig) -> None:
    db = generate(config)
    points = POINTS @ db.R[0].T + db.t[0]
    uv_camera = points @ db.K[0].T
    uv_h = np.column_stack((POINTS[:, :2], np.ones(14))) @ db.H[0].T
    np.testing.assert_allclose(
        uv_camera[:, :2] / uv_camera[:, 2:], uv_h[:, :2] / uv_h[:, 2:], atol=1e-8
    )
    assert uv_h[0, 0] / uv_h[0, 2] < uv_h[1, 0] / uv_h[1, 2]
    matches = query_database(db, db.masks[0], top_k=2)
    assert matches[0].index == 0
    assert matches[0].descriptor_distance == 0
    assert matches[0].success


def test_perspective_recovery_and_ecc_direction(config: DatabaseConfig) -> None:
    db = generate(config)
    perturbation = np.array(
        [[1.005, 0.003, 2.0], [-0.002, 0.995, 1.5], [0.00002, -0.00003, 1.0]]
    )
    query = cv2.warpPerspective(
        db.masks[0],
        perturbation,
        (config.width, config.height),
        flags=cv2.INTER_NEAREST,
    )
    h, warp, score, reason = refine(db.masks[0], query, db.H[0])
    assert h is not None, reason
    assert warp is not None and score is not None and score > 0.95
    ground = np.column_stack((POINTS[:, :2], np.ones(14)))
    expected = ground @ (perturbation @ db.H[0]).T
    actual = ground @ h.T
    error = np.linalg.norm(
        actual[:, :2] / actual[:, 2:] - expected[:, :2] / expected[:, 2:], axis=1
    )
    assert error.max() < 1.0
    np.testing.assert_allclose(warp @ h / (warp @ h)[2, 2], db.H[0], atol=1e-7)


def test_resize_half_pixel_and_nonmatching_aspect(config: DatabaseConfig) -> None:
    db = generate(config)
    original = cv2.resize(db.masks[0], (512, 432), interpolation=cv2.INTER_NEAREST)
    resized, scale = resize_query(original, (256, 144))
    np.testing.assert_array_equal(resized, db.masks[0])
    np.testing.assert_allclose(scale @ [10.5, 11.5, 1], [5.0, 3.5, 1])
    result = query_database(db, original, top_k=1)[0]
    assert (
        result.success
        and result.world_to_query is not None
        and result.query_to_template is not None
    )
    composed = result.query_to_template @ result.world_to_query
    np.testing.assert_allclose(composed / composed[2, 2], db.H[0], atol=1e-7)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((144, 256), np.uint8),
        np.full((144, 256), 255, np.uint8),
        np.ones((144, 256), np.float32),
        np.ones((144, 256, 3), np.uint8),
    ],
)
def test_invalid_query_rejected(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_mask(mask)


def test_ecc_failure_is_explicit(
    config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = generate(config)

    def fail(*args: object, **kwargs: object) -> None:
        raise cv2.error("test nonconvergence")

    monkeypatch.setattr(cv2, "findTransformECC", fail)
    result = query_database(db, db.masks[0], top_k=1)[0]
    assert (
        not result.success
        and result.world_to_query is None
        and result.query_to_template is None
    )
    assert "ecc_failed" in result.reason


def test_invalid_sampling_prior_rejected(config: DatabaseConfig) -> None:
    with pytest.raises(ValueError):
        replace(config, width=257)
    with pytest.raises(ValueError):
        replace(config, camera=replace(config.camera, height_m=(-1.0, 2.0)))


def test_cli_generate_query_and_path_contract(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    from src.utils.paths import PROJECT_ROOT

    common = [
        f"paths.data_root={tmp_path / 'dataset_root'}",
        f"paths.output_root={tmp_path / 'output_root'}",
        "database.count=3",
        "matching.top_k=1",
    ]
    cli = [
        sys.executable,
        "-m",
        "src.synthetic_data_generation.scripts.court_line_database",
    ]
    subprocess.run(
        cli + common + ["output_dir=court_detection/generate/cli/test"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    data = tmp_path / "dataset_root/court_detection/calibration_baseline_v1"
    with np.load(data / "database.npz", allow_pickle=False) as archive:
        assert cv2.imwrite(str(data / "query.png"), archive["masks"][0])
    subprocess.run(
        cli + common + ["mode=query", "output_dir=court_detection/analyze/cli/test"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    result = json.loads(
        (
            tmp_path / "output_root/court_detection/analyze/cli/test/result.json"
        ).read_text()
    )
    assert result["matches"][0]["success"]
    invalid = subprocess.run(
        cli + common + ["database_path=../escape.npz"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert not (tmp_path / "escape.npz").exists()
    missing_query = subprocess.run(
        cli
        + common
        + [
            "mode=query",
            "query_path=missing.png",
            "output_dir=court_detection/analyze/missing/test",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    assert missing_query.returncode != 0
    assert not (tmp_path / "output_root/court_detection/analyze/missing/test").exists()


@pytest.mark.parametrize("field", ["K", "R", "t", "H", "descriptors"])
def test_nonfinite_database_rejected(config: DatabaseConfig, field: str) -> None:
    db = generate(config)
    getattr(db, field).flat[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        db.validate()


def test_reflected_ecc_mapping_rejected(
    config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = generate(config)

    def reflect(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        return 0.99, np.array([[-1.0, 0.0, 255.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    monkeypatch.setattr(cv2, "findTransformECC", reflect)
    result = query_database(db, db.masks[0], top_k=1)[0]
    assert not result.success and result.world_to_query is None
    assert "invalid_refinement" in result.reason


@pytest.mark.parametrize(
    "field", ["geometry_sha256", "implementation_sha256", "opencv"]
)
def test_archive_provenance_mismatch_rejected_before_validation(
    config: DatabaseConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    import json

    path = tmp_path / "database.npz"
    generate(config).save(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata"].item()))
    metadata[field] = "incompatible"
    payload["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)

    def unexpected_validation(self: LineDatabase) -> None:
        pytest.fail("incompatible archive reached numerical validation")

    monkeypatch.setattr(LineDatabase, "validate", unexpected_validation)
    with pytest.raises(ValueError, match="config/source identity mismatch"):
        LineDatabase.load(path, expected_config=config)
