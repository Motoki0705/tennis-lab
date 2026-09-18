# mypy: disallow_untyped_decorators=False
"""Small CPU fixtures check exact comparisons and failure receipts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.utils.checksum import FileIntegrityError


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "court_comparison", Path(__file__).with_name("compare.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))


@pytest.fixture
def view(tmp_path: Path) -> dict[str, Any]:
    production, probe, old = (
        tmp_path / name for name in ("production", "probe", "old")
    )
    for path in (production, probe, old):
        path.mkdir()
    raw: dict[str, np.ndarray] = {
        "keypoints_px": np.ones((9, 14, 2), np.float64),
        "scores": np.ones((9, 14), np.float32),
        "frame_indices": np.arange(9, dtype=np.int64),
    }
    old_h, final_h = np.eye(3), np.eye(3) * 2
    for stage, variant, h in [
        ("initial", "A_margin025", old_h),
        ("refined", "C_savedH_union20", final_h),
    ]:
        receipt = {
            "camera_id": "cam0",
            "pass": stage,
            "status": "fit_returned",
            "homography": h.tolist(),
            "source_roi_xyxy": [0, 0, 100, 100],
            "initial_homography": old_h.tolist(),
        }
        trial = {
            "clip": "fixture/clip",
            "camera": "cam0",
            "variant": variant,
            "status": "fit_returned",
            "homography": h.tolist(),
            "roi_xyxy": [0, 0, 100, 100],
            "repeatability_established": True,
        }
        write(production / f"cam0_court_{stage}.json", receipt)
        write(probe / f"{variant}.json", trial)
        np.savez_compressed(
            production / f"cam0_court_{stage}_samples.npz", allow_pickle=False, **raw
        )
        np.savez_compressed(probe / f"{variant}_raw.npz", allow_pickle=False, **raw)
    for path in (old, production):
        np.savez_compressed(path / "cam0_court_samples.npz", allow_pickle=False, **raw)
    return {
        "label": "fixture/clip",
        "camera": "cam0",
        "production": production,
        "probe": probe,
        "old": old,
        "old_h": old_h,
        "final_h": final_h,
        "initial_roi": [0, 0, 100, 100],
        "final_roi": [0, 0, 100, 100],
    }


def test_exact_initial_refined_and_published_outputs_pass(
    driver: ModuleType, view: dict[str, Any]
) -> None:
    audit = driver.Audit()
    driver.compare_view(audit, **view)
    assert len(audit.checks) == 21
    assert all(check["exact"] for check in audit.checks.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "scores",
        "dtype",
        "indices",
        "homography",
        "roi",
        "initial_homography",
        "published_h",
    ],
)
def test_view_differences_are_detected(
    driver: ModuleType, view: dict[str, Any], mutation: str
) -> None:
    production = view["production"]
    if mutation in ("raw", "scores", "dtype", "indices"):
        path = production / "cam0_court_refined_samples.npz"
        with np.load(path) as source:
            arrays = {key: source[key].copy() for key in source.files}
        if mutation == "dtype":
            arrays["scores"] = arrays["scores"].astype(np.float64)
        else:
            key = {
                "raw": "keypoints_px",
                "scores": "scores",
                "indices": "frame_indices",
            }[mutation]
            arrays[key].flat[0] += 1
        np.savez_compressed(path, allow_pickle=False, **arrays)
    elif mutation == "published_h":
        view["final_h"][0, 0] += 0.01
    else:
        path = production / "cam0_court_refined.json"
        receipt = json.loads(path.read_text())
        if mutation == "roi":
            receipt["source_roi_xyxy"][0] += 1
        else:
            receipt[mutation][0][0] += 0.01
        write(path, receipt)
    audit = driver.Audit()
    driver.compare_view(audit, **view)
    assert any(not check["exact"] for check in audit.checks.values())


def test_nan_and_shape_differences_never_pass(driver: ModuleType) -> None:
    assert not driver.array_comparison(np.array([np.nan]), np.array([np.nan]))["exact"]
    result = driver.array_comparison(np.ones((2, 2)), np.ones(4))
    assert not result["exact"] and result["different_elements"] is None


@pytest.mark.parametrize("mode", ["difference", "mutation", "missing"])
def test_failure_always_writes_receipt(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file = source / "input"
    file.write_bytes(b"original")

    def comparison(audit: Any, **kwargs: Any) -> None:
        audit.digest(file)
        if mode == "difference":
            audit.compare("fixture", np.array([1]), np.array([2]))
        elif mode == "mutation":
            file.write_bytes(b"changed")
        else:
            audit.json(source / "missing.json")

    monkeypatch.setattr(driver, "compare_all", comparison)
    output = tmp_path / "report"
    with pytest.raises((ValueError, FileIntegrityError)):
        driver.run(
            source=source,
            target=tmp_path / "target",
            probe=tmp_path / "probe",
            dataset=tmp_path / "dataset",
            output=output,
            old_samples_root=tmp_path / "old_samples",
        )
    receipt = json.loads((output / "comparison.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["inputs_before"]
    if mode == "mutation":
        assert not receipt["inputs_stable"]
    else:
        assert receipt["error"]
    with pytest.raises(ValueError, match="new and absolute"):
        driver.run(
            source=source,
            target=tmp_path / "target",
            probe=tmp_path / "probe",
            dataset=tmp_path / "dataset",
            output=output,
            old_samples_root=tmp_path / "old_samples",
        )


def test_explicit_old_samples_origin_requires_historical_receipts(
    driver: ModuleType, view: dict[str, Any], tmp_path: Path
) -> None:
    origin = tmp_path / "sample_root" / "fixture" / "clip"
    source = tmp_path / "source_root" / "fixture" / "clip"
    origin.mkdir(parents=True)
    source.mkdir(parents=True)
    snapshots = {}
    for name in ("court.json", "court.npz"):
        (origin / name).write_bytes(b"fixed court bytes")
        (source / name).write_bytes(b"fixed court bytes")
        snapshots[str(origin / name)] = {
            "dual_sha256": driver.dual_sha256(origin / name)
        }
    for camera in ("cam0", "cam1", "cam2"):
        path = origin / f"{camera}_court_samples.npz"
        path.write_bytes((view["old"] / "cam0_court_samples.npz").read_bytes())
        if camera != "cam2":
            snapshots[str(path)] = {"dual_sha256": driver.dual_sha256(path)}
    kwargs = {
        "old_samples_root": tmp_path / "sample_root",
        "source": tmp_path / "source_root",
        "probe_inputs": snapshots,
        "clip_id": "fixture/clip",
    }
    records = driver.validate_old_samples(driver.Audit(), **kwargs)
    assert records["cam0"]["historical_probe_digest_available"] is True
    assert records["cam2"]["historical_probe_digest_available"] is False
    (origin / "cam0_court_samples.npz").unlink()
    with pytest.raises(FileIntegrityError):
        driver.validate_old_samples(driver.Audit(), **kwargs)
