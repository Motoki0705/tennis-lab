"""Tests for real-capture ball calibration import and bundle integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.blcs.calibration import (
    BALL_CALIBRATION_CAPTURE_SCHEMA,
    import_ball_calibration_capture,
    load_ball_calibration_bundle,
    load_ball_calibration_capture,
    load_ball_calibration_import,
    write_ball_calibration_bundle,
)


def _file_ref(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_capture(root: Path) -> Path:
    rgb_root = root / "rgb"
    mask_root = root / "mask"
    rgb_root.mkdir(parents=True)
    mask_root.mkdir()
    width = 20
    height = 12
    intrinsic = [
        25.0,
        0.0,
        width / 2,
        0.0,
        25.0,
        height / 2,
        0.0,
        0.0,
        1.0,
    ]
    views: list[dict[str, object]] = []
    for index, split in ((2, "validation"), (0, "train"), (1, "train")):
        view_id = f"view-{index:03d}"
        rgb_path = rgb_root / f"{view_id}.png"
        mask_path = mask_root / f"{view_id}.png"
        rgb: NDArray[np.uint8] = np.full(
            (height, width, 3), 245 - index, dtype=np.uint8
        )
        rgb[4:8, 8:13] = np.asarray([180 + index, 220, 30], dtype=np.uint8)
        mask: NDArray[np.uint8] = np.zeros(
            (height, width), dtype=np.uint8
        )
        mask[4:8, 8:13] = 255
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(mask).save(mask_path)
        camera = np.eye(4, dtype=np.float64)
        camera[0, 3] = 0.1 * index
        views.append(
            {
                "view_id": view_id,
                "split": split,
                "width": width,
                "height": height,
                "rgb": _file_ref(root, rgb_path),
                "mask": _file_ref(root, mask_path),
                "camera_to_asset": camera.ravel().tolist(),
                "intrinsics": intrinsic,
            }
        )
    manifest = root / "capture.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": BALL_CALIBRATION_CAPTURE_SCHEMA,
                "capture_id": "unit-ball-capture",
                "views": views,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_capture_import_is_sorted_strict_and_reproducible(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path / "capture")
    first_path = import_ball_calibration_capture(
        capture_path,
        tmp_path / "import-a",
        bundle_id="unit-ball-bundle",
    )
    second_path = import_ball_calibration_capture(
        capture_path,
        tmp_path / "import-b",
        bundle_id="unit-ball-bundle",
    )

    first = load_ball_calibration_import(first_path)
    second = load_ball_calibration_import(second_path)
    assert first.manifest["ordered_view_ids"] == [
        "view-000",
        "view-001",
        "view-002",
    ]
    assert (
        first.bundle.manifest["content_fingerprint"]
        == second.bundle.manifest["content_fingerprint"]
    )
    assert _tree_hashes(first.root) == _tree_hashes(second.root)
    assert first.bundle.camera_to_asset.dtype == np.float32
    assert first.bundle.rgb.dtype == np.uint8
    assert first.bundle.mask.dtype == np.bool_
    assert first.bundle.train_indices == (0, 1)
    assert first.bundle.validation_indices == (2,)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_ball_calibration_capture(
            capture_path,
            tmp_path / "import-a",
            bundle_id="unit-ball-bundle",
        )


def test_capture_rejects_antialiased_mask_before_publication(
    tmp_path: Path,
) -> None:
    capture_path = _write_capture(tmp_path / "capture")
    raw = json.loads(capture_path.read_text(encoding="utf-8"))
    view = raw["views"][0]
    mask_path = capture_path.parent / view["mask"]["relative_path"]
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
    mask[0, 0] = 127
    Image.fromarray(mask).save(mask_path)
    view["mask"] = _file_ref(capture_path.parent, mask_path)
    capture_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="only 0 and 255"):
        import_ball_calibration_capture(
            capture_path,
            output,
            bundle_id="unit-ball-bundle",
        )
    assert not output.exists()


def test_capture_rejects_split_leakage_and_unsafe_paths(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path / "capture")
    raw = json.loads(capture_path.read_text(encoding="utf-8"))
    train_rgb = raw["views"][1]["rgb"]
    raw["views"][0]["rgb"] = train_rgb
    capture_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RGB contents must be unique"):
        load_ball_calibration_capture(capture_path)

    raw["views"][0]["rgb"] = {
        "relative_path": "../outside.png",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    capture_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relative_path is unsafe"):
        load_ball_calibration_capture(capture_path)


def test_capture_rejects_improper_camera_rotation(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path / "capture")
    raw = json.loads(capture_path.read_text(encoding="utf-8"))
    camera = np.asarray(raw["views"][0]["camera_to_asset"]).reshape(4, 4)
    camera[0, 0] = 2.0
    raw["views"][0]["camera_to_asset"] = camera.ravel().tolist()
    capture_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="proper rotation"):
        load_ball_calibration_capture(capture_path)


def test_bundle_rejects_full_mask_and_tampered_tensor_bytes(
    tmp_path: Path,
) -> None:
    camera = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    intrinsic = np.broadcast_to(
        np.asarray([[20.0, 0.0, 5.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]),
        (3, 3, 3),
    ).astype(np.float32)
    rgb: NDArray[np.uint8] = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    full_mask: NDArray[np.bool_] = np.ones((3, 8, 10), dtype=np.bool_)
    split = np.asarray([0, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="must not fill"):
        write_ball_calibration_bundle(
            tmp_path / "full-mask",
            bundle_id="unit-ball-bundle",
            camera_to_asset=camera,
            intrinsics=intrinsic,
            rgb=rgb,
            mask=full_mask,
            split=split,
        )

    valid_mask = np.zeros_like(full_mask)
    valid_mask[:, 2:5, 3:7] = True
    manifest_path = write_ball_calibration_bundle(
        tmp_path / "bundle",
        bundle_id="unit-ball-bundle",
        camera_to_asset=camera,
        intrinsics=intrinsic,
        rgb=rgb,
        mask=valid_mask,
        split=split,
    )
    tensor_path = manifest_path.parent / "calibration.npz"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size differs"):
        load_ball_calibration_bundle(manifest_path.parent)
