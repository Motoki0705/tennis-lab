"""Input integrity and a real COLMAP fixture with intentionally stale errors."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from experiments.sfm_comparison.evaluate_model import validate_manifest


def frozen_images(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    entries = []
    for index in range(2):
        path = images / f"frame_{index:06d}.jpg"
        Image.new("RGB", (640, 480)).save(path)
        entries.append(
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"accepted_image_count": 2, "images": entries}))
    return manifest, images


def test_rejects_changed_frozen_input(tmp_path: Path) -> None:
    manifest, images = frozen_images(tmp_path)
    (images / "frame_000001.jpg").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(manifest, images)


def test_rejects_duplicate_manifest_names(tmp_path: Path) -> None:
    manifest, images = frozen_images(tmp_path)
    data = json.loads(manifest.read_text())
    data["images"][1] = data["images"][0]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(manifest, images)


def test_recomputes_stale_error_without_modifying_original(tmp_path: Path) -> None:
    python = os.environ.get("SFM_EVAL_PYTHON")
    evaluator = os.environ.get("SFM_EVAL_NHT_ROOT")
    if not python or not evaluator:
        pytest.skip("Requires the isolated SfM runtime and pinned NHT source")
        return
    manifest, images = frozen_images(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 100 100 320 240\n")
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_000000.jpg\n321 240 1\n"
        "2 1 0 0 0 -1 0 0 1 frame_000001.jpg\n301 240 1\n"
    )
    # Both observations have exactly 1 px residual; saved error is deliberately stale.
    point_text = "1 0 0 5 255 255 255 999 1 0 2 0\n"
    (model / "points3D.txt").write_text(point_text)
    output = tmp_path / "evaluation"
    result = subprocess.run(
        [
            python,
            "-m",
            "experiments.sfm_comparison.evaluate_model",
            "--model",
            str(model),
            "--manifest",
            str(manifest),
            "--image-dir",
            str(images),
            "--evaluator-root",
            evaluator,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    assert metrics["mean_reprojection_error_px"] == pytest.approx(1)
    assert metrics["registration_ratio"] == 1
    assert audit["stored_to_recomputed_max_abs_px"] == pytest.approx(998)
    assert (model / "points3D.txt").read_text() == point_text
