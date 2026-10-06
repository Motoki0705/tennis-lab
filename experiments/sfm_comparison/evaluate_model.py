"""Validate frozen inputs and recompute SfM residuals without changing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.sfm_comparison.run_command import write_json


def validate_manifest(
    manifest_path: Path, image_dir: Path
) -> dict[str, tuple[int, int]]:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["images"]
    if not entries or manifest["accepted_image_count"] != len(entries):
        raise ValueError("Manifest count differs from its nonempty image list")
    dimensions: dict[str, tuple[int, int]] = {}
    for entry in entries:
        name = entry["name"]
        if Path(name).is_absolute() or ".." in Path(name).parts or name in dimensions:
            raise ValueError(f"Invalid or duplicate image name: {name}")
        path = image_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Frozen image hash mismatch: {name}")
        with Image.open(path) as image:
            dimensions[name] = image.size
    return dimensions


def evaluate(
    model: Path, manifest: Path, image_dir: Path, evaluator_root: Path, output: Path
) -> dict[str, Any]:
    # This CPU command runs in the external PyCOLMAP environment, not project .venv.
    import numpy as np
    import pycolmap

    dimensions = validate_manifest(manifest, image_dir)
    reconstruction = pycolmap.Reconstruction(model)
    if not reconstruction.num_reg_images() or not reconstruction.num_points3D():
        raise ValueError("Model has no registered cameras or sparse points")
    names: list[str] = []
    poses: list[dict[str, Any]] = []
    for image in reconstruction.images.values():
        if not image.has_pose or image.name not in dimensions or image.name in names:
            raise ValueError(f"Unposed, unknown or duplicate model image: {image.name}")
        camera = reconstruction.cameras[image.camera_id]
        if (camera.width, camera.height) != dimensions[image.name]:
            raise ValueError(
                f"Camera dimensions differ from frozen image: {image.name}"
            )
        if (
            not np.isfinite(image.cam_from_world().matrix()).all()
            or not np.isfinite(camera.params).all()
        ):
            raise ValueError(f"Nonfinite camera geometry: {image.name}")
        names.append(image.name)
        poses.append(
            {"image": image.name, "center": image.projection_center().tolist()}
        )
    points = list(reconstruction.points3D.values())
    if any(
        point.track.length() == 0 or not np.isfinite(point.xyz).all()
        for point in points
    ):
        raise ValueError("Invalid point geometry or empty observation track")
    stored = np.array([point.error for point in points])
    reconstruction.update_point_3d_errors()
    refreshed = np.array([point.error for point in points])
    if not np.isfinite(refreshed).all() or np.any(refreshed < 0):
        raise ValueError("Invalid recomputed reprojection error")
    valid_stored = np.isfinite(stored) & (stored >= 0)
    delta = np.abs(stored[valid_stored] - refreshed[valid_stored])
    output.mkdir(parents=True, exist_ok=False)
    refreshed_model = output / "model-with-refreshed-errors"
    refreshed_model.mkdir()
    reconstruction.write(refreshed_model)
    audit = {
        "original_model": str(model.resolve()),
        "evaluated_model": str(refreshed_model.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "input_images": len(dimensions),
        "registered_images": sorted(names),
        "missing_images": sorted(set(dimensions) - set(names)),
        "original_dimensions_verified": True,
        "pycolmap_version": pycolmap.__version__,
        "error_definition": "Per-point mean Euclidean pixel residual over its track, recomputed from final geometry; mean/median/p95 across points.",
        "invalid_stored_point_errors": int((~valid_stored).sum()),
        "stored_to_recomputed_max_abs_px": float(delta.max()) if len(delta) else None,
        "stored_to_recomputed_mean_abs_px": float(delta.mean()) if len(delta) else None,
        "changed_point_errors_over_1e_6_px": int((delta > 1e-6).sum()),
        "gate_note": "NHT CLI generic gates are not baseline native short-clip gates; do not rank accepted flags.",
        "poses": sorted(poses, key=lambda pose: pose["image"]),
    }
    write_json(output / "audit.json", audit)
    env = {**os.environ, "PYTHONPATH": str(evaluator_root.resolve())}
    with (output / "evaluator.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nht_pipeline.sfm.metrics",
                "--model",
                str(refreshed_model),
                "--image-dir",
                str(image_dir),
                "--input-image-count",
                str(len(dimensions)),
                "--output",
                str(output / "metrics.json"),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    metrics = json.loads((output / "metrics.json").read_text())
    json.dumps(metrics, allow_nan=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate(
        args.model, args.manifest, args.image_dir, args.evaluator_root, args.output
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "registered_images",
                    "input_images",
                    "sparse_points",
                    "p95_reprojection_error_px",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
