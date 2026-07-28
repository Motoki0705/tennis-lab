"""Publish and strictly load immutable ground-line map artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.artifacts.common import (
    artifact_fingerprint,
    validate_artifact_id,
    validate_sha256,
)
from src.synthetic_data_generation.alignment.scene_provider.bundle import sha256_file

GROUND_LINE_MAP_SCHEMA = "ground_line_map_v1"
_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "evidence_sum": np.dtype(np.float32),
    "weight_sum": np.dtype(np.float32),
    "view_count": np.dtype(np.uint16),
    "mean_probability": np.dtype(np.float32),
}


def publish_ground_line_map_artifact(
    payload: dict[str, Any],
    *,
    arrays: dict[str, NDArray[Any]],
    output_dir: Path,
) -> Path:
    """Atomically publish a fingerprinted manifest, arrays, and preview."""
    _validate_payload_core(payload)
    _validate_arrays(arrays)
    artifact_id = validate_artifact_id(
        payload.get("artifact_id"),
        artifact_type="ground-line",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_id}.",
            suffix=".tmp",
            dir=output_dir,
        )
    )
    try:
        arrays_path = temporary_dir / "arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        preview_path = temporary_dir / "aggregate_evidence.png"
        preview = _render_ground_line_preview(
            np.asarray(arrays["evidence_sum"], dtype=np.float32),
            np.asarray(arrays["view_count"], dtype=np.uint16),
        )
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Failed to write ground-line preview: {preview_path}")
        manifest = dict(payload)
        manifest["files"] = {
            "arrays": _file_record(arrays_path),
            "preview": _file_record(preview_path),
        }
        manifest["artifact_fingerprint"] = artifact_fingerprint(manifest)
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        destination = output_dir / (
            f"{artifact_id}-{manifest['artifact_fingerprint'][:16]}"
        )
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite ground-line artifact: {destination}"
            )
        os.rename(temporary_dir, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def load_ground_line_map_artifact(
    path: Path,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Strict-load and hash-verify a published ground-line map."""
    root = path.resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Ground-line manifest must be a JSON object.")
    manifest = dict(raw)
    _validate_payload_core(manifest)
    fingerprint = validate_sha256(
        manifest.get("artifact_fingerprint"),
        name="ground-line artifact_fingerprint",
    )
    expected = artifact_fingerprint(manifest)
    if fingerprint != expected:
        raise ValueError(
            "Ground-line artifact fingerprint mismatch: "
            f"declared {fingerprint}, computed {expected}."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"arrays", "preview"}:
        raise ValueError("Ground-line manifest files are invalid.")
    for record_value in files.values():
        if not isinstance(record_value, dict):
            raise ValueError("Ground-line file record must be an object.")
        relative_path = record_value.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).name != relative_path
        ):
            raise ValueError("Ground-line artifact paths must be plain file names.")
        file_path = root / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing ground-line artifact file: {file_path}")
        if file_path.stat().st_size != record_value.get("size_bytes"):
            raise ValueError(f"Ground-line artifact size mismatch: {file_path}")
        expected_sha256 = validate_sha256(
            record_value.get("sha256"),
            name=f"ground-line file {relative_path} sha256",
        )
        if sha256_file(file_path) != expected_sha256:
            raise ValueError(f"Ground-line artifact hash mismatch: {file_path}")
    arrays_record = files["arrays"]
    arrays_path = root / str(arrays_record["relative_path"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _validate_arrays(arrays)
    return manifest, arrays


def _validate_payload_core(payload: dict[str, Any]) -> None:
    if payload.get("schema") != GROUND_LINE_MAP_SCHEMA:
        raise ValueError(f"Unsupported ground-line schema: {payload.get('schema')!r}.")
    validate_artifact_id(payload.get("artifact_id"), artifact_type="ground-line")
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "provider",
        "split",
        "detector",
        "ground_plane",
        "projection",
        "records",
        "summary",
        "provenance",
    }
    optional = {"files", "artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Ground-line manifest keys do not match v1 schema.")
    split = payload.get("split")
    if (
        not isinstance(split, dict)
        or split.get("holdout_inference_status") != "not_run"
    ):
        raise ValueError("Ground-line holdout inference status must remain 'not_run'.")
    fit_ids = split.get("fit_camera_ids")
    holdout_ids = split.get("holdout_camera_ids")
    if (
        not isinstance(fit_ids, list)
        or not isinstance(holdout_ids, list)
        or not fit_ids
        or not holdout_ids
        or set(fit_ids).intersection(holdout_ids)
    ):
        raise ValueError("Ground-line fit/holdout camera ids must be disjoint.")
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or [record.get("camera_id") for record in records if isinstance(record, dict)]
        != fit_ids
    ):
        raise ValueError("Ground-line records must exactly match fit_camera_ids.")


def _validate_arrays(arrays: dict[str, NDArray[Any]]) -> None:
    if set(arrays) != set(_ARRAY_DTYPES):
        raise ValueError(f"Ground-line arrays must be exactly {sorted(_ARRAY_DTYPES)}.")
    shapes = {np.asarray(value).shape for value in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes), ())) != 2:
        raise ValueError("All ground-line arrays must share one 2D raster shape.")
    for name, expected_dtype in _ARRAY_DTYPES.items():
        array = np.asarray(arrays[name])
        if array.dtype != expected_dtype:
            raise ValueError(
                f"Ground-line array {name} must have dtype {expected_dtype}, "
                f"got {array.dtype}."
            )
        if not np.isfinite(array).all() or bool(np.any(array < 0)):
            raise ValueError(
                f"Ground-line array {name} must be finite and non-negative."
            )
    expected_mean = np.divide(
        arrays["evidence_sum"],
        arrays["weight_sum"],
        out=np.zeros_like(arrays["evidence_sum"]),
        where=arrays["weight_sum"] > 0.0,
    )
    if not np.allclose(arrays["mean_probability"], expected_mean, atol=1.0e-6):
        raise ValueError("Ground-line mean_probability does not match evidence/weight.")


def _render_ground_line_preview(
    evidence_sum: NDArray[np.float32],
    view_count: NDArray[np.uint16],
) -> NDArray[np.uint8]:
    evidence = np.asarray(evidence_sum, dtype=np.float32)
    support = np.asarray(view_count, dtype=np.uint16)
    if evidence.shape != support.shape or evidence.ndim != 2:
        raise ValueError("Preview inputs must be same-shape 2D arrays.")
    positive = evidence[evidence > 0.0]
    scale = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
    normalized = np.clip(
        np.log1p(evidence) / np.log1p(max(scale, 1.0e-6)),
        0.0,
        1.0,
    )
    intensity = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    colored[support == 0] = 0
    return np.flipud(colored)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
