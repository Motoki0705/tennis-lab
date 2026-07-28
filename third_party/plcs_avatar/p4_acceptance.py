"""Publish a strict P4 acceptance report from two independent PLCS trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCHEMA = "plcs_avatar_p4_acceptance_report_v1"
GEOMETRY_SCHEMA = "plcs_smplx_gaussian_asset_fixture_v1"
NHT_SCHEMA = "plcs_avatar_nht_fit_and_pose_render_v1"
GAUSSIANAVATAR_COMMIT = "d981c62238ef64e89dcc04719d2ebbb4758b080a"
HUGS_COMMIT = "b65721a5946771053e4f1d0d68d06199bc1d8c07"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(root: Path, *, schema: str) -> dict[str, object]:
    root = root.resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != schema:
        raise ValueError(f"Unsupported manifest schema at {path}.")
    fingerprint = manifest.get("content_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError(f"Missing content fingerprint at {path}.")
    unsigned = dict(manifest)
    del unsigned["content_fingerprint"]
    if _canonical_sha256(unsigned) != fingerprint:
        raise ValueError(f"Content fingerprint differs at {path}.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Missing file inventory at {path}.")
    for relative, reference in files.items():
        if not isinstance(relative, str) or not isinstance(reference, dict):
            raise ValueError(f"Invalid file reference at {path}.")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Unsafe or missing file {relative}.")
        expected_sha = reference.get("sha256")
        expected_size = reference.get("size_bytes")
        if (
            _sha256(candidate) != expected_sha
            or candidate.stat().st_size != expected_size
        ):
            raise ValueError(f"Referenced file changed: {candidate}.")
    return manifest


def _load_tensor_pack(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "means",
        "quats",
        "scales",
        "opacities",
        "features",
        "instance_ids",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"Unexpected NHT tensor pack at {path}.")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError(f"Non-tensor value in {path}.")
    return payload


def _metrics(manifest: dict[str, object]) -> dict[str, object]:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("NHT manifest metrics are missing.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-repeat", type=Path, required=True)
    parser.add_argument("--nht", type=Path, required=True)
    parser.add_argument("--nht-repeat", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    geometry_root = args.geometry.resolve()
    geometry_repeat_root = args.geometry_repeat.resolve()
    nht_root = args.nht.resolve()
    nht_repeat_root = args.nht_repeat.resolve()
    geometry = _load_manifest(geometry_root, schema=GEOMETRY_SCHEMA)
    geometry_repeat = _load_manifest(
        geometry_repeat_root,
        schema=GEOMETRY_SCHEMA,
    )
    nht = _load_manifest(nht_root, schema=NHT_SCHEMA)
    nht_repeat = _load_manifest(nht_repeat_root, schema=NHT_SCHEMA)

    pins = json.loads(args.pins.read_text())
    repositories = pins.get("repositories") if isinstance(pins, dict) else None
    if not isinstance(repositories, list):
        raise ValueError("PLCS upstream pins are invalid.")
    pinned_commits = {
        item.get("method"): item.get("commit")
        for item in repositories
        if isinstance(item, dict)
    }
    if pinned_commits.get("GaussianAvatar") != GAUSSIANAVATAR_COMMIT:
        raise ValueError("GaussianAvatar official commit is not pinned.")
    if pinned_commits.get("HUGS") != HUGS_COMMIT:
        raise ValueError("HUGS official commit is not pinned.")
    research_text = args.research.read_text()
    for required in (GAUSSIANAVATAR_COMMIT, HUGS_COMMIT, "SplattingAvatar"):
        if required not in research_text:
            raise ValueError(f"Research record omits {required}.")

    geometry_reproducible = (
        geometry["content_fingerprint"] == geometry_repeat["content_fingerprint"]
        and (geometry_root / "manifest.json").read_bytes()
        == (geometry_repeat_root / "manifest.json").read_bytes()
    )
    if not geometry_reproducible:
        raise RuntimeError("SMPL-X Gaussian geometry is not byte-reproducible.")
    geometry_metrics = geometry.get("metrics")
    if not isinstance(geometry_metrics, dict):
        raise ValueError("Geometry metrics are missing.")
    if (
        geometry_metrics.get("all_frames_emitted") != 3
        or geometry_metrics.get("all_frames_finite") is not True
        or float(geometry_metrics.get("max_p95_attachment_error_mm", np.inf)) > 5.0
    ):
        raise RuntimeError("Geometry control gate failed.")

    packs = (
        _load_tensor_pack(nht_root / "avatar-nht-tensors.pt"),
        _load_tensor_pack(nht_repeat_root / "avatar-nht-tensors.pt"),
    )
    exact_geometry_keys = ("means", "quats", "scales", "opacities", "instance_ids")
    exact_geometry = all(
        torch.equal(packs[0][key], packs[1][key]) for key in exact_geometry_keys
    )
    feature_difference = torch.abs(packs[0]["features"] - packs[1]["features"])
    feature_max_abs = float(feature_difference.max())
    feature_mean_abs = float(feature_difference.mean())

    metrics = (_metrics(nht), _metrics(nht_repeat))
    psnr_values = tuple(float(item["mean_validation_psnr_db"]) for item in metrics)
    psnr_delta = abs(psnr_values[0] - psnr_values[1])
    for item in metrics:
        if (
            item.get("status") != "passed"
            or item.get("native_nht_render") is not True
            or item.get("rgb_overlay_used") is not False
            or item.get("standard_3dgs_features_imported") is not False
            or item.get("all_pose_frames_emitted") != 3
            or item.get("dropped_pose_frames") != []
            or item.get("dropped_joint_indices") != []
        ):
            raise RuntimeError("NHT pose-control contract gate failed.")
    if min(psnr_values) < 25.0:
        raise RuntimeError("Held-out avatar PSNR gate failed.")

    image_names = sorted(
        path.name for path in (nht_root / "poses").glob("*camera-*.png")
    )
    if len(image_names) != 6:
        raise RuntimeError("Expected exactly six pose-view RGB renders.")
    maximum_lsb = 0
    mean_lsb: list[float] = []
    for name in image_names:
        first = np.asarray(Image.open(nht_root / "poses" / name)).astype(np.int16)
        second = np.asarray(
            Image.open(nht_repeat_root / "poses" / name)
        ).astype(np.int16)
        difference = np.abs(first - second)
        maximum_lsb = max(maximum_lsb, int(difference.max()))
        mean_lsb.append(float(difference.mean()))
    image_mean_lsb_max = max(mean_lsb)

    tolerance = {
        "feature_max_abs_at_most": 0.02,
        "feature_mean_abs_at_most": 0.002,
        "validation_psnr_delta_db_at_most": 0.1,
        "render_max_lsb_at_most": 1,
        "render_mean_lsb_at_most": 0.01,
    }
    repeat_within_tolerance = (
        exact_geometry
        and feature_max_abs <= tolerance["feature_max_abs_at_most"]
        and feature_mean_abs <= tolerance["feature_mean_abs_at_most"]
        and psnr_delta <= tolerance["validation_psnr_delta_db_at_most"]
        and maximum_lsb <= tolerance["render_max_lsb_at_most"]
        and image_mean_lsb_max <= tolerance["render_mean_lsb_at_most"]
    )
    if not repeat_within_tolerance:
        raise RuntimeError("Repeated NHT optimization exceeds measured tolerance.")

    unsigned = {
        "schema": SCHEMA,
        "status": "passed",
        "p4_complete": True,
        "selected_method": "GaussianAvatar-style fixed SMPL-X query LBS",
        "comparative_method": "HUGS-style top-k SMPL transform blending",
        "official_commits": {
            "GaussianAvatar": GAUSSIANAVATAR_COMMIT,
            "HUGS": HUGS_COMMIT,
        },
        "inputs": {
            "geometry_manifest_sha256": _sha256(
                geometry_root / "manifest.json"
            ),
            "geometry_repeat_manifest_sha256": _sha256(
                geometry_repeat_root / "manifest.json"
            ),
            "nht_manifest_sha256": _sha256(nht_root / "manifest.json"),
            "nht_repeat_manifest_sha256": _sha256(
                nht_repeat_root / "manifest.json"
            ),
            "research_sha256": _sha256(args.research),
            "pins_sha256": _sha256(args.pins),
        },
        "geometry": {
            "byte_reproducible": geometry_reproducible,
            "content_fingerprint": geometry["content_fingerprint"],
            "gaussian_count": geometry["construction"]["gaussian_count"],  # type: ignore[index]
            "joint_count": geometry["construction"]["joint_count"],  # type: ignore[index]
            "max_p95_attachment_error_mm": geometry_metrics[
                "max_p95_attachment_error_mm"
            ],
        },
        "nht": {
            "native_render": True,
            "rgb_overlay_used": False,
            "standard_3dgs_features_imported": False,
            "validation_psnr_db": psnr_values,
            "validation_psnr_delta_db": psnr_delta,
            "exact_geometry_across_repeats": exact_geometry,
            "feature_max_abs_difference": feature_max_abs,
            "feature_mean_abs_difference": feature_mean_abs,
            "render_max_lsb_difference": maximum_lsb,
            "render_max_mean_lsb_difference": image_mean_lsb_max,
            "repeat_within_tolerance": repeat_within_tolerance,
            "tolerance": tolerance,
        },
        "unsupported_control_policy": {
            "dropped_joint_indices": [],
            "dropped_pose_frames": [],
            "coco17_inverse": "requires-explicit-IK-never-silent",
        },
    }
    report = {
        **unsigned,
        "content_fingerprint": _canonical_sha256(unsigned),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
