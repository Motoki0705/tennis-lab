"""Publish an export-bound single/multi-person PLCS Gaussian scene plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.plcs.artifacts.scene_plan import (
    POSE_IDS,
    build_person_schedule,
)
from src.synthetic_data_generation.scene_contract import (
    ArtifactRef,
    SceneCamera,
    load_scene_contract,
)

SCHEMA = "tennis_plcs_gaussian_scene_plan_v1"
P4_SCHEMA = "plcs_avatar_p4_acceptance_report_v1"
NHT_SCHEMA = "plcs_avatar_nht_fit_and_pose_render_v1"
EXPORT_SCHEMA = "cycle09_export_reload_verification_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_uri_path(uri: object) -> Path:
    if not isinstance(uri, str):
        raise ValueError("Artifact URI must be a string.")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Only local file artifacts are supported: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _verify_nht(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != NHT_SCHEMA:
        raise ValueError("Unsupported PLCS NHT avatar manifest.")
    fingerprint = manifest.get("content_fingerprint")
    unsigned = dict(manifest)
    del unsigned["content_fingerprint"]
    if fingerprint != _canonical_sha256(unsigned):
        raise ValueError("PLCS NHT avatar fingerprint differs.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PLCS NHT avatar file inventory is missing.")
    for relative, reference in files.items():
        candidate = (root / relative).resolve()
        if (
            not isinstance(reference, dict)
            or not candidate.is_relative_to(root)
            or not candidate.is_file()
            or _sha256(candidate) != reference.get("sha256")
            or candidate.stat().st_size != reference.get("size_bytes")
        ):
            raise ValueError(f"PLCS NHT avatar file changed: {relative}.")
    return manifest


def _verify_background_source(
    composition: dict[str, Any],
    contract_artifacts: tuple[ArtifactRef, ...],
    *,
    bundle_fingerprint: str,
    scene_fingerprint: str,
) -> None:
    scene_source = composition.get("scene_source")
    if not isinstance(scene_source, dict):
        raise ValueError("Background composition scene source is missing.")
    provider_path = _file_uri_path(scene_source.get("uri"))
    if (
        not provider_path.is_file()
        or _sha256(provider_path) != scene_source.get("sha256")
        or provider_path.stat().st_size != scene_source.get("size_bytes")
    ):
        raise ValueError("Background composition provider bundle changed.")
    provider = json.loads(provider_path.read_text())
    expected_sources = [
        {
            "artifact_id": artifact.artifact_id,
            "uri": artifact.uri,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in contract_artifacts
    ]
    if (
        provider.get("bundle_fingerprint") != bundle_fingerprint
        or provider.get("scene_fingerprint") != scene_fingerprint
        or provider.get("source_artifacts") != expected_sources
    ):
        raise ValueError("Background provider differs from the verified scene export.")


def _project(
    camera: SceneCamera,
    scene_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_to_scene = np.asarray(camera.camera_to_scene).reshape(4, 4)
    scene_to_camera = np.linalg.inv(camera_to_scene)
    homogeneous = np.concatenate(
        (scene_points, np.ones((*scene_points.shape[:-1], 1))),
        axis=-1,
    )
    camera_points = np.einsum("ij,...j->...i", scene_to_camera, homogeneous)[..., :3]
    depth = camera_points[..., 2]
    intrinsics = np.asarray(camera.intrinsics).reshape(3, 3)
    projected = np.einsum("ij,...j->...i", intrinsics, camera_points)
    uv = projected[..., :2] / projected[..., 2:3]
    visible = (
        (depth > 0.0)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] < camera.width)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] < camera.height)
    )
    return uv, depth, visible


def _select_camera(
    cameras: tuple[SceneCamera, ...],
    centers_scene: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, camera in enumerate(cameras):
        uv, depth, visible = _project(camera, centers_scene)
        if not visible.all():
            continue
        normalized_margin = np.minimum.reduce(
            (
                uv[..., 0] / camera.width,
                (camera.width - 1.0 - uv[..., 0]) / camera.width,
                uv[..., 1] / camera.height,
                (camera.height - 1.0 - uv[..., 1]) / camera.height,
            )
        )
        candidates.append((float(normalized_margin.min()), index, uv, depth, visible))
    if not candidates:
        raise RuntimeError("No captured camera contains all scheduled person centers.")
    margin, index, uv, depth, visible = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    return index, uv, depth, visible, margin


def _court_from_asset_rotation(yaw: float) -> NDArray[np.float64]:
    # SMPL-X: +x right, +y up, +z completes the right-handed frame.
    # Court: +x right sideline, +y far baseline, +z up.
    base = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = rotation @ base
    if not np.isclose(np.linalg.det(result), 1.0, atol=1.0e-10):
        raise RuntimeError("SMPL-X to court rotation is not proper.")
    return cast(NDArray[np.float64], result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--export-verification", type=Path, required=True)
    parser.add_argument("--p4-acceptance", type=Path, required=True)
    parser.add_argument("--avatar-nht", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    contract_path = args.scene_contract.resolve()
    export_path = args.export_verification.resolve()
    p4_path = args.p4_acceptance.resolve()
    nht_root = args.avatar_nht.resolve()
    composition_path = args.background_composition.resolve()
    for path in (contract_path, export_path, p4_path, composition_path):
        if not path.is_file():
            raise SystemExit(f"Required artifact is missing: {path}")

    contract = load_scene_contract(contract_path)
    export = json.loads(export_path.read_text())
    if (
        export.get("schema") != EXPORT_SCHEMA
        or export.get("all_declared_files_verified") is not True
        or export.get("camera_count") != len(contract.cameras)
        or export.get("scene_fingerprint") != contract.scene_fingerprint
    ):
        raise RuntimeError("Export reload verification differs from scene contract.")
    p4 = json.loads(p4_path.read_text())
    if (
        p4.get("schema") != P4_SCHEMA
        or p4.get("status") != "passed"
        or p4.get("p4_complete") is not True
    ):
        raise RuntimeError("P4 acceptance is not passed.")
    nht = _verify_nht(nht_root)
    if p4["inputs"]["nht_manifest_sha256"] != _sha256(nht_root / "manifest.json"):
        raise RuntimeError("P4 report does not reference the selected avatar NHT run.")
    composition = json.loads(composition_path.read_text())
    if not isinstance(composition, dict):
        raise ValueError("Background composition must be a JSON object.")
    _verify_background_source(
        composition,
        contract.artifacts,
        bundle_fingerprint=export["bundle_fingerprint"],
        scene_fingerprint=contract.scene_fingerprint,
    )
    if (
        composition.get("background", {}).get("appearance_space_sha256")
        != nht["target_appearance"]["appearance_space_sha256"]
    ):
        raise RuntimeError("Background composition differs from avatar space.")

    schedule = build_person_schedule(
        mode=args.mode,
        seed=args.seed,
    )
    pose_assets: list[dict[str, Any]] = []
    ground_offsets: list[float] = []
    for index, pose_id in enumerate(POSE_IDS):
        tensor_path = nht_root / "poses" / f"{index:03d}-{pose_id}-nht.pt"
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "means" not in payload:
            raise ValueError(f"Invalid pose tensor pack: {tensor_path}")
        means = payload["means"]
        features = payload.get("features")
        if (
            not isinstance(means, torch.Tensor)
            or tuple(means.shape) != (4096, 3)
            or not isinstance(features, torch.Tensor)
            or tuple(features.shape) != (4096, 48)
        ):
            raise ValueError(f"Unexpected pose means: {tensor_path}")
        ground_offsets.append(float(-means[:, 1].min()))
        pose_assets.append(
            {
                "pose_index": index,
                "pose_id": pose_id,
                "uri": tensor_path.resolve().as_uri(),
                "sha256": _sha256(tensor_path),
                "size_bytes": tensor_path.stat().st_size,
                "gaussian_count": 4096,
                "feature_dim": 48,
            }
        )

    scene_from_court = contract.alignment.scene_from_court
    scene_rotation = np.asarray(scene_from_court.rotation).reshape(3, 3)
    scene_translation = np.asarray(scene_from_court.translation)
    frame_count = schedule.frame_count
    person_count = schedule.person_count
    matrices: NDArray[np.float64] = np.empty(
        (frame_count, person_count, 4, 4),
        dtype=np.float64,
    )
    centers_court = schedule.positions_court_m.copy()
    centers_court[..., 2] = 0.9
    centers_scene = scene_from_court.apply(centers_court)
    for frame in range(frame_count):
        for person in range(person_count):
            pose_index = int(schedule.pose_indices[frame, person])
            court_rotation = _court_from_asset_rotation(
                float(schedule.yaw_radians[frame, person])
            )
            court_translation = schedule.positions_court_m[frame, person].copy()
            court_translation[2] = ground_offsets[pose_index]
            matrix = np.eye(4)
            matrix[:3, :3] = scene_from_court.scale * scene_rotation @ court_rotation
            matrix[:3, 3] = (
                scene_from_court.scale * scene_rotation @ court_translation
                + scene_translation
            )
            matrices[frame, person] = matrix

    camera_index, uv, depth, visible, minimum_margin = _select_camera(
        contract.cameras,
        centers_scene,
    )
    camera = contract.cameras[camera_index]
    arrays: dict[str, np.ndarray] = {
        "instance_ids": schedule.instance_ids,
        "positions_court_m": schedule.positions_court_m,
        "velocities_court_mps": schedule.velocities_court_mps,
        "yaw_radians": schedule.yaw_radians,
        "pose_indices": schedule.pose_indices,
        "present": schedule.present,
        "scene_from_asset": matrices,
        "camera_uv": uv,
        "camera_depth": depth,
        "camera_geometric_visible": visible,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        file_refs = {
            path.name: {
                "relative_path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        unsigned = {
            "schema": SCHEMA,
            "mode": schedule.mode,
            "seed": schedule.seed,
            "fps": schedule.fps,
            "frame_count": frame_count,
            "person_count": person_count,
            "schedule_fingerprint": schedule.schedule_fingerprint,
            "scene": {
                "scene_id": contract.scene_id,
                "scene_fingerprint": contract.scene_fingerprint,
                "contract_uri": contract_path.as_uri(),
                "contract_sha256": _sha256(contract_path),
                "accepted_alignment_id": contract.alignment.alignment_id,
                "scene_from_court": scene_from_court.to_dict(),
            },
            "export_verification": {
                "uri": export_path.as_uri(),
                "sha256": _sha256(export_path),
                "bundle_fingerprint": export["bundle_fingerprint"],
            },
            "p4_acceptance": {
                "uri": p4_path.as_uri(),
                "sha256": _sha256(p4_path),
                "content_fingerprint": p4["content_fingerprint"],
            },
            "background_composition": {
                "uri": composition_path.as_uri(),
                "sha256": _sha256(composition_path),
                "composition_fingerprint": composition["composition_fingerprint"],
            },
            "avatar_nht_manifest": {
                "uri": (nht_root / "manifest.json").as_uri(),
                "sha256": _sha256(nht_root / "manifest.json"),
                "content_fingerprint": nht["content_fingerprint"],
                "appearance_space_sha256": nht["target_appearance"][
                    "appearance_space_sha256"
                ],
            },
            "pose_assets": pose_assets,
            "identities": [
                {
                    "identity_id": schedule.identity_ids[index],
                    "instance_id": int(schedule.instance_ids[index]),
                }
                for index in range(person_count)
            ],
            "camera": {
                **camera.to_dict(),
                "camera_index": camera_index,
                "selection_method": "all-centers-visible-max-min-normalized-margin-v1",
                "minimum_normalized_margin": minimum_margin,
            },
            "coordinate_semantics": {
                "positions_court_m": "ground-footprint:+x_right,+y_far,+z_zero",
                "avatar_axes": "smplx:+x_right,+y_up,+z_right_handed",
                "court_from_avatar_base": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                "ground_offsets_m_by_pose": ground_offsets,
            },
            "label_completeness": {
                "identity": True,
                "instance_id": True,
                "position": True,
                "rotation": True,
                "pose": True,
                "presence": True,
                "camera_projection": True,
                "visibility_semantics": (
                    "geometric center only; exact occlusion labels require render AOV"
                ),
            },
            "metrics": {
                "maximum_speed_mps": float(
                    np.linalg.norm(
                        schedule.velocities_court_mps[..., :2], axis=-1
                    ).max()
                ),
                "minimum_person_separation_m": (
                    None
                    if person_count == 1
                    else float(
                        np.linalg.norm(
                            schedule.positions_court_m[:, 0, :2]
                            - schedule.positions_court_m[:, 1, :2],
                            axis=-1,
                        ).min()
                    )
                ),
                "geometrically_visible_center_count": int(visible.sum()),
                "expected_visible_center_count": int(visible.size),
            },
            "files": file_refs,
        }
        manifest = {
            **unsigned,
            "plan_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    print(f"camera_id={camera.camera_id}")
    print(f"plan_fingerprint={manifest['plan_fingerprint']}")
