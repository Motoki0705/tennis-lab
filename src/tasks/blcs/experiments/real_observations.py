"""Fixed Meiji observation diagnostics without independent 3D ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.blcs.physics.rollout import reconstruct_segments
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import denormalize_court_position

from .evaluation import (
    file_sha256,
    fit_diagnostic,
    geometric_baseline,
    load_binding,
    predict_windows,
    save_json,
)


def load_observations(
    root: Path,
) -> tuple[dict[str, Any], tuple[PinholeCamera, ...], dict[str, Any]]:
    scene = json.loads((root / "scene.json").read_text())
    store = ClipStore(root, scene["source"])
    references = store.references()
    alignment = store.load(
        references["camera_alignment"], ArtifactCodec(CameraAlignmentOutput)
    )
    geometry = alignment.geometry
    if geometry is None:
        raise ValueError("Meiji diagnostics require saved camera alignment")
    provenance = geometry.document["court_reference_provenance"]
    for key in ("reference_from_physical", "physical_from_reference"):
        if not np.array_equal(provenance[key], np.eye(3)):
            raise ValueError(
                "This fixed protocol requires an explicitly physical aligned frame"
            )
    videos = scene["source"]["videos"]
    if [video["camera_id"] for video in videos] != list(geometry.camera_ids):
        raise ValueError("Saved camera and observation order mismatch")
    if len({(v["fps"], v["num_frames"]) for v in videos}) != 1:
        raise ValueError("Meiji cameras must share the recorded clock/length")
    reference = references["ball_triangulation"]
    descriptor = store.descriptor(reference)
    ball = unpack_value(
        descriptor["payload"], (root / reference.path).parent, descriptor["arrays"]
    )["ball"]
    if ball["status"] != "ok":
        raise ValueError("Saved ball observations are unavailable")
    # Match the synthetic 30 Hz context as closely as the observed clock permits.
    # No temporal interpolation of 2D observations and no 3D input are used.
    indices = np.arange(0, videos[0]["num_frames"], 2)
    fps = float(videos[0]["fps"]) / 2
    sizes = np.array([[v["width"], v["height"]] for v in videos])
    visible = np.asarray(ball["visibility"][:, indices]).copy()
    uv = np.where(visible[..., None], ball["uv_px"][:, indices] / sizes[:, None], 0)
    points = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14]
    court_uv, court_vis = [], []
    for camera, size in zip(geometry.cameras, sizes, strict=True):
        projected, front = camera.project(points)
        projected /= size
        supported = front & ((projected >= 0) & (projected <= 1)).all(-1)
        court_uv.append(np.where(supported[:, None], projected, 0))
        court_vis.append(supported)
    length = len(indices)
    batch: dict[str, Any] = {
        "ball_uv": torch.tensor(uv[None], dtype=torch.float32),
        "ball_vis": torch.tensor(visible[None]),
        "court_kp": torch.tensor(
            np.stack(court_uv)[None, :, None].repeat(length, 2), dtype=torch.float32
        ),
        "court_vis": torch.tensor(np.stack(court_vis)[None, :, None].repeat(length, 2)),
        "padding_mask": torch.zeros(1, len(videos), length, dtype=torch.bool),
        "camera_R": torch.tensor(
            np.stack([c.rotation for c in geometry.cameras])[None], dtype=torch.float32
        ),
        "camera_C": torch.tensor(
            np.stack([c.center for c in geometry.cameras])[None], dtype=torch.float32
        ),
    }
    for key, values in {
        "camera_f": [c.intrinsic[0, 0] for c in geometry.cameras],
        "camera_cx": [c.intrinsic[0, 2] for c in geometry.cameras],
        "camera_cy": [c.intrinsic[1, 2] for c in geometry.cameras],
        "camera_w": sizes[:, 0],
        "camera_h": sizes[:, 1],
    }.items():
        batch[key] = torch.tensor(np.asarray(values)[None], dtype=torch.float32)
    metadata = {
        "schema": "blcs.real_observation_diagnostic.v1",
        "source_store": str(root),
        "source_index_sha256": file_sha256(root / "scene.json"),
        "camera_artifact_sha256": references["camera_alignment"].sha256,
        "observation_artifact_sha256": reference.sha256,
        "source": scene["source"],
        "camera_geometry": geometry.document,
        "source_frame_indices": indices.tolist(),
        "evaluation_fps": fps,
        "court_input": "physical_v1 first14 explicitly projected from saved aligned calibration",
        "independent_3d_ground_truth": False,
        "limitations": [
            "Model-assisted fixed 2D annotations; not independent 3D labels",
            "Approximate single-plane camera calibration without distortion correction",
            "Court inputs are calibration-derived rather than raw detector observations",
            "Every second observed frame gives 29.97003 Hz versus 30 Hz training",
            "ODE fits use fixed 32-frame windows; unknown real impacts may be crossed",
            "Reprojection or small fit residual alone does not establish 3D accuracy",
        ],
    }
    return batch, geometry.cameras, metadata


def observation_metrics(
    position: np.ndarray,
    batch: dict[str, Any],
    cameras: tuple[PinholeCamera, ...],
    fps: float,
) -> dict[str, Any]:
    visible = batch["ball_vis"][0].numpy().astype(bool)
    size = np.stack([batch["camera_w"][0].numpy(), batch["camera_h"][0].numpy()], -1)
    uv = batch["ball_uv"][0].numpy() * size[:, None]
    errors, evaluable = [], 0
    for i, camera in enumerate(cameras):
        projected, front = camera.project(position)
        supported = visible[i] & front
        errors.extend(np.linalg.norm((projected - uv[i])[supported], axis=-1).tolist())
        evaluable += int(supported.sum())
    error = np.array(errors)
    return {
        "reprojection_count": len(error),
        "eligible_observation_count": int(visible.sum()),
        "reprojection_rmse_px": float(np.sqrt(np.mean(error**2)))
        if len(error)
        else None,
        "reprojection_p95_px": float(np.quantile(error, 0.95)) if len(error) else None,
        "reprojection_evaluable_rate": evaluable / int(visible.sum())
        if visible.any()
        else None,
        "frame_count": len(position),
        "all_view_missing_frames": int((~visible.any(0)).sum()),
        "view_count_frames": {
            str(n): int((visible.sum(0) == n).sum()) for n in range(len(cameras) + 1)
        },
        "below_ground_rate": float(np.mean(position[:, 2] < -0.001)),
        "difference_acceleration_rms_mps2": float(
            np.sqrt(
                np.mean(np.sum((np.diff(position, n=2, axis=0) * fps**2) ** 2, axis=-1))
            )
        ),
        "fit": fit_diagnostic(
            position,
            np.arange(len(position)) // 32,
            np.array([1 / fps / 4, fps * 4, fps, 4]),
            9.8,
        ),
    }


def evaluate_real(
    *,
    source: Path,
    data_root: Path,
    output: Path,
    checkpoint: Path | None,
    historical: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    batch, cameras, metadata = load_observations(source)
    fps = metadata["evaluation_fps"]
    if checkpoint is None:
        prediction, geometry = geometric_baseline(batch)
        metadata["geometric_interpolation"] = geometry
    else:
        binding, _ = load_binding(checkpoint, data_root, historical=historical)
        binding.model.to(device).eval()
        prediction = predict_windows(binding, batch, device=device)
    position = denormalize_court_position(prediction.position)[0].numpy()
    report = {
        **metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "direct": observation_metrics(position, batch, cameras, fps),
    }
    arrays = {key: value.numpy() for key, value in batch.items()}
    arrays["position_m"] = position
    if prediction.velocity is not None:
        arrays["velocity_mps"] = prediction.velocity[0].numpy() * 11.885
    if prediction.physics_parameters is not None:
        assert prediction.velocity is not None and prediction.event_logits is not None
        events = prediction.event_logits[..., 0] >= 0
        p, _ = reconstruct_segments(
            torch.from_numpy(position)[None],
            prediction.velocity * 11.885,
            prediction.physics_parameters,
            events,
            torch.ones_like(events),
            torch.tensor([[1 / fps / 4, fps * 4, fps, 4]], dtype=torch.float64),
            gravity=9.8,
        )
        finite = bool(torch.isfinite(p).all())
        report["integration_failed"] = not finite
        report["predicted_event_count"] = int(events[:, 1:].sum())
        arrays["integrated_position_m"] = p[0].numpy()
        arrays["predicted_event"] = events[0].numpy()
        arrays["physics_parameters"] = prediction.physics_parameters[0].numpy()
        if finite:
            report["integrated_predicted"] = observation_metrics(
                p[0].numpy(), batch, cameras, fps
            )
    np.savez_compressed(output / "real_predictions.npz", **arrays)
    save_json(output / "evaluation.json", report)
    return report
