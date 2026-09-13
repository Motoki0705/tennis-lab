"""Paired synthetic-test and production-observation clip inference for PLCS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.geometry.footpoint import footpoint_prior
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tennis_scene.reference_pipeline.reference import (
    build_reference,
    reference_metadata,
)
from src.utils.inference.windowed import blend_windows, window_slices
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def summary(error: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(error.mean()),
        "median": float(np.median(error)),
        "p95": float(np.quantile(error, 0.95)),
    }


def synthetic(module: PLCSLightningModule, args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.baseline_config)
    dataset = SceneDataset(
        scene_dir=args.dataset,
        split_file="test.txt",
        config=cfg,
        seed=1234,
        augment=False,
        reference_camera_id="camera_0",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        collate_fn=collate_plcs_batch,
        shuffle=False,
    )
    records: dict[str, list[np.ndarray]] = {
        k: []
        for k in [
            "position",
            "target",
            "rotation",
            "target_rotation",
            "anchor",
            "anchor_valid",
        ]
    }
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            moved = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=False,
            ):
                decoded, _ = module._forward_from_batch(moved)
            _, valid, anchor = footpoint_prior(
                *[
                    moved[k]
                    for k in [
                        "human_kp",
                        "court_kp",
                        "human_vis",
                        "court_vis",
                        "padding_mask",
                    ]
                ]
            )
            for key, value in [
                ("position", decoded.position),
                ("target", moved["position"]),
                ("rotation", decoded.rotation),
                ("target_rotation", moved["rotation"]),
                ("anchor", anchor),
                ("anchor_valid", valid.any(1)),
            ]:
                records[key].append(value.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"test {i + 1}/{len(loader)}", flush=True)
    data = {k: np.concatenate(v) for k, v in records.items()}
    scale = np.array(COURT_COORD_SCALE_XYZ)
    delta = (data["position"] - data["target"]) * scale
    error = np.linalg.norm(delta, axis=-1)
    xy = np.linalg.norm(delta[..., :2], axis=-1)
    prior_xy = np.linalg.norm(
        (data["anchor"] - data["target"])[..., :2] * scale[:2], axis=-1
    )
    yaw = np.arctan2(data["rotation"][..., 1], data["rotation"][..., 0]) - np.arctan2(
        data["target_rotation"][..., 1], data["target_rotation"][..., 0]
    )
    yaw = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw))) * 180 / np.pi
    correction = np.linalg.norm(
        (data["position"] - data["anchor"])[..., :2] * scale[:2], axis=-1
    )
    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_scene_count": len(dataset),
        "test_seed": 1234,
        "inference_precision": "float32",
        "position_3d_m": summary(error),
        "position_xy_m": summary(xy),
        "angular_error_deg": summary(yaw),
        "position_accuracy_0.5m": float((error < 0.5).mean()),
        "zero_residual_xy_m": summary(prior_xy[data["anchor_valid"].astype(bool)]),
        "anchor_valid_fraction": float(data["anchor_valid"].mean()),
        "strata": {},
    }
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, np.inf)]:
        mask = (prior_xy >= lo) & (prior_xy < hi) & data["anchor_valid"].astype(bool)
        if mask.any():
            metrics["strata"][f"{lo}-{hi}"] = {
                "frames": int(mask.sum()),
                "position_3d_m": summary(error[mask]),
                "position_xy_m": summary(xy[mask]),
                "predicted_correction_xy_m": summary(correction[mask]),
                "zero_correction_fraction_lt_5cm": float(
                    (correction[mask] < 0.05).mean()
                ),
            }
    np.savez_compressed(
        args.output / f"{args.label}_test.npz",
        **data,
        scene_ids=np.array([p.name for p in dataset.scenes]),
    )
    (args.output / f"{args.label}_test_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics, indent=2), flush=True)


def clip_inference(module: PLCSLightningModule, args: argparse.Namespace) -> None:
    clip = json.loads((args.clip / "clip.json").read_text())
    raw = json.loads(
        (args.clip / "annotations/manual_court_kp_result.json").read_text()
    )
    raw_court = np.asarray(raw["keypoints"], dtype=np.float32)
    aligned, selection, context = build_reference(
        clip["camera_ids"],
        raw_court,
        [False, False, True],
        "cam0",
        (clip["width"], clip["height"]),
    )
    scene_path = args.clip / "annotations/tennis_scene/scene.npz"
    with np.load(scene_path) as scene:
        human = scene["human_kp_2d"].copy()
        vis = (scene["human_kp_vis"] >= 0.15).astype(np.float32)
        old_position = scene["player_position"].copy()
        # Confirm frozen observations use the declared reference ordering.
        np.testing.assert_allclose(scene["court_kp"], aligned, atol=1e-6)
    court_vis = np.ones(aligned.shape[:-1], np.float32)
    if not np.all(np.asarray(raw["visibility"]) == 1):
        raise ValueError("This clip comparison requires all court points visible")
    predictor = PLCSPredictor(
        model=module.model, adapter=module.io_adapter, device=torch.device(args.device)
    )
    frames = human.shape[2]
    selected = np.arange(0, frames, 2)
    # Use the established production stride=2, 128/64 windows and triangular blend.
    position_chunks = []
    rotation_chunks = []
    for start, end in window_slices(len(selected), 128, 64):
        indices = selected[start:end]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=False,
        ):
            prediction = predictor.predict_multiview_observations(
                human_kp=human[:, :, indices],
                court_kp=aligned[:, indices],
                human_vis=vis[:, :, indices],
                court_vis=court_vis[:, indices],
                padding_mask=np.zeros(human[:, :, indices].shape[:3], bool),
                court_keypoint_metadata=context,
                court_reference_provenance=(selection.provenance,) * human.shape[0],
                reference_metadata=reference_metadata(
                    selection, human.shape[0], "plcs"
                ),
            )
        position_chunks.append((start, prediction.position_meters.transpose(1, 0, 2)))
        yaw = prediction.yaw_radians
        rotation_chunks.append(
            (start, np.stack((np.cos(yaw), np.sin(yaw)), -1).transpose(1, 0, 2))
        )

    def restore(chunks: list) -> np.ndarray:
        sampled = blend_windows(chunks, len(selected))
        flat = sampled.reshape(len(selected), -1)
        restored = np.stack(
            [
                np.interp(np.arange(frames), selected, flat[:, i])
                for i in range(flat.shape[1])
            ],
            -1,
        )
        return cast(
            np.ndarray, restored.reshape(frames, *sampled.shape[1:]).transpose(1, 0, 2)
        )

    position = restore(position_chunks)
    rotation = restore(rotation_chunks)
    yaw = np.arctan2(rotation[..., 1], rotation[..., 0])
    hip = (human[..., 11, :] + human[..., 12, :]) / 2
    hip_valid = (vis[..., 11] > 0) & (vis[..., 12] > 0)
    pixels = np.array([clip["width"], clip["height"]])
    reprojections = []
    errors = []
    for v, fit in enumerate(context["camera_fits"]):
        r = cv2.Rodrigues(np.array(fit["R"]))[0]
        t = np.array(fit["t"])
        k = np.array(fit["K"])
        uv = cv2.projectPoints(position.reshape(-1, 3), r, t, k, None)[0][:, 0].reshape(
            *position.shape[:2], 2
        )
        reprojections.append(uv)
        errors.append(np.linalg.norm(uv - hip[:, v] * pixels, axis=-1))
    _, prior_valid, prior = footpoint_prior(
        torch.from_numpy(human),
        torch.from_numpy(aligned).unsqueeze(0).expand(human.shape[0], -1, -1, -1, -1),
        torch.from_numpy(vis),
        torch.from_numpy(court_vis).unsqueeze(0).expand(human.shape[0], -1, -1, -1),
        torch.zeros(human.shape[:3], dtype=torch.bool),
    )
    prior_m = prior.numpy() * np.array(COURT_COORD_SCALE_XYZ)
    correction = np.linalg.norm(position[..., :2] - prior_m[..., :2], axis=-1)
    projected = np.stack(reprojections, 1)
    error = np.stack(errors, 1)
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "clip": str(args.clip.resolve()),
        "frames": frames,
        "players": human.shape[0],
        "evaluation": "root projection versus observed COCO hip midpoint; approximate court-only camera fits; no 3D ground truth",
        "root_reprojection_px": summary(error[hip_valid]),
        "predicted_correction_xy_m": summary(correction),
        "zero_correction_fraction_lt_5cm": float((correction < 0.05).mean()),
        "anchor_valid_fraction": float(prior_valid.any(1).float().mean()),
        "per_camera": {
            c: summary(error[:, v][hip_valid[:, v]])
            for v, c in enumerate(clip["camera_ids"])
        },
        "per_player": {
            str(p): summary(error[p][hip_valid[p]]) for p in range(human.shape[0])
        },
        "camera_fits": context["camera_fits"],
        "source_stride": 2,
        "window_size": 128,
        "overlap": 64,
        "old_scene_position_difference_m": summary(
            np.linalg.norm(position - old_position, axis=-1)
        ),
    }
    np.savez_compressed(
        args.output / f"{args.label}_clip.npz",
        position=position,
        yaw=yaw,
        projected_root_px=projected,
        hip_observed_uv=hip,
        hip_valid=hip_valid,
        court=aligned,
        human=human,
        human_vis=vis,
        reprojection_error_px=error,
        prior_position=prior_m,
        prior_valid=prior_valid.any(1).numpy(),
    )
    (args.output / f"{args.label}_clip_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    (args.output / "reference_context.json").write_text(json.dumps(context, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--baseline-config", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--clip", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clip-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
    cfg.training.compile.enabled = False
    module = PLCSLightningModule(cfg)
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device).eval()
    if not args.clip_only:
        synthetic(module, args)
    clip_inference(module, args)


if __name__ == "__main__":
    main()
