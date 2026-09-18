"""Export only supported multiview ball segments for real-domain fine-tuning."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.tennis_scene.dataset_pipeline.build import _import_ball, _observe_court
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.tennis_scene.dataset_pipeline.geometry import (
    TriangulationSettings,
    triangulate_ball,
)
from src.tennis_scene.dataset_pipeline.quality import summarize
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.reference_pipeline.observations import sha256
from src.tennis_scene.reference_pipeline.reference import build_reference
from src.utils.io import save_json_atomic
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)


def supported_windows(
    valid: np.ndarray, *, minimum: int, maximum: int, stride: int
) -> list[np.ndarray]:
    """Never bridge an unsupported 3D label or duplicate windows across splits."""
    if (
        valid.ndim != 1
        or valid.dtype != bool
        or not 0 < minimum <= maximum
        or stride <= 0
    ):
        raise ValueError(
            "Require a boolean timeline and positive ordered window lengths"
        )
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    windows: list[np.ndarray] = []
    for begin, end in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    ):
        if end - begin < minimum:
            continue
        starts = list(range(begin, max(begin + 1, end - maximum + 1), stride))
        if end - starts[-1] > maximum:
            starts.append(end - maximum)
        windows.extend(np.arange(start, min(start + maximum, end)) for start in starts)
    return windows


def export_blcs_training(
    cfg: DictConfig,
    destination: Path,
    *,
    synthetic_source: Path,
    replay_count: int,
    video_splits: dict[str, str],
    settings: TriangulationSettings,
) -> None:
    """Publish atomically; preserve exact FPS and clip/frame/hash provenance."""
    runtime = DatasetBuildConfig.from_config(cfg)
    manifest = load_dataset_manifest(runtime.source)
    if set(video_splits) != {r.video_id for r in manifest.clips.values()} or set(
        video_splits.values()
    ) != {"train", "val", "test"}:
        raise ValueError(
            "Explicit recording-disjoint train/val/test assignments required"
        )
    if destination.exists():
        raise FileExistsError(f"Choose a new BLCS dataset version: {destination}")
    temporary = destination.with_name(destination.name + ".building")
    writer = BLCSDatasetWriter(temporary, court_keypoint_contract="camera_view_v2")
    contract = resolve_court_keypoint_contract("camera_view_v2")
    splits: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    provenance: dict[str, Any] = {}
    sampling = int(cfg.sample_stride)
    for clip_id, record in sorted(manifest.clips.items()):
        clip = ClipManifest.load(runtime.source / record.path)
        observed = runtime.observations / clip_id
        observed.mkdir(parents=True, exist_ok=True)
        _import_ball(runtime, clip, observed)
        kp, _ = _observe_court(cfg, runtime, clip, observed)
        _, _, reference = build_reference(
            clip.camera_ids,
            kp,
            list(cfg.view_half_turns),
            cfg.reference_camera,
            (clip.width, clip.height),
        )
        ball = BallDetectionResult.load(observed / "ball_detection_result.json")
        result = triangulate_ball(
            ball.ball_uv,
            ball.visibility,
            reference["camera_fits"],
            size=(clip.width, clip.height),
            fps=clip.fps,
            settings=settings,
        )
        indices = np.arange(0, clip.num_frames, sampling)
        windows = supported_windows(
            result.valid[indices], minimum=32, maximum=128, stride=64
        )
        provenance[clip_id] = {
            "manifest_sha256": clip.digest(),
            "ball_receipt_sha256": sha256(observed / "ball_import.metadata.json"),
            "court_receipt_sha256": sha256(observed / "court.json"),
            "source_fps": clip.fps,
            "valid_fraction": float(result.valid.mean()),
            "windows": len(windows),
            "reprojection_px": summarize(result.reprojection_px[:, result.valid]),
            "rejections": {
                str(k): int(v)
                for k, v in zip(
                    *np.unique(result.rejection_code, return_counts=True), strict=True
                )
            },
            "split": video_splits[clip.video_id],
        }
        for window_number, window in enumerate(windows):
            frames = indices[window]
            position = torch.from_numpy(result.position[frames])
            fps = clip.fps / sampling
            velocity = (
                torch.from_numpy(
                    np.gradient(position.numpy(), axis=0).astype(np.float32)
                )
                * fps
            )
            cameras = []
            for view, fit in enumerate(reference["camera_fits"]):
                params = {
                    **fit,
                    "C": fit["camera_center_court_m"],
                    "f": fit["K"][0][0],
                    "cx": clip.width / 2,
                    "cy": clip.height / 2,
                    "w": clip.width,
                    "h": clip.height,
                }
                cameras.append(
                    CameraData(
                        params,
                        ball.ball_uv[view, frames],
                        ball.visibility[view, frames],
                        float(ball.visibility[view, frames].mean()),
                        np.pad(kp[view, 0], ((0, 6), (0, 0))),
                        np.r_[np.ones(14, bool), np.zeros(6, bool)],
                        14.0,
                        build_court_view_record(
                            camera_id=f"cam_{view}",
                            camera_center_court_m=params["C"],
                            contract=contract,
                        ),
                    )
                )
            scene_id = f"meiji_{clip.video_id}_{clip.clip_name}_{window_number:03d}"
            scene = BLCSSceneData(
                scene_id,
                -1,
                "unknown",
                0,
                "observed_segment",
                None,
                [],
                position,
                normalize_court_position(position),
                velocity,
                normalize_court_velocity(velocity),
                cameras,
                len(cameras),
                fps,
                clip.fps,
                {"source": "outsource_2d_triangulation", "is_ground_truth": False},
                {},
                1,
            )
            saved = writer.save_scene(scene)
            metadata = json.loads((saved / "meta.json").read_text())
            metadata["label_provenance"] = {
                "clip_id": clip_id,
                "frame_indices": frames.tolist(),
                "is_ground_truth": False,
            }
            save_json_atomic(metadata, saved / "meta.json")
            splits[video_splits[clip.video_id]].append(scene_id)
        print(
            f"{clip_id}: supported={result.valid.mean():.3f}, windows={len(windows)}",
            flush=True,
        )
    rng = np.random.default_rng(runtime.seed)
    candidates = (synthetic_source / "train.txt").read_text().splitlines()
    replay = sorted(rng.choice(candidates, size=replay_count, replace=False).tolist())
    original = json.loads((synthetic_source / "meta.json").read_text())
    records = {r["scene_id"]: r for r in original["scenes"]}
    for scene_id in replay:
        target = writer.scenes_dir / scene_id
        target.mkdir()
        for path in (synthetic_source / "scenes" / scene_id).iterdir():
            if path.is_file():
                os.link(path, target / path.name)
        writer.scene_records.append(records[scene_id])
        writer.scene_counter += 1
        splits["train"].append(scene_id)
    if any(not scenes for scenes in splits.values()):
        raise ValueError("Geometry filtering left an empty split")
    recipe = {
        "generation": {
            "mode": "multi",
            "method": "quality-gated measured 2D triangulation + train-only synthetic replay",
        },
        "is_ground_truth": False,
        "settings": asdict(settings),
        "video_splits": video_splits,
        "clips": provenance,
        "synthetic_source": str(synthetic_source.resolve()),
        "synthetic_meta_sha256": sha256(synthetic_source / "meta.json"),
        "synthetic_replay_ids": replay,
        "seed": runtime.seed,
        "sample_stride": sampling,
        "source_config": OmegaConf.to_container(cfg, resolve=True),
    }
    writer.save_meta_json(config=recipe)
    for name, scenes in splits.items():
        (temporary / f"{name}.txt").write_text("\n".join(scenes) + "\n")
    save_json_atomic(
        {
            "n_scenes": {k: len(v) for k, v in splits.items()},
            "video_splits": video_splits,
            "seed": runtime.seed,
        },
        temporary / "split_info.json",
    )
    save_json_atomic(recipe, temporary / "provenance.json")
    temporary.rename(destination)
