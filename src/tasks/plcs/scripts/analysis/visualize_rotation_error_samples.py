"""Find and visualize PLCS samples with large rotation error.

Usage:
    python -m src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples
    python -m src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples \
        analysis.top_k=10 analysis.clip_half_window=15

Notes:
    - Configuration is loaded from
      `src/tasks/plcs/configs/analyze_rotation_error_samples.yaml` via Hydra.
    - The script scores frame-level yaw error on a split, crops short clips
      around the worst frames, and renders each clip through the PLCS
      visualization orchestration used by `visualize.py`.
    - Reports and generated GIFs are written under `run.output_dir`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from src.tasks.plcs.generate_dataset.io.dataset_io import load_scene
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.visualization.adapters.predict_inputs import build_multiview_inputs
from src.tasks.plcs.visualization.orchestrator import RuntimeConfig, run_visualization
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


def _resolve_device(device_cfg: Any) -> str:
    device = str(device_cfg)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_cameras(raw: Any, num_cameras: int) -> list[int]:
    if raw is None or str(raw).strip() == "" or str(raw).strip() == "all":
        return list(range(num_cameras))
    if isinstance(raw, str):
        return [int(part.strip()) for part in raw.split(",")]
    return [int(v) for v in raw]


def _scene_names(scene_dir: Path, split: str) -> list[str]:
    split_path = scene_dir / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [
        line.strip() for line in split_path.read_text().splitlines() if line.strip()
    ]


def _normalized_rotation(rotation: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    return cast(np.ndarray, rotation / np.clip(norm, 1e-8, None))


def _angular_error_deg(
    pred_rotation: np.ndarray, gt_rotation: np.ndarray
) -> np.ndarray:
    pred_norm = _normalized_rotation(pred_rotation)
    gt_norm = _normalized_rotation(gt_rotation)
    pred_yaw = np.arctan2(pred_norm[:, 1], pred_norm[:, 0])
    gt_yaw = np.arctan2(gt_norm[:, 1], gt_norm[:, 0])
    diff = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
    return cast(np.ndarray, np.abs(diff) * 180.0 / np.pi)


def _signed_angular_error_deg(
    pred_rotation: np.ndarray,
    gt_rotation: np.ndarray,
) -> np.ndarray:
    pred_norm = _normalized_rotation(pred_rotation)
    gt_norm = _normalized_rotation(gt_rotation)
    pred_yaw = np.arctan2(pred_norm[:, 1], pred_norm[:, 0])
    gt_yaw = np.arctan2(gt_norm[:, 1], gt_norm[:, 0])
    diff = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
    return cast(np.ndarray, diff * 180.0 / np.pi)


def _score_scene(
    predictor: PLCSPredictor,
    scene_path: Path,
    cameras_cfg: Any,
    candidates_per_scene: int,
) -> list[dict[str, Any]]:
    scene = load_scene(scene_path)
    cameras = _resolve_cameras(cameras_cfg, int(scene.num_cameras))
    outputs = predictor.predict(
        denormalize=False,
        **build_multiview_inputs(scene, cameras),
    )
    pred_position = np.asarray(outputs["position"].squeeze(0).numpy())
    pred_rotation = np.asarray(outputs["rotation"].squeeze(0).numpy())
    gt_position = np.asarray(scene.position)
    gt_rotation = np.asarray(scene.rotation)

    frames = min(pred_rotation.shape[0], gt_rotation.shape[0])
    pred_position = pred_position[:frames]
    pred_rotation = pred_rotation[:frames]
    gt_position = gt_position[:frames]
    gt_rotation = gt_rotation[:frames]

    angular_error = _angular_error_deg(pred_rotation, gt_rotation)
    signed_error = _signed_angular_error_deg(pred_rotation, gt_rotation)
    position_error_norm = np.linalg.norm(pred_position - gt_position, axis=-1)
    scale_xyz = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
    position_error_m = np.linalg.norm(
        (pred_position - gt_position) * scale_xyz, axis=-1
    )

    pred_norm = _normalized_rotation(pred_rotation)
    gt_norm = _normalized_rotation(gt_rotation)
    pred_yaw = np.arctan2(pred_norm[:, 1], pred_norm[:, 0]) * 180.0 / np.pi
    gt_yaw = np.arctan2(gt_norm[:, 1], gt_norm[:, 0]) * 180.0 / np.pi

    top_count = min(max(1, int(candidates_per_scene)), frames)
    top_indices = np.argsort(angular_error)[-top_count:][::-1]
    rows: list[dict[str, Any]] = []
    for frame_idx_raw in top_indices:
        frame_idx = int(frame_idx_raw)
        rows.append(
            {
                "scene_id": scene_path.name,
                "scene_path": str(scene_path),
                "frame_idx": frame_idx,
                "angular_error_deg": float(angular_error[frame_idx]),
                "signed_angular_error_deg": float(signed_error[frame_idx]),
                "position_error_norm": float(position_error_norm[frame_idx]),
                "position_error_m": float(position_error_m[frame_idx]),
                "gt_rotation": [float(v) for v in gt_norm[frame_idx]],
                "pred_rotation": [float(v) for v in pred_norm[frame_idx]],
                "gt_yaw_deg": float(gt_yaw[frame_idx]),
                "pred_yaw_deg": float(pred_yaw[frame_idx]),
                "cameras": cameras,
            }
        )
    return rows


def _select_samples(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    unique_scenes: bool,
) -> list[dict[str, Any]]:
    candidates = sorted(
        candidates,
        key=lambda item: float(item["angular_error_deg"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    for item in candidates:
        scene_id = str(item["scene_id"])
        if unique_scenes and scene_id in used_scenes:
            continue
        selected.append(dict(item))
        used_scenes.add(scene_id)
        if len(selected) >= top_k:
            return selected
    return selected


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _crop_scene(
    src: Path,
    dst: Path,
    *,
    center_frame: int,
    half_window: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Output scene already exists: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    meta = _load_json(src / "meta.json")
    scalars = _load_json(src / "scalars.json")
    position = np.load(src / "position.npy", mmap_mode="r")
    total = int(meta.get("num_frames", position.shape[0]))
    total = min(total, int(position.shape[0]))

    start = max(0, int(center_frame) - int(half_window))
    end = min(total, int(center_frame) + int(half_window) + 1)
    if end - start < 2:
        start = max(0, min(start, total - 2))
        end = min(total, start + 2)
    local_frame = int(center_frame) - start

    meta["source_scene_id"] = meta.get("scene_id", src.name)
    meta["source_frame_idx"] = int(center_frame)
    meta["source_start_frame_idx"] = start
    meta["scene_id"] = dst.name
    meta["num_frames"] = end - start
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    (dst / "scalars.json").write_text(json.dumps(scalars, indent=2))

    for npy_path in src.glob("*.npy"):
        arr = np.load(npy_path)
        if arr.ndim >= 1 and int(arr.shape[0]) == total:
            arr = arr[start:end]
        np.save(dst / npy_path.name, arr)

    return start, end, local_frame


def _render_sample(
    sample: dict[str, Any],
    *,
    checkpoint: Path,
    device: str,
    animation_view: str,
    fps: float,
    save_path: Path,
) -> None:
    runtime = RuntimeConfig(
        mode="predict",
        scene_path=Path(str(sample["sample_scene_path"])),
        checkpoint=checkpoint,
        device=device,
        animation_view=animation_view,
        fps=fps,
        save=save_path,
        camera=0,
        cameras=list(sample["cameras"]),
        info=False,
    )
    exit_code = run_visualization(runtime)
    if exit_code != 0:
        raise RuntimeError(f"Visualization failed for {runtime.scene_path}")


@hydra_main(
    config_path="../../configs",
    config_name="analyze_rotation_error_samples",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    scene_dir = Path(to_absolute_path(str(cfg.run.scene_dir)))
    checkpoint = Path(to_absolute_path(str(cfg.run.checkpoint)))
    out_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.analysis.device)
    predictor = PLCSPredictor.load_from_checkpoint(checkpoint, device=device)
    scene_base = scene_dir / "scenes"

    candidates: list[dict[str, Any]] = []
    names = _scene_names(scene_dir, str(cfg.analysis.split))
    for scene_name in tqdm(names, desc="scoring scenes"):
        candidates.extend(
            _score_scene(
                predictor,
                scene_base / scene_name,
                cfg.analysis.cameras,
                candidates_per_scene=int(cfg.analysis.candidates_per_scene),
            )
        )

    selected = _select_samples(
        candidates,
        top_k=int(cfg.analysis.top_k),
        unique_scenes=bool(cfg.analysis.unique_scenes),
    )

    scene_out_dir = out_dir / str(cfg.analysis.scene_subdir)
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        sample_name = (
            f"sample_{rank:02d}_{item['scene_id']}_frame_{item['frame_idx']:04d}"
        )
        sample_scene_path = scene_out_dir / sample_name
        start, end, local_frame = _crop_scene(
            Path(str(item["scene_path"])),
            sample_scene_path,
            center_frame=int(item["frame_idx"]),
            half_window=int(cfg.analysis.clip_half_window),
            overwrite=bool(cfg.analysis.overwrite),
        )
        item["sample_scene_path"] = str(sample_scene_path)
        item["clip_start_frame_idx"] = start
        item["clip_end_frame_idx_exclusive"] = end
        item["clip_local_frame_idx"] = local_frame

        save_path = out_dir / f"{sample_name}_{cfg.analysis.output_suffix}"
        item["visualization"] = str(save_path)
        if bool(cfg.analysis.render_visualizations):
            _render_sample(
                item,
                checkpoint=checkpoint,
                device=device,
                animation_view=str(cfg.analysis.animation_view),
                fps=float(cfg.analysis.fps),
                save_path=save_path,
            )

    report = {
        "checkpoint": str(checkpoint),
        "scene_dir": str(scene_dir),
        "split": str(cfg.analysis.split),
        "device": device,
        "selection": {
            "top_k": int(cfg.analysis.top_k),
            "unique_scenes": bool(cfg.analysis.unique_scenes),
            "candidates_per_scene": int(cfg.analysis.candidates_per_scene),
        },
        "num_scenes": len(names),
        "samples": selected,
    }
    report_path = out_dir / str(cfg.analysis.report_filename)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved rotation-error sample report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cast(Callable[[], int], main)())
